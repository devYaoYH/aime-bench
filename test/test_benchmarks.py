"""Offline year selection, provenance, mismatch guards and 2026 oracle checks."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
import yaml

from src import benchmarks
from src.attempt import parse_args, ready
from src.attempt_metadata import build_metadata, validate_metadata
from src.attempt_results import build_results
from src.attempt_viewer import AttemptStore
from src.attempt_runners import naive_pass4_v1, speedrun_v1, sweep_speedrun_v1
from src.common import ROOT, load_problems
from src.fetch_dataset import normalize_rows, combine_2024_parts
from test import test_attempt_results


class BenchmarkTests(unittest.TestCase):
    def test_year_selection_defaults_and_answers_never_reach_solver(self):
        for runner in (parse_args, naive_pass4_v1.parse_args, speedrun_v1.parse_args):
            self.assertEqual(runner(['--model', 'test/model']).benchmark_year, 2025)
            args = runner(['--model', 'test/model', '--benchmark-year', '2026'])
            self.assertEqual(args.benchmark_year, 2026)
            self.assertEqual(args.benchmark_role, 'generalization')
        self.assertNotEqual(benchmarks.load_questions([1]), benchmarks.load_questions([1], 2026))
        for year in (2024, 2025, 2026):
            questions = benchmarks.load_questions(year=year)
            self.assertEqual(len(questions), 30)
            self.assertTrue(all(set(q) == {'problem_idx', 'problem'} for q in questions))
            self.assertEqual(len(load_problems(year)), 30)
        with self.assertRaises(ValueError):
            benchmarks.load_questions([31], 2026)
        with self.assertRaises(ValueError):
            benchmarks.benchmark_paths(2023)

    def test_provenance_and_variant_answers_match_bundled_key(self):
        for year, role in ((2024, 'prewarming'), (2025, 'development'), (2026, 'generalization')):
            evidence = benchmarks.dataset_provenance(year)
            self.assertEqual(evidence['role'], role)
            prompts, grader, _ = benchmarks.benchmark_paths(year)
            self.assertEqual(evidence['prompt_sha256'], hashlib.sha256(prompts.read_bytes()).hexdigest())
            self.assertEqual(evidence['grader_sha256'], hashlib.sha256(grader.read_bytes()).hexdigest())
        self.assertEqual(load_problems(2026)[15]['answer'], 178)
        self.assertEqual(load_problems(2026)[24]['answer'], 850)
        self.assertEqual(benchmarks.dataset_provenance(2026, 'development')['role'], 'development')

    def test_prewarming_selection_metadata_and_two_source_mapping(self):
        from src.attempt_runners import speedrun_v2, speedrun_v3, speedrun_v4
        for parser in (parse_args, naive_pass4_v1.parse_args, speedrun_v2.parse_args,
                       speedrun_v3.parse_args, speedrun_v4.parse_args):
            args = parser(['--model', 'test/model', '--benchmark-year', '2024'])
            self.assertEqual(args.benchmark_role, 'prewarming')
        rows = load_problems(2024)
        self.assertEqual(rows[0]['answer'], 204)
        self.assertEqual(rows[15]['answer'], 73)
        self.assertEqual(rows[-1]['answer'], 315)
        provenance = benchmarks.dataset_provenance(2024)
        self.assertEqual([s['rows'] for s in provenance['sources']], [15, 15])
        self.assertEqual([s['index_offset'] for s in provenance['sources']], [0, 15])
        self.assertEqual([s['revision'] for s in provenance['sources']],
                         ['ea5b061c3e8039dc9858defaafc407d04b995e9f',
                          '29d5d31e9b46e215fc24d9b2a3047506823dd101'])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'warmup'; folder.mkdir()
            metadata = build_metadata(folder, {'benchmark_year': 2024, 'dataset_provenance': provenance})
            validate_metadata(metadata, 'warmup')
            self.assertEqual(metadata['controls']['dataset'], 'AIME 2024')
            self.assertEqual(metadata['provenance']['dataset']['role'], 'prewarming')
            self.assertEqual(metadata['provenance']['dataset']['sources'], provenance['sources'])

    def test_combining_2024_papers_preserves_all_indices_and_rejects_missing_paper(self):
        rows = load_problems(2024)
        first = rows[:15]
        second = [{**r, 'problem_idx': r['problem_idx'] - 15} for r in rows[15:]]
        self.assertEqual(combine_2024_parts([list(reversed(first)), second]), rows)
        for parts in ([first], [first[:-1], second], [first, second + [second[0]]],
                      [[{**first[0], 'problem_idx': True}] + first[1:], second]):
            with self.assertRaises(ValueError):
                combine_2024_parts(parts)

    def test_mixed_key_and_modified_prompt_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in ('data/aime_2026_problems.jsonl', 'grader/data/aime_2026.jsonl', 'data/source_2026.json'):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / relative).read_bytes())
            with patch.object(benchmarks, 'ROOT', root):
                grader = root / 'grader/data/aime_2026.jsonl'
                text = grader.read_text()
                grader.write_text(text.replace('"answer": "277"', '"answer": "70"', 1))
                with self.assertRaisesRegex(ValueError, 'Prompt/grader mismatch'):
                    benchmarks.dataset_provenance(2026)
                grader.write_text(text)
                prompts = root / 'data/aime_2026_problems.jsonl'
                prompts.write_text(prompts.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'prompt hash'):
                    benchmarks.dataset_provenance(2026)

    def test_import_validates_complete_rows_and_answers(self):
        rows = load_problems(2026)
        self.assertEqual(normalize_rows(list(reversed(rows))), rows)
        for bad in (rows[:-1], rows + [rows[0]], [{**rows[0], 'answer': 1000}] + rows[1:],
                    [{**rows[0], 'problem': ''}] + rows[1:]):
            with self.assertRaises(ValueError):
                normalize_rows(bad)

    def test_new_metadata_and_legacy_unknown_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'attempt'; folder.mkdir()
            config = {'benchmark_year': 2026, 'dataset_provenance': benchmarks.dataset_provenance(2026)}
            m = build_metadata(folder, config)
            validate_metadata(m, 'attempt')
            self.assertEqual(m['controls']['dataset'], 'AIME 2026')
            self.assertEqual(m['provenance']['dataset']['role'], 'generalization')
            legacy = build_metadata(folder, {'model': 'test/model'})
            evidence = legacy['provenance']['dataset']
            self.assertEqual(evidence['year'], 2025)
            self.assertTrue(evidence['inferred_from_legacy_runner'])
            self.assertIsNone(evidence['revision'])
            self.assertIsNone(evidence['role'])
            self.assertIsNone(evidence['prompt_sha256'])

    def test_viewer_uses_saved_year_and_blocks_cross_year_delta(self):
        fixture = test_attempt_results.AttemptResultsTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.make()
        folder, m = fixture.make('test2026', reference='control')
        config = json.loads((folder/'config.json').read_text())
        config.update(benchmark_year=2026, dataset_provenance=benchmarks.dataset_provenance(2026))
        (folder/'config.json').write_text(json.dumps(config))
        m = build_metadata(folder, config); m['intervention']['reference_attempt_id'] = 'control'
        (folder/'metadata.json').write_text(json.dumps(m))
        (fixture.root/'aime_2026_problems.jsonl').write_bytes((ROOT/'data/aime_2026_problems.jsonl').read_bytes())
        overview = fixture.store.overview('test2026')
        self.assertEqual(overview['attempt']['benchmark_year'], 2026)
        self.assertIn('Patrick', overview['questions'][0]['problem'])
        result = build_results(fixture.store)
        row = next(r for r in result['attempts'] if r['id'] == 'test2026')
        self.assertEqual(row['benchmark_role'], 'generalization')
        self.assertIsNone(row['comparison']['saved_s'])
        self.assertTrue(any('different AIME year' in w for w in result['warnings']))
        (fixture.root/'aime_2026_problems.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            fixture.store.overview('test2026')

    def test_sweep_selection_and_ranking_are_year_specific(self):
        config = json.loads((ROOT/'configs/sweeps/vibe-speedrun-v1.json').read_text())
        config['base_args'] += ['--benchmark-year', '2026']
        plan = sweep_speedrun_v1.build_plan(config)
        self.assertTrue(all(c['benchmark_year'] == 2026 and c['benchmark_role'] == 'generalization' for c in plan['cells']))
        rows = sweep_speedrun_v1.ranked([
            {'cell_id': 'dev', 'benchmark_year': 2025, 'time_to_target_s': 90},
            {'cell_id': 'test', 'benchmark_year': 2026, 'time_to_target_s': 80}])
        self.assertEqual([r['rank'] for r in rows], [1, 1])


class Grader2026Tests(unittest.IsolatedAsyncioTestCase):
    async def test_2024_http_grading(self):
        await self.test_http_health_verdict_and_audit_provenance(2024, ((1, '204', '70'), (16, '73', '178'), (30, '315', '314')))

    async def test_http_health_verdict_and_audit_provenance(self, year=2026, cases=None):
        if not all(importlib.util.find_spec(n) for n in ('sympy', 'loguru', 'regex', 'antlr4')):
            self.skipTest('Install grader/requirements-local.txt for HTTP integration')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            cfg = {'dataset': {'id': f'aime_{year}', 'year': year,
                              'source': str(ROOT/f'grader/data/aime_{year}.jsonl'), 'format': 'jsonl'},
                   'cost_c': 0, 'host': '127.0.0.1', 'port': port, 'audit_log': str(path/'audit.jsonl')}
            (path/'config.yaml').write_text(yaml.safe_dump(cfg))
            with (path/'server.log').open('w') as log:
                process = subprocess.Popen([sys.executable, str(ROOT/'grader/server.py')],
                    env={**os.environ, 'GRADER_CONFIG': str(path/'config.yaml')}, stdout=log, stderr=log)
                try:
                    async with httpx.AsyncClient(trust_env=False) as client:
                        health = await ready(client, f'http://127.0.0.1:{port}/health', 10, process)
                        self.assertEqual(health['dataset']['year'], year)
                        self.assertEqual(health['dataset']['sha256'], benchmarks.dataset_provenance(year)['grader_sha256'])
                        for index, correct, wrong in cases or ((1, '277', '70'), (16, '178', '196'), (25, '850', '340')):
                            for candidate, verdict in ((correct, True), (wrong, False)):
                                response = await client.post(f'http://127.0.0.1:{port}/verify', json={'index': index, 'candidate': candidate})
                                record = response.raise_for_status().json()
                                self.assertEqual(record['verdict'], verdict)
                                self.assertEqual(record['dataset']['year'], year)
                                self.assertNotIn('gold', record)
                        audit = [json.loads(line) for line in (path/'audit.jsonl').read_text().splitlines()]
                        self.assertEqual(len(audit), 6)
                        self.assertTrue(all(r['dataset']['id'] == f'aime_{year}' for r in audit))
                finally:
                    process.terminate(); process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
