"""v2.1 rejects malformed candidates and deduplicates proven expression keys."""
import ast
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from runner_final.core_v2 import runner as old_runner, _streaming as old_streaming
from runner_final.core_v2_1 import runner, _streaming
from runner_final.core_v2_1.syntax import ExpressionValidator, POLICY
from runner_final.integrity import verify_core as verify_v1
from runner_final.integrity_v2 import verify_core as verify_v2
from runner_final.integrity_v2_1 import verify_core
from runner_final.run_frozen_v2 import parse_args as old_args
from runner_final.run_frozen_v2_1 import parse_args
from src.attempt_storage import AttemptArtifacts, DisabledGPUSampler
from src.attempt_metrics import AttemptProfiler
from src.common import ROOT
from test.test_attempt import FakeGPU, Stream, chunk
from test.test_runner_final import capped


class IntegrityTests(unittest.TestCase):
    def test_all_three_manifests_verify_and_controls_match(self):
        for verify in (verify_v1, verify_v2, verify_core):
            self.assertEqual(len(verify(ROOT)), 64)
        new, old = parse_args([]), old_args([])
        for name in ('system_prompt', 'system_prompt_sha256', 'parallelism', 'rollouts',
                     'first_pass_max_tokens', 'max_tokens', 'max_attempts_per_question',
                     'schedule', 'model', 'temperature', 'top_p', 'grader_cost',
                     'benchmark', 'skip_benchmark_prewarm'):
            self.assertEqual(getattr(new, name), getattr(old, name), name)
        self.assertEqual((new.parallelism, new.rollouts, new.first_pass_max_tokens,
                          new.max_tokens, new.max_attempts_per_question), (30, 1, 8192, 16384, 4))

    def test_budget_and_exact_prefix_functions_are_unchanged(self):
        def definition(module, name):
            tree = ast.parse(Path(module.__file__).read_text())
            return ast.dump(next(n for n in tree.body if getattr(n, 'name', None) == name), include_attributes=False)
        for name in ('group_options', 'question_state'):
            self.assertEqual(definition(runner, name), definition(old_runner, name))
        self.assertEqual(definition(_streaming, 'continuation_prefix'), definition(old_streaming, 'continuation_prefix'))


class SyntaxTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.validator = ExpressionValidator()
        await self.validator.start()

    async def asyncTearDown(self):
        self.validator.close()

    async def test_preserve_supported_correct_expression_forms(self):
        for answer in ('2618', '2077992225', '-17', '0.125', 'n-1', 'n(n-1)/2',
                       r'\frac{4}{9}', r'\frac{9\sqrt{30}}{4}',
                       r'\frac{1}{46}\binom{2024}{990}+\frac{1}{2}',
                       '10^{225}-1', '4N^3+9N^2+6N+1', r'\{1,2\}', '(1,2)',
                       r'\sin(\pi/2)', r'\sqrt[3]{2}', r'\left(1+\sqrt{2}\right)'):
            with self.subTest(answer=answer):
                result = await self.validator.validate(answer)
                self.assertTrue(result['valid'], result)
                self.assertIn('canonical_key', result)

    async def test_malformed_and_template_candidates_never_pass(self):
        for answer in ('EXPRESSION', '...', '?', r'\mathrm{EXPRESSION}',
                       r'\mathit{ANSWER}', r'\frac{1}{}', 'x -', '1+2;',
                       r'\frac{1}{2}%suffix', 'the value is 42', '(1,)',
                       r'\{1,,2\}', '{' * 65 + '1' + '}' * 65):
            with self.subTest(answer=answer):
                self.assertFalse((await self.validator.validate(answer))['valid'])

    async def test_equivalence_and_type_sensitive_keys(self):
        async def keys(values):
            return [(await self.validator.validate(s))['canonical_key'] for s in values]
        for equivalent in (['1/2', '2/4', r'\frac{1}{2}'],
                           [r'\sqrt{8}', r'2\sqrt{2}'], ['x+x', '2x'],
                           [r'\emptyset', r'\varnothing', r'\{\}'],
                           [r'\{1,2\}', r'\{2,1,1\}']):
            self.assertEqual(len(set(await keys(equivalent))), 1)
        self.assertEqual(len(set(await keys(['(1,2)', '(2,1)', r'\{1,2\}']))), 3)
        self.assertNotEqual(*(await keys(['x/x', '1'])))
        self.assertNotEqual(*(await keys(['(x^2-1)/(x-1)', 'x+1'])))

    async def test_raw_cache_has_no_second_worker_parse(self):
        first = await self.validator.validate('2/4')
        with patch.object(self.validator, '_receive', side_effect=AssertionError('cache must not call worker')):
            second = await self.validator.validate('2/4')
        self.assertTrue(second['cache_hit'])
        self.assertEqual(first['canonical_key'], second['canonical_key'])
        self.assertEqual(second['syntax_cpu_s'], 0)

    async def test_canonical_timeout_keeps_valid_candidate_and_restarts_worker(self):
        with patch.object(self.validator, '_receive', new=AsyncMock(side_effect=[
                {'stage': 'syntax', 'valid': True}, TimeoutError])):
            result = await self.validator.validate('3/7')
        self.assertTrue(result['valid'])
        self.assertTrue(result['canonical_key'].startswith('raw:'))
        self.assertIsNone(self.validator.process)
        self.assertTrue((await self.validator.validate('42'))['valid'])

    async def test_syntax_timeout_rejects_and_kills_owned_worker(self):
        with patch.object(self.validator, '_receive', new=AsyncMock(side_effect=TimeoutError)):
            result = await self.validator.validate('7/8')
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'syntax_timeout_or_crash')
        self.assertIsNone(self.validator.process)
        self.assertTrue((await self.validator.validate('43'))['valid'])

    async def test_worker_cancellation_releases_lock_and_process(self):
        gate = asyncio.Event()
        async def receive(*args):
            await gate.wait()
        with patch.object(self.validator, '_receive', new=receive):
            task = asyncio.create_task(self.validator.validate('8/9'))
            await asyncio.sleep(.01)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(self.validator.lock.locked())
        self.assertIsNone(self.validator.process)


class StreamIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_syntax_rejects_and_expression_duplicates_never_reach_grader(self):
        checks, streams = [], []
        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == '/v1/chat/completions':
                stream = Stream([chunk(r'\boxed{EXPRESSION}'), chunk(r'\boxed{\frac{1}{}}'),
                                 chunk(r'\boxed{4/8}'), chunk(r'\boxed{\frac{1}{2}}'),
                                 chunk(r'\boxed{2/4}'), chunk(r'\boxed{\frac{4}{9}}')], delay=.004, hang=True)
                streams.append(stream)
                return httpx.Response(200, stream=stream)
            checks.append(body['candidate'])
            await asyncio.sleep(.02)
            return httpx.Response(200, json={'verdict': body['candidate'] == r'\frac{4}{9}'})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(['--target-correct', '1'])
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await _streaming.run_question({'problem_idx': 27, 'problem': 'Compute a fraction.'}, args, client, root, FakeGPU())
            self.assertEqual(checks, ['4/8', r'\frac{4}{9}'])
            self.assertTrue(streams[0].closed)
            self.assertEqual(result['unique_candidates'], 2)
            rows = [json.loads(line) for line in (root/'trace/27/candidate_validation.jsonl').read_text().splitlines()]
            self.assertEqual([r['outcome'] for r in rows], ['rejected', 'rejected', 'enqueued',
                             'duplicate_expression', 'duplicate_expression', 'enqueued'])
            self.assertTrue(all('validation_wall_s' in r for r in rows))

    async def test_continuation_retains_expression_keys_and_buffered_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AttemptArtifacts(root, buffered=True)
            profiler = AttemptProfiler(root, enabled=False, artifacts=store)
            args = parse_args(['--target-correct', '1'])
            requests, checks = [], []
            async def handler(request):
                body = json.loads(request.content)
                if request.url.path == '/verify':
                    checks.append(body['candidate'])
                    await asyncio.sleep(.02)
                    return httpx.Response(200, json={'verdict': body['candidate'] == r'\frac{4}{9}'})
                requests.append(body)
                if len(requests) == 1:
                    return httpx.Response(200, stream=Stream(capped([10], [11], r'\boxed{4/8}\boxed{\frac{4}{')))
                self.assertEqual(body['prompt'], [10, 11])
                return httpx.Response(200, stream=Stream(capped([10, 11], [12], r'9}}\boxed{\frac{1}{2}}', completion=True)))
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await runner.run_speedrun([{'problem_idx': 1, 'problem': 'Compute.'}], args, client, root, DisabledGPUSampler(), time.perf_counter(), profiler)
            self.assertEqual(checks, ['4/8', r'\frac{4}{9}'])
            self.assertEqual(result[0]['status'], 'solved')
            self.assertEqual(len(requests), 2)
            self.assertEqual(len(result[0]['candidate_keys']), 2)
            self.assertEqual(list(root.rglob('*')), [])
            store.flush()
            rows = [json.loads(line) for line in (root/'trace/01/candidate_validation.jsonl').read_text().splitlines()]
            self.assertIn('duplicate_expression', [r['outcome'] for r in rows])


if __name__ == '__main__':
    unittest.main()
