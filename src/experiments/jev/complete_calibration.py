"""Fill the capped-trajectory Jev coverage gap without changing the historical cohort.

Use after the original completed-only calibration. This post-cutoff follow-up
scores every remaining length-capped trace with the identical gold-free 1,500
token prefix/Noul payload. Original outcomes are attached only after judging;
precision is retrospective final-answer recovery under the original cap, not
proof that rejected prefixes cannot be salvaged. Required raw records stay
local; compact decisions and an all-240 summary are saved separately.
Planning is the default; --execute authorizes paid OpenRouter calls.
    python -m src.experiments.jev.complete_calibration --tokenizer-json .local/tokenizers/qwen3.json --execute
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import httpx
from tokenizers import Tokenizer

from src.common import ROOT, atomic_json, load_key, utc_now
from src.tokenizer_utils import prefix_for
from src.experiments.jev.judge_trajectories import read_probability, RETRYABLE

FOLDER = 'jev_calibration_followup'


def exact_prefix(text, tokenizer, limit):
    """Trim a split Unicode codepoint only if the source prefix still encodes to N."""
    try:
        prefix, count = prefix_for(text, tokenizer, limit)
        return prefix, count, None
    except ValueError:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded.ids) < limit:
            raise
        start, end = encoded.offsets[limit-1]
        for boundary in range(end-1, start-1, -1):
            prefix = text[:boundary]
            if len(tokenizer.encode(prefix, add_special_tokens=False).ids) == limit:
                return prefix, len(encoded.ids), 'Trimmed a split Unicode codepoint; source-preserving prefix re-encodes to exactly 1500 tokens.'
        raise ValueError('No exact source-preserving prefix at the token boundary')


def prepare(run, tokenizer_path):
    """Verify the complementary cohort and frozen prompt/tokenizer before payment."""
    original_config = json.loads((run / 'jev_calibration/config.json').read_text())
    original = json.loads((run / 'jev_calibration/summary.json').read_text())
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != original_config['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from the historical calibration')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    existing = {(r['problem_idx'], r['sample_number']) for r in original['rows']}
    paths = sorted((run / 'questions').glob('*.json'))
    paths += sorted((run / 'self_consistency/questions').glob('*/*.json'))
    found, prepared = set(), []
    for path in paths:
        trace = json.loads(path.read_text())
        identity = trace['problem_idx'], trace.get('sample_number', 1)
        if identity in found:
            raise ValueError('Duplicate trajectory identifier')
        found.add(identity)
        if identity in existing:
            if trace['finish_reason'] != 'stop':
                raise ValueError('Historical cohort disagrees with saved outcomes')
            continue
        if trace['finish_reason'] != 'length' or trace.get('candidate') is not None:
            raise ValueError('Unscored trajectory is not a capped missing-answer case')
        reasoning = trace['response']['choices'][0]['message'].get('reasoning') or ''
        prefix, count, adjustment = exact_prefix(reasoning, tokenizer, original_config['prefix_tokens'])
        payload = {
            'model': original_config['model'],
            'state': {'problem': trace['problem'], 'reasoning_trace': prefix,
                      'solver_model': 'qwen/qwen3-30b-a3b',
                      'trace_status': f"This is only the first {original_config['prefix_tokens']} Qwen3 reasoning tokens of a saved trajectory; consider continuation from this prefix.",
                      'continuation_budget_output_tokens': original_config['continuation_budget_tokens']},
            'questions': {'promising_to_extend': deepcopy(original_config['question'])},
        }
        prepared.append({'problem_idx': identity[0], 'sample_number': identity[1],
                         'source_trace_file': str(path.relative_to(run)),
                         'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'reasoning_prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
                         'full_reasoning_tokens_tokenizer': count,
                         'prefix_tokens': original_config['prefix_tokens'],
                         'prefix_boundary_adjustment': adjustment, 'request': payload})
    expected = {(i, s) for i in range(1, 31) for s in range(1, 9)}
    if found != expected or not existing <= found or len(existing) != 125 or len(prepared) != 115:
        raise ValueError('Expected exactly 125 historical and 115 complementary trajectories')
    return original_config, original, prepared


def summarize(original, records, run):
    """Count all observed outcomes, preserving censored status and cohort provenance."""
    new = []
    for record in records:
        path = run / record['source_trace_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['source_sha256']:
            raise ValueError('Source changed after judging')
        trace = json.loads(path.read_text())
        new.append({'problem_idx': record['problem_idx'], 'sample_number': record['sample_number'],
                    'probability': record['probability'], 'correct': bool(trace['correct']),
                    'candidate': trace['candidate'], 'finish_reason': trace['finish_reason'],
                    'completion_tokens': trace['usage']['completion_tokens'],
                    'within_8192_more_tokens': False, 'cohort': 'post-cutoff capped follow-up',
                    'decision_file': f"{FOLDER}/{record['problem_idx']:02d}-{record['sample_number']:02d}.json",
                    'source_trace_file': record['source_trace_file']})
    rows = [dict(r, cohort='historical completed-only') for r in original['rows']] + new
    rows.sort(key=lambda r: (r['problem_idx'], r['sample_number']))
    identities = {(r['problem_idx'], r['sample_number']) for r in rows}
    if len(rows) != 240 or len(identities) != 240:
        raise ValueError('Combined cohort must contain 240 unique trajectories')
    thresholds = []
    for cutoff in (.2, .3, .4, .5, .6, .65, .7):
        retained = [r for r in rows if r['probability'] >= cutoff]
        tp = sum(r['correct'] for r in retained)
        thresholds.append({'retain_at_or_above': cutoff, 'retained': len(retained),
                           'correct_final_retained': tp, 'observed_failure_retained': len(retained)-tp,
                           'capped_retained': sum(r['finish_reason']=='length' for r in retained),
                           'completed_wrong_retained': sum(r['finish_reason']=='stop' and not r['correct'] for r in retained),
                           'correct_final_rejected': sum(r['correct'] and r['probability']<cutoff for r in rows),
                           'observed_final_precision': tp/len(retained) if retained else None})
    return {'schema_version': 1, 'model': original['model'], 'run_id': run.name,
            'post_cutoff': True, 'historical_scored': len(original['rows']), 'new_scored': len(new),
            'total_scored': len(rows), 'prefix_tokens': original['prefix_tokens'],
            'continuation_budget_tokens': original['continuation_budget_tokens'],
            'rows': rows, 'thresholds': thresholds,
            'notes': ['Original 125 decisions are preserved and reused; 115 capped traces are scored in a later batch.',
                      'Jev receives only problem and exact prefix; outcomes/keys are attached afterward.',
                      'Precision uses observed correct final answers under the original 16K cap, not mathematical unsalvageability or prospective continuation accuracy.',
                      'Capped outcomes have no final answer; later reasoning may contain useful intermediate candidates.',
                      'Repeated samples share questions; collection dates differ between cohorts.']}


async def execute(args):
    run = ROOT / 'runs' / args.run
    config, original, prepared = prepare(run, args.tokenizer_json)
    print(f'Prepared {len(prepared)} capped prefixes; historical {len(original["rows"])} preserved.', flush=True)
    if not args.execute:
        return
    out = run / FOLDER
    out.mkdir(exist_ok=True)
    provenance = {'schema_version': 1, 'protocol': config, 'source_selection': '115 finish_reason=length trajectories absent from historical calibration',
                  'post_cutoff': True, 'producer': 'src.experiments.jev.complete_calibration',
                  'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'historical_summary_sha256': hashlib.sha256((run/'jev_calibration/summary.json').read_bytes()).hexdigest(),
                  'started_at_utc': utc_now(), 'expected_new_decisions': 115}
    cp = out / 'config.json'
    if cp.exists():
        saved = json.loads(cp.read_text())
        for key in ('protocol','producer_sha256','historical_summary_sha256'):
            if saved[key] != provenance[key]:
                raise ValueError('Existing follow-up has different controls')
        provenance = saved
    else:
        atomic_json(cp, provenance)
    key = load_key()
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30), limits=httpx.Limits(max_connections=args.concurrency+5)) as client:
        async def one(item):
            path = out / f"{item['problem_idx']:02d}-{item['sample_number']:02d}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['request'] != item['request'] or saved['source_sha256'] != item['source_sha256']:
                    raise ValueError('Existing decision source/payload changed')
                if saved.get('probability') is not None:
                    return saved
            record = dict(item, attempts=[], response=None, probability=None, post_cutoff=True)
            async with semaphore:
                for number in range(1, args.retries+2):
                    attempt = {'number': number, 'started_at_utc': utc_now()}
                    started = time.perf_counter()
                    try:
                        response = await client.post(config['endpoint'], json=item['request'], headers={'Authorization': f'Bearer {key}', 'Content-Type':'application/json','X-Title':'AIME post-cutoff Jev coverage follow-up'})
                        attempt['http_status'] = response.status_code
                        body = response.json()
                        attempt['response'] = body
                        if response.status_code == 200:
                            record['probability'] = read_probability(body, 'promising_to_extend')
                            record['response'] = body
                    except (httpx.HTTPError, ValueError, KeyError) as exc:
                        attempt['error'] = f'{type(exc).__name__}: {exc}'
                    attempt.update(finished_at_utc=utc_now(), latency_s=time.perf_counter()-started)
                    record['attempts'].append(attempt)
                    atomic_json(path, record)
                    if record['probability'] is not None or attempt.get('http_status', 429) not in RETRYABLE:
                        break
                    if number <= args.retries:
                        await asyncio.sleep(min(2**number,8))
            print(f"{path.stem}: promising={record['probability']}", flush=True)
            return record
        records = await asyncio.gather(*(one(item) for item in prepared))
    missing = [r for r in records if r['probability'] is None]
    if missing:
        raise RuntimeError(f'{len(missing)} missing decisions; rerun resumes completed records')
    summary = summarize(original, records, run)
    summary.update(started_at_utc=provenance['started_at_utc'], finished_at_utc=utc_now(),
                   total_input_tokens=sum((r['response'].get('usage') or {}).get('input_tokens',0) for r in records),
                   total_reported_cost=round(sum((r['response'].get('usage') or {}).get('cost',0) for r in records),9))
    atomic_json(out/'summary.json', summary)
    print(f'Saved {summary["total_scored"]}/240 scored trajectories.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='20260930-155212')
    parser.add_argument('--tokenizer-json', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--concurrency', type=int, default=20)
    parser.add_argument('--retries', type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 0:
        parser.error('Invalid concurrency/retries')
    asyncio.run(execute(args))


if __name__ == '__main__':
    main()
