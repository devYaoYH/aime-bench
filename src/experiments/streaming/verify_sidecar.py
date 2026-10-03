"""Probe hosted verification on saved causal prefixes in shadow mode.

Use this experiment to check six first proposals, paired altered candidates,
and one naturally wrong intermediate claim from the streaming extraction pilot.
Only text received by each candidate's timestamp enters model requests. Saves
probe records, latency, and false-approval statistics in runs/verify-sidecar-qwen7b/.
Requires local SSE/raw traces and paid OpenRouter calls; --prepare-only creates
fixtures without inference. No live solver is started or canceled. The reusable
nonblocking runtime remains in src.verify_sidecar.
    python -m src.experiments.streaming.verify_sidecar
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import statistics
import time

import httpx

from src.common import ROOT, OPENROUTER_URL, atomic_json, load_key, utc_now
from src.experiments.streaming.chunk_answer_extractor import load_controls, load_stream_cases
from src.verify_sidecar import INSTRUCTION, validate_verdict, verification_request
from src.stream_answer_markers import delta_text


def causal_prefix(case, arrival_s, source):
    """Use only deltas already delivered at the candidate's timestamp."""
    record = json.loads((ROOT / case['source_file']).read_text())
    stream = source / record['api_attempts'][-1]['stream_file']
    text = {'reasoning': '', 'content': ''}
    for line in stream.read_text().splitlines():
        event = json.loads(line)
        if event['elapsed_s'] > arrival_s: break
        for choice in event['chunk'].get('choices', []):
            if choice.get('index', 0) == 0:
                for part, value in delta_text(choice.get('delta') or {}).items():
                    text[part] += value
    return text['reasoning'] + ('\n\nCONTENT:\n' + text['content'] if text['content'] else '')


def pilot_cases(source):
    """First six saved proposals + paired wrong candidates + natural wrong claim."""
    examples = []
    for case in load_stream_cases(source, 1):
        if not case['jobs']: continue
        job = case['jobs'][0]
        candidate = job['candidates'][0]
        draft = causal_prefix(case, job['arrival_s'], source)
        base = {k: case[k] for k in ['case_id', 'problem_idx', 'problem', 'gold', 'natural_latency_s']}
        base.update(draft=draft, arrival_s=job['arrival_s'], source_candidate=candidate)
        examples.append({**base, 'answer': candidate['answer'], 'condition': 'first_proposal'})
        examples.append({**base, 'case_id': case['case_id']+'-altered',
                         'answer': (candidate['answer']+1) % 1000, 'condition': 'altered_candidate'})
    wrong = next(c for c in load_controls() if c['case_id'] == 'wrong-real-proposal')
    record = json.loads((ROOT / wrong['source_file']).read_text())
    end = wrong['job']['window_start']+len(wrong['job']['window'])
    examples.append({'case_id': 'natural-wrong-600', 'problem_idx': wrong['problem_idx'],
                     'problem': wrong['problem'], 'gold': record['gold_answer'],
                     'draft': record['response']['choices'][0]['message']['reasoning'][:end],
                     'answer': wrong['job']['candidates'][0]['answer'], 'condition': 'natural_wrong',
                     'source_candidate': wrong['job']['candidates'][0], 'source_file': wrong['source_file']})
    return examples


async def run_pilot(args):
    source, output = (ROOT/args.source).resolve(), (ROOT/args.out).resolve()
    if not source.is_relative_to(ROOT/'runs') or not output.is_relative_to(ROOT/'runs'):
        raise ValueError('Runs must stay in this repository')
    output.mkdir(parents=True, exist_ok=True)
    cases = pilot_cases(source)
    atomic_json(output/'manifest.json', {'cases': cases, 'keys_sent_to_model': False})
    config = {'model': args.model, 'max_tokens': args.max_tokens, 'concurrency': args.concurrency,
              'source': args.source, 'instruction': INSTRUCTION,
              'mode': 'causal-prefix shadow probe; no solver cancellation',
              'three_second_sensitivity_is_assumed': True}
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text()) != config:
        raise ValueError('New configuration needs a new output label')
    atomic_json(output/'config.json', config)
    print(f'Prepared {len(cases)} verifier probes. No new solver calls or local inference.', flush=True)
    if args.prepare_only: return
    key = load_key()
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=30)) as client:
        async def one(case):
            request = verification_request(args.model, case['problem'], case['answer'], case['draft'], args.max_tokens)
            fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            path = output/(case['case_id']+'.json')
            if path.exists():
                old = json.loads(path.read_text())
                if old['fingerprint'] != fingerprint: raise ValueError('Saved verifier request differs')
                if old.get('decision') and not old.get('error'): return old
            result = {'request': request, 'fingerprint': fingerprint, 'case_id': case['case_id'],
                      'condition': case['condition'], 'candidate_answer': case['answer'],
                      'started_at_utc': utc_now()}
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(OPENROUTER_URL,
                        headers={'Authorization': f'Bearer {key}', 'X-Title': 'Concurrent candidate verification shadow pilot'},
                        json=request)
                    result['http_status'] = response.status_code
                    result['response'] = response.json()
                    response.raise_for_status()
                    choice = result['response']['choices'][0]
                    if choice['finish_reason'] != 'stop': raise ValueError('Incomplete verifier output')
                    result['decision'] = validate_verdict(json.loads(choice['message']['content']))
                except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                    result['error'] = f'{type(exc).__name__}: {exc}'
                    result['decision'] = {'verdict': 'insufficient', 'reason': result['error']}
                result['service_latency_s'] = time.perf_counter()-started
            # Stored key is consulted only after the HTTP request and decision.
            result['exact_key_match'] = case['answer'] == case['gold']
            result['false_approval'] = result['decision']['verdict'] == 'verified' and not result['exact_key_match']
            if 'arrival_s' in case:
                result['arrival_s'] = case['arrival_s']
                result['natural_latency_s'] = case['natural_latency_s']
                result['ready_s_no_queue'] = case['arrival_s']+result['service_latency_s']
                result['ready_s_assumed_3s'] = case['arrival_s']+3
            atomic_json(path, result)
            print(case['case_id'], result['decision']['verdict'],
                  f"{result['service_latency_s']:.2f}s", 'FALSE APPROVAL' if result['false_approval'] else '', flush=True)
            return result
        results = await asyncio.gather(*(one(c) for c in cases))
    correct = [r for r in results if r['exact_key_match']]
    wrong = [r for r in results if not r['exact_key_match']]
    summary = {'config': config, 'probes': len(results), 'correct_candidates': len(correct),
               'wrong_candidates': len(wrong),
               'correct_approved': sum(r['decision']['verdict']=='verified' for r in correct),
               'false_approvals': sum(r['false_approval'] for r in wrong),
               'wrong_rejected': sum(r['decision']['verdict']=='rejected' for r in wrong),
               'insufficient': sum(r['decision']['verdict']=='insufficient' for r in results),
               'median_service_s': statistics.median(r['service_latency_s'] for r in results),
               'max_service_s': max(r['service_latency_s'] for r in results),
               'reported_cost': sum((r.get('response', {}).get('usage') or {}).get('cost') or 0 for r in results),
               'results': [{k: r[k] for k in ['case_id', 'condition', 'candidate_answer', 'decision',
                          'exact_key_match', 'false_approval', 'service_latency_s']} for r in results]}
    atomic_json(output/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['results', 'config']}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='runs/first-answer-20261002-paired')
    p.add_argument('--out', default='runs/verify-sidecar-qwen7b')
    p.add_argument('--model', default='qwen/qwen-2.5-7b-instruct')
    p.add_argument('--max-tokens', type=int, default=768)
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if args.max_tokens < 1 or args.concurrency < 1: p.error('Positive token and concurrency budgets required')
    asyncio.run(run_pilot(args))
