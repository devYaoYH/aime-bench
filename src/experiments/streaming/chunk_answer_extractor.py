"""Replay hosted scope extraction on saved streaming candidates and development controls.

Use this experiment to compare span selection, focused prompts, formatting
guards, and timed two-slot extraction on the paired first-answer SSE records.
Loads six controls from the original Qwen traces; records manifests, hosted
responses, latency, and extraction statistics in the chosen output directory.
Requires ignored source traces and OPENROUTER_API_KEY. --prepare-only builds
fixtures without model calls; normal execution makes paid small-model requests
but never restarts the original solver. Runtime helpers remain in src.chunk_answer_extractor.
    python -m src.experiments.streaming.chunk_answer_extractor
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import httpx

from src.answer_extraction import extract_events
from src.common import ROOT, OPENROUTER_URL, atomic_json, load_key, utc_now
from src.stream_answer_markers import delta_text
from src.chunk_answer_extractor import WindowBuilder, prompt, grounded_decision, focused_prompt, obvious_scope_rejection, SCHEMA, SCOPES


def load_stream_cases(source, max_jobs):
    cases = []
    for path in sorted((source / 'questions').glob('*.json')):
        if '-progress' in path.name: continue
        record = json.loads(path.read_text())
        if not record.get('response') or not record.get('done_received'): continue
        builder = WindowBuilder(record['problem'])
        jobs = []
        stream = source / record['api_attempts'][-1]['stream_file']
        for line in stream.read_text().splitlines():
            item = json.loads(line)
            for choice in item['chunk'].get('choices', []):
                if choice.get('index', 0) != 0: continue
                for part, value in delta_text(choice.get('delta') or {}).items():
                    job = builder.feed(part, value, item['elapsed_s'])
                    if job: jobs.append(job)
        for part in ['reasoning', 'content']:
            job = builder.feed(part, '', record['generation_latency_s'], final=True)
            if job: jobs.append(job)
        cases.append({'case_id': path.stem, 'problem_idx': record['problem_idx'], 'arm': record['arm'],
                      'problem': record['problem'], 'gold': record['gold_answer'],
                      'natural_latency_s': record['generation_latency_s'], 'available_jobs': len(jobs),
                      'jobs': jobs[:max_jobs], 'source_file': str(path.relative_to(ROOT))})
    return cases


def load_controls():
    # Real, labeled excerpts: semantic scope labels, not labels derived from gold.
    specifications = [
        ('format-answer-line', 3, 2, 'Answer: 016', 'format_or_quote'),
        ('format-box', 3, 8, r'\boxed{016}', 'format_or_quote'),
        ('toy-chairs', 25, 1, 'answer is 3.', 'toy_example'),
        ('toy-square', 26, 3, 'answer would be 3.', 'toy_example'),
        ('wrong-real-proposal', 23, 3, 'answer is 1000 - 400 = 600.', 'requested_answer'),
        ('correct-real-proposal', 1, 1, 'answer is 21 + 49 = 70', 'requested_answer'),
    ]
    root = ROOT / 'runs/20260930-155212'
    controls = []
    for name, q, sample, needle, label in specifications:
        path = root / 'questions' / f'{q:02d}.json' if sample == 1 else root / 'self_consistency/questions' / f'{q:02d}' / f'{sample:02d}.json'
        record = json.loads(path.read_text())
        text = record['response']['choices'][0]['message']['reasoning']
        # Select the known formatting use, not the subsequent final box.
        if name == 'format-box':
            position = text.index(needle, text.index('the exact format is'))
        elif name == 'format-answer-line':
            position = text.index(needle, text.index('So, for example, if the answer is 16'))
        else:
            position = text.index(needle)
        start = max(0, position-2600)
        end = text.find('\n\n', position)
        end = end+2 if end >= 0 else len(text)
        window = text[start:end]
        events = extract_events(window, record['problem'])
        selected = [e for e in events if e['start'] <= position-start < e['end']]
        if not selected: raise ValueError(f'Control passage absent: {name}')
        e = selected[0]
        job = {'arrival_s': 0, 'part': 'reasoning', 'window_start': start, 'window': window,
               'candidates': [{'id': 1, 'answer': e['answer'], 'quote': e['quote'], 'kind': e['kind'],
                               'part': 'reasoning', 'start': start+e['start'], 'end': start+e['end'],
                               'host_context_scope': e['confidence'],
                               'requested_transform': e.get('transform')}], 'candidate_overflow': 0}
        controls.append({'case_id': name, 'problem_idx': q, 'problem': record['problem'],
                         'expected_scope': label, 'source_file': str(path.relative_to(ROOT)), 'job': job})
    return controls


async def main(args):
    source = (ROOT / args.source).resolve()
    output = (ROOT / args.out).resolve()
    if not source.is_relative_to(ROOT / 'runs') or not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('Read/write runs only inside this repository')
    output.mkdir(parents=True, exist_ok=True)
    cases = load_stream_cases(source, args.max_jobs)
    controls = load_controls()
    manifest = {'source': args.source, 'cases': cases, 'controls': controls,
                'policy': 'Inspect completed lines with answer clauses or requested-quantity equations; 4096-character causal windows; six spans/window.',
                'keys_in_model_prompts': False, 'max_jobs_per_trace': args.max_jobs}
    atomic_json(output / 'manifest.json', manifest)
    print('Prepared', len(cases), 'streams;', sum(len(c['jobs']) for c in cases), 'bounded inspection windows;', len(controls), 'scope controls.', flush=True)
    if args.prepare_only: return
    config = {'model': args.model, 'concurrency': args.concurrency, 'max_jobs': args.max_jobs,
              'source': args.source, 'max_output_tokens': 64, 'temperature': 0,
              'instruction': INSTRUCTION, 'protocol': 'JSON span selection, no solving or proof verification',
              'mode': 'timed replay' if args.timed else 'accelerated latency probe',
              'focused': args.focused, 'scope_guard': args.scope_guard,
              'notes': ['No new solver calls or local inference.', 'Model keys/grades are withheld.',
                        'No actual solver cancellation; its saved stream is simulated.',
                        'One result per selected first proposal, not a test of final-answer accuracy.']}
    if (output / 'config.json').exists() and json.loads((output/'config.json').read_text()) != config:
        raise ValueError('Use a new label for a different configuration')
    atomic_json(output / 'config.json', config)
    key = load_key()
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=30), limits=httpx.Limits(max_connections=8)) as client:
        async def classify(case_id, number, problem, job):
            if args.focused and len(job['candidates']) > 1:
                latest = None
                for c in job['candidates'][:3]:
                    singleton = {**job, 'candidates': [c]}
                    latest = await classify(case_id, f'{number}-span{c["id"]}', problem, singleton)
                    if latest['candidate'] is not None: return latest
                return latest
            scope_schema = {'type': 'object', 'properties': {'scope': SCHEMA['properties']['scope']},
                            'required': ['scope'], 'additionalProperties': False}
            request = {'model': args.model, 'messages': [{'role': 'user', 'content': focused_prompt(problem, job) if args.focused else prompt(problem, job)}],
                       'temperature': 0, 'max_tokens': 64,
                       'response_format': {'type': 'json_schema', 'json_schema': {'name': 'proposal_scope', 'strict': True, 'schema': scope_schema if args.focused else SCHEMA}}}
            fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            suffix = f'{number:02d}' if type(number) is int else str(number)
            path = output / f'{case_id}-{suffix}.json'
            if path.exists():
                old = json.loads(path.read_text())
                if old['fingerprint'] != fingerprint: raise ValueError('Saved model request differs')
                if old.get('parsed') is not None: return old
            submitted = time.perf_counter()
            result = {'request': request, 'fingerprint': fingerprint, 'source_job': job,
                      'request_file': path.name, 'submitted_at_utc': utc_now()}
            rejection = obvious_scope_rejection(job['candidates'][0]) if args.scope_guard else None
            if rejection:
                result.update(parsed={'candidate_id': None, 'scope': rejection}, candidate=None,
                              service_latency_s=0, queue_latency_s=0, total_latency_s=0,
                              rule_only=True)
                atomic_json(path, result)
                print(case_id, number, 'RULE REJECT', rejection, flush=True)
                return result
            async with semaphore:
                service_start = time.perf_counter()
                try:
                    response = await client.post(OPENROUTER_URL, headers={'Authorization': f'Bearer {key}', 'X-Title': 'Streaming proposed-answer scope pilot'}, json=request)
                    result['http_status'] = response.status_code
                    try: body = response.json()
                    except ValueError: body = {'raw_text': response.text}
                    result['response'] = body
                    response.raise_for_status()
                    if body['choices'][0]['finish_reason'] != 'stop': raise ValueError('Classifier output did not complete')
                    parsed = json.loads(body['choices'][0]['message']['content'])
                    if args.focused:
                        parsed['candidate_id'] = job['candidates'][0]['id'] if parsed.get('scope') == 'requested_answer' else None
                    result['candidate'] = grounded_decision(parsed, job)
                    result['parsed'] = parsed
                except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                    result['parsed'] = None
                    result['candidate'] = None
                    result['error'] = f'{type(exc).__name__}: {exc}'
                result['service_latency_s'] = time.perf_counter()-service_start
                result['queue_latency_s'] = service_start-submitted
                result['total_latency_s'] = time.perf_counter()-submitted
                atomic_json(path, result)
                print(case_id, number, result['parsed'] or result.get('error'), f"service={result['service_latency_s']:.2f}s", flush=True)
                result['request_file'] = path.name
                return result

        async def one_stream(case):
            records, accepted = [], None
            for number, job in enumerate(case['jobs'], 1):
                if args.timed:
                    await asyncio.sleep(max(0, job['arrival_s'] - (time.perf_counter()-started)))
                    if time.perf_counter()-started >= case['natural_latency_s']: break
                decision = await classify(case['case_id'], number, case['problem'], job)
                ready = time.perf_counter()-started if args.timed else job['arrival_s']+decision['total_latency_s']
                records.append({'request_file': decision.get('request_file', f"{case['case_id']}-{number:02d}.json"), 'arrival_s': job['arrival_s'],
                                'ready_s': ready, 'parsed': decision['parsed'], 'service_latency_s': decision['service_latency_s'],
                                'queue_latency_s': decision['queue_latency_s']})
                if decision['candidate'] is not None:
                    c = decision['candidate']
                    accepted = {'candidate': c, 'arrival_s': job['arrival_s'], 'ready_s': ready,
                                'exact_key_match': c['answer'] == case['gold'],
                                'before_natural_end': ready < case['natural_latency_s']}
                    break
            return {k: case[k] for k in ['case_id', 'problem_idx', 'arm', 'natural_latency_s', 'available_jobs', 'source_file']} | {
                'inspections': records, 'accepted': accepted, 'budget_exhausted': accepted is None and len(case['jobs']) < case['available_jobs']}

        async def one_control(case):
            decision = await classify(case['case_id'], 1, case['problem'], case['job'])
            accepted = decision['candidate'] is not None
            expected = case['expected_scope'] == 'requested_answer'
            return {'case_id': case['case_id'], 'expected_scope': case['expected_scope'], 'parsed': decision['parsed'],
                    'accepted': accepted, 'scope_binary_correct': accepted == expected,
                    'service_latency_s': decision['service_latency_s'], 'request_file': f"{case['case_id']}-01.json"}

        # Controls run first to evaluate false-positive protection before replay.
        control_results = await asyncio.gather(*(one_control(c) for c in controls))
        started = time.perf_counter()
        results = await asyncio.gather(*(one_stream(c) for c in cases))
    summary = {'config': config, 'controls': control_results, 'streams': results,
               'replay_wall_s': time.perf_counter()-started,
               'accepted_streams': sum(r['accepted'] is not None for r in results),
               'correct_accepted': sum(r['accepted'] is not None and r['accepted']['exact_key_match'] for r in results),
               'accepted_before_end': sum(r['accepted'] is not None and r['accepted']['before_natural_end'] for r in results)}
    atomic_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='runs/first-answer-20261002-paired')
    p.add_argument('--out', default='runs/chunk-extractor-gemma4b-pilot')
    p.add_argument('--model', default='google/gemma-3-4b-it')
    p.add_argument('--max-jobs', type=int, default=6)
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--timed', action='store_true')
    p.add_argument('--focused', action='store_true')
    p.add_argument('--scope-guard', action='store_true')
    args = p.parse_args()
    if args.max_jobs < 1 or args.concurrency < 1: p.error('Positive job budget and concurrency required')
    asyncio.run(main(args))
