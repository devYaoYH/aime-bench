"""Compare original and first-answer stopping prompts in a paired streaming Qwen pilot.

Use this experiment to test natural stopping on Q1, Q3, Q18, and Q25 with one
request per prompt arm. It streams and preserves full SSE/timing records but
does not impose stop sequences or cancel on a candidate. Exact-key grading
occurs locally. Requires baseline traces, cached Qwen tokenizer, and a paid
OpenRouter connection; use a fresh output label for a new eight-call pilot.
    python -m src.experiments.streaming.first_answer_pilot --out runs/first-answer-NEW-LABEL
"""
from __future__ import annotations

import argparse
import asyncio
from bisect import bisect_left
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import random
import statistics
import time

import httpx

from src.answer_extraction import extract_events, eligible
from src.common import ROOT, OPENROUTER_URL, RETRYABLE, atomic_json, extract_answer, load_key, utc_now

PROMPT_SUFFIX = (
    " As soon as you derive the first complete candidate answer to the requested problem, "
    "end your reasoning immediately, output the final line Answer: NNN, and stop generating. "
    "Do not recheck that candidate, seek another method, test smaller examples, or repeat the derivation. "
    "Intermediate quantities and answers to toy examples are not complete answers to the requested problem."
)
QUESTIONS = (1, 3, 18, 25)
SELECTION = {
    1: "All eight first proposals correct; median first position 1,408 versus 5,747.5 final tokens.",
    3: "All eight first proposals correct; median first position 3,345 versus 12,924 final tokens.",
    18: "Three correct first proposals before the cap; all eight old finals missing. Hard salvage test.",
    25: "Seven correct first proposals; extensive rechecking and toy cases; five old capped attempts.",
}


async def sse_payloads(lines):
    """Handle SSE comments, multiline data, blank delimiters, and EOF."""
    data = []
    async for line in lines:
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    if data:
        yield '\n'.join(data)


class Accumulator:
    def __init__(self):
        self.parts = {'reasoning': [], 'content': []}
        self.lengths = {'reasoning': 0, 'content': 0}
        self.metadata = {}
        self.usage = None
        self.finish_reason = None
        self.native_finish_reason = None
        self.errors = []
        self.first_output_s = None
        self.first_content_s = None
        self.last_output_s = None
        self.finish_s = None
        self.deliveries = []

    def add(self, body, elapsed):
        for k in ['id', 'created', 'model', 'provider', 'system_fingerprint', 'service_tier']:
            if k in body: self.metadata[k] = body[k]
        if body.get('error'): self.errors.append(body['error'])
        if body.get('usage'): self.usage = body['usage']
        for choice in body.get('choices', []):
            if choice.get('index', 0) != 0: continue
            delta = choice.get('delta') or {}
            reasoning = delta.get('reasoning')
            if not reasoning:
                reasoning = ''.join(d.get('text', '') for d in delta.get('reasoning_details', [])
                                    if d.get('type') == 'reasoning.text')
            emitted = False
            for part, value in [('reasoning', reasoning), ('content', delta.get('content'))]:
                if isinstance(value, str) and value:
                    self.parts[part].append(value)
                    self.lengths[part] += len(value)
                    emitted = True
                    if part == 'content' and self.first_content_s is None: self.first_content_s = elapsed
            if emitted:
                if self.first_output_s is None: self.first_output_s = elapsed
                self.last_output_s = elapsed
                self.deliveries.append({'elapsed_s': elapsed, **self.lengths})
            if choice.get('finish_reason'):
                self.finish_reason = choice['finish_reason']
                self.native_finish_reason = choice.get('native_finish_reason')
                if self.finish_s is None: self.finish_s = elapsed

    def response(self):
        return {**self.metadata, 'object': 'chat.completion', 'usage': self.usage,
                'choices': [{'index': 0, 'message': {'role': 'assistant', **{k: ''.join(v) for k, v in self.parts.items()}},
                             'finish_reason': self.finish_reason, 'native_finish_reason': self.native_finish_reason}]}


def annotate(record, tokenizer):
    if not record.get('response'): return
    response = record['response']
    choice = response['choices'][0]
    message = choice['message']
    events, offset = [], 0
    for part in ['reasoning', 'content']:
        encoded = tokenizer.encode(message[part], add_special_tokens=False)
        ends = [item[1] for item in encoded.offsets]
        deliveries = record['timing']['deliveries']
        for event in extract_events(message[part], record['problem']):
            event['part'] = part
            event['visible_output_token'] = offset + bisect_left(ends, event['end']) + 1
            delivery = next((d for d in deliveries if d[part] >= event['end']), None)
            event['observed_at_s'] = delivery['elapsed_s'] if delivery else None
            event['correct'] = event['answer'] == record['gold_answer']
            events.append(event)
        offset += len(encoded.ids)
    usage = response.get('usage') or {}
    completion = usage.get('completion_tokens')
    overhead = max(0, completion - offset) if completion is not None else 0
    for e in events: e['estimated_stop_tokens'] = min(completion or offset, e['visible_output_token'] + overhead)
    first = next((e for e in events if eligible(e, 'proposal')), None)
    marker = next((e for e in events if eligible(e, 'markers')), None)
    target = next((e for e in events if eligible(e, 'target')), None)
    record.update(candidate=extract_answer(message['content']), usage=usage, finish_reason=choice['finish_reason'])
    record['correct'] = record['candidate'] is not None and int(record['candidate']) == record['gold_answer']
    record['analysis'] = {
        'events': events, 'first_literal_proposal': first, 'first_marker': marker, 'first_requested_quantity': target,
        'visible_output_tokens': offset,
        'tokens_after_first_proposal': (completion or offset) - first['estimated_stop_tokens'] if first else None,
        'seconds_after_first_proposal': record['generation_latency_s'] - first['observed_at_s'] if first and first['observed_at_s'] is not None else None,
        'note': 'Automatic clauses only, no retrospective toy annotations for the new traces. Inspect scope manually.',
    }


async def run(args):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / '.local/tokenizers/qwen3.json'))
    source = ROOT / 'runs' / args.run
    output = (ROOT / args.out).resolve()
    if not output.is_relative_to(ROOT / 'runs'): raise ValueError('Output must be within this repo/runs')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'questions').mkdir(exist_ok=True)
    originals = {q: json.loads((source / 'questions' / f'{q:02d}.json').read_text()) for q in QUESTIONS}
    config = {
        'source_run': args.run, 'questions': list(QUESTIONS), 'selection': SELECTION,
        'model': originals[1]['request']['model'], 'endpoint': OPENROUTER_URL,
        'control_prompt': originals[1]['request']['messages'][0]['content'],
        'treatment_prompt': originals[1]['request']['messages'][0]['content'] + PROMPT_SUFFIX,
        'requests': 8, 'concurrency': 8, 'provider': {'order': ['deepinfra'], 'allow_fallbacks': False},
        'sampling': {k: originals[1]['request'][k] for k in ['temperature', 'top_p', 'top_k', 'max_tokens', 'reasoning']},
        'forced_stop': False, 'client_cancellation': False, 'stream': True,
        'grading': 'local equality against saved key, no external grader',
        'notes': ['One stochastic sample per arm per selected question; descriptive pilot only.',
                  'Both arms stream with the same routing and sampling; only the system message differs.',
                  'Both arms use the historical DeepInfra provider; no provider fallback.',
                  'No seed requested; the samples do not share a stochastic trajectory.',
                  'No API retries after a partial stream; preserve failures separately.'],
    }
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing pilot configuration differs')
    atomic_json(config_path, config)
    key = load_key()
    started_run = time.perf_counter()
    started_utc = utc_now()
    requests = [(q, arm) for q in QUESTIONS for arm in ['control', 'first_answer']]
    random.Random(20261002).shuffle(requests)
    timeout = httpx.Timeout(connect=30, read=args.api_timeout, write=30, pool=30)
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=12)) as client:
        async def one(q, arm):
            path = output / 'questions' / f'{q:02d}-{arm}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get('response') and saved.get('done_received') and not saved.get('stream_errors'):
                    return saved
            old = originals[q]
            body = deepcopy(old['request'])
            body['messages'][0]['content'] = config['treatment_prompt'] if arm == 'first_answer' else config['control_prompt']
            body.update(stream=True, provider=config['provider'])
            record = {'problem_idx': q, 'arm': arm, 'problem': old['problem'], 'gold_answer': old['gold_answer'],
                      'request': body, 'response': None, 'api_attempts': [], 'candidate': None, 'correct': None}
            for attempt_no in range(1, args.retries + 2):
                accumulator = Accumulator()
                start = time.perf_counter()
                attempt = {'attempt': attempt_no, 'started_at_utc': utc_now()}
                stream_path = path.with_name(path.stem + f'-stream-{attempt_no}.jsonl')
                done = False
                last_progress = start
                print(f'START Q{q:02d} {arm}', flush=True)
                try:
                    async with client.stream('POST', OPENROUTER_URL,
                                             headers={'Authorization': f'Bearer {key}', 'X-Title': 'AIME first-answer paired pilot'}, json=body) as response:
                        attempt['http_status'] = response.status_code
                        attempt['headers_received_s'] = time.perf_counter() - start
                        attempt['generation_id_header'] = response.headers.get('x-generation-id')
                        if response.status_code != 200:
                            await response.aread()
                            try: attempt['error_response'] = response.json()
                            except ValueError: attempt['error_response'] = response.text
                        else:
                            with stream_path.open('w') as stream_file:
                                async for payload in sse_payloads(response.aiter_lines()):
                                    elapsed = time.perf_counter() - start
                                    if payload == '[DONE]':
                                        done = True
                                        break
                                    chunk = json.loads(payload)
                                    stream_file.write(json.dumps({'elapsed_s': elapsed, 'chunk': chunk}, ensure_ascii=False) + '\n')
                                    accumulator.add(chunk, elapsed)
                                    if time.perf_counter() - last_progress >= 5:
                                        stream_file.flush()
                                        atomic_json(path.with_name(path.stem + '-progress.json'),
                                                    {'elapsed_s': elapsed, **accumulator.lengths, 'finish_reason': accumulator.finish_reason,
                                                     'reasoning_tail': ''.join(accumulator.parts['reasoning'][-20:]),
                                                     'content_tail': ''.join(accumulator.parts['content'][-20:])})
                                        last_progress = time.perf_counter()
                except (httpx.HTTPError, ValueError) as exc:
                    attempt['error'] = f'{type(exc).__name__}: {exc}'
                attempt['latency_s'] = time.perf_counter() - start
                attempt['finished_at_utc'] = utc_now()
                attempt['stream_file'] = str(stream_path.relative_to(output))
                record['api_attempts'].append(attempt)
                if accumulator.first_output_s is not None:
                    record.update(response=accumulator.response(), generation_latency_s=attempt['latency_s'],
                                  done_received=done, stream_errors=accumulator.errors,
                                  timing={'first_output_s': accumulator.first_output_s, 'first_content_s': accumulator.first_content_s,
                                          'last_output_s': accumulator.last_output_s, 'finish_s': accumulator.finish_s,
                                          'deliveries': accumulator.deliveries})
                    annotate(record, tokenizer)
                    break  # Never silently retry and mix partially generated trajectories.
                if attempt_no > args.retries or attempt.get('http_status', 503) not in RETRYABLE: break
                await asyncio.sleep(2)
            record['total_api_latency_s'] = sum(a['latency_s'] for a in record['api_attempts'])
            record['completed_at_utc'] = utc_now()
            atomic_json(path, record)
            print(f"DONE Q{q:02d} {arm}: answer={record['candidate']} correct={record['correct']} "
                  f"tokens={(record.get('usage') or {}).get('completion_tokens')} "
                  f"seconds={record.get('generation_latency_s', record['total_api_latency_s']):.2f} finish={record.get('finish_reason')}", flush=True)
            return record
        records = await asyncio.gather(*(one(q, arm) for q, arm in requests))
    historical = json.loads((source / 'intermediate_answers/traces.json').read_text())['rows']
    comparisons = []
    for q in QUESTIONS:
        ctrl = next(r for r in records if r['problem_idx'] == q and r['arm'] == 'control')
        early = next(r for r in records if r['problem_idx'] == q and r['arm'] == 'first_answer')
        old_rows = [r for r in historical if r['problem_idx'] == q]
        pairs = {'problem_idx': q, 'gold_answer': originals[q]['gold_answer'],
                 'historical_median_tokens': statistics.median(r['completion_tokens'] for r in old_rows),
                 'historical_median_latency_s': statistics.median(r['api_latency_s'] for r in old_rows)}
        for arm, r in [('control', ctrl), ('first_answer', early)]:
            pairs[arm] = {k: r.get(k) for k in ['candidate', 'correct', 'finish_reason', 'generation_latency_s', 'done_received']}
            pairs[arm].update(completion_tokens=(r.get('usage') or {}).get('completion_tokens'),
                              provider=(r.get('response') or {}).get('provider'),
                              first_proposal=(r.get('analysis') or {}).get('first_literal_proposal'),
                              tokens_after_first_proposal=(r.get('analysis') or {}).get('tokens_after_first_proposal'),
                              seconds_after_first_proposal=(r.get('analysis') or {}).get('seconds_after_first_proposal'))
        if pairs['control']['completion_tokens'] and pairs['first_answer']['completion_tokens'] is not None:
            pairs['token_saving_fraction'] = 1 - pairs['first_answer']['completion_tokens'] / pairs['control']['completion_tokens']
            pairs['latency_saving_fraction'] = 1 - early['generation_latency_s'] / ctrl['generation_latency_s']
        comparisons.append(pairs)
    arm_totals = {}
    for arm in ['control', 'first_answer']:
        group = [r for r in records if r['arm'] == arm]
        valid = [r for r in group if r.get('response') and r.get('done_received') and not r.get('stream_errors')]
        arm_totals[arm] = {'requests': len(group), 'successful_streams': len(valid), 'correct': sum(r.get('correct') is True for r in group),
                           'natural_stops': sum(r.get('finish_reason') == 'stop' for r in group),
                           'completion_tokens': sum((r.get('usage') or {}).get('completion_tokens', 0) for r in group),
                           'sum_latency_s': sum(r.get('generation_latency_s', 0) for r in group),
                           'batch_max_latency_s': max((r.get('generation_latency_s', 0) for r in group), default=0),
                           'reported_cost': sum((r.get('usage') or {}).get('cost', 0) or 0 for r in group)}
    summary = {'config': config, 'invocation_start_utc': started_utc, 'invocation_end_utc': utc_now(),
               'invocation_wall_s': time.perf_counter() - started_run, 'arm_totals': arm_totals, 'comparisons': comparisons}
    atomic_json(output / 'summary.json', summary)
    print(json.dumps({'arm_totals': arm_totals, 'comparisons': comparisons}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='20260930-155212')
    p.add_argument('--out', default='runs/first-answer-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    p.add_argument('--api-timeout', type=float, default=900)
    p.add_argument('--retries', type=int, default=1)
    args = p.parse_args()
    if args.retries < 0: p.error('retries must be nonnegative')
    asyncio.run(run(args))
