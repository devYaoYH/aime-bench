"""Canonical AIME attempt: managed services, streaming fan-out, oracle, telemetry.

Run on callosum: python -m src.attempt --model Qwen/Qwen3.5-4B
See docs/attempts.md. No solver code compares candidates to the answer key.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from copy import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx
import yaml

from src.common import ROOT, atomic_json, utc_now
from src.benchmarks import add_dataset_args, benchmark_paths, dataset_provenance
from src.attempt_metadata import build_metadata
from src.attempt_metrics import AttemptProfiler, Meter, merge_meters, grader_timeline

PROMPT = (
    "Solve the AIME problem. Whenever you have a prospective answer to the original "
    "problem, immediately emit it as \\boxed{N}, where N is an integer from 0 to 999. "
    "You may continue checking your work afterward. End with your final boxed answer."
)


class CandidateDetector:
    """Closed integer boxes and complete answer clauses, independently per channel.

    In particular, a network boundary after one digit never completes an answer.
    Markers remain prospective; correctness comes exclusively from the oracle.
    """
    box = re.compile(r'\\boxed\s*\{\s*(\d{1,3})\s*\}')
    line = re.compile(r'(?i)^\s*(?:\*\*)?Answer\s*:\s*\$?\s*(\d{1,3})\s*\$?\s*(?:\*\*)?\s*[.]?\s*$')

    prose = re.compile(
        r'(?i)\b(?:final\s+)?answer\s+(?:is|would\s+be|should\s+be|must\s+be|might\s+be)'
        r'\s+\$?\s*(\d{1,3})\s*\$?(?=\s*(?:[.。!?;,](?:\s|$)|$))')

    def __init__(self):
        self.text = {'content': '', 'reasoning': ''}
        self.scanned = {'content': 0, 'reasoning': 0}
        self.line_start = {'content': 0, 'reasoning': 0}

    def feed(self, part, delta, eof=False):
        self.text[part] += delta
        text = self.text[part]
        found = []
        # Revisit the incomplete tail of a box, rather than repeatedly scanning
        # the full reasoning transcript on every token.
        for match in self.box.finditer(text, self.scanned[part]):
            found.append({'answer': int(match[1]), 'part': part,
                          'kind': 'boxed', 'end': match.end()})
            self.scanned[part] = match.end()
        tail = text.rfind('\\boxed', self.scanned[part])
        self.scanned[part] = tail if tail >= 0 else max(self.scanned[part], len(text) - 6)
        start = self.line_start[part]
        while '\n' in text[start:] or (eof and start < len(text)):
            end = text.find('\n', start)
            end = len(text) if end < 0 else end + 1
            complete_line = text[start:end]
            match = self.line.fullmatch(complete_line)
            if match:
                found.append({'answer': int(match[1]), 'part': part,
                              'kind': 'answer_line', 'end': end})
            for match in self.prose.finditer(complete_line):
                found.append({'answer': int(match[1]), 'part': part,
                              'kind': 'literal_prose', 'end': start + match.end(),
                              'line': complete_line.strip()})
            start = end
        self.line_start[part] = start
        return found


async def sse_payloads(lines):
    data = []
    async for line in lines:
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip())
    if data:
        yield '\n'.join(data)


def append_json(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + '\n')
    file.flush()


class GPUSampler:
    """Device-level NVML samples, shared by concurrent rollouts (not attribution)."""
    def __init__(self, path, interval=0.2, device=0):
        self.path, self.interval, self.device = path, interval, device
        self.samples = []
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            with self.path.open('w') as file:
                while not self.stop_event.is_set():
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    row = {'monotonic_s': time.perf_counter(), 'timestamp_utc': utc_now(),
                           'vram_used_mib': memory.used / 2**20,
                           'vram_total_mib': memory.total / 2**20, 'gpu_util_pct': util.gpu}
                    self.samples.append(row)
                    append_json(file, row)
                    self.stop_event.wait(self.interval)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    async def start(self):
        self.thread.start()
        for _ in range(100):
            if self.error:
                raise RuntimeError(f'GPU telemetry failed: {self.error}')
            if self.samples:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError('GPU telemetry did not produce a sample')

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def window(self, start, end):
        snapshot = list(self.samples)
        before = [s for s in snapshot if s['monotonic_s'] <= start]
        rows = ([before[-1]] if before else []) + [s for s in snapshot if start < s['monotonic_s'] <= end]
        return {'sample_count': len(rows),
                'start_vram_mib': rows[0]['vram_used_mib'] if rows else None,
                'end_vram_mib': rows[-1]['vram_used_mib'] if rows else None,
                'observed_peak_vram_mib': max((s['vram_used_mib'] for s in rows), default=None),
                'scope': 'shared GPU device; sampled peak, not per-request allocation',
                'error': self.error}


def continuation_prefix(previous, folder, context_limit):
    """Only continue a capped trajectory with complete, exact token-ID evidence."""
    if not previous or not previous['rollouts']:
        return None
    last = previous['rollouts'][-1]
    if last['status'] != 'completed' or last['finish_reason'] != 'length':
        return None
    token_file = folder / f'rollout-{last["rollout"]:02d}' / 'tokens.json'
    if not token_file.exists():
        raise RuntimeError('Cannot continue capped output without saved exact token IDs')
    tokens = json.loads(token_file.read_text())
    prompt, output = tokens['prompt_token_ids'], tokens['output_token_ids']
    if not prompt or not output or not tokens['complete']:
        raise RuntimeError('Incomplete token-ID evidence for continuation')
    prefix = prompt + output
    if len(prefix) >= context_limit - 1:
        return None  # start a fresh trajectory after exhausting the context window
    return {'prompt': prefix, 'parent_rollout': last['rollout'],
            'visible_text': tokens['visible_text'], 'remaining_context': context_limit - len(prefix)}


async def run_question(problem, args, client, output, sampler, *, round_no=1,
                       rollout_offset=0, attempt_start=None, on_solved=None, target_event=None, profiler=None):
    meter = Meter(not getattr(args, 'no_overhead_profile', False),
                  parent=profiler.meter if profiler else None)
    rollout_meters = {}

    def write_json(path, value, scope=meter):
        with scope.measure('artifact_json_write'):
            atomic_json(path, value)

    index = problem['problem_idx']
    folder = output / 'trace' / f'{index:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    previous = json.loads((folder / 'question.json').read_text()) if (folder / 'question.json').exists() else None
    if previous and previous['status'] == 'solved':
        raise ValueError('Solved questions must not be retried')
    if previous and len(previous['rollouts']) + args.rollouts > args.max_attempts_per_question:
        raise ValueError('Per-question generation attempt limit exceeded')
    next_continuation = None
    if args.strategy == 'coverage' and not args.no_continuation:
        next_continuation = continuation_prefix(previous, folder, args.max_context_tokens)
    start, started = time.perf_counter(), utc_now()
    candidates = asyncio.Queue()
    seen = set(previous.get('candidate_answers', [])) if previous else set()
    records = []
    winner = None
    question_error = None
    solved_event = None
    stopped_for_target = False

    def propose(event, rollout):
        candidate = str(event['answer'])
        meter.inc('candidate_proposals')
        if candidate not in seen:
            seen.add(candidate)
            candidates.put_nowait({**event, 'candidate': candidate, 'rollout': rollout,
                                   'observed_at_utc': utc_now(),
                                   'question_elapsed_s': time.perf_counter() - start,
                                   '_enqueued_monotonic_s': time.perf_counter()})
            meter.inc('candidate_unique_enqueued')
            meter.high_water('candidate_queue_depth', candidates.qsize())
        else:
            meter.inc('candidate_duplicates_suppressed')

    async def generate(rollout):
        scope = rollout_meters[rollout] = Meter(meter.enabled, parent=meter)
        scope.inc('generation_requests')

        def detect(part, value, eof=False):
            with scope.measure('candidate_parse_enqueue'):
                for event in detector.feed(part, value, eof=eof):
                    propose(event, rollout)

        path = folder / f'rollout-{rollout:02d}'
        path.mkdir()
        request = {'model': args.model, 'messages': [
            {'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': problem['problem']}],
            'temperature': args.temperature, 'top_p': args.top_p,
            'max_tokens': args.max_tokens, 'seed': args.seed + index * args.rollouts + rollout,
            'stream': True, 'stream_options': {'include_usage': True, 'continuous_usage_stats': True},
            'return_token_ids': args.strategy == 'coverage'}
        if args.disable_thinking:
            request['chat_template_kwargs'] = {'enable_thinking': False}
        continuation = next_continuation
        endpoint = '/v1/chat/completions'
        if continuation:
            endpoint = '/v1/completions'
            request.pop('messages')
            request.pop('chat_template_kwargs', None)
            request.update(prompt=continuation['prompt'],
                           max_tokens=min(args.max_tokens, continuation['remaining_context']))
        write_json(path / 'request.json', request, scope)
        began = time.perf_counter()
        record = {'rollout': rollout, 'round': round_no, 'started_at_utc': utc_now(), 'start_monotonic_s': began,
                  'ttft_s': None, 'last_token_s': None, 'finish_reason': None,
                  'status': 'streaming', 'done_received': False, 'usage': None,
                  'endpoint': endpoint, 'continuation_of_rollout': continuation['parent_rollout'] if continuation else None,
                  'requested_max_tokens': request['max_tokens']}
        detector = CandidateDetector()
        parts = {'reasoning': [], 'content': []}
        prompt_token_ids = list(continuation['prompt']) if continuation else None
        output_token_ids = []
        if continuation:
            detector.feed('content', continuation['visible_text'])
        records.append((path, record, parts))
        try:
            with (path / 'stream.jsonl').open('w') as stream_file:
                async with client.stream('POST', args.vllm_url + endpoint,
                                         json=request, headers={'X-Request-Id': f'{output.name}-q{index}-r{rollout}'}) as response:
                    record['http_status'] = response.status_code
                    record['headers_received_s'] = time.perf_counter() - began
                    response.raise_for_status()
                    async for payload in sse_payloads(response.aiter_lines()):
                        elapsed = time.perf_counter() - began
                        scope.inc('sse_chunks')
                        scope.inc('sse_payload_bytes', len(payload.encode()))
                        with scope.measure('stream_trace_write_flush'):
                            append_json(stream_file, {'elapsed_s': elapsed, 'timestamp_utc': utc_now(), 'data': payload})
                        if payload == '[DONE]':
                            record['done_received'] = True
                            break
                        with scope.measure('sse_json_decode'):
                            body = json.loads(payload)
                        if body.get('error'):
                            raise RuntimeError(f'vLLM stream error: {body["error"]}')
                        if body.get('prompt_token_ids') is not None:
                            prompt_token_ids = body['prompt_token_ids']
                        if body.get('usage'):
                            record['usage'] = body['usage']
                        for choice in body.get('choices', []):
                            if choice.get('index', 0) != 0:
                                continue
                            if choice.get('prompt_token_ids') is not None:
                                prompt_token_ids = choice['prompt_token_ids']
                            if choice.get('token_ids'):
                                output_token_ids.extend(choice['token_ids'])
                            delta = choice.get('delta') or {}
                            if endpoint == '/v1/completions':
                                delta = {'content': choice.get('text', '')}
                            for part, value in [('reasoning', delta.get('reasoning_content') or delta.get('reasoning')),
                                                ('content', delta.get('content'))]:
                                if isinstance(value, str) and value:
                                    if record['ttft_s'] is None:
                                        record['ttft_s'] = elapsed
                                    record['last_token_s'] = elapsed
                                    parts[part].append(value)
                                    detect(part, value)
                            if choice.get('finish_reason'):
                                record['finish_reason'] = choice['finish_reason']
                    if record['finish_reason'] in ('error', 'abort'):
                        raise RuntimeError(f'vLLM terminated with {record["finish_reason"]}')
                    if record['done_received'] or record['finish_reason'] in ('stop', 'length'):
                        # A token cap may cut Answer: 070 after the first digit.
                        # Only a natural stream end can complete an unfinished line.
                        if record['finish_reason'] != 'length':
                            for part in parts:
                                detect(part, '', eof=True)
                        record['status'] = 'completed'
                    else:
                        raise RuntimeError('Stream ended without DONE or a terminal finish reason')
        except asyncio.CancelledError:
            record['status'] = 'cancelled'
            raise
        except Exception as exc:
            record.update(status='error', error=f'{type(exc).__name__}: {exc}')
        finally:
            scope.inc('generation_' + record['status'])
            ended = time.perf_counter()
            record.update(generation_finished_at_utc=utc_now(), generation_end_monotonic_s=ended,
                          generation_latency_s=ended - began,
                          generation_censored=record['status'] != 'completed' or record['finish_reason'] == 'length')
            visible_text = ''.join(parts['reasoning']) + ''.join(parts['content'])
            usage = record['usage'] or {}
            cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
            record.update(generated_token_ids_count=len(output_token_ids),
                          prompt_token_ids_count=len(prompt_token_ids) if prompt_token_ids else None,
                          cached_prompt_tokens=cached,
                          prefix_cache_hit_fraction=cached / len(prompt_token_ids) if cached is not None and prompt_token_ids else None)
            if args.strategy == 'coverage':
                complete_ids = bool(prompt_token_ids and output_token_ids and
                                    usage.get('completion_tokens') == len(output_token_ids))
                write_json(path / 'tokens.json', {'prompt_token_ids': prompt_token_ids,
                    'output_token_ids': output_token_ids, 'complete': complete_ids,
                    'visible_text': (continuation['visible_text'] if continuation else '') + visible_text}, scope)
            write_json(path / 'response.json', {part: ''.join(text) for part, text in parts.items()}, scope)
            record['overhead'] = scope.snapshot()
            write_json(path / 'telemetry.json', record, scope)

    streams = [asyncio.create_task(generate(rollout_offset + r)) for r in range(1, args.rollouts + 1)]

    async def drained():
        try:
            await asyncio.gather(*streams)
        finally:
            candidates.put_nowait(None)

    producer = asyncio.create_task(drained())
    try:
        async with asyncio.timeout(args.question_timeout):
            with (folder / 'verification.jsonl').open('a') as file:
                while True:
                    event = await candidates.get()
                    if event is None:
                        break
                    queue_wait = time.perf_counter() - event.pop('_enqueued_monotonic_s')
                    meter.observe('candidate_local_queue_wait', queue_wait)
                    event.update(verification_started_at_utc=utc_now(), round=round_no,
                                 candidate_queue_wait_s=queue_wait)
                    verify_start = time.perf_counter()
                    meter.inc('verification_submitted')
                    if profiler:
                        profiler.submitted(verify_start - (attempt_start if attempt_start is not None else start))
                    try:
                        response = await client.post(args.grader_url + '/verify', json={
                            'index': index, 'candidate': event['candidate'],
                            'agent_id': f'{output.name}-q{index}-r{event["rollout"]}',
                            'query_id': f'{output.name}-q{index}-a{event["candidate"]}'})
                        response.raise_for_status()
                        verdict = response.json()
                        if type(verdict.get('verdict')) is not bool:
                            raise RuntimeError('Grader response lacks a boolean verdict')
                        event['result'] = verdict
                        meter.inc('verification_completed')
                        meter.inc('verification_correct' if verdict['verdict'] else 'verification_wrong')
                        if verdict.get('queue_wait_s') is not None:
                            meter.observe('grader_queue_wait', verdict['queue_wait_s'])
                        if verdict.get('toll_s') is not None:
                            meter.observe('grader_service', verdict['toll_s'])
                    except asyncio.CancelledError:
                        event['cancelled'] = True
                        meter.inc('verification_cancelled')
                        raise
                    except Exception as exc:
                        meter.inc('verification_errors')
                        event['error'] = f'{type(exc).__name__}: {exc}'
                        raise
                    finally:
                        event.update(verification_finished_at_utc=utc_now(),
                                     verification_latency_s=time.perf_counter() - verify_start)
                        meter.observe('verification_http_wait', event['verification_latency_s'])
                        with meter.measure('verification_trace_write_flush'):
                            append_json(file, event)
                    if verdict['verdict']:
                        winner = event
                        solved_event = {'problem_idx': index, 'round': round_no, 'rollout': event['rollout'],
                                        'candidate': event['candidate'], 'first_solved_at_utc': utc_now(),
                                        'first_solved_elapsed_s': time.perf_counter() - (attempt_start if attempt_start is not None else start),
                                        'grader_answered_at_utc': verdict.get('answered_at'),
                                        'grader_query_id': verdict.get('query_id')}
                        with (output / 'solved.jsonl').open('a') as solved_file:
                            append_json(solved_file, solved_event)
                        break
    except asyncio.CancelledError:
        stopped_for_target = target_event is not None and target_event.is_set()
        question_error = None if stopped_for_target else 'attempt interrupted'
        raise
    except Exception as exc:
        question_error = f'{type(exc).__name__}: {exc}'
    finally:
        cancellation_start = time.perf_counter()
        for task in streams:
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*streams, return_exceptions=True)
        await asyncio.gather(producer, return_exceptions=True)
        meter.observe('generation_cancellation_settlement', time.perf_counter() - cancellation_start)
        ended, ended_at = time.perf_counter(), utc_now()
        for path, record, _ in records:
            record.update(finished_at_utc=ended_at, end_to_end_latency_s=ended - record['start_monotonic_s'],
                          end_to_end_scope='rollout start through question verification/cancellation settlement',
                          gpu=None)
            with meter.measure('gpu_sample_window'):
                record['gpu'] = sampler.window(record['start_monotonic_s'], record['generation_end_monotonic_s'])
            record['overhead'] = rollout_meters[record['rollout']].snapshot()
            write_json(path / 'telemetry.json', record)
        result = {'problem_idx': index, 'started_at_utc': started, 'finished_at_utc': ended_at,
                  'end_to_end_latency_s': ended - start,
                  'status': 'error' if question_error else ('solved' if winner else ('stopped' if stopped_for_target else 'unsolved')),
                  'winner': winner, 'error': question_error, 'unique_candidates': len(seen), 'candidate_answers': sorted(seen),
                  'rollouts': [record for _, record, _ in sorted(records, key=lambda r: r[1]['rollout'])]}
        if any(r['status'] == 'error' for r in result['rollouts']) and not winner:
            result['status'] = 'error'
        result.update(round=round_no, first_solved=solved_event, overhead=meter.snapshot())
        write_json(folder / f'round-{round_no:02d}.json', result)
        rounds = (previous.get('rounds', []) if previous else []) + [
            {k: result[k] for k in ('round', 'status', 'started_at_utc', 'finished_at_utc', 'end_to_end_latency_s', 'winner', 'error')}]
        if previous:
            result['rollouts'] = previous['rollouts'] + result['rollouts']
            result['started_at_utc'] = previous['started_at_utc']
            result['first_solved'] = previous.get('first_solved') or solved_event
        result['overhead'] = merge_meters(previous.get('overhead') if previous else None, meter.snapshot())
        result['rounds'] = rounds
        result['question_start_monotonic_s'] = previous.get('question_start_monotonic_s', start) if previous else start
        result['end_to_end_latency_s'] = ended - result['question_start_monotonic_s']
        write_json(folder / 'question.json', result)
        if solved_event and on_solved:
            on_solved(solved_event)
        print(f'Q{index:02d} round {round_no}: {result["status"]}, {ended - start:.2f}s', flush=True)
    return result


async def run_questions(problems, args, client, output, sampler, *, round_no=1,
                        rollout_offset=0, attempt_start=None, on_solved=None, target_event=None, profiler=None):
    queue = asyncio.Queue()
    for problem in problems:
        queue.put_nowait(problem)
    results = []

    async def worker():
        while not queue.empty() and not (target_event is not None and target_event.is_set()):
            problem = queue.get_nowait()
            results.append(await run_question(problem, args, client, output, sampler,
                round_no=round_no, rollout_offset=rollout_offset, attempt_start=attempt_start,
                on_solved=on_solved, target_event=target_event, profiler=profiler))

    tasks = [asyncio.create_task(worker()) for _ in range(min(args.parallelism, len(problems)))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return sorted(results, key=lambda row: row['problem_idx'])


async def run_coverage(problems, args, client, output, sampler, attempt_start, profiler=None):
    """Barriered single-rollout rounds; stop all remaining work at the solve target."""
    solved = set()
    target_event = asyncio.Event()

    def on_solved(event):
        solved.add(event['problem_idx'])
        print(f'Solved coverage: {len(solved)}/{args.target_correct}', flush=True)
        if len(solved) >= args.target_correct:
            target_event.set()

    for round_no in range(1, min(args.max_rounds, args.max_attempts_per_question) + 1):
        pending = [p for p in problems if p['problem_idx'] not in solved]
        round_args = copy(args)
        round_args.max_tokens = args.first_pass_max_tokens if round_no == 1 else args.max_tokens
        print(f'Coverage round {round_no}: {len(pending)} questions, {round_args.max_tokens} tokens each', flush=True)
        batch = asyncio.create_task(run_questions(pending, round_args, client, output, sampler,
            round_no=round_no, rollout_offset=round_no - 1, attempt_start=attempt_start,
            on_solved=on_solved, target_event=target_event, profiler=profiler))
        target_wait = asyncio.create_task(target_event.wait())
        try:
            await asyncio.wait([batch, target_wait], return_when=asyncio.FIRST_COMPLETED)
            if target_event.is_set():
                batch.cancel()
                await asyncio.gather(batch, return_exceptions=True)
                break
            result = await batch
            if any(q['status'] == 'error' for q in result):
                break
        finally:
            target_wait.cancel()
            if not batch.done() and not batch.cancelling():
                batch.cancel()
            await asyncio.gather(batch, target_wait, return_exceptions=True)
    return [json.loads(p.read_text()) for p in sorted(output.glob('trace/*/question.json'))]


def ensure_free(port):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))


class Services:
    def __init__(self):
        self.processes = []
        self.logs = []

    def launch(self, command, logfile, env=None):
        file = logfile.open('w')
        self.logs.append(file)
        process = subprocess.Popen(command, stdout=file, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True, cwd=ROOT)
        self.processes.append(process)
        return process

    async def close(self):
        for process in reversed(self.processes):
            # Own the entire process group, including vLLM EngineCore children.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.perf_counter() + 15
        while any(p.poll() is None for p in self.processes) and time.perf_counter() < deadline:
            await asyncio.sleep(0.1)
        for process in self.processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for file in self.logs:
            file.close()


async def ready(client, url, timeout, process=None):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'Service exited ({process.returncode}); inspect attempt service logs')
        try:
            response = await client.get(url, timeout=2)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            await asyncio.sleep(0.5)
    raise TimeoutError(f'Service not ready: {url}')


@contextmanager
def attempt_lock():
    with (ROOT / '.attempt.lock').open('a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another canonical attempt owns this checkout') from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def load_questions(indices, year=2025):
    from src.benchmarks import load_questions as load_benchmark_questions
    return load_benchmark_questions(indices, year)



async def warm_inference(args, client, question_count):
    """Warm the configured sampling path at the attempt's maximum batch size."""
    batch_size = min(args.parallelism, question_count) * args.rollouts
    tokens = min(32, args.max_tokens)
    started, start = utc_now(), time.perf_counter()
    print(f'Inference warmup: {batch_size} streams, {tokens} tokens each', flush=True)

    async def one(slot):
        request = {'model': args.model,
                   'messages': [{'role': 'user', 'content': 'Compute 1 + 1.'}],
                   'max_tokens': tokens, 'min_tokens': tokens,
                   'temperature': args.temperature, 'top_p': args.top_p,
                   'seed': args.seed - batch_size + slot, 'stream': False}
        if args.disable_thinking:
            request['chat_template_kwargs'] = {'enable_thinking': False}
        response = await client.post(args.vllm_url + '/v1/chat/completions', json=request)
        response.raise_for_status()
        return {'request': request, 'response': response.json()}

    tasks = [asyncio.create_task(one(slot)) for slot in range(batch_size)]
    try:
        responses = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return {'started_at_utc': started, 'finished_at_utc': utc_now(),
            'latency_s': time.perf_counter() - start, 'batch_size': batch_size,
            'tokens_per_request': tokens, 'requests': responses}


async def run(args):
    services = Services()
    sampler = None
    began = time.perf_counter()
    output = ROOT / 'attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True)
    print(f'Attempt artifacts: {output}', flush=True)
    profiler = AttemptProfiler(output, enabled=not args.no_overhead_profile,
                               interval=args.overhead_interval, engine_interval=args.engine_metrics_interval)
    status, results, error = 'initializing', [], None
    official_start = None
    config = {**vars(args), 'attempt_id': output.name, 'initialization_started_at_utc': utc_now(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True)),
              'system_prompt': PROMPT, 'grading': 'single vendored grader; no local answer-key comparisons',
              'gpu_scope': 'device-level NVML; vLLM preallocates VRAM', 'python': sys.version}
    atomic_json(output / 'config.json', config)
    try:
        config['dataset_provenance'] = dataset_provenance(args.benchmark_year, args.benchmark_role)
        problems = load_questions(args.questions, args.benchmark_year)
        config['question_indices'] = [p['problem_idx'] for p in problems]
        if args.strategy == 'coverage' and args.target_correct > len(problems):
            raise ValueError('Target correct exceeds the number of selected questions')
        profile_path = Path(args.models_dir).expanduser() / args.model / 'vllm.yaml'
        profile_text = profile_path.read_text()
        profile = yaml.safe_load(profile_text)
        overrides = profile.get('override-generation-config', {})
        if isinstance(overrides, str):
            overrides = json.loads(overrides)
        cap = overrides.get('max_new_tokens')
        if cap is not None and max(args.max_tokens, args.first_pass_max_tokens if args.strategy == 'coverage' else 0) > cap:
            raise ValueError(f'Requested max_tokens exceeds model profile ceiling ({cap})')
        atomic_json(output / 'model_profile.json', {'path': str(profile_path), 'yaml': profile_text,
                    'sha256': hashlib.sha256(profile_text.encode()).hexdigest()})
        ensure_free(args.grader_port)
        if not args.reuse_server:
            ensure_free(args.vllm_port)
        sampler = GPUSampler(output / 'gpu.jsonl', args.gpu_interval, args.gpu_device)
        await sampler.start()
        limits = httpx.Limits(max_connections=args.parallelism * (args.rollouts + 1) + 8,
                              max_keepalive_connections=args.parallelism * (args.rollouts + 1) + 8)
        timeout = httpx.Timeout(connect=10, read=args.question_timeout, write=30, pool=30)
        async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
            if not args.reuse_server:
                # Small CUDA matmul warmup before loading the model; cap allocation.
                gpu_warmup = services.launch([args.vllm_python, '-c',
                    'import torch,time; torch.cuda.set_device(' + str(args.gpu_device) + '); '
                    'x=torch.randn((1024,1024),device="cuda",dtype=torch.bfloat16); '
                    'end=time.monotonic()+2; '\
                    '\nwhile time.monotonic()<end: y=x@x; torch.cuda.synchronize()'], output / 'gpu_warmup.log')
                with profiler.meter.measure('initialization_gpu_warmup_wait', cpu=False):
                    await asyncio.wait_for(asyncio.to_thread(gpu_warmup.wait), timeout=120)
                if gpu_warmup.returncode:
                    raise RuntimeError('CUDA warmup failed; inspect gpu_warmup.log')
                command = [args.vllm_binary, 'serve', '--config', str(profile_path),
                           '--host', '127.0.0.1', '--port', str(args.vllm_port), '--enable-prompt-tokens-details']
                config['vllm_command'] = command
                server = services.launch(command, output / 'vllm.log')
            else:
                server = None
            with profiler.meter.measure('initialization_server_ready_wait', cpu=False):
                models = await ready(client, args.vllm_url + '/v1/models', args.startup_timeout, server)
            if args.model not in [m['id'] for m in models.get('data', [])]:
                raise RuntimeError('Inference server does not serve the requested model')
            atomic_json(output / 'server_models.json', models)
            served = next(m for m in models['data'] if m['id'] == args.model)
            args.max_context_tokens = int(served.get('max_model_len') or profile['max-model-len'])
            config['max_context_tokens'] = args.max_context_tokens
            grader_config = {'dataset': {'source': str(benchmark_paths(args.benchmark_year)[1]),
                             'format': 'jsonl', 'idx_field': 'problem_idx', 'gold_field': 'answer'},
                             'cost_c': args.grader_cost, 'host': '127.0.0.1', 'port': args.grader_port,
                             'audit_log': str(output / 'grader_audit.jsonl')}
            grader_config['dataset'].update(id=config['dataset_provenance']['id'], year=args.benchmark_year, revision=config['dataset_provenance']['revision'])
            (output / 'grader_config.yaml').write_text(yaml.safe_dump(grader_config))
            grader = services.launch([args.grader_python, str(ROOT / 'grader/server.py')], output / 'grader.log',
                                     {**os.environ, 'GRADER_CONFIG': str(output / 'grader_config.yaml')})
            with profiler.meter.measure('initialization_grader_ready_wait', cpu=False):
                health = await ready(client, args.grader_url + '/health', 60, grader)
            if health.get('queries_so_far') != 0 or health.get('cost_c') != args.grader_cost:
                raise RuntimeError('Grader did not start with a fresh queue and requested toll')
            if health.get('dataset', {}).get('sha256') != config['dataset_provenance']['grader_sha256']:
                raise RuntimeError('Grader loaded a different benchmark answer key')
            config['grader_health'] = health
            with profiler.meter.measure('initialization_inference_warmup_wait', cpu=False):
                warmup = await warm_inference(args, client, len(problems))
            atomic_json(output / 'inference_warmup.json', warmup)
            config['inference_warmup'] = {k: warmup[k] for k in ('latency_s', 'batch_size', 'tokens_per_request')}
            config.update(model_profile_sha256=hashlib.sha256(profile_text.encode()).hexdigest(),
                          launch_profile=profile, official_started_at_utc=utc_now())
            atomic_json(output / 'config.json', config)
            official_start = time.perf_counter()
            profiler.official_start(client, args.vllm_url)
            status = 'running'
            print('Official solving phase started', flush=True)
            if args.strategy == 'coverage':
                results = await run_coverage(problems, args, client, output, sampler, official_start, profiler)
            else:
                results = await run_questions(problems, args, client, output, sampler, attempt_start=official_start, profiler=profiler)
            await profiler.stop()
            status = 'completed' if all(r['status'] != 'error' for r in results) else 'failed'
            if sampler.error:
                raise RuntimeError(f'GPU telemetry failed: {sampler.error}')
    except asyncio.CancelledError:
        status, error = 'interrupted', 'Attempt interrupted by signal'
        raise
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        await profiler.stop()
        if sampler:
            sampler.stop()
        official_end = time.perf_counter()
        # Include partially completed questions after interruption/failure.
        results = [json.loads(p.read_text()) for p in sorted(output.glob('trace/*/question.json'))]
        summary = {'attempt_id': output.name, 'status': status, 'error': error,
                   'official_started_at_utc': config.get('official_started_at_utc'),
                   'official_finished_at_utc': utc_now(),
                   'official_latency_s': official_end - official_start if official_start else None,
                   'initialization_and_attempt_latency_s': official_end - began,
                   'strategy': args.strategy, 'solved': sum(r['status'] == 'solved' for r in results),
                   'questions_completed': sum(r['status'] != 'stopped' for r in results), 'questions_attempted': len(results),
                   'target_correct': args.target_correct if args.strategy == 'coverage' else None,
                   'target_reached': sum(r['status'] == 'solved' for r in results) >= args.target_correct if args.strategy == 'coverage' else None,
                   'rounds_executed': max((r['round'] for r in results), default=0),
                   'questions': [{**{k: r[k] for k in ('problem_idx', 'status', 'end_to_end_latency_s', 'unique_candidates')},
                                  'verified_answer': r['winner']['candidate'] if r['winner'] else None,
                                  'winning_rollout': r['winner']['rollout'] if r['winner'] else None,
                                  'first_solved': r.get('first_solved')} for r in results]}
        summary['overhead'] = profiler.snapshot()
        atomic_json(output / 'summary.json', summary)  # Durable even if service cleanup fails.
        atomic_json(output / 'config.json', config)
        atomic_json(output / 'metadata.json', build_metadata(output, config))
        cleanup_start = time.perf_counter()
        await services.close()
        summary['service_cleanup_latency_s'] = time.perf_counter() - cleanup_start
        summary['grader_timeline'] = grader_timeline(output / 'grader_audit.jsonl',
            config.get('official_started_at_utc'), args.target_correct, args.grader_cost)
        atomic_json(output / 'overhead.json', summary['overhead'])
        atomic_json(output / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    if status == 'failed':
        raise RuntimeError('One or more questions failed; inspect trace telemetry')
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument('--model', required=True, help='Model ID with ~/models/<ID>/vllm.yaml')
    parser.add_argument('--models-dir', default='~/models')
    parser.add_argument('--vllm-python', default=str(Path('~/.venvs/vllm/bin/python').expanduser()))
    parser.add_argument('--vllm-binary', default=str(Path('~/.venvs/vllm/bin/vllm').expanduser()))
    parser.add_argument('--grader-python', default=str(ROOT / 'grader/.venv/bin/python'))
    parser.add_argument('--reuse-server', action='store_true')
    parser.add_argument('--vllm-port', type=int, default=8000)
    parser.add_argument('--grader-port', type=int, default=8077)
    parser.add_argument('--grader-cost', type=float, default=3.0)
    parser.add_argument('--strategy', choices=('fanout', 'coverage'), default='fanout')
    parser.add_argument('--parallelism', type=int, help='Concurrent questions: fanout default 8, coverage default 30')
    parser.add_argument('--rollouts', type=int, help='Streams per question: fanout default 4, coverage requires 1')
    parser.add_argument('--first-pass-max-tokens', type=int, default=8192)
    parser.add_argument('--no-continuation', action='store_true', help='Coverage: use fresh samples even after capped outputs')
    parser.add_argument('--target-correct', type=int, default=18)
    parser.add_argument('--max-rounds', type=int, default=4, help='Coverage round bound, including first pass')
    parser.add_argument('--max-attempts-per-question', type=int, default=4, help='Counts every generation request, including continuations')
    parser.add_argument('--questions', type=int, nargs='+', help='Smoke subset; default all 30')
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--question-timeout', type=float, default=1800)
    parser.add_argument('--gpu-interval', type=float, default=0.2)
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--no-overhead-profile', action='store_true')
    parser.add_argument('--overhead-interval', type=float, default=0.05, help='Event-loop lag sampling seconds')
    parser.add_argument('--engine-metrics-interval', type=float, default=1.0, help='vLLM metrics polling seconds')
    args = parser.parse_args(argv)
    args.benchmark_role = args.benchmark_role or ("development" if args.benchmark_year == 2025 else "generalization")
    if args.parallelism is None:
        args.parallelism = 30 if args.strategy == 'coverage' else 8
    if args.rollouts is None:
        args.rollouts = 1 if args.strategy == 'coverage' else 4
    if args.rollouts > args.max_attempts_per_question:
        parser.error('Rollouts exceed the per-question attempt limit')
    if args.strategy == 'coverage' and args.rollouts != 1:
        parser.error('Coverage strategy requires one rollout per question per round')
    if any(getattr(args, key) <= 0 for key in ('parallelism', 'rollouts', 'max_tokens', 'startup_timeout', 'question_timeout', 'gpu_interval', 'max_rounds', 'first_pass_max_tokens', 'target_correct', 'max_attempts_per_question', 'overhead_interval', 'engine_metrics_interval')):
        parser.error('Concurrency, token budgets, timeouts, and sampling interval must be positive')
    if args.grader_cost < 0 or args.gpu_device < 0 or not 0 < args.top_p <= 1 or args.temperature < 0:
        parser.error('Invalid grader cost, GPU device, or sampling settings')
    if '/' not in args.model or any(part in ('', '.', '..') for part in args.model.split('/')) or args.model.startswith('/'):
        parser.error('Use a relative organization/model ID')
    if not all(1 <= port <= 65535 for port in (args.vllm_port, args.grader_port)) or args.vllm_port == args.grader_port:
        parser.error('Service ports must be valid and distinct')
    args.max_context_tokens = 32768  # replaced by actual /v1/models metadata during initialization
    args.vllm_url = f'http://127.0.0.1:{args.vllm_port}'
    args.grader_url = f'http://127.0.0.1:{args.grader_port}'
    return args


def main():
    args = parse_args()
    with attempt_lock():
        async def entry():
            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
            await run(args)
        asyncio.run(entry())


if __name__ == '__main__':
    main()
