"""Offline canonical-runner checks with timed fake SSE and a real HTTP grader."""
import asyncio
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import httpx
import yaml

from src.attempt import CandidateDetector, ROOT, parse_args, ready, run_question, run_questions, sse_payloads, warm_inference, run_coverage, continuation_prefix


class FakeGPU:
    def window(self, start, end):
        return {'observed_peak_vram_mib': 123, 'sample_count': 1}


def chunk(text, part='content', finish=None):
    return ('data: ' + json.dumps({'choices': [{'index': 0, 'delta': {part: text}, 'finish_reason': finish}]}) + '\n\n').encode()


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0.002, hang=False):
        self.chunks, self.delay, self.hang = chunks, delay, hang
        self.closed = False

    async def __aiter__(self):
        for value in self.chunks:
            await asyncio.sleep(self.delay)
            yield value
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class DetectorTests(unittest.TestCase):
    def test_split_box_and_digits_do_not_submit_prefix(self):
        d = CandidateDetector()
        for value in ['candidate \\bo', 'xed{0', '7', '0']:
            self.assertEqual(d.feed('reasoning', value), [])
        self.assertEqual(d.feed('reasoning', '}')[0]['answer'], 70)
        self.assertEqual(d.feed('reasoning', ' continuing'), [])

    def test_closed_boxes_lines_and_channel_independence(self):
        d = CandidateDetector()
        self.assertEqual(d.feed('content', 'Answer: 7'), [])
        self.assertEqual(d.feed('reasoning', '0\n'), [])
        self.assertEqual(d.feed('content', '0\n')[0]['answer'], 70)
        self.assertEqual(d.feed('reasoning', '\\boxed{1000}'), [])
        self.assertEqual(d.feed('content', '\\boxed{001}, \\boxed{002}')[0]['answer'], 1)
        self.assertEqual(d.feed('reasoning', '\nAnswer: 082'), [])
        self.assertEqual(d.feed('reasoning', '', eof=True)[0]['answer'], 82)

    def test_prose_answers_need_complete_delimited_integer_not_fraction_or_expression(self):
        d = CandidateDetector()
        self.assertEqual(d.feed('reasoning', 'Final answer is 1'), [])
        self.assertEqual(d.feed('reasoning', '17.\n')[0]['answer'], 117)
        self.assertEqual(d.feed('content', 'I am fairly certain the answer is 49.\n')[0]['answer'], 49)
        for text in ['The answer is $4/17$.\n', 'The answer is 4 + 17.\n',
                     'The answer is 1000.\n', 'The answer is 4.5.\n', 'The answer is 1,000.\n']:
            self.assertEqual(CandidateDetector().feed('content', text), [], text)

    def test_invalid_args(self):
        for args in [['--parallelism', '0'], ['--max-tokens', '-1'], ['--grader-port', '8000']]:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--model', 'Qwen/Qwen3.5-4B', *args])
        args = parse_args(['--model', 'Qwen/Qwen3.5-4B'])
        self.assertEqual((args.parallelism, args.rollouts, args.max_tokens), (8, 4, 16384))
        coverage = parse_args(['--model', 'Qwen/Qwen3.5-4B', '--strategy', 'coverage'])
        self.assertEqual((coverage.parallelism, coverage.rollouts, coverage.first_pass_max_tokens, coverage.max_attempts_per_question), (30, 1, 8192, 4))


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def args(self, **changes):
        args = parse_args(['--model', 'Qwen/Qwen3.5-4B', '--question-timeout', '2'])
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    async def test_warmup_matches_sampling_batch_and_excludes_grader(self):
        requests, active, peak = [], 0, 0
        async def handler(request):
            nonlocal active, peak
            self.assertEqual(request.url.path, '/v1/chat/completions')
            body = json.loads(request.content)
            requests.append(body)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.005)
            active -= 1
            return httpx.Response(200, json={'choices': []})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await warm_inference(self.args(disable_thinking=True), client, 30)
        self.assertEqual((result['batch_size'], peak), (32, 32))
        self.assertEqual(len({r['seed'] for r in requests}), 32)
        for request in requests:
            self.assertEqual((request['max_tokens'], request['min_tokens']), (32, 32))
            self.assertEqual((request['temperature'], request['top_p']), (0.8, 0.95))
            self.assertEqual(request['chat_template_kwargs'], {'enable_thinking': False})

    async def test_sse_multiline_comments_and_eof(self):
        async def lines():
            for line in [': ping', 'data: {', 'data: }', '', 'data: [DONE]']:
                yield line
        self.assertEqual([s async for s in sse_payloads(lines())], ['{\n}', '[DONE]'])

    async def test_correct_verdict_cancels_streams_but_generation_continues_during_grading(self):
        streams, submitted = [], []
        after_candidate = asyncio.Event()

        async def handler(request):
            if request.url.path == '/verify':
                body = json.loads(request.content)
                submitted.append(body['candidate'])
                await after_candidate.wait()
                return httpx.Response(200, json={'verdict': body['candidate'] == '70'})
            stream = Stream([chunk('\\boxed{0', 'reasoning_content'), chunk('70}\n', 'reasoning_content'),
                             chunk('continued checking')], hang=True)
            old = stream.__class__
            class ProgressStream(old):
                async def __aiter__(self):
                    async for value in super().__aiter__():
                        if b'continued checking' in value:
                            after_candidate.set()
                        yield value
            stream.__class__ = ProgressStream
            streams.append(stream)
            return httpx.Response(200, stream=stream)

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await run_question({'problem_idx': 1, 'problem': 'test'}, self.args(), client, Path(tmp), FakeGPU())
            self.assertEqual(result['status'], 'solved')
            self.assertEqual(submitted, ['70'])  # deduplicated across all four rollouts
            self.assertTrue(all(s.closed for s in streams))
            self.assertEqual(len(result['rollouts']), 4)
            for record in result['rollouts']:
                self.assertEqual(record['status'], 'cancelled')
                self.assertGreater(record['ttft_s'], 0)
                self.assertGreaterEqual(record['end_to_end_latency_s'], record['generation_latency_s'])
                self.assertEqual(record['gpu']['observed_peak_vram_mib'], 123)
            self.assertIn('continued checking', (Path(tmp) / 'trace/01/rollout-01/response.json').read_text())

    async def test_wrong_answer_then_correct_at_eof(self):
        submitted = []
        async def handler(request):
            if request.url.path == '/verify':
                candidate = json.loads(request.content)['candidate']
                submitted.append(candidate)
                return httpx.Response(200, json={'verdict': candidate == '70'})
            return httpx.Response(200, stream=Stream([chunk('\\boxed{69}\nAnswer: 070'),
                chunk('', finish='stop'), b'data: [DONE]\n\n']))
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await run_question({'problem_idx': 1, 'problem': 'test'}, self.args(), client, Path(tmp), FakeGPU())
            self.assertEqual(submitted, ['69', '70'])
            self.assertEqual(result['status'], 'solved')
            self.assertTrue(any(r['done_received'] for r in result['rollouts']))

    async def test_token_cap_does_not_complete_a_partial_answer_number(self):
        async def handler(request):
            self.assertNotEqual(request.url.path, '/verify')
            return httpx.Response(200, stream=Stream([chunk('Answer: 7'),
                chunk('', finish='length'), b'data: [DONE]\n\n']))
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await run_question({'problem_idx': 1, 'problem': 'test'}, self.args(), client, Path(tmp), FakeGPU())
            self.assertEqual((result['status'], result['unique_candidates']), ('unsolved', 0))
            self.assertTrue(all(r['generation_censored'] for r in result['rollouts']))

    async def test_errors_and_timeout_persist_partial_traces(self):
        for mode in ['malformed', 'http', 'timeout', 'grader']:
            streams = []
            async def handler(request):
                if request.url.path == '/verify':
                    return httpx.Response(500, json={'error': 'oracle unavailable'})
                if mode == 'http':
                    return httpx.Response(500)
                stream = Stream([b'data: broken-json\n\n'] if mode == 'malformed' else [chunk('\\boxed{70}' if mode == 'grader' else 'partial')], hang=True)
                streams.append(stream)
                return httpx.Response(200, stream=stream)
            with tempfile.TemporaryDirectory() as tmp:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    result = await run_question({'problem_idx': 1, 'problem': 'test'}, self.args(question_timeout=0.05), client, Path(tmp), FakeGPU())
                self.assertEqual(result['status'], 'error', mode)
                self.assertTrue(all(s.closed for s in streams))
                self.assertTrue((Path(tmp) / 'trace/01/rollout-01/telemetry.json').exists())

    async def test_interruption_closes_streams_and_saves_question(self):
        streams = []
        started = asyncio.Event()
        async def handler(request):
            stream = Stream([chunk('partial reasoning')], hang=True)
            streams.append(stream)
            started.set()
            return httpx.Response(200, stream=stream)
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                task = asyncio.create_task(run_question({'problem_idx': 1, 'problem': 'test'}, self.args(), client, Path(tmp), FakeGPU()))
                await started.wait()
                await asyncio.sleep(0.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertTrue(all(s.closed for s in streams))
            record = json.loads((Path(tmp) / 'trace/01/question.json').read_text())
            self.assertEqual(record['error'], 'attempt interrupted')
            self.assertTrue(all(r['status'] == 'cancelled' for r in record['rollouts']))

    async def test_coverage_continues_exact_tokens_and_stitches_split_answer(self):
        requests = []
        args = self.args(strategy='coverage', parallelism=30, rollouts=1,
                         first_pass_max_tokens=1, max_tokens=2, target_correct=1)
        args.max_context_tokens = 5
        async def handler(request):
            if request.url.path == '/verify':
                self.assertEqual(json.loads(request.content)['candidate'], '70')
                return httpx.Response(200, json={'verdict': True, 'query_id': 'oracle-1',
                                                'answered_at': '2026-10-03T20:00:00Z'})
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if len(requests) == 1:
                event = {'prompt_token_ids': [10, 11], 'choices': [{'index': 0,
                    'delta': {'reasoning_content': '\\boxed{0'}, 'token_ids': [70], 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1}}
            else:
                event = {'choices': [{'index': 0, 'text': '70}', 'prompt_token_ids': [10, 11, 70],
                    'token_ids': [71, 72], 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 2,
                              'prompt_tokens_details': {'cached_tokens': 2}}}
            return httpx.Response(200, stream=Stream([('data: '+json.dumps(event)+'\n\n').encode(), b'data: [DONE]\n\n']))
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await run_coverage([{'problem_idx': 1, 'problem': 'test'}], args, client, Path(tmp), FakeGPU(), time.perf_counter())
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[1][0], '/v1/completions')
            self.assertEqual(requests[1][1]['prompt'], [10, 11, 70])
            self.assertNotIn('messages', requests[1][1])
            self.assertEqual(requests[1][1]['max_tokens'], 2)
            q = result[0]
            self.assertEqual(q['status'], 'solved')
            self.assertEqual(len(q['rounds']), 2)
            self.assertEqual(q['rollouts'][1]['continuation_of_rollout'], 1)
            self.assertEqual(q['rollouts'][1]['cached_prompt_tokens'], 2)
            self.assertEqual(q['first_solved']['grader_query_id'], 'oracle-1')
            events = [json.loads(line) for line in (Path(tmp)/'solved.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0], q['first_solved'])

    async def test_coverage_target_cancels_other_questions_after_solve_records_saved(self):
        streams = []
        args = self.args(strategy='coverage', parallelism=30, rollouts=1, target_correct=2)
        async def handler(request):
            if request.url.path == '/verify':
                await asyncio.sleep(0.01)
                return httpx.Response(200, json={'verdict': True})
            q = int(json.loads(request.content)['messages'][1]['content'])
            stream = Stream([chunk('\\boxed{70}' if q <= 2 else 'still thinking')], hang=True)
            streams.append(stream)
            return httpx.Response(200, stream=stream)
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await run_coverage([{'problem_idx': q, 'problem': str(q)} for q in range(1, 6)], args, client, Path(tmp), FakeGPU(), time.perf_counter())
            self.assertEqual(len(result), 5)
            self.assertEqual(sum(q['status'] == 'solved' for q in result), 2)
            self.assertEqual(sum(q['status'] == 'stopped' for q in result), 3)
            self.assertTrue(all(s.closed for s in streams))
            self.assertTrue(all(q['error'] is None for q in result))

    async def test_coverage_budget_hard_caps_each_question_at_four_requests(self):
        requests = []
        args = self.args(strategy='coverage', parallelism=30, rollouts=1,
                         target_correct=1, max_rounds=20, no_continuation=True)
        async def handler(request):
            self.assertNotEqual(request.url.path, '/verify')
            requests.append(json.loads(request.content))
            return httpx.Response(200, stream=Stream([chunk('no answer', finish='stop'), b'data: [DONE]\n\n']))
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await run_coverage([{'problem_idx': 1, 'problem': 'test'}], args, client, Path(tmp), FakeGPU(), time.perf_counter())
            self.assertEqual(len(requests), 4)
            self.assertEqual(len(result[0]['rollouts']), 4)
            self.assertEqual(len({r['seed'] for r in requests}), 4)
            self.assertEqual(result[0]['status'], 'unsolved')

    def test_context_exhaustion_and_incomplete_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'rollout-01').mkdir()
            previous = {'rollouts': [{'rollout': 1, 'status': 'completed', 'finish_reason': 'length'}]}
            token_file = folder/'rollout-01/tokens.json'
            token_file.write_text(json.dumps({'prompt_token_ids': [1,2], 'output_token_ids':[3], 'complete':True, 'visible_text':'abc'}))
            self.assertIsNone(continuation_prefix(previous, folder, 4))
            token_file.write_text(json.dumps({'prompt_token_ids': [1,2], 'output_token_ids':[3], 'complete':False, 'visible_text':'abc'}))
            with self.assertRaises(RuntimeError):
                continuation_prefix(previous, folder, 10)

    async def test_question_parallelism_and_worker_reuse(self):
        active, peak, launched = set(), 0, []
        async def handler(request):
            nonlocal peak
            if request.url.path == '/verify':
                await asyncio.sleep(0.005)
                return httpx.Response(200, json={'verdict': True})
            body = json.loads(request.content)
            q = body['messages'][1]['content']
            active.add(q)
            peak = max(peak, len(active))
            launched.append(q)
            class TrackedStream(Stream):
                async def aclose(self):
                    await super().aclose()
                    active.discard(q)
            return httpx.Response(200, stream=TrackedStream([chunk('\\boxed{70}')], hang=True))
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await run_questions([{'problem_idx': i, 'problem': str(i)} for i in range(1, 6)],
                                            self.args(parallelism=2, rollouts=1), client, Path(tmp), FakeGPU())
            self.assertEqual(len(result), 5)
            self.assertEqual(peak, 2)
            self.assertEqual(set(launched), {'1', '2', '3', '4', '5'})

    async def test_vendored_grader_http_integration(self):
        import importlib.util
        if not all(importlib.util.find_spec(name) for name in ('sympy', 'loguru', 'regex', 'antlr4')):
            self.skipTest('Install grader/requirements-local.txt for HTTP integration')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            cfg = {'dataset': {'source': str(ROOT / 'grader/data/aime_2025.jsonl'), 'format': 'jsonl'},
                   'cost_c': 0.02, 'host': '127.0.0.1', 'port': port, 'audit_log': str(path / 'audit.jsonl')}
            (path / 'config.yaml').write_text(yaml.safe_dump(cfg))
            import os
            with (path / 'server.log').open('w') as log:
                process = subprocess.Popen([sys.executable, str(ROOT / 'grader/server.py')],
                    env={**os.environ, 'GRADER_CONFIG': str(path / 'config.yaml')}, stdout=log, stderr=log)
                try:
                    async with httpx.AsyncClient(trust_env=False) as client:
                        health = await ready(client, f'http://127.0.0.1:{port}/health', 10, process)
                        self.assertEqual(health['n_problems'], 30)
                        start = time.perf_counter()
                        responses = await asyncio.gather(*[client.post(f'http://127.0.0.1:{port}/verify',
                            json={'index': 1, 'candidate': c}) for c in ('69', '70')])
                        self.assertGreaterEqual(time.perf_counter() - start, 0.04)
                        self.assertEqual([r.json()['verdict'] for r in responses], [False, True])
                        self.assertTrue(all('gold' not in r.json() for r in responses))
                finally:
                    process.terminate()
                    process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
