"""Offline v1.1 policy checks: mocked HTTP/SSE only, no services or GPU work."""
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx
import yaml

from runner_final import run_v1_1
from runner_final.integrity import verify_core as verify_v1
from runner_final.integrity_v1_1 import verify_core
from src.attempt_runners import speedrun_v1_1 as runner
from src.attempt_metrics import AttemptProfiler
from src.attempt_storage import AttemptArtifacts, DisabledGPUSampler
from src.common import ROOT


def sse(text, finish="stop"):
    body = {"choices": [{"index": 0, "delta": {"content": text},
                         "finish_reason": finish}]}
    return ("data: " + json.dumps(body) + "\n\n").encode()


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, hang=False, closed=None):
        self.chunks, self.hang, self.on_close = chunks, hang, closed
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk
        if self.hang:
            await asyncio.Event().wait()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        if not self.closed and self.on_close:
            self.on_close()
        self.closed = True


class SettingsTests(unittest.TestCase):
    def test_preset_and_launch_profile_preserve_other_controls(self):
        args = run_v1_1.parse_args([])
        self.assertEqual((args.parallelism, args.rollouts, args.schedule), (30, 1, "eager"))
        self.assertEqual((args.first_pass_max_tokens, args.max_tokens), (65536, 65536))
        self.assertEqual((args.max_attempts_per_question, args.max_rounds), (4, 4))
        self.assertTrue(args.no_continuation)
        self.assertTrue(args.skip_benchmark_prewarm)
        self.assertTrue(args.no_overhead_profile)
        self.assertEqual((args.temperature, args.top_p, args.target_correct), (0.8, 0.95, 18))
        self.assertEqual(args.system_prompt, (ROOT / "runner_final/prompts/adherence_v1.txt").read_text())
        folder = ROOT / "configs/vllm/r0b0tlab/VibeThinker-3B-NVFP4"
        old = yaml.safe_load((folder / "vllm-flashinfer.yaml").read_text())
        new = yaml.safe_load((folder / runner.MODEL_PROFILE).read_text())
        self.assertEqual(json.loads(new.pop("override-generation-config")), {"max_new_tokens": 65536})
        self.assertEqual(json.loads(old.pop("override-generation-config")), {"max_new_tokens": 16384})
        self.assertEqual(new, old)

    def test_rejects_barrier_fanout_over30_and_fifth_request(self):
        for argv in (["--schedule", "barrier"], ["--rollouts", "2"],
                     ["--parallelism", "31"], ["--max-attempts-per-question", "5"],
                     ["--max-tokens", "8192"],
                     ["--max-tokens", "65537", "--first-pass-max-tokens", "65537"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises((ValueError, SystemExit)):
                runner.parse_args(argv)

    def test_both_core_manifests_verify(self):
        self.assertEqual(len(verify_v1(ROOT)), 64)
        self.assertEqual(len(verify_core(ROOT)), 64)


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    def args(self, **changes):
        args = run_v1_1.parse_args([])
        args.max_context_tokens = 65536
        args.question_timeout = 2
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    async def execute(self, problems, handler, **changes):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp)
            artifacts = AttemptArtifacts(output, buffered=True)
            profiler = AttemptProfiler(output, artifacts=artifacts, enabled=False)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await asyncio.wait_for(runner.run_speedrun(
                    problems, self.args(**changes), client, output,
                    DisabledGPUSampler(), time.perf_counter(), profiler), 3)

    async def test_served_tokenization_matches_chat_messages_and_rejects_full_prompt(self):
        args = self.args(disable_thinking=True)
        seen = []

        async def handler(request):
            self.assertEqual(request.url.path, "/tokenize")
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"count": 944})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            problems = await runner.prepare_prompts([{"problem_idx": 1, "problem": "question"}], args, client)
        self.assertEqual(seen[0]["messages"], [{"role": "system", "content": args.system_prompt},
                                               {"role": "user", "content": "question"}])
        self.assertTrue(seen[0]["add_generation_prompt"])
        self.assertEqual(seen[0]["chat_template_kwargs"], {"enable_thinking": False})
        with tempfile.TemporaryDirectory() as temp:
            options, round_no, used, prefixes = runner.group_options(problems[0], args, Path(temp))
        self.assertEqual((options.max_tokens, round_no, used, prefixes), (64592, 1, 0, [None]))
        for invalid in (0, -1, True, 65536, "944"):
            with self.subTest(count=invalid):
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, json={"count": invalid}))) as client:
                    with self.assertRaises(ValueError):
                        await runner.prepare_prompts(problems, args, client)

    async def test_wrong_live_candidate_keeps_same_generation_until_correct(self):
        requests, checks = [], []

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/verify":
                checks.append(body["candidate"])
                await asyncio.sleep(0.001)
                return httpx.Response(200, json={"verdict": body["candidate"] == "2"})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            requests.append(body)
            return httpx.Response(200, stream=Stream([
                sse("\\boxed{1}", None), sse("\\boxed{2}", None)], hang=True))

        rows = await self.execute([{"problem_idx": 1, "problem": "1", "prompt_tokens": 944}],
                                  handler, target_correct=1)
        self.assertEqual(checks, ["1", "2"])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_tokens"], 64592)
        self.assertEqual(rows[0]["status"], "solved")

    async def test_eager_fresh_retry_waits_for_all_verdicts_but_not_other_question(self):
        requests, checks = [], []
        active = {1: 0, 2: 0}
        peak = 0
        last_verdict_done = False

        async def handler(request):
            nonlocal peak, last_verdict_done
            body = json.loads(request.content)
            if request.url.path == "/verify":
                checks.append(body["candidate"])
                await asyncio.sleep(0.002)
                if body["candidate"] == "3":
                    last_verdict_done = True
                return httpx.Response(200, json={"verdict": body["candidate"] == "2"})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            index = int(body["messages"][1]["content"])
            number = 1 + sum(q == index for q, _ in requests)
            if index == 1 and number == 2:
                self.assertTrue(last_verdict_done)
                self.assertEqual(active[2], 1)  # Q2 has not finished its first request.
            self.assertEqual(active[index], 0)  # No sibling admission for this question.
            active[index] += 1
            peak = max(peak, sum(active.values()))
            requests.append((index, body))

            def closed():
                active[index] -= 1

            text = "\\boxed{1} \\boxed{3}" if number == 1 else "\\boxed{2}"
            return httpx.Response(200, stream=Stream(
                [sse(text)] if index == 1 else [], hang=index == 2, closed=closed))

        rows = await self.execute([{"problem_idx": q, "problem": str(q), "prompt_tokens": 204}
                                   for q in (1, 2)], handler, target_correct=1)
        self.assertEqual([q for q, _ in requests], [1, 2, 1])
        self.assertEqual(checks, ["1", "3", "2"])
        self.assertEqual(peak, 2)
        self.assertEqual(active, {1: 0, 2: 0})
        q1 = next(row for row in rows if row["problem_idx"] == 1)
        self.assertEqual(len(q1["rollouts"]), 2)
        self.assertTrue(all(r["continuation_of_rollout"] is None for r in q1["rollouts"]))
        self.assertEqual([b["seed"] for q, b in requests if q == 1], [20261008, 20261009])

    async def test_length_end_and_no_answer_exhaust_four_fresh_requests(self):
        for finish in ("length", "stop"):
            with self.subTest(finish=finish):
                requests = []

                async def handler(request):
                    self.assertEqual(request.url.path, "/v1/chat/completions")
                    requests.append(json.loads(request.content))
                    return httpx.Response(200, stream=Stream([sse("reasoning with no answer", finish)]))

                rows = await self.execute([{"problem_idx": 1, "problem": "1", "prompt_tokens": 944}],
                                          handler, target_correct=1)
                self.assertEqual(len(requests), 4)
                self.assertEqual([r["max_tokens"] for r in requests], [64592] * 4)
                self.assertEqual(rows[0]["status"], "unsolved")
                self.assertEqual(len(rows[0]["rollouts"]), 4)
                self.assertTrue(all(r["continuation_of_rollout"] is None for r in rows[0]["rollouts"]))

    async def test_initial30_target18_cancels_remaining_without_fanout(self):
        requests, streams = [], []
        all_started = asyncio.Event()
        oracle = asyncio.Lock()

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/verify":
                await all_started.wait()
                async with oracle:
                    await asyncio.sleep(0.001)
                    return httpx.Response(200, json={"verdict": True})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            requests.append(body)
            stream = Stream([sse("\\boxed{1}", None)], hang=True)
            streams.append(stream)
            if len(requests) == 30:
                all_started.set()
            return httpx.Response(200, stream=stream)

        rows = await self.execute([{"problem_idx": q, "problem": str(q), "prompt_tokens": 204}
                                   for q in range(1, 31)], handler)
        self.assertEqual(len(requests), 30)
        self.assertEqual(sum(r["status"] == "solved" for r in rows), 18)
        self.assertTrue(all(stream.closed for stream in streams))
        self.assertTrue(all(len(row["rollouts"]) == 1 for row in rows))
