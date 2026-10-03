"""Isolated v1 naive-runner checks; canonical tests remain unchanged."""

import asyncio
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx

from src.attempt_runners.naive_pass4_v1 import (
    PROMPT as BASELINE_PROMPT,
    final_answer,
    parse_args,
    prepare_baseline_prompts,
    run_baseline,
    run_question,
)
from src.attempt_runners._runtime_v1 import warm_inference


class FakeGPU:
    def window(self, start, end):
        return {"observed_peak_vram_mib": 123, "sample_count": 1}


def chunk(text, part="content", finish=None):
    return (
        "data: "
        + json.dumps(
            {"choices": [{"index": 0, "delta": {part: text}, "finish_reason": finish}]}
        )
        + "\n\n"
    ).encode()


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


class FinalAnswerTests(unittest.TestCase):
    def test_cli_is_separate_from_canonical(self):
        from src import attempt as canonical

        before = vars(canonical.parse_args(["--model", "WeiboAI/VibeThinker-3B"]))
        naive = parse_args(["--model", "WeiboAI/VibeThinker-3B"])
        self.assertEqual(
            (naive.parallelism, naive.rollouts, naive.max_tokens), (30, 4, 16384)
        )
        self.assertEqual(naive.model_profile, "vllm-baseline-16k.yaml")
        self.assertNotEqual(BASELINE_PROMPT, canonical.PROMPT)
        after = vars(canonical.parse_args(["--model", "WeiboAI/VibeThinker-3B"]))
        self.assertEqual(before, after)
        self.assertEqual(
            (before["parallelism"], before["rollouts"], before["strategy"]),
            (8, 4, "fanout"),
        )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            canonical.parse_args(
                ["--model", "WeiboAI/VibeThinker-3B", "--strategy", "baseline"]
            )
        for extra in [
            ["--strategy", "coverage"],
            ["--model-profile", "../vllm.yaml"],
            ["--rollouts", "5"],
        ]:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--model", "WeiboAI/VibeThinker-3B", *extra])

    def test_final_answer_uses_last_final_content_box(self):
        self.assertEqual(
            final_answer("<think>\\boxed{69}</think>Final: \\boxed{070}")["answer"], 70
        )
        self.assertEqual(
            final_answer("\\boxed{69}, correction: \\boxed{70}")["answer"], 70
        )
        for text in [
            "<think>\\boxed{70}",
            "\\boxed{70} then \\boxed{4/17}",
            "Answer is 70.",
            "\\boxed{1000}",
            "reasoning without an answer",
        ]:
            self.assertIsNone(final_answer(text), text)


class NaiveRunnerTests(unittest.IsolatedAsyncioTestCase):
    def args(self, **changes):
        args = parse_args(["--model", "Qwen/Qwen3.5-4B", "--question-timeout", "2"])
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    async def test_versioned_runtime_warms_all_120_streams(self):
        requests, active, peak = [], 0, 0

        async def handler(request):
            nonlocal active, peak
            self.assertEqual(request.url.path, "/v1/chat/completions")
            requests.append(json.loads(request.content))
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.005)
            active -= 1
            return httpx.Response(200, json={"choices": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await warm_inference(self.args(), client, 30)
        self.assertEqual((result["batch_size"], peak), (120, 120))
        self.assertEqual(len({r["seed"] for r in requests}), 120)
        self.assertTrue(
            all(
                (r["temperature"], r["top_p"], r["max_tokens"]) == (0.8, 0.95, 32)
                for r in requests
            )
        )

    async def test_baseline_waits_for_natural_end_and_ignores_intermediate_box(self):
        emitted, finish = asyncio.Event(), asyncio.Event()
        submitted, requests = [], []

        class FinalStream(Stream):
            async def __aiter__(self):
                yield chunk("<think>First guess \\boxed{69}. Still checking.")
                emitted.set()
                await finish.wait()
                yield chunk("</think>Final answer: \\boxed{70}", finish="stop")
                yield b"data: [DONE]\n\n"

        async def handler(request):
            if request.url.path == "/verify":
                self.assertTrue(finish.is_set())
                submitted.append(json.loads(request.content)["candidate"])
                return httpx.Response(200, json={"verdict": True})
            requests.append(json.loads(request.content))
            return httpx.Response(200, stream=FinalStream([]))

        args = self.args(
            strategy="baseline", rollouts=1, max_context_tokens=100, max_tokens=100
        )
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                task = asyncio.create_task(
                    run_question(
                        {"problem_idx": 1, "problem": "test", "prompt_tokens": 10},
                        args,
                        client,
                        Path(tmp),
                        FakeGPU(),
                    )
                )
                await emitted.wait()
                await asyncio.sleep(0.01)
                self.assertEqual(submitted, [])
                finish.set()
                result = await task
            self.assertEqual(submitted, ["70"])
            self.assertEqual(result["winner"]["kind"], "final_box")
            self.assertEqual(result["rollouts"][0]["status"], "completed")
            self.assertEqual(requests[0]["max_tokens"], 90)
            self.assertEqual(requests[0]["messages"][0]["content"], BASELINE_PROMPT)

    async def test_baseline_does_not_grade_capped_output_even_with_box(self):
        async def handler(request):
            self.assertNotEqual(request.url.path, "/verify")
            return httpx.Response(
                200,
                stream=Stream(
                    [chunk("\\boxed{70}", finish="length"), b"data: [DONE]\n\n"]
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                result = await run_question(
                    {"problem_idx": 1, "problem": "test", "prompt_tokens": 20},
                    self.args(strategy="baseline"),
                    client,
                    Path(tmp),
                    FakeGPU(),
                )
            self.assertEqual(result["status"], "unsolved")
            self.assertEqual(result["unique_candidates"], 0)
            self.assertTrue(
                all(r["final_answer_extracted"] is False for r in result["rollouts"])
            )

    async def test_baseline_launches_120_streams_and_stops_at_18_final_verdicts(self):
        active = peak = 0
        streams, requests = [], []

        class Tracked(Stream):
            async def aclose(self):
                nonlocal active
                if not self.closed:
                    active -= 1
                await super().aclose()

        async def handler(request):
            nonlocal active, peak
            if request.url.path == "/verify":
                await asyncio.sleep(0.002)
                return httpx.Response(200, json={"verdict": True})
            body = json.loads(request.content)
            requests.append(body)
            q = int(body["messages"][1]["content"])
            active += 1
            peak = max(peak, active)
            stream = Tracked(
                (
                    [chunk("\\boxed{70}", finish="stop"), b"data: [DONE]\n\n"]
                    if q <= 18
                    else [chunk("intermediate \\boxed{69}")]
                ),
                delay=0.01,
                hang=q > 18,
            )
            streams.append(stream)
            return httpx.Response(200, stream=stream)

        args = self.args(strategy="baseline", parallelism=30, target_correct=18)
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                result = await run_baseline(
                    [
                        {"problem_idx": i, "problem": str(i), "prompt_tokens": 20}
                        for i in range(1, 31)
                    ],
                    args,
                    client,
                    Path(tmp),
                    FakeGPU(),
                    time.perf_counter(),
                )
            self.assertEqual((len(requests), peak), (120, 120))
            self.assertEqual(sum(q["status"] == "solved" for q in result), 18)
            self.assertEqual(sum(q["status"] == "stopped" for q in result), 12)
            self.assertTrue(all(len(q["rollouts"]) == 4 for q in result))
            self.assertTrue(all(s.closed for s in streams))
            self.assertTrue(all(q["error"] is None for q in result))
            self.assertTrue(
                all(q["winner"]["kind"] == "final_box" for q in result if q["winner"])
            )

    async def test_baseline_tokenization_matches_prompt_and_checks_context(self):
        args = self.args(
            strategy="baseline", disable_thinking=True, max_context_tokens=100
        )

        async def handler(request):
            self.assertEqual(request.url.path, "/tokenize")
            body = json.loads(request.content)
            self.assertEqual(body["messages"][0]["content"], BASELINE_PROMPT)
            self.assertTrue(body["add_generation_prompt"])
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            return httpx.Response(200, json={"count": 20})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await prepare_baseline_prompts(
                [{"problem_idx": 1, "problem": "test"}], args, client
            )
            self.assertEqual(result[0]["prompt_tokens"], 20)
            args.max_context_tokens = 20
            with self.assertRaises(ValueError):
                await prepare_baseline_prompts(
                    [{"problem_idx": 1, "problem": "test"}], args, client
                )
