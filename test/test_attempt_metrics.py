"""Offline profiling and candidate queue invariants, including cancelled checks."""

import asyncio
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx

from src.attempt import parse_args, run_question
from src.attempt_metrics import (
    AttemptProfiler,
    Meter,
    grader_timeline,
    merge_meters,
    parse_engine_metrics,
)
from test.test_attempt import FakeGPU, Stream, chunk


class MetricsTests(unittest.TestCase):
    def test_scope_cpu_async_wall_and_round_aggregation(self):
        parent = Meter()
        child = Meter(parent=parent)
        with child.measure("sync"):
            sum(range(1000))
        child.observe("network_wait", 0.2)
        child.inc("duplicates", 3)
        child.high_water("queue", 2)
        snap = parent.snapshot()
        self.assertEqual(snap["timings"]["sync"]["cpu_samples"], 1)
        self.assertEqual(snap["timings"]["network_wait"]["cpu_samples"], 0)
        merged = merge_meters(snap, snap)
        self.assertEqual(merged["counters"]["duplicates"], 6)
        self.assertEqual(merged["maxima"]["queue"], 2)
        self.assertEqual(merged["timings"]["sync"]["count"], 2)
        disabled = Meter(False)
        with disabled.measure("sync"):
            disabled.inc("anything")
        self.assertEqual(disabled.snapshot()["timings"], {})

    def test_engine_metrics_filter_preserves_labels(self):
        rows = parse_engine_metrics("""# HELP vllm:num_requests_running gauge
vllm:num_requests_running{model_name="WeiboAI/VibeThinker-3B", engine="0"} 120
vllm:kv_cache_usage_perc{engine="0"} 0.55
vllm:num_preemptions_total{engine="0"} 2
vllm:irrelevant_total 3
vllm:prompt_tokens_total bogus
""")
        self.assertEqual([r["value"] for r in rows], [120.0, 0.55, 2.0])
        self.assertIn('engine="0"', rows[0]["metric"])

    def test_grader_service_floor_and_idle_decomposition_with_partial_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            rows = [
                dict(
                    picked_at="2026-10-03T00:00:10Z",
                    answered_at="2026-10-03T00:00:13Z",
                    verdict=False,
                ),
                dict(
                    picked_at="2026-10-03T00:00:15Z",
                    answered_at="2026-10-03T00:00:18Z",
                    verdict=True,
                ),
            ]
            path.write_text("\n".join(map(json.dumps, rows)) + '\n{"partial":')
            result = grader_timeline(path, "2026-10-03T00:00:00Z", 18, 3)
            self.assertEqual(
                (
                    result["actual_service_s"],
                    result["idle_between_queries_s"],
                    result["first_pick_elapsed_s"],
                ),
                (6, 2, 10),
            )
            self.assertEqual(
                (result["correct"], result["wrong"], result["target_service_floor_s"]),
                (1, 1, 54),
            )
            self.assertEqual(len(result["audit_errors"]), 1)


class AsyncMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_profiler_samples_and_stops_tasks_without_affecting_inference(self):
        async def handler(request):
            self.assertEqual(request.url.path, "/metrics")
            return httpx.Response(
                200,
                text="vllm:num_requests_running 120\nvllm:kv_cache_usage_perc 0.4\n",
            )

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                profiler = AttemptProfiler(
                    Path(tmp), interval=0.002, engine_interval=0.003
                )
                profiler.official_start(client, "http://test")
                await asyncio.sleep(0.01)
                profiler.submitted(2.0)
                profiler.submitted(3.0)
                await profiler.stop()
                sample = profiler.snapshot()
                self.assertGreater(sample["engine"]["samples"], 0)
                self.assertEqual(
                    sample["engine"]["observed_max"]["vllm:num_requests_running"], 120
                )
                self.assertGreater(sample["event_loop_lag"]["official"]["count"], 0)
                self.assertEqual(sample["first_verification_submit_elapsed_s"], 2.0)
                self.assertEqual(profiler.tasks, [])
                self.assertTrue((Path(tmp) / "inference_metrics.jsonl").exists())

    async def test_failed_metrics_poll_is_observational(self):
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r: httpx.Response(404))
            ) as client:
                profiler = AttemptProfiler(
                    Path(tmp), interval=0.001, engine_interval=0.002
                )
                profiler.official_start(client, "http://test")
                await asyncio.sleep(0.007)
                await profiler.stop()
                self.assertGreater(len(profiler.snapshot()["engine"]["errors"]), 0)

    async def test_duplicate_normalization_and_one_check_inflight_per_question(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled), tempfile.TemporaryDirectory() as tmp:
                active = peak = 0
                checked = []
                progressed = asyncio.Event()
                args = parse_args(["--model", "WeiboAI/VibeThinker-3B"])
                args.no_overhead_profile = disabled

                async def handler(request):
                    nonlocal active, peak
                    if request.url.path == "/verify":
                        candidate = json.loads(request.content)["candidate"]
                        active += 1
                        peak = max(peak, active)
                        checked.append(candidate)
                        await progressed.wait()
                        await asyncio.sleep(0.012)
                        active -= 1
                        return httpx.Response(
                            200,
                            json={
                                "verdict": candidate == "70",
                                "queue_wait_s": 0.002,
                                "toll_s": 0.01,
                            },
                        )

                    class Progress(Stream):
                        async def __aiter__(self):
                            async for value in super().__aiter__():
                                if b"checking" in value:
                                    progressed.set()
                                yield value

                    return httpx.Response(
                        200,
                        stream=Progress(
                            [
                                chunk("\\boxed{069} \\boxed{69}"),
                                chunk("\\boxed{070}"),
                                chunk("still checking"),
                            ],
                            hang=True,
                        ),
                    )

                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    row = await run_question(
                        {"problem_idx": 1, "problem": "test"},
                        args,
                        client,
                        Path(tmp),
                        FakeGPU(),
                    )
                self.assertEqual(
                    (checked, peak, row["status"]), (["69", "70"], 1, "solved")
                )
                events = [
                    json.loads(l)
                    for l in (Path(tmp) / "trace/01/verification.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertGreater(events[1]["candidate_queue_wait_s"], 0.01)
                meter = row["overhead"]
                if disabled:
                    self.assertEqual(meter["counters"], {})
                else:
                    self.assertEqual(meter["counters"]["candidate_unique_enqueued"], 2)
                    self.assertGreaterEqual(
                        meter["counters"]["candidate_duplicates_suppressed"], 6
                    )
                    self.assertEqual(meter["counters"]["verification_wrong"], 1)
                    self.assertEqual(meter["counters"]["verification_correct"], 1)
                    self.assertEqual(
                        meter["timings"]["verification_http_wait"]["cpu_samples"], 0
                    )
                    self.assertGreater(
                        meter["timings"]["candidate_parse_enqueue"]["cpu_samples"], 0
                    )
                    self.assertEqual(meter["timings"]["grader_queue_wait"]["count"], 2)
