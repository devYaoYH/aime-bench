"""Policy and deferred sweep checks; no GPU, SSH, or model access."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import socket
import unittest
from unittest.mock import patch

import httpx

from src.attempt_runners.speedrun_v1 import assert_gpu_idle, parse_args, run_speedrun
from src.attempt_runners import sweep_speedrun_v1 as sweep
from src.common import ROOT
from src.attempt_runners._ports_v1 import ensure_free
from test.test_attempt import FakeGPU, Stream, chunk


def capped(prompt, tokens, text, *, completion=False):
    choice = {"index": 0, "token_ids": tokens, "finish_reason": "length"}
    choice.update(
        {"text": text, "prompt_token_ids": prompt}
        if completion
        else {"delta": {"content": text}}
    )
    row = {
        "choices": [choice],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(tokens)},
    }
    if not completion:
        row["prompt_token_ids"] = prompt
    return [("data: " + json.dumps(row) + "\n\n").encode(), b"data: [DONE]\n\n"]


class SpeedrunTests(unittest.IsolatedAsyncioTestCase):
    def args(self, **changes):
        args = parse_args(
            ["--model", "WeiboAI/VibeThinker-3B", "--question-timeout", "2"]
        )
        for k, v in changes.items():
            setattr(args, k, v)
        return args

    async def test_all_120_initial_requests_target18_cancellation_and_budget4(self):
        streams, requests, checks = [], [], []
        all_started = asyncio.Event()
        oracle = asyncio.Lock()

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/verify":
                await all_started.wait()
                async with oracle:
                    checks.append(body)
                    await asyncio.sleep(0.001)
                    return httpx.Response(200, json={"verdict": True})
            requests.append(body)
            q = int(body["messages"][1]["content"])
            stream = Stream(
                [chunk("\\boxed{070}" if q <= 18 else "thinking")],
                delay=0.001,
                hang=True,
            )
            streams.append(stream)
            if len(streams) == 120:
                all_started.set()
            return httpx.Response(200, stream=stream)

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                rows = await asyncio.wait_for(
                    run_speedrun(
                        [{"problem_idx": q, "problem": str(q)} for q in range(1, 31)],
                        self.args(),
                        client,
                        Path(tmp),
                        FakeGPU(),
                        time.perf_counter(),
                    ),
                    3,
                )
            self.assertEqual(len(requests), 120)
            self.assertEqual(
                len(checks), 18
            )  # four identical candidates => one verification/question
            self.assertEqual(len(rows), 30)
            self.assertEqual(sum(r["status"] == "solved" for r in rows), 18)
            self.assertEqual(sum(r["status"] == "stopped" for r in rows), 12)
            self.assertTrue(all(len(r["rollouts"]) == 4 for r in rows))
            self.assertTrue(all(s.closed for s in streams))
            self.assertEqual({r["max_tokens"] for r in requests}, {8192})
            self.assertEqual(
                len((Path(tmp) / "solved.jsonl").read_text().splitlines()), 18
            )

    async def test_pass2_continues_distinct_lanes_and_cap4(self):
        requests = []

        async def handler(request):
            self.assertNotEqual(request.url.path, "/verify")
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path == "/v1/chat/completions":
                prompt, tokens = [body["seed"]], [50]
            else:
                prompt, tokens = body["prompt"], [51]
            return httpx.Response(
                200,
                stream=Stream(
                    capped(prompt, tokens, "thinking", completion="prompt" in body)
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                rows = await run_speedrun(
                    [{"problem_idx": 1, "problem": "test"}],
                    self.args(rollouts=2, target_correct=1),
                    client,
                    Path(tmp),
                    FakeGPU(),
                    time.perf_counter(),
                )
            self.assertEqual(len(requests), 4)
            self.assertEqual(
                [r[0] for r in requests],
                ["/v1/chat/completions"] * 2 + ["/v1/completions"] * 2,
            )
            self.assertEqual(requests[2][1]["prompt"], [requests[0][1]["seed"], 50])
            self.assertEqual(requests[3][1]["prompt"], [requests[1][1]["seed"], 50])
            self.assertNotEqual(requests[2][1]["prompt"], requests[3][1]["prompt"])
            self.assertEqual(
                [r[1]["max_tokens"] for r in requests], [8192, 8192, 16384, 16384]
            )
            self.assertEqual(
                [r["continuation_of_rollout"] for r in rows[0]["rollouts"]],
                [None, None, 1, 2],
            )
            self.assertEqual(rows[0]["status"], "unsolved")

    async def test_fifo_fresh_before_retries_and_eager_vs_barrier(self):
        async def schedule(kind):
            order = []
            release = asyncio.Event()
            q2_started = asyncio.Event()

            async def handler(request):
                body = json.loads(request.content)
                q = int(body["messages"][1]["content"])
                order.append(q)
                if q == 2 and order.count(2) == 1:
                    q2_started.set()
                    await release.wait()
                return httpx.Response(
                    200,
                    stream=Stream(
                        [chunk("none", finish="stop"), b"data: [DONE]\n\n"], delay=0.001
                    ),
                )

            with tempfile.TemporaryDirectory() as tmp:
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    task = asyncio.create_task(
                        run_speedrun(
                            [{"problem_idx": q, "problem": str(q)} for q in (1, 2, 3)],
                            self.args(
                                schedule=kind,
                                rollouts=1,
                                parallelism=2,
                                target_correct=3,
                                max_rounds=2,
                                no_continuation=True,
                            ),
                            client,
                            Path(tmp),
                            FakeGPU(),
                            time.perf_counter(),
                        )
                    )
                    await q2_started.wait()
                    for _ in range(50):
                        if order.count(1) >= 2 or (kind == "barrier" and 3 in order):
                            break
                        await asyncio.sleep(0.001)
                    self.assertEqual(order[:3], [1, 2, 3])
                    self.assertEqual(order.count(1) >= 2, kind == "eager")
                    release.set()
                    await asyncio.wait_for(task, 2)
            self.assertEqual(len(order), 6)

        await schedule("eager")
        await schedule("barrier")

    async def test_worker_failure_does_not_hang_or_retry(self):
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(500)

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(
                        run_speedrun(
                            [{"problem_idx": 1, "problem": "test"}],
                            self.args(rollouts=1),
                            client,
                            Path(tmp),
                            FakeGPU(),
                            time.perf_counter(),
                        ),
                        1,
                    )
            row = json.loads((Path(tmp) / "trace/01/question.json").read_text())
            self.assertEqual(row["status"], "error")
            self.assertEqual(len(requests), 1)

    async def test_pass1_counts_continuations_as_requests_and_stitches_candidate(self):
        requests = []

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/verify":
                self.assertEqual(body["candidate"], "70")
                return httpx.Response(200, json={"verdict": True})
            requests.append(body)
            if len(requests) == 1:
                return httpx.Response(
                    200, stream=Stream(capped([10], [11], "\\boxed{0"))
                )
            return httpx.Response(
                200, stream=Stream(capped([10, 11], [12], "70}", completion=True))
            )

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                rows = await run_speedrun(
                    [{"problem_idx": 1, "problem": "test"}],
                    self.args(rollouts=1, target_correct=1),
                    client,
                    Path(tmp),
                    FakeGPU(),
                    time.perf_counter(),
                )
            self.assertEqual(len(requests), 2)
            self.assertEqual(rows[0]["status"], "solved")
            self.assertEqual(requests[1]["prompt"], [10, 11])


class SweepTests(unittest.TestCase):
    def test_manifest_eight_cells_and_paired_seeds(self):
        plan = sweep.build_plan(
            json.loads((ROOT / "configs/sweeps/vibe-speedrun-v1.json").read_text())
        )
        self.assertEqual(len(plan["cells"]), 8)
        combinations = {
            (c["parameters"]["parallelism"], c["parameters"]["rollouts"])
            for c in plan["cells"]
        }
        self.assertNotIn((8, 1), combinations)
        self.assertNotIn((8, 2), combinations)
        self.assertIn((8, 4), combinations)
        self.assertEqual(
            (
                plan["cells"][0]["parameters"]["parallelism"],
                plan["cells"][0]["parameters"]["rollouts"],
            ),
            (30, 4),
        )
        self.assertEqual(
            (
                plan["cells"][1]["parameters"]["parallelism"],
                plan["cells"][1]["parameters"]["rollouts"],
            ),
            (30, 2),
        )
        self.assertEqual(
            sum(c["parameters"]["schedule"] == "barrier" for c in plan["cells"]), 1
        )
        self.assertEqual(
            max(c["initial_concurrent_requests"] for c in plan["cells"]), 120
        )
        self.assertEqual({c["parameters"]["seed"] for c in plan["cells"]}, {20261003})
        for cell in plan["cells"]:
            args = parse_args(cell["argv"])
            self.assertEqual(
                (args.first_pass_max_tokens, args.max_attempts_per_question), (8192, 4)
            )
            self.assertTrue(cell["command"].startswith("~/.venvs/vllm/bin/python -m"))

    def test_default_plan_is_side_effect_free(self):
        with patch.object(
            sweep, "execute", side_effect=AssertionError("Must not execute")
        ), patch.object(
            sweep, "attempt_lock", side_effect=AssertionError("Must not lock")
        ), redirect_stdout(
            io.StringIO()
        ) as out:
            sweep.main([])
        self.assertEqual(len(json.loads(out.getvalue())["cells"]), 8)

    def test_exclusions_are_validated_and_can_be_disabled(self):
        config = json.loads((ROOT / "configs/sweeps/vibe-speedrun-v1.json").read_text())
        config["exclude"] = []
        self.assertEqual(len(sweep.build_plan(config)["cells"]), 10)
        for excluded in (
            [{}],
            [{"typo": 8}],
            [{"schedule": "eager"}, {"schedule": "barrier"}],
        ):
            config["exclude"] = excluded
            with self.assertRaises(ValueError):
                sweep.build_plan(config)

    def test_invalid_budget_and_profile_escape(self):
        for argv in (
            ["--max-attempts-per-question", "5"],
            ["--model-profile", "../other.yaml"],
            ["--engine-metrics-interval", "0"],
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--model", "WeiboAI/VibeThinker-3B", *argv])

    def test_unmet_or_failed_targets_never_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cell = {"cell_id": "one", "parameters": {}}
            for status, reached in (
                ("completed", False),
                ("failed", True),
                ("completed", True),
            ):
                (folder / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": status,
                            "target_reached": reached,
                            "time_to_target_s": 92.4,
                        }
                    )
                )
                row = sweep.score(cell, folder)
                self.assertEqual(
                    row["time_to_target_s"] is not None,
                    status == "completed" and reached,
                )
            rows = sweep.ranked(
                [
                    dict(cell_id="a", time_to_target_s=None),
                    dict(cell_id="b", time_to_target_s=70),
                    dict(cell_id="c", time_to_target_s=60),
                ]
            )
            self.assertEqual([r["rank"] for r in rows], [None, 2, 1])

    def test_busy_gpu_is_rejected_without_termination(self):
        from types import SimpleNamespace

        fake = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda d: d,
            nvmlDeviceGetComputeRunningProcesses=lambda d: [SimpleNamespace(pid=123)],
        )
        with patch.dict("sys.modules", {"pynvml": fake}), self.assertRaisesRegex(
            RuntimeError, "occupied"
        ):
            assert_gpu_idle(0)


class SweepExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_persists_failed_cell_and_stops_before_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "attempts").mkdir()
            config = json.loads(
                (ROOT / "configs/sweeps/vibe-speedrun-v1.json").read_text()
            )
            plan = sweep.build_plan(config)
            calls = []

            async def fail(args):
                calls.append(args)
                path = root / "attempts" / "failed-cell"
                path.mkdir()
                (path / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "target_reached": False,
                            "error": "out of memory",
                        }
                    )
                )
                raise RuntimeError("out of memory")

            with patch.object(sweep, "ROOT", root), patch.object(
                sweep.runner, "run", fail
            ), patch.object(
                sweep.subprocess, "check_output", return_value="test-commit\n"
            ):
                await sweep.execute(plan, root / "sweep")
            result = json.loads((root / "sweep/summary.json").read_text())
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["cells"][0]["attempt_id"], "failed-cell")
            self.assertIsNone(result["cells"][0]["rank"])

    async def test_interruption_persists_partial_cell_and_does_not_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "attempts").mkdir()
            plan = sweep.build_plan(
                json.loads((ROOT / "configs/sweeps/vibe-speedrun-v1.json").read_text())
            )
            started = asyncio.Event()

            async def interrupted(args):
                path = root / "attempts" / "partial-cell"
                path.mkdir()
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    (path / "summary.json").write_text(
                        json.dumps({"status": "interrupted", "target_reached": False})
                    )

            with patch.object(sweep, "ROOT", root), patch.object(
                sweep.runner, "run", interrupted
            ), patch.object(
                sweep.subprocess, "check_output", return_value="test-commit\n"
            ):
                task = asyncio.create_task(sweep.execute(plan, root / "sweep"))
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            result = json.loads((root / "sweep/summary.json").read_text())
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(len(result["cells"]), 1)
            self.assertEqual(result["cells"][0]["attempt_id"], "partial-cell")
            self.assertIsNone(result["cells"][0]["time_to_target_s"])


class PortProbeTests(unittest.TestCase):
    def test_active_listener_is_rejected(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen()
            with self.assertRaises(OSError):
                ensure_free(server.getsockname()[1])

    def test_recently_closed_server_connection_allows_next_cell(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
            server.listen()
            client.connect(("127.0.0.1", port))
            connection, _ = server.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b"")
                client.close()
        ensure_free(port)
