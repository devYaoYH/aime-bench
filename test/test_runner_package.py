"""Canonical packaging and differential policy replay; no model access."""

import ast
import asyncio
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import yaml

from runner import cli, scheduler, question
from runner.integrity import verify_core
from runner.lib.common import ROOT, atomic_json
from runner.lib.continuations import continuation_prefix
from runner.lib.extraction import CandidateDetector
from runner.lib.grader_questions import question_digest
from runner.lib.metrics import AttemptProfiler
from runner.lib.setup import local_dataset, prepare_grader
from runner.lib.storage import AttemptArtifacts, DisabledGPUSampler
from runner.lib.metadata import validate_metadata
from runner_final.core_v1 import runner as archive, _streaming as archive_stream
from runner_final.run_frozen import parse_args as archived_config
from test.test_attempt import Stream, chunk
from test.test_runner_final import capped


class PackageBoundaryTests(unittest.TestCase):
    def test_default_matches_measured_config_and_explicit_overrides(self):
        old = archived_config(
            ["--preset", str(ROOT / "runner_final/presets/prompt_adherence.json")]
        )
        new = cli.parse_args([])
        changed = {
            "grader_python",
            "preset_file",
            "preset_sha256",
            "system_prompt_file",
            "profile",
            "dataset_manifest",
            "grader_config",
            "reuse_grader",
        }
        self.assertEqual(
            {k: v for k, v in vars(new).items() if k not in changed},
            {k: v for k, v in vars(old).items() if k not in changed},
        )
        self.assertEqual(cli.parse_args(["--seed", "17"]).seed, 17)
        profiled = cli.parse_args(["--profile"])
        self.assertFalse(profiled.benchmark)
        self.assertFalse(profiled.no_gpu_telemetry)
        self.assertFalse(profiled.no_overhead_profile)
        self.assertTrue(profiled.buffer_traces)

    def test_custom_dataset_selection_retains_policy(self):
        for selection in (
            ["--grader-config", "example.yaml"],
            ["--dataset-manifest", "example.json"],
            ["--reuse-grader"],
        ):
            args = cli.parse_args(selection)
            self.assertIsNone(args.benchmark_year)
            self.assertEqual(
                (
                    args.parallelism,
                    args.rollouts,
                    args.schedule,
                    args.max_attempts_per_question,
                ),
                (30, 1, "barrier", 4),
            )

    def test_extraction_and_continuation_primitive_ast_unchanged(self):
        for old, new in (
            (archive_stream.CandidateDetector, CandidateDetector),
            (archive_stream.continuation_prefix, continuation_prefix),
        ):
            import inspect

            self.assertEqual(
                ast.dump(ast.parse(inspect.getsource(old))),
                ast.dump(ast.parse(inspect.getsource(new))),
            )

    def test_historical_manifests_retain_their_identities(self):
        from runner_final import (
            integrity,
            integrity_v1_1,
            integrity_v1_5,
            integrity_v2,
            integrity_v2_1,
            integrity_v2_2,
            integrity_v2_3,
        )

        expected = {
            integrity: "35d6a06315a0e45441b3ac49bd468a2b94553fb172f378bfebc7ea8cc044070f",
            integrity_v2: "8e6cf003ae04b063ce5479c43cc72bc6128cd5e828df054334373017e0a7187e",
            integrity_v2_1: "7ca23622428105d4d1de78fdd4df0250e07aab3a28c0a842657919903c117948",
            integrity_v2_2: "178b45f8327567bdd969ea3e0db32195de1b337d13f3871ec147a298bf93c4a7",
            integrity_v2_3: "d233ef469ce9a54b1b0f293fba54e34c8d1832cae21f26e0afd20ab4b3dcb73d",
        }
        for module, digest in expected.items():
            self.assertEqual(module.verify_core(ROOT), digest)
        for module in (integrity_v1_1, integrity_v1_5):
            self.assertEqual(len(module.verify_core(ROOT)), 64)
        self.assertTrue((ROOT / "runner_final").is_symlink())
        self.assertEqual(
            (ROOT / "runner_final").resolve(), ROOT / "runner/extensions/variants"
        )

    def test_manifest_rejects_source_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runner").mkdir()
            atomic_json(
                root / "runner/manifest.json",
                {
                    "schema_version": 1,
                    "core_id": "runner_core_v1",
                    "sha256": {"runner/__init__.py": "0" * 64},
                },
            )
            (root / "runner/__init__.py").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "source drift"):
                verify_core(root)

    def test_registry_routes_explicit_version_and_promoted_default(self):
        with patch.object(cli, "launch_extension") as launch:
            cli.main(["--version", "v2.3", "--help"])
            launch.assert_called_once_with("v2.3", ["--help"])
        with patch.object(cli, "CANONICAL", "v2.1"), patch.object(
            cli, "launch_extension"
        ) as launch:
            cli.main(["--help"])
            launch.assert_called_once_with("v2.1", ["--help"])

    def test_canonical_runs_with_only_runner_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(
                ROOT / "runner",
                Path(tmp) / "runner",
                ignore=shutil.ignore_patterns("extensions", "__pycache__", ".venv"),
            )
            # No src, root grader/data, runner_final alias, or Git metadata here.
            result = subprocess.run(
                [sys.executable, "-m", "runner", "--help"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--grader-config", result.stdout)
            self.assertIn("--profile", result.stdout)


async def replay(module, args, scripts, *, verdict="70", parallelism=None):
    """Replay fixed SSE outputs and compare actual requests/verdicts and saved state."""
    requests, checks, streams = [], [], []

    async def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/verify":
            checks.append((body["index"], body["candidate"]))
            await asyncio.sleep(0.01)
            return httpx.Response(200, json={"verdict": body["candidate"] == verdict})
        requests.append((request.url.path, body))
        stream = Stream(scripts[len(requests) - 1], delay=0.001)
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = AttemptArtifacts(root, buffered=True)
        profiler = AttemptProfiler(root, enabled=False, artifacts=store)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            problems = [{"problem_idx": 1, "problem": "Test statement."}]
            rows = await module.run_speedrun(
                problems,
                args,
                client,
                root,
                DisabledGPUSampler(),
                time.perf_counter(),
                profiler,
            )
        selfcontained = {
            p.name: v for p, v in store.json_files.items() if p.name == "tokens.json"
        }
        return requests, checks, rows, selfcontained, all(s.closed for s in streams)


class CanonicalReplayTests(unittest.IsolatedAsyncioTestCase):
    async def compare(self, scripts, options=(), verdict="70"):
        args = cli.parse_args(["--target-correct", "1", *options])
        # Both runners receive exactly the same measured prompt and settings.
        old = await replay(archive, args, scripts, verdict=verdict)
        new = await replay(scheduler, args, scripts, verdict=verdict)
        self.assertEqual(old[:2], new[:2])
        fields = (
            "problem_idx",
            "status",
            "candidate_answers",
            "round",
            "unique_candidates",
        )
        for a, b in zip(old[2], new[2]):
            self.assertEqual({k: a[k] for k in fields}, {k: b[k] for k in fields})
            rf = (
                "rollout",
                "round",
                "status",
                "finish_reason",
                "continuation_of_rollout",
                "requested_max_tokens",
            )
            self.assertEqual(
                [{k: r[k] for k in rf} for r in a["rollouts"]],
                [{k: r[k] for k in rf} for r in b["rollouts"]],
            )
            for row in (a, b):
                if row["status"] == "solved":
                    self.assertGreaterEqual(
                        row["first_solved"]["first_solved_elapsed_s"], 0
                    )
                    self.assertLessEqual(
                        row["first_solved"]["first_solved_elapsed_s"],
                        row["end_to_end_latency_s"] + 0.05,
                    )
        self.assertEqual(old[3:], new[3:])
        return new

    async def test_split_integer_continuation_matches_frozen(self):
        value = await self.compare(
            [
                capped([10], [11], "\\boxed{0"),
                capped([10, 11], [12], "70}", completion=True),
            ]
        )
        self.assertEqual(value[0][1][1]["prompt"], [10, 11])
        self.assertEqual([r[1]["max_tokens"] for r in value[0]], [8192, 16384])

    async def test_wrong_duplicate_then_self_correction_matches_frozen(self):
        value = await self.compare(
            [
                [
                    chunk("\\boxed{7}\n"),
                    chunk("\\boxed{007}\n"),
                    chunk("Answer: 70\n"),
                    chunk("", finish="stop"),
                    b"data: [DONE]\n\n",
                ]
            ]
        )
        self.assertEqual(value[1], [(1, "7"), (1, "70")])

    async def test_request_limit_matches_frozen(self):
        value = await self.compare(
            [capped([1], [2], "reasoning")]
            + [
                capped([1, 2], [3], "more reasoning", completion=True),
                capped([1, 2, 3], [4], "more reasoning", completion=True),
                capped([1, 2, 3, 4], [5], "more reasoning", completion=True),
            ]
        )
        self.assertEqual(len(value[0]), 4)
        self.assertEqual(value[2][0]["status"], "unsolved")

    async def test_two_lanes_continue_distinct_exact_prefixes(self):
        value = await self.compare(
            [
                capped([1], [10], "first"),
                capped([1], [20], "second"),
                capped([1, 10], [11], "\\boxed{70}", completion=True),
                capped([1, 20], [21], "other", completion=True),
            ],
            ["--rollouts", "2"],
        )
        self.assertEqual([r[1].get("prompt") for r in value[0][2:]], [[1, 10], [1, 20]])

    async def test_invalid_grader_response_is_error_and_cancels_stream(self):
        async def handler(request):
            if request.url.path == "/verify":
                return httpx.Response(200, json={"verdict": "true"})
            return httpx.Response(200, stream=Stream([chunk("\\boxed{70}")], hang=True))

        with tempfile.TemporaryDirectory() as tmp:
            store = AttemptArtifacts(Path(tmp), buffered=True)
            profiler = AttemptProfiler(Path(tmp), enabled=False, artifacts=store)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                row = await question.run_question(
                    {"problem_idx": 1, "problem": "Test"},
                    cli.parse_args([]),
                    client,
                    Path(tmp),
                    DisabledGPUSampler(),
                    profiler=profiler,
                )
            self.assertEqual(row["status"], "error")
            self.assertIn("boolean verdict", row["error"])
            self.assertEqual(row["rollouts"][0]["status"], "cancelled")

    async def test_target_cancels_sibling_and_other_question_streams(self):
        streams = []

        async def handler(request):
            if request.url.path == "/verify":
                return httpx.Response(200, json={"verdict": True})
            body = json.loads(request.content)
            text = (
                "\\boxed{70}"
                if body["messages"][1]["content"] == "winner"
                else "reasoning"
            )
            stream = Stream([chunk(text)], hang=True)
            streams.append(stream)
            return httpx.Response(200, stream=stream)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AttemptArtifacts(root, buffered=True)
            profiler = AttemptProfiler(root, enabled=False, artifacts=store)
            args = cli.parse_args(["--rollouts", "2", "--target-correct", "1"])
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                rows = await scheduler.run_speedrun(
                    [
                        {"problem_idx": 1, "problem": "winner"},
                        {"problem_idx": 2, "problem": "other"},
                    ],
                    args,
                    client,
                    root,
                    DisabledGPUSampler(),
                    time.perf_counter(),
                    profiler,
                )
            self.assertEqual([r["status"] for r in rows], ["solved", "stopped"])
            self.assertEqual(len(streams), 4)
            self.assertTrue(all(s.closed for s in streams))
            self.assertTrue(
                all(r["status"] == "cancelled" for q in rows for r in q["rollouts"])
            )

    async def test_round_barrier_waits_for_all_questions(self):
        entered, release, events = asyncio.Event(), asyncio.Event(), []

        async def fake(problem, args, client, output, sampler, **kw):
            idx = problem["problem_idx"]
            rnd = kw["round_no"]
            events.append((idx, rnd))
            if idx == 2 and rnd == 1:
                entered.set()
                await release.wait()
            row = {
                "problem_idx": idx,
                "status": "unsolved",
                "round": rnd,
                "rollouts": [
                    {
                        "rollout": rnd,
                        "round": rnd,
                        "status": "completed",
                        "finish_reason": "stop",
                    }
                ],
            }
            kw["profiler"].artifacts.write_json(
                output / "trace" / f"{idx:02d}" / "question.json", row
            )
            return row

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AttemptArtifacts(root, buffered=True)
            p = AttemptProfiler(root, enabled=False, artifacts=store)
            args = cli.parse_args(
                ["--target-correct", "2", "--max-rounds", "2", "--no-continuation"]
            )
            with patch.object(scheduler, "run_question", fake):
                task = asyncio.create_task(
                    scheduler.run_speedrun(
                        [{"problem_idx": 1}, {"problem_idx": 2}],
                        args,
                        None,
                        root,
                        None,
                        0,
                        p,
                    )
                )
                await entered.wait()
                await asyncio.sleep(0.005)
                self.assertEqual(events, [(1, 1), (2, 1)])
                release.set()
                await task
            self.assertEqual(events, [(1, 1), (2, 1), (1, 2), (2, 2)])


class DatasetAndLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_bundled_custom_grader_runs_without_legacy_tree(self):
        from runner.lib.services import Services, ready

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(
                ROOT / "runner",
                root / "runner",
                ignore=shutil.ignore_patterns("extensions", "__pycache__", ".venv"),
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            config = yaml.safe_load(
                (root / "runner/examples/integer_grader.yaml").read_text()
            )
            config.update(port=port, cost_c=0, audit_log=str(root / "audit.jsonl"))
            path = root / "grader.yaml"
            path.write_text(yaml.safe_dump(config))
            services = Services()
            try:
                process = services.launch(
                    [sys.executable, str(root / "runner/grader/server_questions.py")],
                    root / "grader.log",
                    {**os.environ, "GRADER_CONFIG": str(path)},
                )
                async with httpx.AsyncClient(trust_env=False) as client:
                    health = await ready(
                        client, f"http://127.0.0.1:{port}/health", 20, process
                    )
                    response = await client.get(f"http://127.0.0.1:{port}/questions")
                    payload = response.json()
                    self.assertEqual(health["n_problems"], 2)
                    self.assertEqual(
                        [q["problem_idx"] for q in payload["questions"]], [4, 90]
                    )
                    self.assertTrue(
                        all(
                            set(q) == {"problem_idx", "problem"}
                            for q in payload["questions"]
                        )
                    )
                    verdict = await client.post(
                        f"http://127.0.0.1:{port}/verify",
                        json={
                            "index": 4,
                            "candidate": "156",
                            "query_id": "package-test",
                            "agent_id": "test",
                        },
                    )
                    self.assertTrue(verdict.json()["verdict"])
            finally:
                await services.close()

    async def test_custom_api_is_gold_free_and_reused_grader_is_not_owned(self):
        questions = [
            {"problem_idx": 4, "problem": "An integer problem."},
            {"problem_idx": 90, "problem": "Another problem."},
        ]
        digest = question_digest(questions)
        health = {
            "queries_so_far": 0,
            "cost_c": 3.0,
            "n_problems": 2,
            "dataset": {"sha256": "key-sha", "questions_sha256": digest},
        }
        dataset = {"id": "custom", "year": None, "grader_sha256": "key-sha", "rows": 2}

        async def handler(request):
            return httpx.Response(
                200,
                json=(
                    health
                    if request.url.path == "/health"
                    else {
                        "questions": questions,
                        "dataset": dataset,
                        "questions_sha256": digest,
                    }
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            args = cli.parse_args(
                ["--reuse-grader", "--questions", "90", "--target-correct", "1"]
            )
            root = Path(tmp)
            config = {}
            services = SimpleNamespace(launch=unittest.mock.Mock())
            profiler = AttemptProfiler(root, enabled=False)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                selected = await prepare_grader(
                    args, client, services, profiler, root, config, None
                )
            services.launch.assert_not_called()
            self.assertEqual(selected, [questions[1]])
            self.assertEqual(
                json.loads((root / "questions.json").read_text()), questions
            )
            self.assertIsNone(local_dataset(args, {}))

    async def test_full_lifecycle_benchmark_flush_and_metadata(self):
        import runner.run as module

        args = cli.parse_args(["--questions", "1", "--target-correct", "1"])

        async def handler(request):
            if request.url.path == "/verify":
                return httpx.Response(200, json={"verdict": True})
            return httpx.Response(200, stream=Stream(capped([1], [2], "\\boxed{70}")))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        profile = {"max-model-len": 65536, "gpu-memory-utilization": 0.95}

        async def inference(*values):
            args.max_context_tokens = 65536

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            services = SimpleNamespace(close=AsyncMock())
            with patch.object(module, "ROOT", root), patch.object(
                module, "verify_core", return_value="manifest"
            ), patch.object(module, "Services", return_value=services), patch.object(
                module, "ensure_free"
            ), patch.object(
                module,
                "git_state",
                return_value={"git_commit": None, "git_dirty": None},
            ), patch.object(
                module,
                "local_dataset",
                return_value=[{"problem_idx": 1, "problem": "Test"}],
            ), patch.object(
                module,
                "read_profile",
                return_value=(root / "profile.yaml", yaml.safe_dump(profile), profile),
            ), patch.object(
                module, "prepare_inference", side_effect=inference
            ), patch.object(
                module,
                "prepare_grader",
                new=AsyncMock(return_value=[{"problem_idx": 1, "problem": "Test"}]),
            ), patch.object(
                module,
                "warm_inference",
                new=AsyncMock(
                    return_value={
                        "latency_s": 0.0,
                        "batch_size": 1,
                        "tokens_per_request": 32,
                    }
                ),
            ), patch.object(
                module.httpx, "AsyncClient", return_value=client
            ):
                # Model/key identity is normally supplied by startup, never by the policy.
                real_local = module.local_dataset

                def local(a, c):
                    from runner.lib.aime import dataset_provenance

                    c["dataset_provenance"] = dataset_provenance(2025)
                    return real_local(a, c)

                with patch.object(module, "local_dataset", side_effect=local):
                    output = await module.run(args)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["target_reached"])
            self.assertTrue(summary["trace_storage"]["buffered"])
            self.assertTrue((output / "trace/01/rollout-01/tokens.json").exists())
            self.assertIsNone(summary["performance"]["official_gpu"])
            services.close.assert_awaited_once()
            validate_metadata(
                json.loads((output / "metadata.json").read_text()), output.name
            )


if __name__ == "__main__":
    unittest.main()
