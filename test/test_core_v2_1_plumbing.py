"""Core v2 transfer safeguards; no GPU, SSH, model or network is used."""
import ast
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from runner_final.core_v1 import runner as v1
from runner_final.core_v2_1 import benchmarks, runner, _streaming
from runner_final.core_v2_1.extraction import CandidateDetector
from runner_final.core_v2_1.metadata import build_metadata, validate_metadata
from runner_final.core_v2_1.grader_questions import fetch_questions, question_digest
from runner_final.integrity import verify_core as verify_v1
from runner_final.integrity_v2_1 import MANIFEST, verify_core
from runner_final.run_frozen_v2_1 import parse_args
from src.attempt_storage import AttemptArtifacts, DisabledGPUSampler
from src.attempt_metrics import AttemptProfiler
from src.common import ROOT
from test.test_attempt import FakeGPU, Stream, chunk
from test.test_runner_final import capped

APEX = "data/source_apex_shortlist.json"


class ExtractionTests(unittest.TestCase):
    def test_exact_expressions_at_every_stream_split(self):
        answers = ["2618", "2077992225", "-17", "0.125", "n-1", "n(n-1)/2",
                   r"\frac{4}{9}", r"\frac{9\sqrt{30}}{4}",
                   r"\frac{1}{46}\binom{2024}{990}+\frac{1}{2}",
                   "10^{225}-1", "4N^3+9N^2+6N+1", r"\{1,2\}"]
        for answer in answers:
            text = "Thinking. " + r"\boxed{" + answer + "}"
            for split in range(len(text)):
                with self.subTest(answer=answer, split=split):
                    detector = CandidateDetector()
                    self.assertEqual(detector.feed("reasoning", text[:split]), [])
                    events = detector.feed("reasoning", text[split:])
                    self.assertEqual([e["answer"] for e in events], [answer])
                    self.assertEqual(detector.feed("reasoning", ""), [])

    def test_character_stream_and_multiple_independent_channels(self):
        detector = CandidateDetector()
        self.assertEqual(detector.feed("reasoning", r"\boxed{\frac{4}"), [])
        self.assertEqual(detector.feed("content", r"\fbox{8096}")[0]["answer"], "8096")
        self.assertEqual(detector.feed("reasoning", "{9}} proof")[0]["answer"], r"\frac{4}{9}")
        detector = CandidateDetector()
        events = []
        for ch in r"\boxed  {2^{20}-1} then \boxed{n-1}":
            events.extend(detector.feed("content", ch))
        self.assertEqual([e["answer"] for e in events], ["2^{20}-1", "n-1"])

    def test_only_explicit_complete_markers_and_bounded_payload(self):
        detector = CandidateDetector()
        self.assertEqual(detector.feed("content", "the answer is 12.\n"), [])
        self.assertEqual(detector.feed("content", r"\boxed{}\boxed{" + "x" * 4097 + "}"), [])
        self.assertEqual(detector.feed("content", r"\boxed{\frac{4}{9}"), [])
        self.assertEqual(detector.feed("content", "", eof=True), [])

    def test_complete_answer_lines_and_natural_eof(self):
        detector = CandidateDetector()
        self.assertEqual(detector.feed("content", "Answer: $n-1$"), [])
        self.assertEqual(detector.feed("content", "", eof=True)[0]["answer"], "n-1")
        detector = CandidateDetector()
        event = detector.feed("reasoning", "**Answer: \\(\\frac{4}{9}\\)**\n")
        self.assertEqual(event[0]["answer"], r"\frac{4}{9}")
        # A box on an Answer line creates only the box candidate.
        self.assertEqual(len(detector.feed("content", r"Answer: \boxed{n-1}" + "\n")), 1)


class FrozenBoundaryTests(unittest.TestCase):
    def test_both_manifests_verify_and_v2_drift_rejected(self):
        self.assertEqual(len(verify_v1(ROOT)), 64)
        self.assertEqual(len(verify_core(ROOT)), 64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / MANIFEST
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"core_id": "runner_final_core_v2_1", "schema_version": 1,
                                        "sha256": {"engine.py": "0" * 64}}))
            (root / "engine.py").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "Frozen core drift.*engine.py"):
                verify_core(root)

    def test_scheduling_and_exact_continuation_policy_unchanged(self):
        def definition(module, name):
            nodes = ast.parse(Path(module.__file__).read_text()).body
            return ast.dump(next(n for n in nodes if getattr(n, "name", None) == name), include_attributes=False)
        for name in ("group_options", "question_state"):
            self.assertEqual(definition(v1, name), definition(runner, name))
        from runner_final.core_v1 import _streaming as old
        self.assertEqual(definition(old, "continuation_prefix"), definition(_streaming, "continuation_prefix"))

    def test_defaults_and_attempt_cap(self):
        args = parse_args([])
        self.assertEqual((args.parallelism, args.rollouts, args.first_pass_max_tokens, args.max_tokens,
                          args.max_attempts_per_question, args.schedule), (30, 1, 8192, 16384, 4, "barrier"))
        self.assertTrue(args.benchmark)
        self.assertIsNone(parse_args(["--reuse-grader"]).benchmark_year)
        self.assertTrue(parse_args(["--reuse-grader"]).reuse_grader)
        self.assertIn(r"\boxed{EXPRESSION}", args.system_prompt)
        self.assertNotIn("0 to 999", args.system_prompt)
        for argv in (["--max-attempts-per-question", "5"], ["--dataset-manifest", APEX, "--benchmark-year", "2025"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)


class DatasetTests(unittest.TestCase):
    def test_apex_is_pinned_gold_free_and_not_limited_to_thirty(self):
        evidence = benchmarks.dataset_provenance(None, "generalization", APEX)
        questions = benchmarks.load_questions(None, None, APEX)
        self.assertEqual((evidence["id"], evidence["year"], evidence["rows"]), ("apex_shortlist", None, 47))
        self.assertEqual([q["problem_idx"] for q in questions], list(range(1, 48)))
        self.assertTrue(all(set(q) == {"problem_idx", "problem"} for q in questions))
        self.assertEqual(len(evidence["overlap_notes"]), 2)
        self.assertEqual(benchmarks.load_questions([47], None, APEX), [questions[-1]])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            benchmarks.load_questions([48], None, APEX)

    def test_arbitrary_count_and_hash_and_prompt_key_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts, key, manifest = root / "prompts.jsonl", root / "key.jsonl", root / "source.json"
            prompts.write_text(json.dumps({"problem_idx": 7, "problem": "Compute."}) + "\n")
            key.write_text(json.dumps({"problem_idx": 7, "problem": "Compute.", "answer": "n-1"}) + "\n")
            value = {"schema_version": 1, "id": "test", "source": "test", "revision": "test", "split": "train", "rows": 1,
                     "prompt_path": str(prompts), "grader_path": str(key),
                     "prompt_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
                     "grader_sha256": hashlib.sha256(key.read_bytes()).hexdigest()}
            manifest.write_text(json.dumps(value))
            self.assertEqual(benchmarks.dataset_provenance(None, None, str(manifest))["rows"], 1)
            prompts.write_text(json.dumps({"problem_idx": 7, "problem": "Different."}) + "\n")
            with self.assertRaisesRegex(ValueError, "statements disagree"):
                benchmarks.dataset_provenance(None, None, str(manifest))
            prompts.write_text(json.dumps({"problem_idx": 7, "problem": "Compute."}) + "\n\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                benchmarks.dataset_provenance(None, None, str(manifest))

    def test_generic_metadata_validates_and_aime_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--preset", str(ROOT / "runner_final/presets/apex_core_v2_1.json")])
            config = {**vars(args), "runner_id": runner.RUNNER_ID,
                      "dataset_provenance": benchmarks.dataset_provenance(None, None, APEX)}
            metadata = build_metadata(Path(tmp), config)
            validate_metadata(metadata, Path(tmp).name)
            self.assertEqual(metadata["controls"]["dataset"], "apex_shortlist")
            self.assertEqual(metadata["provenance"]["dataset"]["rows"], 47)
            config = {**vars(parse_args([])), "dataset_provenance": benchmarks.dataset_provenance()}
            metadata = build_metadata(Path(tmp), config)
            validate_metadata(metadata, Path(tmp).name)
            self.assertEqual(metadata["controls"]["dataset"], "AIME 2025")

    def test_actual_grader_equivalence_for_transferred_formats(self):
        sys.path.insert(0, str(ROOT / "grader/grader_core"))
        from grade import grade
        pairs = [("2618", "2618"), ("4/9", r"\frac{4}{9}"), ("n*(n-1)/2", "n(n-1)/2"),
                 ("10**225-1", "10^{225}-1"), (r"9\sqrt{30}/4", r"\frac{9\sqrt{30}}{4}")]
        for candidate, gold in pairs:
            with self.subTest(candidate=candidate):
                self.assertTrue(grade(candidate, gold))
        self.assertFalse(grade("4/8", r"\frac{4}{9}"))


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_grader_supplies_arbitrary_questions_without_local_dataset(self):
        questions = [{"problem_idx": 91, "problem": "Compute the requested exact expression."}]
        evidence = {"id": "external", "year": None, "rows": 1, "grader_sha256": "f" * 64}
        health = {"n_problems": 1, "dataset": {"sha256": "f" * 64, "questions_sha256": question_digest(questions)}}
        payload = {"questions": questions, "dataset": evidence, "questions_sha256": question_digest(questions)}
        args = parse_args(["--reuse-grader", "--target-correct", "1"])
        async def handler(request):
            self.assertEqual(request.url.path, "/questions")
            return httpx.Response(200, json=payload)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            selected, _, _, _ = await fetch_questions(client, args, health)
            self.assertEqual(selected, questions)
            payload["questions"][0]["answer"] = "PRIVATE_GOLD"
            with self.assertRaisesRegex(RuntimeError, "only positive indices"):
                await fetch_questions(client, args, health)

    async def test_full_apex_lifecycle_uses_matching_grader_key_and_saves_metadata(self):
        """Exercise initialization, solving and buffered finalization with fake services."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(["--preset", str(ROOT / "runner_final/presets/apex_core_v2_1.json"),
                               "--reuse-server", "--questions", "47", "--target-correct", "1"])
            args.models_dir = str(root / "models")
            profile = Path(args.models_dir) / args.model / args.model_profile
            profile.parent.mkdir(parents=True)
            profile.write_text("max-model-len: 16384\noverride-generation-config:\n  max_new_tokens: 16384\n")
            evidence = benchmarks.dataset_provenance(None, None, APEX)
            all_questions = benchmarks.load_questions(None, None, APEX)
            closed, submissions = [], []
            class Services:
                def launch(self, command, log, env):
                    import yaml
                    settings = yaml.safe_load(Path(env["GRADER_CONFIG"]).read_text())
                    self_case.assertEqual(Path(settings["dataset"]["source"]), ROOT / "grader/data/apex_shortlist.jsonl")
                    return object()
                async def close(self):
                    closed.append(True)
            self_case = self
            async def handler(request):
                if request.url.path == "/questions":
                    return httpx.Response(200, json={"questions": all_questions, "dataset": evidence,
                                                     "questions_sha256": question_digest(all_questions)})
                body = json.loads(request.content)
                if request.url.path == "/verify":
                    submissions.append(body)
                    return httpx.Response(200, json={"verdict": True})
                self.assertEqual(body["messages"][1]["content"], benchmarks.load_questions([47], None, APEX)[0]["problem"])
                return httpx.Response(200, stream=Stream([chunk(r"\boxed{10^{225}-1}")], hang=True))
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            models = {"data": [{"id": args.model, "max_model_len": 16384}]}
            health = {"queries_so_far": 0, "cost_c": args.grader_cost, "n_problems": 47,
                      "dataset": {"sha256": evidence["grader_sha256"], "questions_sha256": question_digest(all_questions)}}
            with (patch.object(runner, "ROOT", root), patch("runner_final.integrity_v2_1.verify_core", return_value="f" * 64),
                  patch.object(runner, "Services", Services), patch.object(runner, "ensure_free"),
                  patch.object(runner.httpx, "AsyncClient", return_value=client),
                  patch.object(runner, "ready", AsyncMock(side_effect=[models, health])),
                  patch.object(runner, "warm_inference", AsyncMock(return_value={"latency_s": 0, "batch_size": 1, "tokens_per_request": 32})),
                  patch.object(runner.subprocess, "check_output", return_value="test-commit"),
                  redirect_stdout(io.StringIO())):
                output = await runner.run(args)
            summary = json.loads((output / "summary.json").read_text())
            config = json.loads((output / "config.json").read_text())
            metadata = json.loads((output / "metadata.json").read_text())
            validate_metadata(metadata, output.name)
            self.assertTrue(summary["target_reached"])
            self.assertEqual(summary["questions"][0]["verified_answer"], "10^{225}-1")
            self.assertEqual(config["dataset_provenance"]["rows"], 47)
            self.assertEqual(config["runner_module"], "runner_final.run_frozen_v2_1")
            self.assertEqual(json.loads((output / "questions.json").read_text()), all_questions)
            self.assertEqual(submissions[0]["index"], 47)
            self.assertEqual(closed, [True])
            from src.attempt_viewer import AttemptStore
            from src.attempt_results import build_results
            store = AttemptStore(root / "attempts", ROOT / "data/aime_2025_problems.jsonl")
            overview = store.overview(output.name)
            self.assertEqual(overview["questions"][0]["problem"], all_questions[46]["problem"])
            results = build_results(store)
            self.assertEqual(results["warnings"], [])
            self.assertEqual(results["attempts"][0]["benchmark_id"], "apex_shortlist")
            self.assertEqual(results["attempts"][0]["benchmark_year"], None)

    async def test_wrong_and_duplicate_expressions_then_correct_cancel_stream(self):
        checks, streams = [], []
        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/v1/chat/completions":
                stream = Stream([chunk(r"\boxed{4/8}"), chunk(r"\boxed{4/8}"), chunk(r"\boxed{\frac{4}{9}}")], delay=.003, hang=True)
                streams.append(stream)
                return httpx.Response(200, stream=stream)
            checks.append(body["candidate"])
            self.assertLess(len(body["query_id"]), 200)
            await asyncio.sleep(.015)
            return httpx.Response(200, json={"verdict": body["candidate"] == r"\frac{4}{9}"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(["--target-correct", "1"])
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await _streaming.run_question({"problem_idx": 27, "problem": "Compute a fraction."}, args, client, root, FakeGPU())
            self.assertEqual(checks, ["4/8", r"\frac{4}{9}"])
            self.assertEqual(result["winner"]["answer"], r"\frac{4}{9}")
            self.assertTrue(streams[0].closed)

    async def test_split_nested_box_across_exact_id_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AttemptArtifacts(root, buffered=True)
            profiler = AttemptProfiler(root, enabled=False, artifacts=store)
            args = parse_args(["--target-correct", "1"])
            requests, checks = [], []
            async def handler(request):
                body = json.loads(request.content)
                if request.url.path == "/verify":
                    checks.append(body["candidate"])
                    return httpx.Response(200, json={"verdict": True})
                requests.append(body)
                if len(requests) == 1:
                    return httpx.Response(200, stream=Stream(capped([10], [11], r"\boxed{\frac{4}{")))
                self.assertEqual(body["prompt"], [10, 11])
                return httpx.Response(200, stream=Stream(capped([10, 11], [12], "9}}", completion=True)))
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await runner.run_speedrun([{"problem_idx": 1, "problem": "Compute."}], args, client, root, DisabledGPUSampler(), time.perf_counter(), profiler)
            self.assertEqual(checks, [r"\frac{4}{9}"])
            self.assertEqual(result[0]["status"], "solved")
            self.assertEqual(len(requests), 2)
            self.assertEqual(list(root.rglob("*")), [])
            store.flush()
            self.assertEqual(len((root / "solved.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
