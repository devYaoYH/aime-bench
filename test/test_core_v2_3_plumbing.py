"""Owned-service initialization, context accounting and mixed viewer dispatch."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import httpx
from runner_final.core_v2_3 import benchmarks, runner
from runner_final.core_v2_3.grader_questions import question_digest
from runner_final.core_v2_3.metadata import validate_metadata
from runner_final.run_frozen_v2_3 import parse_args
from src.common import ROOT
from test.test_attempt import Stream, chunk
APEX="data/source_apex_shortlist.json"

class Lifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_subset_uses_one_slot_with_gold_free_snapshot(self):
        await self._lifecycle(subset=True)

    async def test_full_apex_uses_47_slots_and_stops_at_target(self):
        await self._lifecycle(subset=False)

    async def _lifecycle(self, subset=True):
        """Exercise initialization, solving and buffered finalization with fake services."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(["--preset", str(ROOT / "runner_final/presets/apex_core_v2_3.json"),
                               "--reuse-server", "--grader-cost", "0"] + (["--questions", "47", "--target-correct", "1"] if subset else []))
            args.models_dir = str(root / "models")
            profile = Path(args.models_dir) / args.model / args.model_profile
            profile.parent.mkdir(parents=True)
            profile.write_text("max-model-len: 65536\noverride-generation-config:\n  max_new_tokens: 65536\n")
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
                if request.url.path == "/tokenize":
                    self.assertIn("messages", body)
                    return httpx.Response(200, json={"count": 3, "tokens": [1,2,3]})
                self.assertEqual(request.url.path, "/v1/chat/completions")
                idx=(body["seed"]-args.seed-1)//4
                self.assertEqual(body["messages"][1]["content"], all_questions[idx-1]["problem"])
                self.assertEqual(body["max_tokens"], 65533)
                return httpx.Response(200, stream=Stream([chunk(r"\boxed{10^{225}-1}")], hang=True))
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            models = {"data": [{"id": args.model, "max_model_len": 65536}]}
            health = {"queries_so_far": 0, "cost_c": args.grader_cost, "n_problems": 47,
                      "dataset": {"sha256": evidence["grader_sha256"], "questions_sha256": question_digest(all_questions)}}
            with (patch.object(runner, "ROOT", root), patch("runner_final.integrity_v2_3.verify_core", return_value="f" * 64),
                  patch.object(runner, "Services", Services), patch.object(runner, "ensure_free"),
                  patch.object(runner.httpx, "AsyncClient", return_value=client) as client_factory,
                  patch.object(runner, "ready", AsyncMock(side_effect=[models, health])),
                  patch.object(runner, "warm_inference", AsyncMock(return_value={"latency_s": 0, "batch_size": 1, "tokens_per_request": 32})),
                  patch.object(runner.subprocess, "check_output", return_value="test-commit"),
                  redirect_stdout(io.StringIO())):
                output = await runner.run(args)
            summary = json.loads((output / "summary.json").read_text())
            self.assertIsNone(client_factory.call_args.kwargs['limits'].max_connections)
            config = json.loads((output / "config.json").read_text())
            metadata = json.loads((output / "metadata.json").read_text())
            validate_metadata(metadata, output.name)
            self.assertTrue(summary["target_reached"])
            self.assertEqual(summary["questions"][0]["verified_answer"], "10^{225}-1")
            self.assertEqual(config["dataset_provenance"]["rows"], 47)
            self.assertEqual(config["runner_module"], "runner_final.run_frozen_v2_3")
            self.assertEqual(json.loads((output / "questions.json").read_text()), all_questions)
            self.assertEqual(submissions[0]["index"], 47 if subset else 1)
            n=1 if subset else 47
            self.assertEqual((config["parallelism"], config["max_concurrent_requests"]), (n,n))
            self.assertEqual(config["policy"]["slots"], n)
            self.assertEqual(len(config["served_prompt_tokens"]), n)
            self.assertEqual(metadata["controls"]["hyperparameters"]["max_attempts_per_question"],4)
            self.assertEqual(metadata["controls"]["hyperparameters"]["first_pass_max_tokens"],65536)
            allocation=json.loads((output / "allocation.json").read_text())
            self.assertLessEqual(allocation["peak_active_requests"],n)
            self.assertTrue(all(v["used"] <= 4 for v in allocation["questions"].values()))
            self.assertTrue(config["benchmark"])
            self.assertIsNone(summary["performance"]["official_gpu"])
            self.assertEqual(closed, [True])
            from src.attempt_viewer import AttemptStore
            from src.attempt_results import build_results
            store = AttemptStore(root / "attempts", ROOT / "data/aime_2025_problems.jsonl")
            overview = store.overview(output.name)
            self.assertEqual(overview["questions"][0]["problem"], all_questions[46 if subset else 0]["problem"])
            results = build_results(store)
            self.assertEqual(results["warnings"], [])
            self.assertEqual(results["attempts"][0]["benchmark_id"], "apex_shortlist")
            self.assertEqual(results["attempts"][0]["benchmark_year"], None)
