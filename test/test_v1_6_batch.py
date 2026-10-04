"""Matched historical controls and retention of all declared refactor trials."""

import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from runner.extensions.validation import v1_6_batch as driver
from runner.lib.common import ROOT


class ControlTests(unittest.TestCase):
    def test_declared_seeds_and_nonpolicy_controls_match_all_original_trials(self):
        self.assertEqual(
            driver.SEEDS, [20261011, 20261012, 20261013, 20261014, 20261015]
        )
        for row in driver.reference_trials():
            args = driver.trial_args(row["sampling_seed"], "/external/grader/python")
            old = driver.load(ROOT / "attempts" / row["attempt_id"] / "config.json")
            # Local library versions can differ; the real remote preflight checks them.
            args.runtime_package_versions = old["runtime_package_versions"]
            driver.check_controls(args, old)
            args.rollouts = 2
            with self.assertRaisesRegex(RuntimeError, "rollouts"):
                driver.check_controls(args, old)
        self.assertEqual(len(driver.grader_semantics()), 3)

    def test_long_profile_preserves_the_named_historical_profile_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models/model"
            committed = root / "configs/vllm/model"
            models.mkdir(parents=True)
            committed.mkdir(parents=True)
            original = models / "vllm-flashinfer.yaml"
            profile = models / "vllm-v1_6-long64k.yaml"
            old_values = {
                "gpu-memory-utilization": 0.95,
                "max-model-len": 65536,
                "override-generation-config": '{"max_new_tokens": 16384}',
            }
            new_values = {
                **old_values,
                "override-generation-config": '{"max_new_tokens": 65536}',
            }
            original.write_text(driver.yaml.safe_dump(old_values))
            profile.write_text(driver.yaml.safe_dump(new_values))
            (committed / profile.name).write_bytes(profile.read_bytes())
            old = {
                "profile_path": "/remote/models/model/" + original.name,
                "profile_sha256": driver.hashlib.sha256(
                    original.read_bytes()
                ).hexdigest(),
            }
            args = SimpleNamespace(
                models_dir=str(root / "models"),
                model="model",
                model_profile=profile.name,
            )
            with patch.object(driver, "ROOT", root):
                driver.validate_profile(profile, args, old)
                new_values["gpu-memory-utilization"] = 0.8
                profile.write_text(driver.yaml.safe_dump(new_values))
                (committed / profile.name).write_bytes(profile.read_bytes())
                with self.assertRaisesRegex(
                    RuntimeError, "Unexpected model profile change"
                ):
                    driver.validate_profile(profile, args, old)
            self.assertEqual(
                driver.hashlib.sha256(original.read_bytes()).hexdigest(),
                old["profile_sha256"],
            )

    def test_failure_cannot_pass_median_screen(self):
        rows = [
            {
                "valid": True,
                "time_to_target_s": 77.0,
                "reference_time_to_target_s": 77.0,
                "delta_s": 0.0,
            }
            for _ in range(4)
        ]
        rows.append({"valid": False, "status": "failed", "time_to_target_s": None})
        result = driver.aggregate(rows)
        self.assertEqual(result["valid_trials"], 4)
        self.assertFalse(result["all_five_valid"])
        self.assertFalse(result["within_practical_10pct_median_band"])

    def test_large_latency_regression_is_flagged(self):
        rows = [
            {
                "valid": True,
                "time_to_target_s": 90.0,
                "reference_time_to_target_s": 77.0,
                "delta_s": 13.0,
            }
            for _ in range(5)
        ]
        self.assertFalse(driver.aggregate(rows)["within_practical_10pct_median_band"])


class RetentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_retained_without_replacement_and_server_closed(self):
        seen = []
        services = SimpleNamespace(launch=lambda *a, **kw: object(), close=AsyncMock())
        references = driver.reference_trials()
        original_args = driver.trial_args(driver.SEEDS[0], "/grader/python")
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (root / "attempts").mkdir()
            profile = root / "model.yaml"
            profile.write_text("profile")
            original_args.models_dir = str(root)
            original_args.model = "model"
            original_args.model_profile = "profile.yaml"
            profile = root / "model/profile.yaml"
            profile.parent.mkdir()
            profile.write_text("profile")

            async def run(args):
                seen.append(args.seed)
                output = root / "attempts" / str(args.seed)
                output.mkdir()
                if len(seen) == 3:
                    raise RuntimeError("failed trial")
                return output

            def args(seed, grader):
                original_args.seed = seed
                return original_args

            real_load = driver.load

            def load(path):
                if path.name == "config.json":
                    return {
                        "profile_sha256": driver.hashlib.sha256(b"profile").hexdigest(),
                        "source_commit": "old",
                    }
                return real_load(path)

            with (
                patch.object(driver, "ROOT", root),
                patch.object(driver, "reference_trials", return_value=references),
                patch.object(driver, "trial_args", side_effect=args),
                patch.object(driver, "check_controls"),
                patch.object(driver, "verify_core", return_value="core"),
                patch.object(driver, "validate_profile"),
                patch.object(driver, "grader_semantics", return_value={}),
                patch.object(driver, "load", side_effect=load),
                patch.object(driver, "Services", return_value=services),
                patch.object(driver, "assert_gpu_idle"),
                patch.object(driver, "ensure_free"),
                patch.object(driver, "ready", new=AsyncMock()),
                patch.object(
                    driver, "reset_cache", new=AsyncMock(return_value={"success": True})
                ),
                patch.object(driver, "run", side_effect=run),
                patch.object(
                    driver,
                    "measure",
                    side_effect=lambda out, ref, core: {
                        "valid": True,
                        "time_to_target_s": 77.0,
                        "delta_s": 77.0 - ref["time_to_target_s"],
                    },
                ),
                patch.object(
                    driver.subprocess, "check_output", side_effect=["", "commit"]
                ),
            ):
                batch = await driver.execute(
                    SimpleNamespace(
                        batch="test", grader_python="/grader/python", trial_timeout=30
                    )
                )
            saved = json.loads((batch / "summary.json").read_text())
            self.assertEqual(seen, driver.SEEDS)
            self.assertEqual(saved["trials"][2]["status"], "failed")
            self.assertEqual(saved["trials"][2]["attempt_id"], str(driver.SEEDS[2]))
            self.assertEqual(len(saved["trials"]), 5)
            self.assertFalse(saved["all_five_valid"])
            services.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
