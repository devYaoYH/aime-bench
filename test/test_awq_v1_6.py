"""AWQ quantization-comparison profile and unchanged policy controls."""

import json
from pathlib import Path
import tempfile
import shutil
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch
import yaml
from runner.extensions.validation import awq_v1_6 as driver
from runner.extensions.v1_6.integrity import verify_core
from runner.lib.common import ROOT


class ComparisonTests(unittest.TestCase):
    def test_semantic_token_check_requests_plain_ids_on_new_transformers(self):
        reference = driver.reference_trial(20261011)
        folder = ROOT / "attempts" / reference["attempt_id"]
        reference_ids = {}
        for path in folder.glob("trace/*/rollout-01/request.json"):
            messages = json.loads(path.read_text())["messages"]
            reference_ids[json.dumps(messages)] = json.loads(
                path.with_name("tokens.json").read_text()
            )["prompt_token_ids"]
        test = self

        class Tokenizer:
            all_special_ids = [151643]

            def get_vocab(self):
                return {"test": 1}

            def apply_chat_template(self, messages, **kwargs):
                test.assertIs(kwargs.get("return_dict"), False)
                return reference_ids[json.dumps(messages)]

            def decode(self, tokens):
                return str(tokens)

        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *a, **k: Tokenizer()
        ))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"transformers": fake}):
            old_model = json.loads((folder / "config.json").read_text())["model"]
            for model in (driver.MODEL, old_model):
                location = Path(tmp) / model
                location.mkdir(parents=True)
                config = {"quantization_config": {
                    "quant_method": "awq", "bits": 4,
                    "group_size": 128, "zero_point": True,
                } if model == driver.MODEL else {"quant_method": "modelopt_fp4"}}
                (location / "config.json").write_text(json.dumps(config))
                (location / "generation_config.json").write_text("{}")
            evidence = driver.model_provenance(tmp, reference)
            self.assertEqual(len(evidence["tokenizer_semantic_check"]["questions"]), 30)

    def test_same_policy_and_controls_except_model_identity(self):
        ref = driver.reference_trial(20261011)
        old = driver.harness.baseline.load(
            ROOT / "attempts" / ref["attempt_id"] / "config.json"
        )
        args = driver.trial_args(20261011, "/external/grader/python")
        args.runtime_package_versions = old["runtime_package_versions"]
        driver.check_controls(vars(args), old)
        self.assertEqual(verify_core(ROOT), old["core_manifest_sha256"])
        self.assertEqual(args.model, "AABoyles/VibeThinker-3B-AWQ")
        self.assertEqual(args.max_rollout_tokens, 65536)
        self.assertEqual(args.max_concurrent_requests, 30)
        args.first_pass_max_tokens = 4096
        with self.assertRaisesRegex(RuntimeError, "first_pass_max_tokens"):
            driver.check_controls(vars(args), old)

    def test_artifact_audit_accepts_normalized_model_identity_and_rejects_prefix_drift(
        self,
    ):
        reference = driver.reference_trial(20261011)
        original = ROOT / "attempts" / reference["attempt_id"]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "attempt"
            shutil.copytree(original, output)
            path = output / "config.json"
            config = json.loads(path.read_text())
            config.update(model=driver.MODEL, model_profile=driver.PROFILE)
            path.write_text(json.dumps(config))
            for path in output.glob("trace/*/rollout-*/request.json"):
                request = json.loads(path.read_text())
                request["model"] = driver.MODEL
                path.write_text(json.dumps(request))
            core = verify_core(ROOT)
            result = driver.audit(output, reference, core)
            self.assertTrue(result["valid"])
            continuation = next(
                p
                for p in output.glob("trace/*/rollout-*/request.json")
                if "prompt" in json.loads(p.read_text())
            )
            body = json.loads(continuation.read_text())
            body["prompt"][0] += 1
            continuation.write_text(json.dumps(body))
            result = driver.audit(output, reference, core)
            self.assertFalse(result["valid"])
            self.assertEqual(len(result["continuation_errors"]), 1)

    def test_awq_marlin_and_flashinfer_match_other_profile_settings(self):
        path = ROOT / "configs/vllm" / driver.MODEL / driver.PROFILE
        profile = driver.validate_profile(path, driver.reference_trial(20261011))
        self.assertEqual(profile["attention-backend"], "FLASHINFER")
        self.assertEqual(profile["dtype"], "bfloat16")
        self.assertEqual(profile["linear-backend"], "marlin")
        self.assertEqual(profile["quantization"], "awq")
        self.assertEqual(
            json.loads(profile["override-generation-config"]), {"max_new_tokens": 65536}
        )
        for key, value in [
            ("linear-backend", "auto"),
            ("quantization", "modelopt_fp4"),
            ("attention-backend", "TRITON_ATTN"),
            ("gpu-memory-utilization", 0.8),
            ("kv-cache-dtype", "fp8"),
        ]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                altered = Path(tmp) / "profile.yaml"
                altered.write_text(yaml.safe_dump({**profile, key: value}))
                with self.assertRaises(RuntimeError):
                    driver.validate_profile(altered, driver.reference_trial(20261011))


if __name__ == "__main__":
    unittest.main()
