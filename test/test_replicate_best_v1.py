import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import yaml

from src.experiments.replicate_best_v1 import controls, reset_cache
from src.attempt_runners.speedrun_v2 import parse_args
from src.attempt_runners.speedrun_v4 import parse_args as parse_dynamic
from src.attempt_runners.speedrun_v5 import parse_args as parse_paused


class BestReplicationTests(unittest.TestCase):
    def test_matched_controls_and_four_request_cap(self):
        args = parse_args(controls("WeiboAI/VibeThinker-3B"))
        self.assertEqual((args.parallelism, args.rollouts, args.schedule), (30, 1, "barrier"))
        self.assertEqual((args.first_pass_max_tokens, args.max_tokens), (8192, 16384))
        self.assertEqual(args.seed, 20261003)
        self.assertEqual(args.max_attempts_per_question, 4)
        self.assertEqual(args.target_correct, 18)
        self.assertTrue(args.reuse_server)
        self.assertTrue(args.benchmark)
        self.assertTrue(args.no_gpu_telemetry)

    def test_quantized_comparison_preserves_other_controls(self):
        a = controls("WeiboAI/VibeThinker-3B")
        b = controls("r0b0tlab/VibeThinker-3B-NVFP4")
        self.assertEqual([i for i,(x,y) in enumerate(zip(a,b)) if x != y], [a.index("--model") + 1])

    def test_original_profiled_control_restores_storage_and_sampler(self):
        args = parse_args(controls("WeiboAI/VibeThinker-3B", benchmark=False))
        self.assertFalse(args.benchmark)
        self.assertFalse(args.buffer_traces)
        self.assertFalse(args.no_overhead_profile)
        self.assertFalse(args.no_gpu_telemetry)
        self.assertEqual((args.parallelism, args.rollouts, args.schedule), (30, 1, "barrier"))

    def test_dynamic_recycling_preserves_request_and_memory_caps(self):
        args = parse_dynamic(controls("WeiboAI/VibeThinker-3B", policy="dynamic30"))
        self.assertEqual(args.max_concurrent_requests, 30)
        self.assertEqual(args.max_attempts_per_question, 4)
        self.assertEqual(args.token_budgets, [8192, 16384])
        self.assertEqual(args.seed_stride, 4)
        self.assertEqual((args.parallelism, args.rollouts), (30, 1))
        self.assertTrue(args.benchmark)

    def test_sampling_intervention_changes_only_declared_seed(self):
        original = controls("WeiboAI/VibeThinker-3B")
        changed = controls("WeiboAI/VibeThinker-3B", seed=20261004)
        self.assertEqual([i for i, (a, b) in enumerate(zip(original, changed)) if a != b], [original.index('--seed') + 1])
        self.assertEqual(parse_args(changed).seed, 20261004)

    def test_pending_verdict_policy_matches_dynamic30_request_controls(self):
        model = 'r0b0tlab/VibeThinker-3B-NVFP4'
        control = controls(model, policy='dynamic30', model_profile='vllm-flashinfer.yaml')
        paused = controls(model, policy='paused30', model_profile='vllm-flashinfer.yaml')
        self.assertEqual(control, paused)
        args = parse_paused(paused)
        self.assertEqual(args.max_attempts_per_question, 4)
        self.assertEqual(args.max_concurrent_requests, 30)
        self.assertEqual(args.token_budgets, [8192, 16384])
        self.assertEqual(args.schedule, 'suspend_pending_verdict')

    def test_attention_profile_retains_precision_budget_and_request_controls(self):
        model = 'r0b0tlab/VibeThinker-3B-NVFP4'
        original = controls(model)
        changed = controls(model, model_profile='vllm-flashinfer.yaml')
        self.assertEqual([i for i, (a, b) in enumerate(zip(original, changed)) if a != b], [original.index('--model-profile') + 1])
        root = Path(__file__).resolve().parents[1]
        profile = root / 'configs/vllm' / model
        base = yaml.safe_load((profile / 'vllm.yaml').read_text())
        trial = yaml.safe_load((profile / 'vllm-flashinfer.yaml').read_text())
        self.assertEqual(trial.pop('attention-backend'), 'FLASHINFER')
        self.assertEqual(trial, base)


class CacheResetTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_held_blocks_without_forcing_running_reset(self):
        client = AsyncMock()
        request = httpx.Request("POST", "http://localhost/reset_prefix_cache")
        client.post.side_effect = [httpx.Response(200, json={"success": False}, request=request),
                                   httpx.Response(200, json={"success": True}, request=request)]
        result = await reset_cache(client, "http://localhost", attempts=2)
        self.assertTrue(result["success"])
        self.assertEqual(client.post.await_count, 2)
        client.post.assert_awaited_with("http://localhost/reset_prefix_cache", timeout=10)

    async def test_false_reset_aborts_instead_of_reusing_solution_cache(self):
        client = AsyncMock()
        request = httpx.Request("POST", "http://localhost/reset_prefix_cache")
        client.post.return_value = httpx.Response(200, json={"success": False}, request=request)
        with self.assertRaisesRegex(RuntimeError, "Prefix cache still held"):
            await reset_cache(client, "http://localhost", attempts=1)

    async def test_missing_endpoint_is_not_treated_as_cache_reset(self):
        client = AsyncMock()
        client.post.return_value = httpx.Response(404, request=httpx.Request("POST", "http://localhost/reset_prefix_cache"))
        with self.assertRaises(httpx.HTTPStatusError):
            await reset_cache(client, "http://localhost")
