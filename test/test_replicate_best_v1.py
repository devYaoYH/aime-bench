import unittest
from unittest.mock import AsyncMock

import httpx

from src.experiments.replicate_best_v1 import controls, reset_cache
from src.attempt_runners.speedrun_v2 import parse_args


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
