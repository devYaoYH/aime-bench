"""Declared matched seeds and failed-trial retention for the AWQ expansion."""

import unittest
from runner.extensions.validation import awq_v1_6_batch as driver


class BatchTests(unittest.TestCase):
    def test_pilot_is_reused_without_repeating_its_seed(self):
        plan = driver.plan()
        self.assertEqual(plan["seeds"], [20261011, 20261012, 20261013, 20261014, 20261015])
        self.assertEqual(plan["remaining_seeds"], plan["seeds"][1:])
        self.assertTrue(plan["pilot_trial"]["valid"])
        self.assertEqual(plan["pilot_trial"]["attempt_id"], "20261004T232342.669339Z")

    def test_aggregation_keeps_failures_in_recorded_count(self):
        result = driver.aggregate([
            {"valid": True, "target_reached": True, "time_to_target_s": 100},
            {"valid": False, "target_reached": False, "time_to_target_s": None},
            {"valid": True, "target_reached": True, "time_to_target_s": 80},
        ])
        self.assertEqual(result["declared_trials"], 5)
        self.assertEqual(result["recorded_trials"], 3)
        self.assertEqual(result["valid_target_trials"], 2)
        self.assertEqual(result["median_s"], 90)
        self.assertEqual(result["min_s"], 80)
        self.assertEqual(result["max_s"], 100)

    def test_empty_batch_does_not_invent_success_statistics(self):
        result = driver.aggregate([])
        self.assertEqual(result["valid_target_trials"], 0)
        self.assertIsNone(result["median_s"])
        self.assertIsNone(result["sample_sd_s"])


if __name__ == "__main__":
    unittest.main()
