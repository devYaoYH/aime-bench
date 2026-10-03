import unittest
from src.experiments.replay_best_source_v1 import historical_argv
from src.attempt_runners.speedrun_v2 import parse_args


class HistoricalControlsTests(unittest.TestCase):
    def test_replay_uses_original_storage_and_fresh_managed_server(self):
        args = parse_args(historical_argv('/test/grader/python'))
        self.assertFalse(args.reuse_server)
        self.assertFalse(args.benchmark)
        self.assertFalse(args.buffer_traces)
        self.assertFalse(args.no_gpu_telemetry)
        self.assertEqual(args.grader_python, '/test/grader/python')
        self.assertEqual((args.parallelism, args.rollouts, args.schedule), (30, 1, 'barrier'))
        self.assertEqual((args.first_pass_max_tokens, args.max_tokens), (8192, 16384))
        self.assertEqual((args.max_attempts_per_question, args.seed, args.target_correct), (4, 20261003, 18))
