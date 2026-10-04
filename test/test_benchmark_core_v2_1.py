import unittest
from src.experiments.benchmark_core_v2_1 import plan
from runner_final.run_frozen_v2_1 import parse_args

class BenchmarkPlanTests(unittest.TestCase):
    def test_both_datasets_share_declared_benchmark_controls(self):
        jobs=plan(20261011)
        self.assertEqual([j['dataset'] for j in jobs],['AIME 2025','Apex shortlist'])
        for job in jobs:
            args=parse_args(job['argv'])
            self.assertTrue(args.benchmark)
            self.assertTrue(args.no_overhead_profile)
            self.assertTrue(args.no_gpu_telemetry)
            self.assertTrue(args.buffer_traces)
            self.assertEqual(args.seed,20261011)
            self.assertEqual((args.parallelism,args.rollouts,args.max_attempts_per_question),(30,1,4))
