import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from src.experiments import benchmark_core_v2_3 as batch
from runner_final.run_frozen_v2_3 import parse_args


class BenchmarkPlanTests(unittest.IsolatedAsyncioTestCase):
    async def test_separate_official_deadlines_and_frozen_controls(self):
        jobs=batch.plan(20261011)
        self.assertEqual([j['official_deadline_s'] for j in jobs],[900,300])
        for job in jobs:
            args=parse_args(job['argv'])
            self.assertEqual(args.seed,20261011)
            self.assertTrue(args.benchmark and args.buffer_traces and args.no_gpu_telemetry)
            self.assertEqual((args.max_tokens,args.rollouts,args.max_attempts_per_question),(65536,1,4))
            self.assertIsNone(args.max_concurrent_requests)

    async def test_deadline_failure_is_retained_and_next_job_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            deadline=AsyncMock(side_effect=[{'deadline_reached':True,'attempt_id':'partial'},
                                           {'deadline_reached':False,'attempt_id':None,'error':'startup failed'}])
            options=SimpleNamespace(batch='test',seed=20261011,aime_deadline_s=900,apex_deadline_s=300)
            with (patch.object(batch,'ROOT',root),
                  patch.object(batch.subprocess,'check_output',side_effect=['','test-commit']),
                  patch.object(batch,'verify_core',return_value='f'*64),
                  patch.object(batch,'parse_args',return_value=object()),
                  patch.object(batch,'measure',side_effect=FileNotFoundError('partial artifacts')),
                  patch.object(batch,'with_official_deadline',deadline)):
                await batch.execute(options)
            result=json.loads((root/'runs/experiments/test/summary.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual([c.args[3] for c in deadline.await_args_list],[900,300])
            self.assertEqual(len(result['trials']),2)
            self.assertTrue(result['trials'][0]['deadline_reached'])
            self.assertIn('partial artifacts',result['trials'][0]['measurement_error'])
            self.assertEqual(result['trials'][1]['error'],'startup failed')
