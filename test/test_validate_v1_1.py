"""Audit checks against transformed, committed historical controls; no GPU work."""
import json
from pathlib import Path
import tempfile
import unittest

from runner_final import validate_v1_1 as driver
from src.common import ROOT


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        protocol = json.loads(driver.PROTOCOL.read_text())
        self.seed = protocol['seeds'][0]
        self.reference = protocol['reference_attempts'][str(self.seed)]
        control = ROOT/'attempts'/self.reference
        config = json.loads((control/'config.json').read_text())
        config.update(runner_id='runner_final_core_v1_1', core_manifest_sha256=protocol['core_manifest_sha256'],
                      model_profile_sha256=protocol['profile_sha256'], git_dirty=False,
                      parallelism=30, schedule='eager', rollouts=1, no_continuation=True,
                      first_pass_max_tokens=65536, max_tokens=65536,
                      served_prompt_tokens={str(i): 204 for i in range(1,31)})
        self.write(self.output/'config.json', config)
        for filename in ('summary.json','inference_warmup.json'):
            (self.output/filename).write_bytes((control/filename).read_bytes())
        for path in control.glob('trace/*/question.json'):
            q = json.loads(path.read_text())
            q['rollouts'] = q['rollouts'][:1]
            r = q['rollouts'][0]
            r.update(continuation_of_rollout=None, endpoint='/v1/chat/completions', requested_max_tokens=65536-204)
            self.write(self.output/path.relative_to(control), q)
            req_path = path.parent/'rollout-01/request.json'
            req = json.loads(req_path.read_text())
            req['max_tokens'] = 65536-204
            self.write(self.output/req_path.relative_to(control), req)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def audit(self):
        return driver.score(self.output, self.seed, self.reference)

    def test_accepts_declared_output_cap_change(self):
        self.assertTrue(self.audit()['valid'])

    def test_rejects_continuation_and_extra_tokens(self):
        path = self.output/'trace/01/question.json'
        q = json.loads(path.read_text())
        q['rollouts'][0].update(continuation_of_rollout=1, generated_token_ids_count=65536)
        self.write(path, q)
        self.assertFalse(self.audit()['caps_valid'])
        self.assertFalse(self.audit()['valid'])

    def test_rejects_sampling_change(self):
        path = self.output/'trace/01/rollout-01/request.json'
        req = json.loads(path.read_text())
        req['seed'] += 1
        self.write(path, req)
        self.assertEqual(self.audit()['different_requests_beyond_declared_output_cap'], [[1,1]])
        self.assertFalse(self.audit()['valid'])

    def test_retains_failure_in_aggregate(self):
        protocol = json.loads(driver.PROTOCOL.read_text())
        rows = [{'sampling_seed':s, 'valid':i != 0, 'time_to_target_s':None if i == 0 else 80}
                for i,s in enumerate(protocol['seeds'])]
        result = driver.aggregate(rows,protocol)
        self.assertEqual(result['successful_trials'],4)
        self.assertTrue(result['all_declared_seeds_retained'])
