"""The v1.5 extension retains the completed trial and never replaces failures."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runner_final import validate_v1_5 as driver


class ExtensionTests(unittest.IsolatedAsyncioTestCase):
    def test_existing_attempt_passes_unchanged_core_and_control_checks(self):
        protocol=json.loads(driver.PROTOCOL.read_text())
        row=driver.score(driver.ROOT/'attempts'/protocol['existing_attempt'],protocol['seeds'][0],
                         protocol['reference_attempts'][str(protocol['seeds'][0])])
        self.assertTrue(row['valid'])
        self.assertEqual(row['generation_requests'],73)
        self.assertEqual(row['later_fresh_requests'],29)

    def test_failed_trial_excluded_from_statistics_but_completion_tracks_all_seeds(self):
        protocol=json.loads(driver.PROTOCOL.read_text())
        rows=[{'sampling_seed':s,'valid':s!=20261013,'time_to_target_s':60 if s!=20261013 else None}
              for s in protocol['seeds']]
        result=driver.aggregate(rows,protocol)
        self.assertEqual(result['successful_trials'],4)
        self.assertTrue(result['all_declared_seeds_retained'])
        self.assertEqual(result['median_time_to_target_s'],60)
        self.assertFalse(driver.aggregate(rows[:-1],protocol)['all_declared_seeds_retained'])

    async def test_only_remaining_four_run_once_and_failure_is_retained(self):
        protocol=json.loads(driver.PROTOCOL.read_text());seen=[]
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);(root/'attempts').mkdir()
            profile=root/'models/r0b0tlab/VibeThinker-3B-NVFP4/vllm-flashinfer.yaml'
            profile.parent.mkdir(parents=True)
            profile.write_text(Path(driver.__file__).with_name('vllm-flashinfer.yaml').read_text())
            original=driver.parse_args
            def parse(argv):return original(argv+['--models-dir',str(root/'models')])
            async def run(args):
                seen.append(args.seed);self.assertFalse(args.reuse_server)
                self.assertEqual((args.max_concurrent_requests,args.max_attempts_per_question,args.first_pass_max_tokens,args.max_tokens),(30,4,8192,16384))
                out=root/'attempts'/str(args.seed);out.mkdir()
                if args.seed==20261013:raise RuntimeError('declared failure')
                return out
            def score(out,seed,reference):return {'sampling_seed':seed,'attempt_id':out.name,'valid':True,'time_to_target_s':65,'reference_attempt_id':reference}
            with patch.object(driver,'ROOT',root),patch.object(driver,'verify_core',return_value=protocol['core_manifest_sha256']),patch.object(driver,'parse_args',new=parse),patch.object(driver.runner,'run',new=run),patch.object(driver,'score',new=score),patch.object(driver.subprocess,'check_output',side_effect=['','a'*40]):
                batch=await driver.execute(SimpleNamespace(batch='extension',grader_python='/tmp/grader-python'))
            result=json.loads((batch/'summary.json').read_text())
            self.assertEqual(seen,protocol['new_seeds'])
            self.assertEqual([r['sampling_seed'] for r in result['trials']],protocol['seeds'])
            self.assertTrue(result['trials'][0]['retained_existing_trial'])
            self.assertEqual(result['trials'][2]['status'],'failed')
            self.assertEqual(result['trials'][2]['attempt_id'],'20261013')
            self.assertEqual(result['successful_trials'],4)
            self.assertEqual(result['status'],'complete')
