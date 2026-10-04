"""Core-v2 back-tests retain failed trials and compare matched payloads."""
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from runner_final import backtest_v2 as driver


class ValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_retained_and_all_declared_seeds_run(self):
        seen,closed=[],[]
        class Services:
            def launch(self,*a,**kw):return object()
            async def close(self):closed.append(True)
        protocol=json.loads(driver.PROTOCOL.read_text())
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);(root/'attempts').mkdir()
            profile=root/'models/r0b0tlab/VibeThinker-3B-NVFP4/vllm-flashinfer.yaml'
            profile.parent.mkdir(parents=True)
            profile.write_text(Path(driver.__file__).with_name('vllm-flashinfer.yaml').read_text())
            original=driver.parse_args
            def parse(argv):return original(argv+['--models-dir',str(root/'models')])
            async def run(args):
                seen.append(args.seed)
                self.assertEqual(args.system_prompt_sha256,protocol['system_prompt_sha256'])
                self.assertEqual((args.rollouts,args.first_pass_max_tokens,args.max_attempts_per_question),(1,8192,4))
                out=root/'attempts'/str(args.seed);out.mkdir()
                if len(seen)==3:raise RuntimeError('failed trial')
                return out
            def score(out,seed,reference):return {'attempt_id':out.name,'sampling_seed':seed,'valid':True,'time_to_target_s':60}
            with patch.object(driver,'ROOT',root),patch.object(driver,'Services',Services),patch.object(driver,'ensure_free'),patch.object(driver,'verify_core',return_value=protocol['core_manifest_sha256']),patch.object(driver.runner,'assert_gpu_idle'),patch.object(driver,'parse_args',new=parse),patch.object(driver.runner,'run',new=run),patch.object(driver,'score',new=score),patch.object(driver,'ready',new=AsyncMock()),patch.object(driver,'reset_cache',new=AsyncMock(return_value={'success':True})),patch.object(driver.subprocess,'check_output',side_effect=['','f'*40]):
                batch=await driver.execute(SimpleNamespace(batch='test-frozen-five'))
            saved=json.loads((batch/'summary.json').read_text())
            self.assertEqual(seen,protocol['seeds'])
            self.assertEqual(len(saved['trials']),5)
            self.assertEqual(saved['trials'][2]['status'],'failed')
            self.assertEqual(saved['trials'][2]['attempt_id'],str(seen[2]))
            self.assertEqual(len(saved['paired_seed_comparisons']),4)
            self.assertEqual(saved['declared_trials'],5)
            self.assertEqual(saved['successful_trials'],4)
            self.assertEqual(closed,[True])


class ProtocolTests(unittest.TestCase):
    def test_initial_payloads_match_except_declared_prompt_change(self):
        protocol=json.loads(driver.PROTOCOL.read_text())
        seed=protocol['seeds'][0]
        reference=protocol['reference_attempts'][str(seed)]
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            (output/'config.json').write_text(json.dumps({'system_prompt':'new math prompt'}))
            for index in range(1,31):
                relative=Path('trace')/f'{index:02d}'/'rollout-01'/'request.json'
                expected=json.loads((driver.ROOT/'attempts'/reference/relative).read_text())
                expected['model']='test-model'
                expected['messages'][0]['content']='new math prompt'
                path=output/relative
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(expected))
            matched=driver.compare_initial_requests(output,'test-model',seed,reference)
            self.assertEqual(matched['initial_requests'],30)
            self.assertEqual(matched['different_questions'],[])
            altered=json.loads(path.read_text());altered['max_tokens']+=1
            path.write_text(json.dumps(altered))
            self.assertEqual(driver.compare_initial_requests(output,'test-model',seed,reference)['different_questions'],[30])

    def test_failed_and_unmet_trials_do_not_enter_timing_statistics(self):
        rows=[{'sampling_seed':1,'valid':True,'time_to_target_s':60},
              {'sampling_seed':2,'valid':False,'time_to_target_s':None},
              {'sampling_seed':3,'valid':True,'time_to_target_s':80}]
        result=driver.aggregate(rows,{'1':70,'2':65,'3':75})
        self.assertEqual((result['successful_trials'],result['declared_trials']),(2,3))
        self.assertEqual(result['median_time_to_target_s'],70)
        self.assertEqual([r['difference_s'] for r in result['paired_seed_comparisons']],[-10,5])


if __name__=='__main__':unittest.main()
