"""Check complementary cohort selection, key-free requests and observed-outcome counts."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.experiments.jev import complete_calibration as followup


class JevFollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)/'run'
        (self.run/'jev_calibration').mkdir(parents=True)
        self.tokenizer = Path(self.temp.name)/'tokenizer.json'
        self.tokenizer.write_text('synthetic tokenizer')
        self.config = {'model':'test/jev','endpoint':'https://example.invalid',
                       'prefix_tokens':1500,'continuation_budget_tokens':8192,
                       'tokenizer_sha256':hashlib.sha256(self.tokenizer.read_bytes()).hexdigest(),
                       'question':{'type':'noul','instructions':'same question','criteria':{}}}
        self.original = {'model':'test/jev','prefix_tokens':1500,'continuation_budget_tokens':8192,'rows':[]}
        for n in range(240):
            index,sample = n//8+1,n%8+1
            path = self.run/(f'questions/{index:02d}.json' if sample==1 else f'self_consistency/questions/{index:02d}/{sample:02d}.json')
            path.parent.mkdir(parents=True,exist_ok=True)
            correct = n<120
            trace = {'problem_idx':index,'sample_number':sample,'problem':'PROBLEM_SENTINEL',
                     'candidate':'42' if n<125 else None,'gold_answer':'GOLD_SENTINEL',
                     'correct':correct,'finish_reason':'stop' if n<125 else 'length',
                     'usage':{'completion_tokens':5000 if n<125 else 16384},
                     'response':{'choices':[{'message':{'reasoning':'PREFIX_SENTINEL OUTCOME_SENTINEL'}}]}}
            path.write_text(json.dumps(trace))
            if n<125:
                self.original['rows'].append({'problem_idx':index,'sample_number':sample,
                    'probability':.8,'correct':correct,'finish_reason':'stop','candidate':'42',
                    'completion_tokens':5000,'within_8192_more_tokens':True})
        (self.run/'jev_calibration/config.json').write_text(json.dumps(self.config))
        (self.run/'jev_calibration/summary.json').write_text(json.dumps(self.original))

    def prepare(self):
        with patch.object(followup.Tokenizer,'from_file'), patch.object(followup,'exact_prefix',return_value=('PREFIX_SENTINEL',20000,None)):
            return followup.prepare(self.run,self.tokenizer)

    def test_complementary_115_payloads_exclude_outcomes_and_keys(self):
        _,original,items = self.prepare()
        self.assertEqual(len(items),115)
        self.assertEqual(original,self.original)
        for item in items:
            request=json.dumps(item['request'])
            self.assertIn('PROBLEM_SENTINEL',request)
            self.assertIn('PREFIX_SENTINEL',request)
            for hidden in ('GOLD_SENTINEL','OUTCOME_SENTINEL','finish_reason','correct','candidate'):
                self.assertNotIn(hidden,request)

    def test_tokenizer_drift_blocks_calls(self):
        self.tokenizer.write_text('different tokenizer')
        with self.assertRaisesRegex(ValueError,'Tokenizer differs'):
            self.prepare()

    def test_all_outcomes_counted_without_relabeling_caps_as_unsalvageable(self):
        _,original,items = self.prepare()
        records=[dict(item,probability=.8) for item in items]
        summary=followup.summarize(original,records,self.run)
        at_half=next(t for t in summary['thresholds'] if t['retain_at_or_above']==.5)
        self.assertEqual(at_half['retained'],240)
        self.assertEqual(at_half['correct_final_retained'],120)
        self.assertEqual(at_half['observed_failure_retained'],120)
        self.assertEqual(at_half['capped_retained'],115)
        self.assertEqual(at_half['completed_wrong_retained'],5)
        self.assertEqual(at_half['observed_final_precision'],.5)
        self.assertEqual(original,self.original)
        path=self.run/items[0]['source_trace_file']
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Source changed'):
            followup.summarize(original,records,self.run)

    def test_plan_never_loads_credentials(self):
        with patch.object(followup,'prepare',return_value=(self.config,self.original,[{}]*115)), patch.object(followup,'load_key',side_effect=AssertionError('paid call')):
            asyncio.run(followup.execute(SimpleNamespace(run='run',tokenizer_json=self.tokenizer,execute=False)))


if __name__=='__main__':
    unittest.main()
