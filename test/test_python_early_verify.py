"""Regression checks for multi-round arrival estimates, independent marker extraction, and stdout candidate policy.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_python_early_verify
Tests make no inference calls.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from src.verification_replay import candidate_schedule, simulate
from src.experiments.python_tools.backtest_python_early_verify import adapt_record, literal_markers, policy_rows, stdout_values


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):
        return SimpleNamespace(ids=list(range(len(text))),offsets=[(i,i+1) for i in range(len(text))])


class PythonReplayTests(unittest.TestCase):
    def test_answer_marker_survives_prose_overlap(self):
        events=literal_markers('Answer: 113.\n\nanswer: still checking\n')
        self.assertEqual([e['answer'] for e in events],[113])
        self.assertEqual(events[0]['kind'],'answer_line')
        row={'candidates':events}
        self.assertEqual(policy_rows([row],'permissive')[0]['candidates'],[])
        self.assertEqual(len(policy_rows([row],'markers')[0]['candidates']),1)

    def test_tool_stdout_is_separate_and_prose_false_positives_abstain(self):
        events=stdout_values('Total: 113\n113\nanswer: still checking\nk=1, count=2\n', 'Find the number of ways.')
        self.assertEqual([e['answer'] for e in events],[113,113])
        self.assertTrue(all(e['tool_extension'] for e in events))
        self.assertEqual(stdout_values('Total: 1/2\n113 apples\n','Find the number.'),[])

    def test_requested_transform_and_debug_values_are_gold_blind(self):
        events=stdout_values('DP Result: 2907\nTotal: 2\n','Find the remainder when $N$ is divided by $1000$.')
        self.assertEqual([e['answer'] for e in events],[907,2])

    def test_per_round_timing_keeps_tool_execution_before_output(self):
        record={'problem_idx':1,'sample_idx':1,'problem':'Find the number of ways.',
                'started_at_utc':'2026-10-02T00:00:00.000+00:00','elapsed_s':25,
                'candidate':None,'correct':False,'gold_answer':113,'status':'generation_budget_exhausted',
                'rounds':[{'number':1,'started_at_utc':'2026-10-02T00:00:00.000+00:00','latency_s':10,
                           'response':{'choices':[{'message':{'reasoning':'Answer: 9\n','content':None,'tool_calls':[]}}],
                                       'usage':{'completion_tokens':10}}},
                          {'number':2,'started_at_utc':'2026-10-02T00:00:15.000+00:00','latency_s':10,
                           'response':{'choices':[{'message':{'reasoning':'Answer: 113\n','content':None}}],
                                       'usage':{'completion_tokens':12}}}],
                'tool_executions':[{'round_number':1,'generated_tokens_at_arrival':10,
                                    'result':{'ok':True,'stdout':'Total: 113\n'}}]}
        row=adapt_record(record,CharacterTokenizer(),'test.json')
        original=deepcopy(row)
        same=policy_rows([row],'permissive')[0]
        self.assertFalse(any(e['kind']=='tool_result' for e in same['candidates']))
        tool=next(e for e in row['candidates'] if e['part']=='tool_stdout')
        self.assertEqual(tool['offset_s'],15)
        schedule=candidate_schedule(row,'permissive_tools',exponent=2,first_token_s=3)
        tool_job=next(e for e in schedule if e['part']=='tool_stdout')
        self.assertEqual(tool_job['offset_s'],15)
        self.assertTrue(all(e['offset_s']>=15 for e in schedule if e.get('round_number')==2))
        self.assertEqual(row,original)
        result=simulate([row],policy='permissive_tools',generation_slots=1)
        self.assertEqual(result['wrong_checks'],1)
        self.assertEqual(result['milestones'][0]['time_s'],18)


if __name__=='__main__': unittest.main()
