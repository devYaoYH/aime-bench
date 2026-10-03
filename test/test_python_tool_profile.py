"""Regression checks for mocked multi-round optional-tool execution, budget accounting, and pass@2 grading.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_python_tool_profile
Tests make no inference calls.
"""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from src.experiments.python_tools.python_tool_profile import profile_case, summarize


def response(message, tokens=10, finish='stop'):
    body={'choices':[{'message':message,'finish_reason':finish}],
          'usage':{'prompt_tokens':20,'completion_tokens':tokens,'total_tokens':20+tokens,'cost':.001}}
    return httpx.Response(200,json=body,request=httpx.Request('POST','https://example.invalid'))


class ProfileTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        return {'model':'test','provider':'test','system':'solve','sampling':{'reasoning':{'enabled':True}},
                'total_generation_budget':100,'max_rounds':16,'max_tool_calls':16,'seed_base':100,
                'questions':[1],'samples':2}

    async def test_multi_step_auto_history_budget_and_key_exclusion(self):
        replies=[]
        for i in (1,2):
            replies.append(response({'role':'assistant','content':None,'reasoning':'working',
                'tool_calls':[{'id':f'c{i}','type':'function','function':{'name':'python_math',
                'arguments':json.dumps({'code':f'print({i})'})}}]},finish='tool_calls'))
        replies.append(response({'role':'assistant','content':'Answer: 7'}))
        client=AsyncMock()
        client.post.side_effect=replies
        worker=AsyncMock(side_effect=[{'ok':False,'stdout':'','error':'bad','wall_s':.1},
                                     {'ok':True,'stdout':'2','error':None,'wall_s':.1}])
        with patch('src.experiments.python_tools.python_tool_profile.atomic_json'),patch('src.experiments.python_tools.python_tool_profile.execute_python',worker):
            result=await profile_case(client,'secret',{'problem_idx':1,'problem':'test question','gold_answer':7},
                                      1,self.config(),Path('unused'),asyncio.Semaphore(1))
        self.assertTrue(result['correct'])
        self.assertEqual(result['accounting']['completion_tokens'],30)
        requests=[c.kwargs['json'] for c in client.post.call_args_list]
        self.assertEqual([r['max_tokens'] for r in requests],[100,90,80])
        self.assertTrue(all(r['tool_choice']=='auto' and r['tools'] for r in requests))
        self.assertTrue(all('gold_answer' not in json.dumps(r) and 'secret' not in json.dumps(r) for r in requests))
        self.assertEqual([m['tool_call_id'] for m in requests[-1]['messages'] if m['role']=='tool'],['c1','c2'])
        self.assertEqual(requests[-1]['messages'][2]['reasoning'],'working')
        self.assertEqual(result['accounting']['successful_tool_calls'],1)

    async def test_capped_candidate_not_counted_as_completed_solve(self):
        client=AsyncMock()
        client.post.return_value=response({'role':'assistant','content':'Answer: 7'},tokens=100,finish='length')
        with patch('src.experiments.python_tools.python_tool_profile.atomic_json'):
            result=await profile_case(client,'secret',{'problem_idx':1,'problem':'test','gold_answer':7},
                                      1,self.config(),Path('unused'),asyncio.Semaphore(1))
        self.assertEqual(result['status'],'generation_budget_exhausted')
        self.assertTrue(result['answer_correct'])
        self.assertFalse(result['correct'])
        second=deepcopy(result)
        second['sample_idx']=2
        second['status']='complete'
        second['correct']=True
        summary=summarize([result,second],self.config(),1)
        self.assertTrue(summary['all_attempts_observed'])
        self.assertEqual(summary['questions_correct_pass_at_2'],1)
        self.assertEqual(summary['attempts_correct'],1)


if __name__=='__main__': unittest.main()
