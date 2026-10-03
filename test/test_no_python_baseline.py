"""Offline checks for matched requests and strict no-tool grading."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from src.common import OPENROUTER_URL
from src.experiments.python_tools.no_python_baseline_v1 import baseline_case, baseline_request
from src.python_tool_protocol import SYSTEM


class BaselineTests(unittest.IsolatedAsyncioTestCase):
    def reference(self):
        return {'problem_idx': 1, 'sample_idx': 2, 'problem': 'Find a number.', 'gold_answer': 7,
                'rounds': [{'request': {'model': 'qwen/qwen3.5-35b-a3b',
                    'messages': [{'role': 'system', 'content': SYSTEM+' Python instructions.'},
                                 {'role': 'user', 'content': 'Find a number.'}],
                    'tools': ['tool'], 'tool_choice': 'auto', 'seed': 2026100302,
                    'max_tokens': 16384, 'temperature': .6, 'top_p': .95, 'top_k': 20,
                    'reasoning': {'enabled': True, 'exclude': False},
                    'provider': {'order': ['parasail'], 'allow_fallbacks': False, 'require_parameters': True}}}]}

    def test_request_only_changes_tool_interface_and_suffix(self):
        reference = self.reference()
        unchanged = deepcopy(reference)
        expected = deepcopy(reference['rounds'][0]['request'])
        del expected['tools'], expected['tool_choice']
        expected['messages'][0]['content'] = SYSTEM
        self.assertEqual(baseline_request(reference), expected)
        self.assertEqual(reference, unchanged)

    async def case(self, body):
        client = AsyncMock()
        client.post.return_value = httpx.Response(200, json=body, request=httpx.Request('POST', OPENROUTER_URL))
        with patch('src.experiments.python_tools.no_python_baseline_v1.atomic_json'):
            result = await baseline_case(client, 'private-key', self.reference(),
                {'model': 'qwen/qwen3.5-35b-a3b'}, Path('unused'))
        self.assertEqual(client.post.call_args.args[0], OPENROUTER_URL)
        request = client.post.call_args.kwargs['json']
        self.assertNotIn('gold_answer', str(request))
        self.assertNotIn('private-key', str(request))
        return result

    def response(self, finish='stop', provider='Parasail', tokens=100, calls=None):
        return {'model': 'qwen/qwen3.5-35b-a3b', 'provider': provider,
                'choices': [{'finish_reason': finish, 'message': {'content': 'Answer: 7', 'tool_calls': calls}}],
                'usage': {'prompt_tokens': 20, 'completion_tokens': tokens,
                          'total_tokens': 20+tokens if tokens is not None else None}}

    async def test_complete_and_capped_candidates(self):
        complete = await self.case(self.response())
        self.assertTrue(complete['correct'])
        capped = await self.case(self.response(finish='length', tokens=16384))
        self.assertTrue(capped['answer_correct'])
        self.assertFalse(capped['correct'])
        self.assertEqual(capped['status'], 'generation_budget_exhausted')

    async def test_reject_provider_mismatch_bad_usage_and_unexpected_tools(self):
        for body in [self.response(provider='Other'), self.response(tokens=None),
                     self.response(tokens=16385), self.response(calls=[{'id': 'unexpected'}])]:
            result = await self.case(body)
            self.assertEqual(result['status'], 'error')
            self.assertFalse(result['correct'])
            self.assertEqual(result['tool_executions'], [])


if __name__ == '__main__':
    unittest.main()
