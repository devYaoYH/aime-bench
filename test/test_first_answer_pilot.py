"""Regression checks for SSE parsing, streamed reasoning/content accumulation, usage updates, and stream failures.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_first_answer_pilot
Tests make no inference calls.
"""
import unittest

from src.experiments.streaming.first_answer_pilot import Accumulator, sse_payloads, annotate


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_comments_multiline_done_and_eof(self):
        async def lines():
            for line in [': heartbeat', '', 'data: {"a":', 'data: 1}', '', 'data: [DONE]']:
                yield line
        self.assertEqual([s async for s in sse_payloads(lines())], ['{"a":\n1}', '[DONE]'])

    def test_reasoning_content_and_repeated_usage_finish(self):
        a = Accumulator()
        a.add({'id': 'x', 'provider': 'DeepInfra', 'choices': [{'delta': {'reasoning': 'Answer is 70.'}}]}, 1.0)
        a.add({'choices': [{'delta': {'content': 'Answer: 070'}, 'finish_reason': 'stop'}]}, 2.0)
        a.add({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 12}}, 2.1)
        self.assertEqual(a.response()['choices'][0]['message']['reasoning'], 'Answer is 70.')
        self.assertEqual(a.response()['choices'][0]['message']['content'], 'Answer: 070')
        self.assertEqual(a.first_output_s, 1.0)
        self.assertEqual(a.first_content_s, 2.0)
        self.assertEqual(a.finish_s, 2.0)
        self.assertEqual(len(a.deliveries), 2)
        self.assertEqual(a.usage['completion_tokens'], 12)

    def test_reasoning_details_fallback_and_stream_errors(self):
        a = Accumulator()
        a.add({'choices': [{'delta': {'reasoning': 'once', 'reasoning_details': [{'type': 'reasoning.text', 'text': 'once'}]}}]}, 1)
        a.add({'choices': [{'delta': {'reasoning_details': [{'type': 'reasoning.text', 'text': ' twice'}]}}]}, 2)
        a.add({'error': {'message': 'provider disconnected'}, 'choices': [{'delta': {}, 'finish_reason': 'error'}]}, 3)
        self.assertEqual(a.response()['choices'][0]['message']['reasoning'], 'once twice')
        self.assertEqual(a.finish_reason, 'error')
        self.assertEqual(len(a.errors), 1)


if __name__ == '__main__':
    unittest.main()
