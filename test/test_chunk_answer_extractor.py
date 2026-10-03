"""Regression checks for chunk-safe candidate windows, source-span grounding, scope controls, and causal transforms.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_chunk_answer_extractor
Tests make no inference calls.
"""
import unittest
from src.chunk_answer_extractor import WindowBuilder, grounded_decision, prompt


class ChunkExtractorTests(unittest.TestCase):
    def test_number_across_chunks_requires_completed_line(self):
        w = WindowBuilder('Find the sum.')
        self.assertIsNone(w.feed('reasoning', 'Answer: 0', 1))
        self.assertIsNone(w.feed('reasoning', '7', 2))
        self.assertIsNone(w.feed('reasoning', '0.', 3))
        job = w.feed('reasoning', '\n', 4)
        self.assertEqual([c['answer'] for c in job['candidates']], [70])
        self.assertEqual(job['arrival_s'], 4)
        self.assertIsNone(w.feed('reasoning', 'More checking.\n', 5))

    def test_source_span_selection_cannot_invent_answer_or_evidence(self):
        w = WindowBuilder('Find the sum.')
        job = w.feed('reasoning', 'The answer is 70.\n', 1)
        cid = job['candidates'][0]['id']
        self.assertEqual(grounded_decision({'candidate_id': cid, 'scope': 'requested_answer'}, job)['answer'], 70)
        with self.assertRaises(ValueError):
            grounded_decision({'candidate_id': 999, 'scope': 'requested_answer'}, job)
        with self.assertRaises(ValueError):
            grounded_decision({'candidate_id': cid, 'scope': 'format_or_quote'}, job)
        self.assertIsNone(grounded_decision({'candidate_id': None, 'scope': 'toy_example'}, job))

    def test_format_examples_remain_available_for_scope_rejection(self):
        w = WindowBuilder('Find the sum.')
        job = w.feed('reasoning', 'For example, if the answer is 16, then Answer: 016.\n', 1)
        self.assertTrue(job['candidates'])
        self.assertIn('For example', prompt(w.problem, job))

    def test_requested_transform_and_causal_context(self):
        w = WindowBuilder('Find the remainder when $N$ is divided by $1000$.')
        job = w.feed('reasoning', 'So N = 2907.\n', 1)
        self.assertEqual(job['candidates'][0]['answer'], 907)
        self.assertEqual(job['candidates'][0]['requested_transform'], 'mod1000')
        w.feed('reasoning', 'But I revise the answer to 123.\n', 2)
        self.assertNotIn('123', job['window'])


if __name__ == '__main__':
    unittest.main()
