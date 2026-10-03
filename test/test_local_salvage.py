"""Regression checks for quote grounding, key-free prompts, and capped/control cohort selection from saved traces.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_local_salvage
Tests make no inference calls. The salvage cohort checks require ignored local baseline records.
"""
import unittest
from src.experiments.local_salvage.local_salvage import ROOT, extract_prompt, grounded_answer, load_cases, verify_prompt


class SafeguardTests(unittest.TestCase):
    def test_grounding_rejects_fabricated_quotes_and_substring_numbers(self):
        self.assertEqual(grounded_answer({'answer': 12, 'evidence': 'Answer: 12'}, 'Answer: 112'), (None, False))
        self.assertEqual(grounded_answer({'answer': 12, 'evidence': 'Answer: 112'}, 'Answer: 112'), (None, False))
        self.assertEqual(grounded_answer({'answer': 12, 'evidence': ''}, 'Answer: 12'), (None, False))
        self.assertEqual(grounded_answer({'answer': 12, 'evidence': 'Answer: 012'}, 'Answer: 012'), (12, True))
        self.assertEqual(grounded_answer({'answer': None, 'evidence': ''}, 'work'), (None, None))

    def test_prompts_contain_only_problem_trace_and_candidate(self):
        problem, trace = 'SENTINEL_PROBLEM', 'SENTINEL_TRACE'
        for text in [extract_prompt(problem, trace), verify_prompt(problem, trace, 42)]:
            self.assertIn(problem, text)
            self.assertIn(trace, text)
            for forbidden in ['gold_answer', 'correct_control', 'wrong_control', 'generation_latency_s']:
                self.assertNotIn(forbidden, text)

    def test_selection_and_control_labels(self):
        capped, correct, wrong, count = load_cases(ROOT / 'runs/20260930-155212', 8, 5)
        self.assertEqual(count, 240)
        self.assertEqual(len(capped), 8)
        self.assertEqual(len({c['problem_idx'] for c in capped}), 8)
        self.assertTrue(all(c['finish_reason'] == 'length' for c in capped))
        self.assertEqual(len(correct), 5)
        self.assertEqual(len(wrong), 5)
        self.assertTrue(all(int(c['original_candidate']) == c['gold'] for c in correct))
        self.assertTrue(all(int(c['original_candidate']) != c['gold'] for c in wrong))

    def test_explicit_cohort_keeps_incorrect_candidate(self):
        capped, _, _, _ = load_cases(ROOT / 'runs/20260930-155212', 8, 0, 'explicit')
        from src.common import extract_answer
        self.assertEqual(len(capped), 8)
        self.assertEqual(sum(int(extract_answer(c['trace'])) != c['gold'] for c in capped), 1)


if __name__ == '__main__':
    unittest.main()
