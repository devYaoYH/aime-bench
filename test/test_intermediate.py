"""Regression checks for answer-clause extraction, toy/negated scope, exact arithmetic, and vote tie handling.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_intermediate
Tests make no inference calls.
"""
import unittest
from src.answer_extraction import extract_events, eligible, evaluate_arithmetic, rank_statistics


class ExtractionTests(unittest.TestCase):
    def candidates(self, text, policy='proposal', problem=''):
        return [e['answer'] for e in extract_events(text, problem) if eligible(e, policy)]

    def test_whole_expression_not_first_operand(self):
        self.assertEqual(self.candidates('Therefore, the answer is 21 + 49 = 70. But let me check.'), [70])
        self.assertEqual(self.candidates('The answer is 2907 mod 1000 = 907.'), [907])
        self.assertEqual(self.candidates('The answer is 1/3.'), [])
        self.assertEqual(self.candidates('The answer is 70 or 71.'), [])
        self.assertEqual(self.candidates('The answer is 7.5.'), [])
        self.assertEqual(self.candidates('The answer is 070.'), [70])
        self.assertEqual(self.candidates('The answer is 2*27³.'), [])
        self.assertEqual(self.candidates('The answer is 145, or 129.'), [])

    def test_toy_sections_do_not_create_false_revisions(self):
        text = ('The answer is 113. Let me check with smaller cases. '
                'Take a square.\n' + 'Checking details.\n' * 30 +
                'The answer would be 3. Therefore, returning to the original problem, the answer is 113.')
        spans = [{'start': text.index('Let me check'), 'end': text.index('Therefore, returning')}]
        events = extract_events(text, excluded_spans=spans)
        self.assertEqual([e['answer'] for e in events if eligible(e, 'proposal')], [113, 113])
        self.assertEqual([e['answer'] for e in events if eligible(e, 'literal_prose')], [113, 3, 113])
        self.assertEqual(self.candidates('Let me verify with small cases.\nAnswer: 3\n', 'markers'), [3])
        self.assertEqual(self.candidates('I recall that in some similar problems, the answer is 68.'), [])

    def test_hypotheticals_negation_and_tentative_proposals(self):
        self.assertEqual(self.candidates('For example, if the answer is 16, write 016.'), [])
        self.assertEqual(self.candidates('The answer is not 16.'), [])
        self.assertEqual(self.candidates('I think the answer is 70.'), [70])
        self.assertEqual(self.candidates('I think the answer is 70.', 'committed'), [])
        self.assertEqual(self.candidates('So the answer would be 16?'), [16])

    def test_marker_duplicates_and_order(self):
        self.assertEqual(self.candidates('The answer is \\boxed{70}.\nAnswer: 071\n', 'markers'), [70, 71])
        self.assertEqual(self.candidates('Answer: 70\nBut I revise it. Answer is 71.'), [70, 71])

    def test_requested_quantity_transforms(self):
        p = 'Find the remainder when $N$ is divided by $1000$.'
        self.assertEqual(self.candidates('So N = 2016.', 'target', p), [16])
        p = 'Find $m+n$.'
        self.assertEqual(self.candidates('m + n = 40 + 231 = 271.', 'target', p), [271])
        p = 'Find the difference between $N$ and 2025.'
        self.assertEqual(self.candidates('N = 2304.', 'target', p), [279])

    def test_safe_arithmetic_and_ties(self):
        self.assertIsNone(evaluate_arithmetic('__import__("os").system("false")'))
        self.assertIsNone(evaluate_arithmetic('2**1000000'))
        self.assertEqual(rank_statistics([10, 20, 30], 20, 2)['expected'], 2/3)
        self.assertFalse(rank_statistics([10, 20, 30], 20, 2)['guaranteed'])
        self.assertEqual(rank_statistics([10, 10, 20, None], 20, 2)['expected'], 1)


if __name__ == '__main__':
    unittest.main()
