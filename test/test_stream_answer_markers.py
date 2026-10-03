"""Regression checks for split markers/numbers, genuine EOF, channel independence, and formatting false positives.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_stream_answer_markers
Tests make no inference calls.
"""
import unittest
from src.stream_answer_markers import FinalAnswerDetector


class MarkerTests(unittest.TestCase):
    def test_split_digits_and_marker_do_not_stop_early(self):
        d = FinalAnswerDetector()
        for chunk in ['**Final Ans', 'wer**\nAn', 'swer: 0', '7', '0']:
            self.assertEqual(d.feed('reasoning', chunk), [])
        self.assertEqual(d.feed('reasoning', '\n')[0]['answer'], 70)

    def test_closed_box_and_standalone_math_block(self):
        d = FinalAnswerDetector()
        self.assertEqual(d.feed('reasoning', '### Final Answer\n$$\n\\boxed{08'), [])
        self.assertEqual(d.feed('reasoning', '2}'), [])
        self.assertEqual(d.feed('reasoning', '\n$$\n')[0]['answer'], 82)

    def test_format_examples_and_embedded_quotes_abstain(self):
        examples = [
            'For example, if the answer is 16, then Answer: 016.\n',
            'The exact format is \\boxed{016} or \\boxed{16}.\n',
            'Therefore, Answer: 070? Wait, let me check.\n',
            'For example:\nAnswer: 016\n',
            'For example:\nFinal Answer\nAnswer: 016\n',
            'The instructions use this format:\nFinal Answer\n\\boxed{016}\n',
            '```text\nFinal Answer\nAnswer: 016\n```\n',
            'Final Answer\n"Answer: 016"\n',
            'Final Answer\nAnswer: 016 or 017\n',
            'Final Answer\nThe instructions say Answer: 016.\n',
            'Final Answer\nAnswer: 0164\n',
            'Final Answer\nAnswer: 016?\n',
        ]
        for text in examples:
            d = FinalAnswerDetector()
            self.assertEqual(d.feed('reasoning', text), [], text)
            self.assertEqual(d.finish(), [], text)

    def test_part_independence_and_true_eof(self):
        d = FinalAnswerDetector()
        d.feed('reasoning', 'Final Answer\n')
        self.assertEqual(d.feed('content', 'Answer: 070\n'), [])
        self.assertEqual(d.feed('reasoning', 'Answer: 070'), [])
        self.assertEqual(d.finish(1.5)[0]['answer'], 70)


if __name__ == '__main__':
    unittest.main()
