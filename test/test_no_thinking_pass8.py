"""Check vote counting and answer parsing for the local Qwen pass@8 run."""

import unittest

from src.experiments.local_qwen.no_thinking_pass8 import (
    model_request,
    parse_candidate,
    vote_row,
)


def records(answers):
    return [{"candidate": answer} for answer in answers]


class NoThinkingPass8Tests(unittest.TestCase):
    def test_request_disables_thinking_and_contains_no_key(self):
        request = model_request("Find the answer.", 123, 64)
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(request["seed"], 123)
        self.assertNotIn("113", str(request))

    def test_only_complete_answers_vote(self):
        self.assertEqual(parse_candidate(" 007 ", "stop"), (7, "bare_integer"))
        self.assertEqual(parse_candidate("Answer: 113", "stop"), (113, "answer_marker"))
        self.assertEqual(parse_candidate("113", "length"), (None, "incomplete"))
        self.assertEqual(parse_candidate("I think 113", "stop"), (None, "unparseable"))

    def test_four_four_tie_is_agreement_without_unique_answer(self):
        row = vote_row(26, 113, records([113] * 4 + [7] * 4))
        self.assertTrue(row["four_plus_agreement"])
        self.assertTrue(row["four_plus_votes_for_gold"])
        self.assertTrue(row["four_four_tie"])
        self.assertIsNone(row["unique_four_plus_answer"])

    def test_unique_four_vote_leader_can_be_wrong(self):
        row = vote_row(26, 113, records([7] * 4 + [113] * 3 + [None]))
        self.assertEqual(row["unique_four_plus_answer"], 7)
        self.assertFalse(row["unique_four_plus_correct"])
        self.assertFalse(row["four_plus_votes_for_gold"])

    def test_five_votes_are_strict_majority(self):
        row = vote_row(26, 113, records([113] * 5 + [7] * 3))
        self.assertEqual(row["strict_majority_answer"], 113)
        self.assertTrue(row["unique_four_plus_correct"])


if __name__ == "__main__":
    unittest.main()
