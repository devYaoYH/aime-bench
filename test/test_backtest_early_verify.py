"""Regression checks for verification queue charging, deduplication, generation retirement, and timing sensitivity.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_backtest_early_verify
Tests make no inference calls.
"""
import unittest

from src.verification_replay import simulate, candidate_schedule


def trace(q, sample=1, gold=70, duration=20, final=None, proposals=()):
    return {'problem_idx':q,'sample_number':sample,'gold_answer':gold,
            'api_latency_s':duration,'final_candidate':final,
            'candidates':[{'answer':answer,'output_fraction':t/duration,
                'kind':'prose','confidence':'asserted','quote':str(answer)} for t,answer in proposals]}


class BacktestTests(unittest.TestCase):
    def test_wrong_candidates_consume_service_time_and_generation_continues(self):
        rows=[trace(1,proposals=[(1,9),(2,70)]),trace(2,gold=80,proposals=[(2,80)])]
        result=simulate(rows,generation_slots=2)
        self.assertEqual([m['time_s'] for m in result['milestones']],[7,10])
        self.assertEqual(result['wrong_checks'],1)
        self.assertEqual([c['verification_start_s'] for c in result['checks']],[1,4,7])
        self.assertEqual(result['checks'][0]['verified_at_s'],4)

    def test_same_answer_across_samples_is_graded_once(self):
        rows=[trace(1,proposals=[(1,70),(3,70)]),trace(1,sample=2,proposals=[(2,70)])]
        result=simulate(rows,generation_slots=2)
        self.assertEqual(result['checks_completed'],1)
        self.assertEqual(result['milestones'][0]['time_s'],4)

    def test_slot_is_reclaimed_only_after_correct_verification(self):
        rows=[trace(1,proposals=[(1,9),(2,70)]),trace(2,gold=80,proposals=[(1,80)])]
        result=simulate(rows,generation_slots=1)
        self.assertEqual([r['start_s'] for r in result['generation_starts']],[0,7])
        self.assertEqual([m['time_s'] for m in result['milestones']],[7,11])

    def test_baseline_uses_final_endpoints_and_same_verifier_queue(self):
        rows=[trace(1,duration=2,final=70,proposals=[(1,70)]),trace(2,duration=3,final=80,gold=80)]
        result=simulate(rows,policy='final',generation_slots=2)
        self.assertEqual([m['time_s'] for m in result['milestones']],[5,8])
        self.assertEqual([c['arrival_s'] for c in result['checks']],[2,3])

    def test_arrival_estimate_sensitivity_keeps_final_at_natural_end(self):
        row=trace(1,duration=100,final=70,proposals=[(50,70)])
        jobs=candidate_schedule(row,'permissive',exponent=2,first_token_s=10)
        self.assertEqual(jobs[0]['offset_s'],32.5)
        self.assertEqual(jobs[-1]['offset_s'],100)

    def test_natural_end_does_not_discard_candidate_waiting_for_grader(self):
        result=simulate([trace(1,duration=2,proposals=[(2,70)])],generation_slots=1)
        self.assertEqual(result['correct_questions'],1)
        self.assertEqual(result['milestones'][0]['time_s'],5)


if __name__=='__main__': unittest.main()
