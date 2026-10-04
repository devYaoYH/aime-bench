"""Versioned dynamic budget arrays remain reproducible and schema-valid."""
from pathlib import Path
import tempfile
import unittest
from src.attempt_metadata import build_metadata, validate_metadata


class DynamicMetadataTests(unittest.TestCase):
    def metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            return build_metadata(Path(tmp)/'attempt', {
                'model':'r0b0tlab/VibeThinker-3B-NVFP4', 'runner_id':'speedrun_v4',
                'strategy':'speedrun_v4', 'max_tokens':65536, 'first_pass_max_tokens':8192,
                'max_attempts_per_question':8, 'max_rounds':1, 'no_continuation':False,
                'token_budgets':[8192,16384,32768,65536],
                'max_concurrent_requests':60,'seed_stride':4,'budget_mode':'continuation'})

    def test_budget_array_and_admission_controls_roundtrip(self):
        m=self.metadata();validate_metadata(m,'attempt')
        hp=m['controls']['hyperparameters']
        self.assertEqual(hp['token_budgets'],[8192,16384,32768,65536])
        self.assertEqual(hp['max_concurrent_requests'],60)
        self.assertEqual(hp['max_attempts_per_question'],8)
        self.assertTrue(hp['continuation_enabled'])

    def test_nested_and_object_hyperparameters_remain_invalid(self):
        for invalid in ([[8192]], [{'value':8192}], {'value':8192}):
            m=self.metadata();m['controls']['hyperparameters']['token_budgets']=invalid
            with self.assertRaisesRegex(ValueError,'Metadata schema'):
                validate_metadata(m,'attempt')

    def test_final_warmup_controls_are_schema_valid_and_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=build_metadata(Path(tmp)/'attempt', {
                'runner_id':'runner_final_v1','strategy':'coverage',
                'skip_benchmark_prewarm':False,'prewarm_max_tokens':8192,
                'first_pass_max_tokens':8192,'max_attempts_per_question':4})
        validate_metadata(m,'attempt')
        self.assertEqual(m['controls']['hyperparameters']['prewarm_max_tokens'],8192)
        self.assertFalse(m['controls']['hyperparameters']['skip_benchmark_prewarm'])
