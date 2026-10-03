# Pass@4 early-verification back-test

Final, marker, and permissive candidates from samples 1-4 were replayed with one shared verifier, three seconds per check, and 120/30/8 generation slots.

- **Producers:** [src.experiments.intermediate_answers.backtest_early_verify](../../../src/experiments/intermediate_answers/backtest_early_verify.py).
- **Inputs:** The first four samples of the original Qwen cohort and cached Qwen tokenizer; candidates are rebuilt from their source records.
- **Artifacts:** summary.json, candidate_inventory.json, milestones.csv, per_question.csv, and time_to_correct plots.
- **Interpretation:** Offline counterfactual scheduling. Saved keys stand in for perfect verification, and intermediate delivery is estimated from token fractions.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
