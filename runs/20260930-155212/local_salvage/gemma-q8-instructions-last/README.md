# Gemma prompt-order comparison

Explicit-answer cohort with extraction instructions and verification candidates placed last; eight capped tails plus verification controls, no full-trace comparisons.

- **Producers:** [src.experiments.local_salvage.local_salvage](../../../../src/experiments/local_salvage/local_salvage.py).
- **Inputs:** Parent Qwen traces; --cohort explicit --capped 8 --full 0 --controls 5 --candidate-last --extractor-last.
- **Artifacts:** config.json and summary.json; individual records stay local.
- **Interpretation:** 23 local model requests. Prompt-order sensitivity experiment, not evidence of reliable mathematical verification.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
