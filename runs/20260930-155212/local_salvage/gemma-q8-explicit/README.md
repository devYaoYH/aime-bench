# Syntax-positive Gemma salvage cohort

Eight capped traces with existing explicit answer syntax, plus two full-trace comparisons and no verification controls.

- **Producers:** [src.experiments.local_salvage.local_salvage](../../../../src/experiments/local_salvage/local_salvage.py).
- **Inputs:** Parent Qwen traces selected by --cohort explicit, --capped 8, --full 2, --controls 0, using 4,096-token tails.
- **Artifacts:** config.json and summary.json; individual records stay local.
- **Interpretation:** Ten local model requests. Selection uses existing answer syntax rather than answer-key correctness.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
