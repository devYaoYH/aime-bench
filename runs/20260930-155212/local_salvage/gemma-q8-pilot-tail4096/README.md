# Spread-cohort Gemma salvage pilot

Eight capped questions, two full-trace comparisons, and five controls per verification class; 4,096-token tails, two CPU slots.

- **Producers:** [src.experiments.local_salvage.local_salvage](../../../../src/experiments/local_salvage/local_salvage.py).
- **Inputs:** Parent Qwen traces selected by the spread cohort and the pinned Gemma 3 1B Q8 loopback server.
- **Artifacts:** config.json and summary.json; extraction/verification request and response records stay local.
- **Interpretation:** 25 local model requests. Fallible model verdicts are diagnostic; quote grounding is checked separately.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
