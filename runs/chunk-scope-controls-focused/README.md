# Focused scope-classifier development controls

Gemma 3 4B and Qwen 2.5 7B were compared on two formatting examples, two toy answers, and two genuine proposals.

- **Producers:** [src.experiments.streaming.chunk_scope_controls](../../src/experiments/streaming/chunk_scope_controls.py).
- **Inputs:** Six fixed passages selected from 20260930-155212 baseline/self_consistency reasoning.
- **Artifacts:** summary.json stores the model/control records, parsed scope labels, correctness, and latency; separate job files stay local.
- **Interpretation:** Twelve hosted calls. This run has no separate config.json; model names and settings are recorded inside summary.json. Reused controls are not held-out evaluation.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
