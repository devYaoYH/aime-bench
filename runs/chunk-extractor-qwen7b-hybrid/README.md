# Guarded Qwen chunk-extraction replay

Focused Qwen 2.5 7B scope classification plus deterministic formatting guards, replayed with two shared extraction slots.

- **Producers:** [src.experiments.streaming.chunk_answer_extractor](../../src/experiments/streaming/chunk_answer_extractor.py).
- **Inputs:** first-answer-20261002-paired SSE trajectories and the same six development controls.
- **Artifacts:** config.json and summary.json; request/response files and manifest stay local.
- **Interpretation:** --focused --scope-guard --timed with qwen/qwen-2.5-7b-instruct. Measures extraction and queueing; ideal verifier savings are hypothetical.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
