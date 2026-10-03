# Initial hosted Gemma chunk-extraction replay

Gemma 3 4B span-selection/classification on causal candidate windows, with six scope controls and a timed two-slot replay.

- **Producers:** [src.experiments.streaming.chunk_answer_extractor](../../src/experiments/streaming/chunk_answer_extractor.py).
- **Inputs:** first-answer-20261002-paired SSE trajectories and six known passages from the original Qwen cohort.
- **Artifacts:** config.json and summary.json; per-job responses and the trace-bearing manifest stay local.
- **Interpretation:** Measured hosted classifier latency. Development controls revealed scope false positives; grounded span selection alone does not ensure correct scope.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
