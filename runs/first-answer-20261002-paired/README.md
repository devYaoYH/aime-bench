# Paired first-answer prompt streaming pilot

Original versus first-answer stopping prompts on Q1, Q3, Q18, and Q25: eight simultaneous Qwen requests, pinned to DeepInfra.

- **Producers:** [src.experiments.streaming.first_answer_pilot](../../src/experiments/streaming/first_answer_pilot.py), [src.experiments.streaming.report_first_answer_pilot](../../src/experiments/streaming/report_first_answer_pilot.py), [src.experiments.streaming.audit_stream_markers](../../src/experiments/streaming/audit_stream_markers.py).
- **Inputs:** Problem/request settings from 20260930-155212; two prompt arms with the original sampling and 16k output cap.
- **Artifacts:** config.json, summary.json, delivery_rates.json, cancellation_replay.json, guarded_marker_replay.json, and plots. Raw SSE/question/progress files stay local.
- **Interpretation:** Measured streaming arrivals and natural endpoints. Cancellation replay is hypothetical; no answer-triggered cancellation was performed.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
