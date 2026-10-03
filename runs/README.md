# Saved run provenance

Datasets, summaries, analysis data, reports, and plots are versioned. Full model
conversations, SSE streams, progress files, and trace-bearing manifests remain
local. Each experiment directory below has a short provenance note identifying
its producers, source cohort, settings, outputs, and interpretation.

Producer paths refer to the current module layout; saved configs and statistics
are historical records and have not been regenerated. For reproduction commands,
see the [experiment guide](../docs/experiments.md).

| Run or analysis | Experiment |
| --- | --- |
| [20260930-155212](20260930-155212/README.md) | Original single-pass Qwen baseline |
| [20260930-155212/self_consistency](20260930-155212/self_consistency/README.md) | Eight-sample self-consistency expansion |
| [20260930-155212/jev_review](20260930-155212/jev_review/README.md) | Full-trace Jev continuation review |
| [20260930-155212/jev_prefix_1500](20260930-155212/jev_prefix_1500/README.md) | Jev review of 1,500-token prefixes |
| [20260930-155212/jev_calibration](20260930-155212/jev_calibration/README.md) | Retrospective Jev false-negative check |
| [20260930-155212/analysis](20260930-155212/analysis/README.md) | Reasoning length and escalation analysis |
| [20260930-155212/intermediate_answers](20260930-155212/intermediate_answers/README.md) | Intermediate-answer extraction and voting replay |
| [20260930-155212/early_verify_pass4](20260930-155212/early_verify_pass4/README.md) | Pass@4 early-verification back-test |
| [20260930-155212/local_salvage](20260930-155212/local_salvage/README.md) | Local CPU salvage experiment family |
| [20260930-155212/local_salvage/gemma-q8-pilot-tail4096](20260930-155212/local_salvage/gemma-q8-pilot-tail4096/README.md) | Spread-cohort Gemma salvage pilot |
| [20260930-155212/local_salvage/gemma-q8-explicit](20260930-155212/local_salvage/gemma-q8-explicit/README.md) | Syntax-positive Gemma salvage cohort |
| [20260930-155212/local_salvage/gemma-q8-instructions-last](20260930-155212/local_salvage/gemma-q8-instructions-last/README.md) | Gemma prompt-order comparison |
| [first-answer-20261002-paired](first-answer-20261002-paired/README.md) | Paired first-answer prompt streaming pilot |
| [chunk-extractor-gemma4b-pilot](chunk-extractor-gemma4b-pilot/README.md) | Initial hosted Gemma chunk-extraction replay |
| [chunk-scope-controls-focused](chunk-scope-controls-focused/README.md) | Focused scope-classifier development controls |
| [chunk-extractor-qwen7b-hybrid](chunk-extractor-qwen7b-hybrid/README.md) | Guarded Qwen chunk-extraction replay |
| [verify-sidecar-qwen7b](verify-sidecar-qwen7b/README.md) | Hosted candidate-verification shadow probes |
| [python-tool-qwen35-pilot](python-tool-qwen35-pilot/README.md) | Initial optional-Python paired pilot |
| [python-tool-qwen35-pilot-v2](python-tool-qwen35-pilot-v2/README.md) | Corrected Python paired pilot |
| [python-tool-qwen35-pass2-auto](python-tool-qwen35-pass2-auto/README.md) | Full optional-Python AIME pass@2 profile |
| [python-tool-qwen35-pass2-auto/early_verify_pass2](python-tool-qwen35-pass2-auto/early_verify_pass2/README.md) | Python pass@2 early-verification replay |
| [Reflex-8](qwen35-4b-no-thinking-pass8-20261003/README.md) | System 1 quick-answer probe with eight no-thinking Qwen3.5-4B samples per AIME question |
