# Script guide

Every Python module has a file-level docstring describing its purpose, when to
use it, inputs, outputs, and prerequisites. Run commands from the repository
root with `.venv/bin/python -m <module>`. Full examples are in the
[experiment guide](experiments.md); historical producers and cohorts are in
[run provenance](../runs/README.md).

## Common utilities and libraries

These stay directly in `src/`. Libraries are imported by experiments; only the
dataset fetcher and viewer are user-facing utility commands.

| File | When to use |
| --- | --- |
| [common.py](../src/common.py) | Repository paths, credentials, dataset loading, JSON writes, UTC timestamps, and final-answer grading |
| [answer_extraction.py](../src/answer_extraction.py) | Literal candidate extraction, requested-quantity transforms, source annotations, and voting/trace analysis |
| [chunk_answer_extractor.py](../src/chunk_answer_extractor.py) | Build completed-line causal windows and validate classifier-selected source spans |
| [stream_answer_markers.py](../src/stream_answer_markers.py) | Conservative chunk-safe final-answer markers and OpenRouter delta normalization |
| [verify_sidecar.py](../src/verify_sidecar.py) | Run bounded asynchronous candidate verification while the original stream continues |
| [verification_replay.py](../src/verification_replay.py) | Simulate charged verification queues and generation-slot retirement across experiment families |
| [python_math_tool.py](../src/python_math_tool.py) | Execute restricted arithmetic in an isolated macOS worker |
| [python_tool_protocol.py](../src/python_tool_protocol.py) | Preserve native tool-call history, validate arguments, and total multi-round usage |
| [tokenizer_utils.py](../src/tokenizer_utils.py) | Recover exact source-preserving prefixes with a caller-supplied tokenizer |
| [fetch_dataset.py](../src/fetch_dataset.py) | Refresh MathArena problems, official keys, revision, and dataset checksum; requires network |
| [viewer_server.py](../src/viewer_server.py) | Inspect locally saved baseline responses, votes, and Jev judgments in the browser |

## Baseline experiments

Module prefix: `src.experiments.baseline`.

| Script | When to use | Mode |
| --- | --- | --- |
| [benchmark.py](../src/experiments/baseline/benchmark.py) | Start a fresh one-sample-per-question Qwen benchmark | Paid OpenRouter inference |
| [expand_samples.py](../src/experiments/baseline/expand_samples.py) | Add samples 2–8 to an existing baseline and compute votes | Paid OpenRouter inference |
| [analyze_escalation.py](../src/experiments/baseline/analyze_escalation.py) | Measure reasoning-token lengths, same-model checkpoints, and call budgets | Saved-trace analysis; tokenizer may download |

## Jev experiments

Module prefix: `src.experiments.jev`.

| Script | When to use | Mode |
| --- | --- | --- |
| [judge_trajectories.py](../src/experiments/jev/judge_trajectories.py) | Rank the original capped trajectories using full-trace continuation decisions | Paid Jev judging |
| [judge_prefix.py](../src/experiments/jev/judge_prefix.py) | Score exact 1,500-token prefixes of unfinished original trajectories | Paid Jev judging |
| [calibrate_jev.py](../src/experiments/jev/calibrate_jev.py) | Compare prefix scores against completed trajectories and continuation budgets | Paid Jev judging, retrospective labeling |

## Local salvage experiments

Module prefix: `src.experiments.local_salvage`.

| Script | When to use | Mode |
| --- | --- | --- |
| [local_gemma.py](../src/experiments/local_salvage/local_gemma.py) | Install the pinned Apple Silicon llama.cpp/Gemma artifacts, then serve on loopback | Network setup; local CPU serving |
| [local_salvage.py](../src/experiments/local_salvage/local_salvage.py) | Compare tail/full extraction and verification controls from capped Qwen traces | Local CPU inference; regex baseline is offline |

## Intermediate-answer experiments

Module prefix: `src.experiments.intermediate_answers`.

| Script | When to use | Mode |
| --- | --- | --- |
| [analyze_intermediate.py](../src/experiments/intermediate_answers/analyze_intermediate.py) | Analyze the fixed 240-trace Qwen cohort for proposals, voting, and token budgets | Offline; raw traces and cached tokenizer |
| [render_intermediate.py](../src/experiments/intermediate_answers/render_intermediate.py) | Rebuild plots and the report from derived intermediate-answer records | Offline rendering |
| [backtest_early_verify.py](../src/experiments/intermediate_answers/backtest_early_verify.py) | Compare early checking on original samples 1–4 under 120/30/8-slot schedules | Offline perfect-verifier replay |

## Streaming experiments

Module prefix: `src.experiments.streaming`.

| Script | When to use | Mode |
| --- | --- | --- |
| [first_answer_pilot.py](../src/experiments/streaming/first_answer_pilot.py) | Run paired original/first-answer prompts on Q1, Q3, Q18, and Q25 | Eight paid streamed solver calls |
| [report_first_answer_pilot.py](../src/experiments/streaming/report_first_answer_pilot.py) | Rebuild prompt-pilot plots, delivery rates, and hypothetical cancellation metrics | Offline rendering |
| [audit_stream_markers.py](../src/experiments/streaming/audit_stream_markers.py) | Audit guarded markers on the paired streams and historical 240 traces | Offline raw-stream/trace replay |
| [chunk_answer_extractor.py](../src/experiments/streaming/chunk_answer_extractor.py) | Compare hosted extraction prompts and guards on saved causal windows | Paid small-model calls; prepare-only is offline |
| [chunk_scope_controls.py](../src/experiments/streaming/chunk_scope_controls.py) | Compare Gemma/Qwen scope classification on six development controls | Paid small-model calls |
| [verify_sidecar.py](../src/experiments/streaming/verify_sidecar.py) | Measure verification latency and false approvals on saved causal-prefix probes | Paid shadow probes; prepare-only is offline |

The top-level chunk extractor and sidecar are reusable libraries. Their similarly
named files inside `experiments/streaming/` load fixed fixtures and run hosted
experiments. Candidate extraction does not establish mathematical correctness.

## Python-tool experiments

Module prefix: `src.experiments.python_tools`.

| Script | When to use | Mode |
| --- | --- | --- |
| [python_tool_pilot.py](../src/experiments/python_tools/python_tool_pilot.py) | Compare Python-enabled/no-tool Qwen3.5 arms on Q23/Q25 | Paid inference and macOS worker |
| [python_tool_profile.py](../src/experiments/python_tools/python_tool_profile.py) | Run two optional-tool attempts on each of the 30 questions | Paid inference and macOS worker |
| [report_python_tool_profile.py](../src/experiments/python_tools/report_python_tool_profile.py) | Summarize measured tokens, latency, tool errors, and final-answer arrivals | Offline raw-profile rendering |
| [backtest_python_early_verify.py](../src/experiments/python_tools/backtest_python_early_verify.py) | Replay early candidates across multi-round text and measured tool timing | Offline perfect-verifier replay |

Configs and worker hashes guard resumability. Use a fresh output label after
changing prompts, settings, or worker source; a historical run's config is
provenance rather than a promise that current code recreates it verbatim.
