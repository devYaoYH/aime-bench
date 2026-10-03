# Experiment guide

This repo runs one independent `qwen/qwen3-30b-a3b` sample for each of the 30
[MathArena AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025)
questions. The runner fans requests out concurrently and grades each final answer
locally against MathArena's official answer key. Its timer starts immediately before
the first inference HTTP request and stops after all local grading is completed.

Run all commands below from the repository root. See the [documentation index](README.md) for reports.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.fetch_dataset
```

Place `OPENROUTER_API_KEY=...` in `.env` or export it as an environment variable.
The key is never written into run artifacts. `fetch_dataset.py` saves problem
statements, official answers, and source metadata.

```bash
.venv/bin/python -m src.experiments.baseline.benchmark --concurrency 10
```

Each run creates `runs/<timestamp>/config.json`, `summary.json`, and one JSON file
per question in `questions/`. Each question file contains the exact request (minus
the API key), every API attempt and complete response body, usage counters, client
HTTP latency, extracted answer, and local grade. `summary.json` reports pass@1
over all 30 problems. `completion_tokens` includes
reasoning tokens; any separately reported reasoning count is a subset.

## Response viewer

```bash
python3 -m src.viewer_server
```

Open [the exploratory archive](http://127.0.0.1:8765/exploratory). This viewer is fixed to the original
run `20260930-155212`, shows its wall clock and concurrent request timeline, and
lets you filter questions by correctness, missing answers, or strict output format. Select
a question to inspect its API latency, output length, tokens, final response,
reasoning trace, and raw API response. “Format incorrect” means the final line
did not follow the requested `Answer: NNN` syntax. It is independent of whether
an answer was extracted from a `\\boxed{...}` expression and graded correct.
The local server exposes only viewer assets and
run JSON files; it does not serve `.env`. The root page is the separate
canonical attempt viewer described in [the attempt workflow](attempts.md#viewer).

## Jev continuation review

The 18 unfinished trajectories in run `20260930-155212` were reviewed with
`typesafe/jev-1.13` and four Noul decisions each. The model saw each problem
and its full saved reasoning trace, without the official answer or grade. See
[JEV_REVIEW.md](reports/jev-review.md) for all scores and interpretation. The viewer
also shows the ranking and per-question decisions alongside the original trace.

To reproduce or resume the review with an available `OPENROUTER_API_KEY`:

```bash
.venv/bin/python -m src.experiments.jev.judge_trajectories --run 20260930-155212 --concurrency 6
```

The script saves each full request and response to
`runs/<run>/jev_review/NN.json`, and the ranked scores to
`runs/<run>/jev_review/summary.json`. Existing successful decisions are reused.
The full-trace estimates have not been validated by continuing those capped
trajectories.

For a second Jev review using only the first 1,500 exact Qwen reasoning tokens
of each failed original trace, run:

```bash
.venv/bin/python -m src.experiments.jev.judge_prefix --run 20260930-155212 --tokens 1500 --concurrency 6
```

This uses the official `Qwen/Qwen3-30B-A3B` tokenizer and the same
`promising_to_extend` Noul question. It saves separate artifacts under
`runs/<run>/jev_prefix_1500/`. The viewer places these prefix scores beside
the full-trace Jev scores; both reviews concern the original first attempt.

## Retrospective Jev check on completed trajectories

To check false negatives in the prefix decision, score the first 1,500
reasoning tokens of every Qwen trajectory that completed with a final response:

```bash
.venv/bin/python -m src.experiments.jev.calibrate_jev --run 20260930-155212 --concurrency 20
```

The script reuses existing decisions, withholds all outcomes from Jev, and
saves its full decision records under `runs/<run>/jev_calibration/`. It then
compares scores to observed correct answers, separately for completions within
Jev's 8,192-token continuation budget and later completions. See
[JEV_CALIBRATION.md](reports/jev-calibration.md) for threshold results and limitations.
The viewer shows the Jev score beside each completed attempt and a summary of
missed correct completions.

## Eight-sample self-consistency run

The first benchmark response is attempt 1 for every question. Add seven
independent samples per question using its exact original OpenRouter payload:

```bash
.venv/bin/python -m src.experiments.baseline.expand_samples --run 20260930-155212 --concurrency 30
```

The runner sends up to 30 requests at once, saves full responses and usage in
`runs/<run>/self_consistency/questions/NN/02.json` through `08.json`, and
reuses completed attempts on rerun. Its summary reports **pass@8** (at least
one correct attempt), the modal answer vote, and a strict majority vote
(at least 5 of 8 attempts agree). Missing final answers abstain from the modal
vote; a tie has no modal answer. The viewer groups all eight attempts with
their answers, latency, tokens, response text, and reasoning traces.
The completed results and per-question vote distributions are in
[SELF_CONSISTENCY.md](reports/self-consistency.md). Elapsed wall time is calculated from
the saved UTC timestamps so that laptop sleep is included.

## Reflex-8: local Qwen3.5-4B quick answers

On callosum, launch the Qwen vLLM config in `~/models/Qwen/Qwen3.5-4B/vllm.yaml`,
then collect eight short-answer samples for each of the same 30 problems:

```bash
~/.venvs/vllm/bin/python -m src.experiments.local_qwen.no_thinking_pass8
```

This sends up to 240 independent requests concurrently with
`chat_template_kwargs.enable_thinking=false` and distinct seeds. Each response
is asked for one integer, with a 64-token cap. Full requests, responses, and
retries stay in ignored `runs/qwen35-4b-no-thinking-pass8-20261003/questions/`.
The versioned `config.json` records model and source revisions, the launch-config
hash, sampling, prompt, and grading rules; `samples.csv` and `summary.json` hold
the candidate and vote data. A question has 4-of-8 agreement if any parseable
answer receives at least four votes. A 4–4 tie is reported separately because it
does not select one answer. See the [run provenance note](../runs/qwen35-4b-no-thinking-pass8-20261003/README.md).

## Reasoning-length and escalation analysis

Plot the exact Qwen-tokenized reasoning length of all eight attempts per
question, and summarize the remaining same-model success rate at several
reasoning checkpoints:

```bash
.venv/bin/python -m src.experiments.baseline.analyze_escalation --run 20260930-155212
```

The plot and machine-readable metrics are saved under `runs/<run>/analysis/`
and appear in the viewer. See [ESCALATION_ANALYSIS.md](reports/escalation-analysis.md)
for the checkpoint table, pass@N versus generated calls, and distinct-answer
verification counts. Those observations do not measure the benefit of a
different model; that requires a separate paired experiment.

This benchmark uses the remote model for exploration; it is not the locally hosted
scored run described in the adjacent speedrun brief.

## Local CPU answer salvage with Gemma

Install a pinned llama.cpp macOS ARM64 binary and the llama.cpp maintainers'
Q8 GGUF conversion of `google/gemma-3-1b-it`, then serve it on CPU:

```bash
python3 -m src.experiments.local_salvage.local_gemma setup
python3 -m src.experiments.local_salvage.local_gemma serve
```

Downloads, binaries, and logs live in ignored `.local/`. The server binds to
`127.0.0.1:8091`, uses six CPU threads, disables GPU/KV/operation offloading,
and provides two slots with 32,768 tokens each. `--threads` and `--parallel`
can be adjusted. Setup requires the existing Python requirements and internet
access; the experiment itself sends requests only to loopback.

In another terminal, run the selected capped-trace pilot and verification controls:

```bash
python3 -m src.experiments.local_salvage.local_salvage
python3 -m src.experiments.local_salvage.local_salvage --regex-baseline
python3 -m src.experiments.local_salvage.local_salvage --cohort explicit --capped 8 --full 2 --controls 0 --label gemma-q8-explicit
python3 -m src.experiments.local_salvage.local_salvage --cohort explicit --capped 8 --full 0 --controls 5 --candidate-last --extractor-last --label gemma-q8-instructions-last
python3 -m unittest test.test_local_salvage
```

The default pilot selects eight capped questions, compares two full traces with
4,096 Gemma-token tail windows, and checks five correct, five naturally wrong,
and five deliberately corrupted completed-answer controls. The `explicit`
cohort includes capped traces with existing final-answer syntax, selected without
consulting correctness. The model never receives the answer key. Extracted
answers require a literal supporting quote; ungrounded outputs abstain.
Verification produces `supported`, `inconsistent`, or `insufficient`.

Requests, raw responses, timing/usage counters, and summaries are saved separately
under `runs/<run>/local_salvage/<label>/`. Requests can be resumed with the same
configuration; choose a new label when changing parameters. Original benchmark
records are not modified. Model verdicts are experimental signals and do not
replace an independent mathematical check. This pilot does not run GPU retries
or establish that CPU/GPU concurrency has no throughput penalty.

See [LOCAL_SALVAGE.md](reports/local-salvage.md) for measured results and interpretation.

## Intermediate answers and first-answer stopping

Analyze all 240 saved trajectories offline, compare intermediate answer claims
with the stored key, and replay one vote per trajectory with 1–8 parallel samples:

```bash
.venv/bin/python -m unittest test.test_intermediate
.venv/bin/python -m src.experiments.intermediate_answers.analyze_intermediate --run 20260930-155212
.venv/bin/python -m src.experiments.intermediate_answers.render_intermediate --run 20260930-155212
```

These commands use the cached `.local/tokenizers/qwen3.json` and do not start
inference or contact a remote grader. They preserve all source records and write
claim inventories, a 240-row trace map, subset-averaged voting metrics, token-budget
replays, and plots under `runs/<run>/intermediate_answers/`.

See [INTERMEDIATE_ANSWERS.md](reports/intermediate-answers.md) for the findings and the
per-question map. Literal extraction is reported alongside manually audited toy
example exclusions; top-two coverage and token-work savings are distinguished
from single-answer accuracy and measured wall time.

## Paired first-answer prompt pilot

`first_answer_pilot.py` compares the original prompt with an added instruction to
stop after the first complete requested answer on Q1, Q3, Q18, and Q25. It sends
one request per arm per question to the same Qwen endpoint, streams both arms,
and records natural endpoints, answer accuracy, token usage, and delivery times.
It uses the original sampling and 16k cap, pins both arms to DeepInfra, and does
not impose a stop sequence or cancel at an answer marker.

```bash
.venv/bin/python -m unittest test.test_first_answer_pilot
.venv/bin/python -m src.experiments.streaming.first_answer_pilot --out runs/first-answer-NEW-LABEL
.venv/bin/python -m src.experiments.streaming.report_first_answer_pilot --out runs/first-answer-NEW-LABEL
```

The pilot makes eight paid API calls; successful saved requests are reused at the
same output path. Rendering the report alone is offline. Exact prompts, complete
responses, raw SSE events, and timing records are stored in the chosen directory.
See [FIRST_ANSWER_PILOT.md](reports/first-answer-pilot.md) for the observed results.

For a chunk-safe detector of standalone final-answer blocks, including reasoning
deltas, see `stream_answer_markers.py`. Its offline audit checks the eight new
SSE streams and the original 240 traces without making API calls:

```bash
.venv/bin/python -m unittest test.test_stream_answer_markers
.venv/bin/python -m src.experiments.streaming.audit_stream_markers
```

[STREAMING_ANSWER_MARKERS.md](reports/streaming-answer-markers.md) documents measured
marker arrival times, formatting false positives, the precision/coverage tradeoff,
and the remaining need to test actual stream cancellation and billing.

For hosted small-model extraction from causal stream windows and a nonblocking
verification sidecar that keeps the original solver running, see
[CHUNK_ANSWER_EXTRACTION.md](reports/chunk-answer-extraction.md). The shadow pilot uses
saved SSE trajectories and OpenRouter, without restarting local llama inference:

```bash
.venv/bin/python -m src.experiments.streaming.verify_sidecar --prepare-only
.venv/bin/python -m src.experiments.streaming.verify_sidecar --out runs/verify-sidecar-NEW-LABEL
.venv/bin/python -m unittest test.test_chunk_answer_extractor test.test_verify_sidecar
```

Extraction does not trigger an early exit. Failed, negative, and uncertain
verification decisions leave the solver running; model verification accuracy
must be evaluated separately from candidate extraction and latency.

## Pass@4 early verification back-test

Replay original samples 1–4 with a shared verifier that completes one exact
answer check every three seconds, including the cost of wrong intermediate
candidates. The original trajectory continues after negative checks. Plot the
step count of distinct verified-correct questions for final-only, marker, and
permissive checking policies:

```bash
.venv/bin/python -m unittest test.test_backtest_early_verify
.venv/bin/python -m src.experiments.intermediate_answers.backtest_early_verify
```

This is offline: saved keys stand in for an accurate verifier. Intermediate
arrival times are estimated from token fractions because the original responses
were not streamed. [EARLY_VERIFY_PASS4.md](reports/early-verify-pass4.md) reports the
time to 18/30, generation-capacity scenarios, timing sensitivity, and plot links.

## Qwen3.5 Python-tool pilot

`python_tool_pilot.py` pairs Python-enabled and no-tool runs on Q23 and Q25,
using Qwen3.5-35B-A3B through OpenRouter. It supports multiple native function-call
rounds and records cumulative output/input tokens, tool programs/results, elapsed
time, and final correctness. Generated math programs run in a restricted macOS
worker. See [PYTHON_TOOL_PILOT.md](reports/python-tool-pilot.md) for the interface, limits,
commands, and measured results.

For the full 30-question pass@2 profile with thinking enabled and optional Python
on every model round:

```bash
.venv/bin/python -m unittest test.test_python_tool_profile
.venv/bin/python -m src.experiments.python_tools.python_tool_profile --prepare-only
.venv/bin/python -m src.experiments.python_tools.python_tool_profile
.venv/bin/python -m src.experiments.python_tools.report_python_tool_profile
```

This sends 60 independent tool-enabled attempts through OpenRouter, with eight
active trajectories and two CPU workers. It applies no separate thinking budget
or forced tool call; the existing 16k cumulative output ceiling remains. Broad
16-round/16-call guards prevent indefinite tool loops. Every request, response,
tool program/result, token count, and final local-key comparison is saved under
`runs/python-tool-qwen35-pass2-auto/`, with a live `summary.json`. Terminal records
are reused without paid repeats; an interrupted running record requires inspection
before restart. Both samples run even if the first succeeds. The offline report
includes per-question outcomes, token/latency plots, tool errors and first-call
timing, and a step graph of correct final responses received. That graph does not
include the separate three-second verifier simulation.

Apply the original permissive answer extractor to these multi-round traces and
replay the shared three-second verifier offline:

```bash
.venv/bin/python -m unittest test.test_python_early_verify test.test_backtest_early_verify
.venv/bin/python -m src.experiments.python_tools.backtest_python_early_verify
```

This uses the official cached Qwen3.5 tokenizer, estimates text delivery within
each API round, and keeps measured CPU-tool timing. It reports the unchanged
permissive text policy and a separate Python-total/bare-number extension, charges
wrong candidates, and compares 60/30/8-slot capacity replays with fixed observed
starts. Results, source quotes, a false-positive audit, and plots are in
`runs/python-tool-qwen35-pass2-auto/early_verify_pass2/`. No inference or grader
is called. If needed, cache the official tokenizer as `.local/tokenizers/qwen35.json`
and its pinned source metadata as `.local/tokenizers/qwen35-source.json` first.

## Matched Qwen3.5 no-Python baseline (v1)

`src.experiments.python_tools.no_python_baseline_v1` runs locally against
OpenRouter, preserving every first request in `runs/python-tool-qwen35-pass2-auto/`
except the Python tool definition, tool choice, and tool-specific system suffix.
Parasail routing, disabled fallbacks, required parameter support, thinking,
sampling, per-question/sample seeds, 16,384 output tokens, eight concurrent
trajectories, and all 60 attempts match the historical optional-tool arm.
Responses must identify the same model and Parasail. No model code is executed;
only complete final answers are counted. Saved request hashes and the runner hash
protect the comparison; unfinished paid records block relaunch.

```sh
.venv/bin/python -m unittest test.test_no_python_baseline test.test_python_tool_profile -v
.venv/bin/python -m src.experiments.python_tools.no_python_baseline_v1 --prepare-only
.venv/bin/python -m src.experiments.python_tools.no_python_baseline_v1
```

The historical and baseline runs occur at different times. Matching provider and
seeds cannot freeze provider load, backend changes, or stochastic token output.
Interpret hosted latency differences descriptively.
