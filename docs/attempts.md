# Canonical remote attempts

Develop locally, test, commit and push; then `ssh callosum` and pull the tested
commit into `~/aime-bench` before running experiments. See [AGENTS.md](../AGENTS.md).
The canonical entry point is `python -m src.attempt` (Python 3.11+).

## Remote setup

The remote already has vLLM and NVML bindings in `~/.venvs/vllm`. Install the
vendored grader's dependencies in its own environment once:

```sh
cd ~/aime-bench
python3 -m venv grader/.venv
grader/.venv/bin/pip install -r grader/requirements-local.txt
```

Model weights and profiles live in `~/models/<model ID>/vllm.yaml`. The Qwen and VibeThinker
profiles are also tracked in this repo so its changes can follow the same local
review/test/push/pull workflow. Deploy it after pulling:

```sh
cp configs/vllm/Qwen/Qwen3.5-4B/vllm.yaml ~/models/Qwen/Qwen3.5-4B/vllm.yaml
```

Qwen's `override-generation-config` sets `max_new_tokens: 16384`, the server-wide
generation ceiling. `max-model-len: 32768` remains the total context window.
The runner requests 16,384 output tokens per rollout by default. A smaller
`--max-tokens` is useful for a bounded smoke run. The profile needs a server
restart to take effect; check active work before restarting an existing server.

## Run

With the GPU free and service ports available:

```sh
~/.venvs/vllm/bin/python -m src.attempt --model Qwen/Qwen3.5-4B
```

This creates an exclusive timestamped attempt, warms CUDA with a small matmul,
launches vLLM from the model's profile, starts a fresh grader with a three-second
global toll, waits for both services, and warms inference with the configured sampling settings at the full batch
size (up to 32 requests of 32 tokens each).
The official solving timer starts after initialization and warmup. The directory
ID is the UTC timestamp of attempt initialization; config and summary distinguish
initialization, official solving, and cleanup. The runner stops its own services
on completion, errors, SIGINT, and SIGTERM. SIGKILL cannot run cleanup.

For an already running matching, idle server, explicitly attach:

```sh
~/.venvs/vllm/bin/python -m src.attempt --model Qwen/Qwen3.5-4B \
  --reuse-server --questions 1 3 --max-tokens 2048
```

Attach mode performs the same inference warmup and starts its own fresh grader, but leaves
the existing inference server running. Its exact launch arguments are external;
the attempt saves the on-disk profile and `/v1/models` response, which cannot prove
that the existing server loaded that profile. Managed mode provides that guarantee.
Ports 8000 and 8077 bind loopback by default; occupied ports are rejected in managed
mode. Both ports are configurable. The checkout lock prevents overlapping runners
in this repo. Separate clones must also share the single inference/grader instance
rather than launching simultaneous attempts on one GPU.

Use `tmux` for a long run, and inspect the printed artifact path and service logs.
`--questions` selects a smoke subset; omitting it runs all 30 AIME 2025 questions.
`--parallelism` defaults to eight questions and `--rollouts` to four independent
streams per question (up to 32 active streams). `--disable-thinking` explicitly
sets Qwen's chat-template option; thinking is enabled by the model's default otherwise.
Seeds, prompt, sampling configuration, question selection, and Git provenance
are persisted. There are no automatic retries of partial generations.

## Streaming and verification

Each question worker fans out its streams, and static parsing watches both
`content` and vLLM's `reasoning`/`reasoning_content` channels. A candidate is a
closed integer `\\boxed{N}` anywhere in the text, a complete standalone
`Answer: N` line, or a complete integer clause such as `the answer is 117` (including natural terminal EOF). Chunk boundaries and token caps never terminate an incomplete
answer line. The prompt asks for prospective boxed answers as soon as available.
The parser rejects partial integers, fractions, decimal answers, and arithmetic
expressions in these clauses. Hypothetical integer answer clauses can still become
prospective candidates and consume grader time. It uses no answer key.
Boxes in examples can still become candidates; only the grader decides correctness.

Candidates are deduplicated within a question. Its verification consumer submits
one candidate at a time, while all generation streams keep running. The vendored
grader's single global FIFO worker serializes requests from all questions and
charges the configured toll, including for the first query. Wrong answers allow
later candidates to be checked. On a correct verdict the worker cancels and closes
all remaining streams, waits for their cleanup, writes the question result, and
immediately takes the next question. vLLM handles HTTP disconnect cancellation.
Already queued grader jobs cannot be withdrawn by client cancellation.

When all streams end, the worker drains pending candidate verifications before
marking the question unsolved. The per-question timeout (default 1,800 seconds)
covers both generation and grading. HTTP, malformed stream, and grader failures
are recorded; attempts with failed questions exit unsuccessfully. A legitimate
unsolved question is a completed experiment, not an infrastructure error.


## Coverage-first continuation experiment

```sh
~/.venvs/vllm/bin/python -m src.attempt --model Qwen/Qwen3.5-4B \
  --strategy coverage --first-pass-max-tokens 8192 --max-tokens 16384 \
  --target-correct 18 --max-attempts-per-question 4
```

Switch to `--model WeiboAI/VibeThinker-3B` to run the same strategy with the
other available model. Deploy its tracked profile from
`configs/vllm/WeiboAI/VibeThinker-3B/vllm.yaml` to the corresponding `~/models`
path after pulling; its server also caps each request at 16,384 output tokens.

Coverage mode defaults to 30 concurrent questions and one streaming generation
per question. It tries all questions in the first round, then works only on
unsolved questions. It stops remaining work as soon as 18 questions are verified
correct, or after each question has used its four-request budget. **Every request
counts toward the per-question budget, including continuation segments.**
`--max-rounds` can impose a smaller round limit. The summary explicitly reports
whether the target was reached; exhausting the budget is a valid experiment
outcome, not an infrastructure error.

A token-capped unsolved response is continued from the exact returned prompt and
output token IDs via `/v1/completions`, with no chat retemplating. Each continuation
is linked to its parent rollout and respects both the server's 16,384-token output
ceiling and the model's remaining context (32,768 for Qwen; 65,536 for VibeThinker). If the trajectory ends
naturally with a wrong/no answer, or fills its context window, the next request
starts a fresh sample. `--no-continuation` provides a fresh-retry control.
Incomplete token-ID evidence fails the experiment rather than silently changing
the continuation prefix. The prefix is preserved, but sampling starts with a new
recorded seed for the suffix rather than resuming an internal RNG state.

vLLM's active request ends at the token cap. Reuse on a subsequent request depends
on automatic prefix caching and cache retention/eviction; it is not a persistent
GPU session. Both tracked model profiles explicitly enable prefix caching and prompt-token
usage details. Each segment saves token IDs, TTFT, cached prompt tokens, reported
cache-hit fraction, and VRAM. The experiment measures cache reuse instead of
assuming that saved text guarantees a cache hit. See the
[vLLM completion implementation](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completion/serving/).

`solved.jsonl` appends a record immediately on each first positive verdict, with
question index, client-observed solved UTC timestamp, elapsed seconds from the
official start, grader answer timestamp/query ID, round, and rollout. The same
first-solved record is preserved in `trace/<index>/question.json` and the attempt
summary. Per-round snapshots preserve outcomes and all generation records; retry
rounds never overwrite earlier rollout directories. Previously checked candidate
integers are deduplicated across rounds for that question.

## Naive final-answer baseline

The baseline is an independent versioned runner, separate from `src.attempt`.
See the [runner catalogue](../src/attempt_runners/README.md) for version policies.
Deploy the separate `configs/vllm/WeiboAI/VibeThinker-3B/vllm-baseline-16k.yaml`
profile beside `~/models/WeiboAI/VibeThinker-3B/vllm.yaml`, then run:

```sh
~/.venvs/vllm/bin/python -m src.attempt_runners.naive_pass4_v1 \
  --model WeiboAI/VibeThinker-3B \
  --parallelism 30 --rollouts 4 --max-tokens 16384 --target-correct 18
```

This submits all 120 streaming requests in one pass. It uses the system prompt
"You are a helpful assistant. Solve the math problem carefully and put your final
answer in \\boxed{} notation." It extracts no prospective answers. After a
response naturally finishes, only the last integer box in its final content is
eligible for grading; an explicit `</think>` boundary excludes earlier reasoning.
Token-capped responses are ungraded, even if earlier text contains a box.
No retries or continuations occur. A correct final response cancels its question's
siblings; the 18th solved question cancels all remaining work. If the target is
unmet after all four samples end, the summary records that outcome.

The baseline profile sets a **16,384-token total context**, including the prompt,
and retains the 95% GPU-memory budget. Before the official timer, `/tokenize`
counts each served chat prompt. Each output budget is the smaller of `--max-tokens`
and the remaining context; these prompt counts and request limits are saved.
The configured sampling path warms at the full 120-request batch before solving.
Use timestamped NVML samples together with vLLM's KV-occupancy, waiting-request,
and preemption logs. Preallocated VRAM can stay steady while the token cache fills
and vLLM preempts/recomputes requests, so VRAM alone cannot detect capacity pressure.

## Artifacts and timing

```text
attempts/<UTC timestamp>/
  config.json, summary.json            compact provenance and outcomes
  model_profile.json, server_models.json
  gpu.jsonl                           200 ms timestamped device telemetry
  gpu_warmup.log, vllm.log, grader.log  managed service logs
  grader_config.yaml, grader_audit.jsonl
  solved.jsonl                        first-solved events linked to oracle times
  inference_warmup.json
  trace/01/                           question index, through trace/30/
    question.json, verification.jsonl, round-01.json
    rollout-01/                       through rollout-04/
      request.json, stream.jsonl, response.json, telemetry.json
      tokens.json                     coverage mode exact continuation evidence
```

Each rollout records UTC start and generation-end timestamps, first nonempty
output delta latency (client-observed TTFT), last output time, full observed
stream latency, completion usage when provided, finish reason, and cancellation
or error status. TTFT includes queuing and prefill and may contain several tokens
in its first delta. A cancelled or token-capped stream's observed duration is censored; it is not
an estimate of its hypothetical natural full-completion latency.

Rollout end-to-end time runs from request start until its question has settled
verification and stream cancellation, so it includes waiting after natural
generation completion. `finished_at_utc` marks that settlement. Question and
official-attempt wall times are separately recorded. Initialization and cleanup
are excluded from the official solving timer.

NVML records shared device VRAM and utilization throughout setup and solving;
each rollout reports start, end, and observed peak VRAM over its generation
interval. These measurements include concurrent rollouts and vLLM's preallocated
cache, so they cannot attribute VRAM to an individual request. The sampler is
required for remote attempts; telemetry startup/runtime failures fail the attempt.
The grader audit contains gold answers and remains an ignored raw artifact.
Saved canonical viewer/analysis evidence is Git-allowlisted, including requests,
responses, exact token records, verification events, question/round records, GPU
samples, and launch/warmup metadata. Full SSE dumps and service logs remain remote.

## Viewer

Run `.venv/bin/python -m src.viewer_server` locally and open
[canonical attempts](http://127.0.0.1:8765). The viewer discovers canonical
directories under `attempts/` and offers question and rollout inspection,
positive-verdict timing, request/response evidence, continuation ancestry,
cache/TTFT counters, and sampled device VRAM/utilization. The first solve chart
uses saved oracle events; older attempts use their recorded winning-verdict
timestamps when available. Missing usage and censored generations are labeled.

Configs and summaries alone provide an overview. Committed canonical attempts
include their detailed evidence, so a fresh clone can inspect them. For a new
attempt, copy the selected attempt's `trace/` directory from callosum; `request.json`,
`response.json`, `telemetry.json`, `question.json`, and `verification.jsonl` are
enough for the main views. Copy `gpu.jsonl` and `solved.jsonl` for device plots and
first-solve events. `tokens.json` is optional exact-continuation evidence;
full `stream.jsonl` files are optional and much larger, and remain ignored by Git.
Review sizes and commit the allowlisted evidence after copying. The viewer never
connects to the VM or launches services.
"Refresh files" rereads the local copy, with optional five-second polling for
an attempt being written locally. An unfinished JSONL tail is tolerated.

The [exploratory archive](http://127.0.0.1:8765/exploratory) retains the original
fixed dataset, pass@8 votes, and Jev analyses independently of canonical attempts.

## Experiment metadata and overall results

Every imported attempt has `metadata.json`, validated against
[`data/attempt_metadata.schema.json`](../data/attempt_metadata.schema.json).
`python -m src.attempt_metadata --all` creates missing files and validates existing
annotations without overwriting them. Fill in `label`, `intervention.label`,
`reference_attempt_id`, `changed_variables`, and `comparison_note` after review.
The recorded model/revision, quantization, activation/KV dtype, GPU VRAM envelope,
runner module/version/source commit, question parallelism, rollout fan-out,
sampling, prompt hash, context, continuation budget, and grader settings make
controls explicit. Null values mean that evidence was not recorded. Historical
runner modules and commits are preserved even when the policy later moves.

Open [Overall results](http://127.0.0.1:8765/results) to compare all local attempts.
Times come from the eighteenth distinct first-solved event, including historical
winner-verdict timestamp fallback. Official settlement time is shown separately
and is never used to fill missing target timing. The history plot places each dot
at its initialization start timestamp and traces
the chronological running minimum with a dotted Pareto step. Older records use
their official start timestamp when initialization was not recorded.
Failed/interrupted/unmet attempts
remain visible. The 54-second reference floor assumes a serial 3-second grader,
18 correct queries, zero wrong/duplicate queries, and no initial generation delay.
Generation can overlap grading. Individual run comparisons show observed deltas
and all recorded setting differences; they do not establish isolated causal effects.

## Local checks

```sh
.venv/bin/pip install -r grader/requirements-local.txt
.venv/bin/python -m unittest test.test_attempt -v
.venv/bin/python -m unittest discover -s test -v
```

The canonical tests use fake timed SSE streams and a real loopback grader;
no GPU or inference requests are needed. Loopback binding requires execution
outside a sandbox that blocks listening sockets.

## Runner overhead instrumentation

Canonical scheduling, candidate extraction, and request budgets are unchanged.
The existing question-wide normalized answer set already suppresses duplicates
across rollouts and continuation rounds; one consumer awaits each grader response
before submitting the next candidate for that question. Other questions may queue
one request each, and generation continues while verification is pending.

New `overhead` fields in rollout/question telemetry and the attempt summary record
synchronous thread CPU and wall time for SSE JSON decoding, candidate parsing and
enqueueing, stream trace serialization/write/flush, JSON artifacts, and GPU sample
window scans. Counters include parsed proposals, suppressed duplicates, unique
candidates, generation outcomes, and verification outcomes. Local candidate queue
waits, grader queue/service times, HTTP response waits, initialization readiness,
and cancellation settlement record wall time only. Concurrent wall waits overlap:
do not sum them into an attempt latency. Persisted snapshots precede their own
final write; the attempt aggregate includes those writes.

`overhead.json` also records runner process CPU/RSS, sampled event-loop delay, and
vLLM `/metrics` observations (`inference_metrics.jsonl`). These include running and
waiting requests, KV occupancy, preemptions, cache hits, and queue/prefill/decode
counters when the server exposes them. Missing or failed metrics polling is logged
without changing solving behavior. Engine counters include the warmup baseline;
use first-to-last deltas. Sampled maxima can miss short spikes. NVML continues to
record device VRAM in `gpu.jsonl` independently.

Defaults sample event-loop lag every 50 ms and engine metrics every second.
`--overhead-interval` and `--engine-metrics-interval` adjust those rates.
`--no-overhead-profile` disables timers, counters, and background engine/lag
sampling for an instrumentation control, while retaining normal traces and NVML.
Profiling itself costs CPU; runner process CPU includes that cost.

`grader_timeline` decomposes completed audit jobs into actual service, idle gaps
between jobs, and the delay before the first pick. At a three-second toll, 18
correct checks alone cost at least 54 seconds. First-candidate delay, wrong checks,
and idle gaps add to that floor. Summed client waits include overlapping queue
waits and are not elapsed grader service. Cancelled submissions can still consume
oracle service; completed audit jobs are the evidence for charged completed work.

The deferred [speedrun v1 sweep](../src/attempt_runners/README.md) compares wider
first passes and eager continuations without adding an experimental policy to the
canonical runner.

## Benchmark year and generalization testing

Existing commands continue to use AIME 2025. Select the held-out 2026 benchmark
explicitly, using the same runner/model/sampling settings as the validated approach:

```sh
python -m src.attempt --model <organization>/<model> --benchmark-year 2026
python -m src.attempt_runners.speedrun_v1 --model <organization>/<model> --benchmark-year 2026
python -m src.attempt_runners.naive_pass4_v1 --model <organization>/<model> --benchmark-year 2026
```

`--benchmark-role` accepts `development` or `generalization`; when omitted it is
recorded as development for 2025 and generalization for 2026. This is independent
of the performance-instrumentation `--benchmark` flag. Year selection also applies
to the local speedrun v2 runner. For sweeps, put `--benchmark-year 2026` in the
manifest's `base_args`; each cell and score records its year and role.

The runner selects both prompts and grader key, validates that they agree, and
checks the grader's actual file hash before solving. Config and automatically
written metadata record `dataset_provenance`: year, role, source URL/revision,
split, prompt/key paths, and SHA256 hashes. Metadata stores this under
`provenance.dataset` with a readable `controls.dataset`. Gold answers are stripped
from the solver's problem records. The serial grader toll and solving policies
remain the same for both years.

The canonical viewer selects question text by the attempt's saved year and checks
its recorded prompt hash. The overall-results viewer filters plots and rankings by
year; cross-year annotated references do not produce a speed-improvement delta.

`python -m src.attempt_metadata --all` enriches old metadata while preserving
annotations. Old fixed-dataset runners are marked AIME 2025 with
`inferred_from_legacy_runner: true`; unrecorded historical role, revision and
hashes remain null. Existing configs and source commits are preserved.

Both datasets are versioned locally and usable offline. Refresh only deliberately:

```sh
python -m src.fetch_dataset --year 2026 --revision d2de22f3c656b4f56cf8981212186377d1e23bc3
```

The downloader requires `pyarrow`, pins the upstream Parquet download to a commit,
and generates prompt/key files together. Source manifests are `data/source.json`
(2025) and `data/source_2026.json` (2026). MathArena's upstream split is named
`train`; our experiment role is generalization and does not change that source
split. The 2026 transcription retains MathArena's contest variants, including
AIME II #1 = 178 and #10 = 850; do not mix it with another version's answer key.
See [dataset provenance](../data/README.md).
