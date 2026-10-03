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

Model weights and profiles live in `~/models/<model ID>/vllm.yaml`. The Qwen
profile is also tracked in this repo so its changes can follow the same local
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
global toll, waits for both services, and makes a 32-token inference warmup.
The official solving timer starts after initialization and warmup. The directory
ID is the UTC timestamp of attempt initialization; config and summary distinguish
initialization, official solving, and cleanup. The runner stops its own services
on completion, errors, SIGINT, and SIGTERM. SIGKILL cannot run cleanup.

For an already running matching, idle server, explicitly attach:

```sh
~/.venvs/vllm/bin/python -m src.attempt --model Qwen/Qwen3.5-4B \
  --reuse-server --questions 1 3 --max-tokens 2048
```

Attach mode performs inference warmup and starts its own fresh grader, but leaves
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
closed integer `\\boxed{N}` anywhere in the text or a complete standalone
`Answer: N` line (including true terminal EOF). Chunk boundaries never terminate
an integer. The prompt asks for prospective boxed answers as soon as available.
The parser does not infer answers from arbitrary numbers or use the answer key.
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

## Artifacts and timing

```text
attempts/<UTC timestamp>/
  config.json, summary.json            compact provenance and outcomes
  model_profile.json, server_models.json
  gpu.jsonl                           200 ms timestamped device telemetry
  gpu_warmup.log, vllm.log, grader.log  managed service logs
  grader_config.yaml, grader_audit.jsonl
  inference_warmup.json
  trace/01/                           question index, through trace/30/
    question.json, verification.jsonl
    rollout-01/                       through rollout-04/
      request.json, stream.jsonl, response.json, telemetry.json
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
Only attempt configs, summaries, and README reports are Git-allowlisted; raw
traces stay on the remote unless explicitly copied for local analysis.

## Local checks

```sh
.venv/bin/pip install -r grader/requirements-local.txt
.venv/bin/python -m unittest test.test_attempt -v
.venv/bin/python -m unittest discover -s test -v
```

The canonical tests use fake timed SSE streams and a real loopback grader;
no GPU or inference requests are needed. Loopback binding requires execution
outside a sandbox that blocks listening sockets.
