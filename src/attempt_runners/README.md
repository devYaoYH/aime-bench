# Attempt runner versions

Keep experimental policies in this directory. The canonical runner stays at
`src/attempt.py`; do not add experimental baseline behavior or switches to it.

| Runner | Entry point | Behavior |
| --- | --- | --- |
| [Canonical](../attempt.py) | `python -m src.attempt` | Original streaming candidate extraction, fan-out, and coverage/continuation modes |
| [Naive pass@4 v1](naive_pass4_v1.py) | `python -m src.attempt_runners.naive_pass4_v1` | One batch, 30 questions × four samples; naturally completed final boxes only; stop at 18 correct |

The naive runner is isolated from the canonical module. Its service, warmup,
locking, GPU sampling, and SSE helpers use the frozen [_runtime_v1.py](_runtime_v1.py)
snapshot of the helpers used in commit `0834cfe`. Both runners write the existing
timestamped `attempts/` artifact format and use the same checkout lock and service
ports, so they must run sequentially on the same GPU. The existing attempt viewer
reads either runner's outputs.

For the v1 baseline, deploy the dedicated
[16K profile](../../configs/vllm/WeiboAI/VibeThinker-3B/vllm-baseline-16k.yaml)
to `~/models/WeiboAI/VibeThinker-3B/vllm-baseline-16k.yaml` after pulling:

```sh
~/.venvs/vllm/bin/python -m src.attempt_runners.naive_pass4_v1 \
  --model WeiboAI/VibeThinker-3B
```

Defaults are 30 questions in parallel, four independent samples per question,
18 verified correct as the stopping target, and a four-request per-question
budget. The profile sets 16K total context and an 95% GPU-memory budget. Output
limits reserve the actual prompt length. The runner uses a standard final-answer
prompt, ignores intermediate candidates, and leaves capped responses ungraded.
`--model-profile` selects another YAML filename within that model's directory.

New config and summary files record `runner_id: naive_pass4_v1` and the module
entry point, alongside the source commit, profile, budgets, and timing. Change an
experimental policy by adding a new version (for example `naive_pass4_v2.py`),
with its own tests and a catalogue row; preserve prior versions for comparisons.
Do not monkey-patch or import canonical runner behavior into a versioned policy.

The [recorded v1 baseline](../../attempts/20261003T202152.418590Z/README.md)
originally ran through `src.attempt --strategy baseline` at commit `0834cfe`.
Its saved config, summary, source commit, and original command remain historical
evidence. The policy is now separately runnable here; splitting the code did not
rerun or relabel that experiment.

## Prospective speedrun v1

| Runner | Entry point | Behavior |
| --- | --- | --- |
| [Speedrun v1](speedrun_v1.py) | `python -m src.attempt_runners.speedrun_v1` | Prospective candidates; default 30 questions × four 8K samples; question-wide deduplication; one pending grader request per question; stop at 18 correct |
| [Sweep v1](sweep_speedrun_v1.py) | `python -m src.attempt_runners.sweep_speedrun_v1` | Plan by default; explicitly execute sequential, isolated cells |

The speedrun uses its own frozen [_streaming_v1.py](_streaming_v1.py) policy and
v1 service runtime, with a [port probe](_ports_v1.py) that permits recently closed
HTTP connections in TIME_WAIT while rejecting active listeners. This avoids false
port conflicts between sequential cells. It does not modify the frozen naive
baseline runtime or canonical/naive solving policies.
Shared [instrumentation](../attempt_metrics.py) is observational. Each sample is
seeded by question index and rollout number with a fixed four-request stride,
so changing fan-out does not change the first sample's seed. This differs from
historical canonical seeding; use the matched control, not only the old 92.4-second
coverage result, when attributing a speedup to concurrency.

`--rollouts 4` uses all four allowed requests in the initial 8K batch; it never
issues a fifth continuation or retry. With one or two samples per group, unsolved
capped samples can continue their own exact prompt/output token prefixes using
the remaining request budget. A continuation is a new request, relies on prefix
caching, and is not guaranteed to retain active KV state. Missing/incomplete token
IDs fail the cell; context exhaustion starts a new sample. Later requests allow
up to 16K output, clipped to remaining context. The standard Vibe profile has 64K
**total context** and a 16K **generation ceiling**, unlike the naive baseline's
separate 16K total-context profile. The sweep uses the existing speedrun profile;
no model profile is changed by planning or executing it.

Default `--schedule eager` places all first groups in a FIFO queue, then appends
unsolved retries as workers free up; it does not wait for the slowest first group.
`--schedule barrier` waits for each full round as a comparison. Both preserve
per-question deduplication across groups and a single verifier per question.

The [manifest](../../configs/sweeps/vibe-speedrun-v1.json) has eight cells: question
parallelism 8/16/30 × fan-out 1/2/4, excluding 8 × 1 and 8 × 2, plus a 30 × 1 barrier control, all using the
same seed, temperature 0.8, top-p 0.95, and 8K first-pass budget. Edit the versioned
manifest's temperature/top-p/seed dimensions for additional sweeps. Partial parameter
mappings in `exclude` remove combinations before launch. Maximum initial
concurrency is 120 requests. Generate a plan locally without CUDA or SSH:

```sh
.venv/bin/python -m src.attempt_runners.sweep_speedrun_v1
```

After testing, committing/pushing, pulling on callosum, and confirming the other
experiment is finished, execute in a durable session:

```sh
~/.venvs/vllm/bin/python -m src.attempt_runners.sweep_speedrun_v1 --execute
```

Execution holds the checkout attempt lock and starts fresh managed inference and
grader services for each cell, warms the full configured sampling batch, and
refuses a busy GPU before CUDA warmup. It never stops unrelated GPU processes.
Defaults stop the sweep on an error; `--continue-on-error` records failures and
continues. Completed cells that exhaust the budget without 18 correct have no
target time or rank. Interruption persists partial results and closes owned services.

Each cell gets its own timestamped `attempts/` folder. Config/summary record
`runner_id: speedrun_v1`, scheduling, sampling, budgets, and provenance. Summaries
include first-18 elapsed time, fresh/continuation TTFT, observed VRAM, and the
[canonical overhead measurements](../../docs/attempts.md#runner-overhead-instrumentation).
Scores and the exact plan are saved in `runs/speedrun_sweeps/<timestamp>/`.
Rankings use official solving time to 18 verified correct, excluding startup and
warmup; initialization time is recorded separately. No speedup has been measured
for this new sweep until it is actually executed on the remote GPU.
