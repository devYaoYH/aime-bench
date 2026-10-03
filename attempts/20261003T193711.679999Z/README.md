# Initial Qwen 8 × 4 baseline

Attempt `20261003T193711.679999Z` ran on callosum's A100 80GB with
`Qwen/Qwen3.5-4B`, source commit `4c87c7fa62e6d5ad1480197da29fbe20945848b6`.
See `config.json` for the exact recorded commit and launch configuration.
Eight concurrent questions each launched four independent streaming rollouts,
with 16,384 output tokens allowed per rollout and a three-second global grader.
The solver used closed integer boxes and complete `Answer:` lines; prose extraction
and full-batch sampling warmup were added afterward.

| Measurement | Result |
| --- | ---: |
| Questions solved / attempted | 5 / 30 |
| Official solve wall time | 999.477 s (16m 39s) |
| Initialization plus solve time | 1,191.962 s (19m 52s) |
| Actual peak concurrent generation streams | 32 |
| Completed / cancelled / errored streams | 101 / 19 / 0 |
| Streams ending at the token cap | 100 |
| Completion tokens with complete usage records | 1,649,496 |
| Unique candidates submitted to grader | 5 |
| Correct / wrong grader verdicts | 5 / 0 |
| Grader toll (global worker busy time) | 15.000 s |
| Sum of grader queue waits | 1.261 s |
| Sum of grader query latencies | 16.261 s |
| Median / p95 client TTFT | 0.208 / 7.528 s |
| First batch median TTFT | 7.521 s |
| Observed device VRAM peak during generation | 65,040.188 MiB (63.52 GiB) |
| Observed device VRAM peak including setup | 66,104.188 MiB (64.55 GiB) |

The five submitted candidates were correct. Most other streams exhausted their
budgets without emitting an eligible marker. A broader syntax audit found 13
prose answer-number mentions, including hypothetical or fractional statements;
they are not 13 verified answers. The next runner adds guarded whole-integer
prose extraction, rejecting fractions, decimals, and unfinished numbers.
The 100 capped streams alone account for 1,638,400 output tokens. Cancelled
streams often lack final usage, so the complete-usage total undercounts actual
work. Generation, rather than grader activity, explains the long baseline.

The single greedy warmup left configured sampling kernels cold. vLLM logged
Triton sampling JIT during the first official batch; those spikes are included in
TTFT and the official solve time. Subsequent code warms the actual sampling
configuration at the full batch size before the timer starts. Comparisons to the
next coverage experiment also change model, parsing, and continuation strategy;
latency/accuracy differences cannot be attributed to concurrency alone.

VRAM is shared device memory, mostly preallocated by vLLM, not the memory cost of
one rollout. The server reported 1,474,085 KV tokens of capacity. Observed steady
usage was 64,272 MiB (62.77 GiB); the 80 GiB card retained roughly 17.2 GiB of
physical headroom. A 30-question × 1-stream round uses fewer simultaneous streams
than this measured 32-stream baseline. Capacity for 120 long streams cannot be
inferred from the short-context throughput profile.

## First-solved times

Reconstructed from the original client `verification_finished_at_utc` records,
relative to `config.json`'s official start. Grader timestamps/query IDs remain in
the original verification and grader audits. The current runner writes these
first-solved events directly to `solved.jsonl`, question records, and summaries.

| Question | Verified answer | First solved elapsed s | Client UTC timestamp |
| --- | ---: | ---: | --- |
| 3 | 16 | 59.995 | 2026-10-03T19:41:24.159Z |
| 6 | 504 | 62.996 | 2026-10-03T19:41:27.160Z |
| 1 | 70 | 167.549 | 2026-10-03T19:43:11.713Z |
| 16 | 468 | 459.491 | 2026-10-03T19:48:03.655Z |
| 17 | 49 | 526.392 | 2026-10-03T19:49:10.556Z |

Saved requests/responses, tokens, verification records, and GPU telemetry are
versioned beside this report for the canonical viewer and analysis. Full SSE,
service logs, and grader audits remain on callosum at
`/home/azureuser/aime-bench/attempts/20261003T193711.679999Z/` and are ignored by Git.
The managed services stopped successfully and released the GPU after completion.
