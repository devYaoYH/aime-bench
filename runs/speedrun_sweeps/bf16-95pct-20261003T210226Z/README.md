# VibeThinker BF16 concurrency sweep at 95% VRAM

Three cells reached 18 distinct verified correct questions. The user stopped the
remaining sweep; no 16-question or 8-question cells ran. Cleanup completed with
0 MiB GPU memory in use. The batch's exit code 1 and `interrupted` status reflect
the requested cancellation, not an OOM.

| Setting | Time to 18 correct | Median fresh TTFT | Completed checks (wrong) | Peak KV pool usage |
| --- | ---: | ---: | ---: | ---: |
| 30 questions × 1 rollout, barrier control | **71.135 s** | 0.159 s | 18 (0) | 5.92% |
| 30 questions × 2 rollouts, eager | 77.029 s | 1.885 s | 19 (1) | 9.22% |
| 30 questions × 4 rollouts, eager | 114.615 s | 3.846 s | 19 (1) | 20.04% |

Times end at the first grader verdict that makes the count 18, excluding GPU/model
initialization and subsequent teardown. All three reached the target during the
first 8,192-token pass; none issued continuations. With no next round, the barrier
control's inter-round barrier did not affect its execution. These are intermediate
answer speedruns, not the cancelled final-only naive baseline.

The original canonical coverage reference was rerun at 95% and reached 18 in
97.345 s, with 30 initial requests and 14 continuation segments. It uses the old
seed stride, whereas the experimental runner pairs seeds with a fixed four-request
stride. Its comparison with this control is therefore confounded by the seed
change as well as runtime sampling differences.

## Interpretation

This is one trial per setting, and does not establish a concurrency optimum or
prove that an earlier improvement was entirely a lucky draw. For every question,
the first-rollout request JSON, seed, and prompt token IDs were identical across
the three sweep cells. Nevertheless all 30 output trajectories diverged in both
adjacent comparisons. Median shared prefixes were 127 tokens for 30×4 versus
30×2 and 135 for 30×2 versus 30×1. Thus nominally paired seeds did not produce
matched realized trajectories under these execution settings. The records do not
identify the cause of that divergence. See the per-question comparisons in
[analysis.json](analysis.json).

The serial grader accounts for much of the elapsed time:

| Setting | First grader pickup | Grading service time | Idle between checks |
| --- | ---: | ---: | ---: |
| 30×1 control | 5.055 s | 54.002 s | 12.076 s |
| 30×2 | 9.253 s | 57.002 s | 10.774 s |
| 30×4 | 11.332 s | 57.002 s | 46.279 s |

These components approximately sum to the time to target. Two fanout cells each
paid one extra 3-second wrong check. The 54-second floor assumes 18 serial correct
checks and zero waiting or incorrect checks; it is not a prediction of total run
time.

## KV cache and runner overhead

All completed cells recorded **zero vLLM preemptions**, and no OOM messages. KV pool
occupancy peaked at 20.04%, so the observations do not support capacity-driven
active-request preemption as the explanation for the 30×4 slowdown. That cell had
a brief initial waiting queue of up to 38 requests; the other two recorded zero
waiting requests. Sampled official VRAM peaked at 78,574 MiB for 30×4 and 78,434
MiB for the other cells. vLLM reserves the KV pool up front, so total device
allocation does not represent occupied KV blocks.

Exact prefix-cache block eviction counts were **not recorded**. Installed vLLM
defaults `enable_kv_cache_events` to false, and these launches did not override it.
Unused cached-prefix removal is distinct from active-request preemption; a zero
preemption counter cannot prove zero cached-block evictions. For a future trace,
the installed vLLM supports a `--kv-events-config` ZMQ publisher that emits
`BlockStored`, `BlockRemoved`, and `AllBlocksCleared` events. A subscriber must
capture those events and account for sequence gaps; this cannot be reconstructed
retroactively from this batch's aggregate metrics. No additional GPU run was
started to collect them.

Observed official runner overhead:

| Setting | Runner CPU (user + system) | Event-loop mean / maximum delay |
| --- | ---: | ---: |
| 30×1 control | 34.401 CPU-s | 1.42 / 66.56 ms |
| 30×2 | 55.593 CPU-s | 4.89 / 177.09 ms |
| 30×4 | 94.135 CPU-s | 17.49 / 616.20 ms |

For 30×4, synchronous trace writing used 16.284 CPU-s, JSON decoding 6.196 CPU-s,
and candidate parsing 4.602 CPU-s. These are components of runner CPU usage, not
additional elapsed times to sum with GPU work or overlapping async waits. The
measurements show increasing client load, but cannot isolate its causal effect
from longer generated trajectories and inference scheduling.

## Provenance and retained attempts

All sweep cells ran `src.attempt_runners.speedrun_v1` at source commit `bf5788f`,
using WeiboAI/VibeThinker-3B BF16, temperature 0.8, top-p 0.95, seed 20261003, and
the 95% `vllm.yaml` profile. The server context limit was 65,536; first requests
had an 8,192 generation budget, later segments a 16,384 cap, and at most four
generation requests per question. A pass@4 cell leaves no continuation budget.

- [30×4 eager](../../../attempts/20261003T211054.387701Z/summary.json)
- [30×2 eager](../../../attempts/20261003T211346.861294Z/summary.json)
- [30×1 barrier control](../../../attempts/20261003T211557.382358Z/summary.json)
- [Canonical 95% coverage reference](../../../attempts/20261003T210230.379752Z/summary.json)
- [Interrupted 30×1 eager cell](../../../attempts/20261003T211800.426093Z/summary.json), already started before cancellation; not ranked.
- [Interrupted naive baseline](../../../attempts/20261003T210650.263990Z/summary.json), cancelled on request; not ranked or rerun.
- [Earlier naive startup failure](../../../attempts/20261003T210502.392028Z/summary.json), caused by the closed-port probe and preserved separately.

[grid_summary.json](grid_summary.json) preserves the remote sweep's final board,
including its uncompleted fourth cell. [config.json](config.json) preserves the
batch launch history; [analysis.json](analysis.json) contains derived measurements
and stop/trace limitations. The 2,105 locally imported JSON/JSONL evidence files
were SHA-256 checked against the remote copies; all matched. Full SSE streams,
service logs, and grader audits remain on the remote machine, outside Git.
