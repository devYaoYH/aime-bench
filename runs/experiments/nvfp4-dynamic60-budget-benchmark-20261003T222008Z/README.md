# NVFP4 dynamic 60-slot allocation and growing token budgets

Reached **18 verified correct in 127.565s**, with 19
completed grader checks (18 correct, 1 wrong).
This was slower than the earlier benchmark-mode NVFP4 30×1 / 16K run's 106.931s.
Several controls changed together, and generated paths vary: this is not a causal
measurement of the effect of any single optimization.

![Dynamic allocation and solved verdicts](allocation-timeline.png)

## Policy and actual request behavior

The runner started all 30 questions with one 8192-token sample. The first client
grader submission opened fan-out at **8.935s**.
The scheduler then filled up to 60 active requests, preferring ready continuations
and assigning fresh 8K samples to the least-active unsolved questions with rotated
ties. As questions solved, remaining questions reached as many as
**5 simultaneous samples**.
The recorded admission peak and reconstructed client-lifetime peak both equal 60.

There were **112 fresh requests and 28 continuations**, 140 total; no question
exceeded the explicitly allowed eight requests. Each continuation uses one request
from that cap. Requests can exhaust the cap before reaching the largest budget.
Completed capped streams continue with their exact original prompt plus output
IDs. The cumulative generated-token ladder is 8192 → 16384 → 32768 → 65536;
each request asks only for the additional tokens and clips to the remaining total
context. The 64K total context includes the original prompt.

All 28 actual continuations resumed an 8192-token capped parent and asked for
8192 more tokens, targeting 16384 total. Exact request prefixes and complete parent
ID evidence were verified for all 28. **No trajectory reached the 32K or 64K stages
before the stop-at-18 target**; those stages are covered by offline context-clipping
and budget tests, not demonstrated by this particular GPU run.

The standard 16K-generation launch profile remains intact. This experiment used
`vllm-expanding-64k.yaml`, with a 65536 generation ceiling and unchanged 65536 total
context, Marlin FP4 weights, BF16 activation/KV and 95% memory utilization.
Question-wide deduplication and one pending grader request per question persist
across all samples and segments. Target registration occurs immediately at the
solved verdict, before sibling cancellation settlement.

## Latencies and prefix-cache evidence

| Request group | Count | Median TTFT |
| --- | ---: | ---: |
| Initial single samples | 30 | 3.406s |
| Additional fresh samples | 82 | 0.114s |
| All fresh requests combined | 112 | 1.693s |
| Exact-ID continuations | 28 | 0.284s |

Per-request cache-hit counts were not returned for any of the 28 continuations;
their fractions remain null. Aggregate service-log prefix hit rates cannot be
attributed to individual requests. These groups have different arrival times, queueing and prefix sizes; the TTFT
contrast does not isolate a cache effect; continuations were slower than the later
fresh requests in this run. It does not imply that KV stayed active
between separate requests. Raw per-request cache counts and exact-prefix checks
are in [analysis.json](analysis.json).

Thirteen winning answers came from initial samples, four from additional fresh
samples, and one from a continuation. The continuation supplied the 18th correct
answer. These are first verified winning requests,
not counterfactual estimates of how many would solve without extra samples.
The grader spent 57.002s on completed jobs,
with 61.573s idle between them.
The final verdict linked to its grader answered timestamp is saved in
[solved.jsonl](../../../attempts/20261003T222009.163270Z/solved.jsonl).

## Memory, compilation and storage

Existing service logs showed 60 running requests, zero waiting at their sampled
logging instants, and peak KV usage **17.4%**. There were no
OOM or preemption warning lines; benchmark mode omits the preemption counter and
sampled VRAM peak, so zero evictions cannot be asserted from these observations.
Four sampler JIT warnings occurred during warmup before the official start; no
additional JIT-compilation warning occurred in official solving.
[Compact log audit](log-observations.json) retains the source hash and selected
observations; complete service logs remain remote.

Runner peak RSS was 510.3 MiB.
Official runner CPU was 71.567
CPU-s. Final flush took **4.372s** for
541,506 buffered rows, after timing and service cleanup;
it is excluded from time to 18 and official latency. Exit code 0 and final GPU
memory 0 MiB confirmed. The attempt retains required generation/TTFT/end-to-end
latencies, exact IDs, deduplicated verification events and all first-solved
verdict timestamps. Optional CPU/engine/GPU profiling is disabled.

## Provenance and validation

- Source commit `9bbf5378cad2ede28324f1cfeed3c37ba09da034`; clean remote checkout.
- [Complete command and experiment controls](config.json).
- [Admission ledger](../../../attempts/20261003T222009.163270Z/allocation.json).
- [Summary](../../../attempts/20261003T222009.163270Z/summary.json).
- [Detailed analysis and continuation prefix audit](analysis.json).
- [Timeline data](timeline-analysis.json); adjacent PNG/SVG/PDF exports.

145 offline tests passed before committing/pushing and pulling for execution.
Post-run validation confirmed the 60-stream ceiling, eight-request cap, seed
stride, fresh/continuation budgets, all 28 exact prefixes, required latency fields,
and 18 distinct solved events linked to grader query/answer timestamps. Metadata
schema validation passed after import. A backward-compatible schema extension
accepts scalar budget arrays; all 147 offline tests passed after that fix. Full SSE/audits/service logs are excluded
from Git. Re-render with `python scripts/render_dynamic_attempt.py
attempts/20261003T222009.163270Z runs/experiments/nvfp4-dynamic60-budget-benchmark-20261003T222008Z`.
