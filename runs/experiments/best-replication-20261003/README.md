# Replicating the 71-second best run

The new fastest observed time to 18 verified correct answers is **59.316s**, using NVFP4 with FlashInfer attention. Reliable repetition remains unproven: a separately declared validation failed, with scored times of **87.356 / 59.316 / 86.574s** after a 136.048s settling run. This changes model/backend configuration rather than exactly replicating the original BF16 run. All twenty-seven completed follow-ups are retained; the goal remains active.

![Timing comparison](timing-comparison.png)

| Comparison | Trials | First-18 times | Median |
| --- | ---: | --- | ---: |
| BF16 benchmark, matched 30×1 | 3 | 117.17 / 134.52 / 135.65s | 134.52s |
| BF16, original profiling restored through v2 | 3 | 116.28 / 150.07 / 152.01s | 150.07s |
| NVFP4 benchmark, paired 30×1 | 3 | 115.40 / 96.67 / 96.67s | 96.67s |
| BF16 recycle up to 30 streams, new policy | 3 | 92.01 / 105.40 / 114.30s | 105.40s |
| BF16 exact original source, fresh services | 1 | 102.99s | — |
| NVFP4 benchmark, four predeclared different seeds | 4 | 118.19 / 94.59 / 126.43 / 114.44s | 116.31s |
| NVFP4 FlashInfer, development comparison | 3 | 98.33 / 64.45 / 64.40s | 64.45s |
| NVFP4 FlashInfer, independent scored validation | 3 + 1 settling | 136.05s settling; 87.36 / 59.32 / 86.57s scored | 86.57s scored |
| NVFP4 FlashInfer, suspend pending verdict v5 | 3 | 95.80 / 99.24 / 83.06s | 95.80s |

The reference's grader used 54.002s of service and waited 12.076s between checks. Earlier unsuccessful comparisons waited 26.964–91.676s; the two faster FlashInfer trials waited only 2.611/2.473s despite using 57s of service for 19 checks. The figure stacks first grader pickup, completed service and idle time; their sum agrees with each first-18 time within 0.03s. Initial TTFT is the conventional median of the first request per question. Startup, cleanup and buffered final writes are outside this solving window.

All nine matched warmed trials retained the original seed, 30×1 barrier, 8K first pass, up to 16K additional output on later requests, temperature 0.8, top-p 0.95 and four-request cap. NVFP4 substitutes model/quantization. Prefix caches were reset before each attempt's normal short warmup, so later trials could not reuse prior solution tokens. Every initial payload matches after that declared model substitution.

The dynamic comparison changes scheduling and continuation semantics: its ladder is 8K then 16K cumulative output and freed slots start fresh samples. It preserves the initial payloads but is a distinct optimization. The exact-source replay restores the original code commit, fresh managed services, profiling and immediate writes; it rules out source integration as a sufficient explanation for recovering 71s.

Paired seeds do not guarantee identical output paths. In the earlier BF16 benchmark repeat, question 1 decoded at 129.5 tokens/s versus the reference's 129.6. New warmed repetitions decoded near 128–130 tokens/s, but two generated 1,344 tokens before cancellation versus 1,030 in the reference. This supports a trajectory/answer-arrival explanation; it is not proof of the underlying numerical or scheduling cause. Neither the TTFT nor stopped-at-18 comparisons establish full reasoning accuracy or a BF16 quality advantage.

An active GPU health observation found SW power capping with a 1395 MHz SM clock versus its 1410 MHz maximum, without thermal/hardware slowdown. Earlier idle checks found zero zombies, no retained GPU allocations and 211 GiB RAM available. Clearing memory caches is unsupported by this evidence.

The report retains every result. The four-seed NVFP4 comparison completed with seeds declared before launch; none recovered 71s. Its fastest result, 94.59s, has not been selected for validation. Any selected fast seed would require separate repeat validation and identification as development tuning, rather than an exact replication or a guarantee across arbitrary draws.

A [fresh health check](../../diagnostics/20261003T232632Z/README.md) after this batch again found zero zombies, no GPU allocations and 211 GiB RAM available. The batch logged at most 9.6% KV usage and no preemption/OOM warnings. EngineDeadError appeared during intentional shutdown after the final completed trial. There is no evidence supporting a memory-cache flush.

[Plot data](analysis.json), [PNG](timing-comparison.png), [SVG](timing-comparison.svg), [PDF](timing-comparison.pdf). Render with `python scripts/render_best_replication.py`.

Batch evidence: [BF16 benchmark](../bf16-best-warm-replicate-20261003T224038Z/README.md), [BF16 profiling](../bf16-best-profiled-replicate-20261003T224630Z/README.md), [NVFP4 benchmark](../nvfp4-best-warm-replicate-20261003T224630Z/README.md), [dynamic 30](../bf16-dynamic30-replicate-20261003T225437Z/README.md), [exact source](../bf16-best-exact-source-20261003T230424Z/README.md).

[Declared-seed batch](../nvfp4-30x1-seed-comparison-20261003T231357Z/README.md).

[FlashInfer development batch](../nvfp4-flashinfer-best-replicate-20261003T233008Z/README.md). Its first median initial TTFT was 3.510s, then 0.198/0.240s. The backend change can alter sampled reasoning paths; its faster time to 18 does not isolate a pure throughput improvement. The [independent validation](../nvfp4-flashinfer-validation-20261003T233628Z/README.md) predeclared one settling attempt followed by three scored attempts, each required to beat the original time, retaining all outcomes and resetting prefix caches before every attempt. Only one scored attempt passed. Plot validation trial #1 is settling; #2–#4 are scored. Normal TTFT in the slow validation runs contradicts TTFT as a sufficient explanation for their delay.

The fastest run used 54.003s grader service, started checking at 4.829s and waited only 0.483s between checks. The other scored trials waited 25.657/27.739s. A 59s observed result is therefore close to this grader's floor; the remaining challenge is making correct candidates arrive that early consistently.

[Selected server evidence](flashinfer-server-evidence.json) confirms FlashInfer attention and Marlin NVFP4 weights in both batches, plus startup compilation and throughput observations. [Pending-verdict overlap](pending-verdict-overlap.json) measures 106–143 overlapping stream-seconds after eventual correct candidates in selected runs. Those sums are not additive wall-time savings. A separate versioned policy can test releasing GPU work while verdicts are pending, then safely resume exact observed token IDs after wrong verdicts, preserving the request cap. The [v5 suspension experiment](../nvfp4-pending-verdict30-v5-20261003T235852Z/README.md) now tests this approach; its three repeats did not beat 71.135s. The selected [final runner](../../../runner_final/README.md) retains the barrier policy and adds a separately declared ungraded AIME 2024 workload warmup.
