# Why the five warmed runs vary

For the question-level reasoning patterns behind the tail, see [the audited late-answer report](TAIL_REASONING.md). Q26 repeatedly spends 23–24s recounting after reaching 113; slow Q23 spends 26.6s after reaching 610. Exact saved streaming chunks support these timings.

The fastest/slowest spread is **38.856s**. **37.704s (97.0%)** comes from extra grader idle time. All five use 19 completed grader checks, one wrong, and approximately 57.002s of service. Candidate arrival in the final few questions explains most of the measured difference. This identifies the timing bottleneck; it does not prove the numerical cause of each reasoning trajectory.

| Seed | First 18 | First-round winners among the final 18 | Initial median TTFT | Same capped questions: median generation | Same capped questions: effective decode | Local candidate wait median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261011 | 92.061s | 15 | 175.2ms | 56.594s | 145.4 tok/s | 0.330ms |
| 20261012 | 105.337s | 14 | 197.9ms | 56.791s | 144.9 tok/s | 0.393ms |
| 20261013 | 66.481s | 15 | 201.6ms | 56.469s | 145.7 tok/s | 0.292ms |
| 20261014 | 78.123s | 15 | 198.7ms | 56.509s | 145.6 tok/s | 0.203ms |
| 20261015 | 78.601s | 16 | 200.0ms | 56.666s | 145.2 tok/s | 0.306ms |

Common initial 8K-capped questions: Q07, Q09, Q10, Q11, Q12, Q13, Q14, Q15, Q20, Q23, Q28, Q29, Q30. These requests all produce 8,192 observed token IDs and finish at the cap. Effective decode is `(tokens − 1) / (last visible token − TTFT)`, not a hardware-only kernel benchmark. Comparing identical token counts reduces the bias from early cancellation, but concurrent batch composition can still differ.

The first-pass budget boundary occurs at nearly the same time, while different seeds produce different sets of early verified answers. Slower seeds need later continuations to supply the 17th/18th answer; those arrivals leave the serial grader idle. Barrier scheduling can add waiting before continuations, and continued reasoning does not guarantee a correct candidate at any particular token count.

| Seed | Largest grader idle gap | Next question | Its correct/checked candidate observed | Last three verified questions (winning round) |
| --- | ---: | --- | ---: | --- |
| 20261011 | 21.644s | Q02 | 82.416s | Q05 (r2, 60.8s), Q02 (r2, 85.4s), Q12 (r2, 92.1s) |
| 20261012 | 16.620s | Q26 | 82.312s | Q26 (r2, 85.3s), Q12 (r2, 99.5s), Q23 (r2, 105.3s) |
| 20261013 | 4.754s | Q11 | 57.448s | Q11 (r2, 60.5s), Q24 (r2, 63.5s), Q23 (r2, 66.5s) |
| 20261014 | 15.168s | Q26 | 69.119s | Q26 (r2, 72.1s), Q09 (r2, 75.1s), Q18 (r2, 78.1s) |
| 20261015 | 9.849s | Q07 | 68.855s | Q02 (r1, 59.0s), Q07 (r2, 71.9s), Q18 (r2, 78.6s) |

The new five trials deliberately use different sampling seeds. That changes reasoning paths and answer arrival. Earlier same-seed repetitions also differed: default online vLLM does not guarantee reproducibility across scheduling/batch changes. This is a plausible additional mechanism, not a demonstrated kernel fault here. [vLLM reproducibility documentation](https://docs.vllm.ai/en/stable/usage/reproducibility/). Its [batch-invariance feature](https://docs.vllm.ai/en/latest/features/batch_invariance/) offers a diagnostic direction, but support and performance for this exact NVFP4 Marlin/FlashInfer model build have not been tested.

AIME 2024 warming cost about 59 seconds per trial and did not remove the answer-arrival tail. Its causal speed benefit is not established without a paired comparison. The selected final v2 defaults therefore restore only the cheaper 30-stream × 32-token warmup; the complete warmed v1 source/protocol and every warmed outcome are preserved. No solving, grading, quantization or token-budget change accompanies that default revert.

Optional GPU/CPU/engine profiling was disabled. Required timing/token/verdict evidence supports this decomposition; it cannot rule out every transient hardware or KV event. Standard server logs, separately saved in [server-evidence.json](server-evidence.json), provide coarse throughput/KV observations. Client enqueue and submission medians are far smaller than the multi-second idle gaps; the maximum local wait can include intentional per-question serialization behind a wrong verdict.

[Structured variance evidence](variance.json), [five-trial visual report](README.md).

The earlier identical-seed trace check is direct evidence that request seeds alone
do not freeze output here: [saved comparisons](same-seed-divergence.json) compare
three scored earlier trials on the same server/source with equal request payloads
and equal prompt IDs. Q01 diverges after 35 output tokens between the first and
second scored trials; Q23 diverges after 57. Even the second and third trials,
which share Q01's complete observed prefix, diverge on Q23 after 4,876 tokens.
This rules out treating fixed seed as an identical-output control. It does not
isolate which numerical or scheduler operation caused the divergence.
