# Pending-verdict suspension, three retained repeats

The new v5 policy reached 18 verified correct questions in **95.800 / 99.235 / 83.064 seconds**, median **95.800s**. None beat the original 71.135-second reference. The final runner therefore freezes the faster measured NVFP4 FlashInfer 30×1 barrier policy, whose best is 59.316s and independent scored median is 86.574s.

These three runs use seed 20261003, a private reused server, prefix-cache clearing, fresh serial graders, benchmark storage and source commit `ae9c37e`. Their initial 30 payloads match the original reference after model substitution. Each reaches 18 distinct grader-linked correct verdicts and preserves the four-generation-request cap. v5 changes free-slot scheduling, pauses all streams for a question with a pending prospective verdict, and resumes exact observed prefixes after a wrong verdict. Its output budget is 8K → 16K cumulative, unlike the barrier reference's later 16K additional output. This is a policy comparison, not an isolated throughput measurement.

| Trial | First 18 | Grader checks (wrong) | Grader service | Grader idle | Generation requests | Paused requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 95.800s | 18 (0) | 54.002s | 38.134s | 78 | 28 |
| 2 | 99.235s | 19 (1) | 57.002s | 37.396s | 77 | 29 |
| 3 | 83.064s | 18 (0) | 54.002s | 24.528s | 74 | 30 |

Controller exit was zero; its owned controller, API server and EngineCore were absent afterward and GPU memory usage was 0 MiB. Optional GPU/KV polling was disabled, so no sampled VRAM peak or complete eviction counter is available. Raw SSE/service/grader logs remain in `/home/azureuser/aime-bench-pending-v5-ae9c37e`; required JSON/token/verdict records are versioned locally.

[Configuration](config.json), [all results](summary.json), [selected final runner](../../../runner_final/README.md).
