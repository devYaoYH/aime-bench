# Paired NVFP4 30×1 replication

Three NVFP4/Marlin benchmark trials with an 8K first pass reached 18 but did not reproduce 71.135s. Each used a fresh grader and an owned warmed server, with a successful prefix-cache reset before the usual 30-stream warmup. All initial request payloads match the BF16 reference after substituting the model ID.

| Trial | Attempt | First 18 | Initial TTFT median | Wrong checks | Grader service | Grader idle |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [20261003T225721.655059Z](../../../attempts/20261003T225721.655059Z/summary.json) | 115.403s | 0.218s | 2 | 60.002s | 51.271s |
| 2 | [20261003T225920.250528Z](../../../attempts/20261003T225920.250528Z/summary.json) | 96.669s | 0.199s | 2 | 60.002s | 31.999s |
| 3 | [20261003T230059.890071Z](../../../attempts/20261003T230059.890071Z/summary.json) | 96.667s | 0.198s | 2 | 60.003s | 31.997s |

Median first-18 time: **96.669s**. **0/3** reached the predeclared 75s reporting threshold; the 71s goal remains unmet. Warm repeats two and three differed by only 0.002s, but both required continuations.

Initial TTFT was 0.198–0.218s, versus 0.160–0.168s in the matched BF16 benchmark repeats. These controls do not show the expected NVFP4 TTFT advantage. Generation trajectories and candidate timings differed, so the lower NVFP4 first-18 median (96.669s versus BF16 134.519s) is not a pure kernel-performance or reasoning-quality estimate.

Source commit: `4995023`. The original model profile was retained at 95% and 65,536 total context. Request cap four, temperature 0.8, top-p 0.95, seed 20261003, target 18. No optional GPU/engine/client profiling; server logs remain remote. A separate active health observation at 23:01:58 UTC showed 76°C, 1395 MHz SM clock, 1512 MHz memory clock and SW power capping, with no hardware or thermal slowdown. The SM clock was about 1% below its 1410 MHz maximum; this does not account for the answer-arrival tail.

[Controls](config.json), [all trial summaries](summary.json). Exact request/token/verdict evidence and metadata passed offline import checks. Raw SSE, grader audits and service logs are excluded from Git.
