# NVFP4 FlashInfer development comparison

The second and third warmed repetitions reached 18 verified correct answers in 64.453 and 64.399s, faster than the 71.135s BF16 reference. The first was 98.327s. These observations prompted a separately declared validation batch, which failed its three-for-three gate (87.356 / 59.316 / 86.574s). They do not establish reliable replication.

| Trial | Attempt | First 18 (s) | Initial median TTFT (s) | Grader service (s) | Grader idle gaps (s) | Wrong checks | Generation requests |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20261003T233132.790330Z | 98.327 | 3.510 | 54.002 | 37.824 | 0 | 45 |
| 2 | 20261003T233317.639206Z | 64.453 | 0.198 | 57.002 | 2.611 | 1 | 43 |
| 3 | 20261003T233424.568380Z | 64.399 | 0.240 | 57.002 | 2.473 | 1 | 43 |

This changes only the attention backend relative to the standard NVFP4 profile: FlashInfer attention, Marlin NVFP4 weights, BF16 activations and KV, 95% GPU envelope, 64K total context and 16K per-request generation ceiling. Runner controls remain reference seed 20261003, 30×1 barrier, 8K initial output, then up to 16K additional output per continuation, temperature 0.8, top-p 0.95, and a cap of four generation requests per question including continuations. Benchmark mode disables optional profiling and buffers client traces.

All 90 initial request payloads match the original after model substitution. Every trial has 30 question records, 18 distinct first-solved verdicts, matching target timestamps, no per-question cap violation, and clean source provenance. The driver exited 0 and its workers were gone before the independent validation launch. All outcomes are retained; prefix cache resets succeeded before each short warmup, with a fresh grader per attempt.

The first trial had a 3.510s conventional median initial TTFT even though short warmup completed. Later medians were 0.198/0.240s. Backend initialization included a 23.19s compilation-cache miss before official timing; sampling JIT warnings appeared during short warmup. The precise cause of the first official TTFT spike is unproven. The warmup covers the configured sampling path and 30×32 output tokens, but uses short synthetic prompts.

The faster runs needed 13 continuations and one incorrect check, versus neither in the original. They improved by supplying correct candidates earlier: idle gaps dropped to about 2.5s despite 57s grader service. Changed attention math can change sampled trajectories; these time-to-18 observations do not establish a pure decoding speedup or a reasoning-quality advantage.

A separate four-trial validation protocol was committed as configs/experiments/vibe-nvfp4-flashinfer-validation-v1.json before its launch: first trial settles a new owned server, then all three scored trials must beat 71.134766841s. Prefix caches reset before every attempt and every outcome remains visible. This tests fixed-seed warmed-server repeatability and reports settling/startup cost separately.
