# Independent warmed NVFP4 FlashInfer validation

**Validation failed:** one of three scored trials beat the original 71.134766841s. Scored results were 87.356 / 59.316 / 86.574s. The 59.316s result is the new fastest observed attempt, but the complete batch contradicts reliable repetition at that speed.

| Trial | Role | Attempt | First 18 (s) | Initial median TTFT (s) | Grader service (s) | Grader idle gaps (s) | Wrong checks |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Settling | 20261003T233726.660316Z | 136.048 | 0.248 | 63.003 | 69.841 | 3 |
| 2 | Scored validation | 20261003T233946.145732Z | 87.356 | 0.171 | 57.002 | 25.657 | 1 |
| 3 | Scored validation | 20261003T234116.410095Z | 59.316 | 0.197 | 54.003 | 0.483 | 0 |
| 4 | Scored validation | 20261003T234218.148913Z | 86.574 | 0.199 | 54.002 | 27.739 | 0 |

The protocol was committed before launch in configs/experiments/vibe-nvfp4-flashinfer-validation-v1.json: one full settling attempt on a new owned server, followed by three scored trials; all three had to reach 18 distinct verified correct questions within 71.134766841s. Prefix resets succeeded before every normal short warmup. All outcomes are retained. Setup, the 136.048s settling attempt, final buffer writes and cleanup are outside the three scored solving windows; those costs are not hidden or claimed as a 59s cold end-to-end run.

Controls: reference seed 20261003, 30×1 barrier, temperature 0.8, top-p 0.95, 8K initial output then up to 16K additional output on later requests, four requests per question including continuations, NVFP4 Marlin weights, BF16 activation/KV, FLASHINFER attention, 64K total context and 95% envelope. Benchmark mode disables optional profiling and buffers client traces. Source and profile hashes plus actual server command are in config.json.

All 120 initial payloads match after model substitution. Local validation found 30 question records per trial, at most four generation requests per question, 18 distinct grader-linked solved timestamps, and matching target times. Driver exited 0; no owned workers, inference/grader listeners or GPU memory remained afterward. See validation.json for the explicit failed gate. Full SSE/server/audit logs remain remote.

TTFT was normal even in the slow trials, at 0.171–0.248s. The 59.316s result used 54.003s of serial grader service, began checking at 4.829s and had only 0.483s idle gaps. The other scored trials spent 25.657/27.739s waiting between checks. Thus late correct-candidate arrivals remain the main source of variation; this batch does not establish a throughput or reasoning-quality cause. It also disproves a simple claim that one full settling run always produces 64s outcomes.

The next distinct runner experiment should test suspending generation while a question has a candidate awaiting verification, then resume exact observed token IDs if wrong. Existing policies remain unchanged, the request cap includes resumptions, and any benefit must be tested rather than inferred from summed overlapping stream durations.
