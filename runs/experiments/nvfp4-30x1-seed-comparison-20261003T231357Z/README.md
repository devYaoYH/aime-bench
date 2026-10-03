# NVFP4 declared-seed comparison

All four seeds reached 18 verified correct answers, but none reproduced the original 71.135s result. Median time was 116.314s; fastest was 94.591s. No outcome was discarded or selected for validation.

| Seed | Attempt | First 18 (s) | Initial median TTFT (ms) | Grader service (s) | Idle between checks (s) | Wrong checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20261004 | 20261003T231454.883045Z | 118.188 | 180.6 | 63.002 | 48.576 | 3 |
| 20261005 | 20261003T231656.190953Z | 94.591 | 228.7 | 54.002 | 35.266 | 0 |
| 20261006 | 20261003T231833.718658Z | 126.428 | 200.5 | 57.002 | 62.831 | 1 |
| 20261007 | 20261003T232043.406577Z | 114.441 | 229.0 | 57.002 | 52.536 | 1 |

Controls: NVFP4 Marlin weights, BF16 activations/KV, 95% GPU memory envelope, 64K total model context, 8K initial output then up to 16K additional output per continuation, 30×1 barrier, four generation requests per question including continuations, temperature 0.8 and top-p 0.95. Benchmark mode buffers client traces and disables optional profiling. Source/config hashes and server command are in config.json. One owned warmed inference server served the batch; prefix caches were successfully reset before every normal short warmup, with a fresh grader per attempt.

All 120 initial request bodies match the original after the declared model and seed substitutions. Validation found 30 question records, at most four requests per question, and 18 distinct solved timestamps per attempt. Median TTFT uses the first request for each question. Driver exited 0 and all owned GPU workers/listeners were gone on subsequent inspection.

An EngineDeadError at 23:22:41 UTC occurred during the driver's intentional SIGTERM shutdown after the final target was reached, alongside resource teardown. It was not an inference-time error. Full server/SSE logs remain remote and are excluded from Git.
