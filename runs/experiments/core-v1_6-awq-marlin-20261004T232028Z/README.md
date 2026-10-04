# AWQ / Marlin: one matched v1.6 AIME 2025 attempt

Seed 20261011: AABoyles/VibeThinker-3B-AWQ reached 18 verified correct in **100.163s**. Saved-request, timing, barrier and cap audits passed: **True**.

| Measure | NVFP4 / Marlin | BF16 / native linear | AWQ / Marlin |
| --- | ---: | ---: | ---: |
| Time to 18 (s) | 77.498 | 75.640 | 100.163 |
| Generation HTTP requests | 67 | 30 | 66 |
| Fresh samples | 54 | 30 | 52 |
| Wins: initial / continuation / fresh sibling | 15 / 2 / 1 | 18 / 0 / 0 | 15 / 3 / 0 |
| Initial 30-stream median TTFT (s) | 0.222 | 3.439 | 3.439 |
| First grader pickup (s) | 6.094 | 8.592 | 7.948 |
| Completed grader service (s) | 63.003 | 54.002 | 57.002 |
| Grader idle between checks (s) | 8.400 | 13.046 | 35.212 |
| Completed wrong checks | 3 | 0 | 1 |

Coverage/check barrier released at 58.077s; the continuously refilled 30-slot pool was exercised.

## Matched solving policy

The unchanged v1.6 runner launches 30×1 initial requests at 8,192 output tokens. Its first coverage barrier includes the queued grader checks. Afterward, a 30-slot pool prioritizes exact-token continuations, then least-active unsolved questions with rotating ties. Each question permits at most four fresh trajectories. Each trajectory gets at most one continuation for its remaining cumulative 65,536-output-token budget, clipped to the 65,536 total context including prompt. Continuations do not consume fresh-sample allowance. Correct verdicts cancel that question; the eighteenth distinct first-solved verdict stops the attempt. Wrong answers do not inject feedback.

All three deployments use seed 20261011, the same AIME 2025 prompt/dataset, improved system prompt, temperature 0.8/top-p 0.95, cheap 30-stream arithmetic warmup, fresh owned services, prefix-cache reset, serialized three-second grader, 95% memory, BF16 activations/KV and FlashInfer attention. AWQ uses four-bit/group-128 asymmetric weights and explicit Marlin linear kernels. NVFP4 uses ModelOpt FP4/Marlin; BF16 uses unquantized native linear dispatch. Server ceilings override model-card generation defaults to 65,536.

Official timing begins after service startup, dataset loading, tokenization and warmup, immediately before scheduling. --benchmark disables optional CPU profiling, engine polls and NVML sampling. Required request/token/timing/verdict evidence is held in RAM and flushed after official timing; flush/startup are excluded. Existing server logs provide coarse cache/throughput observations without additional timed polling.

## Provenance and limits

Source `f600cd0261318cc6f2ac50161b72043cb865aca6`; frozen v1.6 manifest `15e943454e6f4840325f560bbfd9d18daafab090d24561e2abd6e1018a435017`. All 30 initial payloads match the NVFP4 control after removing model identity. The original AWQ tokenizer files were retained: their file hashes differ due to reserialization, but vocabulary, special-token IDs, exact rendered IDs for all 30 reference prompts and decoded reference outputs match. The preflight explicitly requests plain token IDs for compatibility with the installed Transformers version. No GPU/scored attempt started during the earlier return-type preflight failure.

Model revisions and asset hashes are in config.json. Both quantized model cards name WeiboAI/VibeThinker-3B as base but do not pin their original base checkpoint/calibration data. This is a comparison of downloaded deployments, not an isolated quantization experiment. One seed does not establish repeatability. Correctness derives only from Boolean grader verdicts; existing startup answer-field validation is retained.

Maximum logged KV usage was 10.6%; 0 memory-pressure warnings were found. These are coarse log observations, not a sampled peak VRAM measurement or a complete eviction-counter audit. Runtime kernel/cache-capacity lines, throughput observations and post-run service/GPU cleanup are saved in server-evidence.json. Throughput samples average preceding intervals with changing concurrency and context lengths.

## Reproduce

```sh
.venv/bin/python runs/experiments/core-v1_6-awq-marlin-20261004T232028Z/reproduce_analysis.py
```

The deployment controls and v1.6 policy tests passed locally and remotely (26 tests). New metadata validates individually. The repository-wide legacy metadata command still fails on an older v2.1 expression-valued extraction field; historical records were preserved. Full SSE streams, raw grader audits, service logs and weights remain remote.
