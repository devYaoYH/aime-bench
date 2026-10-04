# One BF16/FlashInfer v1.6 AIME 2025 comparison

Seed 20261011: BF16 reached 18 verified correct in **75.640s**, versus **77.498s** for the matched NVFP4 deployment (-1.858s; -2.4%). All attempt audits passed: **True**. This is one scored run, with no replacement trial.

| Measure | NVFP4 / Marlin | BF16 / native linear |
| --- | ---: | ---: |
| Time to 18 | 77.498s | 75.640s |
| Generation requests | 67 | 30 |
| Fresh samples | 54 | 30 |
| First grader pickup | 6.094s | 8.592s |
| Completed grader service | 63.003s | 54.002s |
| Grader idle between queries | 8.400s | 13.046s |
| Completed wrong checks | 3 | 0 |
| Fresh request median TTFT | 0.177s | 3.440s |
| Continuation median TTFT | 0.19137765699997544 | None |
| Initial 30-request median TTFT (arithmetic median) | 0.222s | 3.439s |

## Runner and controls

Both use unchanged v1.6: initial 30×1 at 8K, coverage barrier including queued checks, then a 30-slot pool prioritizing exact-token continuations and least-active fresh samples with rotating ties. Each question permits four fresh samples; every fresh trajectory starts at 8K and gets at most one continuation to the remaining cumulative 64K output/context budget. Correct verdicts cancel the question, and 18 distinct correct verdicts stop the attempt.

The target was reached during initial coverage, before the pool opened.

Same A100 80GB, runtime versions, dataset provenance, improved prompt, temperature 0.8/top-p 0.95, base seed and continuation seed mapping, 3-second serialized grader, 95% memory, BF16 activations/KV and FlashInfer attention. Both server output ceilings and total context are 65,536. BF16 uses WeiboAI/VibeThinker-3B with quantization absent and automatic unquantized linear dispatch; NVFP4 uses r0b0tlab/VibeThinker-3B-NVFP4 with ModelOpt FP4 weights and Marlin linear kernels. Profiles and older results remain separate.

Each started a fresh owned server, reset prefix cache, ran cheap 30-stream/32-token arithmetic warmup, and started a fresh grader. The benchmark timer starts after initialization/tokenization/warmup, immediately before scheduling; time to 18 uses the eighteenth distinct first-solved verdict receipt. Optional profiling/engine polls/NVML sampling are disabled. Required client traces buffer in RAM and flush after official timing. The 600-second per-trial wall safety limit includes trial initialization after server readiness.

## Provenance and interpretation

The downloaded tokenizers are byte-identical and the architecture/RoPE settings agree. BF16 weight revision is 77bd2cced09193c8b9a59a32bd8577bbd1f3e01c; the NVFP4 artifact revision is 2fc0013974d1a466e6a5a11839f029d5aff34dc9. Its model card names the WeiboAI base but does not pin that base revision. Saved batch config includes these local model/config/tokenizer identities. This compares the downloaded deployments, including native linear versus Marlin inference math. One seed cannot establish a repeatable quantization speedup or separate decoding speed from changed reasoning/answers.

Source `2c1b9a0cbb54ab0385894511bac6be117e5a7f6d`; unchanged v1.6 manifest `15e943454e6f4840325f560bbfd9d18daafab090d24561e2abd6e1018a435017`. The audit checks all 30 initial request payloads after removing only model identity, the same policy controls, served prompt lengths/dataset hashes, exact continuation prefixes and budgets, first-solved timing, initial barrier settlement, and fresh/global request caps. Correctness uses grader verdicts; answer fields are retained only for existing startup validation.

Existing server logs show maximum KV usage 5.8% across startup/warmup/solve, with 0 logged pressure warnings. See server-evidence.json for runtime backend selection, cache capacity and cleanup. These are coarse existing log samples, not sampled peak VRAM or complete eviction counters. Throughput log observations include mixed active counts and censored requests, so they are diagnostic rather than a fixed-concurrency benchmark.

The 26-live-request endpoint log samples show 4,222.6 tok/s for NVFP4 and 3,332.4 tok/s for BF16. They average the preceding 10-second intervals, whose active counts/context trajectories differ, and are not a controlled fixed-concurrency throughput test. The initial-30 TTFT median was 0.222s versus 3.439s. The four logged Triton sampling JIT warnings occurred during cheap warmup (23:10:02–04 UTC), before official start at 23:10:06.470 UTC; they do not establish the cause of the later TTFT gap. No timed GPU/CPU profiling was enabled to attribute that gap.

BF16's slightly shorter solve time came with 54.002s completed grader service (zero wrong checks) versus 63.003s for NVFP4 (three wrong checks). BF16 first pickup was 2.498s later and grader inter-query idle was 4.645s longer; the saved clock decomposition accounts for the approximately 1.858s net advantage. BF16's final two winners were Q18 and Q9, from their initial streams. It reached the target before any continuation or fresh sibling was launched. That means this trial evaluates initial-coverage solving and the stop rule; it does not exercise the post-barrier pool on BF16.

All new BF16 metadata validates. The repository-wide legacy metadata command stops on the existing v2.1 expression-valued extraction field; older records were preserved. Reviewable imported artifacts total 3.20 MiB before analysis; full SSE streams/grader audits/service logs remain remote and ignored. Both owned ports were closed and no GPU compute processes remained after cleanup.

## Reproduce

```sh
.venv/bin/python runs/experiments/core-v1_6-bf16-flashinfer-20261004T230835Z/reproduce_analysis.py
```

The new BF16 profile/control checks and all v1.6 policy checks passed locally and remotely (17 tests). See config.json, summary.json, analysis.json, server-evidence.json and the referenced attempt for the full evidence.
