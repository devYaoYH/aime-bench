# AWQ versus NVFP4: five paired v1.6 AIME 2025 seeds

AWQ reached 18 in **5/5** trials; median **88.835s**, range **61.211–100.163s**. NVFP4 reached 18 in 5/5 with median **77.498s** and range **63.445–101.123s**. AWQ was faster on 2/5 paired seeds; paired mean delta **+8.100s**. All saved policy/payload/continuation/barrier/cap audits passed: **True**.

| Seed | NVFP4 time to 18 | AWQ time to 18 | AWQ − NVFP4 | AWQ wrong checks | AWQ initial / continued / sibling wins |
| --- | ---: | ---: | ---: | ---: | --- |
| 20261011 | 77.498s | 100.163s | +22.665s | 1 | 15 / 3 / 0 |
| 20261012 | 77.512s | 61.211s | -16.301s | 0 | 17 / 1 / 0 |
| 20261013 | 65.176s | 94.217s | +29.041s | 2 | 15 / 3 / 0 |
| 20261014 | 101.123s | 80.826s | -20.296s | 2 | 15 / 3 / 0 |
| 20261015 | 63.445s | 88.835s | +25.390s | 2 | 16 / 2 / 0 |

| Summary | NVFP4 / Marlin | AWQ / Marlin |
| --- | ---: | ---: |
| Median | 77.498s | 88.835s |
| Mean | 76.951s | 85.051s |
| Minimum | 63.445s | 61.211s |
| Maximum | 101.123s | 100.163s |
| Sample standard deviation | 15.049s | 15.107s |

## Controls and scope

The completed seed 20261011 pilot is trial 1; only seeds 20261012–20261015 were subsequently launched. No seed was replaced or selected based on its result. The frozen v1.6 policy, 30-question initial 8K coverage/check barrier, subsequent 30-slot pool, four-fresh-sample allowance, one exact-ID continuation to the remaining cumulative 64K/context ceiling, improved prompt, sampling seeds/temperature/top-p, serialized three-second grader, dataset provenance and benchmark configuration match. Each new trace is independently audited against its corresponding NVFP4 seed.

AWQ uses AABoyles/VibeThinker-3B-AWQ (int4/group128, revision d32ba299f24c77d241832556d7cf5308699354b7) and explicit Marlin linear kernels. NVFP4 uses r0b0tlab/VibeThinker-3B-NVFP4 with ModelOpt FP4/Marlin. Both retain BF16 activations/KV, FlashInfer attention, 95% memory and 65,536 total context/output ceiling. Original AWQ tokenizer assets are retained; each trial's preflight compares vocabulary/special IDs, all exact reference prompt IDs and reference output decoding. Both quantized artifacts name the same WeiboAI base, but do not pin original base checkpoint/calibration data.

Five fresh AWQ server lifetimes, each prefix-reset and cheaply warmed. NVFP4 control used one shared server lifetime across five seeds. Official solve time excludes startup/warmup; residual compilation/cache state may still affect latency.

These are paired downloaded-deployment results on one machine, not a controlled causal estimate of quantization alone. Five seeds provide descriptive repeatability evidence, not a guarantee on unseen questions. BF16 has only one matched-seed run (75.640s), so it is not presented as a five-run distribution.

## Timing, memory and tail evidence

Official solving time begins after dataset loading/tokenization and cheap 30-stream/32-token arithmetic warmup. It ends at the eighteenth distinct first-solved grader verdict. Required traces buffer in RAM; final flush and service startup/cleanup are outside that clock. Optional CPU profiling, engine polling and NVML samples are disabled with --benchmark. A 600-second trial safety timeout includes initialization after server readiness.

The median of initial-30 TTFT trial medians was 0.239s for NVFP4 and 0.212s for AWQ. Completed wrong checks totaled 6 versus 7; grader inter-query idle totaled 71.901s versus 111.674s. The extra 39.773s of total inter-query idle versus only 3.002s of extra completed grader service accounts for most of AWQ's slower group mean. The per-trial clock decomposition and last-three winners are in analysis.json. These distinguish time spent grading wrong candidates from time waiting for candidate arrivals; they do not identify the causal source of changed reasoning trajectories.

All five server logs confirm Marlin linear kernels and FlashInfer attention. Maximum logged KV occupancy across the trials was 10.6%; 0 memory-pressure warnings were recorded. Per-server runtime Marlin/FlashInfer selection, available KV capacity, coarse KV-use/throughput logs and memory-pressure warnings are in server-evidence.json. No sampled peak VRAM or complete eviction-counter claim is made. Existing throughput intervals have changing request counts and context lengths and are not fixed-batch benchmarks. Post-batch GPU processes and owned ports are checked after all four new servers close.

Expansion driver source `b634e1a7cf4c2c256101be581b30dc8201d1ff11`; unchanged v1.6 manifest `15e943454e6f4840325f560bbfd9d18daafab090d24561e2abd6e1018a435017`. Pilot retains its original source commit. All new metadata is checked individually; the legacy global metadata command still fails on an older v2.1 expression-valued field. Raw SSE streams, grader audits, service logs and weights stay remote; only reviewed evidence is imported.

AWQ questions appearing most often among the last three of the 18 winners: Q7: 2/5, Q12: 2/5, Q18: 2/5, Q23: 2/5, Q29: 2/5. This describes late successful arrivals under the stopping rule, not the difficulty of questions that remained unsolved. Both models' frequencies and exact tail timings are in analysis.json.

## Reproduce

```sh
.venv/bin/python runs/experiments/core-v1_6-awq-five-seeds-20261004T232747Z/reproduce_analysis.py
```

All 29 focused deployment, batch and policy tests passed locally and remotely before the new trials. The runner manifests and historical profiles were preserved.
