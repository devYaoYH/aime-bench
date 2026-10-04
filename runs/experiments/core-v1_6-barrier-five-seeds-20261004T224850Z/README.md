# V1.6 initial coverage barrier and pool: five matched seeds

All five declared seeds are scored, including the first. Failures and unmet targets remain; no replacement or settling trial is permitted. AIME 2025 is development data.

Compared with the latest refactored-v1 batch, the observed worst run fell from 128.744s to 101.123s; sample SD fell from 26.475s to 15.049s and the range narrowed from 66.626s to 37.678s. Mean latency fell from 82.379s to 76.951s, while the median rose from 73.568s to 77.498s. This is encouraging evidence for a smaller observed tail under the combined policy. Original v1 and v1.5 had tighter spreads, and the median is essentially unchanged versus original v1. Five historical trials do not isolate the barrier or establish a general reduction in population variance.

| Runner | Reached 18 | Median | Mean | Sample SD | Range |
| --- | ---: | ---: | ---: | ---: | --- |
| original_v1 | 5/5 | 77.277s | 74.995s | 8.083s | 62.783–82.492s |
| refactored_v1 | 5/5 | 73.568s | 82.379s | 26.475s | 62.118–128.744s |
| v1_5 | 5/5 | 77.352s | 75.633s | 9.393s | 65.015–85.464s |
| v1_6 | 5/5 | 77.498s | 76.951s | 15.049s | 63.445–101.123s |

| Seed | Original v1 | Refactored v1 | v1.5 | v1.6 | Initial barrier | Fresh / continued | Wrong checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261011 | 71.321s | 70.881s | 65.015s | 77.498s | 57.256s | 54 / 13 | 3 |
| 20261012 | 77.277s | 76.586s | 77.352s | 77.512s | 57.900s | 50 / 14 | 2 |
| 20261013 | 82.492s | 73.568s | 85.464s | 65.176s | 58.681s | 50 / 14 | 1 |
| 20261014 | 81.102s | 128.744s | 83.537s | 101.123s | 56.563s | 51 / 14 | 0 |
| 20261015 | 62.783s | 62.118s | 66.795s | 63.445s | 58.350s | 50 / 13 | 0 |

Winning trajectories across all trials: 80 initial streams, 9 continuations of initial streams, and 1 extra fresh sibling. The runner admitted 255 fresh samples and 68 continuations. Most wins therefore came from initial coverage or its continuation; direct fresh-sibling wins are sparse in this batch. Scheduling can also change the numeric decoding trajectory, so win counts alone cannot attribute a latency effect. The 101.123s trial's final two winners were Q7 and Q20, both initial-stream continuations; completed grader service was 54.002s, first pickup 6.030s and inter-query idle 41.090s.

## Policy and controls

Launch exactly 30 initial 8K streams. Hold freed slots idle until all initial streams and their queued grader checks settle. If 18 questions solve during that phase, stop without opening the pool. Otherwise open a 30-request pool: ready exact-token continuations first, then fresh samples for least-active unsolved questions with rotating ties. No further global barriers. Every fresh sample starts at 8K; at most four fresh trajectories per question, with at most one continuation per trajectory to the remaining cumulative 64K budget, clipped to the existing 64K total context. Correct verdicts cancel all streams for that question. Wrong verdicts provide no model feedback.

Same A100 80GB, NVFP4 Marlin, BF16 KV, FlashInfer, 95% memory, improved integer-answer prompt, temperature 0.8/top-p 0.95, five seeds 20261011–20261015, 3-second serialized grader and target 18 as the historical controls. The dedicated profile raises only the server output ceiling from 16K to 64K. All 150 initial request payloads match original v1 exactly. Continuation seeds use a disjoint +124 band for AIME 2025, while fresh sample seeds retain base + index×4 + sample; old v1 continued with the next HTTP-rollout seed. Continuation draws therefore differ as part of this extension, even with matched base seeds. One owned server serves all five trials; prefix cache resets, cheap arithmetic warmup and a fresh grader precede each timed trial. Optional profiling/GPU polls are disabled; required traces buffer in RAM and flush after official solving time. Per-trial wall safety timeout is 600s including initialization.

Time to 18 starts after dataset loading, service startup, prompt tokenization and warmup, immediately before generation scheduling. It ends at the eighteenth distinct correct verdict receipt, excluding cancellation settlement and trace flush. Startup answer-field validation is retained; correctness and analysis use grader verdicts, not gold answer fields.

## Verification and interpretation

Source commit `c0e461c59b043582b39fbf6804fe36e2f9bec745`; v1.6 manifest `15e943454e6f4840325f560bbfd9d18daafab090d24561e2abd6e1018a435017`. The canonical v1 manifest remains unchanged. The runtime audit checks all 30 initial request payloads against original v1, exact continuation prefixes/budgets/seeds, the fresh/HTTP caps, coverage settlement before pool admissions, peak concurrent requests and eighteenth first-solved timing. Inspect `analysis.json` for every audit and grader timeline.

Five paired historical seeds provide a practical comparison, not statistical equivalence or an isolated causal estimate. V1.6 changes the post-coverage scheduler, fresh-only cap, continuation budget and continuation seed mapping together. Historical v1.5 is an additional slot-pool comparison; it did not use this initial barrier. Use batch medians/ranges, not a single best timing, for claims.

The existing vLLM logs recorded 40 KV samples with a maximum of 11.2% across the server lifetime, and no OOM/preemption/eviction pressure warnings. Startup reported 71.87 GiB available KV memory and 2,093,344 token capacity (31.94 full-64K requests). Cleanup left zero GPU compute processes, 0 MiB used VRAM and both owned ports closed. Benchmark mode has no sampled peak VRAM or complete engine preemption counters. `server-evidence.json` contains coarse existing engine-log KV observations and cleanup checks; absence of logged warnings does not prove that no KV blocks were ever evicted.

Validation: 391 full offline suite checks passed from the staged source snapshot, then 19 focused policy/batch checks passed locally and remotely. All five imported attempt audits and metadata schemas pass. The repository-wide legacy metadata command stops on an existing v2.1 expression-valued extraction field that its scalar schema cannot accept; historical records were preserved and each new v1.6 record was validated separately. Only allowlisted artifacts are versioned (34.56 MiB before added local analysis); full SSE streams, grader audits and service logs remain remote/ignored.

## Reproduce

```sh
.venv/bin/python runs/experiments/core-v1_6-barrier-five-seeds-20261004T224850Z/reproduce_analysis.py
```

See `config.json`, `summary.json`, `analysis.json`, `paired-seeds.csv` and the linked attempt IDs for reviewable evidence.
