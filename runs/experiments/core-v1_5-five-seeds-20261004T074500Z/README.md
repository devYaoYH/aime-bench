# Core v1.5: five declared seeds

**All five trials reached 18 verified correct. V1.5 did not improve the batch
median:** 77.352s versus core v1's 77.277s. Its range was 65.015–85.464s, compared
with 62.783–82.492s for v1. Only seed 20261011 was faster than its historical
same-seed control; all four added trials were slower. Keep core v1 as the measured
submission policy and retain v1.5 as an unsuccessful scheduling optimization.

| Seed | Core v1 (s) | V1.5 (s) | V1.5 − v1 (s) | Requests v1 / v1.5 | Wrong checks v1 / v1.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261011 | 71.321 | 65.015 | -6.306 | 45 / 73 | 4 / 2 |
| 20261012 | 77.277 | 77.352 | +0.075 | 43 / 69 | 2 / 2 |
| 20261013 | 82.492 | 85.464 | +2.972 | 43 / 70 | 0 / 2 |
| 20261014 | 81.102 | 83.537 | +2.435 | 45 / 70 | 1 / 1 |
| 20261015 | 62.783 | 66.795 | +4.013 | 30 / 72 | 0 / 0 |

The median paired difference was **+2.435s**. Across all five trials, v1.5 issued
**354 generation requests versus 206** (+71.8%) and recorded **1,455,976 client-
observed output token IDs versus 1,099,511** (+32.4%). Both policies incurred seven
completed wrong grader checks in total. Filling freed slots worked as implemented,
but it did not consistently reduce time to the eighteenth verified answer.
Client-observed token totals are not a complete GPU-compute measurement.

The original single run's 6.306s gain largely aligned with two fewer wrong checks.
That advantage did not carry through the extension: seed 20261013 incurred two
wrong checks versus none for its control. Equal total wrong checks and mixed
per-seed timing are reasons to report the full series rather than the first draw.

## Experimental scope

The [initial single-trial report](../core-v1_5-eager30-single/README.md) is retained.
After seeing it, the user requested the remaining four seeds from the existing
core v1 validation sequence. The [extension protocol](../../../runner_final/five_seeds_core_v1_5.json)
was committed before launch. The original seed 20261011 was included once and
was not reexecuted. Seeds 20261012–20261015 were each executed once, in order,
with no parameter changes, replacement seeds, or unscored settling trials.

Every v1.5 trial used a separate fresh owned inference server and fresh grader,
matching the first v1.5 execution. The historical core v1 controls shared one
inference-server lifetime with cache resets and fresh graders. Startup, arithmetic
warmup, service cleanup and trace flush are outside `time_to_target_s`. The solver,
serving profile and all initial payloads are matched; server lifetimes differ.
These are historical same-seed comparisons, not interleaved causal validation.
AIME 2025 remains development data; this is not a held-out transfer result.

The frozen v1.5 policy is unchanged: 30 initial one-per-question 8K requests,
a fixed ceiling of 30 active generations, immediate admission of ready exact-ID
continuations before fresh starts on least-active unsolved questions, and four
requests per question. Later fresh or continuation requests allow 16K additional
output, clipped to remaining context. Generation continues while verification is
pending. Correct verdicts cancel siblings; the eighteenth distinct positive
verdict stops admission and supplies the measured target time.

Preserved controls: adherence_v1 prompt, temperature 0.8, top-p 0.95, seed stride
four, 30×32-token arithmetic warmup, benchmark buffering, NVFP4/Marlin weights,
BF16 activations/KV, FlashInfer, 95% allocation, 65,536-token total context and
three-second serial grader. No FP8 KV profile was used.

## Evidence and validation

- [Summary](summary.json), [reproducible artifact audit](analysis.json),
  [seed comparison CSV](comparison.csv), and [declared launch inputs](config.json).
- All 150 initial request payloads match their historical same-seed controls.
  Saved parent token IDs, per-request seeds, sampling and additional-token budgets
  passed offline audits. Every trial peaked at 30 active client generations and
  stayed within four requests per question. The maximum concurrent streams for
  one question was three; no extra per-question concurrency cap was introduced.
- [Service observations](service_metrics.json) for the four new trials: periodic
  logs observed at most 30 running requests, zero waiting, at most 10.0% KV cache
  usage, and no OOM/error lines. Optional GPU telemetry was disabled. All owned
  services cleaned up; the host returned to zero GPU memory usage.
- Solver manifest: `d9357052a556cb0271a23acaf44733706dfc9d0a073acf351e5ccd9d4664e017`,
  identical for all five attempts. Prompt SHA256:
  `26b591c39bcf55f4c94f5359dcc90d3c5626a524478eee1c5ce44b5038162364`.
- Initial trial source: `ebb1fb8e1a2379276468a2834943c3b6aaed300e`.
  Extension driver/source: `884b2a24e9a7a8aee2c0d865e86fbd69e2066216`.
  Every scored attempt recorded clean tracked source; unrelated edits were preserved.
- Nine focused driver/policy checks passed. The clean committed-source full suite
  passed 233/235 checks; the two legacy salvage checks need ignored raw trace
  fixtures. All five imported attempt metadata records validate.

The extension ran from `2026-10-04T07:44:54Z` to `2026-10-04T07:54:57Z` in tmux
session `core-v1_5-five-884b2a24`, using this command in a clean detached checkout:

```bash
~/.venvs/vllm/bin/python -m runner_final.validate_v1_5 \
  --batch core-v1_5-five-seeds-20261004T074500Z \
  --grader-python /home/azureuser/aime-bench/grader/.venv/bin/python
```

The driver retains the existing first trial and executes the remaining four;
rerunning that command would create additional repeats, outside this report.
To reproduce the offline audit from saved evidence:

```bash
.venv/bin/python -m src.experiments.analyze_core_v1_5 \
  --batch core-v1_5-five-seeds-20261004T074500Z
```

Full SSE, grader audits and service logs remain in the remote detached checkouts.
Compact request/response, exact-token, verdict, allocation and timing evidence is
versioned under `attempts/` and available to the local viewer.
