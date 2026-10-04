# Core v1.1: five declared seeds

**All five trials reached 18 verified correct. The batch median did not improve:**
**77.652s** versus core v1's **77.277s** (+0.375s, +0.49%). Two historical
same-seed comparisons were faster and three slower. V1.1's range was
**60.906–92.096s**, versus **62.783–82.492s** for v1. These five trials do not
establish a consistent speedup; keep core v1 as the measured submission policy.

| Seed | Core v1 (s) | V1.1 (s) | V1.1 − v1 (s) | Requests v1 / v1.1 | Wrong checks v1 / v1.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261011 | 71.321 | 77.652 | +6.331 | 45 / 30 | 4 / 0 |
| 20261012 | 77.277 | 80.447 | +3.170 | 43 / 30 | 2 / 2 |
| 20261013 | 82.492 | 76.453 | -6.038 | 43 / 30 | 0 / 0 |
| 20261014 | 81.102 | 92.096 | +10.994 | 45 / 30 | 1 / 0 |
| 20261015 | 62.783 | 60.906 | -1.877 | 30 / 30 | 0 / 0 |

The median paired difference was **+3.170s**. V1.1 issued **150 generation
requests versus 206** (27.2% fewer), but recorded **1,130,101 client-observed
output IDs versus 1,099,511** (2.8% more). Removing the 8K boundary reduced request
count without reducing observed decoding work. Client output IDs are partial
observations at cancellation, not a complete GPU-compute measurement.

**No fresh retry occurred in any scored trial.** Each attempt reached 18 using
its 30 initial long rollouts; all 90 winning verdicts came from rollout one.
Across the series, **73 of 150 initial generations** exceeded 8,192 observed
output tokens. This tests the longer uninterrupted initial trajectories, but
provides no execution evidence that eager fresh retries reduce latency. Their
four-request cap and sequencing are verified by offline policy tests.

## What the grader timeline shows

V1.1 completed **two wrong checks**, versus seven for the historical v1 controls.
That reduced serial grader service by roughly 15s across the five attempts, but
did not produce a consistent latency gain. The slowest v1.1 trial had no wrong
checks and **32.055s of idle between grader jobs**; the fastest also had no wrong
checks but only **2.434s** of later idle. Answer availability and ordering remain
material to reaching the eighteenth positive verdict.

| Seed | Completed grader service (s) | First grader pick after start (s) | Idle between jobs (s) |
| --- | ---: | ---: | ---: |
| 20261011 | 54.002 | 8.895 | 14.755 |
| 20261012 | 60.003 | 5.644 | 14.799 |
| 20261013 | 54.002 | 4.368 | 18.082 |
| 20261014 | 54.002 | 6.039 | 32.055 |
| 20261015 | 54.002 | 4.470 | 2.434 |

Service, first-pick delay and later idle nearly reconstruct the client target
time, with small HTTP/settlement differences. This decomposition describes the
observed runs; it does not isolate the cause of changed trajectories or timings.

## Scope and controls

The [five-seed protocol](../../../runner_final/five_seeds_core_v1_1.json) was
committed before launch. Seeds 20261011–20261015 were each run once, in order,
with all outcomes retained and no replacement seeds or unscored settling runs.
AIME 2025 is development data. Comparisons use historical same-seed core v1
controls, not interleaved measurements or held-out transfer. Identical seeds do
not force identical tokens under different serving budgets and batch evolution.

Both batches use one inference-server lifetime, resetting the prefix cache
before each trial, starting a fresh three-second serial grader, and running
30×32-token arithmetic warmup. V1.1 also counts each served chat prompt through
`/tokenize` before scoring. Startup, warmup, tokenization, cancellation settlement,
trace flush and cleanup are excluded from `time_to_target_s`; the endpoint is the
eighteenth distinct positive grader verdict. No Nsight profiler was active.

V1.1 changes the initial 8K cap to **65,536 total context tokens including the
prompt**, removes the round barrier, and replaces continuations with fresh-only
retries. Actual prompt lengths were **204–944 tokens**, so requests allowed
**64,592–65,332 output tokens**. Incorrect verdicts leave live generation running.
After a generation ends and all its candidate checks fail, a new sample may
start, with four requests total per question, one active generation per question
and at most 30 overall. Solved slots do not create siblings for other questions.

Preserved controls: adherence_v1 prompt, temperature 0.8, top-p 0.95, seed stride
four, benchmark buffering, NVFP4/Marlin weights, BF16 activations/KV, FlashInfer,
95% allocation, 64K total context, runtime package versions and serial grader.
The only serving-profile setting change is `max_new_tokens: 16384 → 65536`.
This combines length and scheduling changes; it is not an isolated barrier test.

## Evidence and reproduction

- [Summary](summary.json), [artifact audit](analysis.json), [comparison CSV](comparison.csv),
  [launch inputs](config.json) and [periodic service observations](service_metrics.json).
- All 150 saved requests match their corresponding historical control prompt,
  sampling and request structure except the declared output cap. Warmup payloads
  match exactly. Saved prompt IDs match the served prompt reservations; all
  requests stay inside total context, request and concurrency budgets. All
  first-solved times match positive grader verdicts. Every attempt recorded
  clean tracked source and the same solver, prompt and serving-profile hashes.
- Forty standard server-log observations show at most 30 running requests,
  zero waiting, and at most 7.4% KV-cache usage. No OOM or error lines occurred.
  These periodic observations are not continuous GPU telemetry or measured
  v1.1 roofline counters. All owned services exited; the A100 returned to idle.
- Twelve focused checks passed locally and on the remote before launch; fourteen
  passed locally after adding token-artifact audit checks. All five imported
  metadata records validate. Both v1 and v1.1 manifests still verify.
- Source: `bae54382c52854ebdbb9dab5583fb0c150dadda2`.
  V1.1 manifest SHA256: `324126d7fa66068dd47062457af156615f8763bae62ba8443da39191574a1ab7`.
  Prompt SHA256: `26b591c39bcf55f4c94f5359dcc90d3c5626a524478eee1c5ce44b5038162364`.
  Profile SHA256: `ad6558ace0443f44110b8252b9bdae61d2d7b47be5ef1b79e561713174fefd5f`.

The batch ran from **2026-10-04T08:33:58.608+00:00** to
**2026-10-04T08:42:06.349+00:00** in tmux session `core-v1_1-five-bae54382`, using a
clean detached checkout at `/home/azureuser/aime-bench-v1_1-bae54382`:

```bash
~/.venvs/vllm/bin/python -m runner_final.validate_v1_1 \
  --batch core-v1_1-five-seeds-20261004T083800Z \
  --grader-python /home/azureuser/aime-bench/grader/.venv/bin/python
```

The separate committed profile was provisioned beside the model weights;
`vllm-flashinfer.yaml` was preserved. The versioned profile and manifest retain
preparation-time annotations; their bytes were not refreshed after measurement.
Rerunning the command creates additional trials outside this five-seed report.
Reproduce the saved-evidence audit locally with:

```bash
.venv/bin/python -m src.experiments.analyze_core_v1_1 \
  --batch core-v1_1-five-seeds-20261004T083800Z
```

Compact evidence totals about **19.8 MB** before compression and is versioned
under `attempts/`. Full SSE, grader audits and service logs remain in the remote
checkout; raw logs are excluded from Git. The existing final evidence packet's
five-seed figure continues to describe the selected core v1 batch.
