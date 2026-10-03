# NVFP4 30×1 / 16K with benchmark mode

Reached 18 verified correct answers in **106.931 seconds**.
The matched profiling-on control reached 18 in 106.570s:
this trial shows no end-to-end speedup, despite lower runner CPU use.

| Measurement | Profiled control | Benchmark mode |
| --- | ---: | ---: |
| First 18 correct | 106.570s | 106.931s |
| Official runner CPU | 52.606 CPU-s | 37.509 CPU-s |
| Completed grader checks | 20 (18 correct, 2 wrong) | 19 (18 correct, 1 wrong) |
| Grader service time | 60.002s | 57.002s |
| Grader idle between checks | 41.793s | 45.808s |
| Fresh TTFT, conventional median | 0.244s | 0.209s |

The runner used **28.7% less CPU** during the official window.
This broad process measurement is retained without per-token profiling. It does
not establish that all of the difference was profiler overhead: output paths and
check counts differed. All 30 request payloads, seeds and prompt IDs match; all
30 observed output token sequences differ. Their median common prefix is
113 tokens. See [paired evidence](paired_comparison.json).

Sixteen correct arrived at 64.365s. Question 2 and question 23 supplied the last
two verdicts at 103.931s and 106.931s. The grader was idle for 45.808s between
checks, mostly waiting for these late candidates. This remains an answer-arrival
tail, rather than a demonstrated client telemetry bottleneck.

## Run controls and evidence

- Source commit: `7f345a821faa46176c135e202b0ce612b9e65f34`, clean remote checkout.
- Runner: `src.attempt_runners.speedrun_v2 --benchmark`; complete command in
  [config.json](config.json).
- NVFP4/Marlin on A100, BF16 activations/KV; profile unchanged at 95% memory
  utilization and 65,536 total context tokens. Generation budget is 16,384 tokens.
- AIME 2025, 30 questions, one initial rollout each, barrier schedule; target 18,
  four requests maximum per question. Exactly 30 first-pass requests were made;
  no continuations. 28 streams were cancelled and two completed naturally.
- Checks: 14 literal prose candidates and five boxed candidates. Candidate syntax
  alone does not identify a final answer; the runner retains streaming extraction.
- [Attempt summary](../../../attempts/20261003T215349.326138Z/summary.json),
  [first-solved timestamps](../../../attempts/20261003T215349.326138Z/solved.jsonl),
  [analysis](analysis.json), [intervention metadata](../../../attempts/20261003T215349.326138Z/metadata.json).

## Storage and instrumentation validation

Optional profiling was disabled: empty meter totals/counters, zero engine samples,
no event-loop or GPU polling; no `gpu.jsonl` or `inference_metrics.jsonl` was created.
Every rollout retains TTFT, generation start/end/latency, end-to-end latency,
request/response, and exact token records. All 18 first-solved events link to grader
answer timestamps and query IDs.

Buffered trace storage flushed 261,682 rows across
241 files in **1.943s**,
after official timing and service cleanup. Raw SSE payloads contained about
92.9 million characters. Peak runner RSS was
269.8 MiB. Cleanup took
4.211s, also outside time to target. Initialization
plus official attempt took 173.863s.

The service log reported a maximum KV utilization of 8.3%,
and contained no OOM messages. No sampled VRAM peak or preemption count is available
in this mode; the final GPU check confirmed memory returned to 0 MiB. Exit code was
0. Full SSE, grader audits and service logs remain on the remote. Imported JSON
artifacts total about 4.7 MiB; raw streams are excluded from Git.

The committed-source offline suite passed **134 tests**, using the existing ignored
baseline fixtures unmodified. The shared checkout's unrelated dirty v1 edits were
preserved. Metadata schema validation and per-rollout/first-solved evidence checks
passed after import.
