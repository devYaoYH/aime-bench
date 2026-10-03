# NVFP4 30×1 with 16K first pass, and CPU profile review

The requested experiment reached **18 verified correct questions in 106.570 s**.
It finished successfully and released the GPU. There were 30 first-pass generation
requests, no continuations, and 20 completed grader checks: 18 correct, two wrong.
The prior BF16 30×1/8K control reached the target in 71.135 s. This trial changes
both model quantization and the first-pass budget, and does not isolate their
individual effects or sampling variation.

Configuration: `src.attempt_runners.speedrun_v1` at commit `c469b02`,
`r0b0tlab/VibeThinker-3B-NVFP4`, ModelOpt FP4/Marlin on the A100, BF16 activations
and KV, 95% GPU-memory utilization, 65,536-token total context, 16,384-token
generation budget for both first and later segments, barrier schedule, target 18,
at most four requests per question. Temperature 0.8, top-p 0.95 and seed 20261003
match the control. Canonical solving code was not changed.

Median fresh TTFT was 0.244 s. Official VRAM peaked at 78,502 MiB (76.66 GiB);
including initialization it peaked at 78,604 MiB. Occupied KV peaked at 8.64% of
the reserved pool, with no waiting requests, preemptions, or OOM messages.

The grader's first pickup was at 4.774 s; checks used 60.002 s and idle gaps
between checks totaled 41.793 s. The run reached 16 correct at 63.067 s, then
17 at 96.147 s and 18 at 106.570 s. The last two winning candidates were observed
at about 93.13 s (Q29) and 103.56 s (Q7); those streams delivered 12,579 and
13,768 token IDs by cancellation, respectively. These token counts are censored,
not counts to the first candidate. The late tail aligns with waiting for new
candidates. Official runner CPU was 52.606 CPU-s; event-loop mean/max delay was
1.21/58.42 ms. This run does not show the severe client delays seen at 30×4.

## Previous CPU profiles

Measured official CPU usage, not additional wall time to add to inference:

| Previous setting | Runner CPU | Trace serialize/write/flush | JSON decode | Candidate parse | CPU outside timed sections |
| --- | ---: | ---: | ---: | ---: | ---: |
| Canonical 95% coverage | 42.536 s | 6.565 s | 2.351 s | 1.901 s | 31.407 s |
| BF16 30×1 | 34.401 s | 5.409 s | 1.952 s | 1.467 s | 25.400 s |
| BF16 30×2 | 55.593 s | 8.866 s | 3.266 s | 2.317 s | 40.871 s |
| BF16 30×4 | 94.135 s | 16.284 s | 6.196 s | 4.602 s | 66.538 s |

The remainder subtracts all recorded synchronous CPU sections, including small
artifact/GPU/metrics sections omitted from the table. It includes HTTP/asyncio,
profiling bookkeeping, stream handling and other unmeasured work; it is not
attributable solely to profiling overhead. Roughly 70% of runner CPU remains
outside the current detailed timers.

At 30×4, candidate-local queue delay had median/max 13.08/88.53 ms. The gap from
client verification start to grader submission was 151.71 ms median and
1,127.10 ms maximum; grader answer to client receipt was 29.15/117.67 ms. At 30×1,
the corresponding medians were 0.48 ms, 3.77 ms and 1.49 ms, with maxima
1.72 ms, 7.27 ms and 4.54 ms. These timestamp gaps include scheduling and transport
on both client and service, so they do not isolate a single function. Event-loop
maximum delay increased from 66.56 ms at 30×1 to 616.20 ms at 30×4.

## Optimization priorities

1. **Buffer stream trace writes.** `_runtime_v1.append_json` flushes every SSE
   event synchronously on the inference event loop. The 30×4 run flushed 607,441
   events. Keep exact token/candidate handling immediate, but batch trace flushes
   by count/time and flush on completion or cancellation. Preserve prompt/output
   token records and verification/solved durability; a bounded buffered trace has
   a different crash-loss window and needs explicit handling.
2. **Aggregate profiler meters after streams finish.** Each per-token timing and
   counter recursively updates rollout, question and attempt meters. Retain
   per-rollout timings, then merge totals at completion rather than updating all
   three dictionaries per token. Event-loop/engine sampling remains independent.
   Ensure cancelled streams are aggregated exactly once; continuation rounds
   must not recount prior measurements.
3. **Profile the remaining HTTP/stream loop before changing parsing.** Candidate
   parsing is only about 4–5% of official runner CPU. It already scans new text
   incrementally, although growing string copies and incomplete-line tails are
   further candidates. Prefer a function-level CPU profile or offline async replay
   to determine how much of the remaining cost is HTTP line decoding, context
   managers, counters, or string handling. Keep detector behavior at chunk
   boundaries unchanged. GPU sample-window scans and final artifact writes are
   much smaller costs in these records.

Reducing runner CPU is most likely to help overloaded fanout and verification
dispatch. At 30×1 the measured dispatch gaps are milliseconds; these changes
cannot be assumed to remove the tens of seconds spent waiting for candidates or
the serial grader's 54-second minimum.

## Local replay evidence

A saved BF16 30×4 Q09 rollout (5,987 SSE events) was replayed locally through trace
serialization, JSON decode, token-ID collection and the unchanged candidate
detector. Seven unprofiled repeats per variant produced these median CPU times:

| Variant | Median CPU |
| --- | ---: |
| Current per-event flush, three meter levels | 93.7 ms |
| Flush every 128 chunks | 79.8 ms |
| Leaf meters without parent propagation | 81.6 ms |
| Both changes | 67.8 ms |
| Both buffering and disabled meter timings | 55.6 ms |

Combining buffering and deferred aggregation reduced replay CPU by **27.7%**.
Every variant produced identical extracted candidates and output token IDs. A
separate cProfile replay placed recursive `Meter.observe` first by self time,
followed by replay-loop work, file flushes and JSON encoding. This is a local
macOS benchmark without network, GPU or concurrent async scheduling; it is not a
remote throughput measurement or an estimated end-to-end speedup. The leaf-meter
benchmark omits parent aggregation; an implementation would still need to merge
final totals. No production optimizations were applied during the NVFP4 run.

See [cpu_replay_audit.json](cpu_replay_audit.json) for repeat timings,
cProfile output and per-run dispatch gaps, and [analysis.json](analysis.json) for
the new attempt's derived metrics. [config.json](config.json) contains its launch
command and profile checksum. Full streams/audits/logs remain remote.

Attempt: [20261003T212812.156610Z](../../../attempts/20261003T212812.156610Z/README.md).
