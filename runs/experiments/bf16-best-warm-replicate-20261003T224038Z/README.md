# Warmed BF16 best-run replication

Three fixed-seed benchmark repeats did not reproduce the 71.135-second reference. All 90 initial request payloads match the reference. The same owned server was reused, with successful prefix-cache resets before each trial and the ordinary 30-stream warmup before timing. No trial was dropped.

| Trial | Attempt | First 18 | Initial TTFT median | Wrong checks | Grader service | Grader idle |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [20261003T224137.318281Z](../../../attempts/20261003T224137.318281Z/summary.json) | 117.167s | 0.168s | 3 | 63.002s | 49.130s |
| 2 | [20261003T224337.444734Z](../../../attempts/20261003T224337.444734Z/summary.json) | 134.519s | 0.160s | 2 | 60.002s | 66.954s |
| 3 | [20261003T224555.051695Z](../../../attempts/20261003T224555.051695Z/summary.json) | 135.654s | 0.160s | 2 | 60.002s | 68.088s |

Median first-18 time was **134.519s**; **0/3** were at or below the predeclared 75-second reporting threshold. This threshold does not redefine the 71-second goal.

The reference used 54.002s of grader service, 12.076s idle between checks, and first pickup at 5.055s. These repeats primarily increased the answer-arrival idle tail, while initial TTFT stayed close to the reference’s 0.159s. They used 44/45/45 generation requests and two rounds; the reference needed only its 30 first-pass requests. This does not establish a hardware slowdown or compare full-model reasoning accuracy.

Controls: BF16 WeiboAI/VibeThinker-3B, original profile at 95% and 65,536 total context, 30×1 barrier, 8192 first-pass / 16384 later, four requests maximum per question, temperature 0.8, top-p 0.95, seed 20261003, target 18. Optional profiling disabled, client traces buffered until end. Each trial used a fresh grader. Prefix-cache reset routes were enabled on the owned loopback server; caches of weights and compiled kernels were retained.

Source commit: `98e7dff2f457a7bfaabb62993b1a12811d00b37b`. Driver local checks passed; exact initial-payload comparison passed for each trial. The durable job exited 0, closed its owned inference server, and yielded the GPU to the queued matched controls.

[Batch controls](config.json), [all trial summaries](summary.json). Full SSE streams, grader audits and shared service logs remain remote. Imported attempt JSON evidence totals about 21.3 MB.

The objective remains active: the original profiling/storage control and paired NVFP4 30×1 comparison are running next.
