# Warmed BF16 replication with original profiling settings

Three fixed-seed repeats using v2 with the original profiling and immediate trace writes did not reproduce the 71.135-second reference. All 90 initial payloads match. These are matched controls via the v2 integration, not execution of the original source snapshot. The server was owned/reused, prefix caches successfully reset between trials, and each trial warmed 30 streams and started a fresh grader.

| Trial | Attempt | First 18 | Initial TTFT median | Wrong checks | Grader service | Grader idle |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [20261003T224917.672353Z](../../../attempts/20261003T224917.672353Z/summary.json) | 116.281s | 0.183s | 2 | 60.002s | 51.924s |
| 2 | [20261003T225115.111184Z](../../../attempts/20261003T225115.111184Z/summary.json) | 150.067s | 0.161s | 0 | 54.002s | 91.676s |
| 3 | [20261003T225346.318962Z](../../../attempts/20261003T225346.318962Z/summary.json) | 152.007s | 0.167s | 1 | 57.002s | 90.233s |

Median first-18 time: **150.067s**. **0/3** at or below the predeclared 75s reporting threshold; the 71s objective remains unmet.

The reference had 12.076s idle between grader checks. Here that idle time rose to 51.924/91.676/90.233s while initial TTFT remained 0.161–0.183s. Trial two had no wrong checks and still took 150s, demonstrating a long wait for correct candidates even without verification mistakes. This does not establish the cause of trajectory changes or measure uncensored reasoning accuracy.

Controls and sampling match the reference: BF16/95%, 65,536 total context, 30×1 barrier, 8192 first-pass / 16384 later, four requests per question, temperature 0.8, top-p 0.95, seed 20261003, target 18. Unlike the benchmark repeats, optional GPU/engine/CPU profiling and immediate trace writes were restored. Exact token/request/verdict evidence is versioned; raw SSE/audits/service logs remain remote.

Source commit: `4995023`; six replication-driver tests passed before deployment. Each attempt passed payload, cap, first-solved and metadata-schema checks after import. Imported attempt evidence totals about 25.1 MB. The batch completed and handed the GPU to the paired NVFP4 trials.

[Controls](config.json), [all trial summaries](summary.json).
