# Replicate the best run with benchmark mode

The fastest prior completed run was `20261003T211557.382358Z`, BF16
WeiboAI/VibeThinker-3B at 30×1, reaching 18 correct in 71.135s.
The same controls with `speedrun_v2 --benchmark` reached 18 in
**113.625s**. The 71-second result was not reproduced.

This run used 44 generation requests across
2 rounds, with at most 2 per question.
The grader completed 21 checks: 18 correct
and 3 wrong; service 63.003s,
idle between checks 45.582s.
Conventional median fresh TTFT was 0.174s;
continuation TTFT was 0.174s.

The model/profile, question set, prompts, seeds, temperature/top-p, barrier
schedule, 8K first pass / 16K later, 64K total context and four-request cap
match the reference. Optional profiling/GPU polling were disabled; traces
were buffered until the official window ended. Flush took
1.779s outside timing. No sampled VRAM peak or
preemption counts are available. Exit 0 and final GPU memory 0 MiB confirmed.
Generated paths are not guaranteed identical under paired sampling seeds.

[Analysis](analysis.json), [command/provenance](config.json),
[summary](../../../attempts/20261003T220008.957658Z/summary.json),
[first-solved timestamps](../../../attempts/20261003T220008.957658Z/solved.jsonl).
Full streams and grader audits remain remote; imported JSON evidence is versioned.
