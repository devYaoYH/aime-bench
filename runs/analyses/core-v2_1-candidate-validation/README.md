# Offline v2.1 candidate validation replay

This replays only recorded client candidate/verdict events from the five-seed
[v2 AIME 2025 back-test](../../experiments/core-v2-aime2025-five-seeds-20261004T013100Z/README.md).
No dataset answers, raw SSE streams, grader audits or inference services are read.
The [summary](summary.json) pins the new core manifest and all input file hashes;
[candidates](candidate_audit.json) retains every decision and expression key.

| Recorded evidence | Replay result |
| --- | --- |
| Completed checks | 152 |
| Correct verdicts retained by syntax | 90/90 |
| Wrong submissions rejected | 56/62 |
| Placeholder rejections | 55 |
| Additional equivalent duplicates suppressed | 0 |
| Uncached distinct input strings | 33 |

The historical analysis classified 54 wrong checks as literal `EXPRESSION`,
`...`, `?`, `??` or `???`. The new guard also rejects `...the answer...`, and the
syntax gate rejects one additional malformed candidate, for 56 rejected wrong
submissions. Six wrong submissions remain syntactically valid; only the grader
can decide correctness. All 90 recorded correct candidates pass.

There were no expression-equivalent duplicates among this cohort's submitted
candidates. Independent integration tests cover `4/8`, `2/4` and
`\frac{1}{2}` sharing one grader check, including across continuation rounds.
Symbolic normalization also handles `x+x`/`2x`, radicals and unordered finite
sets, while ordered tuples and original denominator restrictions remain distinct.

Timing is an offline sequential replay on the local CPU with a warm worker and
raw-input cache. This is not a remote GPU benchmark or a prediction of time saved:
removing checks changes queues and cancellation. [Core contract](../../../runner_final/core_v2_1/README.md)
describes bounds, grammar scope and safe normalization fallbacks.

```sh
.venv/bin/python -m scripts.audit_v2_1_candidates
```

Validation: 287 tests passed in a clean checkout excluding unrelated edits
(18.825s). The final empty-set equivalence addition also passed all 26 focused
v2.1 tests. CLI help and the recorded candidate replay ran without GPU work.
