# Grader-triggered 30×1 → 30×4 benchmark

**Target unmet: 17 verified correct; no time-to-18 or rank.** The attempt
exhausted its four 8K fresh samples per question after 130.978s.
It reached 17 correct at 108.373s. Process exit was 0: execution
completed without an implementation/service error, but the requested solve target
was not attained.

| Experiment | Result |
| --- | --- |
| Best historical BF16 30×1 | 18 correct in 71.135s |
| Same controls with `--benchmark` | 18 correct in 113.625s |
| Staged 30×1 → 30×4 with `--benchmark` | 17 correct; target unmet |

## Trigger and policy

All 30 initial requests started before the first client grader submission.
Question 1's prospective answer 70 triggered expansion at
**5.038s**. Three fresh sibling tasks were
released for every question, before awaiting the verdict. Every question used
exactly four requests: **120 total**, no continuations or fifth requests. Seeds
retain the fixed four-request stride. All 30 initial request payloads and prompt
IDs match the benchmark-mode control, though generated token paths differ.

The [versioned runner](../../../src/attempt_runners/speedrun_v3.py) preserves
question-wide candidate deduplication, one pending verification per question,
cancellation of solved siblings, and stop-at-18 behavior. All samples use an
8192-token generation budget. The 16384-token configured later budget is unused
because all four requests are fresh and the cap leaves no continuation budget.
Warmup covers 30 streams, matching the control. The BF16 model profile remains
95% utilization, 65536 total context, and a 16384 generation ceiling.
`fanout.json` expansion times mark task release; individual actual generation
start/header/TTFT timestamps remain in each rollout's telemetry.

Of the 17 winning answers, 9 came from the initial sample and
8 from newly added siblings. This counts winning verification events,
not independent estimates of how many would have solved without expansion.

## Measurements

- Initial 30 requests: median TTFT 0.202s.
- Additional 90 requests: median TTFT 1.555s.
- 18 completed checks: 17 correct,
  1 wrong. Candidate syntax: {'boxed': 6, 'literal_prose': 12}.
- Grader service: 54.002s; idle between
  checks: 49.215s. Increasing fan-out
  did not keep the grader continuously supplied with useful candidates.
- 68 streams cancelled after solves;
  52 completed naturally.
- Runner official CPU: 82.175
  CPU-s; peak RSS 577.9 MiB.
- Final buffered flush: 4.743s for
  642,136 rows; excluded from official timing.
- Service logs report peak KV utilization 20.0% and no OOM
  messages. Optional GPU/engine polling is disabled, so sampled VRAM peak and
  preemption counters are unavailable. Final GPU memory was 0 MiB.

[Saved analysis](analysis.json), [command/provenance](config.json),
[summary](../../../attempts/20261003T220441.154715Z/summary.json),
[fan-out trigger](../../../attempts/20261003T220441.154715Z/fanout.json),
[first-solved verdict timestamps](../../../attempts/20261003T220441.154715Z/solved.jsonl),
[historical decode throughput plot](../../profiling/20261003-decode-throughput/README.md).

This trial does not isolate pure staging effects: the control used its spare
request budget for exact-ID continuations with a larger later budget, whereas
staged pass@4 spends all four requests on fresh 8K samples. Sampling paths also
vary. Preserve this outcome as target-unmet rather than extrapolating a first-18
latency or ranking it against completed runs.

Validation: 139 offline tests passed before code deployment. Imported evidence
checks confirmed request caps/seeds/budgets, trigger ordering, required timing and
17 distinct solved events linked to grader answer timestamps. Metadata schema
validation passed. Raw streams and full service logs/audits remain remote.
