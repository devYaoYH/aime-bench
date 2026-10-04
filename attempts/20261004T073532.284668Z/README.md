# Core v1.5 eager30, seed 20261011

One predeclared trial: **18 verified correct in 65.015s**. Peak generation
concurrency 30; 30 initial requests, 29 fresh retries, 14 continuations; at most
four generation requests per question. The matched historical seed took 71.321s.

See the [experiment report](../../runs/experiments/core-v1_5-eager30-single/README.md)
for the policy, unchanged controls, source/manifest checks, tradeoffs and limits.
`allocation.json` records every admission; `summary.json`, `solved.jsonl` and
`trace/` retain timing, verdict, request/response and exact token evidence.

This trial was retained in the [complete five-seed report](../../runs/experiments/core-v1_5-five-seeds-20261004T074500Z/README.md),
which found no improvement to the batch median.
