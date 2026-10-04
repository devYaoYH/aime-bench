# Canonical v1 refactor validation

Run the latest canonical runner on AIME 2025 with the original improved-prompt
seeds `20261011` through `20261015`. This measurement driver preserves the
canonical source manifest and solving policy.

```bash
python -m runner.extensions.validation.refactor_v1 \
  --batch canonical-v1-refactor-five-seeds-YYYYMMDDTHHMMSSZ \
  --grader-python /absolute/path/to/grader/python
```

Use a clean pinned checkout and an available GPU. The driver starts one owned
inference server, reuses it for all five scored trials, clears prefix cache before
each trial, and lets each canonical attempt start a fresh grader and the cheap
inference warmup. There are no unscored settling trials or replacement seeds.
Model profile bytes and runtime package versions must match the historical
`frozen-core-prompt-five-seeds-20261004T005416Z` batch. Grading ASTs are unchanged.

Each trial has a 600s wall safety timeout including its per-trial initialization;
failed, interrupted and target-unmet attempts are retained. Official solving time
still uses the canonical timer boundary. Benchmark mode disables optional polls.

Audits compare all 30 initial request payloads directly to the same-seed original,
check exact continuation prefixes and remaining-context limits, the four-request
ceiling, dataset identity/hashes, first-solved target timestamps and true grader
verdicts. Summaries retain request counts, candidate formats, wrong checks, grader
service/idle times and TTFT. The 10% median band is a preregistered practical screen,
not a statistical equivalence claim; paired timing differences and stochastic
output differences must also be reported.
