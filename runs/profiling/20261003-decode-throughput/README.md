# Observed decoding throughput versus concurrency

![Decode throughput](throughput.png)

Rates below come from prior BF16/95% sweep engine counters, using only official
samples after every initial request's first token, with no waiting requests at
both endpoints. Warmup is excluded. Windows crossing a concurrency bin are
excluded. Aggregate throughput is generated token-counter growth / elapsed time;
per-active throughput divides by approximate request-seconds using endpoint counts.

| Observed concurrency range | Mean active | Aggregate tok/s | Tok/s per active request | Seconds observed |
| --- | ---: | ---: | ---: | ---: |
| 9–16 | 14.5 | 1568 | 108.1 | 11.1 |
| 17–32 | 24.4 | 2627 | 107.7 | 78.7 |
| 33–64 | 52.9 | 4056 | 76.7 | 94.8 |
| 65–96 | 76.7 | 5246 | 68.4 | 37.6 |
| 97–128 | 113.2 | 9287 | 82.1 | 25.5 |

| Initial configuration | Included window tok/s | Tok/s per active request | Seconds included |
| --- | ---: | ---: | ---: |
| 30×1 | 2491 | 115.5 | 68.6 |
| 30×2 | 4053 | 94.6 | 73.4 |
| 30×4 | 5435 | 69.7 | 105.7 |

Increasing fan-out can raise aggregate GPU token output while reducing each
trajectory's decoding speed. A speedrun depends on when useful candidate answers
reach the serial grader; aggregate tok/s alone does not predict time to 18.
These are observational measurements: context lengths, active question mix and
cancellations change over each run. They do not establish a fixed-context scaling
curve or a causal concurrency effect. The per-active rate is an approximation
because concurrency is sampled once per interval. Benchmark-mode runs deliberately
omit these engine polls; no throughput counter data is inferred for them.

[Window-level data and source attempts](analysis.json). PNG, SVG and PDF exports
are adjacent. Reproduce with `python scripts/report_decode_throughput.py`.

## New experiments

The best historical control reached 18 in 71.135s, but its
[benchmark-mode replication](../../experiments/bf16-best-30x1-benchmark-20261003T220008Z/README.md)
took 113.625s. The
[grader-triggered staged run](../../experiments/bf16-staged-30x4-benchmark-20261003T220440Z/README.md)
expanded at 5.038s, exhausted 120 fresh 8K samples and stopped with 17 correct.
It has no time-to-18 or rank. These outcomes do not demonstrate a staged fan-out
speedup. Benchmark-mode traces preserve required generation/verdict timings,
but omit the optional engine counters used in the plots above.
