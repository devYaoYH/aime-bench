# Post-cutoff Jev coverage follow-up

Produced by `python -m src.experiments.jev.complete_calibration --tokenizer-json .local/tokenizers/qwen3.json --execute`. The source commit and producer hash, historical configuration, original-summary hash, and invocation timestamp are saved in [config.json](config.json). This follow-up does not modify the original 125-decision calibration or the pre-cutoff report.

All **115 previously excluded length-capped trajectories** received fresh `typesafe/jev-1.13` Noul decisions through the same OpenRouter endpoint. Jev saw only each problem and a source-preserving prefix that re-encodes to exactly 1,500 Qwen tokens, with the same 8,192-token continuation question. It saw no final answer, grade or answer key. Q9 sample 3 required trimming a split Unicode codepoint at the prefix boundary; that adjustment is recorded in its decision file. All other prefixes use the historical prefix helper unchanged.

[summary.json](summary.json) combines the 115 new scores with the 125 unchanged historical scores, covering all **240** original/pass@8 trajectories. Each row retains its cohort, outcome and decision-file provenance. Compact requests/responses are versioned here; original generation traces remain local. This new cohort was scored in a separate, later batch, rather than simultaneously with the historical cohort.

| Retain score at least | Retained | Correct final retained | Capped/no-final retained | Completed wrong retained | Correct finals rejected | Observed final-answer precision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 226 | 119 | 102 | 5 | 1 | 52.65% |
| 60% | 162 | 104 | 55 | 3 | 16 | 64.20% |
| 65% | 136 | 99 | 37 | 0 | 21 | 72.79% |
| 70% | 104 | 90 | 14 | 0 | 30 | 86.54% |

These are retrospective **final-answer recovery outcomes under the original 16,384-output-token cap**. They do not establish prospective accuracy within the Noul's 8,192-token continuation budget, mathematical unsalvageability of a prefix, or the usefulness of intermediate reasoning answers. Capped cases are observed failures of the saved continuation, not proofs that another continuation could not succeed. Repeated samples share only 30 problems.

The follow-up used **253,837 input tokens** at a provider-reported cost of **$0.010661154**. Exact timing, counts and thresholds are in the summary. At a 50% cutoff, 102/115 capped traces remained promising and all five completed wrong answers were retained. The apparent absence of false positives in the old self-consistency tiles was a coverage omission, not high precision.
