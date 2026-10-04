# Question-matched first-answer analysis

Pairing by question controls question identity and is useful descriptively. It changes the endpoint from **the earliest submission anywhere in a run** to **average first-submission latency per question**. First submissions include incorrect proposals.

The primary comparison remains BF16 versus NVFP4 with FLASH_ATTN fixed, three warmed historical runs/model. Trial numbers are not paired experimental blocks. All question means below average three repeat times/model.

## Complete observed pairs

Only **15 of 30 questions** have a first submission recorded in all six runs: 1, 3, 4, 5, 6, 8, 16, 17, 18, 19, 21, 22, 24, 25, 26.
NVFP4 is faster on **13 of 15**. Mean latency is **30.042s BF16 versus 22.685s NVFP4**, a 7.358s observed reduction.
The naive one-sided paired t-test across these question means gives **p=0.0097**, t=-2.641, df=14.

This is a selected-subset calculation, not evidence of a generally significant speedup. Stop-at-18 leaves other submissions unobserved. Questions within a run share GPU load, cancellation effects, and batch conditions; independence of question differences is not established. Reusing each run to create more question differences does not create more independent model runs.

## All-question sensitivity without excluding unobserved answers

Every run lasted at least 96.667s. For windows below that stopping time, we can measure **min(first-submission time, H)** for all 30 questions. A question with no submission by H contributes H, without asserting it submitted then. This averages time spent waiting within a common window.

| Window H | BF16 mean | NVFP4 mean | NVFP4 minus BF16 | Naive question-paired p | Run-level Welch p |
| --- | --- | --- | --- | --- | --- |
| 10s | 9.841s | 9.736s | -0.105s | 0.0711 | 0.1950 |
| 30s | 25.546s | 23.795s | -1.751s | 0.0031 | 0.0085 |
| 60s | 42.825s | 40.561s | -2.264s | 0.0503 | 0.0456 |
| 90s | 55.897s | 54.713s | -1.184s | 0.3272 | 0.2863 |

All p-values are one-sided, exploratory and unadjusted. Question tests assume independent differences; run tests use three whole-run means per model. The shared fixed seeds/servers, sequential batches and differing source commits remain limitations. Windows were examined retrospectively; selecting one because it has the smallest p-value would overstate the evidence.

At 90 seconds, including all questions reduces the observed difference to **1.184s** (55.897s versus 54.713s); the run-level Welch p is **0.2863**. This illustrates how much the complete-case conclusion depends on the included questions and endpoint.

## Reporting and next experiment

Use: "Among the 15 questions observed in every repeat, NVFP4 submitted first candidates earlier on 13, reducing their mean latency from 30.04s to 22.68s. This question-matched analysis is exploratory and conditioned on observed submissions."

## Zoom into the fastest NVFP4 question

Question matching followed by selecting the fastest NVFP4 question is useful as a descriptive engineering case study. There are two meanings of fastest:

- **Fastest average:** Question 1, averaging three repeats/model. BF16 6.713s versus NVFP4 5.028s, a 25.1% observed reduction.
- **Fastest single submission:** Question 17, at 4.125s in NVFP4 attempt 20261003T225721.655059Z. Across repeats, this question averages BF16 9.611s versus NVFP4 7.990s, a 16.9% observed reduction. Its best BF16 time is 7.876s; comparing those two minima is a best-observed comparison, not a paired independent replicate.

The three timings for each selected question are retained in [the calculation record](question-pairing-results.json). We selected the question using these same outcomes; a significance test on that selected question would need to account for selection or use fresh validation data. The 15-question paired p-value applies to the complete-case question-average comparison, not to the selected fastest question or the global first-submission minimum.

For a stronger test, predeclare the per-question endpoint and horizon; run every question through that horizon regardless of the 18-answer target; use multiple independent, randomized/interleaved model runs and seeds. Keep whole runs together for uncertainty estimation. If running isolated question pairs, randomize model order and repeat across seeds, while acknowledging that isolated latency differs from 30-way speedrun latency.

Method references: [NIST paired observations](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm); [Pustejovsky and Tipton on small-sample cluster inference](https://arxiv.org/abs/1601.01981).

Audit: [complete question pairs](question-pairs-complete.csv), [all calculations and input hashes](question-pairing-results.json), [source question records](question-submissions.csv), [source run records](runs.csv). Reproduce with `aime-bench/.venv/bin/python analyze_question_pairing.py` from the workspace root. No new inference was launched.
