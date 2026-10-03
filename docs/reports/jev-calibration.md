# Retrospective check of Jev's “promising to extend” decision

I evaluated the saved trajectories that **actually ended with a final Qwen response**.
For each, [`calibrate_jev.py`](../../src/experiments/jev/calibrate_jev.py) sent the problem and exactly the
first 1,500 reasoning tokens (cut with the official Qwen3 tokenizer) to the
same `typesafe/jev-1.13` Noul decision used in the earlier prefix review.
Jev did not see the remainder of the trace, the final answer, the completion
length, the local grade, or the answer key. Each decision was made once; all
125 API calls succeeded without retries. The full requests and responses are
in [`jev_calibration/`](../../runs/20260930-155212/jev_calibration).

The Noul asks whether continuing **from that prefix for at most 8,192 more
output tokens** is likely to produce a mathematically justified correct AIME
integer answer. I count an observed *sufficient* success for that question
when the saved continuation returned a correct final answer by **9,692 total
output tokens** (1,500 + 8,192). Local exact integer matching is the outcome
proxy; it does not independently verify every line of the mathematical proof.
The external grader script was not invoked.

## False negatives at candidate rejection cutoffs

“Reject below” treats a Jev score below the threshold as *not promising*.
The false-negative columns count observed correct completions that this rule
would reject.

| Reject below | Completed attempts rejected | Correct within 8,192 more tokens rejected | Eventually correct attempts rejected | Completed wrong attempts rejected |
| ---: | ---: | ---: | ---: | ---: |
| 50% | 1 / 125 | **0 / 54** | 1 / 120 | 0 / 5 |
| 60% | 18 / 125 | **0 / 54** | 16 / 120 | 2 / 5 |
| 65% | 26 / 125 | **0 / 54** | 21 / 120 | 5 / 5 |
| 70% | 35 / 125 | **2 / 54** | 30 / 120 | 5 / 5 |

The 54 within-budget successes came from **11 distinct questions**. Their
lowest Jev score was 68%; the median was 83%. Thus a 50%, 60%, or 65%
cutoff missed none of these *observed* within-budget successes. At 70%, the
two misses were [question 27, attempt 3](../../runs/20260930-155212/jev_calibration/27-03.json)
and [attempt 7](../../runs/20260930-155212/jev_calibration/27-07.json): both scored
68% and returned the correct answer after 8,606 and 8,639 total output tokens.

The one below-50% case is [question 26, attempt 4](../../runs/20260930-155212/jev_calibration/26-04.json).
Jev gave its prefix 47%, and Qwen eventually returned the correct answer,
but only after **14,216** total output tokens, or about **12,716** after the
prefix. It is a miss if “promising” means eventual success under the original
16,384-token cap; it does **not** establish a miss under the Noul's 8,192-token
continuation budget. The other 66 correct completions after 9,692 total tokens
also cannot be called within-budget successes from the final-response record.

## Interpretation

- **No evidence of an oversensitive 50% rejection rule in the observed
  within-budget positives:** 0 of 54 were rejected. This is a conditional
  observation, not a guarantee. Those 54 attempts cover only 11 problems and
  repeated attempts on the same problem are correlated.
- **50% is very permissive in this completed cohort:** it rejects just one of
  125 attempts, and all five completed wrong answers score at least 52%. This
  cohort cannot establish that Jev separates solvable from unsolvable prefixes.
- **A higher cutoff creates misses quickly:** at 70%, two proven within-budget
  successes would be rejected, both on question 27. If the actual continuation
  budget is the full original cap, even the 60% cutoff discards 16 eventually
  correct attempts.
- **The 115 capped trajectories are censored:** they did not return a correct
  final answer within the original cap, but that does not prove their 1,500-token
  prefixes were unsalvageable. They are not labeled negatives here. The five
  completed wrong attempts are also only observed failures of one continuation,
  not proof of impossibility.

The [local viewer](http://127.0.0.1:8765/?run=20260930-155212&q=26&attempt=4)
shows each scored completed attempt beside its observed result. The
machine-readable [summary](../../runs/20260930-155212/jev_calibration/summary.json)
contains every score, outcome, output length, and threshold count. Jev used
270,570 input tokens for these decisions, at a provider-reported cost of
$0.01136394.
