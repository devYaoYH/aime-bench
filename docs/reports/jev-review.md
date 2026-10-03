# Jev review of the 18 unfinished AIME trajectories

The benchmark's 18 failed `qwen/qwen3-30b-a3b` trajectories all stopped at the
16,384 output-token cap without a final answer. I sent each complete problem
statement and its full saved reasoning trace to
[`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13), using OpenRouter's
[Decisions API](https://openrouter.ai/docs/guides/community/jev-tutorial).
The official answers, grades, and earlier benchmark scores were withheld.

Four [Noul](https://docs.typesafe.ai/primitives/noul) yes/no decisions were
requested per trace. The primary decision asks whether **the same Qwen solver,
continuing from the end of the saved trace for at most 8,192 more output tokens,
could likely reach a mathematically justified correct AIME answer without
restarting**. The supporting decisions ask whether there is a coherent route,
whether the trace is near a solution, and whether it is stuck or repeating.
Numbers below are Jev's probabilities for **yes**, ranked by the primary
decision. They are forecasts for prioritization, not observed continuation
success rates or verified proofs.

| Rank | Question | Promising to extend | Coherent route | Near solution | Stuck / repeating |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [12](../../runs/20260930-155212/jev_review/12.json) | 78% | 93% | 82% | 17% |
| 2 | [25](../../runs/20260930-155212/jev_review/25.json) | 74% | 93% | 71% | 33% |
| 3 | [27](../../runs/20260930-155212/jev_review/27.json) | 73% | 91% | 77% | 28% |
| 4 | [09](../../runs/20260930-155212/jev_review/09.json) | 71% | 93% | 78% | 20% |
| 5 | [18](../../runs/20260930-155212/jev_review/18.json) | 59% | 94% | 60% | 15% |
| 6 | [23](../../runs/20260930-155212/jev_review/23.json) | 57% | 80% | 46% | 61% |
| 7 | [20](../../runs/20260930-155212/jev_review/20.json) | 47% | 77% | 25% | 32% |
| 8 | [11](../../runs/20260930-155212/jev_review/11.json) | 46% | 83% | 13% | 24% |
| 9 | [07](../../runs/20260930-155212/jev_review/07.json) | 45% | 77% | 11% | 31% |
| 10 | [10](../../runs/20260930-155212/jev_review/10.json) | 42% | 71% | 13% | 28% |
| 11 | [02](../../runs/20260930-155212/jev_review/02.json) | 41% | 60% | 18% | 89% |
| 12 | [21](../../runs/20260930-155212/jev_review/21.json) | 41% | 73% | 11% | 71% |
| 13 | [30](../../runs/20260930-155212/jev_review/30.json) | 38% | 64% | 23% | 46% |
| 14 | [13](../../runs/20260930-155212/jev_review/13.json) | 30% | 60% | 7% | 68% |
| 15 | [14](../../runs/20260930-155212/jev_review/14.json) | 26% | 56% | 9% | 71% |
| 16 | [15](../../runs/20260930-155212/jev_review/15.json) | 25% | 47% | 8% | 71% |
| 17 | [22](../../runs/20260930-155212/jev_review/22.json) | 25% | 89% | 18% | 83% |
| 18 | [28](../../runs/20260930-155212/jev_review/28.json) | 22% | 42% | 6% | 71% |

## What the highest and lowest scores reflect

- **Q12, Q25, Q27:** Each trace reaches a concrete candidate answer in its
  reasoning and continues checking it, yet never emits a final answer before
  the output cap. Those are natural first candidates for continuation. A
  candidate written in reasoning is not a graded response or a checked proof.
- **Q9:** The solver had developed a candidate geometric setup and was checking
  an extraneous-root issue near the end.
- **Q22:** A coherent approach score of 89% coexists with an 83% stuck score.
  Its latter reasoning repeatedly worries that the answer may exceed the AIME
  range and speculates about the problem statement.
- **Q28:** The trace moves among speculative substitutions and cycles without
  arriving at a stable invariant; Jev gave it the lowest continuation score.

All 18 per-question JSON files contain the full request, full Jev response,
usage, and API attempt record. The original Qwen traces remain under
[`runs/20260930-155212/questions/`](../../runs/20260930-155212/questions).
Total Jev usage for this review was 318,559 input tokens, with a provider
reported cost of $0.013379478. No solver continuation was run.

## First 1,500-token prefix review

I also reran the same **promising to extend** Noul on each question and the
first **exactly 1,500 Qwen3 reasoning tokens** of its original trace. The
prefixes were cut with the official `Qwen/Qwen3-30B-A3B` tokenizer. Only this
one Noul was requested; the answer key remained withheld. This asks about
continuing from the early prefix, whereas the full-trace review asks about
continuing from the end of the original 16,384-token response.

| Question | Prefix promising | Full trace promising | Prefix minus full |
| ---: | ---: | ---: | ---: |
| [22](../../runs/20260930-155212/jev_prefix_1500/22.json) | 77% | 25% | +52 points |
| [23](../../runs/20260930-155212/jev_prefix_1500/23.json) | 69% | 57% | +12 points |
| [18](../../runs/20260930-155212/jev_prefix_1500/18.json) | 68% | 59% | +9 points |
| [09](../../runs/20260930-155212/jev_prefix_1500/09.json) | 67% | 71% | -4 points |
| [07](../../runs/20260930-155212/jev_prefix_1500/07.json) | 65% | 45% | +20 points |
| [21](../../runs/20260930-155212/jev_prefix_1500/21.json) | 64% | 41% | +23 points |
| [27](../../runs/20260930-155212/jev_prefix_1500/27.json) | 64% | 73% | -9 points |
| [12](../../runs/20260930-155212/jev_prefix_1500/12.json) | 58% | 78% | -20 points |
| [02](../../runs/20260930-155212/jev_prefix_1500/02.json) | 57% | 41% | +16 points |
| [20](../../runs/20260930-155212/jev_prefix_1500/20.json) | 56% | 47% | +9 points |
| [30](../../runs/20260930-155212/jev_prefix_1500/30.json) | 56% | 38% | +18 points |
| [10](../../runs/20260930-155212/jev_prefix_1500/10.json) | 55% | 42% | +13 points |
| [11](../../runs/20260930-155212/jev_prefix_1500/11.json) | 55% | 46% | +9 points |
| [25](../../runs/20260930-155212/jev_prefix_1500/25.json) | 53% | 74% | -21 points |
| [15](../../runs/20260930-155212/jev_prefix_1500/15.json) | 50% | 25% | +25 points |
| [14](../../runs/20260930-155212/jev_prefix_1500/14.json) | 49% | 26% | +23 points |
| [13](../../runs/20260930-155212/jev_prefix_1500/13.json) | 48% | 30% | +18 points |
| [28](../../runs/20260930-155212/jev_prefix_1500/28.json) | 47% | 22% | +25 points |

Q22 looked promising early, but its full trace became repetitive. In contrast,
Q12 and Q25 developed concrete answers after the prefix. These shifts are
qualitative signals; no solver was actually continued from either point.
The prefix score exceeded the full-trace score for 14 of the 18 questions
(mean 58.8% versus 46.7%), suggesting Jev often judged the early route more
favorably than the end of the original capped trace.
The prefix review used 40,508 Jev input tokens at a provider reported cost of
$0.001701336. Full request and response records are in
[`jev_prefix_1500/`](../../runs/20260930-155212/jev_prefix_1500).

For a retrospective false-negative check of this prefix decision against Qwen
trajectories that returned final answers, see [JEV_CALIBRATION.md](jev-calibration.md).
