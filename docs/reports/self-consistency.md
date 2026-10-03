# Eight-sample self-consistency on AIME 2025

The original benchmark response is attempt 1 for each of the 30 MathArena questions. Seven additional independent `qwen/qwen3-30b-a3b` calls per question used each original request payload exactly, on the same OpenRouter chat-completions endpoint. All 30 questions started their first additional call within 0.029 seconds; at most 30 calls were in flight. The model, provider (DeepInfra), prompt, temperature 0.6, top-p 0.95, top-k 20, thinking mode, and 16,384-token cap stayed the same. No external grader script was invoked; final answers were extracted from response content and compared locally with the answer key.

## Scores and timing

- **Pass@1:** 12/30 (40.0%).
- **Pass@8, any one of eight correct:** 22/30 (73.3%).
- **Unique modal-answer vote:** 21/30 (70.0%) correct.
- **Strict majority vote (at least 5 of 8):** 15/30 (50.0%) correct; 15 questions reached a strict majority.
- **Additional-run elapsed wall clock, first request through final grade:** 2620.031 s (43m 40.031s).
- **Additional 210 successful responses:** 52,248 input tokens, 2,740,478 generated tokens, of which 1,887,886 were provider-reported reasoning tokens. Median per-sample API latency was 333.575 s.
- **Output cap:** 97/210 additional responses (46.2%) hit 16,384 tokens, compared with 18/30 (60.0%) of the original responses.
- **HTTP attempts:** 215 for 210 successful response slots, including 5 recovered read-error retries.
- **Provider-reported cost of successful additional responses:** $1.37650876. Failed read attempts had no usage report, so any cost for them is unknown.

The saved UTC timestamps span 2026-09-30T23:40:51.199+00:00 through 2026-10-01T00:24:31.230+00:00. The machine was briefly asleep or disconnected during the run. Its process timer measured 2245.626 s, shorter than elapsed wall clock by 374.405 s. The wall-clock figure above uses the timestamps and includes that interval.

A unique mode is the most frequent parseable final answer; tied top counts yield no modal answer. A strict majority requires the same answer from at least five of the eight attempts. Missing final answers abstain from the modal vote but still occupy an attempt. Pass@8 is true if any attempt has the official answer, regardless of the vote.

## Per-question results

| Q | Correct of 8 | Pass@8 | Modal vote | Strict majority | Answer votes |
| ---: | ---: | :---: | ---: | ---: | :--- |
| 01 | 8/8 | ✓ | 70 | 70 | 70 × 8 |
| 02 | 7/8 | ✓ | 588 | 588 | 588 × 7 |
| 03 | 6/8 | ✓ | 16 | 16 | 16 × 6 |
| 04 | 8/8 | ✓ | 117 | 117 | 117 × 8 |
| 05 | 8/8 | ✓ | 279 | 279 | 279 × 8 |
| 06 | 8/8 | ✓ | 504 | 504 | 504 × 8 |
| 07 | 0/8 | ✗ | 271 | — | 271 × 2 |
| 08 | 8/8 | ✓ | 77 | 77 | 77 × 8 |
| 09 | 1/8 | ✓ | 62 | — | 62 × 1 |
| 10 | 0/8 | ✗ | 66 | — | 66 × 1 |
| 11 | 1/8 | ✓ | 259 | — | 259 × 1 |
| 12 | 3/8 | ✓ | 510 | — | 510 × 3 |
| 13 | 0/8 | ✗ | — | — | No parseable answers |
| 14 | 0/8 | ✗ | — | — | No parseable answers |
| 15 | 0/8 | ✗ | — | — | No parseable answers |
| 16 | 8/8 | ✓ | 468 | 468 | 468 × 8 |
| 17 | 8/8 | ✓ | 49 | 49 | 49 × 8 |
| 18 | 0/8 | ✗ | — | — | No parseable answers |
| 19 | 8/8 | ✓ | 106 | 106 | 106 × 8 |
| 20 | 1/8 | ✓ | 336 | — | 336 × 1 |
| 21 | 5/8 | ✓ | 293 | 293 | 293 × 5 |
| 22 | 5/8 | ✓ | 237 | 237 | 237 × 5 |
| 23 | 1/8 | ✓ | — | — | 600 × 1, 610 × 1 |
| 24 | 8/8 | ✓ | 149 | 149 | 149 × 8 |
| 25 | 3/8 | ✓ | 907 | — | 907 × 3 |
| 26 | 6/8 | ✓ | 113 | 113 | 113 × 6 |
| 27 | 5/8 | ✓ | 19 | 19 | 19 × 5 |
| 28 | 0/8 | ✗ | 208 | — | 208 × 1 |
| 29 | 4/8 | ✓ | 104 | — | 104 × 4 |
| 30 | 0/8 | ✗ | — | — | No parseable answers |

The [original attempt](../../runs/20260930-155212/questions/01.json) and all additional attempts under [`self_consistency/questions/`](../../runs/20260930-155212/self_consistency/questions) retain the full request, response, usage, timing, extracted answer, and local grade. The [summary JSON](../../runs/20260930-155212/self_consistency/summary.json) contains the vote calculation and links to every attempt. The [viewer](http://127.0.0.1:8765/?run=20260930-155212&q=1) groups the eight responses for each question.
