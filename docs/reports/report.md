# Qwen3 30B A3B on MathArena AIME 2025

- **Pass@1:** 12/30 = 40.0%
- **Wall clock, first inference request through all local grading:** 697.276 s (11.62 min)
- **Start / finish (UTC):** 2026-09-30T22:52:12.142+00:00 / 2026-09-30T23:03:49.412+00:00
- **Total input / generated tokens:** 7,464 / 406,324
- **Reported reasoning tokens:** 283,674 (included in generated tokens)
- **Sum of per-question API latencies:** 5987.589 s; median 235.383 s
- **OpenRouter reported cost:** $0.2041
- **Capped outputs without a final answer:** 18

One response was generated per question. Requests ran concurrently with a limit of 10. The request used thinking mode, temperature 0.6, top-p 0.95, top-k 20, and a 16,384-token output cap. Answers were extracted from the final response content and graded by exact integer match to the official MathArena answer key. A capped or missing final answer counts as incorrect. There were no API retries in this run.

| Q | Correct | API latency (s) | Input tokens | Generated tokens | Reasoning tokens | Finish | Trace |
|---:|:---:|---:|---:|---:|---:|:---|:---|
| 1 | ✓ | 68.216 | 88 | 5123 | 2858 | stop | [JSON](../../runs/20260930-155212/questions/01.json) |
| 2 | ✗ | 235.391 | 698 | 16384 | 11383 | length | [JSON](../../runs/20260930-155212/questions/02.json) |
| 3 | ✓ | 189.931 | 171 | 13200 | 9344 | stop | [JSON](../../runs/20260930-155212/questions/03.json) |
| 4 | ✓ | 170.458 | 115 | 12033 | 7170 | stop | [JSON](../../runs/20260930-155212/questions/04.json) |
| 5 | ✓ | 138.912 | 138 | 9929 | 7011 | stop | [JSON](../../runs/20260930-155212/questions/05.json) |
| 6 | ✓ | 53.061 | 141 | 4086 | 2453 | stop | [JSON](../../runs/20260930-155212/questions/06.json) |
| 7 | ✗ | 235.375 | 202 | 16384 | 15887 | length | [JSON](../../runs/20260930-155212/questions/07.json) |
| 8 | ✓ | 95.168 | 172 | 6709 | 3480 | stop | [JSON](../../runs/20260930-155212/questions/08.json) |
| 9 | ✗ | 235.742 | 164 | 16384 | 9677 | length | [JSON](../../runs/20260930-155212/questions/09.json) |
| 10 | ✗ | 235.696 | 355 | 16384 | 14116 | length | [JSON](../../runs/20260930-155212/questions/10.json) |
| 11 | ✗ | 247.597 | 543 | 16384 | 10280 | length | [JSON](../../runs/20260930-155212/questions/11.json) |
| 12 | ✗ | 247.368 | 170 | 16384 | 11373 | length | [JSON](../../runs/20260930-155212/questions/12.json) |
| 13 | ✗ | 246.951 | 139 | 16384 | 16196 | length | [JSON](../../runs/20260930-155212/questions/13.json) |
| 14 | ✗ | 247.809 | 192 | 16384 | 9904 | length | [JSON](../../runs/20260930-155212/questions/14.json) |
| 15 | ✗ | 250.241 | 134 | 16384 | 12024 | length | [JSON](../../runs/20260930-155212/questions/15.json) |
| 16 | ✓ | 89.093 | 159 | 5958 | 3027 | stop | [JSON](../../runs/20260930-155212/questions/16.json) |
| 17 | ✓ | 141.325 | 93 | 9325 | 5524 | stop | [JSON](../../runs/20260930-155212/questions/17.json) |
| 18 | ✗ | 252.247 | 328 | 16384 | 13388 | length | [JSON](../../runs/20260930-155212/questions/18.json) |
| 19 | ✓ | 115.607 | 259 | 7688 | 3939 | stop | [JSON](../../runs/20260930-155212/questions/19.json) |
| 20 | ✗ | 252.392 | 671 | 16384 | 11007 | length | [JSON](../../runs/20260930-155212/questions/20.json) |
| 21 | ✗ | 253.556 | 828 | 16384 | 10444 | length | [JSON](../../runs/20260930-155212/questions/21.json) |
| 22 | ✗ | 248.868 | 150 | 16384 | 12095 | length | [JSON](../../runs/20260930-155212/questions/22.json) |
| 23 | ✗ | 248.198 | 352 | 16384 | 11673 | length | [JSON](../../runs/20260930-155212/questions/23.json) |
| 24 | ✓ | 164.854 | 150 | 10543 | 6117 | stop | [JSON](../../runs/20260930-155212/questions/24.json) |
| 25 | ✗ | 249.183 | 126 | 16384 | 13736 | length | [JSON](../../runs/20260930-155212/questions/25.json) |
| 26 | ✓ | 230.635 | 116 | 15288 | 11162 | stop | [JSON](../../runs/20260930-155212/questions/26.json) |
| 27 | ✗ | 242.480 | 319 | 16384 | 11998 | length | [JSON](../../runs/20260930-155212/questions/27.json) |
| 28 | ✗ | 232.305 | 199 | 16384 | 9067 | length | [JSON](../../runs/20260930-155212/questions/28.json) |
| 29 | ✓ | 159.817 | 152 | 11530 | 5620 | stop | [JSON](../../runs/20260930-155212/questions/29.json) |
| 30 | ✗ | 209.113 | 140 | 16384 | 11721 | length | [JSON](../../runs/20260930-155212/questions/30.json) |

The per-question JSON files contain the entire OpenRouter response, including the final content and returned reasoning fields, usage details, request payload, timing, answer, and local grade. The API key is excluded.

## Eight-sample expansion

Seven additional independent responses per question have now been saved, for
eight attempts per question including this original pass. **Pass@8 is 22/30
(73.3%)**. The unique modal-answer vote is correct on 21/30; a strict
five-of-eight majority is reached and correct on 15/30. The additional run's
elapsed wall time, including the laptop sleep interval, was **2,620.031 s**
(43m 40.031s). See [SELF_CONSISTENCY.md](self-consistency.md) for all 30 vote
distributions, output-cap counts, timing, retries, and saved trace locations.

The original 18 failures were also reviewed by Jev from both their full
reasoning trace and their first 1,500 Qwen reasoning tokens. See
[JEV_REVIEW.md](jev-review.md).
