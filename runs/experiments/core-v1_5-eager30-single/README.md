# Core v1.5: one eager 30-slot trial

This is the original single-trial snapshot. The [later five-seed extension](../core-v1_5-five-seeds-20261004T074500Z/README.md)
retained this attempt, reached 18 in 5/5, and found no median improvement over v1.

The single predeclared v1.5 trial reached **18 verified correct in 65.015s**,
compared with **71.321s** for the historical core v1 run using seed 20261011:
**6.306s (8.84%) faster in this comparison**. This is exploratory evidence from
one run; it does not replace core v1's five-seed headline or establish repeatability.
65.015s lies within the final v1 batch's 62.783–82.492s range.

| Measure | Historical core v1 | Core v1.5 |
| --- | ---: | ---: |
| Time to 18 verified correct | 71.321s | 65.015s |
| Time to 14 verified correct | 52.637s | 52.631s |
| Time from 14 to 18 | 18.684s | 12.384s |
| Generation requests | 45 | 73 |
| Initial / later fresh / continuation requests | 30 / 0 / 15 | 30 / 29 / 14 |
| Completed wrong grader checks | 4 | 2 |
| Client-observed output token IDs | 217,285 | 259,042 |
| Peak concurrent generation requests | At most 30, one per question | 30; up to three for one question |

The curves remain almost identical through 14 correct; the difference appears
in the last four solves. Most of the 6.306s saving aligns with two fewer three-
second wrong checks: grader service fell from 66.002s to 60.002s, and grader idle
between jobs fell only from 4.216s to 3.915s. One fresh retry won Q1; one
continuation won Q18; the other sixteen winning candidates came from initial
requests. Scheduling changes batch composition and candidate order, and identical
initial payloads need not produce identical stochastic outputs. This run therefore
supports testing the policy further, without isolating a generation tail mechanism.
It used 62.2% more generation requests and 19.2% more client-observed output tokens.

## Policy and preserved controls

V1.5 starts one request per question and immediately fills freed generation slots.
Ready continuations have priority; otherwise it starts a fresh sample for the
least-active unsolved question, rotating ties. There is no round barrier or first-
submission fan-out gate. Generation continues while verification is pending.
A correct verdict cancels every generation for that question. Admission stops at
the eighteenth distinct positive verdict. The global pool has 30 slots; available
work and the four-request budgets can prevent it from staying full. There is no
additional per-question concurrency limit; the observed maximum was three.

Every initial request allows 8,192 output tokens. Every later request, fresh or
continued, allows 16,384 **additional** tokens, with continuations clipped to
remaining total context. Exact prompt/output IDs are reused only after a capped
completion, using the frozen v1 continuation helper. Natural completion or context
exhaustion makes a fresh start eligible. Every request consumes the shared four-
request budget. Candidate extraction, deduplication, one verifier per question,
three-second serial grading, and first-solved timing retain core v1 semantics.

The unchanged controls are the adherence_v1 prompt, temperature 0.8, top-p 0.95,
seed 20261011 with stride four, 30×32-token arithmetic warmup, benchmark buffering,
NVFP4/Marlin weights, BF16 activations/KV, FlashInfer, 95% allocation, and 65,536-
token total context. This experiment uses the standard `vllm-flashinfer.yaml`,
not the separate FP8 KV profile. It started a fresh owned server and grader.

## Provenance and checks

- [Predeclared single-trial protocol](../../../configs/experiments/vibe-core-v1_5-eager30-single-v1.json).
- [Core v1.5 manifest](../../../runner_final/core_v1_5/manifest.json), independent of unchanged core v1.
- Source commit: `ebb1fb8e1a2379276468a2834943c3b6aaed300e`; tracked source was clean.
- Launch: `2026-10-04T07:35:32Z`, durable tmux session `core-v1_5-ebb1fb8e` on callosum.
- [New attempt](../../../attempts/20261004T073532.284668Z/) and [same-seed control](../../../attempts/20261004T005518.974361Z/).
- [Audit and milestone data](analysis.json): all 30 initial payloads are exactly equal to the control;
  recorded prompt/profile/sampling controls match; every continuation uses its
  saved parent IDs with the preserved additional-token budget; measured client
  concurrency peaks at 30; request counts never exceed four per question.
- All 73 admissions have request evidence; no admission occurs after the target.
  The service log also observed at most 30 running requests, zero waiting,
  and 7.2% KV cache usage, with no OOM/error lines. These are periodic log observations;
  optional GPU telemetry was disabled as in the control. Owned services cleaned up.
- Eleven focused checks passed. On the clean committed snapshot, 230/232 full-suite
  checks passed; the two legacy salvage checks require ignored raw trace fixtures.
  Existing local/remote edits were preserved and excluded from the experiment.

Command, run from the clean checkout on the provided node:

```bash
~/.venvs/vllm/bin/python -m src.attempt_runners.speedrun_v1_5 \
  --seed 20261011 \
  --grader-python /home/azureuser/aime-bench/grader/.venv/bin/python
```

Full SSE streams, service logs and grader audits remain in the remote checkout
`/home/azureuser/aime-bench-v1_5-ebb1fb8e`. Git contains the compact viewer/analysis
evidence, including exact token records and the allocation ledger. This original protocol executed only seed 20261011; the later extension retained
it and added the other four declared seeds, without replacement seeds or selected
repeats.
