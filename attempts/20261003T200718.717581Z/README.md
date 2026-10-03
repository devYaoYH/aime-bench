# VibeThinker-3B coverage with continuation

Attempt `20261003T200718.717581Z` reached **18 verified correct answers in
92.503 seconds** on callosum's A100 PCIe 80GB. Fifteen solved during the 8K
first pass; three more solved through exact-token continuations. The runner
stopped and cleaned up its inference and grader services at the target.

Source commit: `f7efed1336518ab7391bcaf898bc92b3439c610a`, clean remote checkout.
Model: `WeiboAI/VibeThinker-3B`, BF16, 65,536-token context, server-wide output
ceiling 16,384 tokens. Thirty questions started concurrently with one rollout
each. First-pass budget: 8,192 tokens. Continuation budget: up to 16,384 tokens.
Hard request budget: four per question, including continuations. Temperature
0.8, top-p 0.95, seed 20261003. Initialization included CUDA warmup and a
30-request inference warmup at the configured sampling settings.

```sh
~/.venvs/vllm/bin/python -m src.attempt --model WeiboAI/VibeThinker-3B \
  --strategy coverage --first-pass-max-tokens 8192 --max-tokens 16384 \
  --target-correct 18 --max-attempts-per-question 4
```

| Measurement | Result |
| --- | ---: |
| Verified correct / questions attempted | 18 / 30 |
| Official solving time | 92.503 s |
| Initialization plus solving time | 164.749 s |
| Initialization time, excluded from official timer | 72.245 s |
| Questions solved in rounds 1 / 2 | 15 / 3 |
| Requests in rounds 1 / 2 | 30 / 15 |
| Maximum requests used by any question | 2 (limit 4) |
| Peak concurrent generation streams | 30 |
| Completed / cancelled / errored streams | 15 / 30 / 0 |
| Completed streams ending at 8K cap | 15 |
| Observed output token IDs, including partial cancelled streams | 202,883 |
| Unique candidates submitted; correct / wrong | 18; 18 / 0 |
| Candidate extraction: closed boxes / integer prose | 6 / 12 |
| Grader toll, global worker busy time | 54.000 s |
| Sum of grader queue waits | 47.659 s |
| Sum of client grader query latencies | 101.745 s |
| Median / p95 client grader latency | 4.938 / 10.215 s |
| First-pass median / p95 client TTFT | 3.475 / 3.527 s |
| Continuation median / p95 client TTFT | 0.198 / 0.228 s |
| Observed generation latency median / p95 | 33.442 / 78.145 s |
| Generation VRAM peak | 65,806.188 MiB (64.26 GiB) |
| All-phase VRAM peak | 65,958.188 MiB (64.41 GiB) |
| GPU KV capacity reported by vLLM | 1,637,248 tokens |
| Largest periodically logged KV occupancy | 8.4% |

All 15 continuation requests used exactly the prior returned prompt IDs plus
all 8,192 generated token IDs; each parent's token count matched its usage.
The second round started at 20:09:49.161 UTC. vLLM's subsequent 20:09:56
log reported 95.4% aggregate prefix-cache hits, versus 27.2% before continuations.
This aggregate includes warmup/first-pass traffic and is not a per-request hit
fraction. All second-round streams were cancelled after verification or at
the target, so final per-request usage details were unavailable; their
`cached_prompt_tokens` fields remain null. Faster continuation TTFT and the
aggregate cache metric support cache reuse, without proving its isolated
speedup against a fresh-retry control.

All 18 submitted answers were extracted while their streams were still running,
before natural EOS: six closed boxes and twelve integer prose clauses. Every
winning stream was cancelled after verification with no terminal finish reason.
These are nonterminal prospective answers, even when the text called one a
"final answer" and then continued checking. None was extracted from a completed
final response. First-pass TTFT measures fresh prompts on an already warmed
server; its batch of 30 differs from the continuation batch of 15, so the TTFT
comparison also includes batch-size and prefill differences.

Thirty simultaneous questions fit without inference waiting in this run.
vLLM preallocated most device VRAM; the 15.74 GiB of physical headroom does
not directly indicate additional stream capacity. Its token cache and the
length of concurrent trajectories determine capacity. This run's low cache
occupancy does not establish that 120 full 16K trajectories fit:
1,966,080 output tokens alone exceed the reported cache capacity.

Every solved question has a client first-solved timestamp and elapsed time
linked to the grader's answer timestamp and query ID. Full query IDs are in
`summary.json` and remote `solved.jsonl`.

| Question | Round | Answer | First solved elapsed (s) | Client UTC | Grader UTC |
| --- | --- | --- | ---: | --- | --- |
| 1 | 1 | 70 | 12.233 | 2026-10-03T20:08:43.196+00:00 | 2026-10-03T20:08:43.194316Z |
| 3 | 1 | 16 | 15.469 | 2026-10-03T20:08:46.431+00:00 | 2026-10-03T20:08:46.430088Z |
| 6 | 1 | 504 | 18.469 | 2026-10-03T20:08:49.432+00:00 | 2026-10-03T20:08:49.430382Z |
| 8 | 1 | 77 | 21.471 | 2026-10-03T20:08:52.434+00:00 | 2026-10-03T20:08:52.430706Z |
| 17 | 1 | 49 | 24.470 | 2026-10-03T20:08:55.432+00:00 | 2026-10-03T20:08:55.430998Z |
| 16 | 1 | 468 | 27.472 | 2026-10-03T20:08:58.435+00:00 | 2026-10-03T20:08:58.431303Z |
| 4 | 1 | 117 | 30.471 | 2026-10-03T20:09:01.434+00:00 | 2026-10-03T20:09:01.431610Z |
| 22 | 1 | 237 | 33.472 | 2026-10-03T20:09:04.435+00:00 | 2026-10-03T20:09:04.431932Z |
| 25 | 1 | 907 | 36.471 | 2026-10-03T20:09:07.433+00:00 | 2026-10-03T20:09:07.432222Z |
| 19 | 1 | 106 | 39.471 | 2026-10-03T20:09:10.434+00:00 | 2026-10-03T20:09:10.432508Z |
| 27 | 1 | 19 | 42.471 | 2026-10-03T20:09:13.433+00:00 | 2026-10-03T20:09:13.432822Z |
| 21 | 1 | 293 | 46.289 | 2026-10-03T20:09:17.252+00:00 | 2026-10-03T20:09:17.251114Z |
| 24 | 1 | 149 | 55.855 | 2026-10-03T20:09:26.817+00:00 | 2026-10-03T20:09:26.816776Z |
| 7 | 1 | 821 | 72.126 | 2026-10-03T20:09:43.089+00:00 | 2026-10-03T20:09:43.088104Z |
| 9 | 1 | 62 | 75.870 | 2026-10-03T20:09:46.832+00:00 | 2026-10-03T20:09:46.831378Z |
| 26 | 2 | 113 | 86.428 | 2026-10-03T20:09:57.391+00:00 | 2026-10-03T20:09:57.389564Z |
| 18 | 2 | 82 | 89.429 | 2026-10-03T20:10:00.391+00:00 | 2026-10-03T20:10:00.389891Z |
| 5 | 2 | 279 | 92.428 | 2026-10-03T20:10:03.391+00:00 | 2026-10-03T20:10:03.390204Z |

The complete raw evidence remains on the remote:
`/home/azureuser/aime-bench/attempts/20261003T200718.717581Z/`, including
`trace/`, `solved.jsonl`, `grader_audit.jsonl`, `gpu.jsonl`, `vllm.log`,
and the exact-token request/response records. Original config, summary, saved
requests/responses, token records, verification events, and GPU samples are
versioned beside this report for the canonical viewer and analysis. GPU process inspection after completion
confirmed no remaining inference workload.

The [initial Qwen 8 × 4 baseline](../20261003T193711.679999Z/README.md)
solved 5/30 in 999.477 seconds with five correct candidates, no wrong verdicts,
and 15 seconds of grader toll. This experiment changes model, parser, warmup,
concurrency, and continuation policy together; the difference cannot be
attributed to parallelism alone. Stopping at 18 also leaves the final accuracy
of the other twelve questions unmeasured.

Validation: the runner's 71 local tests passed before its source commit was
pushed and pulled remotely. This real run additionally validated exact
continuation prefixes, first-solved evidence, target cancellation, the request
budget, mandatory GPU telemetry, and managed-service cleanup.
