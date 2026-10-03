# VibeThinker-3B naive pass@4, 16K context

Attempt `20261003T202152.418590Z` reached its 18th verified correct answer
at **336.497 seconds (5m36s)**. Official solving plus cancellation settlement
took 336.661 seconds; initialization plus solving took 408.358 seconds.
It ran on callosum's A100 PCIe 80GB from clean source commit
`0834cfe3acb6b630294782834c60dffa6685a265`.

All 30 AIME 2025 questions started with four independent requests each:
120 simultaneous streams, one round, no retries or continuations. The same
VibeThinker-3B BF16 weights and 80% vLLM memory budget were used. The
baseline profile set **16,384 total context tokens**, including the prompt,
as requested after the initial 64K proposal. Each request reserved its measured
prompt tokens, yielding output budgets of 15,581–16,321 tokens.

The system prompt was:
> You are a helpful assistant. Solve the math problem carefully and put your final answer in \\boxed{} notation.

No prospective answer parsing ran. Only a naturally completed response's last
integer box in final content could be submitted; an explicit `</think>`
boundary excluded preceding reasoning. Token-capped responses would remain
ungraded. A correct final verdict cancelled that question's siblings; the
18th solved question cancelled all remaining work. All 18 checked candidates
were from naturally completed final responses and were correct.

```sh
~/.venvs/vllm/bin/python -m src.attempt --model WeiboAI/VibeThinker-3B \
  --model-profile vllm-baseline-16k.yaml --strategy baseline \
  --parallelism 30 --rollouts 4 --max-tokens 16384 --target-correct 18
```

| Measurement | Result |
| --- | ---: |
| Questions solved / attempted at target stop | 18 / 30 |
| First 18 correct, excluding initialization | 336.497 s |
| Official solve and cancellation settlement | 336.661 s |
| Initialization, excluded from official timer | 71.698 s |
| Initialization plus solve | 408.358 s |
| Submitted streams / peak engine running streams | 120 / 120 |
| Completed naturally / cancelled / errored streams | 23 / 97 / 0 |
| Streams ending at token cap before target stop | 0 |
| Final candidates checked; correct / wrong | 18; 18 / 0 |
| Intermediate candidates checked | 0 |
| Grader toll, global worker busy time | 54.000 s |
| Sum of grader queue waits | 3.320 s |
| Sum of client grader query latencies | 57.598 s |
| Median / p95 client grader latency | 3.010 / 4.119 s |
| Median / p95 client TTFT | 5.347 / 5.467 s |
| Median / p95 observed stream latency | 231.632 / 336.444 s |
| Observed completion usage, including partial cancelled streams | 1,251,682 tokens |
| Rollouts with continuous or final usage records | 120 |
| Peak VRAM during generation | 65,930.188 MiB (64.38 GiB) |
| Peak VRAM including initialization | 65,940.188 MiB (64.39 GiB) |
| Largest periodically logged KV occupancy | 48.2% |
| Preemption / OOM warning lines | 0 / 0 |

The measured peak fit the pre-run **64–66 GiB VRAM estimate**. vLLM allocated
1,637,248 token slots, approximately 56.21 GiB of KV cache. Physical headroom
at the generation peak was about 15.62 GiB. Initial prefill briefly logged
50 waiting requests; afterward the engine ran all 120 together and logged
zero waiting requests. No memory-driven preemption or OOM was logged.
Stopping solved siblings freed active cache before the worst case was reached.
Thus the run does not prove that 120 full 16K trajectories fit: that hypothetical
requires about 67.5 GiB of KV alone and exceeds the configured cache capacity.

All 120 output limits plus saved prompt counts fit the 16K context. Reported
prompt usage matched the initialization token counts wherever usage was available.
Each grader query was traced to a completed `stop` response and a
`completed_final_response` extraction event. All four-request limits, 18 first-solve
records, and managed-service cleanup were validated. The GPU was free after
completion. The 85 local tests passed before source was committed, pushed,
pulled, and executed remotely.

## Strategy comparison

| Measurement | Coverage and continuation | Naive final-answer pass@4 |
| --- | ---: | ---: |
| First 18 correct | 92.428 s | 336.497 s |
| Official phase through cancellation settlement | 92.503 s | 336.661 s |
| Peak simultaneous streams | 30 | 120 |
| Generation requests used | 45 | 120 |
| Observed output tokens | 202,883 token IDs | 1,251,682 usage tokens |
| Checked nonterminal / completed final answers | 18 / 0 | 0 / 18 |
| Correct / wrong verdicts | 18 / 0 | 18 / 0 |
| Grader service time | 54.000 s | 54.000 s |
| Generation VRAM peak | 64.26 GiB | 64.38 GiB |
| Logged KV occupancy peak | 8.4% | 48.2% |

The complete coverage strategy reached 18 **3.64 times faster** in these runs.
This is a comparison of the requested strategies, not a controlled ablation
of intermediate extraction alone: prompts, concurrency, sample seeds,
generation budgets, continuation, and total context differ. Both use the same
model, sampling temperature/top-p, grader toll, and initialization exclusion.
The solved sets overlap on 15 questions: coverage additionally solved 7, 9, 26;
the naive baseline instead solved 11, 12, 23. The twelve stopped questions
in each run have unmeasured final pass@4 accuracy.

In the [coverage run](../20261003T200718.717581Z/README.md), fresh first-pass
prompts on a warmed server had median/p95 TTFT 3.475/3.527 seconds, versus
0.198/0.228 seconds on continuations. All 15 continuation prefixes exactly
matched saved prompt-plus-output IDs, and three additional questions solved
in that round. Aggregate prefix hits rose from 27.2% to 95.4%. Per-request
cache details were unavailable on cancelled continuations; different
batch sizes and prefill also affect the TTFT comparison.

## First solved timestamps

These client observations are linked to grader timestamps and full query IDs
in `summary.json` and remote `solved.jsonl`.

| Question | Winning rollout | Answer | First solved elapsed (s) | Client UTC | Grader UTC |
| --- | --- | --- | ---: | --- | --- |
| 1 | 2 | 70 | 32.836 | 2026-10-03T20:23:36.952+00:00 | 2026-10-03T20:23:36.939435Z |
| 17 | 4 | 49 | 40.378 | 2026-10-03T20:23:44.493+00:00 | 2026-10-03T20:23:44.481173Z |
| 16 | 3 | 468 | 50.741 | 2026-10-03T20:23:54.857+00:00 | 2026-10-03T20:23:54.846384Z |
| 3 | 4 | 16 | 53.738 | 2026-10-03T20:23:57.854+00:00 | 2026-10-03T20:23:57.846767Z |
| 4 | 2 | 117 | 63.537 | 2026-10-03T20:24:07.652+00:00 | 2026-10-03T20:24:07.643089Z |
| 6 | 4 | 504 | 68.120 | 2026-10-03T20:24:12.235+00:00 | 2026-10-03T20:24:12.225043Z |
| 19 | 2 | 106 | 74.215 | 2026-10-03T20:24:18.330+00:00 | 2026-10-03T20:24:18.320010Z |
| 22 | 2 | 237 | 88.356 | 2026-10-03T20:24:32.471+00:00 | 2026-10-03T20:24:32.466386Z |
| 21 | 1 | 293 | 99.414 | 2026-10-03T20:24:43.530+00:00 | 2026-10-03T20:24:43.524041Z |
| 8 | 3 | 77 | 102.409 | 2026-10-03T20:24:46.525+00:00 | 2026-10-03T20:24:46.524309Z |
| 25 | 3 | 907 | 122.300 | 2026-10-03T20:25:06.416+00:00 | 2026-10-03T20:25:06.415127Z |
| 27 | 4 | 19 | 193.238 | 2026-10-03T20:26:17.354+00:00 | 2026-10-03T20:26:17.352598Z |
| 24 | 2 | 149 | 198.097 | 2026-10-03T20:26:22.213+00:00 | 2026-10-03T20:26:22.212209Z |
| 5 | 4 | 279 | 203.166 | 2026-10-03T20:26:27.281+00:00 | 2026-10-03T20:26:27.276432Z |
| 18 | 1 | 82 | 208.324 | 2026-10-03T20:26:32.439+00:00 | 2026-10-03T20:26:32.438740Z |
| 23 | 3 | 610 | 258.429 | 2026-10-03T20:27:22.544+00:00 | 2026-10-03T20:27:22.543263Z |
| 12 | 2 | 510 | 261.429 | 2026-10-03T20:27:25.544+00:00 | 2026-10-03T20:27:25.543605Z |
| 11 | 4 | 259 | 336.497 | 2026-10-03T20:28:40.612+00:00 | 2026-10-03T20:28:40.608001Z |

Raw traces, continuous usage, sampler rows, first-solve records, and service logs
remain at
`/home/azureuser/aime-bench/attempts/20261003T202152.418590Z/`.
Original compact `config.json` and `summary.json` are versioned beside this report.
Observed token totals include client-visible partial work and may omit generation
already performed after the final received chunk. Cancelled stream durations
are censored. Cache occupancy and VRAM peaks are sampled observations.
