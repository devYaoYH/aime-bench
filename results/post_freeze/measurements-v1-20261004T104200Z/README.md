# Post-freeze measurements: 4 October 2026

Both declared tasks completed without failure or replacement. [Summary tables and report paragraphs](summary.md), [control and token audit](analysis.json), and [extended figure](extended-milestones.png) contain the results. Task A: 5/5 reached 18, median 78.305s (63.908–135.071s), with 26/27/26/27/26 solved at request-budget exhaustion. Task B: 57/120 mean sample correctness, 18/30 pass@4 and unique-plurality votes, 13/30 strict three-of-four votes; 63/120 samples hit the cap with no eligible final answer.

## Execution and source

Source commit: `9258e9875a2504a5ea92ea6778ddd3101fe45738`. Source edits were developed and tested locally, committed/pushed, then pulled into the remote primary checkout. GPU jobs ran sequentially in the clean pinned worktree `/home/azureuser/aime-bench-post-freeze-9258e987` and durable tmux `post-freeze-9258e987`:

```sh
~/.venvs/vllm/bin/python -m src.experiments.post_freeze \
  --execute --batch measurements-v1-20261004T104200Z
```

[Config](config.json) records each exact command, protocol/preset/prompt/source hashes and resolved model profiles. The [implementation diff](implementation-diff.patch) and [baseline policy diff](baseline-policy-diff.patch) describe the additions. Canonical and frozen cores, the original naive baseline and grader remain unchanged. Frozen v1's 21-file manifest SHA-256 is `35d6a06315a0e45441b3ac49bd468a2b94553fb172f378bfebc7ea8cc044070f`. Task A uses `prompt_adherence.json`; its prompt SHA-256 is `26b591c39bcf55f4c94f5359dcc90d3c5626a524478eee1c5ce44b5038162364`.

Task A retains the 30×1 barrier, 8K initial and 16K additional exact-ID continuations, four total requests/question, NVFP4 Marlin, FlashInfer, BF16 KV, 65,536 context and 95% allocation. Only target 18 becomes 30, with an external 900-official-second ceiling. Seeds 20261011–20261015 reuse one owned server, reset prefix cache, launch fresh graders and use cheap 30×32-token warmups. All 150 initial payloads match their same-seed controls; all 126 continuation prefixes match parent token IDs.

Task B uses original BF16 VibeThinker-3B, 16K total context, 30×4, original prompt/parser/sampling and seed 20261003. Its requested memory allocation is 95%, instead of the historical timing baseline's 80%. The separate runner disables sibling/global cancellation and completes all 120 samples. Original naive warmup remains 120×32. Unique question/answer pairs are checked once and mapped to every matching sample. Missing outputs abstain for unique-plurality voting; ties/no votes fail. Strict three-of-four voting is separately labeled. This is one complete accuracy seed, not an isolated memory/cancellation comparison.

The approved exception retains existing answer-field startup provenance validation only. No reference answers are passed to generation or read by the new analysis/scoring. Every correctness label comes from a recorded grader verdict. The serial grader's three-second toll is unchanged.

## Evidence and verification

[Task A](task_a.json), [Task B](task_b.json), [first-correct ranks](first-correct-ranks.csv), [per-seed milestones](milestones-per-seed.csv), [120 baseline sample mappings](baseline-samples.csv), [per-question votes/counts](baseline-questions.csv) and the six linked attempt folders preserve every outcome. Unreached times remain null. A min/max band is an observed range, not a confidence interval. Historical E8 is a separate reference, not a curve spliced into the new runs.

Local pre-execution validation passed 261 offline tests, including nine new protocol/baseline checks. Independent post-run analysis validates source/profile/prompt controls, first-correct query IDs and true verdicts, every exact-ID continuation, all 120 terminal baseline samples and deduplicated verdict mappings:

```sh
.venv/bin/python -m scripts.analyze_post_freeze \
  --batch measurements-v1-20261004T104200Z
.venv/bin/python -m src.attempt_metadata --all
```

[Service metrics](service-metrics.json) summarize full remote vLLM logs, with hashes: maximum periodic KV occupancy was 13.7% in A and 49.5% in B; maximum observed waiting requests was zero. No OOM/preemption/eviction messages were found. This log audit is not an instrumented zero-eviction claim. B's 2,143 device-wide NVML samples peak at 78,560.188MiB (76.719GiB); A's optional GPU telemetry is disabled by benchmark mode. Both owned services stopped successfully, exit status was zero, and a final remote check found no GPU compute processes or listeners on 8000/8077.

Required client requests/responses/tokens and timing/verdict evidence are under `attempts/`. Full SSE streams, grader audits and raw service logs remain remote and out of Git. Log maxima come from periodic `Running/Waiting/GPU KV cache usage` lines; raw log paths and hashes are in the metric artifact. [Time allocation](time-allocation.md) logs each unattended job separately from development/reporting wall time, without inventing earlier human hours.
