# Working on aime-bench

## Local development and remote experiments

- Write and edit source files in the local checkout. Run relevant offline tests
  locally before committing; run the full suite when changing shared behavior.
- Commit and push the tested changes to this repository. Then use `ssh callosum`,
  change to `/home/azureuser/aime-bench`, and run `git pull --ff-only` on the
  corresponding branch before executing experiments. Check both checkout states
  and commit IDs; preserve unrelated local and remote changes. Do not copy edited
  source files directly to the remote or develop there.
- The remote is an A100 PCIe 80GB machine. Inspect `nvidia-smi`, active processes,
  and service health before launching workloads. Do not terminate an unrelated
  experiment or inference server. Use an explicitly requested model or document
  the choice for a smoke experiment.
- Remote model weights and launch configurations are in
  `/home/azureuser/models/<organization>/<model>/vllm.yaml`. Read that profile and
  `~/models/README.md` rather than inventing launch settings. vLLM uses
  `~/.venvs/vllm/bin/python` and `~/.venvs/vllm/bin/vllm`.
- Use `python -m src.attempt --model <organization>/<model>` for canonical
  attempts. It manages warmup, vLLM, the vendored grader, traces, and telemetry.
  `--reuse-server` deliberately attaches to an idle matching server; it must not
  stop that server. The default starts its own server and rejects occupied ports.
- Use `tmux` or another durable job session for long remote runs. Save command,
  Git commit, model config, timestamps, and results. Inspect progress and errors;
  a launched process alone does not establish success.
- Version canonical attempt evidence needed for the viewer and analysis under
  `attempts/`: configs, summaries, reports, saved requests/responses, exact token
  records, question/round records, verification events, and GPU samples. Review
  artifact sizes before adding new attempts. Keep full SSE streams, grader audits,
  service logs, credentials, weights, and virtual environments out of Git.
- After importing attempts, run `python -m src.attempt_metadata --all` and annotate
  each `metadata.json` intervention/reference against its recorded configuration.
  Preserve historical runner versions and source commits; unknown controls stay
  null. The overall viewer compares distinct first-solved verdict times, including
  the 54-second serial grader floor; do not substitute settlement or rank unmet runs.
- To bring back results, copy only experiment artifacts (for example with `scp`),
  review them locally, and commit/push reports from the local checkout. Do not
  commit or push code from the remote machine.

## Checks

Run from this repository root:

```sh
.venv/bin/python -m unittest discover -s test -v
```

Tests are offline. Some legacy tests need local ignored traces or macOS sandbox
support; report those limitations rather than modifying fixtures to hide them.

## Coverage experiments

- Keep the canonical `src/attempt.py` policy unchanged when adding experimental
  baselines. Put independently runnable, versioned policies and their tests in
  `src/attempt_runners/` and update its README catalogue. Use the isolated
  `src.attempt_runners.naive_pass4_v1` entry point for the naive final-only baseline.
- Record the runner version in new experimental artifacts. Preserve the original
  source commit and command in historical attempt records after reorganizing code.

- Keep the per-question generation request cap at four initially. Continuation
  segments count toward the cap. Stop at the configured solve target (initially
  18), or report that the target was unmet after exhausting the budget.
- Use 8,192 tokens for the coverage-first pass, then continue capped unsolved
  trajectories with exact token IDs when context permits. Measure prefix-cache
  hits and TTFT; do not assume that a completed request retains active KV state.
- Preserve first-solved timestamps and elapsed times, linked to the grader query
  and answer timestamp, in attempt logs and summaries.

## Speedrun sweeps and overhead

- Use `src.attempt_runners.speedrun_v1` for prospective concurrency experiments;
  use `src.attempt_runners.sweep_speedrun_v1` to plan the versioned manifest in
  `configs/sweeps/vibe-speedrun-v1.json`. Planning is the default and does not
  launch GPU work. Only use `--execute` after the GPU's current experiment ends.
- Preserve the four-request cap even for pass@4: four first-pass samples leave
  no continuation budget. Smaller fan-outs resume each distinct lane's exact IDs.
- Canonical instrumentation is allowed without changing its solving policy.
  Keep synchronous CPU/IO separate from overlapping async wall waits. Measure
  duplicate suppression, local/grader queues, parsing, writes, cancellation,
  event-loop delay, and inference metrics; record missing metric observations.
- Compare against the matched barrier control with paired seeds. Report target
  unmet/failure honestly; do not rank an attempt that did not reach 18 correct.
