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
- Use `python -m runner --model <organization>/<model>` for canonical
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

- Keep historical `src/attempt.py` and frozen policies unchanged when adding experimental
  baselines. Put new independently runnable, versioned policies in
  `runner/extensions/<version>/`, reuse `runner/lib` where suitable, and update
  its catalogue and focused tests. Existing `src/attempt_runners` policies are
  historical. Use the isolated
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

- The user-requested v1.6 extension changes the cap to at most four **fresh
  trajectories** per question. Exact-ID continuation segments are separate and
  may extend each trajectory from an 8K first segment to at most 64K cumulative
  output, clipped to served context, or natural completion. Use a v1.5-style slot
  pool, defaulting to the selected question count. Keep v1/v1.1/v1.5 measured
  policies and manifests unchanged; launch with `python -m runner.extensions.v1_6`.

## Speedrun sweeps and overhead

- The canonical v1 entrypoint is `python -m runner`; usage is in `runner/README.md`.
  Its refactored source is pinned by `runner/manifest.json`. Versioned policies
  live under `runner/extensions/`; `runner_final` aliases `runner/extensions/variants`
  for historical commands/manifests. Keep archived bytes and identities unchanged.
  New extensions reuse `runner/lib`; promotion requires an explicit decision,
  matched validation/evidence, a canonical registry update and documentation.
  The original measured v1 entrypoint remains `python -m runner_final.run_frozen`.
  `runner_final/core_v1/manifest.json` pins the core and repository-owned runtime
  dependencies; startup rejects hash drift. Do not modify v1 behavior or simply
  refresh its manifest. Create a new core version for behavior changes. Keep
  canonical prompts and hyperparameter presets under `runner/prompts/` and
  `runner/presets/`; keep version-specific configuration under
  `runner/extensions/<version>/`. Preserve the archived prompts/presets used by
  existing measurements. Record resolved inputs and their hashes. The baseline
  is 30×1 barrier, initial 8K, four requests including continuations, cheap warmup
  and benchmark mode. The improved prompt has five scored trials: 5/5 reached 18, median
  77.277s, range 62.783–82.492s on one server. Use these batch statistics as the
  headline; historical best timing is secondary context. The 30×2/4K preset is untested. Core freeze alone is not evidence of repeatability.

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

- For benchmarking without optional instrumentation, use canonical `--benchmark`
  or `src.attempt_runners.speedrun_v2 --benchmark`. Record this as a distinct
  telemetry/storage configuration: GPU/engine samples are unavailable, required
  timing/verdict/token evidence stays in RAM, and final trace flush time is outside
  official solving time. Preserve v1 comparison records. `--buffer-traces` alone
  retains profiling while deferring writes; graceful interruption flushes partial
  records, but a hard crash can lose the client buffer.

- The explicitly requested dynamic-budget experiment uses versioned
  `speedrun_v4` and may use eight requests per question, including continuations.
  This exception applies to that experiment; earlier policies retain their four
  request cap. Start each trajectory at 8K and grow cumulative generated output
  to 16K/32K/64K via exact-ID continuations, clipping to total model context.
  Reallocate freed request slots to unsolved questions up to 60 active streams.
  Save admissions, expansion trigger, budgets and actual request counts.

## Frozen mathematical core v2

- Use `python -m runner_final.run_frozen_v2` for generalized exact mathematical
  answers. Keep core v1 and its manifest unchanged. V2 obtains test questions from
  the gold-free `GET /questions` endpoint in `grader/server_v2.py`, saves a
  `questions.json` snapshot, and retains four requests per question including
  continuations. Use `--reuse-grader` only with a fresh dedicated v2 service;
  it leaves that service running. `--grader-config FILE` configures an owned service
  without hardcoding a test dataset in the runner.
- Validate core v2 attempt metadata with `python -m runner_final.core_v2.metadata
  ATTEMPT_DIRECTORY` (or `--all` for a mixed corpus). The frozen v1 metadata CLI
  cannot interpret generalized v2 dataset snapshots. Preserve the v2 manifest;
  subsequent behavior changes require a new version. The user-requested v2.1
  CPU validation/expression-key extension lives in `core_v2_1`, with a separate
  manifest, entrypoint and requirements; do not alter v1 or v2 to add it. Apex #25/#26 overlap AIME 2025;
  account for that before claiming unseen transfer. Offline tests establish
  correctness of the runner plumbing, not a new GPU performance result.

- The user-requested v2.2 runtime-correction extension is separately frozen in
  `runner_final/core_v2_2`. Preserve v2.1 and its measured source. Correction
  requests count toward four; queue and pending-validation checks precede any
  cancellation, and actual grader verdicts alone provide negative feedback.

- V2.3 is separately frozen in `runner_final/core_v2_3`, combining v1.5's shared
  slot pool with v2.2 feedback. Slots equal selected question count; ready
  continuations precede least-active fresh samples with rotating ties. Each fresh
  sample uses one long request clipped to the served 65,536 total context after
  reserving the chat-template prompt. Feedback shares the pool and four-request
  budget. Provision the distinct `vllm-v2_3-long64k.yaml` from committed repo
  configuration; do not change older profiles/manifests. Benchmark mode buffers
  required evidence, including `allocation.json`; optional telemetry is disabled.
