# Frozen runner core v1

The public entrypoint is `python -m runner_final.run_frozen`. It uses the
[baseline preset](presets/baseline.json), preserving final v2's 30×1 / 8K
barrier policy and inexpensive warmup. Historical `run.py`, `run_v1.py` and
validation drivers remain available with their original semantics.

```sh
~/.venvs/vllm/bin/python -m runner_final.run_frozen --seed 20261011
```

[core_v1/manifest.json](core_v1/manifest.json) pins 21 source/dependency files.
Startup verifies their hashes before creating an attempt or launching services;
drift aborts the run. Scheduling, streamed candidate parsing, exact-ID
continuations, per-question deduplication and verification serialization,
correct-verdict cancellation, target stopping, service ownership, buffered
storage and grader behavior are frozen. Behavior changes require a new core
version, not rewriting this manifest in place.

The published five-run statistics and AIME 2026 transfer check use
`--preset runner_final/presets/prompt_adherence.json`; the default baseline
preset preserves the original prompt as a control. `python -m
runner_final.validate_frozen` repeats the declared five-seed configuration.

Prompts, parameter presets and deployed model profiles are configuration. Every
attempt saves the resolved arguments, prompt bytes/hash, preset path/hash, core
manifest hash, Git commit, runtime package versions and deployed profile snapshot.
Changing a preset or prompt therefore changes the experiment while retaining the
same core identity. `--system-prompt-file FILE` overrides the preset prompt;
remaining CLI arguments override preset defaults.

- [baseline.json](presets/baseline.json): selected 30×1, 8K first pass, original prompt.
- [prompt_adherence.json](presets/prompt_adherence.json): same settings, stronger instructions to emit a prospective boxed answer before rechecking.
- [30x2_4k.json](presets/30x2_4k.json): two 4K samples per question and the stronger prompt; four requests total leave one continuation per lane.

Neither alternative is an established strategy improvement. The improved-prompt
preset has since been measured below; 30×2/4K remains untested. No new experiments
were launched during the freeze itself. The baseline
remains selected. The next sweeps can vary configurations without duplicating
solving code. The historical warmed results and the per-question token
[distribution report](../runs/experiments/runner-final-five-seeds-20261004T000926Z/EXTRACTION_TOKENS.md)
remain separate from these untested configurations.

The subsequently requested improved-prompt replication uses
`python -m runner_final.validate_frozen`, with all five seeds declared in
[five_seeds_prompt_core_v1.json](five_seeds_prompt_core_v1.json). It runs the
immutable core with the prompt-adherence preset, retaining all outcomes and
checking initial payloads, core/prompt hashes, correct verdicts and request caps.
It reuses one owned server with fresh graders, cleared prefix cache and cheap
warmup per trial. No settling run or seed replacement is allowed. The predeclared 71.135s gate remains in the raw protocol; the report now
leads with the measured five-run median, range and target success count.

## Historical final v2 and validation

Run from the repository root on the remote machine after local tests, commit,
push and `git pull --ff-only`:

```sh
~/.venvs/vllm/bin/python -m runner_final.run --benchmark --seed 20261011
~/.venvs/vllm/bin/python -m runner_final.validate
```

`validate` runs exactly the five seeds declared in [five_seeds.json](five_seeds.json),
sequentially on one owned inference server. Every trial is scored, including the
first; failures remain in the report. Each has a fresh serial grader, cleared
prefix cache, and the inexpensive 30-stream, 32-token warmup. It checks the deployed profile against
the [frozen profile](vllm-flashinfer.yaml) and requires clean tracked source.
The model weights and matching profile live under
`~/models/r0b0tlab/VibeThinker-3B-NVFP4/`.

| Setting | Default |
| --- | --- |
| Weights / activation / KV | NVFP4 Marlin / BF16 / BF16 |
| Attention / GPU allocation | FlashInfer / 95% |
| Total model context | 65,536 tokens |
| Test set / target | All 30 AIME 2025 questions / 18 distinct verified correct |
| Initial concurrency | 30 questions × one rollout |
| Retry schedule | Barrier: finish the current coverage round before the next |
| Generation budget | First request 8,192; subsequent requests up to 16,384 additional tokens |
| Question budget | Four generation requests, including continuations |
| Sampling | Temperature 0.8, top-p 0.95; seed + question index × 4 + rollout |

The solving policy, streaming parser, service lifecycle and port checks are copied
into this directory from the tested v2 policy at `ae9c37e`. Canonical and historical
experimental runners are preserved. Shared dataset, artifact storage, observational
metrics and metadata utilities remain repository dependencies and are pinned by
the recorded source commit. New solving changes should receive a new final version.

Closed integer boxes and complete answer clauses are prospective candidates.
Each question deduplicates them and permits one outstanding verifier call. A
correct verdict cancels its streams; the 18th distinct solved verdict ends the
attempt. Capped unsolved outputs continue only with complete exact token IDs;
natural completion or context exhaustion can start a fresh sample. There is no
grader-pending stream suspension or dynamic fan-out in this selected policy.

The default uses only the inexpensive 30-stream, 32-token warmup.
`--benchmark-prewarm` explicitly enables the ungraded workload: before the grader
starts, all 30 AIME 2024 questions receive one ungraded sample
with the same prompt and sampling settings, capped at 8,192 output tokens. This
stage does not parse candidates or inspect correctness. It runs to natural
completion or the cap, records requests/responses and per-request latency under
`prewarming-2024/`, and writes `prewarm.json`. Prefixes are cleared before and
afterward without resetting running requests. `--skip-benchmark-prewarm` is an explicit spelling of the current default. The
normal 30-stream, 32-token inference warmup is retained in both modes.

Warmup may exercise paths that a short request does not, but improvement has not
been established. vLLM already performs graph warmup during initialization;
[its compilation configuration](https://docs.vllm.ai/en/v0.30.0/api/vllm/config/compilation/)
describes that behavior. A workload warmup does not replace startup compilation.
The historical five warmed v1 trials measure that configuration; comparison with
historical trials cannot isolate warmup's causal benefit. A paired warmed/unwarmed experiment
would be needed for that claim.

Official timing starts only after initialization and both warmup stages. Reports
retain total initialization cost as well as time to the 18th correct verdict.
`--benchmark` disables optional GPU, engine and CPU profiling and buffers required
client evidence in RAM until timing ends. Exact tokens, request/response records,
rollout timestamps, TTFT and first-solved events remain. Trace flush and service
cleanup cost are recorded separately. Server/grader logs stay on the remote machine;
optional GPU samples and eviction counters are unavailable in benchmark mode.

Selection is based on the fastest measured completed run, **59.316 seconds**, using
NVFP4 FlashInfer 30×1. Previous independent scored repeats were **87.356, 59.316,
86.574 seconds**; consistent sub-71.135-second performance is unconfirmed. The
pending-verdict alternative repeated at **95.800, 99.235, 83.064 seconds**. The five historical warmed v1 trials reached 18 in 92.061, 105.337, 66.481,
78.123 and 78.601 seconds (median 78.601; only one of five within 71.135s).
The short-warmup default has since been restored. The [improved-prompt
five-seed core validation](../runs/experiments/frozen-core-prompt-five-seeds-20261004T005416Z/README.md)
reached 18 in 71.321, 77.277, 82.492, 81.102 and 62.783 seconds (median 77.277;
only 1/5 within 71.135s). It kept the core fixed and used one owned server lifetime;
the supported performance claim is 5/5 reached 18, median 77.277s and range
62.783–82.492s across the declared seeds on one server. The original-
prompt short-warmup five-seed protocol has not been executed as a group. Results must not be used to
select replacement seeds.

The first final-runner version and its warmed protocol are preserved as
[run_v1.py](run_v1.py), [validate_v1.py](validate_v1.py), and
[five_seeds_warmed_v1.json](five_seeds_warmed_v1.json). Their original source
commit `ccca184` and historical attempt records stay unchanged. The user requested
restoring the cheaper warmup after the five warmed trials; v2 changes that default,
not the generation/grading policy. AIME 2024 warming remains opt-in.

The lightweight AIME 2026 transfer protocol is [predeclared here](../configs/experiments/vibe-frozen-core-aime2026-lightweight-v1.json): one new seed, all 30 questions, target 18, the same improved-prompt preset and core.

[The lightweight AIME 2026 transfer check](../runs/experiments/frozen-core-aime2026-lightweight-20261004T011005Z/README.md) reached 18 verified correct in **88.669s** with one wrong check, using the unchanged improved-prompt core. This is one predeclared seed, separate from the five-run 2025 headline statistics.

## Frozen core v2: general mathematical answers and grader-fed questions

Use `python -m runner_final.run_frozen_v2`. This is a separate immutable core;
v1 sources, grader and manifest are unchanged. V2 retains 30×1 barrier scheduling,
8K initial / 16K later requests, at most four requests per question including
continuations, the serial grader toll, first-solved timing, and benchmark storage.
No v2 GPU performance result has been measured yet.

```sh
# Start an owned v2 grader using the pinned Apex bundle (47 questions).
~/.venvs/vllm/bin/python -m runner_final.run_frozen_v2 \
  --preset runner_final/presets/apex_core_v2.json --seed 20261011

# Let a fresh, already-running v2 grader choose the dataset.
~/.venvs/vllm/bin/python -m runner_final.run_frozen_v2 \
  --reuse-grader --grader-port 8077 --seed 20261011

# Start an owned grader with any supported dataset specified in a YAML file.
~/.venvs/vllm/bin/python -m runner_final.run_frozen_v2 \
  --grader-config /absolute/path/to/grader.yaml --seed 20261011
```

The runner obtains question indices and statements from `GET /questions`, never
reads a local answer key, validates the API/health fingerprints and saves a
gold-free `questions.json` snapshot. `--questions 1 2` selects API-provided indices;
the default selects every available question. `--reuse-grader` requires a fresh
v2 service with zero completed queries and the configured toll (default 3s).
It leaves that service running. `--reuse-server` independently attaches to an
idle matching vLLM server. External grader audit logs stay with its owner;
client verification and first-solved evidence are always saved.

Start an external v2 grader with `GRADER_CONFIG=/path/to/grader.yaml
python grader/server_v2.py`. Its dataset configuration supports `jsonl`, `csv`,
`parquet` and `hf`, with `idx_field`, `problem_field` and `gold_field`. JSONL rows
need an index, a nonempty statement and an exact answer. HF/parquet require their
optional loader dependencies. The original grader has no `/questions` endpoint;
use the v2 service for this entrypoint. The endpoint is ungraded and free; `/verify`
uses the original FIFO worker and returns verdicts without gold answers.

The [math prompt](prompts/math_core_v2.txt) requests prospective exact boxed
expressions for the original requested quantity, including variables, signs,
fractions and radicals. Extraction accepts a fully closed balanced `\boxed{...}`
or `\fbox{...}`, including nested/escaped braces across chunks and exact-ID
continuations, or a complete standalone `Answer:` line. It does not interpret
unmarked prose or impose the AIME 0–999 restriction. Empty or over-4096-character
candidates are ignored. Duplicates are suppressed per question using the trimmed
expression string; algebraic equivalence remains the grader's job.

[data/source_apex_shortlist.json](../data/source_apex_shortlist.json) pins the
MathArena Apex revision and input hashes. Its prompt file excludes answers; its
key is used only by the grader. Reproduce the bundle with
`python scripts/import_apex_shortlist.py` (development dependencies: httpx,
pyarrow). Apex #25 and #26 reuse AIME 2025 P14/P15, so this is not a fully unseen
transfer set. Generalization claims must account for that overlap.

[core_v2/manifest.json](core_v2/manifest.json) freezes the v2 runner, extraction,
question API, service and runtime dependencies. Startup rejects drift. Prompt,
preset, profile and dataset choices remain hashed configuration. Validate v2
metadata with `python -m runner_final.core_v2.metadata ATTEMPT_DIRECTORY`; the
legacy `src.attempt_metadata` CLI remains AIME-only. The attempt viewer uses the
saved question snapshot for v2; the results API records the dataset identity,
and Apex results are excluded from the AIME year charts.
