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
