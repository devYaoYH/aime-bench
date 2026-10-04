# Frozen runner contract

The measured AIME submission is **core v1 with `presets/prompt_adherence.json`**:
5/5 declared seeds reached 18, median **77.277s**, range **62.783–82.492s**.
Start with the [repository quickstart and execution diagram](../README.md#run-the-measured-core-v1).
The [validation report](../runs/experiments/frozen-core-prompt-five-seeds-20261004T005416Z/README.md)
and [evidence packet](../docs/reports/final/evidence.md) contain the timing records.

## Core v1: measured policy

```bash
~/.venvs/vllm/bin/python -m runner_final.run_frozen \
  --preset runner_final/presets/prompt_adherence.json --seed 20261011

# All five declared seeds, with no replacement or settling trials.
~/.venvs/vllm/bin/python -m runner_final.validate_frozen
```

Run from the repository root on the provided node, with an available GPU and
ports. The validation controller starts one owned inference server, resets prefix
cache and starts a fresh grader before each trial. Each trial uses only the
30-stream, 32-token arithmetic warmup. The single-run command starts its own
services unless `--reuse-server` attaches to an idle matching inference server;
that external server is left running.

### Generation and verification

The measured preset starts 30 questions with one stream each and an 8,192-output-
token cap. Complete integer boxes, standalone answer lines and recognized prose
answer clauses become prospective candidates. Per-question deduplication spans
all rounds; one verifier call per question can be outstanding, and the grader
services all questions serially at three seconds per check. A wrong verdict
leaves generation running. A correct verdict cancels the question's stream;
the eighteenth distinct first positive verdict supplies the reported stop time.

The barrier waits for each current-round question to complete generation and
its pending verifications, or to become solved, before starting another round.
For every unsolved question with remaining budget, the next round issues one
request:

- **Continue** if the last generation ended at its output cap and complete exact
  token IDs leave room in the 65,536-token total context.
- **Start fresh** if generation ended naturally or the context is exhausted.

Later requests allow up to 16,384 additional output tokens, clipped to remaining
context for continuations. The first request, every continuation and every fresh
sample count toward **four total requests per question**. With the measured
`--rollouts 1` preset, there is one active generation per question and at most 30
overall. A continuation and a fresh sibling are never launched together.
Generator errors fail the attempt; they are not treated as ordinary wrong answers.
Grader queries are separately metered and are not capped at four.

These semantics are implemented by
[`group_options` and `run_speedrun`](core_v1/runner.py),
[`continuation_prefix` and `run_question`](core_v1/_streaming.py), and
[`warm_inference`](core_v1/_runtime.py). Saved initial and continuation requests
are available in the five-seed attempt traces.

### Presets and frozen identity

| Preset | Status |
| --- | --- |
| [prompt_adherence.json](presets/prompt_adherence.json) | Measured final preset; stronger prospective-answer prompt; 30 x 1 / 8K |
| [baseline.json](presets/baseline.json) | Original-prompt control; selected when `--preset` is omitted |
| [30x2_4k.json](presets/30x2_4k.json) | Prepared, untested two-stream / 4K alternative; different from the final policy |

[core_v1/manifest.json](core_v1/manifest.json) pins the core and runtime
dependencies. Startup rejects hash drift. Prompts, presets and model profiles
are separately hashed configuration: `--preset FILE` selects a preset,
`--system-prompt-file FILE` overrides its prompt, and remaining CLI arguments
override preset defaults. Changing core behavior requires a new version.

The final profile is NVFP4/Marlin weights, BF16 activations/KV, FlashInfer
attention, 95% GPU allocation and 65,536-token total context; see
[vllm-flashinfer.yaml](vllm-flashinfer.yaml). Weights and the deployed profile
must already exist under `~/models/r0b0tlab/VibeThinker-3B-NVFP4/`.

### Timing and artifacts

Each `attempts/<UTC timestamp>/` directory records the resolved configuration,
Git commit, core/prompt/preset/profile hashes and runtime package versions.
Per-question traces retain requests, responses, exact token IDs, generation
and verification timestamps, and time to first token. `solved.jsonl` records
distinct first-solved verdicts; `summary.json` reports whether the target was met.

`time_to_target_s` starts after initialization and arithmetic warmup and ends at
the eighteenth positive verdict. `official_latency_s` also includes cancellation
settlement; initialization, final trace flush and service cleanup are separate.
Benchmark mode buffers required evidence in RAM and disables optional CPU/GPU/
engine profiling. Graceful interruption can flush partial records; a hard crash
can lose buffered evidence. Detailed server/SSE/grader logs remain remote.

The [AIME 2026 transfer run](../runs/experiments/frozen-core-aime2026-lightweight-20261004T011005Z/README.md)
used the same v1 preset and reached 18 in **88.669s**, with one wrong check.
It is one fresh-server trial, separate from the five-seed 2025 statistics.

The [Nsight Compute report](../runs/profiling/core-v1-ncu-20261004-single-pass/README.md)
profiles three decode graphs from one completed core v1 diagnostic attempt.
It preserves the frozen core and final solving settings, but its timing is
unranked because profiling perturbs execution. The report includes the measured
roofline, context traffic, limitations and the retained failed initial capture.

The [per-question token and tail analysis](../runs/analyses/core-v1-five-seeds-answer-tokens/README.md)
replays the selected five v1 runs' original SSE traces. It separates winning
candidate production from verification latency and identifies the last two
positive-verdict slots; it does not launch or simulate an alternative policy.

## Core v1.1: long fresh rollouts

The separately frozen [v1.1 policy](core_v1_1/README.md) removes the round barrier
and uses fresh-only requests with 64K total context minus the served prompt,
one active generation per question and at most 30 overall. Its
[five-seed benchmark](../runs/experiments/core-v1_1-five-seeds-20261004T083800Z/README.md)
reached 18 in 5/5 trials, median **77.652s**, range **60.906–92.096s**. Two historical
same-seed comparisons were faster, three slower. All reached the target on their
30 initial long rollouts; fresh retries were not exercised. The measured
submission remains v1 (median 77.277s).

```bash
~/.venvs/vllm/bin/python -m runner_final.validate_v1_1
```

This uses one owned server lifetime with prefix-cache reset, a fresh grader and
arithmetic warmup per trial. Provision the separate
`configs/vllm/r0b0tlab/VibeThinker-3B-NVFP4/vllm-v1_1-long64k.yaml` beside the model
weights first. Length and retry/scheduling changes are combined in this version.

## Core v1.5: exploratory eager allocation

The independently frozen [v1.5 policy](../src/attempt_runners/README.md#core-v15-eager-30-slot-experiment)
retains v1's adherence prompt, sampling, warmup and token/request budgets, while
replacing the barrier with a 30-slot pool that prefers ready continuations, then
fresh starts on least-active unsolved questions. The [five-seed series](../runs/experiments/core-v1_5-five-seeds-20261004T074500Z/README.md)
reached 18 in 5/5 trials, median **77.352s**, range **65.015–85.464s**.
Only one historical same-seed comparison was faster; the median did not improve
on v1's 77.277s despite 71.8% more generation requests. V1.5 used five fresh server
lifetimes versus v1's shared server. The measured submission remains core v1.

```bash
~/.venvs/vllm/bin/python -m src.attempt_runners.speedrun_v1_5 --seed 20261011
```

## Frozen core v2: general mathematical answers and grader-fed questions

Core v2 is a capability extension with the same barrier scheduling and
four-request cap. Its [AIME 2025 back-test](../runs/experiments/core-v2-aime2025-five-seeds-20261004T013100Z/README.md)
reached 18 in 5/5 trials: median **113.415s**, range **86.088–145.437s**.
Every trial was slower than its historical same-seed v1 control. Of 62 wrong
checks, 54 were placeholder boxes such as `EXPRESSION`, `...` or `?`.
The batches changed prompt and extraction together and ran sequentially;
this does not isolate the parser's effect. All outcomes and both cores are preserved.

```bash
# General-answer v2 on AIME 2025; repeat the declared batch with backtest_v2.
~/.venvs/vllm/bin/python -m runner_final.run_frozen_v2 --seed 20261011
~/.venvs/vllm/bin/python -m runner_final.backtest_v2

# Pinned Apex shortlist (47 questions).
~/.venvs/vllm/bin/python -m runner_final.run_frozen_v2 \
  --preset runner_final/presets/apex_core_v2.json --seed 20261011

# Custom dataset or an existing fresh v2 grader.
~/.venvs/vllm/bin/python -m runner_final.run_frozen_v2 \
  --grader-config /absolute/path/to/grader.yaml --seed 20261011
~/.venvs/vllm/bin/python -m runner_final.run_frozen_v2 \
  --reuse-grader --grader-port 8077 --seed 20261011
```

V2 obtains indices and statements from `GET /questions`, validates service
fingerprints and saves `questions.json`. Its parser accepts closed balanced
`\boxed{...}` / `\fbox{...}` expressions or complete standalone `Answer:` lines,
including fractions, radicals and variables. It omits unmarked prose extraction.
Empty or over-4096-character candidates are ignored; deduplication uses trimmed
strings, with mathematical equivalence decided by the grader.

`--reuse-grader` requires a fresh v2 service with zero completed queries and the
configured toll; it leaves the service running. For an owned custom grader,
YAML selects `jsonl`, `csv`, `parquet` or `hf` and the index, problem and gold
fields. HF/parquet need their optional loader dependencies. An external grader
can be started with `GRADER_CONFIG=/path/to/grader.yaml python grader/server_v2.py`.
The original grader has no question endpoint; v2 uses `grader/server_v2.py`.

[data/source_apex_shortlist.json](../data/source_apex_shortlist.json) pins the Apex
bundle. Apex #25/#26 overlap AIME 2025 P14/P15, so it is not a fully unseen
transfer set. The [v2 manifest](core_v2/manifest.json) freezes the new core.
Validate v2 artifacts with `python -m runner_final.core_v2.metadata ATTEMPT_DIRECTORY`;
the historical v1 metadata command is `python -m src.attempt_metadata`.

## Historical policies

`run.py`, `run_v1.py`, `validate.py` and `validate_v1.py` preserve earlier final-
runner iterations; their names predate the immutable `core_v1` / `core_v2`
versioning. Historical `--benchmark-prewarm` runs used an additional ungraded
AIME 2024 workload. The measured prompt-adherence preset uses only the cheap
arithmetic warmup. Those older workloads and the 59.316s best draw remain
separate evidence in the [experiment archive](../runs/README.md).
For staged fan-out and dynamic allocation, see the
[historical runner catalogue](../src/attempt_runners/README.md).
