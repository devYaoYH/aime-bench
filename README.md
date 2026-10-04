# Single-GPU AIME speedrun

Solve AIME problems with a locally hosted open-source model and minimize time to
**18 distinct grader-confirmed answers** on one A100 80GB. The final submission
uses **core v1 with `prompt_adherence.json`**. The canonical package is
[`runner/`](runner/README.md); versioned policies live under
[`runner/extensions/`](runner/extensions/README.md).

| Evaluation | Time to 18 | Evidence |
| --- | --- | --- |
| AIME 2025, five declared seeds | **5/5 reached 18; median 77.277s; range 62.783–82.492s** | [Validation and traces](runs/experiments/frozen-core-prompt-five-seeds-20261004T005416Z/README.md) |
| Packaged canonical v1, same five AIME 2025 seeds | 5/5 reached 18; median 73.568s; range 62.118–128.744s; behavior audits passed, tail latency unresolved | [Refactor validation](runs/experiments/canonical-v1-refactor-five-seeds-20261004T215632Z/README.md) |
| AIME 2025, five post-freeze extended seeds | 5/5 reached 18; median 78.305s; range 63.908–135.071s; 26–27 correct at exhaustion | [Extended curve and full baseline](results/post_freeze/measurements-v1-20261004T104200Z/README.md) |
| AIME 2026, unchanged v1 policy | **88.669s**, one wrong check | [Single transfer run](runs/experiments/frozen-core-aime2026-lightweight-20261004T011005Z/README.md) |
| Core v1.1 long rollouts, AIME 2025 | 5/5 reached 18; median 77.652s; range 60.906–92.096s | [Five-seed long-context experiment](runs/experiments/core-v1_1-five-seeds-20261004T083800Z/README.md) |
| Core v1.5 eager30, AIME 2025 | 5/5 reached 18; median 77.352s; range 65.015–85.464s | [Five-seed scheduling experiment](runs/experiments/core-v1_5-five-seeds-20261004T074500Z/README.md) |
| Core v1.6 initial barrier then pool, AIME 2025 | 5/5 reached 18; median 77.498s; range 63.445–101.123s; smaller tail than latest refactored-v1 batch | [Five-seed pool comparison](runs/experiments/core-v1_6-barrier-five-seeds-20261004T224850Z/README.md) |
| Core v1.6 BF16 + FlashInfer, AIME 2025 seed 20261011 | 18 correct in 75.640s; zero wrong checks; all wins in initial coverage; one-run comparison | [BF16 vs NVFP4](runs/experiments/core-v1_6-bf16-flashinfer-20261004T230835Z/README.md) |
| General-answer core v2, AIME 2025 | 5/5 reached 18; median 113.415s | [Extension back-test](runs/experiments/core-v2-aime2025-five-seeds-20261004T013100Z/README.md) |

**Read the [three-page report](docs/reports/final/output/pdf/callosum-speedrun-report.pdf)
and [evidence packet](docs/reports/final/output/pdf/callosum-evidence-packet.pdf).**
The packet includes the final five-seed marginal curve (E8) and the 54-attempt
history with frontier captions (E9), a new curve through 26 correct (E10/E11),
and complete BF16 baseline accuracy at 95% memory (E12). The historical fastest draw was 59.316s;
the final performance claim is core v1's five-run median above. Those v1 2025
trials share one inference-server lifetime; 2026 is a single fresh-server transfer check.

The [Nsight Compute diagnostic](runs/profiling/core-v1-ncu-20261004-single-pass/README.md)
adds a measured decode roofline: three graph samples attain 18–49% of nominal
HBM bandwidth, with growing context traffic as the active batch shrinks.
Its instrumented timing is excluded from the scored results above.

The [token and tail analysis](runs/analyses/core-v1-five-seeds-answer-tokens/README.md)
shows exact per-question tokens and verification slots 17/18 for the selected
five runs. An 8K cumulative cutoff would discard eight of their 90 recorded wins.

## Start the runner

On a prepared Linux GPU machine, run from the repository root using the inference
Python environment. On the supplied `callosum` node:

```bash
ssh callosum
cd /home/azureuser/aime-bench
git pull --ff-only
nvidia-smi
~/.venvs/vllm/bin/python -m runner --seed 20261011
```

The default selects canonical **v1**, the improved prompt, and AIME 2025. It starts
vLLM and the grader, warms inference, and runs 30×1 with an 8K initial request,
16K subsequent requests, at most four requests per question, and a target of
18 correct. Results are saved under `attempts/<timestamp>/`; inspect
`summary.json` for `target_reached` and `time_to_target_s`. An available GPU,
model weights/profile, and runner/grader Python environments are prerequisites.
See [installation and detailed usage](runner/README.md#install-and-launch).

To select another built-in AIME year:

```bash
python -m runner --benchmark-year 2026 --seed 20261021
```

For a custom JSONL dataset containing `problem_idx`, `problem`, and `answer`,
point a grader YAML at your file:

```yaml
dataset:
  source: /absolute/path/questions.jsonl
  format: jsonl
  idx_field: problem_idx
  problem_field: problem
  gold_field: answer
  id: my_dataset
```

Then launch with a target that fits the selected question count. This example
assumes two questions; indices can be any unique positive integers:

```bash
python -m runner \
  --grader-config /absolute/path/grader.yaml \
  --system-prompt-file runner/examples/integer_prompt.txt \
  --parallelism 2 --target-correct 2
```

The grader loads the keys; the solver receives only question indices and
statements. **Current canonical v1 extracts integer answers from 0 to 999
inclusive; 1000, negative integers, fractions and symbolic expressions are outside
its extraction contract.** For broader mathematical answers, explicitly select a
[v2 extension](runner/extensions/README.md), for example `--version v2.1`.

See [dataset configuration](runner/README.md#other-datasets),
[tuning knobs](runner/README.md#tuning-knobs), and the
[execution diagram and timing](runner/README.md#policy-and-timing) for the full
runner guide. Recorded GPU results retain their original frozen source identities;
the historical commands remain available through the `runner_final` alias.

The new [v1.6 extension](runner/extensions/v1_6/README.md) runs an initial 30×1 coverage barrier, then a shared 30-slot pool,
and caps **four fresh samples per question**, with separate exact-token
continuations up to 64K cumulative output/context limit. Launch it explicitly
with `python -m runner.extensions.v1_6`; canonical v1 retains its measured
four-request policy. V1.6’s five-seed GPU comparison reached 18 in all trials,
with a 77.498s median and a smaller observed tail than the latest refactored-v1
batch; see the comparison above.

## Browse the evidence locally

Use Python 3.11+ from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r grader/requirements-local.txt
.venv/bin/python -m src.viewer_server
```

Open [attempt traces](http://127.0.0.1:8765) or
[results and configuration groups](http://127.0.0.1:8765/results).
Canonical evidence is versioned and available from a fresh clone. Historical
hosted-model raw traces and full SSE/service logs are retained separately;
the [experiment guide](docs/experiments.md) identifies their prerequisites.
OpenRouter is used for archived exploration, and is unnecessary for the final run.

## General-answer extension and experiment archive

[Core v2.1](runner/extensions/v2_1/README.md) adds CPU syntax validation and
expression-key deduplication to v2, preserving its prompt and scheduling.
Its offline replay retained all 90 previously correct candidates and rejected
56/62 wrong submissions. One [benchmark-mode trial on each dataset](runs/experiments/core-v2_1-benchmarks-20261004T161700Z/README.md)
reached 18 on AIME 2025 in 82.445s, with two wrong checks; Apex reached 13/47
at the 900s deadline. Recorded validation CPU work was 0.166s/0.809s.
These are single-trial measurements, not repeatability statistics.

Core v2 adds arbitrary mathematical expressions and grader-provided question
statements. It reached 18 in all five AIME 2025 trials but was slower in every
same-seed comparison. Its broader parser admitted prompt placeholders, causing
54 of 62 wrong checks. The [v2 contract and commands](runner/extensions/variants/README.md#frozen-core-v2-general-mathematical-answers-and-grader-fed-questions)
cover custom datasets and the Apex shortlist. Both frozen cores retain their
recorded behavior; the extension is separate from the measured v1 submission.

The [core v1.1 long-rollout experiment](runs/experiments/core-v1_1-five-seeds-20261004T083800Z/README.md)
reached 18 in all five seeds, median **77.652s** versus v1's **77.277s**. All
trials stopped on their 30 initial rollouts, so eager fresh retries were not
exercised. It used fewer requests but 2.8% more observed output IDs, with two
same-seed gains and three regressions. It does not establish a consistent speedup.

The [core v1.5 scheduling experiment](runs/experiments/core-v1_5-five-seeds-20261004T074500Z/README.md)
eagerly filled 30 request slots with continuations or fresh retries. It reached
18 in all five seeds, but did not improve the median: **77.352s versus 77.277s**.
Only one same-seed comparison was faster, despite 71.8% more generation requests.
V1.5 used a fresh server per trial; v1 shared one server lifetime. The measured
submission remains core v1.

| Location | Contents |
| --- | --- |
| [runner/](runner/README.md) | Canonical v1 package, shared libraries, datasets, grader and launch configuration |
| [runner/extensions/](runner/extensions/README.md) | Versioned policies and archived validation commands |
| [attempts/](attempts/) | Canonical run evidence and verdict timing |
| [runs/](runs/README.md) | Experiment reports, summaries and plots |
| [docs/](docs/README.md) | Report index, experiment guide and historical workflows |
| [src/](src/) | Common utilities, historical policies, exploration and viewers |
| [configs/](configs/) | Versioned launch profiles and experiment configurations |
| [test/](test/) | Offline checks |

Develop locally, test, commit/push, then pull the tested commit on the node before
GPU experiments; see [AGENTS.md](AGENTS.md). To run the offline suite:

```bash
.venv/bin/python -m unittest discover -s test -v
```

Some legacy checks require ignored raw traces or macOS sandbox support. Frozen
core manifests reject source drift; preserve them when adding a new policy.

The independently frozen [v2.2 correction runner](runner/extensions/v2_2/README.md)
adds batched wrong-verdict feedback after queued self-corrections have been checked:
`python -m runner --version v2.2 --benchmark`. It preserves the v2.1 parser,
prompt and four-request budget; see that contract for branching and usage details.

[Core v2.3](runner/extensions/v2_3/README.md) adds v1.5-style shared slots sized to
the selected question count (30 for AIME, 47 for Apex), with freed slots admitting
continuations or fresh siblings without a barrier. Each fresh rollout gets one
long request clipped to the 65,536 total context, including its prompt. It retains
v2.2 feedback and the four-request cap. Use
`python -m runner --version v2.3 --benchmark`; provision its separate
`vllm-v2_3-long64k.yaml` profile first. Its [first measured trials](runs/experiments/core-v2_3-benchmarks-20261004T182444Z/README.md)
reached 18 AIME answers in 84.985s and 3/47 Apex answers at a 300s cutoff; no
repeatability or speedup claim follows from these single trials.
