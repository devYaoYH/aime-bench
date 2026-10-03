# AIME benchmark experiments

Benchmark and analyze mathematical reasoning on the 30 MathArena AIME 2025
problems. The experiments cover Qwen sampling, Jev trajectory review,
intermediate-answer extraction, streaming verification, and Python tool use.
Saved runs include requests, responses, timings, usage, and local exact-match
grades against the official answer key.

## Repository layout

```text
src/                 Common utilities and reusable libraries
  experiments/       Runners grouped by research question
  viewer/            Canonical attempt viewer and fixed exploratory archive
data/                Problem statements, answer key, provenance, scope annotations
runs/                Saved experiment records, generated reports, and plots
attempts/            Canonical remote attempts, traces, and telemetry
configs/vllm/        Versioned remote model launch profiles
test/                Unit tests and offline regression checks
docs/                Experiment guide, report index, and research reports
requirements.txt     Python dependencies
.env.example         API configuration template
```

## Setup

Run commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` or export it in your environment. The dataset
is included in `data/`; to fetch it again:

```bash
.venv/bin/python -m src.fetch_dataset
```

## Run and inspect

Start a new 30-question benchmark (makes paid OpenRouter requests):

```bash
.venv/bin/python -m src.experiments.baseline.benchmark --concurrency 10
```

Browse saved responses locally:

```bash
.venv/bin/python -m src.viewer_server
```

Open [canonical attempts](http://127.0.0.1:8765) to select locally available
`attempts/` artifacts and inspect oracle outcomes, trajectories, continuations,
timing, and GPU telemetry. Saved canonical evidence is versioned, so detailed
views work from a fresh clone. New attempts need their artifacts copied from the
VM and committed after review. Refresh rereads files.

The [exploratory viewer](http://127.0.0.1:8765/exploratory) remains fixed to the
original Qwen run `20260930-155212`, including sample votes and Jev reviews.
Other exploratory experiment families have reports and artifacts under `runs/`.

The [overall results page](http://127.0.0.1:8765/results) aggregates all saved
attempts, plots recorded time to the eighteenth distinct positive grader verdict
against each attempt's start timestamp with a dotted best-so-far Pareto step,
and marks the 54-second minimum serial grader toll (18 × 3s). It links each
intervention to its reference attempt and exposes changed and matched controls.
Unmet targets, failed runs, and missing timing evidence remain explicitly unranked.
Each attempt's `metadata.json` follows [the metadata schema](data/attempt_metadata.schema.json).
After importing new attempts, run `python -m src.attempt_metadata --all`, then
annotate the intervention/reference fields; existing annotations are preserved.
The x-axis uses initialization timestamps; latency uses the official clock after
warmup. Plot logic can be checked with `node test/test_results_history.js`.

## Canonical remote attempts

See [the attempt workflow](docs/attempts.md) and [repo agent instructions](AGENTS.md).
On `callosum`, with a free GPU and configured model under `~/models`:

```bash
~/.venvs/vllm/bin/python -m src.attempt --model Qwen/Qwen3.5-4B
```

This manages CUDA/inference warmup, vLLM, the grader, eight concurrent questions
with four streaming rollouts each, early-verification cancellation, and telemetry.
Qwen uses a 16,384-token generation cap.

## Documentation and checks

- [Documentation index](docs/README.md): grouped experiment reports and results.
- [Experiment guide](docs/experiments.md): commands, configuration, and limitations.
- [Script guide](docs/scripts.md): what to run and when.
- [Run provenance](runs/README.md): producers, source inputs, and experiment context.
- [Baseline report](docs/reports/report.md): original single-pass results.

```bash
.venv/bin/python -m unittest discover -s test -v
```

Tests use local fixtures and mocked requests; they do not make inference calls.
The salvage tests require locally saved raw run traces. The Python worker tests
require macOS and `/usr/bin/sandbox-exec`.

Datasets, run summaries, analysis data, reports, and plots are versioned. Canonical
`attempts/` also includes saved requests/responses, exact token records, question
and round records, verification events, GPU samples, and launch/warmup metadata.
Full SSE streams, grader audits, and legacy exploratory `runs/` raw traces remain
ignored; detailed exploratory views and their analyses need the original local
traces. Credentials, virtual environments, downloaded models, caches, and logs
are also ignored.
Offline token-based analyses need the cached tokenizers described in the
experiment guide under `.local/tokenizers/`.
