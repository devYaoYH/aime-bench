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
  viewer/            Response viewer assets
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

Open [the response viewer](http://127.0.0.1:8765). It shows the original Qwen run `20260930-155212`,
sample votes, and Jev reviews. Other experiment families have reports and
artifacts under `runs/`.

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

Datasets, run summaries, analysis data, reports, and plots are versioned. Raw
request/response traces, SSE streams, manifests containing trace excerpts, and
progress records stay local and are ignored by Git. Reports may link to these
local files; replaying the analyses requires the original traces. Credentials,
virtual environments, downloaded models, caches, and logs are also ignored.
Offline token-based analyses need the cached tokenizers described in the
experiment guide under `.local/tokenizers/`.
