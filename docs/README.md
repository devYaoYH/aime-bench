# Documentation

Start with the [repository README](../README.md) for setup and the
[experiment guide](experiments.md) for runnable commands. Run every command
from the repository root. Utilities use `python -m src.<module>`; experiment
entry points use `python -m src.experiments.<family>.<module>`. See the
[script guide](scripts.md) and [saved-run provenance](../runs/README.md).

See [canonical remote attempts](attempts.md) for the managed vLLM/grader runner
and the local-development-to-remote-experiment workflow.

## Baseline and sampling

| Report | Purpose |
| --- | --- |
| [Single-pass benchmark](reports/report.md) | Original 30-question Qwen run, grades, timing, and traces |
| [Self-consistency](reports/self-consistency.md) | Eight samples per question, pass@8, and answer voting |
| [Escalation analysis](reports/escalation-analysis.md) | Reasoning length, continuation checkpoints, and generation budgets |

## Trajectory judging and local salvage

| Report | Purpose |
| --- | --- |
| [Jev review](reports/jev-review.md) | Full-trace and 1,500-token prefix scores for unfinished trajectories |
| [Jev calibration](reports/jev-calibration.md) | Retrospective false-negative checks against completed trajectories |
| [Local salvage](reports/local-salvage.md) | CPU Gemma extraction and verification controls |

## Intermediate answers and early verification

| Report | Purpose |
| --- | --- |
| [Intermediate answers](reports/intermediate-answers.md) | Offline claim extraction, scope annotations, and voting replay |
| [First-answer prompt pilot](reports/first-answer-pilot.md) | Paired streaming experiment on prompt-driven stopping |
| [Streaming markers](reports/streaming-answer-markers.md) | Chunk-safe marker detection and formatting false positives |
| [Chunk extraction and sidecar](reports/chunk-answer-extraction.md) | Hosted extraction and verification while the solver continues |
| [Pass@4 early verification](reports/early-verify-pass4.md) | Shared-verifier back-test with generation-capacity scenarios |

## Python tool experiments

| Report | Purpose |
| --- | --- |
| [Python tool pilot](reports/python-tool-pilot.md) | Qwen3.5 paired tool/no-tool pilot and restricted worker interface |
| [Full pass@2 profile](../runs/python-tool-qwen35-pass2-auto/report.md) | All 30 questions, token and latency plots, and tool outcomes |
| [Pass@2 early verification](../runs/python-tool-qwen35-pass2-auto/early_verify_pass2/report.md) | Multi-round candidate extraction and shared-verifier replay |

## Evidence and generated reports

The reports in `docs/reports/` link to original evidence in `runs/`. Generated
reports that belong to a specific run stay beside that run's plots and records.
`src.experiments.intermediate_answers.render_intermediate` and
`src.experiments.streaming.report_first_answer_pilot` regenerate their
reports directly into `docs/reports/`. Dataset provenance and manually audited
scope exclusions live in `data/`.

Git includes run summaries, analysis data, reports, and plots. Raw traces, SSE
streams, progress records, and manifests containing trace excerpts stay local.
Links to those files require the original local run artifacts; a fresh clone
contains the published statistics rather than the full model conversations.
