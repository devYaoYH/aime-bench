# Documentation

Start with the [repository README](../README.md) for the final core v1 command,
execution diagram and results. The [runner contract](../runner_final/README.md)
explains retry budgets, continuations, service ownership and the v2 extension.
Run commands from the repository root.

## Final submission and results

| Document | Purpose |
| --- | --- |
| [Three-page report](reports/final/output/pdf/callosum-speedrun-report.pdf) | Strategy, measured results, negative findings and next steps |
| [Evidence packet](reports/final/output/pdf/callosum-evidence-packet.pdf) | Eleven pages of figures, captions and source links; marginal curve in E8, history in E9 |
| [Final v1 validation](../runs/experiments/frozen-core-prompt-five-seeds-20261004T005416Z/README.md) | 5/5 reached 18; median 77.277s; range 62.783–82.492s |
| [AIME 2026 transfer](../runs/experiments/frozen-core-aime2026-lightweight-20261004T011005Z/README.md) | Same v1 policy, one declared seed; 88.669s |
| [General-answer v2 back-test](../runs/experiments/core-v2-aime2025-five-seeds-20261004T013100Z/README.md) | 5/5 reached 18; median 113.415s; placeholder failure analysis |

## Historical local baselines

These are complete-strategy observations, separate from the final five-seed claim.
The [historical managed-runner guide](attempts.md) documents `src.attempt`;
the [experiment guide](experiments.md), [script guide](scripts.md) and
[saved-run provenance](../runs/README.md) cover the archive.

| Attempt | Time to eighteenth positive verdict |
| --- | --- |
| [VibeThinker BF16 coverage](../attempts/20261003T200718.717581Z/README.md) | 92.428s |
| [VibeThinker NVFP4 coverage, 95% memory](../attempts/20261003T205350.742196Z/README.md) | 85.546s |
| [VibeThinker BF16 final-only pass@4](../attempts/20261003T202152.418590Z/README.md) | 336.497s |

The following reports describe hosted-model exploration and offline replays.
Their accuracy and simulated verification timings are separate from scored local runs.

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

Git includes canonical attempt requests/responses, exact tokens, verdict records,
run summaries, reports and plots. Full SSE/service/grader logs and some historical
hosted-model raw traces remain outside Git. Analyses of those archived traces need
the original artifacts; the final canonical evidence is available from a fresh clone.
