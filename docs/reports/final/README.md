# Report and evidence packet

The report describes the measured **core v1 / prompt-adherence** strategy:
**5/5 declared seeds reached 18, median 77.277s, range 62.783–82.492s**.
Use the [repository quickstart](../../../README.md#run-the-measured-core-v1)
and [runner contract](../../../runner_final/README.md) to reproduce it.

| Document | Source | PDF |
| --- | --- | --- |
| Three-page speedrun report | [report.md](report.md) | [Report](output/pdf/callosum-speedrun-report.pdf) |
| Eleven-page evidence packet | [evidence.md](evidence.md) | [Evidence packet](output/pdf/callosum-evidence-packet.pdf) |

The packet covers grader timing, concurrency, CPU overhead, dynamic allocation,
backend trials and exploratory alternatives. E8 (page 10) gives the final v1
marginal curve at 1, 2, 4, 6, 8, 10, 12, 14, 16 and 18 verified answers.
E9 (page 11) plots all 54 target-reaching AIME 2025 attempts with frontier captions.
Figure assets and their fixed source data are in `evidence-assets/` and
`analysis/reporting-figures/`.

[Five-seed validation](../../../runs/experiments/frozen-core-prompt-five-seeds-20261004T005416Z/README.md)
supplies the headline statistics. The [AIME 2026 transfer check](../../../runs/experiments/frozen-core-aime2026-lightweight-20261004T011005Z/README.md)
reached 18 in 88.669s with the same policy; this is a single trial.
Historical best draws, AIME 2024 workload warmups, dynamic60 and the general-answer
v2 extension have separate provenance and are not the final v1 configuration.
[Strategy audit](strategy-audit.json) retains the original audit snapshot and
subsequent validation references.

## Rebuild

Use ReportLab, pypdf and Pillow on macOS with the fonts referenced by the renderer.
Run from the repository root, evidence first because the report links its PDF:

```bash
python docs/reports/final/render_report.py --evidence
python docs/reports/final/render_report.py
```

Sources, figures and supporting analyses are versioned. The included
`analyze_first_grader.py` preserves the historical analysis and its workspace
path assumptions; it is not a solver entrypoint.
