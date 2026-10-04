# Report and evidence packet

The report describes the measured **core v1 / prompt-adherence** strategy:
**5/5 declared seeds reached 18, median 77.277s, range 62.783–82.492s**.
Use the [repository quickstart](../../../README.md#run-the-measured-core-v1)
and [runner contract](../../../runner_final/README.md) to reproduce it.

| Document | Source | PDF |
| --- | --- | --- |
| Three-page speedrun report | [report.md](report.md) | [Report](output/pdf/callosum-speedrun-report.pdf) |
| Fifteen-page evidence packet | [evidence.md](evidence.md) | [Evidence packet](output/pdf/callosum-evidence-packet.pdf) |

The packet covers grader timing, concurrency, CPU overhead, dynamic allocation,
backend trials and exploratory alternatives. E8 (page 10) gives the final v1
marginal curve at 1, 2, 4, 6, 8, 10, 12, 14, 16 and 18 verified answers.
E9 (page 11) plots a fixed historical snapshot of 54 target-reaching AIME 2025 attempts with frontier captions.
Figure assets and their fixed source data are in `evidence-assets/` and
`analysis/reporting-figures/`.

[Five-seed validation](../../../runs/experiments/frozen-core-prompt-five-seeds-20261004T005416Z/README.md)
supplies the headline statistics. The [AIME 2026 transfer check](../../../runs/experiments/frozen-core-aime2026-lightweight-20261004T011005Z/README.md)
reached 18 in 88.669s with the same policy; this is a single trial.
Historical best draws, AIME 2024 workload warmups, dynamic60 and the general-answer
v2 extension have separate provenance and are not the final v1 configuration.
[Strategy audit](strategy-audit.json) retains the original audit snapshot and
subsequent validation references.

The [post-freeze batch](../../../results/post_freeze/measurements-v1-20261004T104200Z/README.md)
adds E10–E13 (pages 12–15). Its five extended core trials reached 18 at median
78.305s (63.908–135.071s), then exhausted four-request budgets at 26–27 correct.
The complete 95%-memory BF16 baseline finished 120 samples: pass@1 47.50%,
pass@4 18/30, unique-plurality vote 18/30 and strict three-of-four vote 13/30.
These are separate from the original stop-at-18 headline batch.

## Rebuild

Use ReportLab, pypdf and Pillow on macOS with the fonts referenced by the renderer.
Run from the repository root, evidence first because the report links its PDF:

```bash
python docs/reports/final/render_report.py --evidence
python docs/reports/final/render_report.py
```

Rebuild the post-freeze analysis/figure with `python -m scripts.analyze_post_freeze --batch measurements-v1-20261004T104200Z` before rendering.

Sources, figures and supporting analyses are versioned. The included
`analyze_first_grader.py` preserves the historical analysis and its workspace
path assumptions; it is not a solver entrypoint.
