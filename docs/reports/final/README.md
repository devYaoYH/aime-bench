# Audited speedrun report

[Three-page report source](report.md) and [PDF](output/pdf/callosum-speedrun-report.pdf).
[Illustrated evidence source](evidence.md) and [PDF](output/pdf/callosum-evidence-packet.pdf).

The final-strategy description was checked against the selected frozen core and
baseline preset at `fc0d724`. [Strategy audit](strategy-audit.json) records exact
configuration, manifest and source hashes, the historical five-trial outcomes,
and which alternate configurations have not been measured. All 205 offline
tests passed for the core. The initial audit launched no GPU experiment. The subsequently requested [improved-prompt five-seed validation](../../../runs/experiments/frozen-core-prompt-five-seeds-20261004T005416Z/README.md) is now included in both PDFs: all five reached 18, median 77.277s and range 62.783-82.492s on one server. These batch statistics are the headline; the historical threshold remains secondary context. Core hashes are unchanged; the original baseline preset remains selected.

The original workspace reports were outside Git. This bundle makes the corrected
sources, renderer, figures and cited analysis artifacts versioned. Relative
citations now resolve to repository evidence. The original workspace copies were
updated as well. Historical figures and analyses retain their recorded data;
their captions do not turn experimental dynamic60 into the selected strategy.

To reproduce the PDFs with ReportLab, pypdf and Pillow, run the evidence renderer
first because the main report links its PDF:

```sh
python docs/reports/final/render_report.py --evidence
python docs/reports/final/render_report.py
```

The renderer preserves the original macOS font setup. The included
`analyze_first_grader.py` is the original analysis-source snapshot and expects its
historical workspace layout; it is evidence for that analysis, not a new runner.
