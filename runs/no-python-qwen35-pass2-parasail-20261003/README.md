# Matched no-Python Qwen3.5 pass@2

No-tool baseline for `../python-tool-qwen35-pass2-auto/`, using the same OpenRouter/Parasail routing and generation controls.

- Producer: `src.experiments.python_tools.no_python_baseline_v1`.
- Analysis: `src.experiments.python_tools.report_no_python_baseline_v1`.
- See [comparison report](report.md), [config](config.json), [summary](summary.json), and [paired metrics](analysis.json).
- [Intermediate-answer comparison](early_verify_pass2/report.md) reuses the historical extractor and three-second verifier simulation.
- Raw paid requests are saved locally; unfinished records block automatic relaunch.
