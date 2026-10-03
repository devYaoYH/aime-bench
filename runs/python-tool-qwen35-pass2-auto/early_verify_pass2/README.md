# Python pass@2 early-verification replay

Final/marker/permissive candidate checking and a separate Python-stdout extension, with a shared three-second verifier and 60/30/8-slot replays.

- **Producers:** [src.experiments.python_tools.backtest_python_early_verify](../../../src/experiments/python_tools/backtest_python_early_verify.py).
- **Inputs:** The parent 60 multi-round raw records and cached Qwen3.5 tokenizer plus source metadata.
- **Artifacts:** summary.json, candidate_inventory.json, report.md, and time_to_correct/eight_slots plots.
- **Interpretation:** Offline counterfactual capacity replay plus observed-start shadow schedules. Within-round text times are estimated; CPU-tool timing is retained. Saved keys stand in for perfect verification.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
