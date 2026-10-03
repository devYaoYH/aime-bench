# Jev review of 1,500-token prefixes

The same continuation-promise decision was applied to exact 1,500-Qwen-token prefixes of the 18 unfinished original attempts.

- **Producers:** [src.experiments.jev.judge_prefix](../../../src/experiments/jev/judge_prefix.py).
- **Inputs:** The parent baseline capped traces and the official Qwen/Qwen3-30B-A3B tokenizer.
- **Artifacts:** config.json and summary.json; full per-question Jev records stay local.
- **Interpretation:** Scores describe continuation from the early prefix; full-trace scores describe continuation from the end of the capped response.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
