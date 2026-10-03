# Original single-pass Qwen baseline

One independent qwen/qwen3-30b-a3b response for each of the 30 AIME 2025 questions; concurrent OpenRouter requests with a 16,384-token cap and local exact-key grading.

- **Producers:** [src.experiments.baseline.benchmark](../../src/experiments/baseline/benchmark.py).
- **Inputs:** data/aime_2025_problems.jsonl and data/source.json. This is the source run for the original Qwen follow-up experiments.
- **Artifacts:** config.json and summary.json; question responses remain local in questions/. Nested folders contain separate follow-up experiments.
- **Interpretation:** Measured request timings and grades. The original pass@1 is 12/30; capped responses count as failures.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
