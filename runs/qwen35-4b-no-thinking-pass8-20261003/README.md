# Local Qwen3.5-4B no-thinking pass@8

Eight independent short-answer samples for each of the 30 MathArena AIME 2025 questions, collected on the callosum A100 through vLLM with the Qwen chat template's thinking mode disabled.

- **Producer:** [src.experiments.local_qwen.no_thinking_pass8](../../src/experiments/local_qwen/no_thinking_pass8.py).
- **Repair producer:** [src.experiments.local_qwen.repair_missing](../../src/experiments/local_qwen/repair_missing.py) can continue the seed sequence for any slot without a parseable answer; its separate source commit and attempts are appended to the saved provenance.
- **Inputs:** [The versioned 30-question dataset](../../data/aime_2025_problems.jsonl) and its [source record](../../data/source.json); locally downloaded Qwen/Qwen3.5-4B BF16 checkpoint and versioned vLLM launch config under `~/models/Qwen/Qwen3.5-4B/` on callosum. Gold answers are used only after responses arrive.
- **Settings:** 240 simultaneous independent requests, eight per question; unique seeds; `enable_thinking: false`; temperature 0.8, top-p 0.9, top-k 40, and 64 output tokens. The exact prompt, endpoint, launch-config hash, source commits, seed formula, and retry policy are recorded in `config.json` after execution.
- **Artifacts:** `config.json`, `summary.json`, and `samples.csv` are versioned. Full requests, responses, and retry records stay local in ignored `questions/NN/SS.json` files.
- **Interpretation:** A question meets the requested agreement threshold when any parseable answer receives at least 4 of 8 votes. A 4–4 tie meets that threshold but does not yield one unique prediction; the summary separately counts questions where the gold answer receives at least four votes and uniquely leading predictions that are correct. A response must stop naturally and contain a parseable final integer to vote.

Run from the repository root on callosum, after launching the Qwen vLLM config:

    ~/.venvs/vllm/bin/python -m src.experiments.local_qwen.no_thinking_pass8

Results will be added here after the run.
