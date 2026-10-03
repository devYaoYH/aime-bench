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

## Results

All 30 questions have eight parseable candidate integers (240 total). The initial concurrent collection took 22.8 seconds. It required 260 generation attempts to fill 239 slots; Q22 sample 3 first hit the output cap twice and then returned `1075`, outside the AIME range. One separately recorded no-thinking repair attempt supplied `154`, bringing the total to 261 attempts and 240 parseable slots. No saved response had reasoning content.

| Measure | Result |
| --- | ---: |
| Questions with at least four votes for one answer | 2 / 30 |
| Of those, questions with at least four votes for the gold answer | 0 / 2 |
| Unique four-vote-or-higher leaders that were correct | 0 / 2 |
| Four–four ties | 0 |
| Questions with any correct sample | 2 / 30 |
| Correct individual samples | 2 / 240 |

The two agreement cases were Q20: `96` received 4 votes versus gold `336`; and Q26: `1` received 5 votes versus gold `113`. Thus the requested 4-of-8 agreement count is **2**, and **0** of those agreed answers are correct. The two isolated correct samples were on Q10 and Q29; neither question reached four votes for that answer.

The `summary.json` wall-clock span includes the pause between initial collection and the targeted repair. It is not an inference-only latency measurement. `samples.csv` is the versioned per-sample candidate table; the ignored `questions/` records preserve each exact request, full response, and retry for audit.
