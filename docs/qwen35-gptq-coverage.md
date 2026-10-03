# Qwen3.5 35B A3B GPTQ Int4 coverage experiment

Prepared for Callosum's single A100 PCIe 80GB. Model:
[Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4),
pinned to `3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b`.

The canonical runner uses all 30 AIME 2025 questions, 30 concurrent questions,
one rollout per question per round, an 8,192-token first pass, and a 16,384-token
maximum for each subsequent request. Capped unsolved trajectories continue using
exact token IDs. Continuations count toward the four-request per-question limit;
natural wrong/no-answer endings start fresh samples in subsequent rounds.
The 16K limit is per request, not a cumulative trajectory limit. Total context is
32,768 tokens, matching the existing Qwen profile. A continuation near that
context limit receives only the remaining token budget.

The existing stopping target remains 18 verified correct questions. Thinking is
enabled by default. Sampling stays at temperature 0.8, top-p 0.95, seed 20261003;
the grader's global toll remains three seconds. Prefix caching and prompt-token
details are enabled. The profile infers GPTQ from model metadata and limits
vLLM scheduler capacity to 32 sequences. Runtime loading and memory capacity are
unverified until an explicitly authorized launch.

## Preparation only

After committing and pushing locally, pull the same commit on Callosum:

```sh
ssh callosum
cd ~/aime-bench
git pull --ff-only
~/.venvs/vllm/bin/python scripts/prepare_qwen35_gptq.py
bash scripts/run_qwen35_gptq_coverage.sh --print-command
```

Preparation downloads the pinned snapshot under
`~/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`, verifies file sizes and Hugging Face LFS
SHA256 digests (including all 14 indexed weight shards), deploys the tracked
`vllm.yaml`, and saves `download_manifest.json` beside the weights. Rerunning
preparation resumes downloads. `--verify-only` checks the saved files and deployed
profile offline. Neither preparation nor printing the command starts CUDA,
vLLM, the grader, or an attempt.

## Launch later

Only when ready to run, inspect `nvidia-smi`, active processes, and ports
8000/8077. Leave unrelated services alone. Then use a durable session:

```sh
cd ~/aime-bench
tmux new-session -s qwen35-gptq-coverage \
  'bash scripts/run_qwen35_gptq_coverage.sh'
```

The launcher verifies the prepared model and profile before invoking
`src.attempt`. The runner records provenance, warmup, first-solve timestamps,
exact-token continuation ancestry, cache usage, traces, and GPU telemetry in a
new `attempts/<UTC timestamp>/` directory. It manages its own vLLM and grader
services. See [canonical attempt behavior](attempts.md).
