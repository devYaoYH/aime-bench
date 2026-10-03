#!/usr/bin/env bash
# Invoke explicitly when ready to run; --print-command is safe during preparation.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
command=("$HOME/.venvs/vllm/bin/python" -m src.attempt
  --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
  --strategy coverage --parallelism 30 --rollouts 1
  --first-pass-max-tokens 8192 --max-tokens 16384
  --max-attempts-per-question 4 --max-rounds 4 --target-correct 18
  --temperature 0.8 --top-p 0.95 --seed 20261003)
if [[ "${1:-}" == "--print-command" && $# == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
if (( $# )); then
  echo "Usage: bash scripts/run_qwen35_gptq_coverage.sh [--print-command]" >&2
  exit 2
fi
"$HOME/.venvs/vllm/bin/python" scripts/prepare_qwen35_gptq.py --verify-only
exec "${command[@]}"
