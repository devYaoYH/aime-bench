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
print_command=false
while (( $# )); do
  case "$1" in
    --print-command) print_command=true; shift ;;
    --benchmark-year)
      if [[ "${2:-}" != 2024 && "${2:-}" != 2025 && "${2:-}" != 2026 ]]; then
        echo "--benchmark-year requires 2024, 2025 or 2026" >&2; exit 2
      fi
      command+=(--benchmark-year "$2"); shift 2 ;;
    --benchmark-role)
      if [[ "${2:-}" != prewarming && "${2:-}" != development && "${2:-}" != generalization ]]; then
        echo "--benchmark-role requires prewarming, development or generalization" >&2; exit 2
      fi
      command+=(--benchmark-role "$2"); shift 2 ;;
    *)
      echo "Usage: bash scripts/run_qwen35_gptq_coverage.sh [--print-command] [--benchmark-year 2024|2025|2026] [--benchmark-role prewarming|development|generalization]" >&2
      exit 2 ;;
  esac
done
if "$print_command"; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
"$HOME/.venvs/vllm/bin/python" scripts/prepare_qwen35_gptq.py --verify-only
exec "${command[@]}"
