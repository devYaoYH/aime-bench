"""Shared repository paths, OpenRouter configuration, JSON writes, and grading helpers.

Import these helpers from runners that need credentials, the 30-problem dataset,
UTC timestamps, retryable HTTP statuses, or final-answer integer extraction.
ROOT always points to the repository, independent of an experiment package.
Importing this module performs no requests; keys are loaded only when requested.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_key() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"\'')
    raise RuntimeError("OPENROUTER_API_KEY is missing from the environment and .env")


def load_problems() -> list[dict]:
    path = ROOT / "data" / "aime_2025_problems.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 30 or [row["problem_idx"] for row in rows] != list(range(1, 31)):
        raise ValueError("Expected the 30 ordered MathArena prompts; run python -m src.fetch_dataset")
    if any(not isinstance(row.get("answer"), int) or not 0 <= row["answer"] <= 999 for row in rows):
        raise ValueError("Expected an AIME integer answer between 0 and 999 for each problem")
    return rows


def extract_answer(content: str | None) -> str | None:
    if not content:
        return None
    # Read only the final answer, never a number in the separate reasoning trace.
    answer_lines = re.findall(r"(?im)^\s*Answer\s*:\s*\$?\s*(\d{1,3})\s*\$?\s*[.。]?\s*$", content)
    if answer_lines:
        return str(int(answer_lines[-1]))
    boxed = re.findall(r"\\boxed\s*\{\s*(\d{1,3})\s*\}", content)
    if boxed:
        return str(int(boxed[-1]))
    return None


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
