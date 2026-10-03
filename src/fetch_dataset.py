"""Fetch the 30 MathArena AIME 2025 problems and official answers with provenance.

Use this utility to refresh data/aime_2025_problems.jsonl and data/source.json.
It requests dataset metadata and rows, validates ordered question IDs, and stores
the revision and dataset SHA256. Existing data files are replaced. Internet access
is required; no model inference occurs. Run from the repository root:
    python -m src.fetch_dataset
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

from src.common import ROOT
DATASET = "MathArena/aime_2025"
API = "https://huggingface.co/api/datasets/MathArena/aime_2025"
ROWS = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        info_response = client.get(API)
        info_response.raise_for_status()
        revision = info_response.json()["sha"]
        response = client.get(
            ROWS,
            params={
                "dataset": DATASET,
                "config": "default",
                "split": "train",
                "offset": 0,
                "length": 100,
            },
        )
        response.raise_for_status()
        payload = response.json()

    if payload.get("num_rows_total") != 30 or len(payload.get("rows", [])) != 30:
        raise ValueError("Expected exactly 30 AIME 2025 problems")
    problems = [
        {
            "problem_idx": int(item["row"]["problem_idx"]),
            "problem": item["row"]["problem"],
            "answer": int(item["row"]["answer"]),
            "problem_type": item["row"].get("problem_type", []),
        }
        for item in payload["rows"]
    ]
    problems.sort(key=lambda row: row["problem_idx"])
    if [row["problem_idx"] for row in problems] != list(range(1, 31)):
        raise ValueError("Missing or duplicate problem indices")

    output = ROOT / "data" / "aime_2025_problems.jsonl"
    output.parent.mkdir(exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in problems)
    output.write_text(content)
    manifest = {
        "source": f"https://huggingface.co/datasets/{DATASET}",
        "revision": revision,
        "split": "train",
        "rows": len(problems),
        "dataset_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "note": "Official problem statements and answers; answers are used only for local grading.",
    }
    (output.parent / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(problems)} prompts to {output}")
    print(f"Source revision: {revision}")


if __name__ == "__main__":
    main()
