"""Reproduce the bundled Apex prompts/key from one immutable HF revision.

Development-only importer; requires httpx and pyarrow. Benchmark runs read the
bundled JSONL files offline and never import this module or access HF.
"""
import hashlib
import io
import json
from pathlib import Path

import httpx
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f3efdf224ef665f129ddaae37699f6098c65781b"
SOURCE = "https://huggingface.co/datasets/MathArena/apex-shortlist"


def main():
    url = f"{SOURCE}/resolve/{REVISION}/data/train-00000-of-00001.parquet"
    raw = httpx.get(url, follow_redirects=True, timeout=60).raise_for_status().content
    rows = sorted(pq.read_table(io.BytesIO(raw)).to_pylist(), key=lambda r: r["problem_idx"])
    if [r["problem_idx"] for r in rows] != list(range(1, 48)):
        raise ValueError("Pinned Apex revision must have 47 question indices")
    prompts = ROOT / "data/apex_shortlist_problems.jsonl"
    grader = ROOT / "grader/data/apex_shortlist.jsonl"
    for path, gold in ((prompts, False), (grader, True)):
        output = [{"problem_idx": row["problem_idx"], "problem": row["problem"],
                   "source": row["source"], **({"answer": str(row["answer"])} if gold else {})}
                  for row in rows]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    manifest = {"schema_version": 1, "id": "apex_shortlist", "source": SOURCE,
                "revision": REVISION, "split": "train", "rows": len(rows),
                "prompt_path": str(prompts.relative_to(ROOT)), "grader_path": str(grader.relative_to(ROOT)),
                "prompt_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
                "grader_sha256": hashlib.sha256(grader.read_bytes()).hexdigest(),
                "download_url": url, "download_sha256": hashlib.sha256(raw).hexdigest(),
                "overlap_notes": [f"Apex #{r['problem_idx']} is sourced from {r['source']} (seen AIME 2025 development data)."
                                  for r in rows if "AIME" in r["source"] and "2025" in r["source"]]}
    (ROOT / "data/source_apex_shortlist.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "revision": REVISION, "overlap_notes": manifest["overlap_notes"]}))


if __name__ == "__main__":
    main()
