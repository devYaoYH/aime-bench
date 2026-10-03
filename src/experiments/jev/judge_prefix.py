"""Score the first 1,500 Qwen reasoning tokens of unfinished trajectories with Jev.

Use this experiment to compare early-prefix continuation promise with the
full-trace Jev review. prefix_for cuts exact tokenizer prefixes; the prompt
withholds final outcomes and keys. Saves decisions in jev_prefix_<tokens>/.
Requires local baseline records, the Qwen tokenizer, and OPENROUTER_API_KEY.
Uncached tokenizer retrieval uses the network; fresh Jev decisions are paid.
    python -m src.experiments.jev.judge_prefix --tokens 1500
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from src.common import ROOT, atomic_json, load_key
from src.experiments.jev.judge_trajectories import MODEL, QUESTIONS, RETRYABLE, URL, EXTRA_TOKENS, failed_indices, read_probability, utc_now


from src.tokenizer_utils import QWEN_TOKENIZER_REPO as TOKENIZER_REPO, prefix_for
PREFIX_TOKENS = 1500


async def main_async(args: argparse.Namespace) -> None:
    run = ROOT / "runs" / args.run
    indices = failed_indices(run)
    if len(indices) != 18:
        raise ValueError(f"Expected the 18 failed trajectories from the original run, got {len(indices)}")
    tokenizer_path = args.tokenizer_json or hf_hub_download(repo_id=TOKENIZER_REPO, filename="tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    out = run / f"jev_prefix_{args.tokens}"
    out.mkdir(exist_ok=True)
    config = {
        "model": MODEL,
        "source_run": args.run,
        "endpoint": URL,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        "reasoning_prefix_tokens": args.tokens,
        "continuation_budget_output_tokens": EXTRA_TOKENS,
        "question": QUESTIONS["promising_to_extend"],
        "uses_gold_answers": False,
    }
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Existing prefix review uses a different configuration")
    atomic_json(config_path, config)
    key = load_key()
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=max(args.concurrency + 2, 8))
    timeout = httpx.Timeout(connect=30, read=120, write=60, pool=30)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def one(index: int) -> None:
            path = out / f"{index:02d}.json"
            if path.exists() and json.loads(path.read_text()).get("probability") is not None:
                print(f"{index:02d}: already judged", flush=True)
                return
            trace = json.loads((run / "questions" / f"{index:02d}.json").read_text())
            full_reasoning = trace["response"]["choices"][0]["message"]["reasoning"]
            prefix, full_token_count = prefix_for(full_reasoning, tokenizer, args.tokens)
            payload = {
                "model": MODEL,
                "state": {
                    "problem": trace["problem"],
                    "reasoning_trace": prefix,
                    "solver_model": "qwen/qwen3-30b-a3b",
                    "trace_status": f"This is only the first {args.tokens} Qwen3 reasoning tokens of a saved trajectory; consider continuation from this prefix.",
                    "continuation_budget_output_tokens": EXTRA_TOKENS,
                },
                "questions": {"promising_to_extend": QUESTIONS["promising_to_extend"]},
            }
            record = {
                "problem_idx": index,
                "prefix_tokens": args.tokens,
                "full_reasoning_tokens_tokenizer": full_token_count,
                "prefix_chars": len(prefix),
                "full_reasoning_chars": len(full_reasoning),
                "reasoning_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                "request": payload,
                "attempts": [],
                "response": None,
                "probability": None,
            }
            async with semaphore:
                for number in range(1, args.retries + 2):
                    started = time.perf_counter()
                    attempt = {"number": number, "started_at_utc": utc_now()}
                    try:
                        response = await client.post(
                            URL,
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "AIME prefix trajectory review"},
                            json=payload,
                        )
                        attempt["http_status"] = response.status_code
                        attempt["latency_s"] = round(time.perf_counter() - started, 3)
                        attempt["finished_at_utc"] = utc_now()
                        try:
                            body = response.json()
                        except ValueError:
                            body = {"raw_text": response.text}
                        attempt["response"] = body
                        record["attempts"].append(attempt)
                        if response.status_code == 200 and isinstance(body, dict) and "answers" in body:
                            record["response"] = body
                            record["probability"] = read_probability(body, "promising_to_extend")
                            break
                        if response.status_code not in RETRYABLE:
                            break
                    except (httpx.HTTPError, ValueError, KeyError) as exc:
                        attempt["latency_s"] = round(time.perf_counter() - started, 3)
                        attempt["finished_at_utc"] = utc_now()
                        attempt["error"] = f"{type(exc).__name__}: {exc}"
                        if not record["attempts"] or record["attempts"][-1] is not attempt:
                            record["attempts"].append(attempt)
                    if number <= args.retries:
                        await asyncio.sleep(min(2 ** number, 8))
            record["latency_s"] = round(sum(attempt["latency_s"] for attempt in record["attempts"]), 3)
            atomic_json(path, record)
            print(f"{index:02d}: prefix promising={record['probability']}", flush=True)

        await asyncio.gather(*(one(index) for index in indices))

    full_review = json.loads((run / "jev_review" / "summary.json").read_text())
    full_by_index = {row["problem_idx"]: row["promising_to_extend"] for row in full_review["ranked"]}
    rows = []
    for index in indices:
        record = json.loads((out / f"{index:02d}.json").read_text())
        if record["probability"] is None:
            raise RuntimeError(f"Question {index} has no valid Jev decision")
        usage = (record["response"] or {}).get("usage") or {}
        rows.append({
            "problem_idx": index,
            "prefix_promising": record["probability"],
            "full_trace_promising": full_by_index[index],
            "delta_prefix_minus_full": round(record["probability"] - full_by_index[index], 3),
            "prefix_tokens": record["prefix_tokens"],
            "prefix_chars": record["prefix_chars"],
            "input_tokens": usage.get("input_tokens"),
            "cost": usage.get("cost"),
            "trace_file": f"jev_prefix_{args.tokens}/{index:02d}.json",
        })
    rows.sort(key=lambda row: row["prefix_promising"], reverse=True)
    summary = {
        "model": MODEL,
        "run_id": args.run,
        "reasoning_prefix_tokens": args.tokens,
        "judged": len(rows),
        "question_wording": QUESTIONS["promising_to_extend"],
        "note": "Scores are subjective Jev forecasts. The prefix review contains only the first Qwen reasoning tokens and no gold answers; it is separate from the full-trace review.",
        "total_input_tokens": sum(row["input_tokens"] or 0 for row in rows),
        "total_reported_cost": round(sum(row["cost"] or 0 for row in rows), 9),
        "ranked": rows,
    }
    atomic_json(out / "summary.json", summary)
    print(f"Judged {len(rows)} prefixes; summary: {out / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="20260930-155212")
    parser.add_argument("--tokens", type=int, default=PREFIX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--tokenizer-json", type=Path)
    args = parser.parse_args()
    if not 1000 <= args.tokens <= 2000 or args.concurrency < 1 or args.retries < 0:
        parser.error("tokens must be 1000–2000; concurrency must be positive; retries nonnegative")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
