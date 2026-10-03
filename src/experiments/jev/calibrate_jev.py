"""Check Jev prefix-score false negatives against completed saved Qwen trajectories.

Use this experiment after baseline/self-consistency generation to evaluate
1,500-token prefix decisions on completed responses. Labels are added only after
judging; the analysis separates success within Jev's continuation budget from
later success. Saves decisions and threshold statistics in jev_calibration/.
Requires local raw records, a Qwen tokenizer, and paid OpenRouter Jev calls.
    python -m src.experiments.jev.calibrate_jev
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
from src.experiments.jev.judge_prefix import PREFIX_TOKENS
from src.tokenizer_utils import QWEN_TOKENIZER_REPO as TOKENIZER_REPO, prefix_for
from src.experiments.jev.judge_trajectories import EXTRA_TOKENS, MODEL, QUESTIONS, RETRYABLE, URL, read_probability, utc_now


def completed_traces(run: Path) -> list[tuple[Path, dict]]:
    paths = sorted((run / "questions").glob("*.json"))
    paths += sorted((run / "self_consistency" / "questions").glob("*/*.json"))
    traces = []
    for path in paths:
        trace = json.loads(path.read_text())
        if trace.get("finish_reason") == "stop":
            traces.append((path, trace))
    return traces


def label_summary(records: list[dict], run: Path) -> dict:
    rows = []
    for record in records:
        trace = json.loads((run / record["source_trace_file"]).read_text())
        usage = trace.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(completion_tokens, int):
            raise ValueError(f"Missing completion tokens: {record['source_trace_file']}")
        rows.append({
            "problem_idx": record["problem_idx"],
            "sample_number": record["sample_number"],
            "probability": record["probability"],
            "correct": trace["correct"],
            "candidate": trace["candidate"],
            "finish_reason": trace["finish_reason"],
            "completion_tokens": completion_tokens,
            "within_8192_more_tokens": completion_tokens <= PREFIX_TOKENS + EXTRA_TOKENS,
            "decision_file": f"jev_calibration/{record['problem_idx']:02d}-{record['sample_number']:02d}.json",
            "source_trace_file": record["source_trace_file"],
        })
    rows.sort(key=lambda row: (row["problem_idx"], row["sample_number"]))
    strict = [row for row in rows if row["correct"] and row["within_8192_more_tokens"]]
    eventual = [row for row in rows if row["correct"]]
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7]
    return {
        "model": MODEL,
        "run_id": run.name,
        "prefix_tokens": PREFIX_TOKENS,
        "continuation_budget_tokens": EXTRA_TOKENS,
        "completed_trajectories": len(rows),
        "correct_completed": len(eventual),
        "correct_completed_within_budget": len(strict),
        "thresholds": [
            {
                "reject_below": threshold,
                "rejected_completed": sum(row["probability"] < threshold for row in rows),
                "strict_budget_false_negatives": sum(row["probability"] < threshold for row in strict),
                "strict_budget_positives": len(strict),
                "eventual_false_negatives": sum(row["probability"] < threshold for row in eventual),
                "eventual_positives": len(eventual),
                "completed_wrong_rejected": sum(row["probability"] < threshold for row in rows if not row["correct"]),
            }
            for threshold in thresholds
        ],
        "rows": rows,
        "notes": [
            "Jev saw only problem text and the first 1,500 reasoning tokens, never the final result or answer key.",
            "Strict-budget positives are correct completed trajectories with total provider output <= 9,692 tokens. Longer trajectories are censored for the exact 8,192-token continuation question.",
            "Correctness is local exact integer comparison; no external grader script was invoked.",
            "Multiple samples of the same problem are correlated. This retrospective selected cohort does not measure prospective precision or overall accuracy.",
        ],
    }


async def main_async(args: argparse.Namespace) -> None:
    run = ROOT / "runs" / args.run
    traces = completed_traces(run)
    if not traces:
        raise ValueError("No completed trajectories found")
    tokenizer_path = args.tokenizer_json or Path(hf_hub_download(repo_id=TOKENIZER_REPO, filename="tokenizer.json"))
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    out = run / "jev_calibration"
    out.mkdir(exist_ok=True)
    config = {
        "model": MODEL,
        "endpoint": URL,
        "source_run": args.run,
        "source_selection": "finish_reason=stop; all original and self-consistency samples",
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "prefix_tokens": PREFIX_TOKENS,
        "continuation_budget_tokens": EXTRA_TOKENS,
        "question": QUESTIONS["promising_to_extend"],
        "uses_gold_answers_in_decisions": False,
    }
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Existing calibration uses a different configuration")
    atomic_json(config_path, config)

    prepared = []
    for source_path, trace in traces:
        sample = trace.get("sample_number", 1)
        reasoning = trace["response"]["choices"][0]["message"].get("reasoning") or ""
        prefix, reasoning_tokens = prefix_for(reasoning, tokenizer, PREFIX_TOKENS)
        relative_source = str(source_path.relative_to(run))
        prepared.append((trace["problem_idx"], sample, trace["problem"], prefix, reasoning_tokens, relative_source))
    if len({(index, sample) for index, sample, *_ in prepared}) != len(prepared):
        raise ValueError("Duplicate trajectory identifiers")

    key = load_key()
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(connect=30, read=120, write=60, pool=30)
    limits = httpx.Limits(max_connections=args.concurrency + 5)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def one(item: tuple) -> None:
            index, sample, problem, prefix, reasoning_tokens, relative_source = item
            path = out / f"{index:02d}-{sample:02d}.json"
            if path.exists() and json.loads(path.read_text()).get("probability") is not None:
                print(f"{index:02d}-{sample:02d}: already judged", flush=True)
                return
            payload = {
                "model": MODEL,
                "state": {
                    "problem": problem,
                    "reasoning_trace": prefix,
                    "solver_model": "qwen/qwen3-30b-a3b",
                    "trace_status": f"This is only the first {PREFIX_TOKENS} Qwen3 reasoning tokens of a saved trajectory; consider continuation from this prefix.",
                    "continuation_budget_output_tokens": EXTRA_TOKENS,
                },
                "questions": {"promising_to_extend": QUESTIONS["promising_to_extend"]},
            }
            record = {
                "problem_idx": index,
                "sample_number": sample,
                "source_trace_file": relative_source,
                "prefix_tokens": PREFIX_TOKENS,
                "full_reasoning_tokens_tokenizer": reasoning_tokens,
                "prefix_chars": len(prefix),
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
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "AIME Jev prefix calibration"},
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
                        record["attempts"].append(attempt)
                    if number <= args.retries:
                        await asyncio.sleep(min(2 ** number, 8))
            record["latency_s"] = round(sum(attempt["latency_s"] for attempt in record["attempts"]), 3)
            atomic_json(path, record)
            print(f"{index:02d}-{sample:02d}: promising={record['probability']}", flush=True)

        await asyncio.gather(*(one(item) for item in prepared))

    records = []
    for index, sample, *_ in prepared:
        record = json.loads((out / f"{index:02d}-{sample:02d}.json").read_text())
        if record["probability"] is None:
            raise RuntimeError(f"No valid Jev decision for {index:02d}-{sample:02d}; rerun to retry")
        records.append(record)
    summary = label_summary(records, run)
    summary["total_input_tokens"] = sum(((record["response"].get("usage") or {}).get("input_tokens") or 0) for record in records)
    summary["total_reported_cost"] = round(sum(((record["response"].get("usage") or {}).get("cost") or 0) for record in records), 9)
    atomic_json(out / "summary.json", summary)
    print(f"Saved calibration of {len(records)} completed trajectories to {out / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="20260930-155212")
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 0:
        parser.error("concurrency must be positive and retries nonnegative")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
