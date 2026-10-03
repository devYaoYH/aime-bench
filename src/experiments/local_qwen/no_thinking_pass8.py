"""Collect eight independent, no-thinking Qwen3.5-4B answers per AIME question.

Run on callosum against its local vLLM server. Full request and response records
stay in ignored questions/ files. Config, candidate table, and vote summary are
versioned under runs/. Existing completed slots are reused on rerun.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import httpx

from src.common import ROOT, atomic_json, extract_answer, load_problems, utc_now

MODEL = "Qwen/Qwen3.5-4B"
SYSTEM_PROMPT = (
    "Give your best answer to the AIME math problem. Respond with exactly one "
    "integer from 0 to 999. Do not include reasoning, words, or punctuation."
)
RUN_NAME = "qwen35-4b-no-thinking-pass8-20261003"
SAMPLES = 8
SEED_BASE = 2026100300
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def model_request(problem: str, seed: int, max_tokens: int) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
    }


def parse_candidate(content: str | None, finish_reason: str | None) -> tuple[int | None, str]:
    if finish_reason != "stop" or not content:
        return None, "incomplete"
    text = content.strip()
    if re.fullmatch(r"\d{1,3}", text):
        return int(text), "bare_integer"
    answer = extract_answer(text)
    if answer is not None:
        return int(answer), "answer_marker"
    return None, "unparseable"


def response_has_reasoning(body: object) -> bool:
    if not isinstance(body, dict):
        return False
    return any(
        bool((choice.get("message") or {}).get("reasoning")
             or (choice.get("message") or {}).get("reasoning_content"))
        for choice in body.get("choices") or []
    )


def vote_row(problem_idx: int, gold: int, records: list[dict]) -> dict:
    if len(records) != SAMPLES:
        raise ValueError(f"Question {problem_idx} has {len(records)} of {SAMPLES} samples")
    votes = Counter(record["candidate"] for record in records if record["candidate"] is not None)
    top = max(votes.values(), default=0)
    leaders = sorted(answer for answer, count in votes.items() if count == top)
    four_plus = top >= 4
    unique_four_plus = four_plus and len(leaders) == 1
    return {
        "problem_idx": problem_idx,
        "gold_answer": gold,
        "candidates": [record["candidate"] for record in records],
        "parseable_samples": sum(record["candidate"] is not None for record in records),
        "correct_samples": sum(record["candidate"] == gold for record in records),
        "votes": [
            {"answer": answer, "count": count}
            for answer, count in sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        ],
        "top_vote_count": top,
        "top_answers": leaders,
        "four_plus_agreement": four_plus,
        "four_plus_votes_for_gold": votes[gold] >= 4,
        "unique_four_plus_answer": leaders[0] if unique_four_plus else None,
        "unique_four_plus_correct": unique_four_plus and leaders[0] == gold,
        "four_four_tie": top == 4 and len(leaders) == 2,
        "strict_majority_answer": leaders[0] if top >= 5 and len(leaders) == 1 else None,
    }


def summarize(rows: list[dict], records: list[dict], started: str, ended: str) -> dict:
    latencies = [record["latency_s"] for record in records]
    usages = [record.get("usage") or {} for record in records]
    return {
        "run_name": RUN_NAME,
        "model": MODEL,
        "questions": len(rows),
        "samples_per_question": SAMPLES,
        "sample_slots_completed": len(records),
        "generation_attempts": sum(len(record["attempts"]) for record in records),
        "parseable_samples": sum(row["parseable_samples"] for row in rows),
        "correct_samples": sum(row["correct_samples"] for row in rows),
        "questions_with_any_correct_sample": sum(row["correct_samples"] > 0 for row in rows),
        "questions_with_four_or_more_matching_votes": sum(row["four_plus_agreement"] for row in rows),
        "questions_with_four_or_more_votes_for_gold": sum(row["four_plus_votes_for_gold"] for row in rows),
        "questions_with_unique_four_plus_leader": sum(row["unique_four_plus_answer"] is not None for row in rows),
        "unique_four_plus_leader_correct": sum(row["unique_four_plus_correct"] for row in rows),
        "questions_with_four_four_tie": sum(row["four_four_tie"] for row in rows),
        "four_four_ties_containing_gold": sum(row["four_four_tie"] and row["four_plus_votes_for_gold"] for row in rows),
        "questions_with_strict_majority": sum(row["strict_majority_answer"] is not None for row in rows),
        "strict_majority_correct": sum(row["strict_majority_answer"] == row["gold_answer"] for row in rows),
        "prompt_tokens": sum(usage.get("prompt_tokens") or 0 for usage in usages),
        "completion_tokens": sum(usage.get("completion_tokens") or 0 for usage in usages),
        "median_request_latency_s": statistics.median(latencies),
        "first_request_at_utc": started,
        "all_samples_completed_at_utc": ended,
        "wall_clock_s": (datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds(),
        "no_thinking_requested": True,
        "responses_with_reasoning_content": sum(record.get("reasoning_content_present", False) for record in records),
        "config_file": "config.json",
        "sample_table_file": "samples.csv",
        "results": rows,
    }


def save_samples_csv(path: Path, records: list[dict]) -> None:
    columns = [
        "problem_idx", "sample_number", "candidate", "gold_answer", "correct",
        "finish_reason", "parse_method", "latency_s", "prompt_tokens",
        "completion_tokens", "reasoning_content_present", "trace_file",
    ]
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in columns})
    temp.replace(path)


async def main_async(args: argparse.Namespace) -> None:
    problems = load_problems()
    source = json.loads((ROOT / "data" / "source.json").read_text())
    dataset_path = ROOT / "data" / "aime_2025_problems.jsonl"
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if dataset_sha256 != source["dataset_sha256"]:
        raise ValueError("Dataset SHA256 differs from data/source.json")
    model_dir = Path(args.model_dir).expanduser().resolve()
    launch_config = model_dir / "vllm.yaml"
    if not launch_config.is_file():
        raise FileNotFoundError(launch_config)
    settings = {
        "model": MODEL,
        "endpoint": args.url + "/v1/chat/completions",
        "source_repo_commit": git_head(ROOT),
        "model_repo_commit": git_head(model_dir.parent.parent),
        "model_launch_config": str(launch_config),
        "model_launch_config_sha256": hashlib.sha256(launch_config.read_bytes()).hexdigest(),
        "vllm_version": version("vllm"),
        "dataset": source,
        "questions": 30,
        "samples_per_question": SAMPLES,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "max_generation_attempts_per_slot": args.attempts,
        "seed_base": SEED_BASE,
        "seed_formula": "seed_base + (problem_idx - 1) * 8 + (sample_number - 1) + (generation_attempt - 1) * 1000",
        "sampling": {"temperature": 0.8, "top_p": 0.9, "top_k": 40},
        "chat_template_kwargs": {"enable_thinking": False},
        "system_prompt": SYSTEM_PROMPT,
        "key_in_model_inputs": False,
        "grading": "Local exact integer match; only complete parseable responses vote",
        "voting": "At least 4 of 8 identical parseable answers. A 4-4 tie meets the agreement threshold but has no unique prediction; report gold-vote count separately.",
    }
    out = Path(args.out).expanduser().resolve()
    if not out.is_relative_to(ROOT / "runs"):
        raise ValueError("Output must stay under repository runs/")
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing["settings"] != settings:
            raise ValueError("Existing run uses different settings; choose another --out")
        config = existing
    else:
        config = {"settings": settings, "started_at_utc": utc_now()}
        atomic_json(config_path, config)
    (out / "questions").mkdir(exist_ok=True)

    timeout = httpx.Timeout(connect=30, read=args.timeout, write=30, pool=30)
    limits = httpx.Limits(max_connections=args.concurrency + 10)
    semaphore = asyncio.Semaphore(args.concurrency)
    new_requests = 0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        models = await client.get(args.url + "/v1/models")
        models.raise_for_status()
        served_ids = [entry.get("id") for entry in models.json().get("data", [])]
        if MODEL not in served_ids:
            raise RuntimeError(f"Expected {MODEL} on vLLM, found {served_ids}")

        async def one(row: dict, sample_number: int) -> dict:
            nonlocal new_requests
            index = row["problem_idx"]
            path = out / "questions" / f"{index:02d}" / f"{sample_number:02d}.json"
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                previous = json.loads(path.read_text())
                if previous.get("complete"):
                    return previous
            record = {
                "problem_idx": index,
                "sample_number": sample_number,
                "gold_answer": row["answer"],
                "problem": row["problem"],
                "attempts": [],
                "candidate": None,
                "complete": False,
            }
            async with semaphore:
                for generation_attempt in range(1, args.attempts + 1):
                    seed = SEED_BASE + (index - 1) * SAMPLES + sample_number - 1 + (generation_attempt - 1) * 1000
                    request = model_request(row["problem"], seed, args.max_tokens)
                    started = time.perf_counter()
                    started_utc = utc_now()
                    attempt = {"generation_attempt": generation_attempt, "started_at_utc": started_utc, "request": request}
                    try:
                        response = await client.post(args.url + "/v1/chat/completions", json=request)
                        attempt["http_status"] = response.status_code
                        attempt["latency_s"] = round(time.perf_counter() - started, 3)
                        attempt["finished_at_utc"] = utc_now()
                        try:
                            body = response.json()
                        except ValueError:
                            body = {"raw_text": response.text}
                        attempt["response"] = body
                        if response.status_code == 200 and isinstance(body, dict) and body.get("choices"):
                            choice = body["choices"][0]
                            message = choice.get("message") or {}
                            candidate, method = parse_candidate(message.get("content"), choice.get("finish_reason"))
                            attempt["candidate"] = candidate
                            attempt["parse_method"] = method
                            record["attempts"].append(attempt)
                            if candidate is not None:
                                record.update({
                                    "candidate": candidate,
                                    "parse_method": method,
                                    "finish_reason": choice.get("finish_reason"),
                                    "usage": body.get("usage"),
                                    "latency_s": attempt["latency_s"],
                                    "reasoning_content_present": bool(message.get("reasoning") or message.get("reasoning_content")),
                                })
                                break
                        else:
                            record["attempts"].append(attempt)
                            if response.status_code not in RETRYABLE:
                                break
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        attempt["latency_s"] = round(time.perf_counter() - started, 3)
                        attempt["finished_at_utc"] = utc_now()
                        attempt["error"] = f"{type(exc).__name__}: {exc}"
                        record["attempts"].append(attempt)
                    if generation_attempt < args.attempts:
                        await asyncio.sleep(min(2 ** generation_attempt, 8))
            if record["attempts"] and record["candidate"] is None:
                last = next(
                    (attempt for attempt in reversed(record["attempts"])
                     if isinstance(attempt.get("response"), dict)
                     and attempt["response"].get("choices")),
                    None,
                )
                body = last["response"] if last else {}
                choices = body.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    record.update({
                        "finish_reason": choices[0].get("finish_reason"),
                        "usage": body.get("usage"),
                        "latency_s": last["latency_s"],
                        "parse_method": last.get("parse_method", "unparseable"),
                        "reasoning_content_present": bool(message.get("reasoning") or message.get("reasoning_content")),
                    })
            record["correct"] = record["candidate"] == row["answer"]
            record["complete"] = any(
                isinstance(attempt.get("response"), dict)
                and attempt["response"].get("choices")
                for attempt in record["attempts"]
            )
            record["reasoning_content_present"] = any(
                response_has_reasoning(attempt.get("response"))
                for attempt in record["attempts"]
            )
            record["completed_at_utc"] = utc_now()
            atomic_json(path, record)
            new_requests += 1
            print(f"{new_requests:03d}/240 Q{index:02d} #{sample_number}: {record['candidate']}", flush=True)
            return record

        tasks = [one(row, sample_number) for sample_number in range(1, SAMPLES + 1) for row in problems]
        all_records = await asyncio.gather(*tasks)

    all_records.sort(key=lambda record: (record["problem_idx"], record["sample_number"]))
    if len(all_records) != 240 or any(not record.get("complete") for record in all_records):
        raise RuntimeError("Expected all 240 slots to finish")
    rows = []
    for row in problems:
        group = [record for record in all_records if record["problem_idx"] == row["problem_idx"]]
        rows.append(vote_row(row["problem_idx"], row["answer"], group))
    ended = utc_now()
    started = min(
        attempt["started_at_utc"]
        for record in all_records for attempt in record["attempts"]
    )
    summary = summarize(rows, all_records, started, ended)
    table_records = []
    for record in all_records:
        usage = record.get("usage") or {}
        table_records.append({
            "problem_idx": record["problem_idx"],
            "sample_number": record["sample_number"],
            "candidate": record["candidate"],
            "gold_answer": record["gold_answer"],
            "correct": record["correct"],
            "finish_reason": record.get("finish_reason"),
            "parse_method": record.get("parse_method"),
            "latency_s": record.get("latency_s"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_content_present": record.get("reasoning_content_present", False),
            "trace_file": f"questions/{record['problem_idx']:02d}/{record['sample_number']:02d}.json",
        })
    save_samples_csv(out / "samples.csv", table_records)
    atomic_json(out / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-dir", default="~/models/Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default=str(ROOT / "runs" / RUN_NAME))
    parser.add_argument("--concurrency", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or args.attempts < 1:
        parser.error("concurrency, max-tokens, and attempts must be positive")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
