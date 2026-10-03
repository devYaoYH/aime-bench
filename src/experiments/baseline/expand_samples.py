"""Expand an existing baseline to eight independent samples and compute answer votes.

Use this experiment after a baseline run to add samples 2-8 using each question
record's original payload. It reuses saved successful attempts and writes
self_consistency/ records plus pass@8, modal-vote, and strict-majority statistics.
Requires local raw baseline traces and OPENROUTER_API_KEY; new attempts are paid.
    python -m src.experiments.baseline.expand_samples --run 20260930-155212
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import httpx

from src.common import OPENROUTER_URL, RETRYABLE, ROOT, atomic_json, extract_answer, load_key


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize_question(run: Path, index: int) -> dict:
    original = read_json(run / "questions" / f"{index:02d}.json")
    attempts = [original]
    for number in range(2, 9):
        attempts.append(read_json(run / "self_consistency" / "questions" / f"{index:02d}" / f"{number:02d}.json"))
    gold = int(original["gold_answer"])
    votes = Counter(str(record["candidate"]) for record in attempts if record.get("candidate") is not None)
    highest = max(votes.values(), default=0)
    leaders = sorted((int(value) for value, count in votes.items() if count == highest))
    modal = leaders[0] if len(leaders) == 1 else None
    strict_majority = modal if highest >= 5 else None
    attempt_rows = []
    for number, record in enumerate(attempts, 1):
        usage = record.get("usage") or {}
        attempt_rows.append({
            "attempt": number,
            "candidate": record.get("candidate"),
            "correct": record.get("correct") is True,
            "finish_reason": record.get("finish_reason"),
            "api_latency_s": record.get("total_api_latency_s"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost": usage.get("cost"),
            "trace_file": f"questions/{index:02d}.json" if number == 1 else f"self_consistency/questions/{index:02d}/{number:02d}.json",
        })
    return {
        "problem_idx": index,
        "gold_answer": gold,
        "attempts": attempt_rows,
        "correct_attempts": sum(row["correct"] for row in attempt_rows),
        "pass_at_8": any(row["correct"] for row in attempt_rows),
        "votes": [{"answer": int(value), "count": count} for value, count in sorted(votes.items(), key=lambda pair: (-pair[1], int(pair[0])))],
        "modal_answer": modal,
        "modal_votes": highest if modal is not None else 0,
        "modal_correct": modal == gold,
        "majority_answer": strict_majority,
        "majority_correct": strict_majority == gold,
    }


async def main_async(args: argparse.Namespace) -> None:
    run = ROOT / "runs" / args.run
    base_summary = read_json(run / "summary.json")
    original_config = read_json(run / "config.json")
    if base_summary["model"] != "qwen/qwen3-30b-a3b" or base_summary["questions"] != 30:
        raise ValueError("Expected the completed 30-question Qwen run")
    key = load_key()
    output = run / "self_consistency"
    output.mkdir(exist_ok=True)
    (output / "questions").mkdir(exist_ok=True)
    config = {
        "source_run": args.run,
        "model": original_config["model"],
        "endpoint": OPENROUTER_URL,
        "new_attempts_per_question": 7,
        "total_attempts_per_question": 8,
        "concurrency": args.concurrency,
        "sampling": {name: original_config[name] for name in ("temperature", "top_p", "top_k", "max_tokens", "reasoning")},
        "grading": "local exact integer match against MathArena answer; no external grader",
        "voting": "modal parseable final answer; tied top counts give no modal answer; strict majority requires at least 5 of 8 votes; missing answers abstain",
    }
    config_path = output / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise RuntimeError("Existing sample expansion has a different configuration")
    atomic_json(config_path, config)
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=max(args.concurrency + 5, 35))
    timeout = httpx.Timeout(connect=30, read=args.api_timeout, write=30, pool=30)
    first_request_mono = None
    first_request_utc = None
    newly_completed = 0

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        async def one(index: int, number: int) -> None:
            nonlocal first_request_mono, first_request_utc, newly_completed
            question_dir = output / "questions" / f"{index:02d}"
            question_dir.mkdir(exist_ok=True)
            path = question_dir / f"{number:02d}.json"
            if path.exists():
                previous = read_json(path)
                if previous.get("response") and previous.get("correct") is not None:
                    return
            original = read_json(run / "questions" / f"{index:02d}.json")
            request = original["request"]  # Exact original endpoint payload and sampling parameters.
            record = {
                "problem_idx": index,
                "sample_number": number,
                "problem": original["problem"],
                "request": request,
                "api_attempts": [],
                "response": None,
                "candidate": None,
                "gold_answer": original["gold_answer"],
                "correct": None,
            }
            async with semaphore:
                for retry in range(1, args.retries + 2):
                    started = time.perf_counter()
                    started_utc = utc_now()
                    if first_request_mono is None:
                        first_request_mono, first_request_utc = started, started_utc
                    attempt = {"attempt": retry, "started_at_utc": started_utc}
                    try:
                        response = await client.post(
                            OPENROUTER_URL,
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "AIME 2025 self consistency"},
                            json=request,
                        )
                        attempt["http_status"] = response.status_code
                        attempt["latency_s"] = round(time.perf_counter() - started, 3)
                        attempt["finished_at_utc"] = utc_now()
                        try:
                            body = response.json()
                        except ValueError:
                            body = {"raw_text": response.text}
                        attempt["response"] = body
                        record["api_attempts"].append(attempt)
                        if response.status_code == 200 and isinstance(body, dict) and body.get("choices"):
                            record["response"] = body
                            record["generation_latency_s"] = attempt["latency_s"]
                            break
                        if response.status_code not in RETRYABLE:
                            break
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        attempt["latency_s"] = round(time.perf_counter() - started, 3)
                        attempt["finished_at_utc"] = utc_now()
                        attempt["error"] = f"{type(exc).__name__}: {exc}"
                        record["api_attempts"].append(attempt)
                    if retry <= args.retries:
                        await asyncio.sleep(min(2 ** retry, 8))
            record["total_api_latency_s"] = round(sum(item["latency_s"] for item in record["api_attempts"]), 3)
            if record["response"]:
                choice = record["response"]["choices"][0]
                record["candidate"] = extract_answer((choice.get("message") or {}).get("content"))
                record["usage"] = record["response"].get("usage")
                record["finish_reason"] = choice.get("finish_reason")
            record["correct"] = record["candidate"] is not None and int(record["candidate"]) == record["gold_answer"]
            record["grading_completed_at_utc"] = utc_now()
            atomic_json(path, record)
            newly_completed += 1
            print(f"{newly_completed:03d}/210 Q{index:02d} #{number}: answer={record['candidate']} correct={record['correct']} api={record['total_api_latency_s']:.1f}s", flush=True)

        # FIFO scheduling gives every problem one request before its second new sample.
        await asyncio.gather(*(one(index, number) for number in range(2, 9) for index in range(1, 31)))

    if first_request_mono is None and (output / "summary.json").is_file():
        print(f"All samples already complete; summary: {output / 'summary.json'}")
        return
    rows = [summarize_question(run, index) for index in range(1, 31)]
    new_attempts = [attempt for row in rows for attempt in row["attempts"][1:]]
    if len(new_attempts) != 210 or any(row["api_latency_s"] is None for row in new_attempts):
        raise RuntimeError("Some new attempts are incomplete")
    http_attempts_total = 0
    saved_records = []
    for index in range(1, 31):
        for number in range(2, 9):
            record = read_json(output / "questions" / f"{index:02d}" / f"{number:02d}.json")
            saved_records.append(record)
            http_attempts_total += len(record["api_attempts"])
            if not record.get("response"):
                raise RuntimeError(f"Q{index:02d} attempt {number} has no valid model response; rerun to retry")
    first_request_utc = min(record["api_attempts"][0]["started_at_utc"] for record in saved_records)
    finished_utc = max(record["grading_completed_at_utc"] for record in saved_records)
    elapsed_wall_s = (datetime.fromisoformat(finished_utc) - datetime.fromisoformat(first_request_utc)).total_seconds()
    summary = {
        "run_id": args.run,
        "model": original_config["model"],
        "questions": 30,
        "attempts_per_question": 8,
        "configured_concurrency": args.concurrency,
        "new_inference_requests": 210,
        "http_attempts_total": http_attempts_total,
        "http_retries": http_attempts_total - 210,
        "requests_completed_this_invocation": newly_completed,
        "pass_at_1": base_summary["pass_at_1"],
        "pass_at_8": sum(row["pass_at_8"] for row in rows) / 30,
        "pass_at_8_correct": sum(row["pass_at_8"] for row in rows),
        "modal_vote_accuracy": sum(row["modal_correct"] for row in rows) / 30,
        "modal_vote_correct": sum(row["modal_correct"] for row in rows),
        "majority_vote_accuracy": sum(row["majority_correct"] for row in rows) / 30,
        "majority_vote_correct": sum(row["majority_correct"] for row in rows),
        "questions_with_strict_majority": sum(row["majority_answer"] is not None for row in rows),
        "first_inference_request_at_utc": first_request_utc,
        "all_grading_completed_at_utc": finished_utc,
        "wall_clock_first_request_to_all_grading_s": round(elapsed_wall_s, 3),
        "process_timer_elapsed_s": round(time.perf_counter() - first_request_mono, 3) if first_request_mono is not None else None,
        "new_prompt_tokens": sum(row["prompt_tokens"] or 0 for row in new_attempts),
        "new_completion_tokens": sum(row["completion_tokens"] or 0 for row in new_attempts),
        "new_reasoning_tokens": sum(row["reasoning_tokens"] or 0 for row in new_attempts),
        "new_reported_cost": round(sum(row["cost"] or 0 for row in new_attempts), 9),
        "new_api_latency_median_s": round(statistics.median(row["api_latency_s"] for row in new_attempts), 3),
        "results": rows,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="20260930-155212")
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--api-timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 0:
        parser.error("concurrency must be positive and retries nonnegative")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
