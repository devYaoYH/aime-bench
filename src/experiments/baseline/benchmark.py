"""Run the original single-pass Qwen/OpenRouter AIME baseline.

Use this experiment to obtain one independent sample for each of 30 questions
with concurrent requests, a configurable output cap, and local exact-key grading.
A fresh runs/<timestamp>/ directory receives config, full question records, and
summary statistics; capped or missing final answers count as incorrect.
Requires OPENROUTER_API_KEY and the dataset; execution makes paid API calls.
    python -m src.experiments.baseline.benchmark --concurrency 10
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import httpx

from src.common import ROOT, OPENROUTER_URL, RETRYABLE, atomic_json, extract_answer, load_key, load_problems, utc_now

MODEL = "qwen/qwen3-30b-a3b"
SYSTEM_PROMPT = (
    "Solve the mathematical problem carefully. Reason step by step. "
    "End your final response with the AIME integer answer in the exact format "
    "Answer: NNN, where NNN is an integer from 0 to 999."
)


async def main_async(args: argparse.Namespace) -> None:
    problems = load_problems()
    key = load_key()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "questions").mkdir()
    source = json.loads((ROOT / "data" / "source.json").read_text())
    run_config = {
        "model": MODEL,
        "dataset": source,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "reasoning": {"enabled": True, "exclude": False},
        "grading": "local exact integer match against MathArena answer",
        "prompt": SYSTEM_PROMPT,
        "started_at_utc": utc_now(),
    }
    atomic_json(out / "config.json", run_config)

    timeout = httpx.Timeout(connect=30, read=args.api_timeout, write=30, pool=30)
    limits = httpx.Limits(max_connections=max(args.concurrency + 5, 15))
    semaphore = asyncio.Semaphore(args.concurrency)
    first_request_mono: float | None = None
    first_request_utc: str | None = None

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        async def one_problem(row: dict) -> dict:
            nonlocal first_request_mono, first_request_utc
            index = row["problem_idx"]
            request_body = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["problem"]},
                ],
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": args.max_tokens,
                "reasoning": {"enabled": True, "exclude": False},
                "stream": False,
            }
            record = {
                "problem_idx": index,
                "problem": row["problem"],
                "request": request_body,
                "api_attempts": [],
                "response": None,
                "candidate": None,
                "gold_answer": row["answer"],
                "correct": None,
            }
            path = out / "questions" / f"{index:02d}.json"
            async with semaphore:
                for attempt_number in range(1, args.retries + 2):
                    started_mono = time.perf_counter()
                    started_utc = utc_now()
                    if first_request_mono is None:
                        first_request_mono = started_mono
                        first_request_utc = started_utc
                    attempt = {"attempt": attempt_number, "started_at_utc": started_utc}
                    try:
                        response = await client.post(
                            OPENROUTER_URL,
                            headers={
                                "Authorization": f"Bearer {key}",
                                "Content-Type": "application/json",
                                "X-Title": "AIME 2025 benchmark",
                            },
                            json=request_body,
                        )
                        attempt["http_status"] = response.status_code
                        attempt["latency_s"] = round(time.perf_counter() - started_mono, 3)
                        attempt["finished_at_utc"] = utc_now()
                        try:
                            body = response.json()
                        except ValueError:
                            body = {"raw_text": response.text}
                        attempt["response"] = body  # Preserve even error payloads.
                        record["api_attempts"].append(attempt)
                        if response.status_code == 200 and isinstance(body, dict) and body.get("choices"):
                            record["response"] = body
                            record["generation_latency_s"] = attempt["latency_s"]
                            break
                        if response.status_code not in RETRYABLE:
                            break
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        attempt["latency_s"] = round(time.perf_counter() - started_mono, 3)
                        attempt["finished_at_utc"] = utc_now()
                        attempt["error"] = f"{type(exc).__name__}: {exc}"
                        record["api_attempts"].append(attempt)
                    if attempt_number <= args.retries:
                        await asyncio.sleep(min(2 ** attempt_number, 8))

            record["total_api_latency_s"] = round(
                sum(item["latency_s"] for item in record["api_attempts"]), 3
            )
            if record["response"]:
                message = record["response"]["choices"][0].get("message", {})
                record["candidate"] = extract_answer(message.get("content"))
                record["usage"] = record["response"].get("usage")
                record["finish_reason"] = record["response"]["choices"][0].get("finish_reason")
            atomic_json(path, record)

            grade_started = time.perf_counter()
            record["correct"] = (
                record["candidate"] is not None
                and int(record["candidate"]) == row["answer"]
            )
            record["grading_latency_s"] = round(time.perf_counter() - grade_started, 3)
            record["grading_completed_at_utc"] = utc_now()
            atomic_json(path, record)
            usage = record.get("usage") or {}
            verdict = record["correct"]
            print(
                f"{index:02d}: {'correct' if verdict else 'wrong'} "
                f"answer={record['candidate']} api={record['total_api_latency_s']:.1f}s "
                f"tokens={usage.get('prompt_tokens')}/{usage.get('completion_tokens')}",
                flush=True,
            )
            return record

        records = await asyncio.gather(*(one_problem(row) for row in problems))

    finished_mono = time.perf_counter()
    if first_request_mono is None:
        raise RuntimeError("No inference request was sent")
    correct = sum(record["correct"] is True for record in records)
    graded = len(records)
    prompt_tokens = sum((record.get("usage") or {}).get("prompt_tokens") or 0 for record in records)
    completion_tokens = sum((record.get("usage") or {}).get("completion_tokens") or 0 for record in records)
    reasoning_tokens = sum(
        ((record.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        for record in records
    )
    latency = [record["total_api_latency_s"] for record in records]
    all_grading_completed_utc = utc_now()
    elapsed_wall_s = (datetime.fromisoformat(all_grading_completed_utc) - datetime.fromisoformat(first_request_utc)).total_seconds()
    summary = {
        "model": MODEL,
        "questions": len(records),
        "graded": graded,
        "correct": correct,
        "pass_at_1": correct / len(records),
        "first_inference_request_at_utc": first_request_utc,
        "all_grading_completed_at_utc": all_grading_completed_utc,
        "wall_clock_first_request_to_all_grading_s": round(elapsed_wall_s, 3),
        "process_timer_elapsed_s": round(finished_mono - first_request_mono, 3),
        "api_latency_sum_s": round(sum(latency), 3),
        "api_latency_median_s": round(statistics.median(latency), 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens_reported": reasoning_tokens,
        "results": [
            {
                "problem_idx": record["problem_idx"],
                "candidate": record["candidate"],
                "correct": record["correct"],
                "gold_answer": record["gold_answer"],
                "total_api_latency_s": record["total_api_latency_s"],
                "generation_latency_s": record.get("generation_latency_s"),
                "prompt_tokens": (record.get("usage") or {}).get("prompt_tokens"),
                "completion_tokens": (record.get("usage") or {}).get("completion_tokens"),
                "reasoning_tokens": ((record.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "finish_reason": record.get("finish_reason"),
                "trace_file": f"questions/{record['problem_idx']:02d}.json",
            }
            for record in records
        ],
    }
    atomic_json(out / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")))
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--api-timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 0:
        parser.error("concurrency must be positive and retries nonnegative")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
