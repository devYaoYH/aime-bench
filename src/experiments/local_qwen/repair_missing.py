"""Resample no-thinking pass@8 slots lacking a valid AIME integer answer.

Use this only after the initial run. It preserves every prior raw attempt,
continues the documented seed sequence, and records this repair's code commit
in the run config. The same summary and candidate table are then rebuilt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import httpx

from src.common import ROOT, atomic_json, load_problems, utc_now
from src.experiments.local_qwen.no_thinking_pass8 import (
    MODEL,
    RUN_NAME,
    SAMPLES,
    SEED_BASE,
    git_head,
    model_request,
    parse_candidate,
    response_has_reasoning,
    save_samples_csv,
    summarize,
    vote_row,
)


def rebuild(out: Path, problems: list[dict]) -> dict:
    records = []
    for row in problems:
        for sample_number in range(1, SAMPLES + 1):
            path = out / "questions" / f"{row['problem_idx']:02d}" / f"{sample_number:02d}.json"
            records.append(json.loads(path.read_text()))
    records.sort(key=lambda item: (item["problem_idx"], item["sample_number"]))
    if len(records) != 240 or any(record["candidate"] is None for record in records):
        raise RuntimeError("All 240 slots must have parseable answers before finalizing")
    rows = []
    for problem in problems:
        group = [record for record in records if record["problem_idx"] == problem["problem_idx"]]
        rows.append(vote_row(problem["problem_idx"], problem["answer"], group))
    started = min(attempt["started_at_utc"] for record in records for attempt in record["attempts"])
    summary = summarize(rows, records, started, utc_now())
    table = []
    for record in records:
        usage = record.get("usage") or {}
        table.append({
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
    save_samples_csv(out / "samples.csv", table)
    atomic_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "runs" / RUN_NAME))
    parser.add_argument("--max-total-attempts", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    if not out.is_relative_to(ROOT / "runs"):
        parser.error("Output must stay under repository runs/")
    config_path = out / "config.json"
    config = json.loads(config_path.read_text())
    settings = config["settings"]
    if settings["model"] != MODEL or settings["chat_template_kwargs"] != {"enable_thinking": False}:
        raise ValueError("Expected the original no-thinking Qwen run")
    launch_config = Path(settings["model_launch_config"])
    if hashlib.sha256(launch_config.read_bytes()).hexdigest() != settings["model_launch_config_sha256"]:
        raise ValueError("vLLM launch config changed since the initial run")
    problems = load_problems()
    missing = []
    for row in problems:
        for sample_number in range(1, SAMPLES + 1):
            path = out / "questions" / f"{row['problem_idx']:02d}" / f"{sample_number:02d}.json"
            record = json.loads(path.read_text())
            if record["candidate"] is None:
                missing.append((row, sample_number, path, record))
    if not missing:
        print("All 240 slots already have parseable answers")
        return
    event = {
        "producer": "src.experiments.local_qwen.repair_missing",
        "producer_commit": git_head(ROOT),
        "started_at_utc": utc_now(),
        "max_total_attempts_per_slot": args.max_total_attempts,
        "slots": [f"{row['problem_idx']:02d}/{number:02d}" for row, number, _, _ in missing],
        "extra_attempts": 0,
    }
    config.setdefault("repair_events", []).append(event)
    atomic_json(config_path, config)
    url = settings["endpoint"]
    with httpx.Client(timeout=httpx.Timeout(connect=30, read=args.timeout, write=30, pool=30)) as client:
        models = client.get(url.removesuffix("/v1/chat/completions") + "/v1/models")
        models.raise_for_status()
        served_ids = [item.get("id") for item in models.json().get("data", [])]
        if MODEL not in served_ids:
            raise RuntimeError(f"Expected {MODEL} on vLLM, found {served_ids}")
        for row, sample_number, path, record in missing:
            index = row["problem_idx"]
            for generation_attempt in range(len(record["attempts"]) + 1, args.max_total_attempts + 1):
                seed = SEED_BASE + (index - 1) * SAMPLES + sample_number - 1 + (generation_attempt - 1) * 1000
                request = model_request(row["problem"], seed, settings["max_tokens"])
                started = time.perf_counter()
                attempt = {
                    "generation_attempt": generation_attempt,
                    "started_at_utc": utc_now(),
                    "request": request,
                }
                response = client.post(url, json=request)
                attempt["latency_s"] = round(time.perf_counter() - started, 3)
                attempt["finished_at_utc"] = utc_now()
                attempt["http_status"] = response.status_code
                try:
                    body = response.json()
                except ValueError:
                    body = {"raw_text": response.text}
                attempt["response"] = body
                choices = (body.get("choices") or []) if isinstance(body, dict) and response.status_code == 200 else []
                candidate = None
                if choices:
                    choice = choices[0]
                    message = choice.get("message") or {}
                    candidate, method = parse_candidate(message.get("content"), choice.get("finish_reason"))
                    attempt["candidate"] = candidate
                    attempt["parse_method"] = method
                    if candidate is not None:
                        record.update({
                            "candidate": candidate,
                            "correct": candidate == row["answer"],
                            "parse_method": method,
                            "finish_reason": choice.get("finish_reason"),
                            "usage": body.get("usage"),
                            "latency_s": attempt["latency_s"],
                            "completed_at_utc": utc_now(),
                        })
                record["attempts"].append(attempt)
                record["reasoning_content_present"] = any(
                    response_has_reasoning(item.get("response")) for item in record["attempts"]
                )
                atomic_json(path, record)
                event["extra_attempts"] += 1
                print(f"Q{index:02d} #{sample_number} attempt {generation_attempt}: {candidate}", flush=True)
                if candidate is not None:
                    break
    event["finished_at_utc"] = utc_now()
    event["resolved_slots"] = sum(json.loads(path.read_text())["candidate"] is not None for _, _, path, _ in missing)
    atomic_json(config_path, config)
    summary = rebuild(out, problems)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
