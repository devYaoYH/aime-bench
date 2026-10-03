"""Rank unfinished baseline trajectories using Jev continuation decisions.

Use this experiment to ask typesafe/jev-1.13 four Noul questions about full
capped Qwen reasoning: promise, coherent route, proximity, and repetition.
Problem/trace inputs exclude grades and keys; results are forecasts rather than
observed continuation successes. Saves reusable decisions and rankings in
jev_review/. Requires local baseline records and a paid OpenRouter connection.
    python -m src.experiments.jev.judge_trajectories --run 20260930-155212
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx

from src.common import ROOT, atomic_json, load_key


MODEL = "typesafe/jev-1.13"
URL = "https://openrouter.ai/api/alpha/decisions"
EXTRA_TOKENS = 8192
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}
QUESTIONS = {
    "promising_to_extend": {
        "type": "noul",
        "instructions": (
            "If the same Qwen3 solver continues from the very end of this truncated reasoning trace "
            "for at most 8192 more output tokens, is it likely to reach a mathematically justified "
            "correct AIME integer answer without restarting the solution?"
        ),
        "criteria": {
            "true": "The current path has a sound, useful setup and remaining steps look feasible within the continuation budget.",
            "false": "The path is fundamentally wrong, persistently looping, or too far from a justified answer to finish within the budget.",
        },
    },
    "coherent_route": {
        "type": "noul",
        "instructions": "Does the reasoning trace contain a coherent mathematical route that could solve the stated problem?",
        "criteria": {
            "true": "A relevant method, equation, invariant, counting setup, or geometric construction has been derived and is usable.",
            "false": "The trace is largely unguided speculation, unrelated calculations, or mutually incompatible approaches with no usable route.",
        },
    },
    "near_solution": {
        "type": "noul",
        "instructions": "Near the end of the trace, is the solver within a few concrete derivation or arithmetic steps of a justified AIME integer answer?",
        "criteria": {
            "true": "The necessary derivation is mostly complete; only a small number of explicit steps remain.",
            "false": "Important mathematical work or a major missing insight remains.",
        },
    },
    "stuck_or_repeating": {
        "type": "noul",
        "instructions": "In the latter half of the trace, is the solver stuck in circular reasoning or repeatedly revisiting approaches without substantial progress?",
        "criteria": {
            "true": "Repeated attempts or contradictions consume substantial text with little new progress.",
            "false": "The latter half makes meaningful forward progress, even if unfinished.",
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def source_run(name: str | None) -> Path:
    runs = ROOT / "runs"
    if name:
        path = runs / name
    else:
        choices = sorted(path for path in runs.iterdir() if (path / "summary.json").is_file())
        if not choices:
            raise FileNotFoundError("No completed benchmark runs found")
        path = choices[-1]
    if not path.is_dir() or not (path / "summary.json").is_file():
        raise FileNotFoundError(path)
    return path


def failed_indices(run: Path) -> list[int]:
    summary = json.loads((run / "summary.json").read_text())
    return [int(row["problem_idx"]) for row in summary["results"] if not row["correct"]]


def payload_for(trace: dict) -> dict:
    response = trace.get("response") or {}
    message = (response.get("choices") or [{}])[0].get("message") or {}
    reasoning = message.get("reasoning") or ""
    if not reasoning:
        raise ValueError(f"Question {trace['problem_idx']} has no reasoning trace")
    # Do not send gold_answer, grading result, or any other answer-key data.
    return {
        "model": MODEL,
        "state": {
            "problem": trace["problem"],
            "reasoning_trace": reasoning,
            "solver_model": "qwen/qwen3-30b-a3b",
            "trace_status": "Generation stopped at its 16,384-token output cap before a final answer.",
            "continuation_budget_output_tokens": EXTRA_TOKENS,
        },
        "questions": QUESTIONS,
    }


def read_probability(body: dict, question_id: str) -> float:
    value = body["answers"][question_id]["noul"]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
        raise ValueError(f"Invalid Jev probability for {question_id}: {value!r}")
    return float(value)


def write_summary(run: Path, indices: list[int]) -> dict:
    out = run / "jev_review"
    records = []
    for index in indices:
        path = out / f"{index:02d}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("probabilities"):
                records.append(record)
    ordered = sorted(records, key=lambda row: row["probabilities"]["promising_to_extend"], reverse=True)
    summary = {
        "model": MODEL,
        "run_id": run.name,
        "failed_trajectories": len(indices),
        "judged": len(records),
        "continuation_budget_output_tokens": EXTRA_TOKENS,
        "question_wording": QUESTIONS,
        "note": "Jev Noul values are subjective predictions about continuation, not observed success rates or validated math proofs. No gold answers were sent.",
        "ranked": [
            {
                "problem_idx": row["problem_idx"],
                **row["probabilities"],
                "latency_s": row["latency_s"],
                "input_tokens": (row["response"].get("usage") or {}).get("input_tokens"),
                "cost": (row["response"].get("usage") or {}).get("cost"),
                "trace_file": f"jev_review/{row['problem_idx']:02d}.json",
            }
            for row in ordered
        ],
    }
    atomic_json(out / "summary.json", summary)
    return summary


async def main_async(args: argparse.Namespace) -> None:
    run = source_run(args.run)
    indices = failed_indices(run)
    if args.indices:
        requested = set(args.indices)
        if not requested.issubset(indices):
            raise ValueError(f"Only failed questions may be judged: {sorted(requested - set(indices))}")
        selected = [index for index in indices if index in requested]
    else:
        selected = indices
    if not selected:
        raise ValueError("No failed trajectories selected")

    out = run / "jev_review"
    out.mkdir(exist_ok=True)
    config_path = out / "config.json"
    config = {
        "model": MODEL,
        "source_run": run.name,
        "endpoint": URL,
        "continuation_budget_output_tokens": EXTRA_TOKENS,
        "questions": QUESTIONS,
        "uses_gold_answers": False,
    }
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Existing Jev review uses a different configuration")
    atomic_json(config_path, config)
    key = load_key()
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=max(args.concurrency + 2, 8))
    timeout = httpx.Timeout(connect=30, read=120, write=60, pool=30)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def one(index: int) -> None:
            path = out / f"{index:02d}.json"
            if path.exists() and json.loads(path.read_text()).get("probabilities"):
                print(f"{index:02d}: already judged", flush=True)
                return
            trace = json.loads((run / "questions" / f"{index:02d}.json").read_text())
            payload = payload_for(trace)
            reasoning = payload["state"]["reasoning_trace"]
            record = {
                "problem_idx": index,
                "request": payload,
                "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
                "reasoning_chars": len(reasoning),
                "attempts": [],
                "response": None,
                "probabilities": None,
            }
            async with semaphore:
                for number in range(1, args.retries + 2):
                    started = time.perf_counter()
                    attempt = {"number": number, "started_at_utc": utc_now()}
                    try:
                        response = await client.post(
                            URL,
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "AIME trajectory review"},
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
                            record["probabilities"] = {
                                question_id: read_probability(body, question_id) for question_id in QUESTIONS
                            }
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
            record["latency_s"] = round(sum(item["latency_s"] for item in record["attempts"]), 3)
            atomic_json(path, record)
            if record["probabilities"]:
                p = record["probabilities"]
                print(f"{index:02d}: promising={p['promising_to_extend']:.2f} coherent={p['coherent_route']:.2f} near={p['near_solution']:.2f} stuck={p['stuck_or_repeating']:.2f}", flush=True)
            else:
                print(f"{index:02d}: no valid decision; see {path}", flush=True)

        await asyncio.gather(*(one(index) for index in selected))
    summary = write_summary(run, indices)
    print(f"Judged {summary['judged']}/{summary['failed_trajectories']} failed trajectories")
    print(f"Summary: {out / 'summary.json'}")
    if not args.indices and summary["judged"] != summary["failed_trajectories"]:
        raise RuntimeError("Some trajectories were not judged; rerun to retry them")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Run directory name; defaults to latest completed run")
    parser.add_argument("--indices", type=int, nargs="*", help="Judge only these failed question indices")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 0:
        parser.error("concurrency must be positive and retries nonnegative")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
