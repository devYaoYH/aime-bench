"""Analyze reasoning lengths and same-model continuation checkpoints retrospectively.

Use this baseline analysis after the eight-sample expansion to measure exact
Qwen reasoning tokens, checkpoint success, and generation/verification budgets.
Writes trajectory_metrics.json, checkpoint_summary.json, budget_summary.json,
and plots under the source run's analysis/. Requires raw local traces and a
Qwen tokenizer; an uncached tokenizer is downloaded. No model calls occur, and
these results do not measure the benefit of switching to a different model.
    python -m src.experiments.baseline.analyze_escalation
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from src.common import ROOT, atomic_json
from src.tokenizer_utils import QWEN_TOKENIZER_REPO as TOKENIZER_REPO


CHECKPOINTS = (1500, 3000, 5000, 7000, 9000, 11000, 13000, 15000)
WINDOWS = (2048, 4096, 8192)


def trace_paths(run: Path) -> list[Path]:
    paths = sorted((run / "questions").glob("*.json"))
    paths += sorted((run / "self_consistency" / "questions").glob("*/*.json"))
    return paths


def metric_rows(run: Path, tokenizer: Tokenizer) -> list[dict]:
    rows = []
    for path in trace_paths(run):
        trace = json.loads(path.read_text())
        message = trace["response"]["choices"][0]["message"]
        reasoning = message.get("reasoning") or ""
        usage = trace.get("usage") or {}
        sample = trace.get("sample_number", 1)
        row = {
            "problem_idx": trace["problem_idx"],
            "sample_number": sample,
            "reasoning_tokens": len(tokenizer.encode(reasoning, add_special_tokens=False).ids),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": trace.get("finish_reason"),
            "candidate": trace.get("candidate"),
            "correct": trace.get("correct"),
            "api_latency_s": trace.get("generation_latency_s"),
            "trace_file": str(path.relative_to(run)),
        }
        if row["correct"] is True:
            row["outcome"] = "correct"
        elif row["finish_reason"] == "stop":
            row["outcome"] = "wrong"
        else:
            row["outcome"] = "capped"
        rows.append(row)
    rows.sort(key=lambda row: (row["problem_idx"], row["sample_number"]))
    if len(rows) != 240 or any(len([row for row in rows if row["problem_idx"] == q]) != 8 for q in range(1, 31)):
        raise ValueError("Expected eight trajectories for each of 30 questions")
    return rows


def bootstrap_interval(rows: list[dict], checkpoint: int, window: int, seed: int = 2025) -> list[float]:
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["problem_idx"]].append(row)
    questions = sorted(by_question)
    rng = random.Random(seed + checkpoint + window)
    rates = []
    for _ in range(2000):
        sampled = [row for _ in questions for row in by_question[rng.choice(questions)]]
        active = [row for row in sampled if row["reasoning_tokens"] >= checkpoint]
        if not active:
            continue
        hit = sum(row["correct"] and row["completion_tokens"] <= checkpoint + window for row in active)
        rates.append(hit / len(active))
    rates.sort()
    return [round(rates[int(0.025 * (len(rates) - 1))], 4), round(rates[int(0.975 * (len(rates) - 1))], 4)]


def checkpoint_summary(rows: list[dict]) -> list[dict]:
    result = []
    for checkpoint in CHECKPOINTS:
        active = [row for row in rows if row["reasoning_tokens"] >= checkpoint]
        item = {
            "reasoning_checkpoint_tokens": checkpoint,
            "active_trajectories": len(active),
            "active_questions": len({row["problem_idx"] for row in active}),
            "eventually_correct_by_original_cap": sum(row["correct"] for row in active),
            "windows": [],
        }
        for window in WINDOWS:
            hit = sum(row["correct"] and row["completion_tokens"] <= checkpoint + window for row in active)
            item["windows"].append({
                "additional_output_tokens": window,
                "correct_final_answers": hit,
                "rate": round(hit / len(active), 4) if active else None,
                "question_cluster_bootstrap_95pct": bootstrap_interval(rows, checkpoint, window) if active else None,
            })
        result.append(item)
    return result


def budget_summary(rows: list[dict]) -> dict:
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["problem_idx"]].append(row)
    generation_frontier = []
    for n in range(1, 9):
        subsets = [group[:n] for _, group in sorted(by_question.items())]
        generation_frontier.append({
            "generations_per_question": n,
            "correct_questions": sum(any(row["correct"] for row in group) for group in subsets),
            "final_answers_generated": sum(row["candidate"] is not None for group in subsets for row in group),
            "model_requests": 30 * n,
        })

    verification_frontier = []
    for n in range(1, 9):
        correct_questions = submissions = questions_with_candidates = 0
        for group in by_question.values():
            unique = []
            seen = set()
            for row in group:
                candidate = row["candidate"]
                if candidate is not None and candidate not in seen:
                    seen.add(candidate)
                    unique.append(row)
            selected = unique[:n]
            correct_questions += any(row["correct"] for row in selected)
            submissions += len(selected)
            questions_with_candidates += bool(selected)
        verification_frontier.append({
            "max_distinct_final_answers_verified_per_question": n,
            "correct_questions": correct_questions,
            "submissions_total": submissions,
            "questions_with_final_candidate": questions_with_candidates,
            "model_requests_to_collect_all_candidates": 240,
        })

    stop_on_first_final = []
    for question, group in sorted(by_question.items()):
        requests = next((i for i, row in enumerate(group, 1) if row["candidate"] is not None), 8)
        selected = group[requests - 1]
        stop_on_first_final.append({
            "problem_idx": question,
            "requests_until_first_final_or_8": requests,
            "first_final_candidate": selected["candidate"],
            "first_final_correct": selected["correct"],
            "serial_per_question_api_latency_s": round(sum(row["api_latency_s"] for row in group[:requests]), 3),
        })
    return {
        "generation_pass_at_n": generation_frontier,
        "distinct_candidate_verification_budget": verification_frontier,
        "stop_on_first_final": {
            "model_requests": sum(row["requests_until_first_final_or_8"] for row in stop_on_first_final),
            "correct_questions": sum(row["first_final_correct"] for row in stop_on_first_final),
            "questions_with_final_candidate": sum(row["first_final_candidate"] is not None for row in stop_on_first_final),
            "max_serial_per_question_api_latency_s": round(max(row["serial_per_question_api_latency_s"] for row in stop_on_first_final), 3),
            "by_question": stop_on_first_final,
        },
    }


def plot_distribution(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"correct": "#11765b", "wrong": "#b4812e", "capped": "#c35a43"}
    labels = {"correct": "Correct final answer", "wrong": "Wrong final answer", "capped": "No final answer · output cap"}
    fig, (axis, count_axis) = plt.subplots(
        1, 2, figsize=(14, 15.3), sharey=True,
        gridspec_kw={"width_ratios": [8.5, 1.15], "wspace": 0.04},
    )
    fig.patch.set_facecolor("white")
    for question in range(1, 31):
        axis.axhspan(question - 0.5, question + 0.5, color="#f5f7f4" if question % 2 == 0 else "white", zorder=0)
        group = [row for row in rows if row["problem_idx"] == question]
        for row in group:
            jitter = (row["sample_number"] - 4.5) * 0.075
            marker = "o" if row["outcome"] != "wrong" else "x"
            axis.scatter(
                row["reasoning_tokens"], question + jitter,
                s=34 if marker == "o" else 46, marker=marker,
                color=colors[row["outcome"]], alpha=0.85, linewidths=1.4,
                zorder=3,
            )
        count_axis.barh(question, sum(row["correct"] for row in group), color=colors["correct"], height=0.56, zorder=3)
    axis.axvline(1500, color="#364d4a", linestyle="--", linewidth=1.2, zorder=2)
    axis.set_xlim(0, 16850)
    axis.set_ylim(30.6, 0.3)
    axis.set_yticks(range(1, 31), [f"Q{question:02d}" for question in range(1, 31)])
    axis.set_xticks(range(0, 17000, 2000), [f"{tick // 1000}k" if tick else "0" for tick in range(0, 17000, 2000)])
    axis.set_xlabel("Reasoning trace tokens (official Qwen3 tokenizer)", labelpad=10)
    axis.grid(axis="x", color="#e0e6e1", linewidth=0.8, zorder=1)
    axis.tick_params(axis="both", labelsize=9, colors="#465653", length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#cbd5ce")
    count_axis.set_xlim(0, 8.5)
    count_axis.set_xticks([0, 2, 4, 6, 8])
    count_axis.set_xlabel("Correct / 8", labelpad=10)
    count_axis.tick_params(axis="x", labelsize=9, colors="#465653", length=0)
    count_axis.tick_params(axis="y", left=False)
    count_axis.grid(axis="x", color="#e0e6e1", linewidth=0.8, zorder=1)
    for side in ("top", "right", "left"):
        count_axis.spines[side].set_visible(False)
    count_axis.spines["bottom"].set_color("#cbd5ce")
    handles = [Line2D([0], [0], marker="x" if outcome == "wrong" else "o", linestyle="", color=colors[outcome], label=labels[outcome], markersize=7) for outcome in ("correct", "wrong", "capped")]
    handles.append(Line2D([0], [0], color="#364d4a", linestyle="--", label="1,500-token Jev checkpoint"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.1, 0.935), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("AIME 2025 · reasoning length across eight Qwen attempts per question", x=0.1, y=0.986, ha="left", fontsize=15, fontweight="bold", color="#182b28")
    fig.text(0.1, 0.956, "Each mark is one saved trajectory (n = 240). Capped attempts ended at 16,384 total output tokens.", fontsize=9, color="#52645e")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.925, bottom=0.055)
    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="20260930-155212")
    parser.add_argument("--tokenizer-json", type=Path)
    args = parser.parse_args()
    run = ROOT / "runs" / args.run
    tokenizer_path = args.tokenizer_json or Path(hf_hub_download(repo_id=TOKENIZER_REPO, filename="tokenizer.json"))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = metric_rows(run, tokenizer)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    metadata = {
        "run_id": args.run,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "attempts": len(rows),
        "reasoning_tokens_definition": "tokenizer.encode(message.reasoning, add_special_tokens=False)",
        "rows": rows,
    }
    atomic_json(out / "trajectory_metrics.json", metadata)
    atomic_json(out / "checkpoint_summary.json", {"run_id": args.run, "checkpoints": checkpoint_summary(rows)})
    atomic_json(out / "budget_summary.json", {"run_id": args.run, **budget_summary(rows)})
    plot_distribution(rows, out / "reasoning_tokens_by_question")
    print(f"Analyzed {len(rows)} trajectories in {out}")


if __name__ == "__main__":
    main()
