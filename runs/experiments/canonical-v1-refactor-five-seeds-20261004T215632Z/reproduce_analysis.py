"""Re-audit saved paired trials without reading grader keys or audit gold fields."""

import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from runner.extensions.validation.refactor_v1 import aggregate, measure

BATCH = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def questions(attempt):
    return [read(p) for p in sorted(attempt.glob("trace/*/question.json"))]


def initial_overlap(old, new):
    rows = []
    for index in range(1, 31):
        relative = Path("trace") / f"{index:02d}" / "rollout-01/tokens.json"
        a = read(old / relative)["output_token_ids"]
        b = read(new / relative)["output_token_ids"]
        overlap = min(len(a), len(b))
        prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), overlap)
        rows.append({
            "problem_idx": index,
            "reference_retained_tokens": len(a),
            "new_retained_tokens": len(b),
            "matching_prefix_tokens": prefix,
            "retained_overlap_equal": prefix == overlap,
        })
    return rows


def late_solves(attempt):
    solved = sorted(
        (q for q in questions(attempt) if q.get("first_solved")),
        key=lambda q: q["first_solved"]["first_solved_elapsed_s"],
    )
    return [{"problem_idx": q["problem_idx"], **q["first_solved"]} for q in solved[-3:]]


def q9_evidence(attempt):
    folder = attempt / "trace/09"
    q = read(folder / "question.json")
    checks = [json.loads(line) for line in (folder / "verification.jsonl").read_text().splitlines() if line]
    correct = next(row for row in checks if row["result"]["verdict"] is True)
    telemetry = read(folder / f'rollout-{correct["rollout"]:02d}/telemetry.json')
    first = read(folder / "rollout-01/telemetry.json")
    return {
        "first_solved_elapsed_s": q["first_solved"]["first_solved_elapsed_s"],
        "first_segment_latency_s": first["generation_latency_s"],
        "first_segment_retained_tokens": first["generated_token_ids_count"],
        "winning_segment_latency_s": telemetry["generation_latency_s"],
        "winning_segment_retained_tokens": telemetry["generated_token_ids_count"],
        "candidate_observed_at_utc": correct["observed_at_utc"],
        "candidate_seconds_from_continuation_round_start": correct["question_elapsed_s"],
        "candidate_queue_wait_s": correct["candidate_queue_wait_s"],
        "verification_latency_s": correct["verification_latency_s"],
        "retained_token_scope": "All received segment tokens, including tokens received while grading; not an extraction-token threshold.",
    }


def main():
    summary, config = read(BATCH / "summary.json"), read(BATCH / "config.json")
    original = read(ROOT / "runs/experiments" / config["reference_batch"] / "summary.json")
    references = {r["sampling_seed"]: r for r in original["trials"]}
    rows = []
    for saved in summary["trials"]:
        reference = references[saved["sampling_seed"]]
        old = ROOT / "attempts" / reference["attempt_id"]
        new = ROOT / "attempts" / saved["attempt_id"]
        audited = measure(new, reference, config["core_manifest_sha256"])
        assert all(saved[k] == v for k, v in audited.items()), saved["sampling_seed"]
        old_summary = read(old / "summary.json")
        old_questions = questions(old)
        old_solved = {q["problem_idx"] for q in old_questions if q.get("first_solved")}
        new_solved = set(audited["solved_indices"])
        rows.append({
            "seed": saved["sampling_seed"],
            "audit": audited,
            "reference_performance": old_summary["performance"],
            "reference_grader_timeline": old_summary["grader_timeline"],
            "reference_continuations": sum(r["continuation_of_rollout"] is not None for q in old_questions for r in q["rollouts"]),
            "solved_overlap_count": len(old_solved & new_solved),
            "reference_only_solved": sorted(old_solved - new_solved),
            "new_only_solved": sorted(new_solved - old_solved),
            "initial_token_overlap": initial_overlap(old, new),
            "reference_late_solves": late_solves(old),
            "new_late_solves": late_solves(new),
        })
    screen = aggregate(summary["trials"])
    assert all(summary[k] == v for k, v in screen.items())
    new_times = [r["audit"]["time_to_target_s"] for r in rows]
    old_times = [r["audit"]["reference_time_to_target_s"] for r in rows]
    slow = max(rows, key=lambda r: r["audit"]["delta_s"])
    analysis = {
        "batch": BATCH.name,
        "practical_screen": screen,
        "reference_mean_s": statistics.mean(old_times),
        "new_mean_s": statistics.mean(new_times),
        "mean_relative_change": statistics.mean(new_times) / statistics.mean(old_times) - 1,
        "reference_median_fresh_trial_ttft_s": statistics.median(r["reference_performance"]["fresh_ttft"]["median_s"] for r in rows),
        "new_median_fresh_trial_ttft_s": statistics.median(r["audit"]["performance"]["fresh_ttft"]["median_s"] for r in rows),
        "verified_initial_payloads": sum(30 - len(r["audit"]["initial_payload_mismatches"]) for r in rows),
        "verified_continuations": sum(r["audit"]["continuation_requests"] for r in rows),
        "trials": rows,
        "slow_seed": slow["seed"],
        "slow_seed_q9_reference": q9_evidence(ROOT / "attempts" / slow["audit"]["reference_attempt_id"]),
        "slow_seed_q9_new": q9_evidence(ROOT / "attempts" / slow["audit"]["attempt_id"]),
        "interpretation": "Median screening passes, but tail worsens and decodes differ. This historical comparison does not establish identical stochastic behavior, statistical equivalence, or causal attribution of the outlier to or away from refactoring.",
    }
    (BATCH / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    with (BATCH / "paired_timings.csv").open("w", newline="") as stream:
        fields = ["seed", "reference_s", "new_s", "delta_s", "solved", "requests", "continuations", "wrong_checks", "grader_service_s", "grader_idle_s"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            r = row["audit"]
            writer.writerow(dict(zip(fields, [row["seed"], r["reference_time_to_target_s"], r["time_to_target_s"], r["delta_s"], r["solved"], r["generation_requests"], r["continuation_requests"], r["wrong_checks"], r["grader_timeline"]["actual_service_s"], r["grader_timeline"]["idle_between_queries_s"]])))
    print(json.dumps({k: v for k, v in analysis.items() if k not in ["trials", "slow_seed_q9_reference", "slow_seed_q9_new"]}, indent=2))


if __name__ == "__main__":
    main()
