"""Audit post-freeze client evidence and render extended milestone figures.

Never reads benchmark datasets or local answer keys. Reproduce from versioned
attempts and batch records without running generation or the grader.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics

from src.common import ROOT, atomic_json
from src.experiments.post_freeze_analysis import (
    MILESTONES, aggregate_milestones, baseline_accuracy, events, marginal_trial, read_json,
)
from runner_final.integrity import verify_core


def write_csv(path, rows, fields):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_a(folder, retained, protocol, source):
    config = read_json(folder / "config.json")
    assert config["git_commit"] == source and not config["git_dirty"]
    assert config["core_manifest_sha256"] == verify_core(ROOT) == protocol["core_manifest_sha256"]
    assert config["system_prompt_sha256"] == protocol["system_prompt_sha256"]
    assert config["target_correct"] == 30 and config["seed"] == retained["seed"]
    assert config["benchmark"] and config["skip_benchmark_prewarm"]
    assert config["question_indices"] == list(range(1, 31))
    assert config["max_attempts_per_question"] == 4
    control = read_json(ROOT / "attempts" / protocol["reference_attempts"][str(retained["seed"])] / "config.json")
    for key in ("model", "launch_profile", "runtime_package_versions", "temperature", "top_p", "schedule",
                "max_tokens", "first_pass_max_tokens", "max_attempts_per_question", "max_rounds", "parallelism",
                "rollouts", "grader_cost", "question_timeout", "benchmark", "skip_benchmark_prewarm"):
        assert config.get(key) == control.get(key), f"Frozen control changed: {key}"
    prefix_checks = 0
    unsolved_patterns = []
    question_paths = sorted(folder.glob("trace/*/question.json"))
    assert len(question_paths) == 30
    for q in question_paths:
        question = read_json(q)
        index = question["problem_idx"]
        if not question.get("first_solved"):
            unsolved_patterns.append({"problem_idx": index, "unique_candidates": question["unique_candidates"],
                                      "output_tokens": sum(r["generated_token_ids_count"] for r in question["rollouts"]),
                                      "requests": len(question["rollouts"]),
                                      "finish_reasons": [r.get("finish_reason") for r in question["rollouts"]]})
        assert 1 <= len(question["rollouts"]) <= 4
        verdicts = events(q.parent / "verification.jsonl")
        pairs = [event["candidate"] for event in verdicts]
        assert len(pairs) == len(set(pairs)), "Repeated per-question grading"
        if question.get("first_solved"):
            solved = question["first_solved"]
            matched = [event for event in verdicts if event.get("result", {}).get("query_id") == solved["grader_query_id"]]
            assert len(matched) == 1 and matched[0]["result"]["verdict"] is True
            assert solved["candidate"] == matched[0]["candidate"]
            assert solved["grader_answered_at_utc"] == matched[0]["result"]["answered_at"]
        for rollout in question["rollouts"]:
            path = q.parent / f"rollout-{rollout['rollout']:02d}"
            request = read_json(path / "request.json")
            tokens = read_json(path / "tokens.json")
            assert request["seed"] == retained["seed"] + index * 4 + rollout["rollout"]
            assert len(tokens["output_token_ids"]) == rollout["generated_token_ids_count"]
            assert request["max_tokens"] <= (8192 if rollout["rollout"] == 1 else 16384)
            if rollout["rollout"] == 1:
                ref = ROOT / "attempts" / protocol["reference_attempts"][str(retained["seed"])] / "trace" / f"{index:02d}" / "rollout-01/request.json"
                assert request == read_json(ref), "Initial payload drift"
            if rollout["continuation_of_rollout"]:
                parent = read_json(q.parent / f"rollout-{rollout['continuation_of_rollout']:02d}" / "tokens.json")
                assert request["prompt"] == parent["prompt_token_ids"] + parent["output_token_ids"]
                assert tokens["prompt_token_ids"] == request["prompt"]
                prefix_checks += 1
    measured = marginal_trial(folder)
    assert measured == retained["measurement"], "Driver/independent measurement drift"
    return {**retained, "audit_passed": True, "exact_prefix_checks": prefix_checks,
            "measurement": measured, "first18_grader": first18_grader(folder), "unsolved_patterns": unsolved_patterns}



def first18_grader(folder):
    """Reconstruct service/idle through the eighteenth positive client verdict."""
    config = read_json(folder / "config.json")
    questions = [read_json(p) for p in folder.glob("trace/*/question.json")]
    first = sorted((q["first_solved"] for q in questions if q.get("first_solved")),
                   key=lambda e: e["first_solved_elapsed_s"])
    if len(first) < 18:
        return None
    ts = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    cutoff = ts(first[17]["grader_answered_at_utc"])
    checks = [e["result"] for p in folder.glob("trace/*/verification.jsonl") for e in events(p)
              if type(e.get("result", {}).get("verdict")) is bool and ts(e["result"]["answered_at"]) <= cutoff]
    checks.sort(key=lambda e: ts(e["picked_at"]))
    service = sum(ts(e["answered_at"]) - ts(e["picked_at"]) for e in checks)
    correct = sum(e["verdict"] is True for e in checks)
    assert correct == 18
    return {"correct": correct, "wrong": len(checks)-correct,
            "first_pick_s": ts(checks[0]["picked_at"])-ts(config["official_started_at_utc"]),
            "service_s": service,
            "wrong_service_s": sum(ts(e["answered_at"])-ts(e["picked_at"]) for e in checks if e["verdict"] is False),
            "idle_s": ts(checks[-1]["answered_at"])-ts(checks[0]["picked_at"])-service,
            "endpoint": "Grader answered eighteenth distinct correct; client target adds small HTTP receipt delay",
            "accounting": "service_s includes wrong_service_s. Do not add wrong service twice."}


def render(batch, milestones, historical, rows):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aime-post-freeze-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    for row in rows:
        if row.get("audit_passed"):
            ranks = row["measurement"]["first_correct"]
            ax.plot([r["rank"] for r in ranks], [r["first_solved_elapsed_s"] for r in ranks],
                    color="#91b7c2", lw=.8, alpha=.65)
    reached = [m for m in milestones if m["reached"]]
    x = np.array([r["milestone"] for r in reached])
    ax.fill_between(x, [r["min_s"] for r in reached], [r["max_s"] for r in reached], color="#bcd5dc", alpha=.45)
    ax.plot(x, [r["median_s"] for r in reached], "o-", color="#245f73", lw=2, ms=4, label="New target-30 median (range shaded)")
    hx = [int(r["verified_correct"]) for r in historical]
    ax.plot(hx, [float(r["median_s"]) for r in historical], "s:", color="#bd643f", lw=1.7, ms=3.5, label="Historical stop-at-18 median")
    ax.plot([0, 30], [0, 90], "--", color="#646d77", lw=1.3, label="Serial grader floor: 3s x M")
    for m in reached:
        if m["milestone"] >= 18:
            ax.annotate(f"{m['reached']}/5", (m["milestone"], m["median_s"]),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.text(.985, .33, f"28: {next(r['reached'] for r in milestones if r['milestone']==28)}/5 reached\n30: {next(r['reached'] for r in milestones if r['milestone']==30)}/5 reached", transform=ax.transAxes,
            ha="right", va="center", fontsize=8.3, color="#5b6875")
    ax.set(xlim=(0, 30.7), ylim=(0, None), xlabel="Distinct verified-correct questions",
           ylabel="Official time (seconds)")
    ax.set_xticks([0, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    ax.grid(axis="y", color="#e2e7ea")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.3)
    fig.tight_layout()
    fig.savefig(batch / "extended-milestones.png", dpi=190)
    fig.savefig(batch / "extended-milestones.pdf")
    plt.close(fig)


def analyze(batch):
    config = read_json(batch / "config.json")
    protocol = config["protocol"]
    assert protocol["protocol_sha256"] == hashlib.sha256((ROOT/"configs/experiments/post-freeze-measurements-v1.json").read_bytes()).hexdigest()
    for path, digest in config["source_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, f"Measurement source drift: {path}"
    task_a, task_b = read_json(batch / "task_a.json"), read_json(batch / "task_b.json")
    rows = []
    for retained in task_a["trials"]:
        row = dict(retained)
        if retained.get("measurement"):
            try:
                row = audit_a(ROOT / "attempts" / retained["attempt_id"], retained, protocol["task_a"], config["source_commit"])
            except Exception as error:
                row.update(audit_passed=False, identity_valid=False, audit_error=f"{type(error).__name__}: {error}")
        else:
            row.update(audit_passed=False, identity_valid=False)
        rows.append(row)
    milestones = aggregate_milestones(rows)
    with (ROOT / "docs/reports/final/analysis/reporting-figures/core-v1-milestone-summary.csv").open() as file:
        historical = list(csv.DictReader(file))
    historical_map = {int(r["verified_correct"]): float(r["median_s"]) for r in historical}
    comparisons = [{**r, "historical_e8_median_s": historical_map.get(r["milestone"]),
                    "new_minus_historical_s": r["median_s"] - historical_map[r["milestone"]]
                    if r["median_s"] is not None and r["milestone"] in historical_map else None}
                   for r in milestones]
    accuracy = baseline_accuracy(ROOT / "attempts" / task_b["attempt_id"])
    assert accuracy == read_json(batch / "baseline_accuracy.json")
    baseconfig = read_json(ROOT / "attempts" / task_b["attempt_id"] / "config.json")
    assert baseconfig["git_commit"] == config["source_commit"] and not baseconfig["git_dirty"]
    assert baseconfig["launch_profile"]["gpu-memory-utilization"] == .95
    assert baseconfig["max_context_tokens"] == 16384
    assert not baseconfig["stop_at_target"] and not baseconfig["cancel_siblings_on_correct"]
    assert baseconfig["seed"] == protocol["task_b"]["seed"]
    old = read_json(ROOT / "attempts" / protocol["task_b"]["historical_reference"] / "config.json")
    assert baseconfig["system_prompt"] == old["system_prompt"]
    for key in ("temperature", "top_p", "max_tokens", "parallelism", "rollouts", "max_attempts_per_question"):
        assert baseconfig[key] == old[key], f"Baseline control changed: {key}"
    for index in range(1,31):
        for rollout in range(1,5):
            request = read_json(ROOT / "attempts" / task_b["attempt_id"] / "trace" / f"{index:02d}" / f"rollout-{rollout:02d}" / "request.json")
            assert request["seed"] == baseconfig["seed"] + index*4 + rollout
            assert request["max_tokens"] == min(16384, 16384-baseconfig["prompt_tokens_by_question"][str(index)])
    strict_vote = sum(q["vote_correct"] and max(q["answer_counts"].values(), default=0) >= 3 for q in accuracy["per_question"])
    first = rows[0].get("measurement", {}).get("milestones_s", {}).get("18")
    firstconfig = read_json(ROOT / "attempts" / rows[0]["attempt_id"] / "config.json") if rows[0].get("attempt_id") else {}
    parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    cold = ((parse(firstconfig["official_started_at_utc"]) - parse(task_a["started_at_utc"])).total_seconds() + first
            if first is not None and firstconfig.get("official_started_at_utc") else None)
    result = {"batch": batch.name, "source_commit": config["source_commit"],
              "all_task_a_audits_passed": len(rows)==5 and all(r.get("audit_passed") for r in rows),
              "task_a": rows, "milestones": comparisons, "task_b_accuracy": accuracy,
              "task_b_strict_majority_3_of_4": strict_vote,
              "task_b_timing": {k: task_b.get("summary", {}).get(k) for k in
                                ("official_latency_s", "generation_completed_latency_s", "time_to_18_s", "initialization_and_attempt_latency_s")},
              "task_b_first18_grader": first18_grader(ROOT/"attempts"/task_b["attempt_id"]),
              "first_after_launch_to_18_s": cold,
              "cold_scope": "Task A server-start phase through first trial eighteenth verdict, including launch and cheap warmup; one observation, not repeated cold-start validation.",
              "missing_scope": "Unreached milestones are null. Statistics use reached identity-valid trials and show reach counts, not imputed 900-second results.",
              "comparison_scope": "Historical same-seed controls; sequential batches, not interleaved. Core policy before the eighteenth verdict is unchanged, but matching seeds/payloads does not guarantee identical generated paths.",
              "scoring_scope": "Client/oracle verdicts only; dataset answer fields are never read by this analysis."}
    atomic_json(batch / "analysis.json", result)
    ranks = [{"seed":r["seed"],"attempt_id":r.get("attempt_id"),**e} for r in rows for e in r.get("measurement",{}).get("first_correct",[])]
    write_csv(batch/"first-correct-ranks.csv",ranks,["seed","attempt_id","rank","problem_idx","first_solved_elapsed_s","first_solved_at_utc","grader_query_id"])
    per_seed = [{"seed":r["seed"],"attempt_id":r.get("attempt_id"),"end_condition":r["end_condition"],
                 "solved":r.get("measurement",{}).get("solved"),**r.get("measurement",{}).get("milestones_s",{})} for r in rows]
    write_csv(batch/"milestones-per-seed.csv",per_seed,["seed","attempt_id","end_condition","solved",*map(str,MILESTONES)])
    write_csv(batch/"milestone-summary.csv",comparisons,list(comparisons[0]))
    write_csv(batch/"baseline-samples.csv",accuracy["samples"],list(accuracy["samples"][0]) if accuracy["samples"] else ["problem_idx"])
    write_csv(batch/"baseline-questions.csv",accuracy["per_question"],list(accuracy["per_question"][0]) if accuracy["per_question"] else ["problem_idx"])
    render(batch,milestones,historical,rows)
    summary_markdown(batch,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("task_a","task_b_accuracy")},indent=2))
    return result



def summary_markdown(batch, result):
    """Readable deliverable alongside the complete compact records."""
    fmt = lambda value: f"{value:.3f}" if value is not None else ""
    lines = ["# Post-freeze measurements", "", f"Batch: \x60{batch.name}\x60. Source: \x60{result['source_commit']}\x60.",
             "", "## Task A: immutable core v1, target 30", "",
             "| Seed | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["task_a"]:
        times = row.get("measurement", {}).get("milestones_s", {})
        lines.append("| " + str(row["seed"]) + " | " + " | ".join(fmt(times.get(str(n))) for n in range(14,31,2)) + " |")
    lines += ["", "Times are official seconds. A blank is unreached, not zero.", "",
              "| Milestone | Reached | Median (s) | Minimum (s) | Maximum (s) | Historical E8 median (s) | Difference (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in result["milestones"]:
        lines.append(f"| {m['milestone']} | {m['reached']}/5 | {fmt(m['median_s'])} | {fmt(m['min_s'])} | {fmt(m['max_s'])} | {fmt(m['historical_e8_median_s'])} | {fmt(m['new_minus_historical_s'])} |")
    lines += ["", "Medians and ranges use reached, identity-valid trials. They are not confidence intervals and later milestones may have fewer observations.",
              "", "| Seed | Solved | Checks / wrong | Service (s) | Later idle (s) | Official total (s) | End condition |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in result["task_a"]:
        m = r.get("measurement", {}); g = m.get("grader_timeline") or {}
        lines.append(f"| {r['seed']} | {m.get('solved','')} | {m.get('completed_client_checks','')} / {m.get('wrong_client_checks','')} | {fmt(g.get('actual_service_s'))} | {fmt(g.get('idle_between_queries_s'))} | {fmt(m.get('official_latency_s'))} | {r['end_condition']} |")
    lines += ["", "Every first-correct rank and grader query ID is in [first-correct-ranks.csv](first-correct-ranks.csv). Request counts and never-solved questions are retained in [analysis.json](analysis.json); every continuation prefix is audited against exact token IDs.",
              "", "![Five declared-seed milestone curve, reached-trial median and observed range](extended-milestones.png)", "",
              "The orange curve is the retained historical stop-at-18 batch, rather than the first section of the new curve. The dashed line is 3 seconds times the number of distinct successes.",
              "", "### Task A report text", ""]
    by = {m["milestone"]: m for m in result["milestones"]}
    if by[18]["reached"]:
        lines.append(f"The frozen core reached 18 in {by[18]['reached']}/5 extended trials, with median {fmt(by[18]['median_s'])}s and range {fmt(by[18]['min_s'])}-{fmt(by[18]['max_s'])}s. The historical stop-at-18 median was 77.277s; the new median differs by {fmt(by[18]['new_minus_historical_s'])}s. At 24 correct, {by[24]['reached']}/5 trials reached the milestone" + (f", with median {fmt(by[24]['median_s'])}s." if by[24]["median_s"] is not None else ".") + f" The 28- and 30-answer milestones were reached by {by[28]['reached']}/5 and {by[30]['reached']}/5 trials, respectively. These are matched historical-seed comparisons across sequential server batches, not interleaved measurements.")
    a = result["task_b_accuracy"]; t = result["task_b_timing"]
    lines += ["", "## Task B: complete BF16 final-only pass@4", "",
              "Seed 20261003; 30 questions, four samples each, 16K total context, 95% GPU allocation. All samples finish naturally or at the token cap; correct verdicts and the eighteenth success never cancel generation.",
              "", "| Metric | Result | Definition |", "| --- | ---: | --- |",
              f"| Completed samples | {a['observed_samples']}/120 | Full batch required for scored accuracy |",
              f"| pass@1 | {a['correct_samples']}/120" + (f" ({100*a['pass1']:.2f}%)" if a["pass1"] is not None else " (unscored)") + " | Mean per-question correct fraction |",
              f"| pass@4 coverage | {a['pass4_correct_questions']}/30 | At least one correct sample |",
              f"| Unique-plurality vote | {a['majority_vote_correct_questions']}/30 | Missing answers abstain; ties/no votes incorrect |",
              f"| Strict 3-of-4 vote | {result['task_b_strict_majority_3_of_4']}/30 | At least three identical correct final votes |",
              f"| Token capped | {a['token_capped_count']}/120 | No answer extracted from capped samples |",
              f"| No answer | {a['no_answer_count']}/120 | Capped or no eligible final integer |",
              f"| Unique grader checks | {a['unique_grader_checks']} | One check per question/answer pair |",
              f"| All generation finished | {fmt(t['generation_completed_latency_s'])}s | Official start to final generation end |",
              f"| Generation and grading finished | {fmt(t['official_latency_s'])}s | Includes draining all candidate checks |",
              f"| Time to 18 | {fmt(t['time_to_18_s'])}s | Eighteenth distinct positive verdict within full run |",
              "", "See [baseline-samples.csv](baseline-samples.csv) for all sample/verdict mappings and [baseline-questions.csv](baseline-questions.csv) for correct counts out of four and vote outcomes.",
              "", "### Task B report text", ""]
    if a["valid"]:
        lines.append(f"The full BF16 baseline completed all 120 samples with seed 20261003. Pass@1 was {a['correct_samples']}/120 ({100*a['pass1']:.2f}%), pass@4 covered {a['pass4_correct_questions']}/30 questions, and unique-plurality voting covered {a['majority_vote_correct_questions']}/30 (ties and no votes count incorrect). Token-capped and no-answer rates were {100*a['token_capped_rate']:.2f}% and {100*a['no_answer_rate']:.2f}%. Generation finished in {fmt(t['generation_completed_latency_s'])}s and all grading in {fmt(t['official_latency_s'])}s, with 18 correct reached at {fmt(t['time_to_18_s'])}s. This is one seed at 95% allocation with cancellation disabled, whereas the historical timing baseline used 80% allocation and cancellation.")
    else:
        lines.append("Accuracy is unscored because the full terminal/verdict evidence did not validate; the failure and observed samples are retained.")
    lines += ["", "## Provenance, timing and limitations", "",
              "- [Declared protocol and hashes](config.json), [Task A outcomes](task_a.json), [Task B outcomes](task_b.json), [implementation diff](implementation-diff.patch) and [baseline-policy diff](baseline-policy-diff.patch).",
              "- The frozen v1 manifest is unchanged. The only Task A solving-control override is target 30; the external 900-second deadline excludes setup and warmup.",
              "- Existing startup answer-field provenance validation was explicitly approved. Model inputs and analysis omit reference answers; correctness uses grader verdicts only.",
              f"- First after-launch time through 18: {fmt(result['first_after_launch_to_18_s'])}s, including Task A server startup and cheap warmup. This is one cold observation.",
              "- Service already contains wrong-check time. Adding wrong service again would double-count it.",
              "- No replacement trials or tuned seeds. Raw streams, full grader audits and service logs remain on the remote.",
              "- [Time allocation and unattended compute log](time-allocation.md); wall-clock activity intervals are separate from measured human focused hours.", ""]
    (batch / "summary.md").write_text("\n".join(lines))



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",required=True)
    args=parser.parse_args()
    if Path(args.batch).name != args.batch or args.batch in (".",".."):
        parser.error("Use a plain batch directory name")
    analyze(ROOT/"results/post_freeze"/args.batch)


if __name__=="__main__":main()
