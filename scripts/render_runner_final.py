"""Validate and render all five final-runner results from saved artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aime-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def analyze(batch):
    config = json.loads((batch / "config.json").read_text())
    summary = json.loads((batch / "summary.json").read_text())
    protocol = config["protocol"]
    assert summary["status"] == "complete"
    assert [r["sampling_seed"] for r in summary["trials"]] == protocol["seeds"]
    assert len(summary["trials"]) == 5
    rows = []
    for trial in summary["trials"]:
        row = {"trial": trial["trial"], "seed": trial["sampling_seed"],
               "attempt_id": trial.get("attempt_id"), "valid": trial.get("valid", False),
               "error": trial.get("error")}
        if not row["valid"]:
            rows.append(row)
            continue
        folder = ROOT / "attempts" / row["attempt_id"]
        saved = json.loads((folder / "config.json").read_text())
        s = json.loads((folder / "summary.json").read_text())
        prewarm = json.loads((folder / "prewarm.json").read_text())
        assert saved["git_commit"] == config["source_commit"] and not saved["git_dirty"]
        assert saved["runner_id"] == "runner_final_v1" and saved["benchmark_year"] == 2025
        assert (prewarm["year"], prewarm["questions"], prewarm["generation_requests"], prewarm["grader_queries"]) == (2024, 30, 30, 0)
        assert not prewarm["correctness_evaluated"]
        assert prewarm["prefix_cache_reset_before"]["success"] and prewarm["prefix_cache_reset_after"]["success"]
        assert prewarm["finished_at_utc"] < saved["official_started_at_utc"]
        questions = [json.loads(p.read_text()) for p in sorted(folder.glob("trace/*/question.json"))]
        assert len(questions) == 30 and all(len(q["rollouts"]) <= 4 for q in questions)
        solved = [q for q in questions if q.get("first_solved")]
        assert len({q["problem_idx"] for q in solved}) == 18
        for q in solved:
            event, verdict = q["first_solved"], q["winner"]["result"]
            assert verdict["verdict"] is True and event["grader_query_id"] == verdict["query_id"]
            assert event["grader_answered_at_utc"] == verdict["answered_at"]
            assert event["first_solved_at_utc"] >= saved["official_started_at_utc"]
        times = sorted(q["first_solved"]["first_solved_elapsed_s"] for q in solved)
        assert abs(times[17] - s["time_to_target_s"]) < 1e-8
        grader = s["grader_timeline"]
        assert abs(sum(grader[k] for k in ("first_pick_elapsed_s", "actual_service_s", "idle_between_queries_s")) - s["time_to_target_s"]) < .03
        ttft = [q["rollouts"][0]["ttft_s"] for q in questions if q["rollouts"][0]["ttft_s"] is not None]
        tokens = sum((q["usage"] or {}).get("completion_tokens", 0) for q in prewarm["rollouts"])
        row.update(time_to_target_s=s["time_to_target_s"],
                   prewarm_s=prewarm["total_latency_s"], prewarm_tokens=tokens,
                   initialization_and_attempt_s=s["initialization_and_attempt_latency_s"],
                   first_pick_s=grader["first_pick_elapsed_s"], grader_service_s=grader["actual_service_s"],
                   grader_idle_s=grader["idle_between_queries_s"],
                   grader_queries=grader["completed_queries"], wrong_queries=grader["wrong"],
                   generation_requests=s["performance"]["generation_requests"],
                   initial_ttft_median_s=statistics.median(ttft), solved_times=times,
                   seed_controls_valid=not trial["matched_requests"]["different_questions"])
        rows.append(row)
    return {"source_commit": config["source_commit"], "protocol_sha256": config["protocol_sha256"],
            "reference_time_s": protocol["reference_time_to_target_s"],
            "confirmed": summary["confirmed"], "rows": rows}


def render(batch, data):
    rows = data["rows"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.7), gridspec_kw={"width_ratios": [1.7, 1.1, 1.5]})
    y = range(len(rows))
    left = [0.0] * len(rows)
    for key, label, color in [("first_pick_s", "First pickup", "#8b98a7"),
                              ("grader_service_s", "Grader service", "#287caf"),
                              ("grader_idle_s", "Grader idle", "#e9ac42")]:
        values = [r.get(key, 0) for r in rows]
        a.barh(y, values, left=left, label=label, color=color, height=.6)
        left = [x + z for x, z in zip(left, values)]
    for i, row in enumerate(rows):
        if row["valid"]:
            a.text(row["time_to_target_s"] + 1.5, i, f"{row['time_to_target_s']:.2f}s", va="center", fontsize=9)
        else:
            a.text(2, i, "Failed / unmet", va="center")
    a.set_yticks(y, [f"Seed {r['seed']}" for r in rows]); a.invert_yaxis()
    a.axvline(data["reference_time_s"], linestyle="--", color="#344454", linewidth=1)
    a.set_xlim(0, max([r.get("time_to_target_s", 0) for r in rows] + [data["reference_time_s"]]) + 18)
    a.set_xlabel("Seconds to 18 verified correct")
    a.set_title("Timed AIME 2025 attempt", loc="left")
    a.legend(loc="upper left", bbox_to_anchor=(0, -.19), ncols=3, frameon=False, fontsize=8)
    b.barh(y, [r.get("prewarm_s", 0) for r in rows], color="#8756a5", height=.6)
    for i, row in enumerate(rows):
        if row.get("prewarm_s") is not None:
            b.text(row["prewarm_s"] + 1, i, f"{row['prewarm_s']:.1f}s", va="center", fontsize=9)
    b.set_yticks(y, []); b.invert_yaxis()
    b.set_xlim(0, max([r.get("prewarm_s", 0) for r in rows] + [1]) * 1.2)
    b.set_title("Ungraded AIME 2024 warmup", loc="left")
    b.set_xlabel("Seconds; outside official timer")
    for row in rows:
        if row["valid"]:
            c.step([0] + row["solved_times"], [0] + list(range(1, 19)), where="post", label=str(row["seed"])[-2:])
    c.axhline(18, color="#344454", linestyle="--", linewidth=1)
    c.axvline(data["reference_time_s"], color="#344454", linestyle="--", linewidth=1)
    c.set_ylim(0, 19); c.set_yticks([0, 6, 12, 18])
    c.set_xlabel("Official elapsed seconds"); c.set_ylabel("Distinct verified correct")
    c.set_title("First-solved progression", loc="left")
    c.legend(title="Seed suffix", frameon=False, fontsize=8)
    for ax in [a, b, c]: ax.grid(axis="x", alpha=.15)
    fig.suptitle("Final NVFP4 FlashInfer 30×1 runner — five predeclared seeds", x=.01, ha="left", weight="bold", fontsize=14)
    fig.text(.01, .01, "Every trial retained. No unscored settling run. Dashed reference: 71.135s. Warmup cost and candidate timing are separate; no paired warmup-benefit claim.", color="#536271", fontsize=8)
    fig.tight_layout(rect=[0, .06, 1, .94])
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(batch / f"five-seed-timing.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(batch, data):
    rows = data["rows"]
    valid = [r for r in rows if r["valid"]]
    times = [r["time_to_target_s"] for r in valid]
    passed = sum(t <= data["reference_time_s"] for t in times)
    text = "# Final runner: five predeclared seeds\n\n"
    if times:
        text += f"**{len(valid)}/5 reached 18 distinct verified correct questions.** Median time: **{statistics.median(times):.3f}s**; range **{min(times):.3f}–{max(times):.3f}s**. **{passed}/5** met the predeclared 71.135-second threshold. Consistent sub-reference speed is **{'confirmed for these five declared seeds' if data['confirmed'] else 'not confirmed'}**.\n\n"
    else:
        text += "None of the five trials produced a valid completed target. All failures are retained.\n\n"
    text += "![Five-seed timing](five-seed-timing.png)\n\n| Seed | First 18 | Warmup | Init + attempt | Initial TTFT | Grader queries (wrong) | Grader idle | Generation requests |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for row in rows:
        if row["valid"]:
            text += f"| {row['seed']} | {row['time_to_target_s']:.3f}s | {row['prewarm_s']:.3f}s | {row['initialization_and_attempt_s']:.3f}s | {row['initial_ttft_median_s']*1000:.1f}ms | {row['grader_queries']} ({row['wrong_queries']}) | {row['grader_idle_s']:.3f}s | {row['generation_requests']} |\n"
        else:
            text += f"| {row['seed']} | Failed / unmet | — | — | — | — | — | — |\n"
    text += "\nThe selected policy is frozen in [runner_final](../../../runner_final/README.md): NVFP4 Marlin weights, BF16 activation/KV, FlashInfer attention, 95% GPU allocation, 64K total context, 30×1 barrier coverage, first-pass 8K then up to 16K additional output per continuation, temperature 0.8, top-p 0.95, cap four requests per question including continuations. Benchmark mode buffers required traces and disables optional CPU/GPU/engine profiling.\n\n"
    text += f"Source commit: `{data['source_commit']}`. The [protocol](config.json) declared every seed before execution. One private inference server is reused; each trial has a fresh serial 3-second grader, successful prefix-cache clearing, all 30 ungraded AIME 2024 generations before grader launch, and the standard 30×32-token warmup. No warmup answers were checked. Warmup uses an 8K output cap, the same sampling controls, and no early candidate cancellation. Warmup and startup are excluded from the official 2025 solving timer; initialization plus attempt is retained to expose their cost. Server startup before the first runner invocation is an additional batch cost recorded in batch timestamps.\n\n"
    text += "All valid initial payloads match the original reference after the declared model and seed substitutions. Each has 30 question records, at most four generation requests per question, 18 distinct first-solved events linked to true grader verdicts and timestamps, and an 18th verdict time equal to the reported target time. The stacked grader timeline sums within 0.03s of that time. Exact token and verdict evidence is retained; full SSE and service/grader logs remain remote. Optional GPU peaks and complete eviction counters are unavailable in benchmark mode.\n\n"
    if valid:
        text += f"The workload warmup cost a median **{statistics.median(r['prewarm_s'] for r in valid):.3f}s per trial**, generating a median **{statistics.median(r['prewarm_tokens'] for r in valid):,.0f} tokens**. It did not require grader service. This workload can exercise longer prompts/decodes, but vLLM already performs graph/kernel initialization warmup. These five different-seed warmed trials are not a paired warmed/unwarmed experiment and cannot establish the warmup's causal benefit. Historical same-seed independent FlashInfer times were 87.356, 59.316 and 86.574s; the 59.316s fastest draw was not consistently replicated.\n\n"
    text += "AIME 2025 remains the development benchmark used for tuning; this is a repeatability check, not a held-out generalization result. All five outcomes, including the first, are scored and retained.\n\n[All results](summary.json), [validated plot data](analysis.json), [PNG](five-seed-timing.png), [SVG](five-seed-timing.svg), [PDF](five-seed-timing.pdf).\n"
    (batch / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", help="Plain directory name under runs/experiments")
    args = parser.parse_args()
    if Path(args.batch).name != args.batch or args.batch in (".", ".."):
        parser.error("Use a plain batch directory name")
    batch = ROOT / "runs/experiments" / args.batch
    data = analyze(batch)
    (batch / "analysis.json").write_text(json.dumps(data, indent=2) + "\n")
    render(batch, data)
    report(batch, data)
    print(batch)


if __name__ == "__main__":
    main()
