"""Render grader timing and initial TTFT from saved best-run repetitions."""

import json
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aime-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/experiments/best-replication-20261003"
BATCHES = [
    ("BF16 benchmark", "bf16-best-warm-replicate-20261003T224038Z"),
    ("BF16 profiling", "bf16-best-profiled-replicate-20261003T224630Z"),
    ("NVFP4 benchmark", "nvfp4-best-warm-replicate-20261003T224630Z"),
    ("BF16 recycle 30", "bf16-dynamic30-replicate-20261003T225437Z"),
    ("BF16 exact source", "bf16-best-exact-source-20261003T230424Z"),
    ("NVFP4 varied seed", "nvfp4-30x1-seed-comparison-20261003T231357Z"),
]


def initial_ttft(folder):
    values = []
    for f in (folder / "trace").glob("*/question.json"):
        r = json.loads(f.read_text())["rollouts"][0]
        if r["ttft_s"] is not None:
            values.append(r["ttft_s"])
    return statistics.median(values) if values else None


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = "20261003T211557.382358Z"
    entries = [("Original best", reference, None)]
    groups = []
    for label, batch in BATCHES:
        p = ROOT / "runs/experiments" / batch / "summary.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        groups.append({"label": label, "batch": batch, "status": data["status"]})
        entries += [(f"{label} #{r['trial']}", r["attempt_id"], batch) for r in data["trials"]]
    rows = []
    for label, attempt, batch in entries:
        folder = ROOT / "attempts" / attempt
        s = json.loads((folder / "summary.json").read_text())
        if not s["target_reached"]:
            continue
        g = s["grader_timeline"]
        assert abs(s["time_to_target_s"] - sum(g[k] for k in ["first_pick_elapsed_s", "actual_service_s", "idle_between_queries_s"])) < .03
        rows.append({"label": label, "attempt_id": attempt, "batch": batch,
                     "time_to_target_s": s["time_to_target_s"], "initial_ttft_median_s": initial_ttft(folder),
                     "first_pick_s": g["first_pick_elapsed_s"], "grader_service_s": g["actual_service_s"],
                     "grader_idle_s": g["idle_between_queries_s"], "wrong_checks": g["wrong"]})
    (OUTPUT / "analysis.json").write_text(json.dumps({"reference": reference, "groups": groups, "trials": rows}, indent=2) + "\n")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.6, max(7.5, len(rows) * .45 + 1.2)), gridspec_kw={"width_ratios": [3.5, 1]}, sharey=True)
    colors = ["#8b98a7", "#287caf", "#e9ac42"]
    left = [0] * len(rows)
    for key, label, color in zip(["first_pick_s", "grader_service_s", "grader_idle_s"],
                                 ["First grader pickup", "Grader service", "Idle between checks"], colors):
        values = [r[key] for r in rows]
        a.barh(range(len(rows)), values, left=left, color=color, height=.66, label=label)
        left = [x + y for x, y in zip(left, values)]
    for i, r in enumerate(rows):
        a.text(r["time_to_target_s"] + 1.5, i, f"{r['time_to_target_s']:.1f}s", va="center", fontsize=8)
        b.scatter(r["initial_ttft_median_s"] * 1000, i,
                  color="#8756a5" if "NVFP4" in r["label"] else "#287caf", s=32)
    a.axvline(rows[0]["time_to_target_s"], color="#304050", linestyle="--", linewidth=1.2)
    a.set_yticks(range(len(rows)), [r["label"] for r in rows])
    a.invert_yaxis()
    a.set_xlim(0, max(r["time_to_target_s"] for r in rows) + 13)
    a.set_xlabel("Seconds to 18 verified correct answers")
    a.set_title("The extra time is mostly waiting for correct candidates", loc="left", fontsize=11)
    a.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncols=3, frameon=False, fontsize=8)
    a.grid(axis="x", alpha=.18)
    b.set_xlim(70, 300)
    b.set_xticks([100, 200, 300])
    b.set_xlabel("Median initial TTFT (ms)")
    b.set_title("First 30 requests", fontsize=10)
    b.grid(axis="x", alpha=.18)
    fig.suptitle("Best-run replication: original and completed comparisons", x=.02, ha="left", fontsize=15, weight="bold")
    fig.text(.02, .015, "Official solving window only; startup excluded. Dashed line: original 71.135s. NVFP4 points are purple. Paired seeds do not guarantee identical output paths.", fontsize=8, color="#536271")
    fig.tight_layout(rect=[0, .045, 1, .95])
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(OUTPUT / f"timing-comparison.{suffix}", dpi=180, bbox_inches="tight")
    svg = OUTPUT / "timing-comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
