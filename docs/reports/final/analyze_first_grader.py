"""Retrospective run-level first-grader analysis; launches no inference.

Run with aime-bench/.venv/bin/python analyze_first_grader.py.
Inputs are frozen, explicitly listed experimental batches; additional runs are
not silently admitted into the comparison.
"""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import itertools
import math
import numpy as np
import scipy
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "aime-bench"
OUT = ROOT / "analysis/first-grader"
BATCHES = {
    "bf16_flash_attn": "bf16-best-warm-replicate-20261003T224038Z",
    "nvfp4_flash_attn": "nvfp4-best-warm-replicate-20261003T224630Z",
    "nvfp4_flashinfer_development": "nvfp4-flashinfer-best-replicate-20261003T233008Z",
    "nvfp4_flashinfer_validation": "nvfp4-flashinfer-validation-20261003T233628Z",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elapsed(utc, start):
    return (datetime.fromisoformat(utc.replace("Z", "+00:00")) - start).total_seconds()


def extract_rows():
    runs, question_rows, sources, stream_rows = [], [], [], []
    for group, batch in BATCHES.items():
        batch_summary = REPO / "runs/experiments" / batch / "summary.json"
        batch_config = batch_summary.with_name("config.json")
        config = json.loads(batch_config.read_text())
        sources.extend({"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                       for p in (batch_summary, batch_config))
        for trial in json.loads(batch_summary.read_text())["trials"]:
            attempt = trial["attempt_id"]
            folder = REPO / "attempts" / attempt
            summary_path = folder / "summary.json"
            s = json.loads(summary_path.read_text())
            start = datetime.fromisoformat(s["official_started_at_utc"])
            events = []
            files = sorted(folder.glob("trace/*/verification.jsonl"))
            assert files, f"Missing verification records: {attempt}"
            for path in files:
                for line in path.read_text().splitlines():
                    e = json.loads(line)
                    if e.get("verification_started_at_utc"):
                        events.append((elapsed(e["verification_started_at_utc"], start),
                                       int(path.parent.name), e, path))
            assert events, f"No observed submissions: {attempt}"
            first, question, event, event_path = min(events, key=lambda x: x[0])
            role = "settling" if (group == "nvfp4_flashinfer_validation" and trial["trial"] == 1) else "scored"
            timeline = s["grader_timeline"]
            pick = timeline["first_pick_elapsed_s"]
            assert -0.002 <= pick - first < 0.050
            assert s["status"] == "completed" and s["target_reached"]
            assert trial["cache_reset"]["success"]
            initial_rates, initial_ttfts = [], []
            for q in range(1, 31):
                telemetry_path = folder / f"trace/{q:02d}/rollout-01/telemetry.json"
                token_path = telemetry_path.with_name("tokens.json")
                telemetry = json.loads(telemetry_path.read_text())
                tokens = json.loads(token_path.read_text())
                count = len(tokens["output_token_ids"])
                assert count == telemetry["generated_token_ids_count"]
                assert tokens["complete"] and telemetry["round"] == 1
                duration = telemetry["last_token_s"] - telemetry["ttft_s"]
                assert duration > 0 and count > 0
                rate = count / duration
                initial_rates.append(rate)
                initial_ttfts.append(telemetry["ttft_s"])
                stream_rows.append({"group": group, "attempt_id": attempt, "role": role,
                                    "question": q, "output_token_ids": count,
                                    "ttft_s": telemetry["ttft_s"], "last_token_s": telemetry["last_token_s"],
                                    "approx_observed_decode_tok_s": rate,
                                    "generation_censored": telemetry["generation_censored"],
                                    "telemetry_evidence": str(telemetry_path.relative_to(ROOT))})
                sources.extend({"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                               for p in (telemetry_path, token_path))
            assert np.isclose(np.median(initial_ttfts), trial["matched_requests"]["initial_ttft_median_s"])
            row = {
                "group": group, "batch": batch, "trial": trial["trial"],
                "role": role, "attempt_id": attempt,
                "official_started_at_utc": s["official_started_at_utc"],
                "first_client_submit_s": first,
                "first_grader_pick_s": pick,
                "first_submit_question": question,
                "first_submit_correct": event.get("result", {}).get("verdict"),
                "first_submit_timestamp": event["verification_started_at_utc"],
                "first_submit_evidence": str(event_path.relative_to(ROOT)),
                "grader_service_s": timeline["actual_service_s"],
                "grader_idle_s": timeline["idle_between_queries_s"],
                "wrong_checks": timeline["wrong"],
                "time_to_18_s": s["time_to_target_s"],
                "initial_ttft_median_s": trial["matched_requests"]["initial_ttft_median_s"],
                "initial_decode_rate_median_tok_s": float(np.median(initial_rates)),
                "question_1_decode_rate_tok_s": initial_rates[0],
                "source_commit": config["source_commit"],
            }
            runs.append(row)
            for q in range(1, 31):
                candidates = [e for e in events if e[1] == q]
                question_rows.append({"group": group, "attempt_id": attempt, "question": q,
                                      "first_client_submit_s": min(e[0] for e in candidates) if candidates else None,
                                      "scope": "Observed before stop-at-18; missing submissions are censored, not failures"})
            sources.append({"path": str(summary_path.relative_to(ROOT)), "sha256": digest(summary_path)})
            sources.extend({"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in files)
    return runs, question_rows, sources, stream_rows


def welch(treatment, control):
    x, y = np.asarray(treatment), np.asarray(control)
    result = stats.ttest_ind(x, y, equal_var=False, alternative="less")
    two_sided = stats.ttest_ind(x, y, equal_var=False)
    ci = two_sided.confidence_interval(confidence_level=0.95)
    one_ci = result.confidence_interval(confidence_level=0.95)
    # A sensitivity calculation only: label exchangeability is unverified because
    # historical model batches were not randomly interleaved.
    pooled = np.concatenate((x, y))
    observed = float(x.mean() - y.mean())
    differences = []
    for ix in itertools.combinations(range(len(pooled)), len(x)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(ix)] = True
        differences.append(float(pooled[mask].mean() - pooled[~mask].mean()))
    exact_p = sum(d <= observed + 1e-12 for d in differences) / len(differences)
    return {
        "n_treatment": len(x), "n_control": len(y),
        "mean_treatment_s": float(x.mean()), "mean_control_s": float(y.mean()),
        "mean_difference_treatment_minus_control_s": observed,
        "observed_reduction_percent": float((y.mean() - x.mean()) / y.mean() * 100),
        "t": float(result.statistic), "df": float(result.df),
        "p_one_sided_raw": float(result.pvalue),
        "difference_95pct_two_sided_ci_s": [float(ci.low), float(ci.high)],
        "difference_95pct_one_sided_upper_s": float(one_ci.high),
        "exact_label_permutation_p_sensitivity": exact_p,
        "permutation_assignments": len(differences),
        "significant_at_0_05_using_welch": bool(result.pvalue < 0.05),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, question_rows, sources, stream_rows = extract_rows()
    for filename, rows in [("runs.csv", runs), ("question-submissions.csv", question_rows),
                           ("initial-streams.csv", stream_rows)]:
        with (OUT / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    grouped = {key: [r for r in runs if r["group"] == key and r["role"] == "scored"] for key in BATCHES}
    descriptives = {}
    for key, rows in grouped.items():
        values = [r["first_client_submit_s"] for r in rows]
        descriptives[key] = {
            "n": len(rows), "values_s": values, "mean_s": float(np.mean(values)),
            "sample_sd_s": float(np.std(values, ddof=1)),
            "mean_time_to_18_s": float(np.mean([r["time_to_18_s"] for r in rows])),
            "mean_grader_idle_s": float(np.mean([r["grader_idle_s"] for r in rows])),
            "initial_ttft_run_medians_s": [r["initial_ttft_median_s"] for r in rows],
            "mean_initial_ttft_run_median_s": float(np.mean([r["initial_ttft_median_s"] for r in rows])),
            "initial_decode_rate_run_medians_tok_s": [r["initial_decode_rate_median_tok_s"] for r in rows],
            "mean_initial_decode_rate_run_median_tok_s": float(np.mean([r["initial_decode_rate_median_tok_s"] for r in rows])),
            "question_1_decode_rates_tok_s": [r["question_1_decode_rate_tok_s"] for r in rows],
        }
    values = lambda key: descriptives[key]["values_s"]
    comparisons = {
        "primary_quantization_with_flash_attn": welch(values("nvfp4_flash_attn"), values("bf16_flash_attn")),
        "secondary_flashinfer_validation_vs_bf16": welch(values("nvfp4_flashinfer_validation"), values("bf16_flash_attn")),
        "secondary_flashinfer_validation_vs_nvfp4_flash_attn": welch(values("nvfp4_flashinfer_validation"), values("nvfp4_flash_attn")),
        "secondary_flashinfer_development_vs_nvfp4_flash_attn": welch(values("nvfp4_flashinfer_development"), values("nvfp4_flash_attn")),
    }
    output = {
        "metric": "Earliest client verification_started_at_utc minus official_started_at_utc, per complete 30-question run",
        "timestamp_precision": "UTC fields are rounded to milliseconds; error is negligible relative to second-scale differences",
        "hypothesis": "H1: mean first submission latency is lower in NVFP4 than BF16; primary comparison holds FLASH_ATTN fixed",
        "method": "One-sided unequal-variance Welch t-test, raw alpha=0.05; retrospective exploratory analysis",
        "unit": "One full run, not one of its 30 concurrently generated questions",
        "paired_design": False,
        "decode_rate_definition": "Approximate received output token IDs / (last output delta arrival - first output delta arrival), median across 30 initial streams per run. Numerator includes first delta, so rate has a small boundary bias; streams can be cancelled or capped. Client delivery rate, not an isolated GPU kernel benchmark or fixed-concurrency throughput.",
        "ttft_definition": "Mean across runs of each run's median TTFT over 30 initial requests; includes queue, prefill and client transport. This is not the earliest token anywhere in the run.",
        "limitations": [
            "Only three scored runs per group; normality cannot be assessed reliably.",
            "Each batch reuses one owned warmed server and the same seed; independence and between-server variability are not established.",
            "Model batches were sequential rather than randomized/interleaved; source commits differ.",
            "Trial numbers do not define experimental blocks; a shared fixed seed does not justify a paired t-test.",
            "Endpoint/direction are requested retrospectively after timings were observed; these are not confirmatory tests.",
            "Secondary p-values are exploratory, unadjusted, and do not constitute multiple independent confirmatory findings.",
            "Question-level records share a run and are censored at the global target; do not treat them as 30 independent runs.",
            "First submission tests answer-arrival latency, not kernel throughput or isolated reasoning quality.",
        ],
        "settling_exclusion": "The validation trial designated settling before launch is preserved in runs.csv and shown separately; it is not in scored comparisons.",
        "descriptives": descriptives, "comparisons": comparisons,
        "software": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "sources": sources,
    }
    (OUT / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    labels = ["BF16\nFLASH_ATTN", "NVFP4\nFLASH_ATTN", "NVFP4 FlashInfer\ndevelopment", "NVFP4 FlashInfer\nscored validation"]
    colors = ["#245f73", "#c67a25", "#8a63ae", "#2d8f74"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, metric, title, ylabel in zip(axes,
            ["first_client_submit_s", "grader_idle_s", "time_to_18_s"],
            ["First answer sent to grader", "Idle gaps between grader jobs", "Time to 18 verified correct"],
            ["Seconds from official start", "Seconds", "Seconds from official start"]):
        for i, key in enumerate(BATCHES):
            data = [r[metric] for r in grouped[key]]
            ax.scatter(i + np.array([-0.09, 0, 0.09]), data, s=55, color=colors[i], zorder=3)
            ax.errorbar(i, np.mean(data), yerr=np.std(data, ddof=1), fmt="_", color="#182536",
                        markersize=18, capsize=5, zorder=4)
        settling = next(r for r in runs if r["role"] == "settling")
        ax.scatter(3.2, settling[metric], s=65, marker="D", facecolors="none", edgecolors=colors[3],
                   label="Predeclared settling run", zorder=4)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].legend(loc="upper right", fontsize=8)
    fig.suptitle("Recorded VibeThinker 30 x 1 / 8K batches: first answer versus later grader starvation", fontsize=14)
    fig.text(0.5, 0.015, "Dots: all three scored runs per group. Black bars: mean +/- sample SD. Open diamond: retained settling run.\nPrimary first-submission comparison: NVFP4 vs BF16 with FLASH_ATTN, one-sided Welch p = 0.0561 (exploratory).", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(OUT / "first-grader-comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"descriptives": descriptives, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
