"""Audit all five declared AWQ seeds and reproduce the paired NVFP4 report."""

import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from runner.extensions.validation import awq_v1_6 as driver
from runner.extensions.validation import awq_v1_6_batch as batch_driver

BATCH = Path(__file__).resolve().parent
load = driver.harness.baseline.load
spec = importlib.util.spec_from_file_location("pilot_report", ROOT / "runs/experiments" / batch_driver.PILOT / "reproduce_analysis.py")
pilot_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot_report)


def summary_stats(rows):
    times = [r["time_to_18_s"] for r in rows if r["target_reached"]]
    return {"reached": len(times), "trials": len(rows),
            "median_s": statistics.median(times) if times else None,
            "mean_s": statistics.mean(times) if times else None,
            "min_s": min(times) if times else None,
            "max_s": max(times) if times else None,
            "sample_sd_s": statistics.stdev(times) if len(times) > 1 else None}


def seconds(value, *, signed=False):
    if value is None:
        return "unmet"
    return f"{value:+.3f}s" if signed else f"{value:.3f}s"


def main():
    config, saved, evidence = (load(BATCH / n) for n in (
        "config.json", "summary.json", "server-evidence.json"
    ))
    if not all(t.get("marlin_and_flashinfer_logged") for t in evidence["trials"]):
        raise RuntimeError("Runtime kernel dispatch evidence is incomplete")
    if [t["sampling_seed"] for t in saved["trials"]] != batch_driver.SEEDS:
        raise RuntimeError("Declared paired seed sequence did not complete")
    pairs = []
    for trial in saved["trials"]:
        seed = trial["sampling_seed"]
        reference = driver.reference_trial(seed)
        folder = ROOT / "attempts" / trial["attempt_id"]
        audit = driver.audit(folder, reference, config["core_manifest_sha256"])
        if audit["valid"] != trial["valid"]:
            raise RuntimeError(f"Seed {seed} validity failed reproduction")
        awq = pilot_report.metrics("AWQ / Marlin", folder)
        nv = pilot_report.metrics("NVFP4 / Marlin", ROOT / "attempts" / reference["attempt_id"])
        pairs.append({"seed": seed, "audit": audit, "awq": awq, "nvfp4": nv,
                      "delta_s": awq["time_to_18_s"] - nv["time_to_18_s"] if awq["target_reached"] and audit["valid"] else None})
    awq_stats = summary_stats([p["awq"] for p in pairs])
    nv_stats = summary_stats([p["nvfp4"] for p in pairs])
    deltas = [p["delta_s"] for p in pairs if p["delta_s"] is not None]
    tail_counts = {}
    nv_tail_counts = {}
    for p in pairs:
        for win in p["awq"]["wins"][-3:]:
            index = str(win["question_index"])
            tail_counts[index] = tail_counts.get(index, 0) + 1
        for win in p["nvfp4"]["wins"][-3:]:
            index = str(win["question_index"])
            nv_tail_counts[index] = nv_tail_counts.get(index, 0) + 1
    result = {"source_commit": config["source_commit"], "pairs": pairs,
              "awq": awq_stats, "nvfp4": nv_stats,
              "paired_mean_delta_s": statistics.mean(deltas) if deltas else None,
              "paired_median_delta_s": statistics.median(deltas) if deltas else None,
              "awq_faster_seeds": sum(d < 0 for d in deltas),
              "awq_last_three_question_frequencies": tail_counts,
              "nvfp4_last_three_question_frequencies": nv_tail_counts,
              "server_lifetime_note": config["server_lifetime_note"]}
    (BATCH / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    table = ["| Seed | NVFP4 time to 18 | AWQ time to 18 | AWQ − NVFP4 | AWQ wrong checks | AWQ initial / continued / sibling wins |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for p in pairs:
        a, n = p["awq"], p["nvfp4"]
        wins = " / ".join(str(a["win_sources"][k]) for k in ("initial", "continuation", "fresh_sibling"))
        table.append(f"| {p['seed']} | {seconds(n['time_to_18_s'])} | {seconds(a['time_to_18_s'])} | {seconds(p['delta_s'], signed=True)} | {a['grader_timeline']['wrong']} | {wins} |")
    aggregates = ["| Summary | NVFP4 / Marlin | AWQ / Marlin |", "| --- | ---: | ---: |"]
    for title, key in (("Median", "median_s"), ("Mean", "mean_s"),
                       ("Minimum", "min_s"), ("Maximum", "max_s"), ("Sample standard deviation", "sample_sd_s")):
        aggregates.append(f"| {title} | {nv_stats[key]:.3f}s | {awq_stats[key]:.3f}s |")
    pressure_count = sum(len(t.get("pressure_warnings", [])) for t in evidence["trials"])
    max_kv = max(t["max_logged_kv_cache_usage_pct"] for t in evidence["trials"] if t["max_logged_kv_cache_usage_pct"] is not None)
    aggregate_metrics = []
    for name in ("nvfp4", "awq"):
        rows = [p[name] for p in pairs]
        aggregate_metrics.append({
            "initial_30_ttft_median_of_trial_medians_s": statistics.median(r["initial_30_ttft_median_s"] for r in rows),
            "grader_service_total_s": sum(r["grader_timeline"]["actual_service_s"] for r in rows),
            "grader_idle_total_s": sum(r["grader_timeline"]["idle_between_queries_s"] for r in rows),
            "wrong_checks_total": sum(r["grader_timeline"]["wrong"] for r in rows),
            "requests_total": sum(r["requests"] for r in rows),
        })
    result["aggregate_diagnostics"] = dict(zip(("nvfp4", "awq"), aggregate_metrics))
    (BATCH / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    tail_pattern = ", ".join(f"Q{q}: {count}/5" for q, count in sorted(tail_counts.items(), key=lambda x: (-x[1], int(x[0])))[:5])
    text = [
        "# AWQ versus NVFP4: five paired v1.6 AIME 2025 seeds", "",
        f"AWQ reached 18 in **{awq_stats['reached']}/5** trials; median **{awq_stats['median_s']:.3f}s**, "
        f"range **{awq_stats['min_s']:.3f}–{awq_stats['max_s']:.3f}s**. NVFP4 reached 18 in "
        f"5/5 with median **{nv_stats['median_s']:.3f}s** and range "
        f"**{nv_stats['min_s']:.3f}–{nv_stats['max_s']:.3f}s**. "
        f"AWQ was faster on {result['awq_faster_seeds']}/5 paired seeds; paired mean delta "
        f"**{result['paired_mean_delta_s']:+.3f}s**. All saved policy/payload/continuation/barrier/cap audits passed: "
        f"**{all(p['audit']['valid'] for p in pairs)}**.", "",
        *table, "", *aggregates, "",
        "## Controls and scope", "",
        "The completed seed 20261011 pilot is trial 1; only seeds 20261012–20261015 "
        "were subsequently launched. No seed was replaced or selected based on its result. "
        "The frozen v1.6 policy, 30-question initial 8K coverage/check barrier, subsequent "
        "30-slot pool, four-fresh-sample allowance, one exact-ID continuation to the "
        "remaining cumulative 64K/context ceiling, improved prompt, sampling seeds/temperature/top-p, "
        "serialized three-second grader, dataset provenance and benchmark configuration match. "
        "Each new trace is independently audited against its corresponding NVFP4 seed.", "",
        "AWQ uses AABoyles/VibeThinker-3B-AWQ (int4/group128, revision "
        "d32ba299f24c77d241832556d7cf5308699354b7) and explicit Marlin linear kernels. "
        "NVFP4 uses r0b0tlab/VibeThinker-3B-NVFP4 with ModelOpt FP4/Marlin. "
        "Both retain BF16 activations/KV, FlashInfer attention, 95% memory and 65,536 total "
        "context/output ceiling. Original AWQ tokenizer assets are retained; each trial's "
        "preflight compares vocabulary/special IDs, all exact reference prompt IDs and "
        "reference output decoding. Both quantized artifacts name the same WeiboAI base, "
        "but do not pin original base checkpoint/calibration data.", "",
        config["server_lifetime_note"], "",
        "These are paired downloaded-deployment results on one machine, not a controlled "
        "causal estimate of quantization alone. Five seeds provide descriptive repeatability "
        "evidence, not a guarantee on unseen questions. BF16 has only one matched-seed run "
        "(75.640s), so it is not presented as a five-run distribution.", "",
        "## Timing, memory and tail evidence", "",
        "Official solving time begins after dataset loading/tokenization and cheap "
        "30-stream/32-token arithmetic warmup. It ends at the eighteenth distinct "
        "first-solved grader verdict. Required traces buffer in RAM; final flush and "
        "service startup/cleanup are outside that clock. Optional CPU profiling, engine "
        "polling and NVML samples are disabled with --benchmark. A 600-second trial "
        "safety timeout includes initialization after server readiness.", "",
        f"The median of initial-30 TTFT trial medians was {aggregate_metrics[0]['initial_30_ttft_median_of_trial_medians_s']:.3f}s "
        f"for NVFP4 and {aggregate_metrics[1]['initial_30_ttft_median_of_trial_medians_s']:.3f}s for AWQ. "
        f"Completed wrong checks totaled {aggregate_metrics[0]['wrong_checks_total']} versus "
        f"{aggregate_metrics[1]['wrong_checks_total']}; grader inter-query idle totaled "
        f"{aggregate_metrics[0]['grader_idle_total_s']:.3f}s versus {aggregate_metrics[1]['grader_idle_total_s']:.3f}s. "
        f"The extra {aggregate_metrics[1]['grader_idle_total_s'] - aggregate_metrics[0]['grader_idle_total_s']:.3f}s of total "
        f"inter-query idle versus only {aggregate_metrics[1]['grader_service_total_s'] - aggregate_metrics[0]['grader_service_total_s']:.3f}s "
        "of extra completed grader service accounts for most of AWQ's slower group mean. "
        "The per-trial clock decomposition and last-three winners are in analysis.json. "
        "These distinguish time spent grading wrong candidates from time waiting for "
        "candidate arrivals; they do not identify the causal source of changed reasoning trajectories.", "",
        f"All five server logs confirm Marlin linear kernels and FlashInfer attention. "
        f"Maximum logged KV occupancy across the trials was {max_kv:.1f}%; "
        f"{pressure_count} memory-pressure warnings were recorded. "
        "Per-server runtime Marlin/FlashInfer selection, available KV capacity, coarse "
        "KV-use/throughput logs and memory-pressure warnings are in server-evidence.json. "
        "No sampled peak VRAM or complete eviction-counter claim is made. Existing throughput "
        "intervals have changing request counts and context lengths and are not fixed-batch benchmarks. "
        "Post-batch GPU processes and owned ports are checked after all four new servers close.", "",
        f"Expansion driver source `{config['source_commit']}`; unchanged v1.6 manifest "
        f"`{config['core_manifest_sha256']}`. Pilot retains its original source commit. "
        "All new metadata is checked individually; the legacy global metadata command "
        "still fails on an older v2.1 expression-valued field. Raw SSE streams, grader "
        "audits, service logs and weights stay remote; only reviewed evidence is imported.", "",
        "AWQ questions appearing most often among the last three of the 18 winners: " + tail_pattern + ". "
        "This describes late successful arrivals under the stopping rule, not the difficulty "
        "of questions that remained unsolved. Both models' frequencies and exact tail timings are in analysis.json.", "",
        "## Reproduce", "", "```sh",
        f".venv/bin/python runs/experiments/{BATCH.name}/reproduce_analysis.py", "```", "",
        "All 29 focused deployment, batch and policy tests passed locally and remotely "
        "before the new trials. The runner manifests and historical profiles were preserved.",
    ]
    (BATCH / "README.md").write_text("\n".join(text) + "\n")
    print(json.dumps({k: result[k] for k in ("awq", "nvfp4", "paired_mean_delta_s", "awq_faster_seeds")}, indent=2))


if __name__ == "__main__":
    main()
