"""Render compact paired v1 speculative-decoding evidence after importing a batch."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from speculative_v1_ab import decode_analysis


def report(batch):
    raw = json.loads((batch/"summary.json").read_text())
    rows = raw["cells"]
    by_seed = {}
    windows, blocks, requests = [], [], []
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
        folder = batch/f"{row['seed']}-{row['arm']}"
        analysis_path = folder/"analysis.json"
        if not analysis_path.exists():
            continue
        attempt = batch.parents[2]/"attempts"/row["attempt_id"]
        # Post-hoc block analysis also measures the long-context part of each stream.
        analysis = decode_analysis(attempt)
        (folder/"posthoc_metrics.json").write_text(json.dumps(analysis, indent=2)+"\n")
        windows.extend({"seed": row["seed"], "arm": row["arm"], **w} for w in analysis["windows"])
        blocks.extend({"seed": row["seed"], "arm": row["arm"], **b} for b in analysis["request_blocks"])
        requests.extend({"seed": row["seed"], "arm": row["arm"], **r} for r in analysis["requests"])
        # Public engine counter changes between samples inside official timing.
        summary = json.loads((attempt/"summary.json").read_text())
        start = datetime.fromisoformat(summary["official_started_at_utc"])
        end = datetime.fromisoformat(summary["official_finished_at_utc"])
        metrics = folder/"metrics.jsonl"
        if metrics.exists():
            samples = [json.loads(line) for line in metrics.read_text().splitlines()]
            samples = [s for s in samples if "metrics" in s and start <= datetime.fromisoformat(s["utc"]) <= end]
            (folder/"engine_metrics.json").write_text(json.dumps({"scope": "samples inside official interval",
                "official_started_at_utc": start.isoformat(), "official_finished_at_utc": end.isoformat(),
                "samples": samples}, indent=2)+"\n")
            if len(samples) >= 2:
                first, last = samples[0]["metrics"], samples[-1]["metrics"]
                delta = {k: v-first.get(k, 0) for k, v in last.items() if k.endswith("_total")}
                row["sampled_official_counter_deltas"] = delta
                drafted = delta.get("spec_decode_num_draft_tokens_total")
                accepted = delta.get("spec_decode_num_accepted_tokens_total")
                row["sampled_official_acceptance_fraction"] = accepted/drafted if drafted and accepted is not None else None
                row["sampled_official_seconds"] = (datetime.fromisoformat(samples[-1]["utc"])-datetime.fromisoformat(samples[0]["utc"])).total_seconds()
    paired = []
    control_audits = []
    for seed, cells in by_seed.items():
        if "A" in cells and "B" in cells:
            a, b = cells["A"], cells["B"]
            configs = [json.loads((batch.parents[2]/"attempts"/c["attempt_id"]/"config.json").read_text())
                       for c in (a, b) if c.get("attempt_id")]
            if len(configs) == 2:
                keys = ["core_manifest_sha256", "system_prompt_sha256", "runner_id", "seed",
                        "benchmark_year", "question_indices", "target_correct", "grader_cost",
                        "parallelism", "rollouts", "schedule", "first_pass_max_tokens", "max_tokens",
                        "max_attempts_per_question", "max_rounds", "temperature", "top_p",
                        "benchmark", "buffer_traces", "skip_benchmark_prewarm", "no_continuation"]
                mismatches = [k for k in keys if configs[0].get(k) != configs[1].get(k)]
                profiles = [dict(c["launch_profile"]) for c in configs]
                spec = profiles[1].pop("speculative-config", None)
                if mismatches or profiles[0] != profiles[1] or not spec:
                    raise RuntimeError(f"Unmatched controls for {seed}: {mismatches}")
                control_audits.append({"seed": seed, "matched_controls": keys,
                    "core_manifest_sha256": configs[0]["core_manifest_sha256"],
                    "serving_difference": {"speculative-config": spec}})
            valid = all(c.get("target_reached") and c.get("status") == "completed" for c in (a, b))
            paired.append({"seed": seed, "valid": valid, "A_s": a.get("time_to_target_s"),
                           "B_s": b.get("time_to_target_s"),
                           "B_acceptance_fraction": b.get("sampled_official_acceptance_fraction"),
                           "B_minus_A_s": b["time_to_target_s"]-a["time_to_target_s"] if valid else None,
                           "A_over_B": a["time_to_target_s"]/b["time_to_target_s"] if valid else None})
    valid_seeds = {p["seed"] for p in paired if p["valid"]}
    scored_rows = [r for r in rows if r["seed"] in valid_seeds]
    windows = [w for w in windows if w["seed"] in valid_seeds]
    blocks = [b for b in blocks if b["seed"] in valid_seeds]
    requests = [r for r in requests if r["seed"] in valid_seeds]
    bins = {}
    for label in ("0-4", "4-8", "8-16", "16-30"):
        bins[label] = {}
        for arm in ("A", "B"):
            cells = [r["concurrency_bins"][label] for r in scored_rows if r["arm"] == arm and "concurrency_bins" in r]
            exposure = sum(c["decode_request_seconds"] for c in cells)
            tokens = sum(c["tokens"] for c in cells)
            bins[label][arm] = {"windows": sum(c["windows"] for c in cells),
                                "request_seconds": exposure, "tokens": tokens,
                                "tps": tokens/exposure if exposure else None,
                                "context_tokens": sum((c["mean_context_tokens"] or 0)*c["decode_request_seconds"]
                                                      for c in cells)/exposure if exposure else None}
        a, b = bins[label]["A"]["tps"], bins[label]["B"]["tps"]
        bins[label]["B_over_A"] = b/a if a and b else None
    context_bins = {}
    for label, lower, upper in (("0-8K", 0, 8192), ("8-16K", 8192, 16384),
                               ("16-32K", 16384, 32768), ("32K+", 32768, float("inf"))):
        context_bins[label] = {}
        for arm in ("A", "B"):
            cells = [b for b in blocks if b["arm"] == arm and b["seconds"] >= 4.5
                     and lower < b["context_token_seconds"]/b["seconds"] <= upper]
            exposure = sum(b["seconds"] for b in cells)
            tokens = sum(b["tokens"] for b in cells)
            context_bins[label][arm] = {"blocks": len(cells), "request_seconds": exposure,
                                       "tokens": tokens, "tps": tokens/exposure if exposure else None}
        a, b = context_bins[label]["A"]["tps"], context_bins[label]["B"]["tps"]
        context_bins[label]["B_over_A"] = b/a if a and b else None
    request_rates = {}
    for arm in ("A", "B"):
        values = [r["decode_tps"] for r in requests if r["arm"] == arm and r.get("decode_tps") is not None
                  and r["last"]-r["first"] >= 5]
        request_rates[arm] = {"count": len(values), "median_tps": statistics.median(values) if values else None,
                             "p10_tps": statistics.quantiles(values, n=10, method="inclusive")[0]
                             if len(values) >= 2 else None,
                             "scope": "request mean observed decode rate, at least 5s first-to-last arrival; includes policy-censored streams"}
    result = {"cells": rows, "paired": paired, "pooled_concurrency_bins": bins,
              "pooled_context_bins": context_bins,
              "request_rate_distribution": request_rates,
              "control_audits": control_audits,
              "analysis_source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                         for name in ("report_speculative_v1_ab.py", "speculative_v1_ab.py")},
              "median_time_to_target_s": {arm: statistics.median([r["time_to_target_s"] for r in scored_rows
                  if r["arm"] == arm and r.get("target_reached") and r["status"] == "completed"])
                  if any(r["arm"] == arm and r.get("target_reached") and r["status"] == "completed" for r in scored_rows)
                  else None for arm in ("A", "B")}}
    (batch/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    if windows:
        with (batch/"decode_windows.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(windows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(windows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    colors = {"A": "#2563a5", "B": "#d46b25"}
    labels = list(bins)
    for arm, offset in (("A", -.18), ("B", .18)):
        axes[0].bar([i+offset for i in range(len(labels))], [bins[l][arm]["tps"] or float("nan") for l in labels],
                    width=.35, color=colors[arm], label="Baseline" if arm == "A" else "Draft 0.5B, k=3")
        for i, label in enumerate(labels):
            if bins[label][arm]["tps"] is None:
                axes[0].text(i+offset, 3, "n/a", ha="center", fontsize=8, rotation=90)
    axes[0].set(xticks=range(len(labels)), xticklabels=labels, xlabel="Mean active decoding streams (5s windows)",
                ylabel="Observed tok/s per active stream", title="v1 decode speed by concurrency")
    axes[0].legend(loc="upper left", frameon=False)
    context_labels = list(context_bins)
    for arm, offset in (("A", -.18), ("B", .18)):
        axes[1].bar([i+offset for i in range(len(context_labels))],
                    [context_bins[l][arm]["tps"] or float("nan") for l in context_labels],
                    width=.35, color=colors[arm])
        for i, label in enumerate(context_labels):
            if context_bins[label][arm]["tps"] is None:
                axes[1].text(i+offset, 3, "n/a", ha="center", fontsize=8, rotation=90)
    axes[1].set(xticks=range(len(context_labels)), xticklabels=context_labels,
                xlabel="Mean logical context per request block", ylabel="Observed tok/s per stream",
                title="Longer-context decoding")
    for i, p in enumerate(paired):
        if p["valid"]:
            axes[2].plot([0, 1], [p["A_s"], p["B_s"]], marker="o", label=str(p["seed"]))
    axes[2].set(xticks=[0, 1], xticklabels=["Baseline", "Speculative"], ylabel="Seconds to 18 verified answers",
                title="Paired seeds; identical v1 policy")
    if any(p["valid"] for p in paired):
        axes[2].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.savefig(batch/"comparison.png", dpi=180)
    fig.savefig(batch/"comparison.svg")
    svg = batch/"comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)
    md = "# V1 speculative decoding A/B\n\n"
    medians = result["median_time_to_target_s"]
    if all(medians.values()):
        md += f"Median time to 18: **{medians['A']:.2f}s baseline, {medians['B']:.2f}s speculative** "
        md += f"({medians['B']/medians['A']:.2f}× baseline latency). "
        md += f"{sum(p['valid'] for p in paired)} completed pairs; see observed decode rates below.\n\n"
    md += "VibeThinker-3B derives from [Qwen2.5-Coder-3B](https://github.com/WeiboAI/VibeThinker). "
    md += "The draft is [Qwen2.5-Coder-0.5B (0.49B parameters)](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B), "
    md += "revision `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`, in BF16, with three greedy draft tokens per verification.\n\n"
    md += "The target retains NVFP4/Marlin, FLASHINFER, BF16 KV, 95% memory, prefix caching and 64K context. "
    md += "Both arms run unchanged canonical v1: prompt adherence, AIME 2025, 30×1 barrier, 8K initial/16K subsequent requests, "
    md += "four-request cap, temperature 0.8/top-p 0.95, 3s serial grader and target18. "
    md += "Each cell uses a fresh server and the same canonical warmup; order is AB, BA, AB across three paired seeds. "
    md += "The runner stays in benchmark mode; public engine counters are polled externally once per second.\n\n"
    md += "![Comparison](comparison.png)\n\n| Seed | Baseline s | Speculative s | Difference s | Baseline/speculative | Draft acceptance |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    def fmt(v, digits=2):
        return f"{v:.{digits}f}" if v is not None else "unavailable"
    for p in paired:
        acceptance = 100*p["B_acceptance_fraction"] if p["B_acceptance_fraction"] is not None else None
        md += f"| {p['seed']} | {fmt(p['A_s'])} | {fmt(p['B_s'])} | {fmt(p['B_minus_A_s'])} | {fmt(p['A_over_B'])}× | {fmt(acceptance, 1)}% |\n"
    md += "\n| Active stream bin | Baseline tok/s | Speculative tok/s | Spec/baseline | Windows A/B | Mean context A/B |\n| --- | ---: | ---: | ---: | --- | --- |\n"
    for label, cell in bins.items():
        a, b = cell["A"], cell["B"]
        md += f"| {label} | {fmt(a['tps'], 1)} | {fmt(b['tps'], 1)} | {fmt(cell['B_over_A'])}× | {a['windows']}/{b['windows']} | {fmt(a['context_tokens'], 0)}/{fmt(b['context_tokens'], 0)} |\n"
    md += "\n| Mean logical context | Baseline tok/s | Speculative tok/s | Spec/baseline | Request blocks A/B |\n| --- | ---: | ---: | ---: | --- |\n"
    for label, cell in context_bins.items():
        a, b = cell["A"], cell["B"]
        md += f"| {label} | {fmt(a['tps'], 1)} | {fmt(b['tps'], 1)} | {fmt(cell['B_over_A'])}× | {a['blocks']}/{b['blocks']} |\n"
    md += "\n| Request decode-rate distribution | Baseline | Speculative |\n| --- | ---: | ---: |\n"
    for key, label in (("count", "Requests observed for at least 5s"), ("median_tps", "Median tok/s"), ("p10_tps", "P10 tok/s (slow tail)")):
        md += f"| {label} | {fmt(request_rates['A'][key], 1)} | {fmt(request_rates['B'][key], 1)} |\n"
    md += "\nDecode rates count exact token IDs in saved SSE chunks, including multi-token chunks. "
    md += "The first chunk of each request is excluded from rate counting; TTFT and cancellation cleanup are outside decode exposure. "
    md += "Rates pool tokens over summed first-to-last decoding request-seconds in complete five-second windows. "
    md += "Context bins classify each request's block by its time-weighted mean logical context (prompt plus generated IDs), retaining blocks at least 4.5s long. "
    md += "Request-rate quantiles use each request's mean observed rate with at least 5s decode exposure, including policy-censored streams; they are descriptive, without matching trajectories or contexts. "
    md += "Low-concurrency bins describe the observed tail; changing contexts and question mixes prevent a fixed-context causal interpretation. "
    md += "Seeds pair workloads but speculative sampling can change trajectories. Three pairs are exploratory, without a significance claim. "
    md += "No speed is inferred for bins with no exposure. The draft's native context is 32K while the target is 64K; inspect saved serving logs for the installed runtime's limit handling.\n\n"
    md += "Every draft vocabulary entry has the same token ID in the target; the target adds `<think>` and `</think>`. "
    md += "Both LM heads have 151,936 entries. Preflight records tokenizer hashes and vLLM version. "
    md += "Counter deltas labelled `including_warmup` include warmup; sampled official deltas use only snapshots inside the solve interval.\n\n"
    md += "Reproduce on idle Callosum with `~/.venvs/vllm/bin/python scripts/speculative_v1_ab.py --execute`. "
    md += "Render this report locally with `.venv/bin/python scripts/report_speculative_v1_ab.py BATCH_DIRECTORY`. "
    md += "[Analysis](analysis.json) and [window measurements](decode_windows.csv) are adjacent; full streams, weights and service logs stay outside Git.\n"
    if (batch/"validation.md").exists():
        md += "\n"+(batch/"validation.md").read_text()
    (batch/"README.md").write_text(md)
    print(json.dumps({"paired": paired, "bins": bins, "medians": result["median_time_to_target_s"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    report(parser.parse_args().batch)
