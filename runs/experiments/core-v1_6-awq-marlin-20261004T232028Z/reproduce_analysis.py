"""Reproduce the matched AWQ, BF16 and NVFP4 single-seed comparison."""

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from runner.extensions.validation import awq_v1_6 as driver

BATCH = Path(__file__).resolve().parent
BF16_ATTEMPT = "20261004T231002.053899Z"
load = driver.harness.baseline.load


def metrics(label, folder):
    summary = load(folder / "summary.json")
    questions = [load(p) for p in folder.glob("trace/*/question.json")]
    winners = [q for q in questions if q.get("first_solved")]
    allocation = load(folder / "allocation.json")
    ttft = [load(p)["ttft_s"] for p in folder.glob("trace/*/rollout-01/telemetry.json")]
    timeline = summary["grader_timeline"]
    win_sources = dict(initial=0, continuation=0, fresh_sibling=0)
    for q in winners:
        rollout = next(r for r in q["rollouts"] if r["rollout"] == q["winner"]["rollout"])
        source = "continuation" if rollout["segment"] > 1 else "fresh_sibling" if rollout["fresh_sample"] > 1 else "initial"
        win_sources[source] += 1
    return {
        "label": label, "attempt_id": folder.name,
        "time_to_18_s": summary["time_to_target_s"],
        "status": summary["status"], "target_reached": summary["target_reached"],
        "solved": len(winners),
        "requests": sum(len(q["rollouts"]) for q in questions),
        "fresh_samples": sum(s["fresh"] for s in allocation["questions"].values()),
        "initial_30_ttft_median_s": statistics.median(x for x in ttft if x is not None),
        "grader_timeline": timeline,
        "barrier_release": allocation["barrier_release"],
        "win_sources": win_sources,
        "wins": [dict(question_index=q["problem_idx"], rollout=q["first_solved"]["rollout"], first_solved_elapsed_s=q["first_solved"]["first_solved_elapsed_s"], grader_query_id=q["first_solved"]["grader_query_id"])
                 for q in sorted(winners, key=lambda q: q["first_solved"]["first_solved_elapsed_s"])],
    }


def main():
    config, scored, evidence = (load(BATCH / n) for n in (
        "config.json", "summary.json", "server-evidence.json"
    ))
    reference = driver.reference_trial(scored["seed"])
    folder = ROOT / "attempts" / scored["attempt_id"]
    audit = driver.audit(folder, reference, config["core_manifest_sha256"])
    if audit["valid"] != scored["valid"]:
        raise RuntimeError("Recorded validity did not reproduce")
    rows = [metrics(label, ROOT / "attempts" / attempt) for label, attempt in (
        ("NVFP4 / Marlin", reference["attempt_id"]),
        ("BF16 / native linear", BF16_ATTEMPT),
        ("AWQ / Marlin", scored["attempt_id"]),
    )]
    actual = rows[-1]
    deltas = {r["label"]: actual["time_to_18_s"] - r["time_to_18_s"]
              for r in rows[:-1]} if actual["target_reached"] else {}
    result = {"seed": scored["seed"], "audit": audit, "rows": rows,
              "delta_s": deltas, "server_evidence": evidence,
              "source_commit": config["source_commit"]}
    (BATCH / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    outcome = (f"reached 18 verified correct in **{actual['time_to_18_s']:.3f}s**"
               if actual["target_reached"] else f"did not reach 18; status {actual['status']}")
    table = ["| Measure | NVFP4 / Marlin | BF16 / native linear | AWQ / Marlin |",
             "| --- | ---: | ---: | ---: |"]
    fields = (
        ("Time to 18 (s)", lambda r: r["time_to_18_s"]),
        ("Generation HTTP requests", lambda r: r["requests"]),
        ("Fresh samples", lambda r: r["fresh_samples"]),
        ("Wins: initial / continuation / fresh sibling", lambda r: " / ".join(str(r["win_sources"][k]) for k in ("initial", "continuation", "fresh_sibling"))),
        ("Initial 30-stream median TTFT (s)", lambda r: r["initial_30_ttft_median_s"]),
        ("First grader pickup (s)", lambda r: r["grader_timeline"]["first_pick_elapsed_s"]),
        ("Completed grader service (s)", lambda r: r["grader_timeline"]["actual_service_s"]),
        ("Grader idle between checks (s)", lambda r: r["grader_timeline"]["idle_between_queries_s"]),
        ("Completed wrong checks", lambda r: r["grader_timeline"]["wrong"]),
    )
    for title, get in fields:
        values = [get(r) for r in rows]
        table.append("| " + title + " | " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in values
        ) + " |")
    barrier = actual["barrier_release"]
    phase = (f"Coverage/check barrier released at {barrier['elapsed_s']:.3f}s; "
             "the continuously refilled 30-slot pool was exercised."
             if barrier else "The target was reached during initial coverage; the post-barrier pool was not exercised.")
    provenance = config["model_provenance"]
    token_check = provenance["tokenizer_semantic_check"]
    text = [
        "# AWQ / Marlin: one matched v1.6 AIME 2025 attempt", "",
        f"Seed {scored['seed']}: AABoyles/VibeThinker-3B-AWQ {outcome}. "
        f"Saved-request, timing, barrier and cap audits passed: **{audit['valid']}**.", "",
        *table, "", phase, "",
        "## Matched solving policy", "",
        "The unchanged v1.6 runner launches 30×1 initial requests at 8,192 output tokens. "
        "Its first coverage barrier includes the queued grader checks. Afterward, a 30-slot "
        "pool prioritizes exact-token continuations, then least-active unsolved questions "
        "with rotating ties. Each question permits at most four fresh trajectories. Each "
        "trajectory gets at most one continuation for its remaining cumulative 65,536-output-token "
        "budget, clipped to the 65,536 total context including prompt. Continuations do not "
        "consume fresh-sample allowance. Correct verdicts cancel that question; the eighteenth "
        "distinct first-solved verdict stops the attempt. Wrong answers do not inject feedback.", "",
        "All three deployments use seed 20261011, the same AIME 2025 prompt/dataset, improved "
        "system prompt, temperature 0.8/top-p 0.95, cheap 30-stream arithmetic warmup, fresh "
        "owned services, prefix-cache reset, serialized three-second grader, 95% memory, "
        "BF16 activations/KV and FlashInfer attention. AWQ uses four-bit/group-128 asymmetric "
        "weights and explicit Marlin linear kernels. NVFP4 uses ModelOpt FP4/Marlin; BF16 "
        "uses unquantized native linear dispatch. Server ceilings override model-card "
        "generation defaults to 65,536.", "",
        "Official timing begins after service startup, dataset loading, tokenization and warmup, "
        "immediately before scheduling. --benchmark disables optional CPU profiling, engine "
        "polls and NVML sampling. Required request/token/timing/verdict evidence is held in "
        "RAM and flushed after official timing; flush/startup are excluded. Existing server "
        "logs provide coarse cache/throughput observations without additional timed polling.", "",
        "## Provenance and limits", "",
        f"Source `{config['source_commit']}`; frozen v1.6 manifest `{config['core_manifest_sha256']}`. "
        "All 30 initial payloads match the NVFP4 control after removing model identity. "
        "The original AWQ tokenizer files were retained: their file hashes differ due to "
        "reserialization, but vocabulary, special-token IDs, exact rendered IDs for all "
        f"{len(token_check['questions'])} reference prompts and decoded reference outputs match. "
        "The preflight explicitly requests plain token IDs for compatibility with the installed "
        "Transformers version. No GPU/scored attempt started during the earlier return-type preflight failure.", "",
        "Model revisions and asset hashes are in config.json. Both quantized model cards name "
        "WeiboAI/VibeThinker-3B as base but do not pin their original base checkpoint/calibration "
        "data. This is a comparison of downloaded deployments, not an isolated quantization "
        "experiment. One seed does not establish repeatability. Correctness derives only "
        "from Boolean grader verdicts; existing startup answer-field validation is retained.", "",
        f"Maximum logged KV usage was {evidence['max_logged_kv_cache_usage_pct']}%; "
        f"{len(evidence['pressure_warnings'])} memory-pressure warnings were found. "
        "These are coarse log observations, not a sampled peak VRAM measurement or a complete "
        "eviction-counter audit. Runtime kernel/cache-capacity lines, throughput observations "
        "and post-run service/GPU cleanup are saved in server-evidence.json. Throughput "
        "samples average preceding intervals with changing concurrency and context lengths.", "",
        "## Reproduce", "", "```sh",
        f".venv/bin/python runs/experiments/{BATCH.name}/reproduce_analysis.py", "```", "",
        "The deployment controls and v1.6 policy tests passed locally and remotely (26 tests). "
        "New metadata validates individually. The repository-wide legacy metadata command "
        "still fails on an older v2.1 expression-valued extraction field; historical records "
        "were preserved. Full SSE streams, raw grader audits, service logs and weights remain remote.",
    ]
    (BATCH / "README.md").write_text("\n".join(text) + "\n")
    print(json.dumps({"valid": audit["valid"], "deltas_s": deltas, "awq": actual}, indent=2))


if __name__ == "__main__":
    main()
