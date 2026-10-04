"""Reproduce the single BF16 versus NVFP4 deployment comparison."""

import json
from pathlib import Path
import sys
import statistics

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from runner.extensions.validation import bf16_v1_6 as driver

BATCH = Path(__file__).resolve().parent


def main():
    batch_config = driver.baseline.load(BATCH / "config.json")
    scored = driver.baseline.load(BATCH / "summary.json")
    ref = driver.reference_trial(scored["seed"])
    output = ROOT / "attempts" / scored["attempt_id"]
    measured = driver.audit(output, ref, batch_config["core_manifest_sha256"])
    if measured["valid"] != scored["valid"]:
        raise RuntimeError("Scored validity failed reproduction")
    old = driver.baseline.load(ROOT / "attempts" / ref["attempt_id"] / "summary.json")
    evidence = driver.baseline.load(BATCH / "server-evidence.json")
    delta = (
        measured["time_to_target_s"] - ref["time_to_target_s"]
        if measured["time_to_target_s"] is not None
        else None
    )
    result = dict(
        seed=scored["seed"],
        source_commit=batch_config["source_commit"],
        audit=measured,
        reference=ref,
        delta_s=delta,
        relative_change=delta / ref["time_to_target_s"] if delta is not None else None,
    )
    (BATCH / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    table = [
        "| Measure | NVFP4 / Marlin | BF16 / native linear |",
        "| --- | ---: | ---: |",
    ]
    table.append(
        f"| Time to 18 | {ref['time_to_target_s']:.3f}s | {measured['time_to_target_s']:.3f}s |"
    )
    table.append(
        f"| Generation requests | {ref['generation_requests']} | {measured['generation_requests']} |"
    )
    table.append(
        f"| Fresh samples | {ref['fresh_samples']} | {measured['fresh_samples']} |"
    )
    for title, key in (
        ("First grader pickup", "first_pick_elapsed_s"),
        ("Completed grader service", "actual_service_s"),
        ("Grader idle between queries", "idle_between_queries_s"),
    ):
        table.append(
            f"| {title} | {old['grader_timeline'][key]:.3f}s | {measured['grader_timeline'][key]:.3f}s |"
        )
    table.append(
        f"| Completed wrong checks | {old['grader_timeline']['wrong']} | {measured['grader_timeline']['wrong']} |"
    )
    for title, key in (
        ("Fresh request median TTFT", "fresh_ttft"),
        ("Continuation median TTFT", "continuation_ttft"),
    ):
        a, b = (
            old["performance"][key]["median_s"],
            measured["performance"][key]["median_s"],
        )
        table.append(
            f"| {title} | {a:.3f}s | {b:.3f}s |"
            if a is not None and b is not None
            else f"| {title} | {a} | {b} |"
        )
    initial_ttft = {}
    for label, folder in (
        ("bf16", output),
        ("nvfp4", ROOT / "attempts" / ref["attempt_id"]),
    ):
        values = [
            driver.baseline.load(p)["ttft_s"]
            for p in folder.glob("trace/*/rollout-01/telemetry.json")
        ]
        initial_ttft[label] = statistics.median(values)
    result["initial_30_ttft_median_s"] = initial_ttft
    table.append(
        f"| Initial 30-request median TTFT (arithmetic median) | {initial_ttft['nvfp4']:.3f}s | {initial_ttft['bf16']:.3f}s |"
    )
    (BATCH / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    barrier = measured["barrier_release"]
    phase = (
        f"The initial coverage/check barrier released at {barrier['elapsed_s']:.3f}s, then the pool refilled without further global barriers."
        if barrier
        else "The target was reached during initial coverage, before the pool opened."
    )
    text = [
        "# One BF16/FlashInfer v1.6 AIME 2025 comparison",
        "",
        f"Seed {scored['seed']}: BF16 reached 18 verified correct in **{measured['time_to_target_s']:.3f}s**, versus **{ref['time_to_target_s']:.3f}s** for the matched NVFP4 deployment ({delta:+.3f}s; {result['relative_change']:+.1%}). All attempt audits passed: **{measured['valid']}**. This is one scored run, with no replacement trial.",
        "",
        *table,
        "",
        "## Runner and controls",
        "",
        "Both use unchanged v1.6: initial 30×1 at 8K, coverage barrier including queued checks, then a 30-slot pool prioritizing exact-token continuations and least-active fresh samples with rotating ties. Each question permits four fresh samples; every fresh trajectory starts at 8K and gets at most one continuation to the remaining cumulative 64K output/context budget. Correct verdicts cancel the question, and 18 distinct correct verdicts stop the attempt.",
        "",
        phase,
        "",
        "Same A100 80GB, runtime versions, dataset provenance, improved prompt, temperature 0.8/top-p 0.95, base seed and continuation seed mapping, 3-second serialized grader, 95% memory, BF16 activations/KV and FlashInfer attention. Both server output ceilings and total context are 65,536. BF16 uses WeiboAI/VibeThinker-3B with quantization absent and automatic unquantized linear dispatch; NVFP4 uses r0b0tlab/VibeThinker-3B-NVFP4 with ModelOpt FP4 weights and Marlin linear kernels. Profiles and older results remain separate.",
        "",
        "Each started a fresh owned server, reset prefix cache, ran cheap 30-stream/32-token arithmetic warmup, and started a fresh grader. The benchmark timer starts after initialization/tokenization/warmup, immediately before scheduling; time to 18 uses the eighteenth distinct first-solved verdict receipt. Optional profiling/engine polls/NVML sampling are disabled. Required client traces buffer in RAM and flush after official timing. The 600-second per-trial wall safety limit includes trial initialization after server readiness.",
        "",
        "## Provenance and interpretation",
        "",
        "The downloaded tokenizers are byte-identical and the architecture/RoPE settings agree. BF16 weight revision is 77bd2cced09193c8b9a59a32bd8577bbd1f3e01c; the NVFP4 artifact revision is 2fc0013974d1a466e6a5a11839f029d5aff34dc9. Its model card names the WeiboAI base but does not pin that base revision. Saved batch config includes these local model/config/tokenizer identities. This compares the downloaded deployments, including native linear versus Marlin inference math. One seed cannot establish a repeatable quantization speedup or separate decoding speed from changed reasoning/answers.",
        "",
        f"Source `{batch_config['source_commit']}`; unchanged v1.6 manifest `{batch_config['core_manifest_sha256']}`. The audit checks all 30 initial request payloads after removing only model identity, the same policy controls, served prompt lengths/dataset hashes, exact continuation prefixes and budgets, first-solved timing, initial barrier settlement, and fresh/global request caps. Correctness uses grader verdicts; answer fields are retained only for existing startup validation.",
        "",
        f"Existing server logs show maximum KV usage {evidence['max_logged_kv_cache_usage_pct']}% across startup/warmup/solve, with {len(evidence['pressure_warnings'])} logged pressure warnings. See server-evidence.json for runtime backend selection, cache capacity and cleanup. These are coarse existing log samples, not sampled peak VRAM or complete eviction counters. Throughput log observations include mixed active counts and censored requests, so they are diagnostic rather than a fixed-concurrency benchmark.",
        "",
        "The 26-live-request endpoint log samples show 4,222.6 tok/s for NVFP4 and 3,332.4 tok/s for BF16. They average the preceding 10-second intervals, whose active counts/context trajectories differ, and are not a controlled fixed-concurrency throughput test. The initial-30 TTFT median was 0.222s versus 3.439s. The four logged Triton sampling JIT warnings occurred during cheap warmup (23:10:02–04 UTC), before official start at 23:10:06.470 UTC; they do not establish the cause of the later TTFT gap. No timed GPU/CPU profiling was enabled to attribute that gap.",
        "",
        "BF16's slightly shorter solve time came with 54.002s completed grader service (zero wrong checks) versus 63.003s for NVFP4 (three wrong checks). BF16 first pickup was 2.498s later and grader inter-query idle was 4.645s longer; the saved clock decomposition accounts for the approximately 1.858s net advantage. BF16's final two winners were Q18 and Q9, from their initial streams. It reached the target before any continuation or fresh sibling was launched. That means this trial evaluates initial-coverage solving and the stop rule; it does not exercise the post-barrier pool on BF16.",
        "",
        "All new BF16 metadata validates. The repository-wide legacy metadata command stops on the existing v2.1 expression-valued extraction field; older records were preserved. Reviewable imported artifacts total 3.20 MiB before analysis; full SSE streams/grader audits/service logs remain remote and ignored. Both owned ports were closed and no GPU compute processes remained after cleanup.",
        "",
        "## Reproduce",
        "",
        "```sh",
        f".venv/bin/python runs/experiments/{BATCH.name}/reproduce_analysis.py",
        "```",
        "",
        "The new BF16 profile/control checks and all v1.6 policy checks passed locally and remotely (17 tests). See config.json, summary.json, analysis.json, server-evidence.json and the referenced attempt for the full evidence.",
        "",
    ]
    (BATCH / "README.md").write_text("\n".join(text))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("audit", "reference")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
