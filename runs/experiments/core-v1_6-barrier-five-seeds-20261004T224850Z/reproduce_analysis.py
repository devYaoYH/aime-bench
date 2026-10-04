"""Reproduce the matched policy comparison using configs, tokens and verdicts."""

import csv
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from runner.extensions.validation import v1_6_batch as driver

BATCH = Path(__file__).resolve().parent
REFS = {
    "original_v1": driver.REFERENCE,
    "refactored_v1": "canonical-v1-refactor-five-seeds-20261004T215632Z",
    "v1_5": "core-v1_5-five-seeds-20261004T074500Z",
}


def stats(rows):
    values = [
        r["time_to_target_s"]
        for r in rows
        if r.get("target_reached", r.get("success", True))
        and r.get("time_to_target_s") is not None
    ]
    return dict(
        declared=len(rows),
        reached_target=len(values),
        median_s=statistics.median(values) if values else None,
        mean_s=statistics.mean(values) if values else None,
        sample_sd_s=statistics.stdev(values) if len(values) > 1 else None,
        spread_s=max(values) - min(values) if values else None,
        min_s=min(values) if values else None,
        max_s=max(values) if values else None,
    )


def main():
    batch_config, saved = (
        driver.load(BATCH / "config.json"),
        driver.load(BATCH / "summary.json"),
    )
    references = {
        name: driver.load(ROOT / "runs/experiments" / path / "summary.json")["trials"]
        for name, path in REFS.items()
    }
    original = {r["sampling_seed"]: r for r in references["original_v1"]}
    rows, policy_rows = [], []
    for declared in saved["trials"]:
        seed = declared["sampling_seed"]
        measured = (
            driver.measure(
                ROOT / "attempts" / declared["attempt_id"],
                original[seed],
                batch_config["core_manifest_sha256"],
            )
            if declared.get("status") != "failed"
            else declared
        )
        if measured.get("valid") != declared.get("valid"):
            raise RuntimeError("Reproduced validity differs from scored batch")
        if declared.get("status") != "failed":
            output = ROOT / "attempts" / declared["attempt_id"]
            questions = [driver.load(p) for p in output.glob("trace/*/question.json")]
            winners = []
            for q in questions:
                if not q.get("first_solved"):
                    continue
                r = next(
                    r for r in q["rollouts"] if r["rollout"] == q["winner"]["rollout"]
                )
                winners.append(
                    dict(
                        problem_idx=q["problem_idx"],
                        elapsed_s=q["first_solved"]["first_solved_elapsed_s"],
                        fresh_sample=r["fresh_sample"],
                        segment=r["segment"],
                    )
                )
            measured["last_two_solved"] = sorted(winners, key=lambda v: v["elapsed_s"])[
                -2:
            ]
            measured["winning_trajectories"] = dict(
                initial=sum(
                    w["fresh_sample"] == 1 and w["segment"] == 1 for w in winners
                ),
                fresh_sibling=sum(w["fresh_sample"] > 1 for w in winners),
                initial_continuation=sum(
                    w["fresh_sample"] == 1 and w["segment"] == 2 for w in winners
                ),
            )
            allocation = driver.load(output / "allocation.json")
            first_pool = [a for a in allocation["admissions"] if a["phase"] == "pool"][
                :30
            ]
            measured["first_pool_fill"] = dict(
                fresh=sum(a["segment"] == 1 for a in first_pool),
                continuations=sum(a["segment"] == 2 for a in first_pool),
                question_counts={
                    str(q): sum(a["problem_idx"] == q for a in first_pool)
                    for q in sorted({a["problem_idx"] for a in first_pool})
                },
            )
        policy_rows.append(measured)
        row = dict(
            seed=seed,
            attempt_id=declared.get("attempt_id"),
            valid=measured.get("valid"),
            v1_6_s=measured.get("time_to_target_s"),
            barrier_s=(measured.get("barrier_release") or {}).get("elapsed_s"),
            fresh=measured.get("fresh_samples"),
            continuations=measured.get("continuation_requests"),
            checks=measured.get("completed_checks"),
            wrong=measured.get("wrong_checks"),
        )
        for name, trials in references.items():
            ref = next(r for r in trials if r["sampling_seed"] == seed)
            row[name + "_s"] = ref["time_to_target_s"]
            row["delta_" + name + "_s"] = (
                row["v1_6_s"] - ref["time_to_target_s"]
                if row["v1_6_s"] is not None
                else None
            )
        rows.append(row)
    result = dict(
        source_commit=batch_config["source_commit"],
        rows=rows,
        audit=driver.aggregate(policy_rows),
        statistics={
            **{name: stats(trials) for name, trials in references.items()},
            "v1_6": stats(policy_rows),
        },
        policy_audits=policy_rows,
    )
    (BATCH / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    file = io.StringIO()
    writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (BATCH / "paired-seeds.csv").write_text(file.getvalue())
    table = [
        "| Seed | Original v1 | Refactored v1 | v1.5 | v1.6 | Initial barrier | Fresh / continued | Wrong checks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        seconds = lambda value: f"{value:.3f}s" if value is not None else "unmet/failed"
        table.append(
            f"| {r['seed']} | {seconds(r['original_v1_s'])} | {seconds(r['refactored_v1_s'])} | {seconds(r['v1_5_s'])} | {seconds(r['v1_6_s'])} | {seconds(r['barrier_s']) if r['barrier_s'] is not None else 'target before pool'} | {r['fresh']} / {r['continuations']} | {r['wrong']} |"
        )
    comparison = [
        "| Runner | Reached 18 | Median | Mean | Sample SD | Range |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, s in result["statistics"].items():
        comparison.append(
            f"| {name} | {s['reached_target']}/{s['declared']} | {s['median_s']:.3f}s | {s['mean_s']:.3f}s | {s['sample_sd_s']:.3f}s | {s['min_s']:.3f}–{s['max_s']:.3f}s |"
        )
    latest, new = result["statistics"]["refactored_v1"], result["statistics"]["v1_6"]
    tail_interpretation = f"Compared with the latest refactored-v1 batch, the observed worst run fell from {latest['max_s']:.3f}s to {new['max_s']:.3f}s; sample SD fell from {latest['sample_sd_s']:.3f}s to {new['sample_sd_s']:.3f}s and the range narrowed from {latest['spread_s']:.3f}s to {new['spread_s']:.3f}s. Mean latency fell from {latest['mean_s']:.3f}s to {new['mean_s']:.3f}s, while the median rose from {latest['median_s']:.3f}s to {new['median_s']:.3f}s. This is encouraging evidence for a smaller observed tail under the combined policy. Original v1 and v1.5 had tighter spreads, and the median is essentially unchanged versus original v1. Five historical trials do not isolate the barrier or establish a general reduction in population variance."
    wins = {
        key: sum(r.get("winning_trajectories", {}).get(key, 0) for r in policy_rows)
        for key in ("initial", "initial_continuation", "fresh_sibling")
    }
    request_counts = dict(
        fresh=sum(r.get("fresh_samples", 0) for r in policy_rows),
        continuations=sum(r.get("continuation_requests", 0) for r in policy_rows),
    )
    outcomes = f"Winning trajectories across all trials: {wins['initial']} initial streams, {wins['initial_continuation']} continuations of initial streams, and {wins['fresh_sibling']} extra fresh sibling. The runner admitted {request_counts['fresh']} fresh samples and {request_counts['continuations']} continuations. Most wins therefore came from initial coverage or its continuation; direct fresh-sibling wins are sparse in this batch. Scheduling can also change the numeric decoding trajectory, so win counts alone cannot attribute a latency effect. The 101.123s trial's final two winners were Q7 and Q20, both initial-stream continuations; completed grader service was 54.002s, first pickup 6.030s and inter-query idle 41.090s."
    report = "\n".join(
        [
            "# V1.6 initial coverage barrier and pool: five matched seeds",
            "",
            "All five declared seeds are scored, including the first. Failures and unmet targets remain; no replacement or settling trial is permitted. AIME 2025 is development data.",
            "",
            tail_interpretation,
            "",
            *comparison,
            "",
            *table,
            "",
            outcomes,
            "",
            "## Policy and controls",
            "",
            "Launch exactly 30 initial 8K streams. Hold freed slots idle until all initial streams and their queued grader checks settle. If 18 questions solve during that phase, stop without opening the pool. Otherwise open a 30-request pool: ready exact-token continuations first, then fresh samples for least-active unsolved questions with rotating ties. No further global barriers. Every fresh sample starts at 8K; at most four fresh trajectories per question, with at most one continuation per trajectory to the remaining cumulative 64K budget, clipped to the existing 64K total context. Correct verdicts cancel all streams for that question. Wrong verdicts provide no model feedback.",
            "",
            "Same A100 80GB, NVFP4 Marlin, BF16 KV, FlashInfer, 95% memory, improved integer-answer prompt, temperature 0.8/top-p 0.95, five seeds 20261011–20261015, 3-second serialized grader and target 18 as the historical controls. The dedicated profile raises only the server output ceiling from 16K to 64K. All 150 initial request payloads match original v1 exactly. Continuation seeds use a disjoint +124 band for AIME 2025, while fresh sample seeds retain base + index×4 + sample; old v1 continued with the next HTTP-rollout seed. Continuation draws therefore differ as part of this extension, even with matched base seeds. One owned server serves all five trials; prefix cache resets, cheap arithmetic warmup and a fresh grader precede each timed trial. Optional profiling/GPU polls are disabled; required traces buffer in RAM and flush after official solving time. Per-trial wall safety timeout is 600s including initialization.",
            "",
            "Time to 18 starts after dataset loading, service startup, prompt tokenization and warmup, immediately before generation scheduling. It ends at the eighteenth distinct correct verdict receipt, excluding cancellation settlement and trace flush. Startup answer-field validation is retained; correctness and analysis use grader verdicts, not gold answer fields.",
            "",
            "## Verification and interpretation",
            "",
            f"Source commit `{batch_config['source_commit']}`; v1.6 manifest `{batch_config['core_manifest_sha256']}`. The canonical v1 manifest remains unchanged. The runtime audit checks all 30 initial request payloads against original v1, exact continuation prefixes/budgets/seeds, the fresh/HTTP caps, coverage settlement before pool admissions, peak concurrent requests and eighteenth first-solved timing. Inspect `analysis.json` for every audit and grader timeline.",
            "",
            "Five paired historical seeds provide a practical comparison, not statistical equivalence or an isolated causal estimate. V1.6 changes the post-coverage scheduler, fresh-only cap, continuation budget and continuation seed mapping together. Historical v1.5 is an additional slot-pool comparison; it did not use this initial barrier. Use batch medians/ranges, not a single best timing, for claims.",
            "",
            "The existing vLLM logs recorded 40 KV samples with a maximum of 11.2% across the server lifetime, and no OOM/preemption/eviction pressure warnings. Startup reported 71.87 GiB available KV memory and 2,093,344 token capacity (31.94 full-64K requests). Cleanup left zero GPU compute processes, 0 MiB used VRAM and both owned ports closed. Benchmark mode has no sampled peak VRAM or complete engine preemption counters. `server-evidence.json` contains coarse existing engine-log KV observations and cleanup checks; absence of logged warnings does not prove that no KV blocks were ever evicted.",
            "",
            "Validation: 391 full offline suite checks passed from the staged source snapshot, then 19 focused policy/batch checks passed locally and remotely. All five imported attempt audits and metadata schemas pass. The repository-wide legacy metadata command stops on an existing v2.1 expression-valued extraction field that its scalar schema cannot accept; historical records were preserved and each new v1.6 record was validated separately. Only allowlisted artifacts are versioned (34.56 MiB before added local analysis); full SSE streams, grader audits and service logs remain remote/ignored.",
            "",
            "## Reproduce",
            "",
            "```sh",
            f".venv/bin/python runs/experiments/{BATCH.name}/reproduce_analysis.py",
            "```",
            "",
            "See `config.json`, `summary.json`, `analysis.json`, `paired-seeds.csv` and the linked attempt IDs for reviewable evidence.",
            "",
        ]
    )
    (BATCH / "README.md").write_text(report)
    print(
        json.dumps(
            {"statistics": result["statistics"], "audit": result["audit"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
