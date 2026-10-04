"""Audit paired AIME 2025 evidence and render the core-v2 back-test comparison.

Run from the repository root; no inference, network access or answer-key reads.
"""
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aime-core-v2-matplotlib")
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "core-v2-aime2025-five-seeds-20261004T013100Z"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from runner_final import backtest_v2
BATCH = Path(__file__).resolve().parent
summary = json.loads((BATCH / "summary.json").read_text())
config = json.loads((BATCH / "config.json").read_text())
protocol = config["protocol"]
assert summary["status"] == "complete"
assert [r["sampling_seed"] for r in summary["trials"]] == protocol["seeds"]
rows = []
for trial in summary["trials"]:
    seed = trial["sampling_seed"]
    folder = ROOT / "attempts" / trial["attempt_id"]
    if not trial["valid"]:
        rows.append({"seed": seed, "attempt_id": folder.name, "valid": False,
                     "status": trial["status"]})
        continue
    checked = backtest_v2.score(folder, seed, protocol["reference_attempts"][str(seed)])
    assert checked["valid"]
    attempt_config = json.loads((folder / "config.json").read_text())
    assert attempt_config["git_commit"] == config["source_commit"]
    assert attempt_config["model_profile_sha256"] == config["profile_sha256"]
    assert attempt_config["benchmark"] and attempt_config["skip_benchmark_prewarm"]
    assert attempt_config["max_context_tokens"] == 65536
    attempt = json.loads((folder / "summary.json").read_text())
    questions = [json.loads(p.read_text()) for p in sorted(folder.glob("trace/*/question.json"))]
    solved = sorted((q for q in questions if q.get("first_solved")),
                    key=lambda q: q["first_solved"]["first_solved_elapsed_s"])
    assert len(solved) == len({q["problem_idx"] for q in solved}) == 18
    assert abs(solved[-1]["first_solved"]["first_solved_elapsed_s"] - attempt["time_to_target_s"]) < .01
    for question in solved:
        first, verdict = question["first_solved"], question["winner"]["result"]
        assert first["grader_query_id"] == verdict["query_id"]
        assert first["grader_answered_at_utc"] == verdict["answered_at"]
    timeline = attempt["grader_timeline"]
    assert not timeline["audit_errors"] and timeline["correct"] == 18
    assert abs(sum(timeline[k] for k in ("first_pick_elapsed_s", "actual_service_s", "idle_between_queries_s")) - attempt["time_to_target_s"]) < .03
    continuations = 0
    for question in questions:
        for rollout in question["rollouts"]:
            parent = rollout.get("continuation_of_rollout")
            if parent is None:
                continue
            request = json.loads((folder / "trace" / f"{question['problem_idx']:02d}" / f"rollout-{rollout['rollout']:02d}" / "request.json").read_text())
            tokens = json.loads((folder / "trace" / f"{question['problem_idx']:02d}" / f"rollout-{parent:02d}" / "tokens.json").read_text())
            parent_record = next(r for r in question["rollouts"] if r["rollout"] == parent)
            assert tokens["complete"] and parent_record["finish_reason"] == "length"
            assert parent_record["status"] == "completed"
            assert request["prompt"] == tokens["prompt_token_ids"] + tokens["output_token_ids"]
            assert len(request["prompt"]) + request["max_tokens"] <= 65536
            continuations += 1
    wrong = []
    for path in sorted(folder.glob("trace/*/verification.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("result", {}).get("verdict") is False:
                wrong.append({"question": int(path.parent.name), "candidate": event["candidate"],
                              "kind": event["kind"], "round": event["round"]})
    assert len(wrong) == timeline["wrong"]
    reference = protocol["reference_times_s"][str(seed)]
    rows.append({"seed": seed, "attempt_id": folder.name, "valid": True,
                 "v2_s": attempt["time_to_target_s"], "v1_s": reference,
                 "difference_s": attempt["time_to_target_s"] - reference,
                 "first_pick_s": timeline["first_pick_elapsed_s"],
                 "service_s": timeline["actual_service_s"],
                 "idle_s": timeline["idle_between_queries_s"], "wrong_checks": len(wrong),
                 "wrong_candidates": wrong,
                 "wrong_candidate_counts": dict(Counter(e["candidate"] for e in wrong)),
                 "literal_expression_wrong_checks": sum(e["candidate"] == "EXPRESSION" for e in wrong),
                 "placeholder_wrong_checks": sum(e["candidate"] in {"EXPRESSION", "...", "?", "??", "???"} for e in wrong),
                 "requests": attempt["performance"]["generation_requests"],
                 "exact_id_continuations": continuations,
                 "winner_kinds": dict(Counter(q["winner"]["kind"] for q in solved)),
                 "last_three_solved": [q["problem_idx"] for q in solved[-3:]],
                 "fresh_ttft_median_s": attempt["performance"]["fresh_ttft"]["median_s"],
                 "continuation_ttft_median_s": attempt["performance"]["continuation_ttft"]["median_s"]})
valid = [row for row in rows if row["valid"]]
times = [row["v2_s"] for row in valid]
analysis = {"source_commit": config["source_commit"],
            "core_manifest_sha256": config["core_manifest_sha256"],
            "system_prompt_sha256": config["system_prompt_sha256"], "rows": rows,
            "successful_trials": len(valid), "declared_trials": len(rows),
            "median_s": statistics.median(times) if times else None,
            "min_s": min(times) if times else None, "max_s": max(times) if times else None,
            "v1_median_s": statistics.median(protocol["reference_times_s"].values()),
            "median_paired_difference_s": statistics.median(row["difference_s"] for row in valid) if valid else None,
            "limits": ["Historical seed pairing is sequential, not randomized/interleaved.",
                       "The bundle changes the prompt, extraction and question service together.",
                       "Each five-trial batch shares one inference-server lifetime.",
                       "AIME 2025 is development data; benchmark mode disables optional profiling."]}
(BATCH / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for row in valid:
    axes[0].plot([0, 1], [row["v1_s"], row["v2_s"]], "o-", label=str(row["seed"]))
axes[0].set_xticks([0, 1], ["Core v1 + adherence prompt", "Core v2 + math prompt"])
axes[0].set_ylabel("Seconds to 18 distinct correct verdicts")
axes[0].set_title("Historical matched-seed comparison", loc="left", fontweight="bold")
positions = range(len(valid))
first = [r["first_pick_s"] for r in valid]
service = [r["service_s"] for r in valid]
idle = [r["idle_s"] for r in valid]
axes[1].bar(positions, first, label="Before first pickup", color="#849dac")
axes[1].bar(positions, service, bottom=first, label="Grader service", color="#286f8b")
axes[1].bar(positions, idle, bottom=[a+b for a,b in zip(first,service)], label="Later grader idle", color="#dc9c55")
axes[1].set_xticks(list(positions), [str(r["seed"])[-2:] for r in valid])
axes[1].set_xlabel("Seed suffix (202610xx)")
axes[1].set_title("Core v2 time decomposition", loc="left", fontweight="bold")
for i, row in enumerate(valid):
    axes[1].text(i, row["v2_s"] + 1, f"{row['v2_s']:.2f}s\n{row['wrong_checks']} wrong", ha="center", fontsize=8)
axes[1].set_ylim(0, max(times, default=60)+16)
for ax in axes:
    ax.axhline(54, color="#666666", linestyle=":", linewidth=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncols=2, fontsize=8)
fig.suptitle("AIME 2025 · frozen core v2 back-test · 54s serial verification floor", fontweight="bold")
fig.subplots_adjust(left=.075, right=.985, top=.85, bottom=.25, wspace=.25)
for extension in ("png", "svg"):
    fig.savefig(BATCH / f"comparison.{extension}", dpi=170, metadata={"Date":None} if extension=="svg" else {})
svg = BATCH / "comparison.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
plt.close(fig)
placeholders = sum(r["placeholder_wrong_checks"] for r in valid)
wrong_total = sum(r["wrong_checks"] for r in valid)
literal = sum(r["literal_expression_wrong_checks"] for r in valid)
table = "\n".join(
    f"| {r['seed']} | {r['v1_s']:.3f}s | {r['v2_s']:.3f}s | {r['difference_s']:+.3f}s | {r['wrong_checks']} | {r['placeholder_wrong_checks']} | {r['requests']} |"
    for r in valid)
decomposition = "\n".join(
    f"| {r['seed']} | {r['first_pick_s']:.3f}s | {r['service_s']:.3f}s | {r['idle_s']:.3f}s | {r['exact_id_continuations']} |"
    for r in valid)
if valid:
    conclusion = ("V2 is slower in this batch; retain v1 as the AIME speed reference."
                  if analysis["median_s"] > analysis["v1_median_s"]
                  else "V2 has a lower observed median in this historical comparison.")
    report = f"""# Frozen core v2: AIME 2025 back-test

**{len(valid)}/{len(rows)} declared trials reached 18 distinct verified correct.** V2 median
**{analysis['median_s']:.3f}s**, range **{analysis['min_s']:.3f}–{analysis['max_s']:.3f}s**.
The historical same-seed v1 improved-prompt controls had a **{analysis['v1_median_s']:.3f}s** median.
{conclusion} This is a back-test of the new mathematical prompt/extraction bundle, not an
isolated or interleaved parser experiment. All declared outcomes are retained in
[the batch summary](summary.json).

![Matched-seed comparison and v2 grader decomposition](comparison.png)

| Seed | V1 control | V2 | Paired change | Wrong checks | Placeholder checks | Requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

The concrete failure mode is explicit non-answer boxes. Across the valid trials,
**{placeholders}/{wrong_total} wrong checks** were the literal strings `EXPRESSION`,
`...`, `?`, `??` or `???`; **{literal}** were exactly `EXPRESSION`, matching the
example in the system prompt. Each wrong completed check occupies three seconds
of the serial grader, so those placeholder checks consumed approximately
**{placeholders * 3} seconds** across the batch. Per-question string deduplication
suppresses repeats of the same spelling; distinct placeholders still incur checks.
General mathematical extraction accepts such payloads, while v1's integer
extraction rejected them. These are saved trace observations; this batch did not
change the parser or prompt midway. The full wrong-candidate inventory is in
[analysis.json](analysis.json).

| Seed | First pickup | Completed grader service | Later idle | Exact-ID continuations |
| --- | ---: | ---: | ---: | ---: |
{decomposition}

First pickup + completed service + later idle reconciles with each successful
time to target within 0.03 seconds. The first-solved events link to true grader
verdicts; no candidate extraction alone counts as solved. There is a 54-second
floor from eighteen three-second grader checks. Server initialization, cheap
warmup, final trace flushing and service cleanup are outside time to target.

## Reproduce and inspect

```bash
# On callosum, from a clean tested checkout with a free GPU:
~/.venvs/vllm/bin/python -m runner_final.backtest_v2

# Locally, after importing the allowlisted attempt evidence:
.venv/bin/python runs/experiments/{BATCH.name}/reproduce_analysis.py
```

Source commit `{config['source_commit']}`; core manifest
`{config['core_manifest_sha256']}`; prompt SHA256
`{config['system_prompt_sha256']}`. The [predeclared protocol](../../../runner_final/five_seeds_core_v2_aime2025.json)
uses seeds 20261011–20261015, the unchanged NVFP4 Marlin / BF16 KV / FlashInfer
95% profile, 65,536-token context, 30×1 barrier coverage, 8,192 initial output
tokens, up to 16,384 additional tokens per continuation, and four requests per
question including continuations. Temperature is 0.8 and top-p is 0.95. Each
trial has a fresh v2 grader, cleared prefix cache and cheap 30×32-token warmup;
one owned inference server is reused across all five. There are no settling
trials, replacement seeds, workload warmups or settings changes between seeds.

The controller checks all thirty initial request payloads against the same-seed
v1 control, allowing only the declared prompt/model substitutions. The local
audit verifies core/prompt/source identity, the gold-free dataset snapshot and
grader key fingerprint, eighteen distinct first-solved verdicts, request caps,
and every saved exact-ID continuation prefix. Attempt metadata links each trial
to its same-seed control in the viewer.

AIME 2025 is development data. The two five-seed batches ran sequentially rather
than interleaved, and each shares one server lifetime; differences do not isolate
prompt, parser or run-state effects. Benchmark mode disables optional CPU/GPU/
engine profiling, so a measured peak VRAM and complete eviction counters are
unavailable. Coarse server logs and cleanup evidence are separate observations
in [server-evidence.json](server-evidence.json); raw SSE and audit/service logs
remain on the remote machine. Both frozen cores remain unchanged.
"""
    (BATCH / "README.md").write_text(report)
print(json.dumps(analysis, indent=2))
