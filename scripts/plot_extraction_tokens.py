"""Exact generated-token counts at successful candidate detection in five trials.

Replays the frozen detector against saved SSE and verifies exact output IDs.
Counts tokens, not prompt tokens, later verification-time generation, or chars.
Unverified questions remain missing observations, never zero-token successes.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics

from runner_final._streaming import CandidateDetector
from src.common import ROOT

BATCH = "runner-final-five-seeds-20261004T000926Z"


def replay(folder, stream, winner):
    tokens = json.loads((folder / "tokens.json").read_text())
    telemetry = json.loads((folder / "telemetry.json").read_text())
    detector = CandidateDetector()
    parent = telemetry.get("continuation_of_rollout")
    if parent:
        prefix = json.loads((folder.parent / f"rollout-{parent:02d}/tokens.json").read_text())["visible_text"]
        detector.feed("content", prefix)
    observed_ids, found = [], None

    def consider(events):
        nonlocal found
        for event in events:
            if found is None and all(event[k] == winner[k] for k in ("answer", "kind", "part", "end")):
                found = len(observed_ids)

    for line in stream.open():
        chunk = json.loads(line)
        if chunk["data"] == "[DONE]":
            break
        body = json.loads(chunk["data"])
        for choice in body.get("choices", []):
            assert choice.get("index", 0) == 0
            observed_ids.extend(choice.get("token_ids") or [])
            delta = choice.get("delta") or {}
            if telemetry["endpoint"] == "/v1/completions":
                delta = {"content": choice.get("text", "")}
            for part, value in (("reasoning", delta.get("reasoning_content") or delta.get("reasoning")),
                                ("content", delta.get("content"))):
                if isinstance(value, str) and value:
                    consider(detector.feed(part, value))
            if choice.get("finish_reason") == "stop":
                for part in ("content", "reasoning"):
                    consider(detector.feed(part, "", eof=True))
    assert observed_ids == tokens["output_token_ids"], folder
    assert detector.text["content"] == tokens["visible_text"], folder
    assert found is not None and found > 0, (folder, winner)
    return found


def generated_prefix(folder, rollout):
    telemetry = json.loads((folder / f"rollout-{rollout:02d}/telemetry.json").read_text())
    parent = telemetry.get("continuation_of_rollout")
    if not parent:
        return 0
    tokens = json.loads((folder / f"rollout-{parent:02d}/tokens.json").read_text())
    return len(tokens["output_token_ids"]) + generated_prefix(folder, parent)


def analyze(repo, streams_root):
    trials = json.loads((repo / "runs/experiments" / BATCH / "summary.json").read_text())["trials"]
    rows = []
    for trial in trials:
        seed, attempt = trial["sampling_seed"], trial["attempt_id"]
        for q in range(1, 31):
            relative = Path("attempts") / attempt / "trace" / f"{q:02d}"
            folder = repo / relative
            question = json.loads((folder / "question.json").read_text())
            winner = question.get("winner")
            observed = sum(len(json.loads(p.read_text())["output_token_ids"]) for p in folder.glob("rollout-*/tokens.json"))
            row = {"seed": seed, "attempt_id": attempt, "question": q,
                   "verified": bool(winner), "status": question["status"],
                   "observed_question_output_tokens": observed,
                   "trajectory_tokens_to_extraction": None,
                   "question_output_tokens_through_extraction": None}
            if winner:
                assert winner["result"]["verdict"] is True
                rollout = winner["rollout"]
                segment = replay(folder / f"rollout-{rollout:02d}", streams_root / relative / f"rollout-{rollout:02d}/stream.jsonl", winner)
                previous = sum(len(json.loads((folder / f"rollout-{r:02d}/tokens.json").read_text())["output_token_ids"]) for r in range(1, rollout))
                row.update(trajectory_tokens_to_extraction=generated_prefix(folder, rollout) + segment,
                           question_output_tokens_through_extraction=previous + segment,
                           winning_rollout=rollout, candidate_kind=winner["kind"],
                           candidate_observed_at_utc=winner["observed_at_utc"],
                           verifier_query_id=winner["result"]["query_id"])
            rows.append(row)
    verified = [r for r in rows if r["verified"]]
    assert len(rows) == 150 and len(verified) == 90
    thresholds = []
    for cap in (2048, 4096, 8192, 16384):
        per_seed = {str(t["sampling_seed"]): sum(r["verified"] and r["trajectory_tokens_to_extraction"] <= cap for r in rows if r["seed"] == t["sampling_seed"]) for t in trials}
        thresholds.append({"tokens": cap, "verified_observations": sum(per_seed.values()),
                           "per_seed_distinct_questions": per_seed})
    early_sets = [{r["question"] for r in verified if r["seed"] == t["sampling_seed"] and r["trajectory_tokens_to_extraction"] <= 4096} for t in trials]
    return {"method": "Replay frozen CandidateDetector; exact streamed output token IDs at the winning candidate event plus its continuation ancestors. All saved IDs and visible text checked. No local gold-key selection.",
            "scope": "Five warmed 30x1 runs; 90 verified answers among 150 question/trial observations. Success-token distribution is conditional on solving before global stop at 18; 60 others are unknown, not failed or statistical right-censoring points.",
            "interpretation": "4K threshold describes these observed trajectories; it cannot estimate pass@2 or predict changed-batch reasoning. Each 30x2 lane would have one continuation within the four-request cap.",
            "four_k_distinct_question_union": sorted(set.union(*early_sets)),
            "four_k_distinct_question_intersection": sorted(set.intersection(*early_sets)),
            "thresholds": thresholds, "rows": rows}


def render(result, folder):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aime-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2b67aa", "#d98127", "#238254", "#8656a8", "#b94b5f"]
    seeds = sorted({r["seed"] for r in result["rows"]})
    fig, (ax, coverage) = plt.subplots(1, 2, figsize=(13.5, 10), gridspec_kw={"width_ratios": [3.3, 1]})
    for q in range(1, 31):
        rows = [r for r in result["rows"] if r["question"] == q]
        successes = [r["trajectory_tokens_to_extraction"] for r in rows if r["verified"]]
        if len(successes) >= 2:
            ax.boxplot([successes], positions=[q], vert=False, widths=.55, showfliers=False, manage_ticks=False,
                       medianprops={"color":"#263548"}, boxprops={"color":"#9da9b5"}, whiskerprops={"color":"#9da9b5"}, capprops={"color":"#9da9b5"})
        for row in rows:
            if row["verified"]:
                i=seeds.index(row["seed"])
                ax.scatter(row["trajectory_tokens_to_extraction"],q+(i-2)*.09,s=20,color=colors[i],zorder=3)
        coverage.barh(q, len(successes), color="#728fae", height=.6)
        coverage.text(len(successes)+.12,q,str(len(successes)),va="center",fontsize=9)
    for cap,color in ((4096,"#b66a17"),(8192,"#748393")):
        ax.axvline(cap,color=color,ls="--",lw=1.3,label=f"{cap//1024}K boundary")
    ax.set_ylim(30.8,.2);coverage.set_ylim(30.8,.2)
    ax.set_yticks(range(1,31),[f"Q{q:02d}" for q in range(1,31)])
    ax.set_xlabel("Generated tokens before verified-correct candidate extraction (winning trajectory)")
    ax.set_xlim(left=0)
    coverage.set_xlim(0,6);coverage.set_xticks([0,5]);coverage.set_yticks([])
    coverage.set_xlabel("Verified / 5 trials")
    ax.grid(axis="x",alpha=.17)
    ax.legend(frameon=False,loc="upper right",fontsize=9)
    for chart in (ax,coverage):chart.spines[["top","right"]].set_visible(False)
    fig.suptitle("Successful answer extraction by question · five 30×1 trials",fontsize=14)
    fig.text(.02,.018,"Dots are individual seeds; boxes summarize observed successes only. Blank rows have no verified win before the global 18-answer stop.",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    for ext in ("png","svg","pdf"):fig.savefig(folder/f"extraction-token-distribution.{ext}",dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5))
    for i,seed in enumerate(seeds):
        vals=sorted(r["trajectory_tokens_to_extraction"] for r in result["rows"] if r["seed"]==seed and r["verified"])
        ax.step([0]+vals,[0]+list(range(1,len(vals)+1)),where="post",color=colors[i],label=str(seed))
    ax.axvline(4096,color="#b66a17",ls="--",label="4K proposed first-pass cap")
    ax.axvline(8192,color="#748393",ls="--",label="8K current first-pass cap")
    ax.axhline(18,color="#555",ls=":",lw=1)
    ax.set_xlabel("Generated tokens before candidate extraction (winning trajectory)")
    ax.set_ylabel("Distinct verified questions with extraction at or below token count")
    ax.set_title("Observed coverage at a token budget · excludes unverified stopped questions")
    ax.set_ylim(0,19);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ("png","svg","pdf"):fig.savefig(folder/f"extraction-token-coverage.{ext}",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=ROOT)
    parser.add_argument("--streams-root",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.repo,args.streams_root)
    folder=args.repo/"runs/experiments"/BATCH
    (folder/"extraction-token-distribution.json").write_text(json.dumps(result,indent=2)+"\n")
    render(result,folder)
    print(json.dumps(result["thresholds"],indent=2))
