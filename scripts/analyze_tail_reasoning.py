"""Retrospective, manually audited reasoning milestones in the final five trials.

Raw SSE artifacts remain outside Git. This does not implement an online answer
extractor: the selected expressions use the subsequently verified answer and a
human reading of the requested quantity. Reproduce with --streams-root pointing
to the extracted aime-final-tail-streams archive from the historical worktree.
"""
import argparse
import bisect
import json
from datetime import datetime
from pathlib import Path


BATCH = "runner-final-five-seeds-20261004T000926Z"
# Position of the first manually reviewed result for the requested quantity.
# Q02 deliberately excludes earlier 588s that refer only to triangle ABC.
AUDIT = {
    (20261011, 5): (22100, "late_derivation", "N − 2025 computed just after the 8K boundary"),
    (20261011, 2): (33911, "late_derivation", "heptagon shoelace calculation; earlier 588 is triangle area"),
    (20261011, 12): (31329, "post_result_rechecking", "a+b=510, then rechecks finite/unbounded regions and inequality signs"),
    (20261012, 26): (27328, "post_result_rechecking", "total 113, then recounts cycle matchings and duplicate chord lengths"),
    (20261012, 12): (38270, "late_derivation", "sign-case/region analysis finally yields a+b=510"),
    (20261012, 23): (27971, "post_result_rechecking", "successes=610, then revisits residue counts and the N<25 boundary"),
    (20261013, 11): (19731, "late_derivation", "requested parameter sum 259 followed promptly by recognized answer"),
    (20261013, 24): (23325, "error_repair", "corrects n=135 to n=139; n+t changes from 145 to 149"),
    (20261013, 23): (24145, "late_derivation", "successes=610 followed promptly by recognized answer"),
    (20261014, 26): (22558, "post_result_rechecking", "total 113, then repeats even/odd cycle and matching counts"),
    (20261014, 9): (20907, "late_derivation", "rotated-parabola parameter sum 62 followed promptly by answer"),
    (20261014, 18): (28707, "error_repair", "enumeration gives 82, contradicting 98; diagnoses wrong case multiplicity"),
    (20261015, 2): (23851, "late_derivation", "heptagon shoelace result; earlier 588 is triangle area"),
    (20261015, 7): (32789, "late_derivation", "probability 128/693 yields requested m+n=821"),
    (20261015, 18): (30203, "error_repair", "48+2+32=82 contradicts previous 83; repeats the nine contributions"),
}


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def stream_text(path):
    text, ends, times = "", [], []
    for line in path.open():
        chunk = json.loads(line)
        if chunk["data"] == "[DONE]":
            continue
        body = json.loads(chunk["data"])
        for choice in body.get("choices", []):
            delta = choice.get("delta", {})
            piece = choice.get("text", "") or delta.get("content", "") or ""
            if piece:
                text += piece
                ends.append(len(text))
                times.append(timestamp(chunk["timestamp_utc"]))
    return text, ends, times


def analyze(repo, streams_root):
    batch = repo / "runs/experiments" / BATCH
    summary = json.loads((batch / "summary.json").read_text())
    rows = []
    for trial in summary["trials"]:
        seed, attempt = trial["sampling_seed"], trial["attempt_id"]
        root = repo / "attempts" / attempt
        start = timestamp(json.loads((root / "summary.json").read_text())["official_started_at_utc"])
        questions = [json.loads(p.read_text()) for p in (root / "trace").glob("*/question.json")]
        solved = sorted((q for q in questions if q.get("winner")),
                        key=lambda q: q["winner"]["result"]["answered_at"])
        assert len(solved) == 18
        for question in solved[-3:]:
            q = question["problem_idx"]
            winner = question["winner"]
            position, category, note = AUDIT[seed, q]
            text, ends, times = "", [], []
            prefix_length = 0
            for rollout in range(1, winner["rollout"] + 1):
                relative = Path("attempts") / attempt / "trace" / f"{q:02d}" / f"rollout-{rollout:02d}"
                segment, segment_ends, segment_times = stream_text(streams_root / relative / "stream.jsonl")
                prefix_length = len(text)
                text += segment
                ends.extend(prefix_length + end for end in segment_ends)
                times.extend(segment_times)
                saved = json.loads((repo / relative / "tokens.json").read_text())["visible_text"]
                assert text == saved, (seed, q, rollout, "SSE differs from saved visible_text")
            answer = str(winner["answer"])
            assert text[position:position + len(answer)] == answer
            milestone = times[bisect.bisect_left(ends, position + len(answer))]
            observed = timestamp(winner["observed_at_utc"])
            reconstructed = times[bisect.bisect_left(ends, winner["end"])]
            # Literal-prose detection can wait for completion of its line, after
            # the matched numeral is streamed; keep that wait in the evidence.
            assert observed >= reconstructed, (seed, q, observed, reconstructed)
            lag = (observed - milestone).total_seconds()
            assert lag >= 0
            rows.append({
                "seed": seed, "attempt_id": attempt, "question": q,
                "answer": winner["answer"], "winning_rollout": winner["rollout"],
                "category": category, "note": note,
                "milestone_character": position,
                "milestone_at_utc": milestone.isoformat(),
                "milestone_elapsed_s": (milestone - start).total_seconds(),
                "parser_candidate_elapsed_s": (observed - start).total_seconds(),
                "matched_marker_streamed_elapsed_s": (reconstructed - start).total_seconds(),
                "marker_to_detection_delay_s": (observed - reconstructed).total_seconds(),
                "result_to_candidate_delay_s": lag,
                "verified_elapsed_s": (timestamp(winner["result"]["answered_at"]) - start).total_seconds(),
                "milestone_in_continuation": position >= prefix_length and winner["rollout"] > 1,
                "excerpt": text[max(0, position - 160):position + 330],
                "winner_kind": winner["kind"],
                "winner_line": winner.get("line"),
                "question_record": f"attempts/{attempt}/trace/{q:02d}/question.json",
            })
    assert len(rows) == len(AUDIT) == 15
    return {
        "scope": "Last three of eighteen verified questions in each of five trials; selected retrospectively",
        "method": "Manual semantic milestone audit; exact text reconstruction and chunk timestamps; not an online extraction policy",
        "limitations": "Correct numerals identified with later verifier result. Milestones are provisional reasoning statements, not proof of correctness or safe early-stop points. UTC chunks have millisecond precision. Categories are descriptive, not a statistical topic effect.",
        "tail_winners": len(rows),
        "continuation_winners": sum(r["winning_rollout"] > 1 for r in rows),
        "category_counts": {c: sum(r["category"] == c for r in rows) for c in sorted({r["category"] for r in rows})},
        "rows": rows,
    }


def render(result, folder):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aime-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"]
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, row in enumerate(rows):
        y = len(rows) - i - 1
        a, b, c = (row[k] for k in ("milestone_elapsed_s", "parser_candidate_elapsed_s", "verified_elapsed_s"))
        ax.plot([a, b], [y, y], color="#e49a32", lw=5, alpha=.7)
        ax.plot([b, c], [y, y], color="#6a9c7e", lw=3, alpha=.7)
        ax.scatter([a], [y], color="#b96b06", marker="D", s=35,
                   label="Requested result appears in reasoning (retrospective)" if i == 0 else None, zorder=3)
        ax.scatter([b], [y], color="#2b67aa", s=32,
                   label="Parser detects candidate" if i == 0 else None, zorder=4)
        ax.scatter([c], [y], color="#238254", marker="|", s=150,
                   label="Grader verifies correct" if i == 0 else None, zorder=5)
        if b - a > 5:
            ax.text((a + b) / 2, y + .19, f"{b-a:.1f}s", ha="center", fontsize=9, color="#8a5108")
    ax.set_yticks(range(len(rows)), [f"seed {r['seed']} · Q{r['question']:02d}" for r in reversed(rows)])
    for y in [2.5, 5.5, 8.5, 11.5]:
        ax.axhline(y, color="#d1d5dc", lw=.8)
    ax.set_xlim(40, 110)
    ax.set_xlabel("Seconds from official attempt start")
    fig.suptitle("Late answers: correct-valued reasoning can precede parser detection by 23–27s", y=.98, fontsize=13)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=1, frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .015, "Last three verified questions per trial. Earlier reasoning statements use hindsight; they are not safe online stop points.", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / f"tail-reasoning.{extension}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--streams-root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.repo, args.streams_root)
    out = args.repo / "runs/experiments" / BATCH / "tail-reasoning.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    render(result, out.parent)
    for row in result["rows"]:
        print(f"{row['seed']} Q{row['question']:02d}: {row['category']} "
              f"result {row['milestone_elapsed_s']:.3f}s → candidate "
              f"{row['parser_candidate_elapsed_s']:.3f}s ({row['result_to_candidate_delay_s']:.3f}s)")
