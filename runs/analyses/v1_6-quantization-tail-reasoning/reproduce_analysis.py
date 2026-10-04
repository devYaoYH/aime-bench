"""Replay selected winning streams; never read dataset answer keys.

Raw SSE and the tokenizer stay outside Git. See README.md for prerequisites.
The frozen detector is audited, not modified. Earlier matching answer clauses
are hindsight landmarks, not replayed grader submissions or policy outcomes.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from runner.lib.extraction import CandidateDetector

EXTRACTOR_SHA = "4e611524d163ad2e9779280a249ba5fd03ef62b58286de2df02a7c022a95e909"
TOKENIZER_SHA = "296e081e2f5ecf9d87814aa9b0f4b12d670ed2b2e2be6c84e01a9466c953afb7"
CASES = [
    ("AWQ12", "20261004T232909.168373Z", 29, "Thus area = 104 √3."),
    ("AWQ11", "20261004T232342.669339Z", 29, "So area of quadrilateral BKLC = 104 sqrt(3)."),
    ("NV12", "20261004T225113.454134Z", 29, "Thus area = 104 sqrt(3). So n = 104."),
    ("NV13", "20261004T225234.136076Z", 23,
     "Now total successes = total numbers (1000) - failures (390) = 610."),
    ("AWQ13", "20261004T233116.619910Z", 23,
     "Thus successes = total N (1..1000) = 1000 - 390 = 610."),
    ("NV15", "20261004T225526.955249Z", 2,
     "heptagon area in actual = model heptagon area * (6/13) = 1274 * (6/13) = 588."),
    ("AWQ15", "20261004T233625.124298Z", 2,
     "Thus the area of the heptagon AFNBCEM equals 588"),
    ("NV14", "20261004T225342.150827Z", 7, None),
    ("AWQ14", "20261004T233357.876262Z", 7, None),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def analyze(label, attempt, index, conclusion, stream_root, tokenizer):
    folder = ROOT / "attempts" / attempt
    trace = folder / "trace" / f"{index:02}"
    summary, question = read(folder / "summary.json"), read(trace / "question.json")
    winner = question["winner"]
    assert winner["result"]["verdict"] is True
    assert question["unique_candidates"] == 1, "These examples each have one submitted scalar"
    rolls = {r["rollout"]: r for r in question["rollouts"]}
    chain = [winner["rollout"]]
    while rolls[chain[-1]]["continuation_of_rollout"] is not None:
        chain.append(rolls[chain[-1]]["continuation_of_rollout"])
    chain.reverse()
    segments, ids, text, sources = [], [], "", []
    for r in chain:
        part = trace / f"rollout-{r:02}"
        response, tokens = read(part / "response.json"), read(part / "tokens.json")
        assert not response["reasoning"], "These selected traces use the content channel"
        segment_text = response["content"]
        text += segment_text
        ids.extend(tokens["output_token_ids"])
        assert tokens["visible_text"] == text
        stream = stream_root / attempt / "trace" / f"{index:02}" / f"rollout-{r:02}" / "stream.jsonl"
        assert stream.is_file(), stream
        segments.append((r, segment_text, tokens["output_token_ids"], stream))
        for p in (part / "response.json", part / "tokens.json", part / "request.json"):
            sources.append({"path": str(p.relative_to(ROOT)), "sha256": sha(p)})
        sources.append({"path": str(stream.relative_to(stream_root)),
                        "storage": "ignored raw SSE cache", "sha256": sha(stream)})
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    assert decoded.startswith(text), "Exact generated IDs must decode to the saved prefix"

    def token_at(end):
        low, high = 0, len(ids)
        while low < high:
            middle = (low + high) // 2
            if len(tokenizer.decode(ids[:middle], skip_special_tokens=False)) >= end:
                high = middle
            else:
                low = middle + 1
        assert tokenizer.decode(ids[:low], skip_special_tokens=False)[:end] == text[:end]
        return low

    landmarks = {}
    if conclusion:
        start = text.index(conclusion)
        landmarks["target_quantity_conclusion"] = {
            "character_end": start + len(conclusion), "quote": conclusion,
            "interpretation": "Manual reading of the target quantity; not a submitted candidate",
        }
    early = re.search(r"(?i)\banswer\s*[:=]\s*" + str(winner["answer"]) + r"\s*(?=[.\n])",
                      text[:winner["end"]])
    if early:
        end = early.end() + 1  # Require the complete punctuation-delimited clause.
        landmarks["earlier_explicit_matching_clause"] = {
            "character_end": end, "quote": text[early.start():end].strip(),
            "interpretation": "Hindsight scalar match to the later True verdict; never graded here",
        }
    landmarks["recognized_candidate"] = {
        "character_end": winner["end"], "quote": winner.get("line", ""),
        "kind": winner["kind"], "interpretation": "Actual candidate whose saved grader verdict is True",
    }
    for landmark in landmarks.values():
        landmark["generated_tokens_through_character_end"] = token_at(landmark["character_end"])

    official = stamp(summary["official_started_at_utc"])
    prefix, cumulative_ids, accepted, boundaries = "", [], None, []
    for r, expected_text, expected_ids, stream in segments:
        detector = CandidateDetector()
        if prefix:
            detector.feed("content", prefix)  # Same initialization as a real continuation.
        current_text, current_ids, last_event = "", [], None
        for raw in stream.read_text().splitlines():
            event = json.loads(raw)
            if event["data"] == "[DONE]":
                continue
            body = json.loads(event["data"])
            for choice in body.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta", {}).get("content") or choice.get("text") or ""
                current_text += delta
                current_ids.extend(choice.get("token_ids") or [])
                clock = {
                    "timestamp_utc": event["timestamp_utc"],
                    "official_elapsed_s": (stamp(event["timestamp_utc"]) - official).total_seconds(),
                    "rollout": r, "chunk_generated_tokens": len(cumulative_ids) + len(current_ids),
                }
                last_event = clock
                for name, landmark in landmarks.items():
                    if name != "recognized_candidate" and "observed_event" not in landmark:
                        if len(prefix) + len(current_text) >= landmark["character_end"]:
                            landmark["observed_event"] = dict(clock)
                for candidate in detector.feed("content", delta):
                    if candidate["answer"] == winner["answer"] and accepted is None:
                        assert candidate["end"] == winner["end"], "The saved winner must be the first matching extraction"
                        accepted = dict(clock)
                        landmarks["recognized_candidate"]["observed_event"] = accepted
        assert current_text == expected_text
        assert current_ids == expected_ids
        prefix += current_text
        cumulative_ids.extend(current_ids)
        boundaries.append({"rollout": r, "finish_reason": rolls[r]["finish_reason"],
                           "last_received_event": last_event,
                           "generation_finished_at_utc": rolls[r]["generation_finished_at_utc"],
                           "started_at_utc": rolls[r]["started_at_utc"]})
    assert accepted is not None, (label, index, "Actual winner must replay with the frozen detector")
    assert abs((stamp(winner["observed_at_utc"]) - stamp(accepted["timestamp_utc"])).total_seconds()) <= 0.01
    for landmark in landmarks.values():
        assert "observed_event" in landmark
        landmark["gap_to_recognized_s"] = accepted["official_elapsed_s"] - landmark["observed_event"]["official_elapsed_s"]
    verdict_s = (stamp(winner["verification_finished_at_utc"]) - official).total_seconds()
    return {"label": label, "attempt_id": attempt, "question_index": index,
            "winning_rollout_chain": chain, "landmarks": landmarks,
            "recognized_to_verdict_s": verdict_s - accepted["official_elapsed_s"],
            "correct_verdict_official_elapsed_s": verdict_s,
            "candidate_queue_wait_s": winner["candidate_queue_wait_s"],
            "request_boundaries": boundaries,
            "sources": sources + [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
                                  for p in (trace / "question.json", folder / "summary.json")]}


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    ax = axes[0]
    values = [r["landmarks"]["recognized_candidate"]["generated_tokens_through_character_end"] for r in rows]
    labels = [f"Q{r['question_index']} · {r['label']}" for r in rows]
    colors = ["#2864a0" if r["label"].startswith("NV") else "#dd8132" for r in rows]
    ax.barh(range(len(rows)), values, color=colors)
    for y, value in enumerate(values):
        ax.text(value + 90, y, f"{value:,}", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.axvline(8192, color="#555555", linestyle="--", linewidth=1)
    ax.text(8250, -0.65, "8K initial cap", color="#555555")
    ax.set_xlim(0, 14500)
    ax.set_xlabel("Exact cumulative generated tokens through the accepted candidate")
    ax.set_title("Selected winning trajectories: route length and answer emission differ")
    ax.spines[["right", "top"]].set_visible(False)
    ax = axes[1]
    q7 = [r for r in rows if r["question_index"] == 7]
    for y, r in enumerate(q7):
        early = r["landmarks"]["earlier_explicit_matching_clause"]["observed_event"]["official_elapsed_s"]
        recognized = r["landmarks"]["recognized_candidate"]["observed_event"]["official_elapsed_s"]
        verdict = r["correct_verdict_official_elapsed_s"]
        ax.plot([early, recognized, verdict], [y] * 3, color="#999999", linewidth=3)
        ax.scatter([early, recognized, verdict], [y] * 3, c=["#dd8132", "#2864a0", "#3a8757"], s=65, zorder=3)
        ax.annotate(f"missed clause\n{early:.3f}s", (early, y), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"accepted\n{recognized:.3f}s", (recognized, y), xytext=(0, -30), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"True verdict\n{verdict:.3f}s", (verdict, y), xytext=(20, 12), textcoords="offset points", ha="center", fontsize=9)
    ax.set_yticks(range(len(q7)), [r["label"] for r in q7])
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(48, 110)
    ax.set_xlabel("Seconds from the official attempt start; saved SSE timestamps")
    ax.set_title('Q7: “answer: 821.” arrives well before the accepted “answer is 821.”')
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(OUT / "reasoning-tail.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-root", type=Path,
                        default=ROOT / ".cache/analyses/v1_6-quantization-tail-reasoning/selected-streams")
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    assert sha(ROOT / "runner/lib/extraction.py") == EXTRACTOR_SHA
    assert sha(args.tokenizer) == TOKENIZER_SHA
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = [analyze(*case, args.stream_root, tokenizer) for case in CASES]
    result = {"scope": "Nine purposively selected winning trajectories from five paired v1.6 seeds",
              "extractor_sha256": EXTRACTOR_SHA, "tokenizer_sha256": TOKENIZER_SHA,
              "timing": "Official-clock SSE UTC timestamps; exact IDs count generated output only",
              "selection_limit": "Not an all-question census or counterfactual policy replay",
              "cases": rows}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(rows)
    for r in rows:
        c = r["landmarks"]["recognized_candidate"]
        earlier = {k: round(v["gap_to_recognized_s"], 3) for k, v in r["landmarks"].items() if k != "recognized_candidate"}
        print(r["label"], r["question_index"], c["generated_tokens_through_character_end"],
              c["observed_event"]["official_elapsed_s"], earlier)


if __name__ == "__main__":
    main()
