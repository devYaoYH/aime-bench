"""Post-freeze measurements from client records and grader verdicts only.

This module never opens dataset files or imports a local answer-key grader.
"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics

MILESTONES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)


def read_json(path):
    return json.loads(Path(path).read_text())


def events(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def baseline_accuracy(folder, expected_samples=120, rollouts=4):
    """Map each unique oracle verdict to every sample; missing evidence stays unknown."""
    folder = Path(folder)
    samples, questions, pairs = [], [], set()
    valid = True
    for path in sorted(folder.glob("trace/*/question.json")):
        question = read_json(path)
        index = question["problem_idx"]
        verdicts = {}
        for event in events(path.parent / "verification.jsonl"):
            verdict = event.get("result", {}).get("verdict")
            if type(verdict) is bool:
                candidate = event["candidate"]
                if candidate in verdicts:
                    raise ValueError("Repeated grading of a question/answer pair")
                verdicts[candidate] = verdict
                pairs.add((index, candidate))
        counts = Counter()
        correct = capped = no_answer = 0
        for record in question["rollouts"]:
            candidate = record.get("extracted_candidate")
            terminal = record["status"] == "completed" and record.get("finish_reason") in ("stop", "length")
            if record.get("finish_reason") != "stop" and candidate is not None:
                raise ValueError("A capped/failed sample was extracted or graded")
            verdict = verdicts.get(candidate) if candidate is not None else None
            valid &= terminal and (candidate is None or type(verdict) is bool)
            if candidate is not None:
                counts[candidate] += 1
            correct += verdict is True
            capped += record.get("finish_reason") == "length"
            no_answer += candidate is None
            samples.append({"problem_idx": index, "rollout": record["rollout"],
                            "candidate": candidate, "verdict": verdict, "status": record["status"],
                            "finish_reason": record.get("finish_reason"),
                            "no_answer": candidate is None, "token_capped": record.get("finish_reason") == "length",
                            "generation_latency_s": record.get("generation_latency_s"),
                            "completion_tokens": (record.get("usage") or {}).get("completion_tokens")})
        leaders = [answer for answer, count in counts.items() if count == max(counts.values())] if counts else []
        vote = leaders[0] if len(leaders) == 1 else None
        vote_correct = vote is not None and verdicts.get(vote) is True
        valid &= len(question["rollouts"]) == rollouts
        questions.append({"problem_idx": index, "correct_out_of_4": correct,
                          "samples": len(question["rollouts"]), "pass4": correct > 0,
                          "vote_answer": vote, "vote_correct": vote_correct,
                          "vote_tied": len(leaders) > 1, "no_answer": no_answer,
                          "token_capped": capped, "answer_counts": dict(counts)})
    valid = bool(valid and len(samples) == expected_samples and len(questions) * rollouts == expected_samples and expected_samples > 0)
    return {"valid": valid, "expected_samples": expected_samples, "observed_samples": len(samples),
            "questions": len(questions), "unique_grader_checks": len(pairs),
            "correct_samples": sum(s["verdict"] is True for s in samples),
            "pass1": sum(s["verdict"] is True for s in samples) / expected_samples if valid else None,
            "pass4_correct_questions": sum(q["pass4"] for q in questions) if valid else None,
            "majority_vote_correct_questions": sum(q["vote_correct"] for q in questions) if valid else None,
            "vote_rule": "Unique most frequent extracted integer among four samples; missing answers abstain; ties or no votes count incorrect",
            "token_capped_count": sum(s["token_capped"] for s in samples),
            "token_capped_rate": sum(s["token_capped"] for s in samples) / expected_samples if valid else None,
            "no_answer_count": sum(s["no_answer"] for s in samples),
            "no_answer_rate": sum(s["no_answer"] for s in samples) / expected_samples if valid else None,
            "per_question": questions, "samples": samples,
            "scope": "Accuracy comes only from oracle verdicts, not an answer key. Failed/incomplete batches retain evidence without a scored accuracy."}


def marginal_trial(folder):
    """Distinct first-correct milestones, verified against saved oracle responses."""
    folder = Path(folder)
    summary = read_json(folder / "summary.json")
    config = read_json(folder / "config.json")
    records = [read_json(path) for path in sorted(folder.glob("trace/*/question.json"))]
    solved = sorted((q for q in records if q.get("first_solved")),
                    key=lambda q: q["first_solved"]["first_solved_elapsed_s"])
    seen = set()
    first = []
    for question in solved:
        event, result = question["first_solved"], question["winner"]["result"]
        assert result["verdict"] is True and question["problem_idx"] not in seen
        assert event["grader_query_id"] == result["query_id"]
        seen.add(question["problem_idx"])
        first.append({"rank": len(first) + 1, **event})
    checks = [event for path in folder.glob("trace/*/verification.jsonl") for event in events(path)]
    completed = [e for e in checks if type(e.get("result", {}).get("verdict")) is bool]
    return {"attempt_id": folder.name, "status": summary["status"],
            "solved": len(seen), "first_correct": first,
            "milestones_s": {str(n): first[n-1]["first_solved_elapsed_s"] if len(first) >= n else None for n in MILESTONES},
            "total_client_checks": len(checks), "completed_client_checks": len(completed),
            "wrong_client_checks": sum(e["result"]["verdict"] is False for e in completed),
            "grader_timeline": summary.get("grader_timeline"),
            "requests_per_question": {str(q["problem_idx"]): len(q["rollouts"]) for q in records},
            "questions_never_solved": [i for i in config.get("question_indices", list(range(1, 31))) if i not in seen],
            "official_latency_s": summary.get("official_latency_s"),
            "initialization_and_attempt_latency_s": summary.get("initialization_and_attempt_latency_s")}


def aggregate_milestones(rows):
    result = []
    for n in MILESTONES:
        values = [r["measurement"]["milestones_s"][str(n)] for r in rows
                  if r.get("identity_valid", True) and r.get("measurement") and r["measurement"]["milestones_s"][str(n)] is not None]
        result.append({"milestone": n, "reached": len(values), "declared_trials": len(rows), "identity_invalid_trials": sum(r.get("identity_valid") is False for r in rows),
                       "median_s": statistics.median(values) if values else None,
                       "min_s": min(values) if values else None, "max_s": max(values) if values else None})
    return result
