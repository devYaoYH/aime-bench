"""Serve canonical attempts and the fixed exploratory archive in local viewers.

Open / for canonical src.attempt outputs under attempts/, or /exploratory for
original Qwen trajectories, self-consistency, and Jev records from the fixed run.
It serves src/viewer assets and selected artifact APIs on 127.0.0.1:8765 by default.
Detailed views require ignored raw run files, which are absent from a fresh clone.
The server does not make inference calls or expose .env. Start it with:
    python -m src.viewer_server
"""
from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlparse


from src.common import ROOT
from src.attempt_viewer import AttemptStore
RUNS = ROOT / "runs"
VIEWER = ROOT / "src" / "viewer"
ATTEMPTS = AttemptStore(ROOT / "attempts", ROOT / "data" / "aime_2025_problems.jsonl")
SAFE_RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
QUESTION = re.compile(r"^/api/runs/([^/]+)/questions/(\d{1,3})$")
JEV_REVIEW = re.compile(r"^/api/runs/([^/]+)/jev-review/(\d{1,3})$")
JEV_PREFIX = re.compile(r"^/api/runs/([^/]+)/jev-prefix/(\d{1,3})$")
JEV_CALIBRATION = re.compile(r"^/api/runs/([^/]+)/jev-calibration/(\d{1,3})/([1-8])$")
SAMPLE = re.compile(r"^/api/runs/([^/]+)/attempts/(\d{1,3})/([1-8])$")
OVERVIEW = re.compile(r"^/api/runs/([^/]+)/overview$")
REASONING_PLOT = re.compile(r"^/api/runs/([^/]+)/reasoning-tokens\.svg$")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def run_dir(name: str) -> Path:
    if not SAFE_RUN.fullmatch(name) or name in {".", ".."}:
        raise ValueError("Invalid run identifier")
    path = RUNS / name
    if not path.is_dir():
        raise FileNotFoundError(name)
    return path


def seconds_after(timestamp: str | None, start: str) -> float | None:
    if not timestamp:
        return None
    try:
        return round((datetime.fromisoformat(timestamp) - datetime.fromisoformat(start)).total_seconds(), 3)
    except ValueError:
        return None


def build_overview(name: str) -> dict:
    path = run_dir(name)
    summary = read_json(path / "summary.json")
    config = read_json(path / "config.json")
    review_path = path / "jev_review" / "summary.json"
    review = read_json(review_path) if review_path.is_file() else None
    review_by_index = {int(item["problem_idx"]): item for item in review["ranked"]} if review else {}
    prefix_path = path / "jev_prefix_1500" / "summary.json"
    prefix_review = read_json(prefix_path) if prefix_path.is_file() else None
    prefix_by_index = {int(item["problem_idx"]): item for item in prefix_review["ranked"]} if prefix_review else {}
    calibration_path = path / "jev_calibration" / "summary.json"
    calibration = read_json(calibration_path) if calibration_path.is_file() else None
    samples_path = path / "self_consistency" / "summary.json"
    samples = read_json(samples_path) if samples_path.is_file() else None
    samples_by_index = {int(item["problem_idx"]): item for item in samples["results"]} if samples else {}
    analysis_dir = path / "analysis"
    checkpoints = read_json(analysis_dir / "checkpoint_summary.json") if (analysis_dir / "checkpoint_summary.json").is_file() else None
    budgets = read_json(analysis_dir / "budget_summary.json") if (analysis_dir / "budget_summary.json").is_file() else None
    start = summary["first_inference_request_at_utc"]
    items = []
    for result in summary["results"]:
        index = int(result["problem_idx"])
        trace = read_json(path / "questions" / f"{index:02d}.json")
        attempts = trace.get("api_attempts") or []
        first = attempts[0] if attempts else {}
        last = attempts[-1] if attempts else {}
        response = trace.get("response") or {}
        choices = response.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        final = message.get("content") or ""
        reasoning = message.get("reasoning") or ""
        final_line = final.strip().splitlines()[-1].strip() if final.strip() else ""
        requested_format_valid = bool(re.fullmatch(r"Answer:\s*\d{1,3}", final_line))
        usage = trace.get("usage") or {}
        items.append({
            "problem_idx": index,
            "problem": trace.get("problem") or "",
            "candidate": trace.get("candidate"),
            "gold_answer": trace.get("gold_answer"),
            "correct": trace.get("correct"),
            "format_valid": requested_format_valid,
            "answer_parseable": trace.get("candidate") is not None,
            "finish_reason": trace.get("finish_reason"),
            "api_latency_s": trace.get("total_api_latency_s"),
            "generation_latency_s": trace.get("generation_latency_s"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "final_chars": len(final),
            "reasoning_chars": len(reasoning),
            "cost": usage.get("cost"),
            "attempt_count": len(attempts),
            "request_start_s": seconds_after(first.get("started_at_utc"), start),
            "request_end_s": seconds_after(last.get("finished_at_utc"), start),
            "grading_end_s": seconds_after(trace.get("grading_completed_at_utc"), start),
            "http_status": last.get("http_status"),
            "jev_review": review_by_index.get(index),
            "prefix_review": prefix_by_index.get(index),
            "self_consistency": samples_by_index.get(index),
        })
    return {"run_id": name, "summary": summary, "config": config, "questions": items, "jev_review": review, "prefix_review": prefix_review, "jev_calibration": calibration, "self_consistency": samples, "reasoning_checkpoints": checkpoints, "budget_analysis": budgets}


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: object, status: int = 200) -> None:
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/" and parse_qs(parsed.query).get("run") == ["20260930-155212"]:
            self.send_response(303)
            self.send_header("Location", "/exploratory?" + parsed.query)
            self.end_headers()
            return
        assets = {
            "/": ("attempts/index.html", "text/html; charset=utf-8"),
            "/attempts": ("attempts/index.html", "text/html; charset=utf-8"),
            "/attempts/viewer.css": ("attempts/viewer.css", "text/css; charset=utf-8"),
            "/attempts/viewer.js": ("attempts/viewer.js", "text/javascript; charset=utf-8"),
            "/exploratory": ("exploratory/index.html", "text/html; charset=utf-8"),
            "/exploratory/": ("exploratory/index.html", "text/html; charset=utf-8"),
            "/exploratory/viewer.css": ("exploratory/viewer.css", "text/css; charset=utf-8"),
            "/exploratory/viewer.js": ("exploratory/viewer.js", "text/javascript; charset=utf-8"),
        }
        if path in assets:
            filename, content_type = assets[path]
            self.send_bytes((VIEWER / filename).read_bytes(), content_type)
            return
        try:
            if path == "/api/attempts":
                self.send_json(ATTEMPTS.list())
                return
            if match := re.fullmatch(r"/api/attempts/([^/]+)/overview", path):
                self.send_json(ATTEMPTS.overview(match[1]))
                return
            if match := re.fullmatch(r"/api/attempts/([^/]+)/gpu", path):
                self.send_json(ATTEMPTS.gpu(match[1]))
                return
            if match := re.fullmatch(r"/api/attempts/([^/]+)/questions/(\d+)", path):
                self.send_json(ATTEMPTS.question(match[1], int(match[2])))
                return
            if match := re.fullmatch(r"/api/attempts/([^/]+)/questions/(\d+)/rollouts/(\d+)", path):
                self.send_json(ATTEMPTS.rollout(match[1], int(match[2]), int(match[3])))
                return
            if match := re.fullmatch(r"/api/attempts/([^/]+)/files/(.+)", path):
                artifact = ATTEMPTS.artifact(match[1], match[2])
                self.send_bytes(artifact.read_bytes(), "application/json; charset=utf-8" if artifact.suffix == ".json" else "application/x-ndjson; charset=utf-8")
                return
            if match := OVERVIEW.fullmatch(path):
                self.send_json(build_overview(match.group(1)))
                return
            if match := REASONING_PLOT.fullmatch(path):
                folder = run_dir(match.group(1))
                self.send_bytes((folder / "analysis" / "reasoning_tokens_by_question.svg").read_bytes(), "image/svg+xml; charset=utf-8")
                return
            if match := QUESTION.fullmatch(path):
                folder = run_dir(match.group(1))
                index = int(match.group(2))
                self.send_bytes((folder / "questions" / f"{index:02d}.json").read_bytes(), "application/json; charset=utf-8")
                return
            if match := JEV_REVIEW.fullmatch(path):
                folder = run_dir(match.group(1))
                index = int(match.group(2))
                self.send_bytes((folder / "jev_review" / f"{index:02d}.json").read_bytes(), "application/json; charset=utf-8")
                return
            if match := JEV_PREFIX.fullmatch(path):
                folder = run_dir(match.group(1))
                index = int(match.group(2))
                self.send_bytes((folder / "jev_prefix_1500" / f"{index:02d}.json").read_bytes(), "application/json; charset=utf-8")
                return
            if match := JEV_CALIBRATION.fullmatch(path):
                folder = run_dir(match.group(1))
                index, number = int(match.group(2)), int(match.group(3))
                self.send_bytes((folder / "jev_calibration" / f"{index:02d}-{number:02d}.json").read_bytes(), "application/json; charset=utf-8")
                return
            if match := SAMPLE.fullmatch(path):
                folder = run_dir(match.group(1))
                index, number = int(match.group(2)), int(match.group(3))
                file = (folder / "questions" / f"{index:02d}.json") if number == 1 else (folder / "self_consistency" / "questions" / f"{index:02d}" / f"{number:02d}.json")
                self.send_bytes(file.read_bytes(), "application/json; charset=utf-8")
                return
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 404)
            return
        self.send_json({"error": "Not found"}, 404)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AIME viewer: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
