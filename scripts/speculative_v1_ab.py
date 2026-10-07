"""Paired serving A/B around unchanged canonical v1; run on the idle A100.

Each cell owns a fresh server, runs the canonical benchmark, polls public engine
counters externally, and retains failures. Source is developed locally and pulled
on the node. Full streams/logs remain ignored; compact analysis is publishable.
"""
import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL = "r0b0tlab/VibeThinker-3B-NVFP4"
PROFILES = {"A": "vllm-flashinfer.yaml", "B": "vllm-spec-qwen05-k3.yaml"}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def get(path):
    with urlopen("http://127.0.0.1:8000" + path, timeout=3) as response:
        return response.read().decode()


def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def counters(text):
    values = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        key, value = line.rsplit(None, 1)
        name = key.split("{", 1)[0].removeprefix("vllm:")
        if name in ("generation_tokens_total", "num_requests_running", "num_requests_waiting",
                    "num_preemptions_total", "kv_cache_usage_perc") or name.startswith("spec_decode"):
            # Histograms have bucket/count/sum suffixes; keep each name distinct.
            values[name] = values.get(name, 0) + float(value)
    return values


def preflight():
    from transformers import AutoConfig, AutoTokenizer
    target = Path.home() / "models" / MODEL
    draft = Path.home() / "models/Qwen/Qwen2.5-Coder-0.5B"
    t, d = [AutoTokenizer.from_pretrained(p).get_vocab() for p in (target, draft)]
    mismatches = [token for token, idx in d.items() if t.get(token) != idx]
    if mismatches:
        raise RuntimeError(f"Draft token IDs differ: {mismatches[:10]}")
    tc, dc = [AutoConfig.from_pretrained(p) for p in (target, draft)]
    if tc.vocab_size != dc.vocab_size:
        raise RuntimeError("LM-head vocab sizes differ")
    import vllm
    return {"vllm_version": vllm.__version__, "shared_token_ids_match": True,
            "target_only_tokens": {k: v for k, v in t.items() if k not in d},
            "lm_head_vocab_size": tc.vocab_size, "draft_native_context": dc.max_position_embeddings,
            "draft_revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
            "tokenizer_hashes": [hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                                 for v in (t, d)]}


def decode_analysis(attempt, width=5):
    """Audit exact IDs, then integrate first-to-last arrival exposure in shared windows.

    Speculation can return multiple IDs per SSE chunk. Count IDs, never chunks.
    Exclude the entire first chunk from rate counting to avoid zero-time tokens.
    """
    config = json.loads((attempt / "config.json").read_text())
    origin = datetime.fromisoformat(config["official_started_at_utc"])
    requests, blocks = [], []
    for path in sorted(attempt.glob("trace/*/rollout-*/telemetry.json")):
        record = json.loads(path.read_text())
        saved = json.loads((path.parent / "tokens.json").read_text())
        ids, arrivals = [], []
        began = (datetime.fromisoformat(record["started_at_utc"]) - origin).total_seconds()
        stream = path.parent / "stream.jsonl"
        digest = hashlib.sha256()
        with stream.open("rb") as handle:
            for line in handle:
                digest.update(line)
                event = json.loads(line)
                if event["data"] == "[DONE]":
                    continue
                payload = json.loads(event["data"])
                chunk = []
                for choice in payload.get("choices", []):
                    if choice.get("index", 0) == 0:
                        chunk.extend(choice.get("token_ids") or [])
                if chunk:
                    ids.extend(chunk)
                    arrivals.append((began + event["elapsed_s"], len(chunk)))
        if ids != saved["output_token_ids"] or len(ids) != record["generated_token_ids_count"]:
            raise RuntimeError(f"Token evidence mismatch: {path}")
        row = {"request": str(path.parent.relative_to(attempt)), "tokens": len(ids),
               "chunks": len(arrivals), "status": record["status"], "ttft_s": record["ttft_s"],
               "stream_sha256": digest.hexdigest(), "exact_ids_audited": True}
        if len(arrivals) < 2 or arrivals[-1][0] <= arrivals[0][0]:
            row["decode_tps"] = None
            requests.append(row)
            continue
        first, last = arrivals[0][0], arrivals[-1][0]
        row.update(first=first, last=last, decode_tps=(len(ids)-arrivals[0][1])/(last-first))
        requests.append(row)
        times = [x[0] for x in arrivals]
        cumulative = []
        total = 0
        for _, count in arrivals:
            total += count
            cumulative.append(total)
        prompt = record["prompt_token_ids_count"] or 0
        for k in range(int(first // width), int(last // width) + 1):
            a, b = max(first, k*width), min(last, (k+1)*width)
            if b <= a:
                continue
            n = sum(count for t, count in arrivals[1:] if a < t <= b)
            points = [a] + [t for t in times if a < t < b] + [b]
            context_area = sum((right-left)*(prompt+cumulative[bisect_right(times, left)-1])
                               for left, right in zip(points, points[1:]))
            blocks.append({"window": k, "request": row["request"], "tokens": n,
                           "seconds": b-a, "context_token_seconds": context_area})
    windows = []
    for k in sorted({b["window"] for b in blocks}):
        cells = [b for b in blocks if b["window"] == k]
        exposure = sum(b["seconds"] for b in cells)
        tokens = sum(b["tokens"] for b in cells)
        # Fixed-width windows avoid labelling a short final fragment low concurrency.
        complete = (k+1)*width <= max((r.get("last", 0) for r in requests), default=0)
        windows.append({"start_s": k*width, "end_s": (k+1)*width, "complete": complete,
                        "tokens": tokens, "decode_request_seconds": exposure,
                        "mean_active_decoders": exposure/width, "per_active_tps": tokens/exposure,
                        "mean_context_tokens": sum(b["context_token_seconds"] for b in cells)/exposure})
    bins = {}
    for label, lower, upper in (("0-4", 0, 4), ("4-8", 4, 8), ("8-16", 8, 16), ("16-30", 16, 31)):
        cells = [w for w in windows if w["complete"] and lower < w["mean_active_decoders"] <= upper]
        exposure = sum(w["decode_request_seconds"] for w in cells)
        bins[label] = {"windows": len(cells), "decode_request_seconds": exposure,
                       "tokens": sum(w["tokens"] for w in cells),
                       "per_active_tps": sum(w["tokens"] for w in cells)/exposure if exposure else None,
                       "mean_context_tokens": sum(w["mean_context_tokens"]*w["decode_request_seconds"]
                                                  for w in cells)/exposure if exposure else None}
    return {"requests": requests, "windows": windows, "concurrency_bins": bins,
            "method": "5s client arrival windows; exact token IDs audited; first SSE token chunk excluded; rates weighted by decoding request-seconds; incomplete final window excluded from bins. Mean concurrency and logical context are observational and trajectory-dependent."}


def cell(arm, seed, output, args):
    folder = output / f"{seed}-{arm}"
    folder.mkdir()
    server = runner = None
    profile = ROOT / "configs/vllm" / MODEL / PROFILES[arm]
    command = [str(Path.home()/".venvs/vllm/bin/vllm"), "serve", "--config", str(profile)]
    runner_command = [args.python, "-m", "runner", "--model", MODEL, "--models-dir",
                      str(ROOT/"configs/vllm"), "--model-profile", PROFILES[arm],
                      "--reuse-server", "--seed", str(seed), "--startup-timeout", "900"]
    row = {"arm": arm, "seed": seed, "started_utc": utc(), "server_command": command,
           "runner_command": runner_command, "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest()}
    save(folder/"config.json", row)
    try:
        for port in (8000, 8077):
            with socket.socket() as sock:
                # Same check as canonical ensure_free: TIME_WAIT is not a listener.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
        gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if gpu.strip():
            raise RuntimeError(f"GPU occupied by {gpu.strip()}")
        before = set((ROOT/"attempts").glob("*"))
        with (folder/"vllm.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"})
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}")
                try:
                    models = json.loads(get("/v1/models"))
                    if MODEL not in [m["id"] for m in models["data"]]:
                        raise RuntimeError("Unexpected served model")
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup timed out")
                    time.sleep(1)
            save(folder/"metrics_before.json", counters(get("/metrics")))
            with (folder/"runner.log").open("w") as rlog, (folder/"metrics.jsonl").open("w") as metrics:
                runner = subprocess.Popen(runner_command, cwd=ROOT, stdout=rlog,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                while runner.poll() is None:
                    try:
                        metrics.write(json.dumps({"utc": utc(), "metrics": counters(get("/metrics"))})+"\n")
                        metrics.flush()
                    except OSError as exc:
                        metrics.write(json.dumps({"utc": utc(), "error": str(exc)})+"\n")
                    time.sleep(1)
            save(folder/"metrics_after.json", counters(get("/metrics")))
            row["runner_exit"] = runner.returncode
        new = set((ROOT/"attempts").glob("*")) - before
        if len(new) != 1:
            raise RuntimeError(f"Expected one attempt, got {len(new)}")
        attempt = new.pop()
        row["attempt_id"] = attempt.name
        summary = json.loads((attempt/"summary.json").read_text())
        row.update(status=summary["status"], target_reached=summary["target_reached"],
                   time_to_target_s=summary["time_to_target_s"], performance=summary["performance"])
        if runner.returncode == 0:
            analysis = decode_analysis(attempt)
            save(folder/"analysis.json", analysis)
            row["concurrency_bins"] = analysis["concurrency_bins"]
        # Counter deltas include warmup; official intervals are retained separately.
        initial = json.loads((folder/"metrics_before.json").read_text())
        final = json.loads((folder/"metrics_after.json").read_text())
        row["counter_deltas_including_warmup"] = {k: v-initial.get(k, 0) for k, v in final.items() if k.endswith("_total")}
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        stop(runner)
        stop(server)
        row["finished_utc"] = utc()
        save(folder/"summary.json", row)
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261011, 20261012, 20261013])
    parser.add_argument("--python", default=str(Path.home()/".venvs/vllm/bin/python"))
    args = parser.parse_args()
    plan = [{"seed": seed, "arms": ["A", "B"] if i % 2 == 0 else ["B", "A"]}
            for i, seed in enumerate(args.seeds)]
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    output = ROOT/"runs/experiments"/datetime.now(timezone.utc).strftime("speculative-v1-ab-%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True)
    config = {"started_utc": utc(), "plan": plan, "preflight": preflight(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
              "hardware": subprocess.check_output(["nvidia-smi"], text=True),
              "policy": "unchanged canonical v1 prompt_adherence benchmark; each cell owns a fresh server"}
    save(output/"config.json", config)
    print(f"Batch artifacts: {output}", flush=True)
    rows = []
    for pair in plan:
        for arm in pair["arms"]:
            row = cell(arm, pair["seed"], output, args)
            rows.append(row)
            save(output/"summary.json", {"config": config, "cells": rows})
            if row["status"] == "failed":
                # Preserve failure; do not silently substitute another profile/seed.
                raise RuntimeError(f"Cell failed: {pair['seed']}-{arm}; inspect {output}")


if __name__ == "__main__":
    main()
