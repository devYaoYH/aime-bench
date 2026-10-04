"""Run the frozen runner for five predeclared seeds on one owned server."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess

import httpx
import yaml

from src.common import ROOT, utc_now
from runner_final import run as runner
from runner_final._runtime import Services, attempt_lock, ready
from runner_final._ports import ensure_free
from runner_final.prewarm import reset_cache

PROTOCOL = Path(__file__).with_name("five_seeds.json")


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def compare_initial_requests(output, model, seed, reference):
    differences, observed = [], 0
    for index in range(1, 31):
        relative = Path("trace") / f"{index:02d}" / "rollout-01" / "request.json"
        expected = json.loads((ROOT / "attempts" / reference / relative).read_text())
        expected["model"] = model
        expected["seed"] += seed - 20261003
        actual_path = output / relative
        if not actual_path.exists():
            differences.append(index)
            continue
        observed += 1
        if expected != json.loads(actual_path.read_text()):
            differences.append(index)
    return {"initial_requests": observed, "different_questions": differences,
            "sampling_seed": seed, "reference_attempt": reference}


def score(output, seed, reference):
    summary = json.loads((output / "summary.json").read_text())
    questions = [json.loads(p.read_text()) for p in sorted(output.glob("trace/*/question.json"))]
    solved = [q for q in questions if q.get("first_solved")]
    comparison = compare_initial_requests(output, summary_model(output), seed, reference)
    caps_valid = all(len(q["rollouts"]) <= 4 for q in questions)
    valid = (summary["status"] == "completed" and summary["target_reached"]
             and len({q["problem_idx"] for q in solved}) >= 18
             and caps_valid and not comparison["different_questions"])
    return {"sampling_seed": seed, "attempt_id": output.name, "status": summary["status"],
            "target_reached": summary["target_reached"], "valid": valid,
            "time_to_target_s": summary["time_to_target_s"] if valid else None,
            "recorded_time_to_target_s": summary["time_to_target_s"],
            "initialization_and_attempt_latency_s": summary["initialization_and_attempt_latency_s"],
            "benchmark_prewarm": summary.get("benchmark_prewarm"),
            "performance": summary["performance"], "grader_timeline": summary["grader_timeline"],
            "matched_requests": comparison, "request_cap_valid": caps_valid,
            "distinct_solved": len({q["problem_idx"] for q in solved})}


def summary_model(output):
    return json.loads((output / "config.json").read_text())["model"]


def aggregate(rows, threshold, count=5):
    times = [r["time_to_target_s"] for r in rows if r.get("valid") and r["time_to_target_s"] is not None]
    passed = sum(t <= threshold for t in times)
    return {"successful_trials": len(times), "at_or_below_reference_trials": passed,
            "confirmed": len(rows) == count and passed == count,
            "median_time_to_target_s": statistics.median(times) if times else None,
            "min_time_to_target_s": min(times) if times else None,
            "max_time_to_target_s": max(times) if times else None,
            "statistics_scope": "Successful valid trials only; all failures/unmet targets retained in trials"}


async def execute(options):
    protocol = json.loads(PROTOCOL.read_text())
    seeds = protocol["seeds"]
    argv = protocol["argv"] + ["--reuse-server"]
    args = runner.parse_args(argv)
    batch = ROOT / "runs/experiments" / options.batch
    batch.mkdir(parents=True, exist_ok=False)
    profile = Path(args.models_dir).expanduser() / args.model / args.model_profile
    if yaml.safe_load(profile.read_text()) != yaml.safe_load(Path(__file__).with_name("vllm-flashinfer.yaml").read_text()):
        raise RuntimeError("Remote launch profile differs from frozen final configuration")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Validation requires a clean tracked source checkout")
    config = {"driver": "runner_final.validate", "started_at_utc": utc_now(),
              "protocol": protocol, "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "argv": argv, "seed_sequence": seeds, "profile_path": str(profile),
              "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
              "server_reused": True, "prefix_cache_reset_between_trials": True,
              "server_env_override": {"VLLM_SERVER_DEV_MODE": "1"}}
    save(batch / "config.json", config)
    services = Services()
    result = {"status": "initializing", "trials": [], "seed_sequence": seeds}
    save(batch / "summary.json", result)
    try:
        ensure_free(args.vllm_port)
        ensure_free(args.grader_port)
        runner.assert_gpu_idle(args.gpu_device)
        command = [args.vllm_binary, "serve", "--config", str(profile), "--host", "127.0.0.1",
                   "--port", str(args.vllm_port), "--enable-prompt-tokens-details"]
        config["server_command"] = command
        save(batch / "config.json", config)
        server = services.launch(command, batch / "vllm.log", env=dict(os.environ, VLLM_SERVER_DEV_MODE="1"))
        async with httpx.AsyncClient(trust_env=False) as client:
            await ready(client, args.vllm_url + "/v1/models", args.startup_timeout, server)
            result["status"] = "running"
            save(batch / "summary.json", result)
            for trial, seed in enumerate(seeds, 1):
                before = set((ROOT / "attempts").iterdir())
                row = {"trial": trial, "sampling_seed": seed, "valid": False, "time_to_target_s": None}
                try:
                    cache = await reset_cache(client, args.vllm_url)
                    print(f"FINAL TRIAL {trial}/5 seed={seed}", flush=True)
                    output = await runner.run(runner.parse_args(argv + ["--seed", str(seed)]))
                    row.update(score(output, seed, protocol["reference_attempt"]), cache_reset=cache)
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                    created = set((ROOT / "attempts").iterdir()) - before
                    if len(created) == 1:
                        row["attempt_id"] = created.pop().name
                    print(f"FINAL TRIAL FAILURE {trial}: {row['error']}", flush=True)
                result["trials"].append(row)
                save(batch / "summary.json", result)
                print("FINAL RESULT " + json.dumps({k: v for k, v in row.items() if k != "benchmark_prewarm"}), flush=True)
            result.update(status="complete", completed_at_utc=utc_now(),
                          **aggregate(result["trials"], protocol["reference_time_to_target_s"]))
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        await services.close()
        save(batch / "summary.json", result)
        print("FINAL BATCH " + json.dumps({k: v for k, v in result.items() if k != "trials"}), flush=True)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="runner-final-five-seeds-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    options = parser.parse_args(argv)
    if Path(options.batch).name != options.batch or options.batch in (".", ".."):
        parser.error("Use a plain batch directory name")
    with attempt_lock():
        async def entry():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await execute(options)
        asyncio.run(entry())


if __name__ == "__main__":
    main()
