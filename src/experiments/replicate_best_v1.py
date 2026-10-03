"""Sequential matched best-run repeats on one owned, warmed vLLM server."""

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

from src.common import ROOT, utc_now
from src.attempt_runners import speedrun_v2 as runner
from src.attempt_runners import speedrun_v4 as dynamic_runner
from src.attempt_runners._ports_v1 import ensure_free
from src.attempt_runners._runtime_v1 import Services, attempt_lock, ready

MANIFEST = ROOT / "configs/experiments/vibe-bf16-best-30x1-benchmark-v2.json"
REFERENCE = "20261003T211557.382358Z"


def controls(model, benchmark=True, policy="reference"):
    manifest = (MANIFEST if policy == "reference" else
                ROOT / "configs/experiments/vibe-bf16-dynamic30-8k-v4.json")
    argv = json.loads(manifest.read_text())["argv"]
    argv[argv.index("--model") + 1] = model
    if not benchmark:
        argv.remove("--benchmark")
    return argv + ["--reuse-server"]


def compare_initial_requests(output, model):
    differences = []
    ttft = []
    for index in range(1, 31):
        relative = Path("trace") / f"{index:02d}" / "rollout-01"
        expected = json.loads((ROOT / "attempts" / REFERENCE / relative / "request.json").read_text())
        actual = json.loads((output / relative / "request.json").read_text())
        expected["model"] = model
        if expected != actual:
            differences.append(index)
        question = json.loads((output / "trace" / f"{index:02d}" / "question.json").read_text())
        value = question["rollouts"][0]["ttft_s"]
        if value is not None:
            ttft.append(value)
    return {"initial_request_count": 30, "different_questions": differences,
            "initial_ttft_median_s": statistics.median(ttft) if ttft else None,
            "initial_ttft_count": len(ttft),
            "model_override": model != "WeiboAI/VibeThinker-3B"}


async def reset_cache(client, url, attempts=75):
    """Refuse to inherit previous trial prefixes; never reset running requests."""
    for _ in range(attempts):
        response = await client.post(url + "/reset_prefix_cache", timeout=10)
        response.raise_for_status()
        if response.json().get("success") is True:
            return {"success": True, "at_utc": utc_now()}
        await asyncio.sleep(0.2)
    raise RuntimeError("Prefix cache still held by outstanding requests; stopping repeats")


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


async def execute(options):
    module = runner if options.policy == "reference" else dynamic_runner
    argv = controls(options.model, benchmark=not options.profiled_control, policy=options.policy)
    args = module.parse_args(argv)
    batch = ROOT / "runs/experiments" / options.batch
    batch.mkdir(parents=True, exist_ok=False)
    profile = Path(args.models_dir).expanduser() / args.model / args.model_profile
    config = {"driver": "replicate_best_v1", "started_at_utc": utc_now(),
              "reference_attempt": REFERENCE, "trials_requested": options.trials,
              "near_reference_threshold_s": options.threshold_s,
              "argv": argv, "model": options.model, "profiled_control": options.profiled_control,
              "policy": options.policy, "runner_module": module.__name__,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "profile_path": str(profile), "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
              "server_reused": True, "prefix_cache_reset_between_trials": True,
              "server_env_override": {"VLLM_SERVER_DEV_MODE": "1"},
              "sampling": "Original fixed seeds in every trial; no successful trials selected or discarded"}
    save(batch / "config.json", config)
    services = Services()
    result = {"status": "initializing", "trials": [], "reference_attempt": REFERENCE}
    save(batch / "summary.json", result)
    try:
        ensure_free(args.vllm_port)
        ensure_free(args.grader_port)
        runner.assert_gpu_idle(args.gpu_device)
        env = dict(os.environ, VLLM_SERVER_DEV_MODE="1")
        command = [args.vllm_binary, "serve", "--config", str(profile), "--host", "127.0.0.1",
                   "--port", str(args.vllm_port), "--enable-prompt-tokens-details"]
        config["server_command"] = command
        save(batch / "config.json", config)
        server = services.launch(command, batch / "vllm.log", env=env)
        async with httpx.AsyncClient(trust_env=False) as client:
            await ready(client, args.vllm_url + "/v1/models", args.startup_timeout, server)
            result["status"] = "running"
            save(batch / "summary.json", result)
            for trial in range(1, options.trials + 1):
                cache_reset = await reset_cache(client, args.vllm_url)
                print(f"REPLICATION TRIAL {trial}/{options.trials} model={options.model}", flush=True)
                output = await module.run(module.parse_args(argv))
                summary = json.loads((output / "summary.json").read_text())
                comparison = compare_initial_requests(output, options.model)
                row = {"trial": trial, "attempt_id": output.name, "cache_reset": cache_reset,
                       "target_reached": summary["target_reached"], "time_to_target_s": summary["time_to_target_s"],
                       "performance": summary["performance"], "grader_timeline": summary["grader_timeline"],
                       "matched_requests": comparison}
                result["trials"].append(row)
                save(batch / "summary.json", result)
                print("REPLICATION RESULT " + json.dumps(row), flush=True)
                if comparison["different_questions"]:
                    raise RuntimeError("Initial request controls differ from the best reference")
            times = [r["time_to_target_s"] for r in result["trials"] if r["target_reached"]]
            result.update(status="complete", completed_at_utc=utc_now(),
                          target_reached_trials=len(times),
                          near_reference_trials=sum(t <= options.threshold_s for t in times),
                          median_time_to_target_s=statistics.median(times) if times else None,
                          min_time_to_target_s=min(times) if times else None,
                          max_time_to_target_s=max(times) if times else None)
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        await services.close()
        save(batch / "summary.json", result)
        print("REPLICATION BATCH " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["WeiboAI/VibeThinker-3B", "r0b0tlab/VibeThinker-3B-NVFP4"], default="WeiboAI/VibeThinker-3B")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threshold-s", type=float, default=75)
    parser.add_argument("--profiled-control", action="store_true", help="Restore original profiling and immediate trace writes for the matched control")
    parser.add_argument("--policy", choices=["reference", "dynamic30"], default="reference",
                        help="Reference barrier control, or reallocate solved slots while retaining only 30 active requests")
    parser.add_argument("--batch", default="best-replication-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    options = parser.parse_args()
    if options.trials < 1 or options.threshold_s <= 0 or Path(options.batch).name != options.batch:
        parser.error("Positive trials/threshold and a plain batch directory name required")
    with attempt_lock():
        async def entry():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await execute(options)
        asyncio.run(entry())


if __name__ == "__main__":
    main()
