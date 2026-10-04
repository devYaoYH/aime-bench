"""Sequential post-freeze milestone and complete-baseline measurements.

Plan by default. Execute only from clean tested source on the idle remote GPU.
Frozen cores, the historical naive runner and the grader are never modified.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import httpx
import yaml

from src.common import ROOT, utc_now
from runner_final.integrity import verify_core
from runner_final.run_frozen import parse_args as frozen_args
from runner_final.core_v1 import runner as frozen
from runner_final.core_v1._runtime import Services, attempt_lock, ready
from runner_final.core_v1._ports import ensure_free
from runner_final.core_v1.prewarm import reset_cache
from src.attempt_runners import naive_pass4_full_v1 as full
from src.experiments.post_freeze_analysis import marginal_trial, aggregate_milestones, read_json

PROTOCOL = ROOT / "configs/experiments/post-freeze-measurements-v1.json"


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def plan():
    protocol = read_json(PROTOCOL)
    return {**protocol, "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "commands": ["~/.venvs/vllm/bin/python -m src.experiments.post_freeze --execute"],
            "task_a_cli": ["python", "-m", "runner_final.run_frozen", "--preset", protocol["task_a"]["preset"],
                           *protocol["task_a"]["argv"], "--reuse-server", "--seed", "<declared-seed>"],
            "task_b_cli": ["python", "-m", "src.attempt_runners.naive_pass4_full_v1", *protocol["task_b"]["argv"]],
            "expected_runtime": "Task A at most 900 official seconds per seed (75 minutes total), plus setup; Task B provisionally 10–20 minutes including generation and grading."}


async def with_official_deadline(run, args, attempts, seconds, state=None):
    """Cancel gracefully at the official deadline, excluding initialization/warmup.

    The core persists its official UTC timestamp immediately before starting its
    monotonic timer. The watcher polls only during initialization, then sleeps
    once to the deadline; it adds no recurring poll to the solving window.
    """
    before = set(attempts.iterdir())
    task = asyncio.create_task(run(args))
    state = state if state is not None else {}
    state.update(deadline_reached=False, attempt_id=None)

    async def watch():
        while not task.done():
            created = set(attempts.iterdir()) - before
            if len(created) == 1:
                folder = created.pop()
                state["attempt_id"] = folder.name
                path = folder / "config.json"
                if path.exists():
                    timestamp = read_json(path).get("official_started_at_utc")
                    if timestamp:
                        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp.replace("Z", "+00:00"))).total_seconds()
                        state["deadline_armed_at_utc"] = utc_now()
                        await asyncio.sleep(max(0, seconds - elapsed))
                        if not task.done():
                            state["deadline_reached"] = True
                            state["deadline_fired_at_utc"] = utc_now()
                            task.cancel()
                        return
            await asyncio.sleep(.05)

    watcher = asyncio.create_task(watch())
    try:
        output = await task
        state["attempt_id"] = output.name
    except asyncio.CancelledError:
        if not state["deadline_reached"]:
            raise
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)
        created = set(attempts.iterdir()) - before
        if state["attempt_id"] is None and len(created) == 1:
            state["attempt_id"] = created.pop().name
    return state


def matched_initial_payloads(folder, reference):
    different, observed = [], 0
    for index in range(1, 31):
        relative = Path("trace") / f"{index:02d}" / "rollout-01/request.json"
        path = folder / relative
        if not path.exists():
            different.append(index)
            continue
        observed += 1
        if read_json(path) != read_json(ROOT / "attempts" / reference / relative):
            different.append(index)
    return {"observed": observed, "different_question_indices": different,
            "reference_attempt_id": reference}


async def task_a(batch, protocol):
    declared = protocol["task_a"]
    args = frozen_args(["--preset", declared["preset"], *declared["argv"], "--reuse-server"])
    profile = Path(args.models_dir).expanduser() / args.model / args.model_profile
    if yaml.safe_load(profile.read_text()) != yaml.safe_load((ROOT / "runner_final/vllm-flashinfer.yaml").read_text()):
        raise RuntimeError("Task A profile differs from the measured frozen profile")
    services = Services()
    job = {"task": "A", "status": "initializing", "started_at_utc": utc_now(),
           "compute_mode": "unattended background compute", "trials": []}
    save(batch / "task_a.json", job)
    command = [args.vllm_binary, "serve", "--config", str(profile), "--host", "127.0.0.1",
               "--port", str(args.vllm_port), "--enable-prompt-tokens-details"]
    save(batch / "task_a_server.json", {"command": command, "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                                       "server_env": {"VLLM_SERVER_DEV_MODE": "1"}})
    try:
        ensure_free(args.vllm_port)
        ensure_free(args.grader_port)
        frozen.assert_gpu_idle(args.gpu_device)
        server = services.launch(command, batch / "task_a_vllm.log", env=dict(os.environ, VLLM_SERVER_DEV_MODE="1"))
        async with httpx.AsyncClient(trust_env=False) as client:
            await ready(client, args.vllm_url + "/v1/models", args.startup_timeout, server)
            job.update(status="running", server_ready_at_utc=utc_now(),
                       server_startup_latency_s=(datetime.now(timezone.utc)-datetime.fromisoformat(job["started_at_utc"].replace("Z","+00:00"))).total_seconds())
            for number, seed in enumerate(declared["seeds"], 1):
                row = {"trial": number, "seed": seed, "started_at_utc": utc_now(),
                       "compute_mode": "unattended background compute"}
                job["trials"].append(row)
                save(batch / "task_a.json", job)
                print(f"POST-FREEZE A {number}/5 seed={seed}", flush=True)
                try:
                    row["cache_reset"] = await reset_cache(client, args.vllm_url)
                    parsed = frozen_args(["--preset", declared["preset"], *declared["argv"],
                                          "--reuse-server", "--seed", str(seed)])
                    state = await with_official_deadline(frozen.run, parsed, ROOT / "attempts", declared["official_deadline_s"], state=row)
                    row.update(state)
                    if state["attempt_id"]:
                        folder = ROOT / "attempts" / state["attempt_id"]
                        if (folder / "summary.json").exists():
                            row["measurement"] = marginal_trial(folder)
                            config = read_json(folder / "config.json")
                            row["initial_payload_check"] = matched_initial_payloads(folder, declared["reference_attempts"][str(seed)])
                            row["identity_valid"] = (config["core_manifest_sha256"] == verify_core(ROOT)
                                                     and config["system_prompt_sha256"] == declared["system_prompt_sha256"]
                                                     and not config["git_dirty"]
                                                     and config["git_commit"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                                                     and all(n <= 4 for n in row["measurement"]["requests_per_question"].values())
                                                     and not row["initial_payload_check"]["different_question_indices"])
                            row["end_condition"] = ("official_900s_deadline" if state["deadline_reached"] else
                                                    "failed" if state.get("error") or row["measurement"]["status"] == "failed" else
                                                    "all_30_solved" if row["measurement"]["solved"] == 30 else
                                                    "request_budget_exhausted_and_checks_drained")
                except asyncio.CancelledError:
                    row.update(end_condition="interrupted", error="External cancellation; partial attempt retained in attempts/")
                    raise
                except Exception as error:
                    row.update(error=f"{type(error).__name__}: {error}", end_condition="failed")
                finally:
                    row.setdefault("end_condition", "failed")
                    row["finished_at_utc"] = utc_now()
                    save(batch / "task_a.json", job)
                print("POST-FREEZE A RESULT " + json.dumps(row), flush=True)
            job.update(status="complete", milestones=aggregate_milestones(job["trials"]))
    except BaseException as error:
        job.update(status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        await services.close()
        job["finished_at_utc"] = utc_now()
        save(batch / "task_a.json", job)
    return job


async def task_b(batch, protocol):
    job = {"task": "B", "status": "initializing", "seed": protocol["task_b"]["seed"],
           "started_at_utc": utc_now(), "compute_mode": "unattended background compute"}
    save(batch / "task_b.json", job)
    before = set((ROOT / "attempts").iterdir())
    try:
        args = full.parse_args(protocol["task_b"]["argv"])
        profile = Path(args.models_dir).expanduser() / args.model / args.model_profile
        if yaml.safe_load(profile.read_text()) != yaml.safe_load((ROOT / "configs/vllm/WeiboAI/VibeThinker-3B/vllm-baseline-16k.yaml").read_text()):
            raise RuntimeError("Task B requires the declared 16K BF16 95% baseline profile")
        frozen.assert_gpu_idle(args.gpu_device)
        output = await full.run(args)
        job.update(attempt_id=output.name, status="complete", summary=read_json(output / "summary.json"))
        save(batch / "baseline_accuracy.json", job["summary"]["accuracy"])
    except BaseException as error:
        job.update(status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        created = set((ROOT / "attempts").iterdir()) - before
        if len(created) == 1:
            output = created.pop()
            job["attempt_id"] = output.name
            if (output / "summary.json").exists():
                job["summary"] = read_json(output / "summary.json")
                if job["summary"].get("accuracy"):
                    save(batch / "baseline_accuracy.json", job["summary"]["accuracy"])
        job["finished_at_utc"] = utc_now()
        save(batch / "task_b.json", job)
        print("POST-FREEZE B RESULT " + json.dumps({k: v for k, v in job.items() if k != "summary"}), flush=True)
    return job


async def execute(options):
    declared = plan()
    if verify_core(ROOT) != declared["task_a"]["core_manifest_sha256"]:
        raise RuntimeError("Frozen core differs from the declared measured core")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Experiments require clean tracked source")
    batch = ROOT / "results/post_freeze" / options.batch
    batch.mkdir(parents=True, exist_ok=False)
    config = {"protocol": declared, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "started_at_utc": utc_now(), "compute_mode": "unattended background compute",
              "driver": "src.experiments.post_freeze",
              "source_sha256": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in
                                  ("src/experiments/post_freeze.py", "src/experiments/post_freeze_analysis.py", "src/attempt_runners/naive_pass4_full_v1.py")}}
    save(batch / "config.json", config)
    summary = {"status": "running", "tasks": {}}
    save(batch / "summary.json", summary)
    try:
        summary["tasks"]["A"] = await task_a(batch, declared)
        save(batch / "summary.json", summary)
        summary["tasks"]["B"] = await task_b(batch, declared)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["finished_at_utc"] = utc_now()
        save(batch / "summary.json", summary)
        print("POST-FREEZE BATCH " + str(batch), flush=True)
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--batch", default="measurements-v1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    options = parser.parse_args()
    if Path(options.batch).name != options.batch or options.batch in (".", ".."):
        parser.error("Use a plain batch directory name")
    if not options.execute:
        print(json.dumps(plan(), indent=2))
        return
    with attempt_lock():
        async def entry():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await execute(options)
        asyncio.run(entry())


if __name__ == "__main__":
    main()
