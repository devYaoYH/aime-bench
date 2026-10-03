"""Replay the best attempt's immutable source and managed-server configuration."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess

from src.common import ROOT, utc_now
from src.attempt_runners._runtime_v1 import attempt_lock
from src.experiments.replicate_best_v1 import controls, compare_initial_requests, save, REFERENCE


def historical_argv(grader_python):
    argv = controls("WeiboAI/VibeThinker-3B", benchmark=False)
    argv.remove("--reuse-server")
    return argv + ["--grader-python", str(grader_python)]


async def execute(options):
    reference = json.loads((ROOT / "attempts" / REFERENCE / "config.json").read_text())
    source = reference["git_commit"]
    batch = ROOT / "runs/experiments" / options.batch
    batch.mkdir(parents=True, exist_ok=False)
    checkout = ROOT.parent / (ROOT.name + "-best-source-" + options.batch)
    subprocess.run(["git", "worktree", "add", "--detach", str(checkout), source], cwd=ROOT, check=True)
    command = [str(Path("~/.venvs/vllm/bin/python").expanduser()), "-u", "-m",
               "src.attempt_runners.speedrun_v1", *historical_argv(ROOT / "grader/.venv/bin/python")]
    config = {"driver": "replay_best_source_v1", "reference_attempt": REFERENCE,
              "source_commit": source, "checkout": str(checkout), "command": command,
              "trials_requested": options.trials, "started_at_utc": utc_now(),
              "driver_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "server_reused": False, "policy": "Original v1 source, original profiling and immediate writes"}
    save(batch / "config.json", config)
    result = {"status": "running", "trials": [], "source_commit": source}
    process = None
    try:
        for trial in range(1, options.trials + 1):
            before = set((checkout / "attempts").glob("*"))
            print(f"HISTORICAL REPLAY {trial}/{options.trials} source={source}", flush=True)
            with (batch / f"trial-{trial:02d}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=checkout, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = await asyncio.to_thread(process.wait)
            after = set((checkout / "attempts").glob("*")) - before
            if len(after) != 1:
                raise RuntimeError(f"Expected one historical attempt, found {len(after)}")
            output = after.pop()
            if code:
                raise RuntimeError(f"Historical runner exited {code}; inspect trial log")
            summary = json.loads((output / "summary.json").read_text())
            comparison = compare_initial_requests(output, "WeiboAI/VibeThinker-3B")
            recorded = json.loads((output / "config.json").read_text())
            if recorded["git_commit"] != source or comparison["different_questions"]:
                raise RuntimeError("Historical source or initial payloads do not match reference")
            row = {"trial": trial, "attempt_id": output.name, "output": str(output),
                   "time_to_target_s": summary["time_to_target_s"], "target_reached": summary["target_reached"],
                   "performance": summary["performance"], "grader_timeline": summary["grader_timeline"],
                   "matched_requests": comparison}
            result["trials"].append(row)
            save(batch / "summary.json", result)
            print("HISTORICAL RESULT " + json.dumps(row), flush=True)
        result.update(status="complete", completed_at_utc=utc_now())
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), 30)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await asyncio.to_thread(process.wait)
        save(batch / "summary.json", result)
    print("HISTORICAL BATCH " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--batch", default="bf16-exact-source-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    options = parser.parse_args()
    if options.trials < 1 or Path(options.batch).name != options.batch:
        parser.error("Positive trial count and a plain batch directory name required")
    with attempt_lock():
        async def entry():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await execute(options)
        asyncio.run(entry())


if __name__ == "__main__":
    main()
