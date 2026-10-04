"""Expand the scored AWQ seed into five paired seeds using fresh servers."""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import signal
import statistics
import subprocess
from types import SimpleNamespace

from runner.extensions.validation import awq_v1_6 as awq
from runner.extensions.v1_6.integrity import verify_core
from runner.lib.common import ROOT, atomic_json, utc_now
from runner.lib.services import attempt_lock

PILOT = "core-v1_6-awq-marlin-20261004T232028Z"
SEEDS = awq.harness.baseline.SEEDS


def plan(pilot=PILOT):
    folder = ROOT / "runs/experiments" / pilot
    config, result = (awq.harness.baseline.load(folder / n)
                      for n in ("config.json", "summary.json"))
    if result["seed"] != SEEDS[0] or config["seed"] != SEEDS[0]:
        raise RuntimeError("Pilot must use the first declared comparison seed")
    reference = awq.reference_trial(SEEDS[0])
    measured = awq.audit(ROOT / "attempts" / result["attempt_id"], reference, verify_core(ROOT))
    if not measured["valid"] or not result["valid"]:
        raise RuntimeError("Existing AWQ pilot failed its unchanged-policy audit")
    return {"pilot_batch": pilot, "pilot_source_commit": config["source_commit"],
            "seeds": list(SEEDS), "remaining_seeds": list(SEEDS[1:]),
            "pilot_trial": {**measured, "sampling_seed": SEEDS[0], "batch": pilot}}


def aggregate(trials):
    times = [t["time_to_target_s"] for t in trials
             if t.get("valid") and t.get("target_reached")]
    return {"declared_trials": 5, "recorded_trials": len(trials),
            "valid_target_trials": len(times),
            "mean_s": statistics.mean(times) if times else None,
            "median_s": statistics.median(times) if times else None,
            "min_s": min(times) if times else None,
            "max_s": max(times) if times else None,
            "sample_sd_s": statistics.stdev(times) if len(times) > 1 else None}


async def execute(options):
    prepared = plan(options.pilot)
    folder = ROOT / "runs/experiments" / options.batch
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / "config.json", {
        "driver": "runner.extensions.validation.awq_v1_6_batch",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "core_manifest_sha256": verify_core(ROOT), "started_at_utc": utc_now(),
        "model": awq.MODEL, "profile": awq.PROFILE,
        "reference_batch": awq.REFERENCE,
        "plan": {k: v for k, v in prepared.items() if k != "pilot_trial"},
        "server_lifetime_note": "Five fresh AWQ server lifetimes, each prefix-reset and cheaply warmed. NVFP4 control used one shared server lifetime across five seeds. Official solve time excludes startup/warmup; residual compilation/cache state may still affect latency.",
        "trial_wall_timeout_s": options.trial_timeout,
        "replacement_trials": False,
    })
    trials = [prepared["pilot_trial"]]
    try:
        for seed in prepared["remaining_seeds"]:
            atomic_json(folder / "summary.json", {"status": "running", "trials": trials, **aggregate(trials)})
            batch = options.batch + f"-seed{seed}"
            try:
                await awq.harness.execute(SimpleNamespace(
                    seed=seed, batch=batch, grader_python=options.grader_python,
                    trial_timeout=options.trial_timeout,
                ), deployment=awq)
            except Exception as error:
                saved = ROOT / "runs/experiments" / batch / "summary.json"
                trial = awq.harness.baseline.load(saved) if saved.exists() else {
                    "status": "failed", "valid": False, "target_reached": False,
                    "time_to_target_s": None,
                }
                trial["error"] = f"{type(error).__name__}: {error}"
            else:
                trial = awq.harness.baseline.load(ROOT / "runs/experiments" / batch / "summary.json")
            trials.append({**trial, "sampling_seed": seed, "batch": batch})
    finally:
        state = aggregate(trials)
        status = "completed" if state["valid_target_trials"] == 5 else "partial_or_failed"
        atomic_json(folder / "summary.json", {"status": status, "trials": trials,
                    **state, "updated_at_utc": utc_now()})
        print(f"AWQ five-seed status: {status}; {state}", flush=True)


async def entry(options):
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    await execute(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default=PILOT)
    parser.add_argument("--batch", default="core-v1_6-awq-five-seeds-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--grader-python", default=str(ROOT / "runner/grader/.venv/bin/python"))
    parser.add_argument("--trial-timeout", type=float, default=600)
    options = parser.parse_args()
    if any(Path(v).name != v or v in (".", "..") for v in (options.pilot, options.batch)) or options.trial_timeout <= 0:
        parser.error("Use plain batch names and a positive timeout")
    with attempt_lock():
        asyncio.run(entry(options))


if __name__ == "__main__":
    main()
