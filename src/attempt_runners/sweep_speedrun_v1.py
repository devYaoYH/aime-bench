"""Deferred speedrun v1 sweeps: print a plan by default; --execute runs cells.

Cells run sequentially with fresh managed inference/grader services and warmup.
No remote command or CUDA operation occurs while planning.
"""

from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shlex
import signal
import subprocess

from src.common import ROOT, atomic_json, utc_now
from src.attempt_runners import speedrun_v1 as runner
from src.attempt_runners._runtime_v1 import attempt_lock

KEYS = ("parallelism", "rollouts", "temperature", "top_p", "seed")


def build_plan(config):
    if config["schema_version"] != 1 or set(config["grid"]) != set(KEYS):
        raise ValueError(
            "Expected schema 1 and a grid of parallelism, rollouts, temperature, top_p, seed"
        )
    grid = config["grid"]
    if any(not isinstance(grid[k], list) or not grid[k] for k in KEYS):
        raise ValueError("Every sweep dimension must be a nonempty list")
    parameters = [
        dict(zip(KEYS, values), schedule="eager")
        for values in itertools.product(*(grid[key] for key in KEYS))
    ]
    parameters = config.get("controls", []) + parameters
    excluded = config.get("exclude", [])
    if not isinstance(excluded, list) or any(
        not isinstance(pattern, dict)
        or not pattern
        or not set(pattern) <= set(KEYS) | {"schedule"}
        for pattern in excluded
    ):
        raise ValueError(
            "Exclusions must be nonempty parameter mappings with known keys"
        )
    parameters = [
        values
        for values in parameters
        if not any(
            all(values.get(key) == value for key, value in pattern.items())
            for pattern in excluded
        )
    ]
    if not parameters:
        raise ValueError("Sweep exclusions removed every cell")
    order = config.get("order", "listed")
    if order not in ("listed", "parallelism_desc"):
        raise ValueError("Unknown sweep order")
    if order == "parallelism_desc":
        parameters.sort(
            key=lambda values: (values["parallelism"], values["rollouts"]), reverse=True
        )
    cells = []
    for index, values in enumerate(parameters, 1):
        if set(values) != set(KEYS) | {"schedule"}:
            raise ValueError("A control must specify all grid fields and schedule")
        argv = list(config["base_args"])
        for key, value in values.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        args = runner.parse_args(argv)
        if args.reuse_server:
            raise ValueError("Sweep v1 requires fresh managed servers per cell")
        if args.strategy != runner.RUNNER_ID:
            raise ValueError("Unexpected runner strategy")
        question_count = len(runner.load_questions(args.questions))
        if args.target_correct > question_count:
            raise ValueError("Target exceeds selected question count")
        cells.append(
            {
                "cell_id": f"cell-{index:02d}",
                "parameters": values,
                "initial_concurrent_requests": min(args.parallelism, question_count)
                * args.rollouts,
                "argv": argv,
                "command": "~/.venvs/vllm/bin/python "
                + shlex.join(["-m", "src.attempt_runners.speedrun_v1", *argv]),
            }
        )
    return {
        "schema_version": 1,
        "runner_id": runner.RUNNER_ID,
        "cells": cells,
        "excluded_parameters": excluded,
        "order": order,
        "execution": "sequential; fresh grader and inference server; full-batch warmup before official timer",
        "scope": "time to target excludes model startup and warmup; generation requests capped at four per question",
        "ranking": "Only completed cells reaching the target have a time_to_target_s and rank",
    }


def score(cell, output, error=None):
    summary = (
        json.loads((output / "summary.json").read_text())
        if output and (output / "summary.json").exists()
        else {}
    )
    valid = (
        summary.get("status") == "completed"
        and summary.get("target_reached") is True
        and error is None
    )
    return {
        "cell_id": cell["cell_id"],
        "parameters": cell["parameters"],
        "attempt_id": output.name if output else None,
        "status": summary.get("status", "failed"),
        "error": error or summary.get("error"),
        "target_reached": summary.get("target_reached", False),
        "solved": summary.get("solved", 0),
        "time_to_target_s": summary.get("time_to_target_s") if valid else None,
        "official_latency_s": summary.get("official_latency_s"),
        "grader_timeline": summary.get("grader_timeline"),
        "performance": summary.get("performance"),
        "overhead": summary.get("overhead"),
    }


def ranked(rows):
    valid = sorted(
        (r for r in rows if r["time_to_target_s"] is not None),
        key=lambda r: r["time_to_target_s"],
    )
    order = {r["cell_id"]: i for i, r in enumerate(valid, 1)}
    return [{**row, "rank": order.get(row["cell_id"])} for row in rows]


async def execute(plan, output, *, continue_on_error=False):
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "config.json",
        {
            **plan,
            "started_at_utc": utc_now(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )
    rows, status = [], "running"
    try:
        for cell in plan["cells"]:
            print(f"Starting {cell['cell_id']}: {cell['parameters']}", flush=True)
            before = set((ROOT / "attempts").glob("*"))
            attempt, error = None, None
            try:
                attempt = await runner.run(runner.parse_args(cell["argv"]))
            except asyncio.CancelledError:
                status = "interrupted"
                error = "Sweep interrupted"
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if attempt is None:
                    created = sorted(set((ROOT / "attempts").glob("*")) - before)
                    attempt = created[0] if len(created) == 1 else None
                rows.append(score(cell, attempt, error))
                atomic_json(
                    output / "summary.json",
                    {
                        "status": status,
                        "updated_at_utc": utc_now(),
                        "cells": ranked(rows),
                    },
                )
            if error and not continue_on_error:
                status = "failed"
                break
        else:
            status = (
                "completed_with_errors"
                if any(r["error"] or r["status"] != "completed" for r in rows)
                else "completed"
            )
    finally:
        atomic_json(
            output / "summary.json",
            {"status": status, "finished_at_utc": utc_now(), "cells": ranked(rows)},
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/sweeps/vibe-speedrun-v1.json"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch GPU experiments; otherwise only print the plan",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Execution artifact directory; generated timestamp by default",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(json.loads(args.config.read_text()))
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    destination = args.output or ROOT / "runs/speedrun_sweeps" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S.%fZ")
    with attempt_lock():

        async def entry():
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
            await execute(plan, destination, continue_on_error=args.continue_on_error)

        asyncio.run(entry())


if __name__ == "__main__":
    main()
