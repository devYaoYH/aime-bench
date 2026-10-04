"""Matched five-seed GPU validation of the packaged canonical v1 runner."""

import argparse
import ast
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess

import httpx

from runner.cli import parse_args
from runner.integrity import verify_core
from runner.lib.common import ROOT, atomic_json, utc_now
from runner.lib.gpu import assert_gpu_idle
from runner.lib.prewarm import reset_cache
from runner.lib.services import Services, attempt_lock, ensure_free, ready
from runner.run import run

REFERENCE = "frozen-core-prompt-five-seeds-20261004T005416Z"
SEEDS = [20261011, 20261012, 20261013, 20261014, 20261015]
CONTROLS = (
    "benchmark_year",
    "benchmark_role",
    "model",
    "model_profile",
    "parallelism",
    "rollouts",
    "first_pass_max_tokens",
    "max_tokens",
    "max_attempts_per_question",
    "max_rounds",
    "schedule",
    "no_continuation",
    "temperature",
    "top_p",
    "disable_thinking",
    "grader_cost",
    "target_correct",
    "questions",
    "benchmark",
    "buffer_traces",
    "no_gpu_telemetry",
    "no_overhead_profile",
    "skip_benchmark_prewarm",
    "prewarm_max_tokens",
    "system_prompt_sha256",
    "runtime_package_versions",
    "question_timeout",
    "startup_timeout",
)


def load(path):
    return json.loads(path.read_text())


def reference_trials():
    summary = load(ROOT / "runs/experiments" / REFERENCE / "summary.json")
    if [r["sampling_seed"] for r in summary["trials"]] != SEEDS:
        raise RuntimeError("Historical seed sequence changed")
    return summary["trials"]


def check_controls(args, old):
    differences = [key for key in CONTROLS if getattr(args, key) != old[key]]
    if differences:
        raise RuntimeError("Unmatched historical controls: " + ", ".join(differences))


def grader_semantics():
    """Permit vendored whitespace changes only; preserve all grading ASTs."""
    evidence = {}
    for name in ("grade.py", "parser.py", "parse_manual.py"):
        old = ast.dump(ast.parse((ROOT / "grader/grader_core" / name).read_text()))
        new = ast.dump(
            ast.parse((ROOT / "runner/grader/grader_core" / name).read_text())
        )
        if old != new:
            raise RuntimeError("Grader behavior changed: " + name)
        evidence[name] = hashlib.sha256(new.encode()).hexdigest()
    return evidence


def measure(output, reference, core_hash):
    config, summary = load(output / "config.json"), load(output / "summary.json")
    old = load(ROOT / "attempts" / reference["attempt_id"] / "config.json")
    check_controls(type("Config", (), config)(), old)
    dataset_fields = (
        "id",
        "year",
        "role",
        "revision",
        "rows",
        "prompt_sha256",
        "grader_sha256",
    )
    if any(
        config["dataset_provenance"][k] != old["dataset_provenance"][k]
        for k in dataset_fields
    ):
        raise RuntimeError("Dataset provenance differs from historical trial")
    initial_differences, continuation_errors = [], []
    questions = [load(p) for p in sorted(output.glob("trace/*/question.json"))]
    records = [r for q in questions for r in q["rollouts"]]
    for index in range(1, 31):
        relative = Path("trace") / f"{index:02d}" / "rollout-01/request.json"
        path = output / relative
        if not path.exists() or load(path) != load(
            ROOT / "attempts" / reference["attempt_id"] / relative
        ):
            initial_differences.append(index)
    for q in questions:
        folder = output / "trace" / f'{q["problem_idx"]:02d}'
        for r in q["rollouts"]:
            parent = r["continuation_of_rollout"]
            if parent is None:
                continue
            prior = load(folder / f"rollout-{parent:02d}" / "tokens.json")
            request = load(folder / f'rollout-{r["rollout"]:02d}' / "request.json")
            prefix = prior["prompt_token_ids"] + prior["output_token_ids"]
            if (
                not prior["complete"]
                or request.get("prompt") != prefix
                or request["max_tokens"]
                != min(config["max_tokens"], config["max_context_tokens"] - len(prefix))
            ):
                continuation_errors.append(
                    {"problem_idx": q["problem_idx"], "rollout": r["rollout"]}
                )
    solved = [q for q in questions if q.get("first_solved")]
    times = sorted(q["first_solved"]["first_solved_elapsed_s"] for q in solved)
    checks = [
        json.loads(line)
        for path in output.glob("trace/*/verification.jsonl")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    completed = [r for r in checks if type(r.get("result", {}).get("verdict")) is bool]
    cap_valid = len(questions) == 30 and all(len(q["rollouts"]) <= 4 for q in questions)
    correct_valid = all(q["winner"]["result"]["verdict"] is True for q in solved)
    target_valid = (
        len(times) >= 18 and abs(times[17] - summary["time_to_target_s"]) < 1e-6
    )
    valid = (
        summary["status"] == "completed"
        and summary["target_reached"]
        and cap_valid
        and correct_valid
        and target_valid
        and not initial_differences
        and not continuation_errors
        and config["runner_id"] == "runner_core_v1"
        and not config["git_dirty"]
        and config["core_manifest_sha256"] == core_hash
        and config["max_context_tokens"] == 65536
    )
    return {
        "sampling_seed": config["seed"],
        "attempt_id": output.name,
        "valid": valid,
        "status": summary["status"],
        "target_reached": summary["target_reached"],
        "time_to_target_s": summary["time_to_target_s"],
        "reference_attempt_id": reference["attempt_id"],
        "reference_time_to_target_s": reference["time_to_target_s"],
        "delta_s": (
            summary["time_to_target_s"] - reference["time_to_target_s"]
            if summary["time_to_target_s"] is not None
            else None
        ),
        "solved": len(solved),
        "solved_indices": [q["problem_idx"] for q in solved],
        "generation_requests": len(records),
        "continuation_requests": sum(
            r["continuation_of_rollout"] is not None for r in records
        ),
        "initial_payload_mismatches": initial_differences,
        "continuation_errors": continuation_errors,
        "request_cap_valid": cap_valid,
        "first_solved_target_valid": target_valid,
        "completed_checks": len(completed),
        "wrong_checks": sum(not r["result"]["verdict"] for r in completed),
        "candidate_formats": dict(Counter(r["kind"] for r in completed)),
        "performance": summary["performance"],
        "grader_timeline": summary["grader_timeline"],
        "official_latency_s": summary["official_latency_s"],
        "trace_storage": summary["trace_storage"],
    }


def aggregate(rows):
    valid = [
        r for r in rows if r.get("valid") and r.get("time_to_target_s") is not None
    ]
    times = [r["time_to_target_s"] for r in valid]
    old = [r["reference_time_to_target_s"] for r in valid]
    complete = len(rows) == 5 and len(valid) == 5
    relative = statistics.median(times) / statistics.median(old) - 1 if times else None
    return {
        "valid_trials": len(valid),
        "all_five_valid": complete,
        "median_time_to_target_s": statistics.median(times) if times else None,
        "min_time_to_target_s": min(times) if times else None,
        "max_time_to_target_s": max(times) if times else None,
        "reference_median_s": statistics.median(old) if old else None,
        "median_relative_change": relative,
        "paired_mean_delta_s": (
            statistics.mean(r["delta_s"] for r in valid) if valid else None
        ),
        "within_practical_10pct_median_band": complete and abs(relative) <= 0.10,
        "interpretation": "Five paired historical seeds; a 10% median band is a practical screen, not statistical equivalence. Keep failures/unmet targets and stochastic output differences.",
    }


def trial_args(seed, grader_python):
    return parse_args(
        [
            "--benchmark",
            "--skip-benchmark-prewarm",
            "--reuse-server",
            "--benchmark-year",
            "2025",
            "--seed",
            str(seed),
            "--grader-python",
            grader_python,
        ]
    )


async def execute(options):
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError(
            "Use a clean pinned worktree; preserve primary-checkout changes"
        )
    references = reference_trials()
    args = trial_args(SEEDS[0], options.grader_python)
    check_controls(
        args, load(ROOT / "attempts" / references[0]["attempt_id"] / "config.json")
    )
    core_hash = verify_core(ROOT)
    profile = Path(args.models_dir).expanduser() / args.model / args.model_profile
    profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
    old_batch = load(ROOT / "runs/experiments" / REFERENCE / "config.json")
    if profile_hash != old_batch["profile_sha256"]:
        raise RuntimeError("Model profile differs from original five-seed batch")
    grader_hashes = grader_semantics()
    batch = ROOT / "runs/experiments" / options.batch
    batch.mkdir(parents=True, exist_ok=False)
    config = {
        "driver": "runner.extensions.validation.refactor_v1",
        "runner_module": "runner",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "core_manifest_sha256": core_hash,
        "started_at_utc": utc_now(),
        "seed_sequence": SEEDS,
        "reference_batch": REFERENCE,
        "reference_source_commit": old_batch["source_commit"],
        "profile_path": str(profile),
        "profile_sha256": profile_hash,
        "system_prompt_sha256": args.system_prompt_sha256,
        "runtime_package_versions": args.runtime_package_versions,
        "grader_semantic_hashes": grader_hashes,
        "shared_server": True,
        "fresh_grader_each_trial": True,
        "prefix_cache_reset_each_trial": True,
        "trial_wall_timeout_s": options.trial_timeout,
        "practical_median_band": 0.10,
        "scoring": "Score all five original seeds including first; retain failures, timeouts and unmet targets; no replacement seeds or settling trial.",
        "scope": "AIME 2025 development data, matched historical hardware/runtime/profile/prompt/sampling; optional profiling disabled.",
    }
    atomic_json(batch / "config.json", config)
    result = {"status": "initializing", "seed_sequence": SEEDS, "trials": []}
    atomic_json(batch / "summary.json", result)
    services = Services()
    try:
        ensure_free(args.vllm_port)
        ensure_free(args.grader_port)
        assert_gpu_idle(args.gpu_device)
        command = [
            args.vllm_binary,
            "serve",
            "--config",
            str(profile),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.vllm_port),
            "--enable-prompt-tokens-details",
        ]
        config["server_command"] = command
        config["server_env_override"] = {"VLLM_SERVER_DEV_MODE": "1"}
        atomic_json(batch / "config.json", config)
        server = services.launch(
            command, batch / "vllm.log", env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
        )
        async with httpx.AsyncClient(trust_env=False) as client:
            await ready(
                client, args.vllm_url + "/v1/models", args.startup_timeout, server
            )
            result["status"] = "running"
            atomic_json(batch / "summary.json", result)
            for number, reference in enumerate(references, 1):
                seed = reference["sampling_seed"]
                before = set((ROOT / "attempts").iterdir())
                row = {
                    "trial": number,
                    "sampling_seed": seed,
                    "valid": False,
                    "time_to_target_s": None,
                    "reference_attempt_id": reference["attempt_id"],
                    "reference_time_to_target_s": reference["time_to_target_s"],
                }
                try:
                    row["cache_reset"] = await reset_cache(client, args.vllm_url)
                    print(f"REFACTOR TRIAL {number}/5 seed={seed}", flush=True)
                    async with asyncio.timeout(options.trial_timeout):
                        output = await run(trial_args(seed, options.grader_python))
                    row.update(measure(output, reference, core_hash))
                except Exception as error:
                    row.update(
                        status="failed", error=f"{type(error).__name__}: {error}"
                    )
                    created = set((ROOT / "attempts").iterdir()) - before
                    if len(created) == 1:
                        row["attempt_id"] = created.pop().name
                result["trials"].append(row)
                atomic_json(batch / "summary.json", result)
                print("REFACTOR RESULT " + json.dumps(row), flush=True)
            result.update(
                status="complete",
                completed_at_utc=utc_now(),
                **aggregate(result["trials"]),
            )
    except BaseException as error:
        result.update(
            status=(
                "interrupted" if isinstance(error, asyncio.CancelledError) else "failed"
            ),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        await services.close()
        atomic_json(batch / "summary.json", result)
        print(
            "REFACTOR BATCH "
            + json.dumps({k: v for k, v in result.items() if k != "trials"}),
            flush=True,
        )
    return batch


async def entry(options):
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM, asyncio.current_task().cancel
    )
    await execute(options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        default="canonical-v1-refactor-five-seeds-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--grader-python", default=str(ROOT / "runner/grader/.venv/bin/python")
    )
    parser.add_argument(
        "--trial-timeout",
        type=float,
        default=600,
        help="Total per-trial wall safety timeout, including initialization; partial runs are retained",
    )
    options = parser.parse_args(argv)
    if (
        Path(options.batch).name != options.batch
        or options.batch in (".", "..")
        or options.trial_timeout <= 0
    ):
        parser.error("Use a plain batch name and positive trial timeout")
    with attempt_lock():
        asyncio.run(entry(options))


if __name__ == "__main__":
    main()
