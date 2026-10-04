"""Five matched-seed measurements of v1.6 coverage-barrier/pool policy."""

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

import yaml
from runner.extensions.v1_6.cli import parse_args, prepare_problems, metadata, RUNNER_ID
from runner.extensions.v1_6.integrity import verify_core
from runner.extensions.v1_6.policy import run_speedrun
from runner.lib.common import ROOT, atomic_json, utc_now
from runner.lib.gpu import assert_gpu_idle
from runner.lib.prewarm import reset_cache
from runner.lib.services import Services, attempt_lock, ensure_free, ready
from runner.lib.policy_runtime import run_policy


async def run(args):
    return await run_policy(
        args,
        policy=run_speedrun,
        prepare_problems=prepare_problems,
        verify=verify_core,
        runner_id=RUNNER_ID,
        runner_module="runner.extensions.v1_6",
        metadata_builder=metadata,
    )


REFERENCE = "frozen-core-prompt-five-seeds-20261004T005416Z"
SEEDS = [20261011, 20261012, 20261013, 20261014, 20261015]
POLICY_CHANGES = {
    "model_profile",
    "max_tokens",
    "max_attempts_per_question",
    "max_rounds",
    "schedule",
}
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
    differences = [
        key
        for key in CONTROLS
        if key not in POLICY_CHANGES and getattr(args, key) != old[key]
    ]
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
    fields = (
        "id",
        "year",
        "role",
        "revision",
        "rows",
        "prompt_sha256",
        "grader_sha256",
    )
    if any(
        config["dataset_provenance"][k] != old["dataset_provenance"][k] for k in fields
    ):
        raise RuntimeError("Dataset provenance changed")
    questions = [load(p) for p in sorted(output.glob("trace/*/question.json"))]
    records = [r for q in questions for r in q["rollouts"]]
    allocation = load(output / "allocation.json")
    admissions = allocation["admissions"]
    mismatches, errors = [], []
    for index in range(1, 31):
        relative = Path("trace") / f"{index:02d}" / "rollout-01/request.json"
        if not (output / relative).exists() or load(output / relative) != load(
            ROOT / "attempts" / reference["attempt_id"] / relative
        ):
            mismatches.append(index)
    for q in questions:
        folder = output / "trace" / f"{q['problem_idx']:02d}"
        samples = {}
        for r in q["rollouts"]:
            samples.setdefault(r["fresh_sample"], []).append(r)
            request = load(folder / f"rollout-{r['rollout']:02d}" / "request.json")
            expected_seed = (
                config["seed"]
                + q["problem_idx"] * 4
                + r["fresh_sample"]
                + (r["segment"] - 1) * config["segment_seed_stride"]
            )
            valid = (
                request["seed"] == expected_seed
                and r["trajectory_generated_tokens"] <= config["max_rollout_tokens"]
            )
            if r["segment"] == 1:
                valid &= r["continuation_of_rollout"] is None and request[
                    "max_tokens"
                ] == min(
                    config["first_pass_max_tokens"],
                    config["max_rollout_tokens"],
                    config["max_context_tokens"]
                    - config["served_prompt_tokens"][str(q["problem_idx"])],
                )
            else:
                parent = r["continuation_of_rollout"]
                prior = load(folder / f"rollout-{parent:02d}" / "tokens.json")
                prior_record = next(t for t in q["rollouts"] if t["rollout"] == parent)
                prefix = prior["prompt_token_ids"] + prior["output_token_ids"]
                valid &= (
                    r["segment"] == 2
                    and prior_record["fresh_sample"] == r["fresh_sample"]
                    and prior_record["segment"] == 1
                    and prior["complete"]
                    and request.get("prompt") == prefix
                    and request["max_tokens"]
                    == min(
                        config["max_rollout_tokens"] - len(prior["output_token_ids"]),
                        config["max_context_tokens"] - len(prefix),
                    )
                )
            if not valid:
                errors.append(dict(problem_idx=q["problem_idx"], rollout=r["rollout"]))
        if len(samples) > 4 or any(len(v) > 2 for v in samples.values()):
            errors.append(
                dict(problem_idx=q["problem_idx"], error="fresh/continuation cap")
            )
    release = allocation["barrier_release"]
    coverage = [a for a in admissions if a["phase"] == "coverage"]
    pool = [a for a in admissions if a["phase"] == "pool"]
    barrier_valid = (
        len(coverage) == 30
        and [a["problem_idx"] for a in coverage] == list(range(1, 31))
        and all(a["fresh_sample"] == 1 and a["segment"] == 1 for a in coverage)
        and (
            not pool
            or release is not None
            and release["active_requests"] == 0
            and all(a["elapsed_s"] >= release["elapsed_s"] for a in pool)
        )
    )
    if release is not None:
        # Includes generation retirement and the initial verifier drain/settlement.
        barrier_valid &= all(
            s["coverage_settled"]
            and s["coverage_settled_elapsed_s"] <= release["elapsed_s"]
            and s["initial_generation_retired_elapsed_s"] is not None
            and s["initial_generation_retired_elapsed_s"] <= release["elapsed_s"]
            for s in allocation["questions"].values()
        )
    cap_valid = (
        allocation["peak_active_requests"] <= 30
        and all(a["active_requests"] <= 30 for a in admissions)
        and all(
            s["fresh"] <= 4 and s["requests"] <= 8
            for s in allocation["questions"].values()
        )
        and len(admissions) == len(records)
    )
    solved = [q for q in questions if q.get("first_solved")]
    times = sorted(q["first_solved"]["first_solved_elapsed_s"] for q in solved)
    target_valid = (
        len(times) >= 18
        and abs(times[17] - summary["time_to_target_s"]) < 1e-6
        and all(q["winner"]["result"]["verdict"] is True for q in solved)
    )
    checks = [
        json.loads(line)
        for p in output.glob("trace/*/verification.jsonl")
        for line in p.read_text().splitlines()
        if line.strip()
    ]
    completed = [r for r in checks if type(r.get("result", {}).get("verdict")) is bool]
    valid = (
        summary["status"] == "completed"
        and summary["target_reached"]
        and len(questions) == 30
        and barrier_valid
        and cap_valid
        and target_valid
        and not mismatches
        and not errors
        and config["runner_id"] == RUNNER_ID
        and not config["git_dirty"]
        and config["core_manifest_sha256"] == core_hash
        and config["max_context_tokens"] == 65536
        and config["initial_coverage_barrier"] is True
    )
    return dict(
        sampling_seed=config["seed"],
        attempt_id=output.name,
        valid=valid,
        status=summary["status"],
        target_reached=summary["target_reached"],
        time_to_target_s=summary["time_to_target_s"],
        reference_attempt_id=reference["attempt_id"],
        reference_time_to_target_s=reference["time_to_target_s"],
        delta_s=summary["time_to_target_s"] - reference["time_to_target_s"]
        if summary["time_to_target_s"] is not None
        else None,
        solved=len(solved),
        solved_indices=[q["problem_idx"] for q in solved],
        generation_requests=len(records),
        fresh_samples=sum(r["segment"] == 1 for r in records),
        continuation_requests=sum(r["segment"] == 2 for r in records),
        initial_payload_mismatches=mismatches,
        continuation_errors=errors,
        request_cap_valid=cap_valid,
        initial_barrier_valid=barrier_valid,
        barrier_release=release,
        peak_active_requests=allocation["peak_active_requests"],
        first_solved_target_valid=target_valid,
        completed_checks=len(completed),
        wrong_checks=sum(not r["result"]["verdict"] for r in completed),
        candidate_formats=dict(Counter(r["kind"] for r in completed)),
        performance=summary["performance"],
        grader_timeline=summary["grader_timeline"],
        official_latency_s=summary["official_latency_s"],
        trace_storage=summary["trace_storage"],
    )


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
        "interpretation": "Five paired historical seeds; policy changes include the post-coverage slot pool, fresh-only cap and long continuation. Historical controls are not simultaneous randomized trials; retain failures/unmet targets.",
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


def validate_profile(profile, args, old_batch):
    profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
    committed_profile = ROOT / "configs/vllm" / args.model / args.model_profile
    if profile_hash != hashlib.sha256(committed_profile.read_bytes()).hexdigest():
        raise RuntimeError("Remote profile differs from committed v1.6 configuration")
    old_profile = (
        Path(args.models_dir).expanduser()
        / args.model
        / Path(old_batch["profile_path"]).name
    )
    if (
        hashlib.sha256(old_profile.read_bytes()).hexdigest()
        != old_batch["profile_sha256"]
    ):
        raise RuntimeError("Historical model profile changed")
    old_values, new_values = (
        yaml.safe_load(old_profile.read_text()),
        yaml.safe_load(profile.read_text()),
    )
    if {k: v for k, v in old_values.items() if k != "override-generation-config"} != {
        k: v for k, v in new_values.items() if k != "override-generation-config"
    }:
        raise RuntimeError("Unexpected model profile change")
    if json.loads(new_values["override-generation-config"]) != {
        "max_new_tokens": 65536
    }:
        raise RuntimeError("Wrong long-continuation server ceiling")


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
    validate_profile(profile, args, old_batch)
    grader_hashes = grader_semantics()
    batch = ROOT / "runs/experiments" / options.batch
    batch.mkdir(parents=True, exist_ok=False)
    config = {
        "driver": "runner.extensions.validation.v1_6_batch",
        "runner_module": "runner.extensions.v1_6",
        "policy_changes": sorted(POLICY_CHANGES),
        "initial_coverage_barrier": True,
        "pool_slots": 30,
        "fresh_cap": 4,
        "output_budget": 65536,
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
        "scope": "AIME 2025 development data, matched historical hardware/runtime/prompt/sampling; profile output ceiling raised from16K to64K; original profile preserved; optional profiling disabled.",
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
                    print(f"V1.6 TRIAL {number}/5 seed={seed}", flush=True)
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
                print("V1.6 RESULT " + json.dumps(row), flush=True)
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
            "V1.6 BATCH "
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
        default="core-v1_6-barrier-five-seeds-"
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
