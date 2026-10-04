"""Core v1.5: eager fixed-30 allocation with unchanged core v1 sampling and budgets."""

from __future__ import annotations

import argparse
import asyncio
from copy import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import yaml

from src.common import ROOT, atomic_json, utc_now
from src.benchmarks import add_dataset_args, benchmark_paths, dataset_provenance, default_benchmark_role
from src.attempt_metadata import build_metadata
from src.attempt_storage import (AttemptArtifacts, DisabledGPUSampler,
                                 add_benchmark_args, apply_benchmark_args)
from src.attempt_metrics import AttemptProfiler, grader_timeline
from runner_final.core_v1._runtime import (
    GPUSampler,
    Services,
    attempt_lock,
    load_questions,
    ready,
    warm_inference,
)
from runner_final.core_v1._ports import ensure_free
from runner_final.core_v1.prewarm import prewarm_benchmark
from src.attempt_runners._streaming_v1_5 import run_question
from src.attempt_runners._allocation_v1_5 import AllocationPool
from runner_final.core_v1.runner import parse_args as parse_core_args

RUNNER_ID = "runner_final_core_v1_5"


def assert_gpu_idle(device):
    """Fail before warmup if another CUDA workload is present; never terminate it."""
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if processes:
            raise RuntimeError(
                f"GPU {device} occupied by PIDs {[p.pid for p in processes]}; defer the sweep"
            )
    finally:
        pynvml.nvmlShutdown()


async def run_speedrun(problems, args, client, output, sampler, attempt_start, profiler=None):
    if args.max_concurrent_requests < len(problems):
        raise ValueError("Concurrent-request limit must fit all initial questions")
    artifacts = profiler.artifacts if profiler else AttemptArtifacts(output)
    pool = AllocationPool([p["problem_idx"] for p in problems], args, artifacts, output, attempt_start)
    target, solved = asyncio.Event(), set()

    def on_solved(event):
        solved.add(event["problem_idx"])
        if len(solved) >= args.target_correct:
            target.set()
            pool.halt_requested = True
            pool.changed.set()

    async def one(problem):
        row = await run_question(problem, args, client, output, sampler,
                                 attempt_start=attempt_start, on_solved=on_solved,
                                 target_event=target, profiler=profiler, pool=pool)
        if row["status"] == "error":
            raise RuntimeError(f"Q{problem['problem_idx']} failed; inspect its trace")
        return row

    tasks_by_index = {p["problem_idx"]: asyncio.create_task(one(p)) for p in problems}
    tasks = list(tasks_by_index.values())
    work = asyncio.gather(*tasks)
    allocator = asyncio.create_task(pool.run())
    reached = asyncio.create_task(target.wait())
    try:
        await asyncio.wait([work, allocator, reached], return_when=asyncio.FIRST_COMPLETED)
        if allocator.done():
            await allocator  # Fail fast on generation errors.
            await asyncio.wait([work, reached], return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            await work
    finally:
        # Winning questions finish trace settlement without a second cancellation.
        for index, task in tasks_by_index.items():
            if index not in solved and not task.done() and not task.cancelling():
                task.cancel()
        for task in (allocator, reached):
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, work, allocator, reached, return_exceptions=True)
        pool.snapshot()
    return artifacts.questions()


async def run(args):
    from runner_final.integrity_v1_5 import verify_core
    args.core_manifest_sha256 = verify_core(ROOT)
    if not isinstance(args.system_prompt, str) or not args.system_prompt.strip():
        raise ValueError("A nonempty system prompt must be supplied by run_frozen")
    services = Services()
    sampler = None
    began = time.perf_counter()
    output = (
        ROOT / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    output.mkdir(parents=True)
    print(f"Attempt artifacts: {output}", flush=True)
    artifacts = AttemptArtifacts(output, buffered=args.buffer_traces)
    profiler = AttemptProfiler(
        output,
        artifacts=artifacts,
        enabled=not args.no_overhead_profile,
        interval=args.overhead_interval,
        engine_interval=args.engine_metrics_interval,
    )
    status, results, error = "initializing", [], None
    official_start = None
    config = {
        **vars(args),
        "attempt_id": output.name,
        "initialization_started_at_utc": utc_now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                text=True,
            )
        ),
        "runner_id": RUNNER_ID,
        "runner_module": "src.attempt_runners.speedrun_v1_5",
        "system_prompt": args.system_prompt,
        "allocation_policy": "eager30; continuations first, then least-active fresh starts",
        "token_budget_scope": "8192 initial, 16384 additional per later request, clipped to context",
        "grading": "single vendored grader; no local answer-key comparisons",
        "gpu_scope": ("disabled for benchmark" if args.no_gpu_telemetry else "device-level NVML; vLLM preallocates VRAM"),
        "python": sys.version,
    }
    atomic_json(output / "config.json", config)
    try:
        config['dataset_provenance'] = dataset_provenance(args.benchmark_year, args.benchmark_role)
        problems = load_questions(args.questions, args.benchmark_year)
        config["question_indices"] = [p["problem_idx"] for p in problems]
        if args.target_correct > len(problems):
            raise ValueError("Target correct exceeds the number of selected questions")
        profile_path = (
            Path(args.models_dir).expanduser() / args.model / args.model_profile
        )
        profile_text = profile_path.read_text()
        profile = yaml.safe_load(profile_text)
        overrides = profile.get("override-generation-config", {})
        if isinstance(overrides, str):
            overrides = json.loads(overrides)
        cap = overrides.get("max_new_tokens")
        if cap is not None and max(args.max_tokens, args.first_pass_max_tokens, 0 if args.skip_benchmark_prewarm else args.prewarm_max_tokens) > cap:
            raise ValueError(
                f"Requested max_tokens exceeds model profile ceiling ({cap})"
            )
        atomic_json(
            output / "model_profile.json",
            {
                "path": str(profile_path),
                "yaml": profile_text,
                "sha256": hashlib.sha256(profile_text.encode()).hexdigest(),
            },
        )
        ensure_free(args.grader_port)
        if not args.reuse_server:
            ensure_free(args.vllm_port)
        sampler = (DisabledGPUSampler() if args.no_gpu_telemetry else
                   GPUSampler(artifacts.gpu_path(output / "gpu.jsonl"), args.gpu_interval, args.gpu_device))
        await sampler.start()
        limits = httpx.Limits(
            max_connections=args.parallelism * (args.rollouts + 1) + 8,
            max_keepalive_connections=args.parallelism * (args.rollouts + 1) + 8,
        )
        timeout = httpx.Timeout(
            connect=10, read=args.question_timeout, write=30, pool=30
        )
        async with httpx.AsyncClient(
            timeout=timeout, limits=limits, trust_env=False
        ) as client:
            if not args.reuse_server:
                assert_gpu_idle(args.gpu_device)
                # Small CUDA matmul warmup before loading the model; cap allocation.
                gpu_warmup = services.launch(
                    [
                        args.vllm_python,
                        "-c",
                        "import torch,time; torch.cuda.set_device("
                        + str(args.gpu_device)
                        + "); "
                        'x=torch.randn((1024,1024),device="cuda",dtype=torch.bfloat16); '
                        "end=time.monotonic()+2; "
                        "\nwhile time.monotonic()<end: y=x@x; torch.cuda.synchronize()",
                    ],
                    output / "gpu_warmup.log",
                )
                with profiler.meter.measure(
                    "initialization_gpu_warmup_wait", cpu=False
                ):
                    await asyncio.wait_for(
                        asyncio.to_thread(gpu_warmup.wait), timeout=120
                    )
                if gpu_warmup.returncode:
                    raise RuntimeError("CUDA warmup failed; inspect gpu_warmup.log")
                command = [
                    args.vllm_binary,
                    "serve",
                    "--config",
                    str(profile_path),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.vllm_port),
                    "--enable-prompt-tokens-details",
                ]
                config["vllm_command"] = command
                server = services.launch(command, output / "vllm.log",
                                         env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"})
            else:
                server = None
            with profiler.meter.measure("initialization_server_ready_wait", cpu=False):
                models = await ready(
                    client, args.vllm_url + "/v1/models", args.startup_timeout, server
                )
            if args.model not in [m["id"] for m in models.get("data", [])]:
                raise RuntimeError(
                    "Inference server does not serve the requested model"
                )
            atomic_json(output / "server_models.json", models)
            served = next(m for m in models["data"] if m["id"] == args.model)
            args.max_context_tokens = int(
                served.get("max_model_len") or profile["max-model-len"]
            )
            config["max_context_tokens"] = args.max_context_tokens
            if not args.skip_benchmark_prewarm:
                config["benchmark_prewarm"] = await prewarm_benchmark(args, client, output)
                atomic_json(output / "config.json", config)
            grader_config = {
                "dataset": {
                    "source": str(benchmark_paths(args.benchmark_year)[1]),
                    "format": "jsonl",
                    "idx_field": "problem_idx",
                    "gold_field": "answer",
                },
                "cost_c": args.grader_cost,
                "host": "127.0.0.1",
                "port": args.grader_port,
                "audit_log": str(output / "grader_audit.jsonl"),
            }
            grader_config['dataset'].update(id=config['dataset_provenance']['id'], year=args.benchmark_year, revision=config['dataset_provenance']['revision'])
            (output / "grader_config.yaml").write_text(yaml.safe_dump(grader_config))
            grader = services.launch(
                [args.grader_python, str(ROOT / "grader/server.py")],
                output / "grader.log",
                {**os.environ, "GRADER_CONFIG": str(output / "grader_config.yaml")},
            )
            with profiler.meter.measure("initialization_grader_ready_wait", cpu=False):
                health = await ready(client, args.grader_url + "/health", 60, grader)
            if (
                health.get("queries_so_far") != 0
                or health.get("cost_c") != args.grader_cost
            ):
                raise RuntimeError(
                    "Grader did not start with a fresh queue and requested toll"
                )
            if health.get('dataset', {}).get('sha256') != config['dataset_provenance']['grader_sha256']:
                raise RuntimeError('Grader loaded a different benchmark answer key')
            config["grader_health"] = health
            with profiler.meter.measure(
                "initialization_inference_warmup_wait", cpu=False
            ):
                warmup = await warm_inference(args, client, len(problems))
            atomic_json(output / "inference_warmup.json", warmup)
            config["inference_warmup"] = {
                k: warmup[k] for k in ("latency_s", "batch_size", "tokens_per_request")
            }
            config.update(
                model_profile_sha256=hashlib.sha256(profile_text.encode()).hexdigest(),
                launch_profile=profile,
                official_started_at_utc=utc_now(),
            )
            atomic_json(output / "config.json", config)
            official_start = time.perf_counter()
            profiler.official_start(client, args.vllm_url)
            status = "running"
            print("Official solving phase started", flush=True)
            results = await run_speedrun(
                problems, args, client, output, sampler, official_start, profiler
            )
            await profiler.stop()
            status = (
                "completed"
                if all(r["status"] != "error" for r in results)
                else "failed"
            )
            if sampler.error:
                raise RuntimeError(f"GPU telemetry failed: {sampler.error}")
    except asyncio.CancelledError:
        status, error = "interrupted", "Attempt interrupted by signal"
        raise
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        await profiler.stop()
        if sampler:
            sampler.stop()
        official_end = time.perf_counter()
        # Include partially completed questions after interruption/failure.
        results = artifacts.questions()
        summary = {
            "attempt_id": output.name,
            "status": status,
            "error": error,
            "official_started_at_utc": config.get("official_started_at_utc"),
            "official_finished_at_utc": utc_now(),
            "official_latency_s": (
                official_end - official_start if official_start else None
            ),
            "initialization_and_attempt_latency_s": official_end - began,
            "benchmark_prewarm": config.get("benchmark_prewarm"),
            "runner_id": RUNNER_ID,
            "schedule": args.schedule,
            "solved": sum(r["status"] == "solved" for r in results),
            "questions_completed": sum(r["status"] != "stopped" for r in results),
            "questions_attempted": len(results),
            "target_correct": args.target_correct,
            "target_reached": sum(r["status"] == "solved" for r in results)
            >= args.target_correct,
            "rounds_executed": max((r["round"] for r in results), default=0),
            "questions": [
                {
                    **{
                        k: r[k]
                        for k in (
                            "problem_idx",
                            "status",
                            "end_to_end_latency_s",
                            "unique_candidates",
                        )
                    },
                    "verified_answer": (
                        r["winner"]["candidate"] if r["winner"] else None
                    ),
                    "winning_rollout": r["winner"]["rollout"] if r["winner"] else None,
                    "first_solved": r.get("first_solved"),
                }
                for r in results
            ],
        }
        solved_times = sorted(
            r["first_solved"]["first_solved_elapsed_s"]
            for r in results
            if r.get("first_solved")
        )
        summary["time_to_target_s"] = (
            solved_times[args.target_correct - 1]
            if len(solved_times) >= args.target_correct
            else None
        )
        rollouts = [rollout for question in results for rollout in question["rollouts"]]

        def ttft(continued):
            values = sorted(
                r["ttft_s"]
                for r in rollouts
                if r["ttft_s"] is not None
                and (r["continuation_of_rollout"] is not None) == continued
            )
            return {
                "count": len(values),
                "median_s": values[len(values) // 2] if values else None,
                "p95_s": values[int((len(values) - 1) * 0.95)] if values else None,
            }

        summary["performance"] = {
            "generation_requests": len(rollouts),
            "fresh_ttft": ttft(False),
            "continuation_ttft": ttft(True),
            "official_gpu": (
                sampler.window(official_start, official_end)
                if sampler and official_start
                else None
            ),
            "initialization_gpu": (
                sampler.window(began, official_start or official_end)
                if sampler
                else None
            ),
        }
        summary["overhead"] = profiler.snapshot()
        atomic_json(
            output / "summary.json", summary
        )  # Durable even if service cleanup fails.
        atomic_json(output / "config.json", config)
        cleanup_start = time.perf_counter()
        try:
            await services.close()
        finally:
            summary["service_cleanup_latency_s"] = time.perf_counter() - cleanup_start
            try:
                summary["trace_storage"] = artifacts.flush()
            except Exception as exc:
                summary.update(status="failed", error=f"Trace flush failed: {type(exc).__name__}: {exc}")
                atomic_json(output / "summary.json", summary)
                raise
            summary["trace_storage"]["scope"] = "Client trace flush after official timing; excluded from time to target and official latency"
            atomic_json(output / "summary.json", summary)
            atomic_json(output / "metadata.json", build_metadata(output, config))

        summary["grader_timeline"] = grader_timeline(
            output / "grader_audit.jsonl",
            config.get("official_started_at_utc"),
            args.target_correct,
            args.grader_cost,
        )
        atomic_json(output / "overhead.json", summary["overhead"])
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    if status == "failed":
        raise RuntimeError("One or more questions failed; inspect trace telemetry")
    return output


def parse_args(argv=None):
    args = parse_core_args(argv)
    if args.parallelism != 30 or args.rollouts != 1:
        raise ValueError("Core v1.5 fixes 30 request slots and one initial rollout per question")
    args.max_concurrent_requests = 30
    args.schedule = "eager_pool"
    return args


def main(argv=None):
    from runner_final.run_v1_5 import main as frozen_main
    frozen_main(argv)


if __name__ == "__main__":
    main()
