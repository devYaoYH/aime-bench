"""Prospective-answer speedrun v1: 8K first samples, lane continuations, cap four.

Run only on an idle GPU. See configs/sweeps/vibe-speedrun-v1.json for a deferred
concurrency sweep; planning that sweep does not launch this module.
"""

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
from src.attempt_metrics import AttemptProfiler, grader_timeline
from src.attempt_runners._runtime_v1 import (
    GPUSampler,
    Services,
    attempt_lock,
    load_questions,
    ready,
    warm_inference,
)
from src.attempt_runners._ports_v1 import ensure_free
from src.attempt_runners._streaming_v1 import PROMPT, continuation_prefix, run_question

RUNNER_ID = "speedrun_v1"


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


def question_state(output, index):
    path = output / "trace" / f"{index:02d}" / "question.json"
    return json.loads(path.read_text()) if path.exists() else None


def group_options(problem, args, output):
    """Reserve at most the remaining budget, continuing each distinct lane once."""
    previous = question_state(output, problem["problem_idx"])
    used = len(previous["rollouts"]) if previous else 0
    options = copy(args)
    options.rollouts = min(args.rollouts, args.max_attempts_per_question - used)
    options.max_tokens = args.max_tokens if used else args.first_pass_max_tokens
    round_no = previous["round"] + 1 if previous else 1
    prefixes = [None] * options.rollouts
    if previous and not args.no_continuation:
        folder = output / "trace" / f'{problem["problem_idx"]:02d}'
        # The latest group contains one result per independent lane. Each new
        # lane resumes its own exact prefix, never the same last sample four times.
        last_group = [
            r for r in previous["rollouts"] if r["round"] == previous["round"]
        ]
        prefixes = [
            continuation_prefix({"rollouts": [r]}, folder, args.max_context_tokens)
            for r in last_group[: options.rollouts]
        ]
    return options, round_no, used, prefixes


async def run_speedrun(
    problems, args, client, output, sampler, attempt_start, profiler=None
):
    """FIFO fresh coverage before retries; optional barrier is a matched control."""
    target = asyncio.Event()
    solved = set()

    def on_solved(event):
        solved.add(event["problem_idx"])
        if len(solved) >= args.target_correct:
            target.set()

    def eligible(problem):
        row = question_state(output, problem["problem_idx"])
        return not row or (
            row["status"] == "unsolved"
            and len(row["rollouts"]) < args.max_attempts_per_question
            and row["round"] < args.max_rounds
        )

    async def batch(pending, retry):
        queue = asyncio.Queue()
        for problem in pending:
            queue.put_nowait(problem)

        async def worker():
            while not target.is_set():
                problem = await queue.get()
                try:
                    options, round_no, used, prefixes = group_options(
                        problem, args, output
                    )
                    row = await run_question(
                        problem,
                        options,
                        client,
                        output,
                        sampler,
                        round_no=round_no,
                        rollout_offset=used,
                        attempt_start=attempt_start,
                        on_solved=on_solved,
                        target_event=target,
                        profiler=profiler,
                        continuations=prefixes,
                    )
                    if row["status"] == "error":
                        raise RuntimeError(
                            f'Q{problem["problem_idx"]} failed; inspect its trace'
                        )
                    if retry and not target.is_set() and eligible(problem):
                        queue.put_nowait(problem)
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(args.parallelism, len(pending)))
        ]
        all_workers = asyncio.gather(*workers)
        drained = asyncio.create_task(queue.join())
        try:
            await asyncio.wait(
                [all_workers, drained], return_when=asyncio.FIRST_COMPLETED
            )
            if all_workers.done():
                await all_workers  # Propagate worker failure rather than hang queue.join().
        finally:
            drained.cancel()
            for task in workers:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*workers, drained, return_exceptions=True)
            # Retrieve aggregate cancellation/exception to avoid unhandled futures.
            await asyncio.gather(all_workers, return_exceptions=True)

    for _ in range(args.max_rounds):
        pending = [p for p in problems if eligible(p)]
        if not pending or target.is_set():
            break
        work = asyncio.create_task(batch(pending, retry=args.schedule == "eager"))
        reached = asyncio.create_task(target.wait())
        try:
            await asyncio.wait([work, reached], return_when=asyncio.FIRST_COMPLETED)
            if target.is_set():
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                break
            await work
        finally:
            reached.cancel()
            if not work.done() and not work.cancelling():
                work.cancel()
            await asyncio.gather(work, reached, return_exceptions=True)
        if args.schedule == "eager":
            break
    return [
        json.loads(p.read_text()) for p in sorted(output.glob("trace/*/question.json"))
    ]


async def run(args):
    services = Services()
    sampler = None
    began = time.perf_counter()
    output = (
        ROOT / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    output.mkdir(parents=True)
    print(f"Attempt artifacts: {output}", flush=True)
    profiler = AttemptProfiler(
        output,
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
        "runner_module": __name__,
        "system_prompt": PROMPT,
        "grading": "single vendored grader; no local answer-key comparisons",
        "gpu_scope": "device-level NVML; vLLM preallocates VRAM",
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
        if cap is not None and max(args.max_tokens, args.first_pass_max_tokens) > cap:
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
        sampler = GPUSampler(output / "gpu.jsonl", args.gpu_interval, args.gpu_device)
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
                server = services.launch(command, output / "vllm.log")
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
        results = [
            json.loads(p.read_text())
            for p in sorted(output.glob("trace/*/question.json"))
        ]
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
        atomic_json(output / 'metadata.json', build_metadata(output, config))
        cleanup_start = time.perf_counter()
        await services.close()
        summary["service_cleanup_latency_s"] = time.perf_counter() - cleanup_start
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
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument(
        "--model", required=True, help="Model ID with a launch profile under ~/models"
    )
    parser.add_argument("--models-dir", default="~/models")
    parser.add_argument(
        "--vllm-python", default=str(Path("~/.venvs/vllm/bin/python").expanduser())
    )
    parser.add_argument(
        "--vllm-binary", default=str(Path("~/.venvs/vllm/bin/vllm").expanduser())
    )
    parser.add_argument(
        "--grader-python", default=str(ROOT / "grader/.venv/bin/python")
    )
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--grader-port", type=int, default=8077)
    parser.add_argument("--grader-cost", type=float, default=3.0)
    parser.add_argument("--schedule", choices=("eager", "barrier"), default="eager")
    parser.add_argument("--model-profile", default="vllm.yaml")
    parser.add_argument(
        "--parallelism", type=int, default=30, help="Concurrent question groups"
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=4,
        help="Concurrent samples per question group; initial pass uses 8K each",
    )
    parser.add_argument("--first-pass-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--no-continuation",
        action="store_true",
        help="Coverage: use fresh samples even after capped outputs",
    )
    parser.add_argument("--target-correct", type=int, default=18)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=4,
        help="Coverage round bound, including first pass",
    )
    parser.add_argument(
        "--max-attempts-per-question",
        type=int,
        default=4,
        help="Counts every generation request, including continuations",
    )
    parser.add_argument(
        "--questions", type=int, nargs="+", help="Smoke subset; default all 30"
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--question-timeout", type=float, default=1800)
    parser.add_argument("--gpu-interval", type=float, default=0.2)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--no-overhead-profile", action="store_true")
    parser.add_argument(
        "--overhead-interval",
        type=float,
        default=0.05,
        help="Event-loop lag sampling seconds",
    )
    parser.add_argument(
        "--engine-metrics-interval",
        type=float,
        default=1.0,
        help="vLLM metrics polling seconds",
    )
    args = parser.parse_args(argv)
    args.benchmark_role = args.benchmark_role or default_benchmark_role(args.benchmark_year)
    args.strategy = "speedrun_v1"
    if args.rollouts > args.max_attempts_per_question:
        parser.error("Rollouts exceed the per-question attempt limit")
    if args.max_attempts_per_question > 4:
        parser.error(
            "Speedrun v1 has a hard ceiling of four generation requests per question"
        )
    if Path(
        args.model_profile
    ).name != args.model_profile or not args.model_profile.endswith(".yaml"):
        parser.error("Model profile must be a YAML filename within the model directory")
    if any(
        getattr(args, key) <= 0
        for key in (
            "parallelism",
            "rollouts",
            "max_tokens",
            "startup_timeout",
            "question_timeout",
            "gpu_interval",
            "max_rounds",
            "first_pass_max_tokens",
            "target_correct",
            "max_attempts_per_question",
            "overhead_interval",
            "engine_metrics_interval",
        )
    ):
        parser.error(
            "Concurrency, token budgets, timeouts, and sampling interval must be positive"
        )
    if (
        args.grader_cost < 0
        or args.gpu_device < 0
        or not 0 < args.top_p <= 1
        or args.temperature < 0
    ):
        parser.error("Invalid grader cost, GPU device, or sampling settings")
    if (
        "/" not in args.model
        or any(part in ("", ".", "..") for part in args.model.split("/"))
        or args.model.startswith("/")
    ):
        parser.error("Use a relative organization/model ID")
    if (
        not all(1 <= port <= 65535 for port in (args.vllm_port, args.grader_port))
        or args.vllm_port == args.grader_port
    ):
        parser.error("Service ports must be valid and distinct")
    args.max_context_tokens = (
        32768  # replaced by actual /v1/models metadata during initialization
    )
    args.vllm_url = f"http://127.0.0.1:{args.vllm_port}"
    args.grader_url = f"http://127.0.0.1:{args.grader_port}"
    return args


def main():
    args = parse_args()
    with attempt_lock():

        async def entry():
            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
            await run(args)

        asyncio.run(entry())


if __name__ == "__main__":
    main()
