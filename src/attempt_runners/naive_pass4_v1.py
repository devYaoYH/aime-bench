"""Naive final-answer pass@4 baseline, version 1.

Run: python -m src.attempt_runners.naive_pass4_v1 --model WeiboAI/VibeThinker-3B
Based on the completed baseline policy at commit 0834cfe. The canonical runner
is separate and is never imported or patched by this module.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import httpx
import yaml

from src.common import ROOT, atomic_json, utc_now
from src.benchmarks import add_dataset_args, benchmark_paths, dataset_provenance, default_benchmark_role
from src.attempt_metadata import build_metadata
from src.attempt_runners._runtime_v1 import (
    GPUSampler,
    Services,
    append_json,
    attempt_lock,
    ensure_free,
    load_questions,
    ready,
    sse_payloads,
    warm_inference,
)

RUNNER_ID = "naive_pass4_v1"
RUNNER_MODULE = "src.attempt_runners.naive_pass4_v1"
PROMPT = "You are a helpful assistant. Solve the math problem carefully and put your final answer in \\boxed{} notation."


def final_answer(text):
    """Read only the last box in the final content of a naturally ended response."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None
    boxes = list(re.finditer("\\\\boxed\\s*\\{", text))
    match = (
        re.match("\\\\boxed\\s*\\{\\s*(\\d{1,3})\\s*\\}", text[boxes[-1].start() :])
        if boxes
        else None
    )
    if match is None:
        return None
    return {
        "answer": int(match[1]),
        "part": "content",
        "kind": "final_box",
        "extraction_stage": "completed_final_response",
    }


async def run_question(
    problem,
    args,
    client,
    output,
    sampler,
    *,
    round_no=1,
    rollout_offset=0,
    attempt_start=None,
    on_solved=None,
    target_event=None,
):
    index = problem["problem_idx"]
    folder = output / "trace" / f"{index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    previous = (
        json.loads((folder / "question.json").read_text())
        if (folder / "question.json").exists()
        else None
    )
    if previous and previous["status"] == "solved":
        raise ValueError("Solved questions must not be retried")
    if (
        previous
        and len(previous["rollouts"]) + args.rollouts > args.max_attempts_per_question
    ):
        raise ValueError("Per-question generation attempt limit exceeded")
    start, started = (time.perf_counter(), utc_now())
    candidates = asyncio.Queue()
    seen = set(previous.get("candidate_answers", [])) if previous else set()
    records = []
    winner = None
    question_error = None
    solved_event = None
    stopped_for_target = False

    def propose(event, rollout):
        candidate = str(event["answer"])
        if candidate not in seen:
            seen.add(candidate)
            candidates.put_nowait(
                {
                    **event,
                    "candidate": candidate,
                    "rollout": rollout,
                    "observed_at_utc": utc_now(),
                    "question_elapsed_s": time.perf_counter() - start,
                }
            )

    async def generate(rollout):
        path = folder / f"rollout-{rollout:02d}"
        path.mkdir()
        request = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": problem["problem"]},
            ],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed + index * args.rollouts + rollout,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "return_token_ids": False,
        }
        request["max_tokens"] = min(
            args.max_tokens, args.max_context_tokens - problem["prompt_tokens"]
        )
        if request["max_tokens"] <= 0:
            raise ValueError("Baseline prompt fills the model context")
        if args.disable_thinking:
            request["chat_template_kwargs"] = {"enable_thinking": False}
        endpoint = "/v1/chat/completions"
        atomic_json(path / "request.json", request)
        began = time.perf_counter()
        record = {
            "rollout": rollout,
            "round": round_no,
            "started_at_utc": utc_now(),
            "start_monotonic_s": began,
            "ttft_s": None,
            "last_token_s": None,
            "finish_reason": None,
            "status": "streaming",
            "done_received": False,
            "usage": None,
            "endpoint": endpoint,
            "continuation_of_rollout": None,
            "requested_max_tokens": request["max_tokens"],
            "grading_mode": "completed_final_only",
        }
        parts = {"reasoning": [], "content": []}
        prompt_token_ids = None
        output_token_ids = []
        final_event = None
        records.append((path, record, parts))
        try:
            with (path / "stream.jsonl").open("w") as stream_file:
                async with client.stream(
                    "POST",
                    args.vllm_url + endpoint,
                    json=request,
                    headers={"X-Request-Id": f"{output.name}-q{index}-r{rollout}"},
                ) as response:
                    record["http_status"] = response.status_code
                    record["headers_received_s"] = time.perf_counter() - began
                    response.raise_for_status()
                    async for payload in sse_payloads(response.aiter_lines()):
                        elapsed = time.perf_counter() - began
                        append_json(
                            stream_file,
                            {
                                "elapsed_s": elapsed,
                                "timestamp_utc": utc_now(),
                                "data": payload,
                            },
                        )
                        if payload == "[DONE]":
                            record["done_received"] = True
                            break
                        body = json.loads(payload)
                        if body.get("error"):
                            raise RuntimeError(f"vLLM stream error: {body['error']}")
                        if body.get("prompt_token_ids") is not None:
                            prompt_token_ids = body["prompt_token_ids"]
                        if body.get("usage"):
                            record["usage"] = body["usage"]
                        for choice in body.get("choices", []):
                            if choice.get("index", 0) != 0:
                                continue
                            if choice.get("prompt_token_ids") is not None:
                                prompt_token_ids = choice["prompt_token_ids"]
                            if choice.get("token_ids"):
                                output_token_ids.extend(choice["token_ids"])
                            delta = choice.get("delta") or {}
                            for part, value in [
                                (
                                    "reasoning",
                                    delta.get("reasoning_content")
                                    or delta.get("reasoning"),
                                ),
                                ("content", delta.get("content")),
                            ]:
                                if isinstance(value, str) and value:
                                    if record["ttft_s"] is None:
                                        record["ttft_s"] = elapsed
                                    record["last_token_s"] = elapsed
                                    parts[part].append(value)
                            if choice.get("finish_reason"):
                                record["finish_reason"] = choice["finish_reason"]
                    if record["finish_reason"] in ("error", "abort"):
                        raise RuntimeError(
                            f"vLLM terminated with {record['finish_reason']}"
                        )
                    if record["done_received"] or record["finish_reason"] in (
                        "stop",
                        "length",
                    ):
                        record["status"] = "completed"
                        final_event = (
                            final_answer("".join(parts["content"]))
                            if record["finish_reason"] == "stop"
                            else None
                        )
                        record["final_answer_extracted"] = final_event is not None
                    else:
                        raise RuntimeError(
                            "Stream ended without DONE or a terminal finish reason"
                        )
            if final_event:
                propose(final_event, rollout)
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            ended = time.perf_counter()
            record.update(
                generation_finished_at_utc=utc_now(),
                generation_end_monotonic_s=ended,
                generation_latency_s=ended - began,
                generation_censored=record["status"] != "completed"
                or record["finish_reason"] == "length",
            )
            usage = record["usage"] or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            record.update(
                generated_token_ids_count=len(output_token_ids),
                prompt_token_ids_count=(
                    len(prompt_token_ids) if prompt_token_ids else None
                ),
                cached_prompt_tokens=cached,
                prefix_cache_hit_fraction=(
                    cached / len(prompt_token_ids)
                    if cached is not None and prompt_token_ids
                    else None
                ),
            )
            atomic_json(
                path / "response.json",
                {part: "".join(text) for part, text in parts.items()},
            )
            atomic_json(path / "telemetry.json", record)

    streams = [
        asyncio.create_task(generate(rollout_offset + r))
        for r in range(1, args.rollouts + 1)
    ]

    async def drained():
        try:
            await asyncio.gather(*streams)
        finally:
            candidates.put_nowait(None)

    producer = asyncio.create_task(drained())
    try:
        async with asyncio.timeout(args.question_timeout):
            with (folder / "verification.jsonl").open("a") as file:
                while True:
                    event = await candidates.get()
                    if event is None:
                        break
                    event.update(verification_started_at_utc=utc_now(), round=round_no)
                    verify_start = time.perf_counter()
                    try:
                        response = await client.post(
                            args.grader_url + "/verify",
                            json={
                                "index": index,
                                "candidate": event["candidate"],
                                "agent_id": f"{output.name}-q{index}-r{event['rollout']}",
                                "query_id": f"{output.name}-q{index}-a{event['candidate']}",
                            },
                        )
                        response.raise_for_status()
                        verdict = response.json()
                        if type(verdict.get("verdict")) is not bool:
                            raise RuntimeError(
                                "Grader response lacks a boolean verdict"
                            )
                        event["result"] = verdict
                    except Exception as exc:
                        event["error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        event.update(
                            verification_finished_at_utc=utc_now(),
                            verification_latency_s=time.perf_counter() - verify_start,
                        )
                        append_json(file, event)
                    if verdict["verdict"]:
                        winner = event
                        solved_event = {
                            "problem_idx": index,
                            "round": round_no,
                            "rollout": event["rollout"],
                            "candidate": event["candidate"],
                            "first_solved_at_utc": utc_now(),
                            "first_solved_elapsed_s": time.perf_counter()
                            - (attempt_start if attempt_start is not None else start),
                            "grader_answered_at_utc": verdict.get("answered_at"),
                            "grader_query_id": verdict.get("query_id"),
                        }
                        with (output / "solved.jsonl").open("a") as solved_file:
                            append_json(solved_file, solved_event)
                        break
    except asyncio.CancelledError:
        stopped_for_target = target_event is not None and target_event.is_set()
        question_error = None if stopped_for_target else "attempt interrupted"
        raise
    except Exception as exc:
        question_error = f"{type(exc).__name__}: {exc}"
    finally:
        for task in streams:
            if not task.done() and (not task.cancelling()):
                task.cancel()
        await asyncio.gather(*streams, return_exceptions=True)
        await asyncio.gather(producer, return_exceptions=True)
        ended, ended_at = (time.perf_counter(), utc_now())
        for path, record, _ in records:
            record.update(
                finished_at_utc=ended_at,
                end_to_end_latency_s=ended - record["start_monotonic_s"],
                end_to_end_scope="rollout start through question verification/cancellation settlement",
                gpu=sampler.window(
                    record["start_monotonic_s"], record["generation_end_monotonic_s"]
                ),
            )
            atomic_json(path / "telemetry.json", record)
        result = {
            "problem_idx": index,
            "started_at_utc": started,
            "finished_at_utc": ended_at,
            "end_to_end_latency_s": ended - start,
            "status": (
                "error"
                if question_error
                else (
                    "solved"
                    if winner
                    else "stopped" if stopped_for_target else "unsolved"
                )
            ),
            "winner": winner,
            "error": question_error,
            "unique_candidates": len(seen),
            "candidate_answers": sorted(seen),
            "rollouts": [
                record
                for _, record, _ in sorted(records, key=lambda r: r[1]["rollout"])
            ],
        }
        if any((r["status"] == "error" for r in result["rollouts"])) and (not winner):
            result["status"] = "error"
        result.update(round=round_no, first_solved=solved_event)
        atomic_json(folder / f"round-{round_no:02d}.json", result)
        rounds = (previous.get("rounds", []) if previous else []) + [
            {
                k: result[k]
                for k in (
                    "round",
                    "status",
                    "started_at_utc",
                    "finished_at_utc",
                    "end_to_end_latency_s",
                    "winner",
                    "error",
                )
            }
        ]
        if previous:
            result["rollouts"] = previous["rollouts"] + result["rollouts"]
            result["started_at_utc"] = previous["started_at_utc"]
            result["first_solved"] = previous.get("first_solved") or solved_event
        result["rounds"] = rounds
        result["question_start_monotonic_s"] = (
            previous.get("question_start_monotonic_s", start) if previous else start
        )
        result["end_to_end_latency_s"] = ended - result["question_start_monotonic_s"]
        atomic_json(folder / "question.json", result)
        if solved_event and on_solved:
            on_solved(solved_event)
        print(
            f"Q{index:02d} round {round_no}: {result['status']}, {ended - start:.2f}s",
            flush=True,
        )
    return result


async def run_questions(
    problems,
    args,
    client,
    output,
    sampler,
    *,
    round_no=1,
    rollout_offset=0,
    attempt_start=None,
    on_solved=None,
    target_event=None,
):
    queue = asyncio.Queue()
    for problem in problems:
        queue.put_nowait(problem)
    results = []

    async def worker():
        while not queue.empty() and (
            not (target_event is not None and target_event.is_set())
        ):
            problem = queue.get_nowait()
            results.append(
                await run_question(
                    problem,
                    args,
                    client,
                    output,
                    sampler,
                    round_no=round_no,
                    rollout_offset=rollout_offset,
                    attempt_start=attempt_start,
                    on_solved=on_solved,
                    target_event=target_event,
                )
            )

    tasks = [
        asyncio.create_task(worker())
        for _ in range(min(args.parallelism, len(problems)))
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done() and (not task.cancelling()):
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return sorted(results, key=lambda row: row["problem_idx"])


async def prepare_baseline_prompts(problems, args, client):
    """Count the actual served chat template before the official solving timer."""

    async def one(problem):
        request = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": problem["problem"]},
            ],
            "add_generation_prompt": True,
        }
        if args.disable_thinking:
            request["chat_template_kwargs"] = {"enable_thinking": False}
        response = await client.post(args.vllm_url + "/tokenize", json=request)
        response.raise_for_status()
        count = response.json()["count"]
        if type(count) is not int or not 0 < count < args.max_context_tokens:
            raise ValueError("Invalid baseline prompt token count")
        return {**problem, "prompt_tokens": count}

    return await asyncio.gather(*(one(p) for p in problems))


async def run_baseline(problems, args, client, output, sampler, attempt_start):
    """One pass@4 batch, final responses only, with the same solve-target stop."""
    solved = set()
    target_event = asyncio.Event()

    def on_solved(event):
        solved.add(event["problem_idx"])
        print(f"Solved baseline: {len(solved)}/{args.target_correct}", flush=True)
        if len(solved) >= args.target_correct:
            target_event.set()

    batch = asyncio.create_task(
        run_questions(
            problems,
            args,
            client,
            output,
            sampler,
            attempt_start=attempt_start,
            on_solved=on_solved,
            target_event=target_event,
        )
    )
    target_wait = asyncio.create_task(target_event.wait())
    try:
        await asyncio.wait([batch, target_wait], return_when=asyncio.FIRST_COMPLETED)
        if target_event.is_set():
            batch.cancel()
            await asyncio.gather(batch, return_exceptions=True)
        else:
            await batch
    finally:
        target_wait.cancel()
        if not batch.done() and (not batch.cancelling()):
            batch.cancel()
        await asyncio.gather(batch, target_wait, return_exceptions=True)
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
    status, results, error = ("initializing", [], None)
    official_start = None
    config = {
        **vars(args),
        "runner_id": RUNNER_ID,
        "runner_module": RUNNER_MODULE,
        "runtime_version": "v1",
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
        "system_prompt": PROMPT,
        "grading": "single vendored grader; no local answer-key comparisons",
        "answer_extraction": "completed final box only; capped outputs ungraded",
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
        if cap is not None and args.max_tokens > cap:
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
                gpu_warmup = services.launch(
                    [
                        args.vllm_python,
                        "-c",
                        "import torch,time; torch.cuda.set_device("
                        + str(args.gpu_device)
                        + '); x=torch.randn((1024,1024),device="cuda",dtype=torch.bfloat16); end=time.monotonic()+2; \nwhile time.monotonic()<end: y=x@x; torch.cuda.synchronize()',
                    ],
                    output / "gpu_warmup.log",
                )
                await asyncio.wait_for(asyncio.to_thread(gpu_warmup.wait), timeout=120)
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
            models = await ready(
                client, args.vllm_url + "/v1/models", args.startup_timeout, server
            )
            if args.model not in [m["id"] for m in models.get("data", [])]:
                raise RuntimeError(
                    "Inference server does not serve the requested model"
                )
            atomic_json(output / "server_models.json", models)
            served = next((m for m in models["data"] if m["id"] == args.model))
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
            problems = await prepare_baseline_prompts(problems, args, client)
            config["prompt_tokens_by_question"] = {
                str(p["problem_idx"]): p["prompt_tokens"] for p in problems
            }
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
            status = "running"
            print("Official solving phase started", flush=True)
            results = await run_baseline(
                problems, args, client, output, sampler, official_start
            )
            status = (
                "completed"
                if all((r["status"] != "error" for r in results))
                else "failed"
            )
            if sampler.error:
                raise RuntimeError(f"GPU telemetry failed: {sampler.error}")
    except asyncio.CancelledError:
        status, error = ("interrupted", "Attempt interrupted by signal")
        raise
    except Exception as exc:
        status, error = ("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if sampler:
            sampler.stop()
        official_end = time.perf_counter()
        results = [
            json.loads(p.read_text())
            for p in sorted(output.glob("trace/*/question.json"))
        ]
        summary = {
            "runner_id": RUNNER_ID,
            "runner_module": RUNNER_MODULE,
            "attempt_id": output.name,
            "status": status,
            "error": error,
            "official_started_at_utc": config.get("official_started_at_utc"),
            "official_finished_at_utc": utc_now(),
            "official_latency_s": (
                official_end - official_start if official_start else None
            ),
            "initialization_and_attempt_latency_s": official_end - began,
            "strategy": "baseline",
            "solved": sum((r["status"] == "solved" for r in results)),
            "questions_completed": sum((r["status"] != "stopped" for r in results)),
            "questions_attempted": len(results),
            "target_correct": args.target_correct,
            "target_reached": sum((r["status"] == "solved" for r in results))
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
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "config.json", config)
        atomic_json(output / "metadata.json", build_metadata(output, config))
        await services.close()
        print(json.dumps(summary, indent=2), flush=True)
    if status == "failed":
        raise RuntimeError("One or more questions failed; inspect trace telemetry")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument(
        "--model", required=True, help="Model ID with ~/models/<ID>/vllm.yaml"
    )
    parser.add_argument("--models-dir", default="~/models")
    parser.add_argument(
        "--model-profile",
        default="vllm-baseline-16k.yaml",
        help="YAML filename within the model directory",
    )
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
    parser.set_defaults(strategy="baseline")
    parser.add_argument(
        "--parallelism", type=int, help="Concurrent questions (default 30)"
    )
    parser.add_argument(
        "--rollouts", type=int, help="Independent samples per question (default 4)"
    )
    parser.add_argument("--target-correct", type=int, default=18)
    parser.add_argument(
        "--max-attempts-per-question",
        type=int,
        default=4,
        help="Hard sample budget per question",
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
    args = parser.parse_args(argv)
    args.benchmark_role = args.benchmark_role or default_benchmark_role(args.benchmark_year)
    if args.parallelism is None:
        args.parallelism = 30
    if args.rollouts is None:
        args.rollouts = 4
    if args.rollouts > args.max_attempts_per_question:
        parser.error("Rollouts exceed the per-question attempt limit")
    if any(
        (
            getattr(args, key) <= 0
            for key in (
                "parallelism",
                "rollouts",
                "max_tokens",
                "startup_timeout",
                "question_timeout",
                "gpu_interval",
                "target_correct",
                "max_attempts_per_question",
            )
        )
    ):
        parser.error(
            "Concurrency, token budgets, timeouts, and sampling interval must be positive"
        )
    if (
        args.grader_cost < 0
        or args.gpu_device < 0
        or (not 0 < args.top_p <= 1)
        or (args.temperature < 0)
    ):
        parser.error("Invalid grader cost, GPU device, or sampling settings")
    if (
        "/" not in args.model
        or any((part in ("", ".", "..") for part in args.model.split("/")))
        or args.model.startswith("/")
    ):
        parser.error("Use a relative organization/model ID")
    if Path(
        args.model_profile
    ).name != args.model_profile or not args.model_profile.endswith(".yaml"):
        parser.error("Model profile must be a YAML filename within the model directory")
    if (
        not all((1 <= port <= 65535 for port in (args.vllm_port, args.grader_port)))
        or args.vllm_port == args.grader_port
    ):
        parser.error("Service ports must be valid and distinct")
    args.max_context_tokens = 16384  # Replaced by served model metadata before solving.
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
