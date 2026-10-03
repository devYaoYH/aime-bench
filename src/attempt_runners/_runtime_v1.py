"""v1 service/telemetry runtime snapshot, isolated from the canonical runner.

Copied from the helpers used by the baseline at commit 0834cfe.
Keep this version stable; behavior changes belong in a new runner/runtime version.
"""

import asyncio
from contextlib import contextmanager
import fcntl
import json
import os
import signal
import socket
import subprocess
import threading
import time

import httpx

from src.common import ROOT, utc_now


async def sse_payloads(lines):
    data = []
    async for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield "\n".join(data)


def append_json(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.flush()


class GPUSampler:
    """Device-level NVML samples, shared by concurrent rollouts (not attribution)."""

    def __init__(self, path, interval=0.2, device=0):
        self.path, self.interval, self.device = path, interval, device
        self.samples = []
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            with self.path.open("w") as file:
                while not self.stop_event.is_set():
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    row = {
                        "monotonic_s": time.perf_counter(),
                        "timestamp_utc": utc_now(),
                        "vram_used_mib": memory.used / 2**20,
                        "vram_total_mib": memory.total / 2**20,
                        "gpu_util_pct": util.gpu,
                    }
                    self.samples.append(row)
                    append_json(file, row)
                    self.stop_event.wait(self.interval)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    async def start(self):
        self.thread.start()
        for _ in range(100):
            if self.error:
                raise RuntimeError(f"GPU telemetry failed: {self.error}")
            if self.samples:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("GPU telemetry did not produce a sample")

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def window(self, start, end):
        snapshot = list(self.samples)
        before = [s for s in snapshot if s["monotonic_s"] <= start]
        rows = ([before[-1]] if before else []) + [
            s for s in snapshot if start < s["monotonic_s"] <= end
        ]
        return {
            "sample_count": len(rows),
            "start_vram_mib": rows[0]["vram_used_mib"] if rows else None,
            "end_vram_mib": rows[-1]["vram_used_mib"] if rows else None,
            "observed_peak_vram_mib": max(
                (s["vram_used_mib"] for s in rows), default=None
            ),
            "scope": "shared GPU device; sampled peak, not per-request allocation",
            "error": self.error,
        }


def ensure_free(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


class Services:
    def __init__(self):
        self.processes = []
        self.logs = []

    def launch(self, command, logfile, env=None):
        file = logfile.open("w")
        self.logs.append(file)
        process = subprocess.Popen(
            command,
            stdout=file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            cwd=ROOT,
        )
        self.processes.append(process)
        return process

    async def close(self):
        for process in reversed(self.processes):
            # Own the entire process group, including vLLM EngineCore children.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.perf_counter() + 15
        while (
            any(p.poll() is None for p in self.processes)
            and time.perf_counter() < deadline
        ):
            await asyncio.sleep(0.1)
        for process in self.processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for file in self.logs:
            file.close()


async def ready(client, url, timeout, process=None):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Service exited ({process.returncode}); inspect attempt service logs"
            )
        try:
            response = await client.get(url, timeout=2)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Service not ready: {url}")


@contextmanager
def attempt_lock():
    with (ROOT / ".attempt.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another canonical attempt owns this checkout") from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def load_questions(indices):
    # Deliberately strip answers; only the grader reads the key during execution.
    rows = [
        json.loads(line)
        for line in (ROOT / "data/aime_2025_problems.jsonl").read_text().splitlines()
        if line.strip()
    ]
    problems = [
        {"problem_idx": row["problem_idx"], "problem": row["problem"]} for row in rows
    ]
    if indices:
        requested = set(indices)
        if not requested <= {p["problem_idx"] for p in problems}:
            raise ValueError("Unknown question index")
        problems = [p for p in problems if p["problem_idx"] in requested]
    return problems


async def warm_inference(args, client, question_count):
    """Warm the configured sampling path at the attempt's maximum batch size."""
    batch_size = min(args.parallelism, question_count) * args.rollouts
    tokens = min(32, args.max_tokens)
    started, start = utc_now(), time.perf_counter()
    print(f"Inference warmup: {batch_size} streams, {tokens} tokens each", flush=True)

    async def one(slot):
        request = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Compute 1 + 1."}],
            "max_tokens": tokens,
            "min_tokens": tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed - batch_size + slot,
            "stream": False,
        }
        if args.disable_thinking:
            request["chat_template_kwargs"] = {"enable_thinking": False}
        response = await client.post(
            args.vllm_url + "/v1/chat/completions", json=request
        )
        response.raise_for_status()
        return {"request": request, "response": response.json()}

    tasks = [asyncio.create_task(one(slot)) for slot in range(batch_size)]
    try:
        responses = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return {
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "latency_s": time.perf_counter() - start,
        "batch_size": batch_size,
        "tokens_per_request": tokens,
        "requests": responses,
    }
