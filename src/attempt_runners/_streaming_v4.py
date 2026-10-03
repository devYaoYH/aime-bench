"""Streaming candidates with dynamic allocation and exact-ID budget expansion.

Same v1 solving policy with attempt-owned buffered telemetry. Differences:
paired seeds across concurrency widths, exact IDs for every sample, and explicit
per-lane continuation prefixes. Do not import or mutate the canonical runner.
"""

import asyncio
import json
import re
import time
from src.attempt_storage import AttemptArtifacts
from src.common import atomic_json, utc_now
from src.attempt_metrics import Meter, merge_meters
from src.attempt_runners._runtime_v1 import sse_payloads

PROMPT = (
    "Solve the AIME problem. Whenever you have a prospective answer to the original "
    "problem, immediately emit it as \\boxed{N}, where N is an integer from 0 to 999. "
    "You may continue checking your work afterward. End with your final boxed answer."
)


class CandidateDetector:
    """Closed integer boxes and complete answer clauses, independently per channel.

    In particular, a network boundary after one digit never completes an answer.
    Markers remain prospective; correctness comes exclusively from the oracle.
    """

    box = re.compile(r"\\boxed\s*\{\s*(\d{1,3})\s*\}")
    line = re.compile(
        r"(?i)^\s*(?:\*\*)?Answer\s*:\s*\$?\s*(\d{1,3})\s*\$?\s*(?:\*\*)?\s*[.]?\s*$"
    )

    prose = re.compile(
        r"(?i)\b(?:final\s+)?answer\s+(?:is|would\s+be|should\s+be|must\s+be|might\s+be)"
        r"\s+\$?\s*(\d{1,3})\s*\$?(?=\s*(?:[.。!?;,](?:\s|$)|$))"
    )

    def __init__(self):
        self.text = {"content": "", "reasoning": ""}
        self.scanned = {"content": 0, "reasoning": 0}
        self.line_start = {"content": 0, "reasoning": 0}

    def feed(self, part, delta, eof=False):
        self.text[part] += delta
        text = self.text[part]
        found = []
        # Revisit the incomplete tail of a box, rather than repeatedly scanning
        # the full reasoning transcript on every token.
        for match in self.box.finditer(text, self.scanned[part]):
            found.append(
                {
                    "answer": int(match[1]),
                    "part": part,
                    "kind": "boxed",
                    "end": match.end(),
                }
            )
            self.scanned[part] = match.end()
        tail = text.rfind("\\boxed", self.scanned[part])
        self.scanned[part] = (
            tail if tail >= 0 else max(self.scanned[part], len(text) - 6)
        )
        start = self.line_start[part]
        while "\n" in text[start:] or (eof and start < len(text)):
            end = text.find("\n", start)
            end = len(text) if end < 0 else end + 1
            complete_line = text[start:end]
            match = self.line.fullmatch(complete_line)
            if match:
                found.append(
                    {
                        "answer": int(match[1]),
                        "part": part,
                        "kind": "answer_line",
                        "end": end,
                    }
                )
            for match in self.prose.finditer(complete_line):
                found.append(
                    {
                        "answer": int(match[1]),
                        "part": part,
                        "kind": "literal_prose",
                        "end": start + match.end(),
                        "line": complete_line.strip(),
                    }
                )
            start = end
        self.line_start[part] = start
        return found


def continuation_prefix(previous, folder, context_limit, artifacts=None):
    """Only continue a capped trajectory with complete, exact token-ID evidence."""
    artifacts = artifacts or AttemptArtifacts(folder.parent.parent)
    if not previous or not previous["rollouts"]:
        return None
    last = previous["rollouts"][-1]
    if last["status"] != "completed" or last["finish_reason"] != "length":
        return None
    token_file = folder / f'rollout-{last["rollout"]:02d}' / "tokens.json"
    if not artifacts.has_json(token_file):
        raise RuntimeError(
            "Cannot continue capped output without saved exact token IDs"
        )
    tokens = artifacts.read_json(token_file)
    prompt, output = tokens["prompt_token_ids"], tokens["output_token_ids"]
    if not prompt or not output or not tokens["complete"]:
        raise RuntimeError("Incomplete token-ID evidence for continuation")
    prefix = prompt + output
    if len(prefix) >= context_limit - 1:
        return None  # start a fresh trajectory after exhausting the context window
    return {
        "prompt": prefix,
        "parent_rollout": last["rollout"],
        "visible_text": tokens["visible_text"],
        "remaining_context": context_limit - len(prefix),
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
    profiler=None,
    continuations=None,
    pool=None,
):
    meter = Meter(
        not getattr(args, "no_overhead_profile", False),
        parent=profiler.meter if profiler else None,
    )
    rollout_meters = {}
    artifacts = profiler.artifacts if profiler else AttemptArtifacts(output)

    def write_json(path, value, scope=meter):
        with scope.measure("artifact_json_write"):
            artifacts.write_json(path, value)

    index = problem["problem_idx"]
    folder = output / "trace" / f"{index:02d}"
    if not artifacts.buffered:
        folder.mkdir(parents=True, exist_ok=True)
    previous = artifacts.read_json(folder / "question.json") if artifacts.has_json(folder / "question.json") else None
    if previous and previous["status"] == "solved":
        raise ValueError("Solved questions must not be retried")
    if (
        previous
        and len(previous["rollouts"]) + args.rollouts > args.max_attempts_per_question
    ):
        raise ValueError("Per-question generation attempt limit exceeded")
    next_continuation = None
    if continuations is None and not args.no_continuation:
        next_continuation = continuation_prefix(
            previous, folder, args.max_context_tokens, artifacts
        )
    start, started = time.perf_counter(), utc_now()
    candidates = asyncio.Queue()
    seen = set(previous.get("candidate_answers", [])) if previous else set()
    records = []
    winner = None
    question_error = None
    solved_event = None
    stopped_for_target = False

    def propose(event, rollout):
        candidate = str(event["answer"])
        meter.inc("candidate_proposals")
        if candidate not in seen:
            seen.add(candidate)
            candidates.put_nowait(
                {
                    **event,
                    "candidate": candidate,
                    "rollout": rollout,
                    "observed_at_utc": utc_now(),
                    "question_elapsed_s": time.perf_counter() - start,
                    "_enqueued_monotonic_s": time.perf_counter(),
                }
            )
            meter.inc("candidate_unique_enqueued")
            meter.high_water("candidate_queue_depth", candidates.qsize())
        else:
            meter.inc("candidate_duplicates_suppressed")

    async def generate(rollout, plan):
        scope = rollout_meters[rollout] = Meter(meter.enabled, parent=meter)
        scope.inc("generation_requests")

        def detect(part, value, eof=False):
            with scope.measure("candidate_parse_enqueue"):
                for event in detector.feed(part, value, eof=eof):
                    propose(event, rollout)

        path = folder / f"rollout-{rollout:02d}"
        if not artifacts.buffered:
            path.mkdir()
        request = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": problem["problem"]},
            ],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": plan["target_generated_tokens"],
            "seed": args.seed + index * args.seed_stride + rollout,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "return_token_ids": True,
        }
        if args.disable_thinking:
            request["chat_template_kwargs"] = {"enable_thinking": False}
        continuation = plan["continuation"]
        endpoint = "/v1/chat/completions"
        if continuation:
            endpoint = "/v1/completions"
            request.pop("messages")
            request.pop("chat_template_kwargs", None)
            request.update(
                prompt=continuation["prompt"],
                max_tokens=min(plan["target_generated_tokens"] - continuation["generated_tokens"],
                               continuation["remaining_context"]),
            )
        write_json(path / "request.json", request, scope)
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
            "continuation_of_rollout": (
                continuation["parent_rollout"] if continuation else None
            ),
            "requested_max_tokens": request["max_tokens"],
            "budget_stage": plan["stage"],
            "target_generated_tokens": plan["target_generated_tokens"],
        }
        detector = CandidateDetector()
        parts = {"reasoning": [], "content": []}
        prompt_token_ids = list(continuation["prompt"]) if continuation else None
        output_token_ids = []
        if continuation:
            detector.feed("content", continuation["visible_text"])
        records.append((path, record, parts))
        try:
            with artifacts.open_jsonl(path / "stream.jsonl", "w") as stream_file:
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
                        if scope.enabled:
                            scope.inc("sse_chunks")
                            scope.inc("sse_payload_bytes", len(payload.encode()))
                        with scope.measure("stream_trace_write_flush"):
                            stream_file.append(
                                {
                                    "elapsed_s": elapsed,
                                    "timestamp_utc": utc_now(),
                                    "data": payload,
                                },
                            )
                        if payload == "[DONE]":
                            record["done_received"] = True
                            break
                        with scope.measure("sse_json_decode"):
                            body = json.loads(payload)
                        if body.get("error"):
                            raise RuntimeError(f'vLLM stream error: {body["error"]}')
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
                            if endpoint == "/v1/completions":
                                delta = {"content": choice.get("text", "")}
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
                                    detect(part, value)
                            if choice.get("finish_reason"):
                                record["finish_reason"] = choice["finish_reason"]
                    if record["finish_reason"] in ("error", "abort"):
                        raise RuntimeError(
                            f'vLLM terminated with {record["finish_reason"]}'
                        )
                    if record["done_received"] or record["finish_reason"] in (
                        "stop",
                        "length",
                    ):
                        # A token cap may cut Answer: 070 after the first digit.
                        # Only a natural stream end can complete an unfinished line.
                        if record["finish_reason"] != "length":
                            for part in parts:
                                detect(part, "", eof=True)
                        record["status"] = "completed"
                    else:
                        raise RuntimeError(
                            "Stream ended without DONE or a terminal finish reason"
                        )
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            scope.inc("generation_" + record["status"])
            ended = time.perf_counter()
            record.update(
                generation_finished_at_utc=utc_now(),
                generation_end_monotonic_s=ended,
                generation_latency_s=ended - began,
                generation_censored=record["status"] != "completed"
                or record["finish_reason"] == "length",
            )
            visible_text = "".join(parts["reasoning"]) + "".join(parts["content"])
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
            complete_ids = bool(
                prompt_token_ids
                and output_token_ids
                and usage.get("completion_tokens") == len(output_token_ids)
            )
            write_json(
                path / "tokens.json",
                {
                    "prompt_token_ids": prompt_token_ids,
                    "output_token_ids": output_token_ids,
                    "complete": complete_ids,
                    "visible_text": (
                        continuation["visible_text"] if continuation else ""
                    )
                    + visible_text,
                },
                scope,
            )
            write_json(
                path / "response.json",
                {part: "".join(text) for part, text in parts.items()},
                scope,
            )
            record["overhead"] = scope.snapshot()
            write_json(path / "telemetry.json", record, scope)

        if record["status"] == "error":
            raise RuntimeError(record["error"])
        if record["status"] == "completed" and record["finish_reason"] == "length" and plan["stage"] + 1 < len(args.token_budgets):
            if not complete_ids:
                raise RuntimeError("Cannot expand capped trajectory without complete exact token IDs")
            prefix = prompt_token_ids + output_token_ids
            original_prompt_tokens = continuation["base_prompt_tokens"] if continuation else len(prompt_token_ids)
            generated = len(prefix) - original_prompt_tokens
            remaining = args.max_context_tokens - len(prefix)
            next_stage = plan["stage"] + 1
            if remaining > 0 and args.token_budgets[next_stage] > generated:
                return {"stage": next_stage,
                        "target_generated_tokens": args.token_budgets[next_stage],
                        "continuation": {"prompt": prefix, "parent_rollout": rollout,
                                         "visible_text": (continuation["visible_text"] if continuation else "") + visible_text,
                                         "remaining_context": remaining,
                                         "base_prompt_tokens": original_prompt_tokens,
                                         "generated_tokens": generated}}
        return None

    pool.register(index, generate, lambda: candidates.put_nowait(None))
    try:
        async with asyncio.timeout(args.question_timeout):
            with artifacts.open_jsonl(folder / "verification.jsonl") as file:
                while True:
                    event = await candidates.get()
                    if event is None:
                        break
                    queue_wait = time.perf_counter() - event.pop(
                        "_enqueued_monotonic_s"
                    )
                    meter.observe("candidate_local_queue_wait", queue_wait)
                    event.update(
                        verification_started_at_utc=utc_now(),
                        round=round_no,
                        candidate_queue_wait_s=queue_wait,
                    )
                    verify_start = time.perf_counter()
                    meter.inc("verification_submitted")
                    if profiler:
                        profiler.submitted(
                            verify_start
                            - (attempt_start if attempt_start is not None else start)
                        )
                    pool.submitted(index, event, verify_start)
                    try:
                        response = await client.post(
                            args.grader_url + "/verify",
                            json={
                                "index": index,
                                "candidate": event["candidate"],
                                "agent_id": f'{output.name}-q{index}-r{event["rollout"]}',
                                "query_id": f'{output.name}-q{index}-a{event["candidate"]}',
                            },
                        )
                        response.raise_for_status()
                        verdict = response.json()
                        if type(verdict.get("verdict")) is not bool:
                            raise RuntimeError(
                                "Grader response lacks a boolean verdict"
                            )
                        event["result"] = verdict
                        meter.inc("verification_completed")
                        meter.inc(
                            "verification_correct"
                            if verdict["verdict"]
                            else "verification_wrong"
                        )
                        if verdict.get("queue_wait_s") is not None:
                            meter.observe("grader_queue_wait", verdict["queue_wait_s"])
                        if verdict.get("toll_s") is not None:
                            meter.observe("grader_service", verdict["toll_s"])
                    except asyncio.CancelledError:
                        event["cancelled"] = True
                        meter.inc("verification_cancelled")
                        raise
                    except Exception as exc:
                        meter.inc("verification_errors")
                        event["error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        event.update(
                            verification_finished_at_utc=utc_now(),
                            verification_latency_s=time.perf_counter() - verify_start,
                        )
                        meter.observe(
                            "verification_http_wait", event["verification_latency_s"]
                        )
                        with meter.measure("verification_trace_write_flush"):
                            file.append(event)
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
                        with artifacts.open_jsonl(output / "solved.jsonl") as solved_file:
                            solved_file.append(solved_event)
                        if on_solved:
                            on_solved(solved_event)
                        break
    except asyncio.CancelledError:
        stopped_for_target = target_event is not None and target_event.is_set()
        question_error = None if stopped_for_target else "attempt interrupted"
        raise
    except Exception as exc:
        question_error = f"{type(exc).__name__}: {exc}"
    finally:
        cancellation_start = time.perf_counter()
        await pool.close_question(index)
        meter.observe(
            "generation_cancellation_settlement",
            time.perf_counter() - cancellation_start,
        )
        ended, ended_at = time.perf_counter(), utc_now()
        for path, record, _ in records:
            record.update(
                finished_at_utc=ended_at,
                end_to_end_latency_s=ended - record["start_monotonic_s"],
                end_to_end_scope="rollout start through question verification/cancellation settlement",
                gpu=None,
            )
            with meter.measure("gpu_sample_window"):
                record["gpu"] = sampler.window(
                    record["start_monotonic_s"], record["generation_end_monotonic_s"]
                )
            record["overhead"] = rollout_meters[record["rollout"]].snapshot()
            write_json(path / "telemetry.json", record)
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
                    else ("stopped" if stopped_for_target else "unsolved")
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
        if any(r["status"] == "error" for r in result["rollouts"]) and not winner:
            result["status"] = "error"
        result.update(
            round=round_no, first_solved=solved_event, overhead=meter.snapshot()
        )
        write_json(folder / f"round-{round_no:02d}.json", result)
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
        result["overhead"] = merge_meters(
            previous.get("overhead") if previous else None, meter.snapshot()
        )
        result["rounds"] = rounds
        result["question_start_monotonic_s"] = (
            previous.get("question_start_monotonic_s", start) if previous else start
        )
        result["end_to_end_latency_s"] = ended - result["question_start_monotonic_s"]
        write_json(folder / "question.json", result)
        print(
            f'Q{index:02d} round {round_no}: {result["status"]}, {ended - start:.2f}s',
            flush=True,
        )
    return result
