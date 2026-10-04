"""Long streaming rollouts in a shared pool with queue-aware wrong-verdict forks.

V2.1 syntax and expression keys, v2.2 feedback, and exact token evidence are
retained. Admissions and feedback continuations share the input-sized slot pool.
"""

import asyncio
import json
import hashlib
import re
import time
from src.attempt_storage import AttemptArtifacts
from src.common import atomic_json, utc_now
from src.attempt_metrics import Meter, merge_meters
from runner_final.core_v2_3._runtime import sse_payloads

PROMPT = (
    "Solve the mathematics problem. Whenever you have a prospective exact answer "
    "to the original requested quantity, immediately emit it as \\boxed{EXPRESSION}. "
    "Preserve variables, fractions and radicals; put only the expression in the box. "
    "You may continue checking your work afterward. End with your final boxed answer."
)


from runner_final.core_v2_3.extraction import CandidateDetector
from runner_final.core_v2_3.syntax import with_validator
from runner_final.core_v2_3.feedback import feedback_text, tokenize_feedback
from runner_final.core_v2_3.budgets import prompt_tokens


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


@with_validator
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
    syntax_validator=None,
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
    if pool is None:
        raise ValueError("V2.3 question execution requires the shared slot pool")
    if artifacts.has_json(folder / "question.json"):
        raise ValueError("V2.3 starts a fresh attempt; do not resume question artifacts")
    previous = None
    start, started = time.perf_counter(), utc_now()
    candidates = asyncio.Queue()
    seen = set(previous.get("candidate_keys", [])) if previous else set()
    accepted_answers = set(previous.get("candidate_answers", [])) if previous else set()
    records = []
    winner = None
    question_error = None
    solved_event = None
    stopped_for_target = False
    pending_detections = 0
    queued_unique = 0
    rejected_answers = list(previous.get("rejected_answers", [])) if previous else []
    pending_feedback = []
    streams = {}
    live = {}
    lanes = {}
    blocked_signature = None
    wake_queued = False

    def wake():
        nonlocal wake_queued
        if not wake_queued:
            wake_queued = True
            candidates.put_nowait({"_wake": True})

    def feedback_log(kind, **fields):
        with artifacts.open_jsonl(folder / "feedback.jsonl") as audit:
            audit.append({"kind": kind, "timestamp_utc": utc_now(), "round": round_no,
                          "question_elapsed_s": time.perf_counter() - start, **fields})

    async def fork_after_wrong():
        """Pause admissions during preparation; enqueue forks in the shared pool."""
        nonlocal blocked_signature
        if not pending_feedback:
            return
        if queued_unique or pending_detections:
            feedback_log("deferred", queued_unique=queued_unique,
                         pending_detections=pending_detections, answers=list(pending_feedback))
            return
        state = pool.states[index]
        remaining = max(0, args.max_attempts_per_question - state["used"] - len(state["ready"]))
        active = [r for r, task in streams.items() if not task.done()]
        eligible = [r for r in active if live.get(r, {}).get("complete")
                    and not live[r].get("terminal")]
        signature = (tuple(pending_feedback), tuple(eligible), remaining)
        if not remaining or not eligible or args.no_continuation or not pool.eligible(index):
            if signature != blocked_signature:
                feedback_log("not_forked", reason=("request_cap" if not remaining else
                             "continuations_disabled" if args.no_continuation else "no_live_exact_prefix"),
                             active=active, remaining_requests=remaining, answers=list(pending_feedback))
                blocked_signature = signature
            return
        if signature == blocked_signature:
            return
        # Reserving eligibility prevents fresh admissions for this question from
        # consuming its last request budget while /tokenize awaits the server.
        pool.pause(index, True)
        try:
            text = feedback_text(rejected_answers)
            prepared = time.perf_counter()
            try:
                tokens = await tokenize_feedback(client, args, text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                blocked_signature = signature
                feedback_log("not_forked", reason="tokenize_error", error=str(exc))
                return
            feedback_log("tokenized", answers=list(rejected_answers), token_count=len(tokens),
                         preparation_s=time.perf_counter() - prepared)
            if queued_unique or pending_detections:
                feedback_log("deferred_after_tokenize", queued_unique=queued_unique,
                             pending_detections=pending_detections)
                return
            remaining = max(0, args.max_attempts_per_question - state["used"] - len(state["ready"]))
            selected = [r for r in eligible if not streams[r].done() and live[r]["complete"]
                        and not live[r].get("terminal")
                        and len(live[r]["prompt"]) + len(live[r]["output"]) + len(tokens)
                        < args.max_context_tokens][:remaining]
            if not selected:
                blocked_signature = signature
                feedback_log("not_forked", reason="ended_or_context_full")
                return
            batch = list(pending_feedback)
            for r in selected:
                pool.correction_cancellations.add(streams[r])
                streams[r].cancel()
            await asyncio.gather(*(streams[r] for r in selected), return_exceptions=True)
            queued = False
            for parent in selected:
                saved = artifacts.read_json(folder / f"rollout-{parent:02d}" / "tokens.json")
                prefix = saved["prompt_token_ids"] + saved["output_token_ids"]
                if not saved["complete"] or len(prefix) + len(tokens) >= args.max_context_tokens:
                    feedback_log("not_forked", reason="final_prefix_incomplete_or_full", parent=parent)
                    continue
                continuation = {
                    "prompt": prefix + tokens, "parent_rollout": parent,
                    "visible_text": saved["visible_text"] + text,
                    "remaining_context": args.max_context_tokens - len(prefix) - len(tokens),
                    "feedback_answers": list(rejected_answers),
                    "lane_id": lanes[parent],
                }
                queued |= pool.queue_correction(index, {
                    "stage": live[parent]["stage"] + 1, "target_generated_tokens": args.max_tokens,
                    "continuation": continuation})
                feedback_log("fork_queued", parent_rollout=parent, lane_id=lanes[parent],
                             batch_answers=batch, all_rejected_answers=list(rejected_answers),
                             prefix_token_count=len(prefix), feedback_token_ids=tokens,
                             exact_parent_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest())
            if queued:
                pending_feedback.clear()
                blocked_signature = None
        finally:
            pool.pause(index, False)

    async def propose(event, rollout):
        nonlocal queued_unique
        candidate = str(event["answer"])
        observed_at = utc_now()
        observed_s = time.perf_counter()
        meter.inc("candidate_proposals")
        validation = await syntax_validator.validate(candidate)
        if not validation["valid"]:
            outcome = "rejected"
            meter.inc("candidate_syntax_rejected")
        elif validation["canonical_key"] in seen:
            outcome = "duplicate_expression"
            meter.inc("candidate_duplicates_suppressed")
        else:
            outcome = "enqueued"
            seen.add(validation["canonical_key"])
            accepted_answers.add(candidate)
            queued_unique += 1
            candidates.put_nowait({
                **event, "candidate": candidate, "rollout": rollout,
                "canonical_key": validation["canonical_key"],
                "observed_at_utc": observed_at,
                "question_elapsed_s": observed_s - start,
                "_enqueued_monotonic_s": time.perf_counter(),
            })
            meter.inc("candidate_unique_enqueued")
            meter.high_water("candidate_queue_depth", candidates.qsize())
        # Required evidence remains present even with optional profiling disabled.
        with artifacts.open_jsonl(folder / "candidate_validation.jsonl") as audit:
            audit.append({
                **event, "candidate": candidate, "rollout": rollout,
                "observed_at_utc": observed_at, "validated_at_utc": utc_now(),
                "question_elapsed_s": observed_s - start,
                "outcome": outcome, **validation,
            })

    async def generate(rollout, plan):
        streams[rollout] = asyncio.current_task()
        continuation = plan["continuation"]
        lanes[rollout] = (continuation.get("lane_id", continuation["parent_rollout"])
                          if continuation else rollout)
        request_feedback = list(rejected_answers)
        scope = rollout_meters[rollout] = Meter(meter.enabled, parent=meter)
        scope.inc("generation_requests")

        async def detect_many(values, eof=False):
            nonlocal pending_detections
            with scope.measure("candidate_parse_enqueue"):
                found = [event for part, value in values
                         for event in detector.feed(part, value, eof=eof)]
            # Count the whole chunk before awaiting the first parse: a second
            # box in that same chunk may already be a self-corrected answer.
            pending_detections += len(found)
            left = len(found)
            try:
                for event in found:
                    await propose(event, rollout)
                    pending_detections -= 1
                    left -= 1
            finally:
                pending_detections -= left
                if found:
                    wake()

        path = folder / f"rollout-{rollout:02d}"
        if not artifacts.buffered:
            path.mkdir()
        request = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": problem["problem"] + (feedback_text(request_feedback) if request_feedback else "")},
            ],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed + index * args.max_attempts_per_question + rollout,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "return_token_ids": True,
        }
        if args.disable_thinking:
            request["chat_template_kwargs"] = {"enable_thinking": False}
        if continuation and request_feedback and continuation.get("feedback_answers") != request_feedback:
            text = feedback_text(request_feedback)
            tokens = await tokenize_feedback(client, args, text)
            if len(continuation["prompt"]) + len(tokens) >= args.max_context_tokens:
                continuation = None  # fresh request carries the same feedback
                lanes[rollout] = rollout
            else:
                continuation = {**continuation, "prompt": continuation["prompt"] + tokens,
                    "remaining_context": continuation["remaining_context"] - len(tokens),
                    "visible_text": continuation["visible_text"] + text,
                    "feedback_answers": request_feedback}
        endpoint = "/v1/chat/completions"
        if continuation:
            endpoint = "/v1/completions"
            request.pop("messages")
            request.pop("chat_template_kwargs", None)
            request.update(
                prompt=continuation["prompt"],
                max_tokens=min(args.max_tokens, continuation["remaining_context"]),
            )
        else:
            count = (problem.get("prompt_tokens") if not request_feedback else None)
            if count is None:
                count = await prompt_tokens(client, args, request["messages"])
            if type(count) is not int or not 0 < count < args.max_context_tokens:
                raise ValueError("Invalid reserved served prompt budget")
            request["max_tokens"] = min(args.max_tokens, args.max_context_tokens - count)
        write_json(path / "request.json", request, scope)
        began = time.perf_counter()
        record = {
            "rollout": rollout,
            "round": round_no,
            "lane_id": lanes[rollout],
            "feedback_answers": continuation.get("feedback_answers", []) if continuation else request_feedback,
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
            "context_budget_scope": "output clipped to 65536 total context including prompt and feedback",
        }
        detector = CandidateDetector()
        parts = {"reasoning": [], "content": []}
        prompt_token_ids = list(continuation["prompt"]) if continuation else None
        output_token_ids = []
        live[rollout] = {"prompt": prompt_token_ids, "output": output_token_ids, "complete": False, "stage": plan["stage"]}
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
                            live[rollout].update(prompt=prompt_token_ids,
                                complete=bool(prompt_token_ids and output_token_ids and
                                              body["usage"].get("completion_tokens") == len(output_token_ids)))
                        for choice in body.get("choices", []):
                            if choice.get("index", 0) != 0:
                                continue
                            if choice.get("prompt_token_ids") is not None:
                                prompt_token_ids = choice["prompt_token_ids"]
                            if choice.get("token_ids"):
                                output_token_ids.extend(choice["token_ids"])
                            live[rollout].update(prompt=prompt_token_ids,
                                complete=bool(prompt_token_ids and output_token_ids and
                                              (record["usage"] or {}).get("completion_tokens") == len(output_token_ids)))
                            delta = choice.get("delta") or {}
                            if endpoint == "/v1/completions":
                                delta = {"content": choice.get("text", "")}
                            if choice.get("finish_reason"):
                                record["finish_reason"] = choice["finish_reason"]
                                live[rollout]["terminal"] = True
                            arrived = []
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
                                    arrived.append((part, value))
                            await detect_many(arrived)
                            if choice.get("finish_reason"):
                                record["finish_reason"] = choice["finish_reason"]
                            if pending_feedback and live[rollout]["complete"]:
                                wake()
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
                            await detect_many([(part, "") for part in parts], eof=True)
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
        if not args.no_continuation and record["finish_reason"] == "length":
            prefix = continuation_prefix({"rollouts": [record]}, folder,
                                         args.max_context_tokens, artifacts)
            if prefix:
                prefix["lane_id"] = lanes[rollout]
                return {"stage": plan["stage"] + 1,
                        "target_generated_tokens": args.max_tokens,
                        "continuation": prefix}
        return None

    pool.register(index, generate, wake)

    try:
        async with asyncio.timeout(args.question_timeout):
            with artifacts.open_jsonl(folder / "verification.jsonl") as file:
                while True:
                    await fork_after_wrong()
                    if (not queued_unique and not pending_detections
                            and pool.states[index]["exhausted"]):
                        # Retrieve task exceptions (e.g. failure before a stream
                        # record exists) instead of silently losing setup errors.
                        for task in streams.values():
                            if not task.cancelled() and task.exception():
                                raise task.exception()
                        break
                    event = await candidates.get()
                    if event.get("_wake"):
                        wake_queued = False
                        continue
                    queued_unique -= 1
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
                                "query_id": f'{output.name}-q{index}-a{hashlib.sha256(event["candidate"].encode()).hexdigest()}',
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
                    if not verdict["verdict"]:
                        rejected_answers.append(event["candidate"])
                        pending_feedback.append(event["candidate"])
                        feedback_log("wrong_verdict", candidate=event["candidate"],
                                     query_id=verdict.get("query_id"), rollout=event["rollout"])
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
            "candidate_answers": sorted(accepted_answers),
            "candidate_keys": sorted(seen),
            "rejected_answers": rejected_answers,
            "pending_feedback": pending_feedback,
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
