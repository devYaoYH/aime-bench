# Qwen3.5 Python-tool pilot

Two selected questions from `runs/20260930-155212` are paired with and without
Python using `qwen/qwen3.5-35b-a3b` through OpenRouter, pinned to Parasail with
fallback disabled. Q23 is coin-change optimality; Q25 counts seating subsets.
This is a new model, so historical Qwen3-30B-A3B timings are not a matched baseline.

## Native tool interface and multiple steps

Every tool-enabled request includes one function in OpenRouter's `tools` field:

```json
{
  "type": "function",
  "function": {
    "name": "python_math",
    "description": "Execute a short Python program for exact arithmetic, finite enumeration, or dynamic programming. Print your result and compact checks.",
    "parameters": {
      "type": "object",
      "properties": {"code": {"type": "string"}},
      "required": ["code"],
      "additionalProperties": false
    }
  }
}
```

The actual definition adds allowed imports, independent-call behavior, and
resource limits. The corrected pilot uses `tool_choice: "required"` on the
first Python round, then `"auto"` on follow-ups. The system prompt asks Qwen to
formulate a short computation and call Python immediately. Neither the prompt
nor the tool schema contains answers or prewritten solution programs.

The loop is:

1. Send the question, history, and tool schema to Qwen.
2. Preserve Qwen's assistant message, tool-call IDs, and reasoning fields.
3. Parse and execute each `python_math` program in a fresh restricted process.
4. Append a `role: "tool"` message with the matching `tool_call_id` and a JSON
   result containing `ok`, `stdout`, `error`, and execution/wall timings.
5. Query Qwen again with the updated history and the same tool schema. Repeat
   until it answers or a budget is exhausted. Errors are returned as tool results
   so Qwen can correct its program in a later call.

This supports multiple tool rounds. Variables do not persist between calls, but
programs and printed results persist in the conversation. The limits are 3 tool
execution attempts, 4 API rounds, and **16,384 cumulative generated tokens** per
case. Model-emitted multiple calls in a single response are handled sequentially.
Requests beyond the tool execution budget receive an error without execution.

The no-tool arm uses the same model, provider, sampling parameters, seed, and
answer format, with a no-tools instruction instead of the tool instruction.
All four cases launch concurrently. Provider seed support does not guarantee
determinism, especially when prompts differ.

## Execution and accounting

Only `math`, `itertools`, `functools`, `collections`, `fractions`, and `decimal`
are available. The worker restricts Python syntax/builtins and runs under macOS
`sandbox-exec`, with no inherited API credentials. The OS sandbox denies network,
writes, process creation, and reads from user/home/temp locations; it permits
the Python runtime, the worker source, and necessary system resources.
Limits: 3 seconds CPU, 5 seconds wall, a 256 MiB RSS watchdog, 20,000 code
characters, and 6,000 output characters. RSS polling is not a hard allocation
limit. Unsupported environments fail before paid API calls; there is no
unsandboxed fallback.

The evaluation compares final extracted integers to saved keys **after** the
model/tool loop. It does not invoke the provided grader, SSH, or local inference.

Each raw API request/response and generated program is saved. Completion tokens
sum across every round, including reasoning and program generation. Prompt
tokens also sum across rounds, including repeated history and tool outputs.
These are different metrics: output tokens approximate decode work; total
tokens include added prefill work. Hosted wall time includes provider/network
delays and does not measure CPU/GPU contention on a local A100.

## Run and inspect

```bash
.venv/bin/python -m unittest test.test_python_tool_pilot -v
.venv/bin/python -m src.experiments.python_tools.python_tool_pilot --prepare-only
.venv/bin/python -m src.experiments.python_tools.python_tool_pilot
```

The runner needs network access and permission to start its nested OS sandbox.
Results live in `runs/python-tool-qwen35-pilot-v2/`. Completed cases can be reused;
incomplete paid cases and changed configurations require a fresh output label
to preserve their records. No automatic retry hides or discards failed usage.

Primary interface references:
[OpenRouter tool calling](https://openrouter.ai/docs/guides/features/tool-calling),
[reasoning preservation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
[model](https://openrouter.ai/qwen/qwen3.5-35b-a3b).

## Initial pilot and correction

The initial optional-tool pilot is preserved in `runs/python-tool-qwen35-pilot/`.
Q23 hit the 16,384-token cap in both arms without a tool call or final answer.
Q25's control answered 907 correctly using 13,785 output tokens in 91.06 seconds.
The Python arm generated 10,840 tokens before its first call. An overly strict
worker filter then rejected ordinary `_` loop variables; repeated calls received
the same error, so the arm exhausted its round budget without a final answer.
These failed results are not evidence for tool-enabled correctness or savings.

The filter now allows ordinary underscore-prefixed variables while retaining
restrictions on double-underscore names and private attributes. A regression
test covers the exact list-comprehension pattern. The corrected pilot explicitly
requires the first tool call to address the observed late/absent calls. Therefore
its results apply to this stronger policy, not merely making a tool available.

## Corrected pilot results

All four corrected cases completed their budgeted loops through Parasail.
**Q25 is the useful primary case:** Python can count the valid seating patterns
exactly, avoiding lengthy manual enumeration. Qwen generated a binomial formula
and a separate dynamic program. The DP tracks sequence length, occupied count,
and the trailing run of occupied chairs, disallowing runs of three. Both methods
returned 2,907; the final model response correctly gave remainder 907.

| Q25 metric | Python tool | No tools |
|---|---:|---:|
| Final answer | 907, correct | 907, correct |
| Generated tokens, all rounds | 9,726 | 15,313 |
| Prompt tokens, all rounds | 1,666 | 134 |
| Total tokens | 11,392 | 15,447 |
| Provider-reported reasoning tokens | 6,195 | 8,608 |
| Model API rounds | 2 | 1 |
| Successful Python calls | 1 | 0 |
| Case elapsed | 67.006 s | 105.288 s |
| Python execution including worker startup | 0.137 s | — |
| Reported API cost | $0.0099759 | $0.0153331 |

Q25 used **36.49% fewer output tokens**, **36.36% less elapsed time**, and
**26.25% fewer total tokens**. Those totals include the additional model query
after the tool result. The added input work was 1,532 tokens. Native computation
took only 0.00238 seconds; worker launch, sandbox, and communication account for
most of the tool's wall time.

The first Python round still generated **7,569 tokens in 50.40 seconds** before
issuing the call; the follow-up used 2,157 tokens in 16.46 seconds. Thus requiring
a call does not by itself bound the reasoning that precedes it. Constraining that
first turn is a remaining experimental opportunity, not a measured improvement.

Q23 hit 16,384 generated tokens in both arms, with no final answer and **no tool
call**, even though the Python request used `tool_choice: "required"`. Elapsed
time was 112.866 seconds with tools and 112.855 without. These capped cases offer
no evidence of a Python benefit and cannot be counted as successful solves.

The corrected four-case pilot cost $0.0582456; the preserved initial pilot cost
$0.0604874. Initial errors and capped responses remain in their original folders.
The corrected observations are a single paired sample on a selected question,
not an estimate of general speedup or progress toward 18/30. In fact, Q25's
control differed from the initial control despite using the same seed.

Raw records and summary:

- [`25-python.json`](../../runs/python-tool-qwen35-pilot-v2/25-python.json): generated
  program, tool result, both exact requests/responses, and per-round usage.
- [`25-control.json`](../../runs/python-tool-qwen35-pilot-v2/25-control.json).
- [`summary.json`](../../runs/python-tool-qwen35-pilot-v2/summary.json): all four cases.

Validation: all 43 repository tests passed, including 9 new protocol/worker
checks for reasoning/call-ID preservation, usage after errors, argument schema,
arithmetic, Python error feedback, ordinary `_` loop variables, resource/output
limits, and OS-level repository-read/network denial. The initial failed live
run exercised repeated tool-call/error-response/follow-up rounds; the corrected
Q25 run exercised a successful tool-call/result/final-answer sequence.
