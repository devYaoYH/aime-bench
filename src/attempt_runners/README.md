# Attempt runner versions

Keep experimental policies in this directory. The canonical runner stays at
`src/attempt.py`; do not add experimental baseline behavior or switches to it.

| Runner | Entry point | Behavior |
| --- | --- | --- |
| [Canonical](../attempt.py) | `python -m src.attempt` | Original streaming candidate extraction, fan-out, and coverage/continuation modes |
| [Naive pass@4 v1](naive_pass4_v1.py) | `python -m src.attempt_runners.naive_pass4_v1` | One batch, 30 questions × four samples; naturally completed final boxes only; stop at 18 correct |

The naive runner is isolated from the canonical module. Its service, warmup,
locking, GPU sampling, and SSE helpers use the frozen [_runtime_v1.py](_runtime_v1.py)
snapshot of the helpers used in commit `0834cfe`. Both runners write the existing
timestamped `attempts/` artifact format and use the same checkout lock and service
ports, so they must run sequentially on the same GPU. The existing attempt viewer
reads either runner's outputs.

For the v1 baseline, deploy the dedicated
[16K profile](../../configs/vllm/WeiboAI/VibeThinker-3B/vllm-baseline-16k.yaml)
to `~/models/WeiboAI/VibeThinker-3B/vllm-baseline-16k.yaml` after pulling:

```sh
~/.venvs/vllm/bin/python -m src.attempt_runners.naive_pass4_v1 \
  --model WeiboAI/VibeThinker-3B
```

Defaults are 30 questions in parallel, four independent samples per question,
18 verified correct as the stopping target, and a four-request per-question
budget. The profile sets 16K total context and an 80% GPU-memory budget. Output
limits reserve the actual prompt length. The runner uses a standard final-answer
prompt, ignores intermediate candidates, and leaves capped responses ungraded.
`--model-profile` selects another YAML filename within that model's directory.

New config and summary files record `runner_id: naive_pass4_v1` and the module
entry point, alongside the source commit, profile, budgets, and timing. Change an
experimental policy by adding a new version (for example `naive_pass4_v2.py`),
with its own tests and a catalogue row; preserve prior versions for comparisons.
Do not monkey-patch or import canonical runner behavior into a versioned policy.

The [recorded v1 baseline](../../attempts/20261003T202152.418590Z/README.md)
originally ran through `src.attempt --strategy baseline` at commit `0834cfe`.
Its saved config, summary, source commit, and original command remain historical
evidence. The policy is now separately runnable here; splitting the code did not
rerun or relabel that experiment.
