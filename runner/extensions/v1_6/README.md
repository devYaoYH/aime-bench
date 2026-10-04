# V1.6: four fresh samples, continuations to 64K

This version uses v1.5-style allocation and v1's integer extraction and grading.
Each question has **at most four fresh samples**. Continuation requests do not
consume that allowance. Each fresh trajectory starts with up to 8,192 output
tokens, then continues with exact token IDs in segments of up to 16,384 output
tokens until natural completion or its cumulative 65,536-output-token budget.
The served total context includes the prompt: on the existing 64K-context model,
available output is slightly less than 64K. All generated tokens in the trajectory
count toward its budget. A natural end or exhausted context/budget can free a
slot for another fresh sample; a correct verdict cancels every stream for that
question. The overall attempt stops at 18 distinct correct verdicts by default.

The pool initially covers every selected question. Its size defaults to the
selected question count (30 for AIME 2025). As generation ends or questions solve,
ready continuations take priority, followed by fresh samples on least-active
unsolved questions, with rotating ties. Multiple fresh trajectories for one
question may run concurrently, but their cumulative fresh count never exceeds
four. This is a finite allowance, not a requirement to use all four.

```bash
~/.venvs/vllm/bin/python -m runner.extensions.v1_6 \
  --benchmark-year 2025 --seed 20261011 --benchmark
```

Use the prepared remote environments and the existing 95%-memory
`vllm-flashinfer.yaml` profile. The per-request 16K server generation ceiling
already fits these segmented continuations; a 64K single-request ceiling is
unnecessary. The runner owns warmup, inference/grader services and cleanup, or
can attach to an idle matching server with `--reuse-server`.

| Control | Default | Meaning |
| --- | ---: | --- |
| `--max-fresh-samples-per-question` | 4 | Total fresh trajectories, allowed range 1–4 |
| `--max-rollout-tokens` | 65536 | Cumulative output per trajectory, clipped to served total context |
| `--max-concurrent-requests` | Question count | Shared inference slots; must fit initial question coverage |
| `--first-pass-max-tokens` | 8192 | First segment of every fresh trajectory |
| `--max-tokens` | 16384 | Additional output per continuation segment |
| `--question-timeout` | 1800s | Overall generation/verification timeout for each question |

`--max-attempts-per-question` is an alias for the fresh-sample limit **only in
this extension**. `--max-rounds`, `--schedule`, `--rollouts`, `--parallelism` and
`--no-continuation` are rejected because they describe different scheduling
contracts. Other service, dataset, sampling, prompt and benchmark flags are
shared with [canonical usage](../../README.md). Custom datasets use the same
gold-free grader question API and retain the integer 0–999 answer restriction.

The improved prompt and shared sampling defaults come from the canonical v1
preset. Separate [policy defaults](policy.json), their hash, actual slot count,
served prompt lengths, resolved configuration and source identity are saved.
Fresh seeds retain `base_seed + question_index * 4 + fresh_sample_id`.
Continuation segments use disjoint seed bands whose stride is four times one
plus the maximum selected question index. Changing which sibling finishes first
therefore does not renumber another trajectory's seeds.

Saved `rollout-NN` folders index **HTTP segments**. Their telemetry identifies
`fresh_sample`, `segment`, `continuation_of_rollout`, generated tokens before the
segment and cumulative trajectory tokens. `allocation.json` records each
admission and separate fresh/request counters. Summaries likewise distinguish
fresh samples from continuation requests. There is no independent four-HTTP-call
or four-round cutoff. For example, one sample with a large enough context can
use 8K + 16K + 16K + 16K + 8K = 64K across five HTTP requests; four such samples
can use twenty requests. Actual budgets are clipped for the prompt/context.

The [manifest](manifest.json) pins this policy, the shared extension lifecycle
and the unchanged canonical primitive dependency. The current canonical v1,
v1.1 and v1.5 policies/manifests remain unchanged. Launch v1.6 using its direct
module; the frozen canonical selector has not been changed to register/promote it.
This implementation has offline policy/lifecycle tests, **no GPU performance
measurement yet**. Run those checks with:

Validation: all 12 focused v1.6 checks and the full 385-test offline suite passed
from an exact staged-source snapshot, excluding unrelated working-tree edits.

```bash
.venv/bin/python -m unittest test.test_speedrun_v1_6 -v
```

## Difference from v1.1 and v1.5

| Policy | Scheduling | Request budget | Four-count interpretation |
| --- | --- | --- | --- |
| v1.1 | Eager, at most one live generation per question; freed solved slots do not create siblings | One long request, up to 64K total context; fresh retry after it ends and queued checks finish | Four fresh requests; no continuations |
| v1.5 | Fixed 30-slot pool; continuations first, then least-active fresh samples | First request per question 8K; subsequent requests up to 16K additional output | Four HTTP requests including continuations |
| v1.6 | Question-count pool with the same admission priorities | Every fresh sample starts at 8K; segmented continuations to cumulative 64K/context cap | Four fresh trajectories; continuation segments are separate |

The original five-seed medians were 77.652s for v1.1 and 77.352s for v1.5.
Their different budgets and allocation policies prevent attributing that small
difference to a single change. These measurements do not establish a v1.6 speedup.
