# V1.6: four fresh samples, continuations to 64K

This version uses v1.5-style allocation and v1's integer extraction and grading.
Each question has **at most four fresh samples**. Continuation requests do not
consume that allowance. Each fresh trajectory starts with up to 8,192 output
tokens, then sends **one exact-token continuation for the remaining output
budget**, with no further 16K boundaries. It runs until natural completion or its
cumulative 65,536-output-token budget.
The served total context includes the prompt: on the existing 64K-context model,
available output is slightly less than 64K. All generated tokens in the trajectory
count toward its budget. A natural end or exhausted context/budget can free a
slot for another fresh sample; a correct verdict cancels every stream for that
question. The overall attempt stops at 18 distinct correct verdicts by default.

The initial phase launches exactly one 8K request per selected question (30×1
for AIME 2025), with a **coverage barrier**. Freed slots stay idle until all initial
generations and their queued grader checks settle, matching v1’s first-round
barrier. Correct verdicts still cancel their question immediately, and reaching
the solve target ends the attempt without opening the pool.

After that one barrier, the pool opens with a slot count defaulting to the
selected question count (30 for AIME 2025). Ready continuations take priority,
followed by fresh samples on least-active unsolved questions, with rotating ties.
There are no subsequent global barriers: slots refill as requests end or
questions solve. Multiple fresh trajectories for one
question may run concurrently, but their cumulative fresh count never exceeds
four. This is a finite allowance, not a requirement to use all four.

```bash
~/.venvs/vllm/bin/python -m runner.extensions.v1_6 \
  --benchmark-year 2025 --seed 20261011 --benchmark
```

Use the prepared remote environments and the dedicated 95%-memory
`vllm-v1_6-long64k.yaml` profile from
[the committed configuration](../../../configs/vllm/r0b0tlab/VibeThinker-3B-NVFP4/vllm-v1_6-long64k.yaml).
Its output ceiling is 65,536; total context stays 65,536. Provision it under the
matching `~/models/<organization>/<model>/` directory. Older profiles stay
unchanged. The runner owns warmup, inference/grader services and cleanup, or
can attach to an idle matching server with `--reuse-server`.

| Control | Default | Meaning |
| --- | ---: | --- |
| `--max-fresh-samples-per-question` | 4 | Total fresh trajectories, allowed range 1–4 |
| `--max-rollout-tokens` | 65536 | Cumulative output per trajectory, clipped to served total context |
| `--max-concurrent-requests` | Question count | Shared inference slots; must fit initial question coverage |
| `--first-pass-max-tokens` | 8192 | First segment of every fresh trajectory |
| `--model-profile` | `vllm-v1_6-long64k.yaml` | Dedicated profile with a 64K output ceiling |
| `--question-timeout` | 1800s | Overall generation/verification timeout for each question |

`--max-attempts-per-question` is an alias for the fresh-sample limit **only in
this extension**. `--max-tokens` aliases the cumulative `--max-rollout-tokens`
budget; it is not an additional segment size. `--max-rounds`, `--schedule`, `--rollouts`, `--parallelism` and
`--no-continuation` are rejected because they describe different scheduling
contracts. Other service, dataset, sampling, prompt and benchmark flags are
shared with [canonical usage](../../README.md). Custom datasets use the same
gold-free grader question API and retain the integer 0–999 answer restriction.

The improved prompt and shared sampling defaults come from the canonical v1
preset. Separate [policy defaults](policy.json), their hash, actual slot count,
served prompt lengths, resolved configuration and source identity are saved.
Fresh seeds retain `base_seed + question_index * 4 + fresh_sample_id`.
The continuation uses a disjoint seed band whose stride is four times one
plus the maximum selected question index. Changing which sibling finishes first
therefore does not renumber another trajectory's seeds.

Saved `rollout-NN` folders index **HTTP segments**. Their telemetry identifies
`fresh_sample`, `segment`, `continuation_of_rollout`, generated tokens before the
segment and cumulative trajectory tokens. `allocation.json` records each
admission, its coverage/pool phase, the barrier release time, and separate
fresh/request counters. Summaries likewise distinguish
fresh samples from continuation requests. There is no independent four-HTTP-call
or four-round cutoff. For example, one sample with a large enough context can
use 8K + 56K = 64K across two HTTP requests; four such samples can use eight
requests. Actual budgets are clipped for the prompt/context. There is one HTTP
handoff after the initial capped request, not recurring 16K handoffs.

The unbenchmarked prototype at `b7214b3e` used recurring 16K segments. This
revision changes that planner and its dedicated profile before GPU measurements;
the earlier source remains available in Git.

The [manifest](manifest.json) pins this policy, the shared extension lifecycle
and the unchanged canonical primitive dependency. The current canonical v1,
v1.1 and v1.5 policies/manifests remain unchanged. Launch v1.6 using its direct
module; the frozen canonical selector has not been changed to register/promote it.
The corrected coverage-barrier policy was tested on the GPU with all five
historical seeds: **5/5 reached 18; median 77.498s; range 63.445–101.123s**.
Compared with the latest refactored-v1 batch, the worst run fell from 128.744s
and the sample SD fell from 26.475s to 15.049s. Original v1 and v1.5 had tighter
spreads; the median is essentially unchanged versus original v1. The combined
policy and continuation seed changes prevent attributing the observed tail
reduction to the initial barrier alone. See the [full comparison](../../../runs/experiments/core-v1_6-barrier-five-seeds-20261004T224850Z/README.md).

Validation: 391 offline suite checks passed from the staged-source snapshot;
all 19 focused policy/batch checks passed after adding the named-profile control
check. Unrelated working-tree edits were excluded from the suite snapshot.

The initial-coverage barrier is covered by offline tests for freed slots, queued
wrong verdicts, excess configured capacity and cancellation. The batch comparison
uses the original five seeds without replacement.

Run the focused checks with:

```bash
.venv/bin/python -m unittest test.test_speedrun_v1_6 -v
```

## Difference from v1.1 and v1.5

| Policy | Scheduling | Request budget | Four-count interpretation |
| --- | --- | --- | --- |
| v1.1 | Eager, at most one live generation per question; freed solved slots do not create siblings | One long request, up to 64K total context; fresh retry after it ends and queued checks finish | Four fresh requests; no continuations |
| v1.5 | Fixed 30-slot pool; continuations first, then least-active fresh samples | First request per question 8K; subsequent requests up to 16K additional output | Four HTTP requests including continuations |
| v1.6 | Initial coverage/check barrier, then question-count pool with the same admission priorities | Every fresh sample starts at 8K; one long continuation to cumulative 64K/context cap | Four fresh trajectories; the continuation is separate |

The original five-seed medians were 77.652s for v1.1 and 77.352s for v1.5.
Their different budgets and allocation policies prevent attributing that small
difference to a single change. These measurements do not establish a v1.6 speedup.

## Five-seed comparison batch

On the idle remote GPU, from a clean worktree of the pushed source:

```bash
~/.venvs/vllm/bin/python -m runner.extensions.validation.v1_6_batch \
  --grader-python /home/azureuser/aime-bench/grader/.venv/bin/python
```

This scores seeds 20261011–20261015, reuses one owned server, resets prefix cache
and performs cheap arithmetic warmup before each trial, and starts a fresh grader.
All failures and unmet targets remain in the batch; there are no replacement
seeds. Server startup and trace flush are outside solving latency.
A 600-second per-trial wall safety timeout includes initialization.

## BF16 with FlashInfer comparison

For the unquantized `WeiboAI/VibeThinker-3B` weights on the prepared A100, use the
separate committed `vllm-v1_6-bf16-flashinfer.yaml` profile. It retains 95% memory,
64K context/output ceilings, BF16 KV and FlashInfer attention, while unquantized
linear layers use automatic native dispatch with no Marlin override.

```bash
~/.venvs/vllm/bin/python -m runner.extensions.validation.bf16_v1_6 \
  --seed 20261011 \
  --grader-python /home/azureuser/aime-bench/grader/.venv/bin/python
```

This runs **one** attempt against the same-seed NVFP4 v1.6 control. It owns the
server, resets prefix cache, runs the same cheap warmup, starts a fresh grader,
and retains failures. It checks tokenizer/architecture/RoPE agreement and the
unchanged v1.6 controls; inference math and native linear kernels differ. The
quantization model card identifies the base model but does not pin its base
revision, so this is a comparison of the downloaded deployments. One seed does
not establish a repeatable quantization speedup.
