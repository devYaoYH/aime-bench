# Post-freeze measurements v1

The protocol is [versioned](../../configs/experiments/post-freeze-measurements-v1.json).
Plan without GPU work:

```sh
python -m src.experiments.post_freeze
```

After local tests, commit/push, remote pull and an idle-GPU check, execute from a
clean pinned checkout in a durable tmux session:

```sh
~/.venvs/vllm/bin/python -m src.experiments.post_freeze --execute --batch <unique-batch>
```

Task A retains the immutable core v1 and improved prompt. Five declared seeds
20261011–20261015 reuse one owned NVFP4/FlashInfer server, with a cache reset,
fresh grader and cheap 30×32-token warmup before each trial. Only the target
changes to 30. An external watchdog cancels at 900 official seconds, excluding
initialization and warmup. Natural exhaustion and failures remain outcomes;
there are no replacement trials.

Task B uses the separate full-batch naive runner at BF16, 16K total context and
95% GPU allocation, seed 20261003. All 120 samples run to natural stop or token
cap. Its prompt, sampling and last-integer-box extractor match the historical
naive runner. Capped responses are ungraded. Completed final candidates stream
to the grader; each question/answer pair is graded once, and its verdict scores
every matching sample. No correct verdict or solve target cancels generation.
Pass@1 averages sample correctness; pass@4 counts questions with a correct
sample. Voting uses the unique plurality of extracted integers, with missing
answers abstaining and ties/no votes incorrect.

The user approved retaining the frozen startup answer-field provenance check
only. Model inputs and analysis exclude reference answers, and correctness
comes exclusively from grader verdicts. The grader remains serial with a
three-second toll. Both tasks execute sequentially and own their services.
Task A has a 75-minute official-time ceiling plus setup; Task B provisionally
needs 10–20 minutes including generation and grading.

Each batch records commands, source and configuration hashes, all outcomes,
first-correct milestones, baseline sample verdict mappings and timestamps.
Canonical traces remain under `attempts/`; service logs and grader audits
remain on the remote and out of Git. Missing milestones stay null, with reach
counts reported alongside reached-trial medians. Setup and unattended compute
time are recorded separately from development activity. Frozen cores, the
original baseline and the grader are unchanged.
