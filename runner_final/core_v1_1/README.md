# Core v1.1 — prepared, unrun

This experimental version starts one long fresh rollout per question, with eager
retries and no round barrier. It has **no GPU results** and does not replace the
measured core v1 submission policy. Preparing it does not change or deploy the
core v1 runner, model profile, or profiling setup.

- Initially run all 30 questions, one request each. At most 30 requests may run
  concurrently, with at most one generation per question. Solved slots do not
  create siblings for other questions; active concurrency can fall below 30.
- Each request allows up to **65,536 total context tokens**, including the chat
  prompt. Count the actual served chat template through `/tokenize` before the
  solving timer, then request at most `65536 - prompt_tokens` output tokens.
- Keep core v1's prospective candidate detector and grader handling. Incorrect
  candidates do not interrupt a live generation. A correct verdict cancels that
  question's generation. Stop the attempt at 18 distinct verified answers.
- Once generation ends, wait for all queued candidate verdicts. If none is
  correct (or no candidate was emitted), start a **fresh** sample immediately,
  without waiting for other questions. A token-limited ending also retries fresh;
  this version sends no exact-prefix continuation requests.
- Allow four total generation requests per question, including its first request.
  Preserve v1's prompt, question/sample seed mapping, sampling, short arithmetic
  warmup, model, attention backend, KV dtype, grader cost, and benchmark storage.

This changes both the 8K first-request boundary and the round barrier. It is not
an isolated test of removing the barrier. Longer contexts may increase KV traffic
and decoding cost; offline tests establish policy behavior, not a speedup.

Entrypoint: `python -m runner_final.run_v1_1`. Its default preset is
`runner_final/presets/prompt_adherence_v1_1.json`; the equivalent versioned entry
point is `python -m src.attempt_runners.speedrun_v1_1`.

The separate, undeployed model profile is
`configs/vllm/r0b0tlab/VibeThinker-3B-NVFP4/vllm-v1_1-long64k.yaml`. Its only launch
setting difference from the v1 FlashInfer profile is raising the generation
ceiling from 16K to 64K; total context stays 64K. Before a future authorized run,
commit/push the tested version, pull remotely, and place this profile under the
matching model directory. Do not change the standard profile or use an occupied
profiling server. No deployment or scored run is part of this preparation.

Run the policy checks offline with:

```sh
.venv/bin/python -m unittest test.test_speedrun_v1_1 -v
```

The manifest pins the new policy and its unchanged frozen v1 runtime dependencies.
Prompt, preset, model profile, served prompt lengths, and resolved launch controls
are recorded in attempt artifacts when an execution is separately authorized.
