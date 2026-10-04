# Canonical v1 package: matched five-seed validation

**5/5 reached 18 distinct verified correct questions.** The packaged runner's
median was **73.568s**, compared with **77.277s** for the same original five seeds
(-4.8%). All request and correctness audits passed. Median performance is within
the predeclared ±10% practical band, but **tail latency is not confirmed unchanged**:
one trial took 128.744s, increasing the mean from 74.995s to 82.379s (+9.8%).
Do not interpret this batch as statistical equivalence or a causal speedup.

| Seed | Original time to 18 | Packaged v1 | Paired change | New requests / continuations | New wrong checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261011 | 71.321s | 70.881s | -0.440s | 44 / 14 | 2 |
| 20261012 | 77.277s | 76.586s | -0.691s | 43 / 13 | 3 |
| 20261013 | 82.492s | 73.568s | -8.924s | 45 / 15 | 0 |
| 20261014 | 81.102s | 128.744s | +47.642s | 44 / 14 | 0 |
| 20261015 | 62.783s | 62.118s | -0.665s | 43 / 13 | 0 |

The original range was 62.783–82.492s; the new range is 62.118–128.744s. Four
paired runs were faster, and the slow trial is retained without a replacement.
Both batches scored their first trial. All five share one inference-server
lifetime within each batch. These are historical paired controls on development
data, not randomized interleaved controls or independent server restarts.

## Matched setup and audit

The new source is `f30d9dfe0f8bf9c87e3b744e4dc5f282a1606c9c`, checked out cleanly
on `callosum` at `/home/azureuser/aime-bench-refactor-v1-f30d9dfe` after the primary
checkout pulled the pushed changes. The canonical manifest is
`165d659a47e0f3e789c9a1cfd4ec9fc4184081af91cbda35d719fe41c339d688`.
The [reference batch](../frozen-core-prompt-five-seeds-20261004T005416Z/README.md)
used frozen v1 at `7c40f7af5586bc698b444c6b7934013baf99e5cd`.
Module paths and package layout changed; the archived runner remains available.

Both use A100 PCIe 80GB, NVFP4 VibeThinker-3B, Marlin, BF16 activations/KV,
FlashInfer, 95% memory allocation and 65,536-token total context. Profile bytes,
runtime package versions, dataset provenance, improved prompt, temperature 0.8,
top-p 0.95 and seeds match. Scheduling is 30×1 barrier with 8,192 initial output
tokens and up to 16,384 additional tokens per continuation, clipped to context.
The per-question four-request cap includes continuations. One owned vLLM server
is reused, with cleared prefix cache, a fresh serial three-second grader and the
cheap 30×32-token arithmetic warmup for every scored trial. Optional profiling is
disabled; required traces stay in memory until the solve phase ends.

The official clock starts after questions, services and warmup are ready, before
request scheduling. Time to 18 is the eighteenth distinct first-solved verdict
receipt. Cancellation settlement, service cleanup and the final trace flush are
separate; flushes took 1.560–2.476s and are excluded from the target measure.

The driver and local replay audit establish:

- **150/150 initial request payloads match exactly** against their same-seed originals.
- **69/69 continuations** contain their parent's complete prompt and generated
  token IDs and the correct remaining-context budget.
- Each trial retains all 30 question records, respects the request ceiling and
  has 18 distinct winners with true grader verdicts. The eighteenth first-solved
  timestamp reconciles with the summary's time to target.
- The vendored grader's three grading modules have identical ASTs to the
  original modules. Dataset and prompt hashes and all recorded policy controls match.
- The median of trial-level fresh TTFT medians is **240.556ms**, versus
  **240.289ms** originally, a 0.267ms difference.

Behavioral contracts pass, but generated text is not identical. First-pass
retained token overlaps match in only 4–10 questions per seed. Earliest differing
tokens occur after 2,973–4,790 matching tokens; cancelled traces are censored.
This is consistent with decoding sensitivity to asynchronous batching, but this
comparison does not isolate why the trajectories diverged.

## Slow trial: late candidates, not slow grading

Seed 20261014 completed 18 correct checks and **zero wrong checks**, versus one
wrong check originally. Its observed timeline is:

| Component | Original | Packaged v1 |
| --- | ---: | ---: |
| First grader pickup | 6.034s | 6.031s |
| Completed grader service | 57.002s | 54.002s |
| Grader idle between checks | 18.065s | 68.711s |

The sixteenth correct question, Q26, arrived at 54.036s; Q29 arrived at 119.445s
and Q9 was the eighteenth at 128.744s. Originally the late trio was Q7/Q12/Q9,
with Q9 arriving at 81.102s. Q9's first capped segment took **56.575s**, versus
**56.587s** originally. Its trajectories differ from token 3,463, so the later
continuation prefixes differ even though continuation construction is correct.
Q9's winning candidate was observed **69.111s into its continuation round**,
versus **21.454s** originally; verification took **3.005s** in both runs, with
less than 1ms of local candidate queue wait. The continuation retained 8,615
generated tokens versus 3,208, including generation while grading.

The immediate source of the longer elapsed time is waiting for late correct
candidates. Stable first-pass duration and TTFT provide no sign of a broad
inference slowdown. They do not prove that refactoring had no effect on batching
or decoding, or establish unchanged tail reliability. A contemporaneous
interleaved original/new control would be needed to resolve that attribution.

Coarse server logs contain 42 KV-occupancy samples, a maximum of 10.0%, and no
OOM/preemption/eviction warning lines. Complete eviction counters and peak VRAM
are unavailable because optional polling was disabled. [Cleanup evidence](server-evidence.json)
shows no remaining compute workers, no owned listeners on 8000/8077 and 0 MiB
GPU allocation after completion. Full streams and service logs remain remote.

## Reproduce

The [batch driver](../../../runner/extensions/validation/README.md) was tested
locally and remotely before launch (four focused tests passed). The command was:

```bash
~/.venvs/vllm/bin/python -m runner.extensions.validation.refactor_v1 \
  --batch canonical-v1-refactor-five-seeds-20261004T215632Z \
  --grader-python /home/azureuser/aime-bench/grader/.venv/bin/python
```

All five trials have a 600s wall safety limit, including per-trial initialization.
No timeouts, failed trials, added settling runs or replacement seeds occurred.
[Configuration](config.json), [all outcomes](summary.json),
[paired CSV](paired_timings.csv) and [analysis](analysis.json) are versioned with
the five attempts. From the repository root, replay the saved-payload, token and
verdict audits and regenerate the analysis with:

```bash
python runs/experiments/canonical-v1-refactor-five-seeds-20261004T215632Z/reproduce_analysis.py
```

New attempt metadata was annotated against each historical reference and passed
both metadata validators. The repository-wide legacy `src.attempt_metadata --all`
stops at an existing v2.1 expression-validation schema incompatibility; historical
metadata was preserved and all five new v1 attempts were validated individually.
