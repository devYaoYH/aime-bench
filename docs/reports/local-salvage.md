# Local CPU Gemma answer recovery experiment

The useful result is cheap recovery of answers already present in the reasoning.
The Gemma 3 1B verification judgments were unreliable in this pilot and highly
sensitive to prompt placement. All work stayed within this repository; no SSH or
provided grader was used. Exact-match evaluation used the answer key already in
the saved run artifacts, after model inference.

## Setup and data

Run `20260930-155212` has 240 saved Qwen trajectories. Of these, 115 ended with
`finish_reason=length`; each reported exactly 16,384 completion tokens.

The final local server used the llama.cpp maintainers'
[GGUF conversion of google/gemma-3-1b-it](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF),
file `gemma-3-1b-it-Q8_0.gguf` (1,069,306,368 bytes), revision
`f9c28bcd85737ffc5aef028638d3341d49869c27`, with llama.cpp `b11352`.
The model file SHA-256 is
`b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`.

The machine reported macOS ARM64 and 12 logical CPUs. The server used six CPU
threads, two slots, and 32,768 context tokens per slot. GPU layers, KV offloading,
and operation offloading were disabled; the endpoint is `http://127.0.0.1:8091`.
Generation was deterministic, with temperature 0, seed 2026, repetition penalty
1.1, compact JSON grammars, and limits of 160 extraction / 224 verification output
tokens. Strings were bounded to 240 characters. Prompt caching was disabled.

The initial official QAT Q4 download produced repetitive, garbled long-context
output and was excluded from the reported results. The model's
[discussion includes the same symptom and a maintainer's metadata diagnosis](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/discussions/1).
The final experiments used the original IT checkpoint's Q8 conversion. This
llama.cpp release also required its built-in Gemma template (`--no-jinja
--chat-template gemma`) to avoid a JSON-sampler initialization failure.

## A. Extract an existing final answer

The deterministic baseline scans reasoning plus response for the last existing
`Answer: NNN` line or integer `\\boxed{NNN}` expression. It found eight candidate
answers across the 115 capped trajectories, seven matching the key. The scan took
about 0.15 seconds with files already cached locally. Syntax can still refer to a
tentative or retracted answer, so these are candidates for checking.

| Question | Sample | Recovered integer | Exact match |
| --- | --- | --- | --- |
| 7 | 2 | 271 | No |
| 9 | 4 | 62 | Yes |
| 18 | 8 | 82 | Yes |
| 20 | 6 | 336 | Yes |
| 21 | 8 | 293 | Yes |
| 23 | 6 | 610 | Yes |
| 25 | 8 | 907 | Yes |
| 26 | 7 | 113 | Yes |

Question 18 was absent from the existing successful completions. Counting all
correct recovered candidates retrospectively would raise oracle pass@8 from
22/30 to 23/30 (73.3% to 76.7%). This does not establish a deployable selection
policy: the answer key is unavailable when choosing a candidate in practice.

Gemma saw only supplied work and the problem, without grades or the answer key.
The revised extraction prompt omitted the question and placed extraction
instructions after the work. Extraction required a literal quote containing the
proposed integer; fabricated quotes, empty evidence, and unrelated numbers were
rejected. Literal grounding alone does not establish semantic finality or prove
the mathematics.

| Cohort / prompt | Input | Traces | Raw non-null outputs | Raw exact matches | Quote-grounded candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| Spread sample, original prompt | Last 4,096 Gemma tokens | 8 | 3 | 0 | 0 |
| First two spread traces, original prompt | Full trace | 2 | 1 | 1 | 0 |
| All syntax-positive traces, original prompt | Last 4,096 Gemma tokens | 8 | 0 | 0 | 0 |
| First two syntax-positive traces, original prompt | Full trace | 2 | 2 | 0 | 0 |
| All syntax-positive traces, instructions last | Last 4,096 Gemma tokens | 8 | 6 | 3 | 0 |

The revised prompt recovered the correct raw integers for questions 20, 21, and
26, but its evidence did not pass the quote check. The other three non-null
outputs were wrong. Gemma added no grounded candidates beyond the syntax scan in
these selected cases. For traces that never reached an answer, extraction cannot
recover a conclusion that does not exist; inferring one becomes additional solving.

## B. Check logical consistency

Controls comprised the first five correct completed trajectories, all five
naturally wrong completed trajectories, and the same five correct trajectories
with their proposed integers increased by one modulo 1,000. The verifier saw the
problem, candidate, and up to 4,096 tail tokens, with labels withheld.

| Control group | Cases | Original prompt: supported / inconsistent | Candidate repeated last: supported / inconsistent |
| --- | ---: | ---: | ---: |
| Correct completed answers | 5 | 4 / 1 | 0 / 5 |
| Naturally wrong completed answers | 5 | 3 / 2 | 0 / 5 |
| Deliberately changed answers | 5 | 4 / 1 | 0 / 5 |

Neither protocol returned `insufficient`. The original prompt falsely supported
7/10 incorrect controls. Repeating the candidate and requiring an explicit
comparison after the work caused the verifier to reject all 15 controls,
including all five correct ones. For example, the correct answer 70 to question 1
was supported by the first prompt and rejected by the revised prompt despite the
work deriving bases 21 and 49 and their sum.

These judgments often commented on the general shape of the work or invented
errors, rather than checking the required answer. They are unsuitable as a signal
to accept an answer or stop a retry. These small, selected controls are diagnostic;
they do not estimate general verifier accuracy. Two wrong controls share question
7, and tail excerpts omit earlier proof steps.

## CPU cost and concurrency

The final experiments made 58 inference requests across three runs. All returned
parseable, non-truncated JSON. The quote check rejected every model-derived
candidate; parseable output does not establish correct reasoning.

| Experiment | Requests | First-run wall time | Median tail extraction | Median tail verification | Full-trace extraction |
| --- | ---: | ---: | ---: | ---: | --- |
| Spread pilot + controls | 25 | 240.9 s | 8.6 s | 15.9 s | 68.9 s, 80.2 s |
| Syntax-positive cohort | 10 | 132.3 s | 15.3 s | — | 34.0 s, 94.0 s |
| Instructions-last comparison + controls | 23 | 239.2 s | 12.1 s | 16.3 s | — |

These client service times exclude the runner's semaphore queue but include
server scheduling and interference between its two slots. One request's long
prefill can delay the other request's decoding. The first pilot's median prompt
processing time was approximately 7.0 seconds for tails and 40.2 seconds for full
traces; full traces were around 16.6–16.9k Gemma tokens. Timing counters under
concurrent batching include scheduling effects and are not isolated kernel speeds.

The original capped Qwen generations had a median recorded latency of 385.2
seconds, but that was a different model, remote hardware, sampling configuration,
and request concurrency. It is context for the saved workload, not a controlled
speedup measurement.

Extraction and verification still require transformer forward passes. Savings
come from producing fewer output tokens; reading the input remains significant.
Two CPU slots successfully ran concurrently. No simultaneous GPU retries or
CPU/GPU contention benchmark was performed, so this does not establish that a CPU
side task has zero impact on GPU throughput.

## Reproduce and inspect

See [README.md](../experiments.md#local-cpu-answer-salvage-with-gemma) for installation and
commands. `local_gemma.py` pins and verifies the downloaded artifacts.
`local_salvage.py` restricts inference to loopback, retains exact prompts/raw
responses, supports cached resumption, and keeps original records unchanged.
Four offline safeguard tests passed; the actual server and all three experiments
were also exercised locally.

- [Offline syntax baseline](../../runs/20260930-155212/local_salvage/regex_baseline.json)
- [Spread pilot summary](../../runs/20260930-155212/local_salvage/gemma-q8-pilot-tail4096/summary.json)
- [Syntax-positive summary](../../runs/20260930-155212/local_salvage/gemma-q8-explicit/summary.json)
- [Instructions-last summary](../../runs/20260930-155212/local_salvage/gemma-q8-instructions-last/summary.json)

Each summary's directory also contains the full request and response records.
Use cheap syntax recovery first. Keep Gemma's answers and judgments as
experimental diagnostics; acceptance needs an independent mathematical check.
