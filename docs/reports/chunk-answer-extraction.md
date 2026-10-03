# Hosted chunk extraction and a continuing solver

The prototype finds candidate answers from completed clauses while a solver
continues generating. The user's intended policy is to verify those candidates
in a sidecar and exit early only on a verified answer. Extraction alone does
not trigger cancellation. Local llama inference remains stopped.

All experiments used OpenRouter small models and saved streams inside this
repository. No new Qwen 30B solver requests, SSH, or provided grader were used.
The exact stored integer keys were consulted only after model responses.

## Extraction pilot

The source is the eight SSE streams from `runs/first-answer-20261002-paired`:
two prompt arms for Q1, Q3, Q18, and Q25. Jobs use only text already received
at their timestamp, a rolling 4,096-character window, and the original problem.
The host extracts literal integer proposals and safely evaluates requested
operations such as `N mod 1000`. The classifier selects a grounded source span;
it cannot invent a new answer. Scope classification remains fallible.

Six controls contain real saved passages: two formatting examples, two toy
problems, one correct proposal, and one mathematically wrong genuine proposal
(600, later corrected to 610). The last must pass extraction, because extraction
asks what was proposed, not whether it was correct.

| Setup | False acceptance among four negatives | Recognized two genuine proposals |
| --- | ---: | ---: |
| Gemma 3 4B, initial span-selection prompt | 4/4 | 2/2 |
| Gemma 3 4B, focused scope prompt | 1/4 | 0/2 |
| Qwen 2.5 7B, focused scope prompt | 2/4 | 2/2 |
| Formatting guards + focused Qwen 2.5 7B | 0/4 | 2/2 |

The guards rejected the two formatting cases; Qwen rejected the two toy cases.
These are development controls reused while adjusting the prompt, not a held-out
precision estimate. Small model speed did not imply reliable semantic decisions.
For the wider-net verification policy, such scope errors chiefly increase the
number of verification requests. A verifier's false approval remains an accuracy
risk, even if extraction is perfectly grounded.

The final hybrid used a timed replay with two shared classifier slots:

| Stream | Candidate | Candidate ready, including extraction | Natural endpoint |
| --- | ---: | ---: | ---: |
| Q1 control | 70 | 36.11 s | 70.36 s |
| Q1 first-answer prompt | 70 | 16.46 s | 65.76 s |
| Q3 control | 16 | 36.57 s | 137.70 s |
| Q3 first-answer prompt | 16 | 47.15 s | 161.12 s |
| Q18 control | none | — | 205.98 s, capped |
| Q18 first-answer prompt | none | — | 206.09 s, capped |
| Q25 control | 907 | 46.77 s | 206.50 s, capped |
| Q25 first-answer prompt | 907 | 77.85 s | 178.20 s |

All six extracted candidates match the stored keys. Ten hosted calls, including
controls, had median service latency **0.412 s**, range **0.262–0.892 s**, and
reported total cost **$0.0009447**. This measures hosted inference, not CPU
inference or CPU/GPU resource contention.

If every correct candidate could be verified in an additional three seconds,
the sum of the eight stream durations would fall from 1,231.71 s to 690.98 s,
a **43.9% hypothetical reduction**. This is summed per-request duration, not
measured batch wall time or token/billing savings. Q18 would still determine
the batch's roughly 206-second completion time. Actual solver abort behavior
was not exercised. A model verdict is not a correctness certificate.

## Continuing-generation sidecar

`verify_sidecar.py` implements a nonblocking producer and bounded worker queue.
It accepts broad literal answer events, including tentative or formatting uses,
and requested-quantity equations. Repeated integers are deduplicated. A snapshot
of the causal draft is taken when the candidate arrives. No future solver text
or answer key enters a verification request.

`feed()` never awaits hosted inference. Rejected, insufficient, malformed, or
failed checks leave generation running. An optional `on_verified` callback can
signal an early exit; it is omitted in the shadow pilot. `close(drain=False)`
cancels pending checks when the original stream finishes. Queue overflow drops
work rather than blocking the original stream, and dropped integers may be
reconsidered at a later occurrence. The first prototype checks each admitted
integer once; retrying an insufficient verdict after substantial new evidence
would be a separate policy.

The cheap gate scans only completed lines, never treats an SSE chunk boundary
as the end of an integer, and keeps reasoning/content channels separate. It is
not a universal semantic detector: paraphrased answers without a recognized
clause or quantity equation can be missed. At high sample concurrency, verifier
queueing must be included in the time-to-verdict calculation.

Run the extraction replay or verification shadow probes:

```bash
.venv/bin/python -m src.experiments.streaming.chunk_answer_extractor --focused --scope-guard --timed \
  --model qwen/qwen-2.5-7b-instruct --out runs/chunk-extractor-NEW-LABEL
.venv/bin/python -m src.experiments.streaming.verify_sidecar --out runs/verify-sidecar-NEW-LABEL
.venv/bin/python -m unittest test.test_chunk_answer_extractor test.test_verify_sidecar
```

The verifier probe uses six first candidates, six altered candidates, and one
naturally wrong intermediate answer. It supplies the full received causal
prefix, since a short local excerpt can omit necessary mathematical evidence.
Consequently verification has a larger input and output budget than extraction;
the user's three-second estimate is a sensitivity assumption to measure.

## Verification results

The 13 hosted Qwen 2.5 7B probes finished with median service latency **1.163 s**,
maximum **1.852 s**, and reported cost **$0.0061683**. These are separate latency
probes, not a timed combined solver/verifier replay; shared queueing under a
large sampling batch remains unmeasured. Prefix inputs ranged from 1,604 to
11,505 provider-reported tokens. No response exhausted its 768-token output cap.

| Candidate class | Approved | Rejected | Insufficient |
| --- | ---: | ---: | ---: |
| Correct first candidates | 5/6 | 1/6 | 0/6 |
| Altered wrong candidates | 4/6 | 2/6 | 0/6 |
| Naturally wrong intermediate 600 | 1/1 | 0/1 | 0/1 |

The verifier falsely approved **five of seven wrong candidates**. Three outputs
said `verified` while their explanations explicitly said the candidate was
incorrect. Another approved 908 with contradictory arithmetic. The natural
600 claim was also approved, repeating the unfinished draft's mistaken count;
the stored correct answer is 610. Even one correct 907 approval claimed the sum
was 3007 and that a remainder of 007 “rounds to 907,” so answer agreement did
not imply a sound verification explanation.

This supports the latency feasibility of a continuing-generation sidecar and
does **not** establish a reliable model-only early-exit policy. A stronger
verifier, checked computational evidence, or a better calibrated approval
protocol is needed before using this model's approval as a stop signal. Returning
an independently computed answer and comparing it to the candidate would remove
some verdict-binding errors, but cannot by itself repair incorrect mathematics.
The runtime is therefore left in shadow mode. The 43.9% duration reduction above
is an idealized opportunity estimate, not a result achieved by this verifier.

Raw requests, responses, source spans, costs, and latency records:

- `runs/chunk-extractor-gemma4b-pilot/`
- `runs/chunk-scope-controls-focused/`
- `runs/chunk-extractor-qwen7b-hybrid/`
- `runs/verify-sidecar-qwen7b/`

Model calls are reusable at the same output path. Use a new label for a fresh
latency measurement: cached responses do not reproduce real queueing or wall time.
