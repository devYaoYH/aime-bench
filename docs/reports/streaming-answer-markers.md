# Streaming answer markers and false positives

Explicit markers are usable for application-enforced interruption, including
markers inside the returned **reasoning** stream. The first final-content marker
often comes much later. A bare substring search can also match formatting
examples, hypothetical answers, or quoted instructions, so the marker is a
candidate signal rather than a verified answer.

This analysis replays all eight raw SSE streams from the paired four-question
pilot offline. It makes no new inference calls and does not actually cancel a
request. The earlier pilot results are in [FIRST_ANSWER_PILOT.md](first-answer-pilot.md).

## Observed marker arrival times

The guarded detector requires a complete standalone `Answer: NNN` or integer
`\boxed{NNN}` line below a standalone `Final Answer` heading. It rejects a heading
immediately introduced as an example, format/instruction discussion, or code
quotation. It accepts Markdown headings, bold headings, and standalone math
wrappers. This is a deliberately conservative syntax/context rule, not a semantic
scope classifier.

| Question | Arm | Guarded candidate | Marker received | Natural request end | Time after marker |
|---|---|---:|---:|---:|---:|
| 1 | Original | 70 | 61.36s | 70.36s | 9.00s |
| 1 | Modified | 70 | 49.55s | 65.76s | 16.22s |
| 3 | Original | 16 | 86.59s | 137.70s | 51.11s |
| 3 | Modified | 16 | 143.26s | 161.12s | 17.86s |
| 18 | Original | — | — | 205.98s, capped | — |
| 18 | Modified | — | — | 206.09s, capped | — |
| 25 | Original | — | — | 206.50s, capped | — |
| 25 | Modified | 907 | 165.14s | 178.20s | 13.07s |

All five detected blocks are proposals for the requested problem and match the
stored key. Guarded interruption would reduce the summed request-end time by
**9.7% in the original arm and 7.7% in the modified arm**. Requests without a
marker are charged through their observed endpoint. The longest request remains
about 206 seconds in each arm; waiting for every request still exposes that tail.
These are retrospective arrival-time estimates, without measured abort overhead
or post-abort billing.

In the original 240-trace corpus the guarded detector still finds 133 candidate
blocks, of which 127 match the key. Those six wrong answers are mathematical
errors, rather than a reason to treat key agreement as evidence of safe syntax.
Full contexts are retained in the audit artifact. The guard can delay a useful
inline box until a later final-answer block and may miss usable candidates in
other formats.

## False positives found in the saved traces

Two examples show why exact-key correctness cannot validate a marker's meaning:

- Q3, saved attempt 2: “for example, if the answer is 16, then Answer: 016.” This
  is a formatting example. A broad `Answer:` regex would falsely treat it as an
  emitted answer even though 16 happens to be the key.
- Q3, saved attempt 8: “the exact format is `\boxed{016}` or `\boxed{16}`.” These
  boxes discuss notation, rather than emit a standalone answer. The guarded
  detector waits for a subsequent final-answer block.

The new modified Q1 stream also contains “Therefore, Answer: 070? Wait ...” at
28.52 seconds. This is an actual tentative proposal for this question, not a toy
answer, but it remains reasoning prose. The conservative detector waits until
49.55 seconds. The modified Q3 stream similarly contains an inline `Answer: 016?`
at 131.51 seconds and a guarded block at 143.26 seconds. The acceptance policy
must specify whether tentative inline proposals are eligible for cancellation.

A further limit is **absence of tags**. The original Q25 control explicitly
proposes the correct 907 at 47.87 seconds in “the answer is 907”, then checks it
until the cap at 206.50 seconds, without either explicit marker. Marker-only
interruption cannot recover that opportunity. Broader prose extraction has more
coverage and more semantic ambiguity.

## Chunk-safe interruption

`FinalAnswerDetector` in [stream_answer_markers.py](../../src/stream_answer_markers.py) is
ready to feed from incoming deltas. It keeps independent state for reasoning and
content, so a heading in one channel cannot authorize a marker in the other.
It requires a received newline to delimit the whole marker line; genuine stream
EOF can delimit the final line. **Never use the end of a network chunk as the
end of a number:** `Answer: 0`, `7`, `0` can arrive in three chunks. The detector
waits and yields 70, rather than stopping at 0 or 7.

An integration can return from the streaming context or close the response as
soon as the detector returns a candidate. [OpenRouter's streaming documentation](https://openrouter.ai/docs/api_reference/streaming#stream-cancellation)
lists DeepInfra as supporting connection-abort cancellation. This pilot did not
exercise that cancellation path. Early abort can also prevent the final usage
frame from arriving, so missing token/cost accounting must be recorded rather
than mistaken for zero cost.

For a more reliable protocol, request one reserved output such as
`<candidate_answer>070</candidate_answer>` or a tool call expressly defined to
contain the requested quantity; avoid asking the model to quote numeric examples
of that protocol in its reasoning. The application should validate the complete
marker and cancel when it arrives. This improves the distinction between prose
and an answer emission, while the resulting answer still needs sampling or
verification to assess correctness.

## Evidence and reproduction

- [Raw-SSE guarded replay](../../runs/first-answer-20261002-paired/guarded_marker_replay.json)
- [All 240 guarded first-marker contexts](../../runs/20260930-155212/intermediate_answers/guarded_marker_audit.json)
- [Detector and false-positive/chunk-boundary tests](../../test/test_stream_answer_markers.py)
- [Offline replay implementation](../../src/experiments/streaming/audit_stream_markers.py)

```bash
.venv/bin/python -m unittest test.test_stream_answer_markers
.venv/bin/python -m src.experiments.streaming.audit_stream_markers
```

No local inference server was restarted. The original records remain unchanged.
