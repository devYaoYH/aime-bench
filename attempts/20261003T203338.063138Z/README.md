# Qwen3.5-35B-A3B GPTQ coverage attempt

Produced on callosum by `python -m src.attempt` at source commit
`70c00ef65065b74840354337b5fa77c3c5f96568`, using
`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` on the A100 PCIe 80GB.

This coverage experiment started all 30 AIME 2025 questions with one rollout
each and an 8,192-token first pass. Later rounds allowed 16,384 tokens per
request, with exact-token continuations and a four-request cap per question.
The recorded seed was `20261003`; three rounds submitted 71 generation requests.

The target was reached: **18 verified solves**, with official solving and
cancellation settlement taking **517.843 seconds (8m38s)**. The official
interval was `2026-10-03T20:37:42.003+00:00` through
`2026-10-03T20:46:19.849+00:00`; initialization is excluded from this timer.

[config.json](config.json) preserves settings, prompt, launch profile, and Git
provenance. [summary.json](summary.json) preserves outcomes and first-solved
events. Trajectories, verification logs, exact token evidence, and GPU telemetry
are versioned for the canonical viewer and analysis. Full SSE dumps and service
logs remain excluded from Git and were not needed for this local copy.
