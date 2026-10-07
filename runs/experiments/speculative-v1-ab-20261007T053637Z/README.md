# Speculative v1 preflight failure

The baseline reached target18 in 76.123s. The B cell failed before starting vLLM: the harness interpreted a recently closed socket in TIME_WAIT as an occupied listener. The server and GPU had shut down cleanly. No speculative performance result exists for this batch.

Commit `188dcae3` fixes the harness to use the canonical SO_REUSEADDR port check and the actual grader port, 8077. The entire three-pair plan was then restarted as [speculative-v1-ab-20261007T054017Z](../speculative-v1-ab-20261007T054017Z/README.md). This baseline is excluded from that comparison; no seed or serving profile was substituted.

Saved config, cell summaries and baseline attempt `20261007T053756.499642Z` retain the original failed batch evidence. Full streams and logs are ignored by Git.
