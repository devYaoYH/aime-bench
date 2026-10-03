# Hosted candidate-verification shadow probes

Qwen 2.5 7B verification of six first candidates, six altered wrong candidates, and the naturally wrong intermediate 600.

- **Producers:** [src.experiments.streaming.verify_sidecar](../../src/experiments/streaming/verify_sidecar.py).
- **Inputs:** Causal prefixes from first-answer-20261002-paired, plus the original Qwen wrong-proposal control.
- **Artifacts:** config.json and summary.json; trace-bearing manifests and per-probe requests/responses stay local.
- **Interpretation:** 13 hosted probes with 768-token cap and two verifier slots. No live solver was canceled; false approvals prevent treating this verifier as a correctness certificate.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
