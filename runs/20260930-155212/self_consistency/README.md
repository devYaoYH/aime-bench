# Eight-sample self-consistency expansion

Seven new independent samples per question, combined with the original response as sample 1, to measure pass@8 and answer voting.

- **Producers:** [src.experiments.baseline.expand_samples](../../../src/experiments/baseline/expand_samples.py).
- **Inputs:** The parent baseline questions/ records, including their exact original request payloads.
- **Artifacts:** config.json and summary.json; the 210 additional full responses stay local in questions/.
- **Interpretation:** Observed pass@8 is 22/30. Saved UTC timing includes a laptop sleep interval; this is independent resampling, not continuation.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
