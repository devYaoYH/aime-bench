# Reasoning length and escalation analysis

Exact reasoning-token measurements, checkpoint continuation statistics, and generation-versus-verification budgets across eight samples per question.

- **Producers:** [src.experiments.baseline.analyze_escalation](../../../src/experiments/baseline/analyze_escalation.py).
- **Inputs:** All 240 original and self_consistency attempts, plus the official Qwen tokenizer.
- **Artifacts:** trajectory_metrics.json, checkpoint_summary.json, budget_summary.json, and reasoning-length PNG/SVG plots.
- **Interpretation:** Offline same-model analysis. It does not establish gains from changing models or measured live checkpoint timing.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
