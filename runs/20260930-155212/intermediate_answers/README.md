# Intermediate-answer extraction and voting replay

All 240 trajectories were analyzed for first/intermediate answer proposals, scope exclusions, parallel voting, and token-budget stopping. Conservative marker detection was audited separately.

- **Producers:** [src.experiments.intermediate_answers.analyze_intermediate](../../../src/experiments/intermediate_answers/analyze_intermediate.py), [src.experiments.intermediate_answers.render_intermediate](../../../src/experiments/intermediate_answers/render_intermediate.py), [src.experiments.streaming.audit_stream_markers](../../../src/experiments/streaming/audit_stream_markers.py).
- **Inputs:** Original/self_consistency raw records, data/intermediate_exclusions.json, and the cached Qwen tokenizer.
- **Artifacts:** summary.json, trace_map.csv, claim_inventory.csv, guarded_marker_audit.json, and plots. Derived traces.json stays local. The renderer also writes docs/reports/intermediate-answers.md.
- **Interpretation:** Offline retrospective analysis. Manual scope annotations are not an online semantic detector, and token-work savings are not measured wall time.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
