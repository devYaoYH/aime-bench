# Retrospective Jev false-negative check

Prefix decisions on 125 completed trajectories were compared with observed correct/wrong outcomes and continuation budgets.

- **Producers:** [src.experiments.jev.calibrate_jev](../../../src/experiments/jev/calibrate_jev.py).
- **Inputs:** Completed baseline and self_consistency records, cut to 1,500 reasoning tokens before Jev sees them.
- **Artifacts:** config.json and threshold/result summary.json; individual judging records stay local.
- **Interpretation:** Conditional calibration on completed traces; capped trajectories remain censored rather than proven negatives.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
