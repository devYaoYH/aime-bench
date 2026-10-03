# Initial optional-Python paired pilot

Qwen3.5-35B-A3B through Parasail, Python-enabled versus no-tool arms on Q23 and Q25, with cumulative 16k output and four-round/three-call guards.

- **Producers:** [src.experiments.python_tools.python_tool_pilot](../../src/experiments/python_tools/python_tool_pilot.py).
- **Inputs:** Original Q23/Q25 problem records from 20260930-155212 and the then-current restricted macOS worker.
- **Artifacts:** config.json, worker_preflight.json, and summary.json; multi-round arm records stay local.
- **Interpretation:** Historical optional-tool policy. The worker rejected ordinary underscore loop variables, contributing to failure. The current runner includes the subsequent correction/stronger first-call policy, so it does not reproduce this initial policy verbatim.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
