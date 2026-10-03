# Full optional-Python AIME pass@2 profile

Two independent Qwen3.5-35B-A3B/Parasail attempts per question, with thinking enabled, optional tool calls, eight active trajectories, and two CPU workers.

- **Producers:** [src.experiments.python_tools.python_tool_profile](../../src/experiments/python_tools/python_tool_profile.py), [src.experiments.python_tools.report_python_tool_profile](../../src/experiments/python_tools/report_python_tool_profile.py).
- **Inputs:** All 30 original problem records from 20260930-155212, plus the restricted macOS worker; cumulative 16k output and 16-round/16-call guards.
- **Artifacts:** config.json, worker_preflight.json, summary.json, analysis.json, report.md, and plots; 60 multi-round raw records stay local.
- **Interpretation:** Measured final-answer profile: 19/30 pass@2. Every question received both attempts; this has no matched no-tool arm. Saved worker hashes identify historical code, so edited source requires a fresh output label.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
