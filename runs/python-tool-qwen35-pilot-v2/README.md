# Corrected Python paired pilot

The Q23/Q25 paired pilot was repeated after allowing ordinary underscore variables and strengthening the first-tool-call policy.

- **Producers:** [src.experiments.python_tools.python_tool_pilot](../../src/experiments/python_tools/python_tool_pilot.py).
- **Inputs:** Original Q23/Q25 records, corrected macOS worker, and saved v2 request settings.
- **Artifacts:** config.json, worker_preflight.json, and summary.json; full round/tool records stay local.
- **Interpretation:** Selected-question four-call pilot. Q25 was correctly solved in both arms; Q23 remained capped without tool use. Saved config/request records define the historical settings.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
