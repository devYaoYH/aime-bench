# VibeThinker BF16 · 95% VRAM

Produced by `python -m src.attempt` on callosum at source commit
`7f2d726bc757fc10a25f6212b8a975e9acaa41a0` for the 95% VRAM + overhead profiling experiment.

Status: **completed**; 18 verified correct questions. Official solving and settlement: 97.441s.

Coverage policy, model, seed, and sampling match the earlier BF16 run. The VRAM budget and instrumentation changed together; this measured repeat was slower.

[metadata.json](metadata.json) records the intervention and reference, model,
quantization, GPU VRAM envelope, runner source/version, and control variables.
[config.json](config.json) and [summary.json](summary.json) preserve original
settings and outcomes. Saved trajectories, verification, GPU, and profiling
evidence are versioned for the viewer and analysis; full SSE/audits/logs stay excluded.
