# VibeThinker NVFP4, 30×1, 16K first pass

Reached 18 verified correct in **106.570 s**. Twenty completed checks included
two wrong answers. All 30 generation requests were first-pass samples; there
were no continuations. The GPU was released after successful completion.

The NVFP4/Marlin profile uses 95% VRAM, BF16 activations/KV and a 65,536-token
total context. This attempt increased the first-pass generation budget to 16,384
tokens, retained a four-request cap per question and used the same seed/sampling,
prospective answer prompt and barrier schedule as the BF16 30×1 control.

See [the experiment and CPU review](../../runs/experiments/nvfp4-30x1-16k-20261003T212811Z/README.md)
for timing decomposition, limitations, and optimization candidates. Config,
summary, metadata, saved requests/responses, exact token records and telemetry
are retained here. Full SSE streams and service audits remain on the remote.
