# Local CPU salvage experiment family

Gemma 3 1B IT Q8 extraction/verification on saved capped Qwen traces, including full-versus-tail windows and prompt-order controls.

- **Producers:** [src.experiments.local_salvage.local_gemma](../../../src/experiments/local_salvage/local_gemma.py), [src.experiments.local_salvage.local_salvage](../../../src/experiments/local_salvage/local_salvage.py).
- **Inputs:** Parent baseline/self_consistency traces and the pinned llama.cpp/GGUF model served on loopback.
- **Artifacts:** regex_baseline.json; each gemma-* child has its own configuration, summary, and provenance note.
- **Interpretation:** The regex baseline is offline; child model pilots use local CPU inference. These are not simultaneous CPU/GPU contention measurements.

Producer links show current module locations after repository organization. Saved
configuration and statistics retain the original experiment settings. Raw traces
remain local under the Git ignore policy; this note does not rerun the experiment.
