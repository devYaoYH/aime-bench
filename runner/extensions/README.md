# Versioned runner policies

Canonical selection is v1. Launch another policy with `python -m runner --version
VERSION ...`, or use its module below. Each delegates to its preserved entrypoint
and verifies the original source manifest. Use `--help` to inspect version-specific
flags; canonical v1 preset files cannot be used as another core's presets.
New v1.6 uses its direct module while the canonical selector stays frozen.

| Version | Module | Difference from v1 |
| --- | --- | --- |
| v1.1 | `runner.extensions.v1_1` | Eager serial-per-question fresh long requests within 64K context; no continuation or sibling fan-out |
| v1.5 | `runner.extensions.v1_5` | Shared fixed-30 request-slot pool; retain v1 sampling/segmented budgets |
| [v1.6](v1_6/README.md) | `runner.extensions.v1_6` (direct module) | Initial coverage barrier, then question-count slot pool; cap four fresh samples, continue each from 8K to cumulative 64K/context limit |
| v2 | `runner.extensions.v2` | Gold-free grader question API, broader answer formats and relaxed mathematical prompt; v1 round scheduling |
| v2.1 | `runner.extensions.v2_1` | CPU LaTeX syntax validation and expression-key deduplication before grader submission |
| v2.2 | `runner.extensions.v2_2` | Wrong-answer feedback forks exact-prefix continuations after other unique queued answers are assessed |
| v2.3 | `runner.extensions.v2_3` | Combine feedback with a question-count slot pool and long requests clipped to 64K total context |

```bash
python -m runner.extensions.v2_1 --help
python -m runner --version v2.3 --help
```

V2.1 and later need their pinned expression-parser requirements; v1.1 and v2.3
need their distinct long-context model profiles, not the canonical 16K output
profile. Their original entrypoint defaults, prompts, budgets and manifests are
preserved. `variants/` holds the former `runner_final` tree, old batch plans,
validation/report tools, and the frozen v1 reference. Its core-version directory
aliases point to the corresponding directories here. Legacy root source helpers
remain available for frozen dependencies; canonical v1 has no dependency on them.

Put new policies under `runner/extensions/<version>/`, with a public entrypoint,
manifest, requirements where needed, presets, and focused tests. Reuse
`runner.lib.requests`, `transport`, `continuations`, `services`, `storage`,
`metrics`, and dataset helpers where their contracts fit. Keep policy decisions
in the extension rather than silently changing v1. Register the public selector
in `runner.lib.entrypoints.EXTENSIONS`. Adding a version does not promote it.
See [canonical usage and promotion](../README.md#layout-extensions-and-promotion).

## Matched v1.6 model deployments

The policy can use the committed NVFP4/Marlin, BF16/native or AWQ/Marlin
profiles without changing its manifest. Provision the chosen profile under
`~/models/<organization>/<model>/`, alongside the downloaded weights. For AWQ:

```bash
~/.venvs/vllm/bin/python -m runner.extensions.v1_6 \
  --model AABoyles/VibeThinker-3B-AWQ \
  --model-profile vllm-v1_6-awq-marlin.yaml \
  --seed 20261011 --benchmark --skip-benchmark-prewarm
```

The [AWQ profile](../../configs/vllm/AABoyles/VibeThinker-3B-AWQ/vllm-v1_6-awq-marlin.yaml)
uses 95% memory, 64K total context/output ceiling, BF16 activations/KV,
FlashInfer attention and explicit Marlin linear kernels. Check the startup log
for `Using MarlinLinearKernel for AutoAWQMarlinLinearMethod`; a requested flag
alone does not prove runtime dispatch.

`runner.extensions.validation.awq_v1_6` is the single matched-seed comparison
driver. It starts owned services in a clean pinned worktree, validates all
policy controls and reference prompt token IDs, retains failures, and compares
against the existing NVFP4 seed. It requires the reference model/assets and
versioned reference attempt; the ordinary policy command above does not.
