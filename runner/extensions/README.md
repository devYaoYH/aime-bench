# Versioned runner policies

Canonical selection is v1. Launch another policy with `python -m runner --version
VERSION ...`, or use its module below. Each delegates to its preserved entrypoint
and verifies the original source manifest. Use `--help` to inspect version-specific
flags; canonical v1 preset files cannot be used as another core's presets.

| Version | Module | Difference from v1 |
| --- | --- | --- |
| v1.1 | `runner.extensions.v1_1` | Eager serial-per-question fresh long requests within 64K context; no continuation or sibling fan-out |
| v1.5 | `runner.extensions.v1_5` | Shared fixed-30 request-slot pool; retain v1 sampling/segmented budgets |
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
