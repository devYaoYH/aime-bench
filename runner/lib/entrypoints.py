"""Explicit policy registry: adding an extension never changes the canonical default."""

from importlib import import_module

CANONICAL = "v1"
EXTENSIONS = {
    "v1.1": "runner_final.run_v1_1",
    "v1.5": "runner_final.run_v1_5",
    "v2": "runner_final.run_frozen_v2",
    "v2.1": "runner_final.run_frozen_v2_1",
    "v2.2": "runner_final.run_frozen_v2_2",
    "v2.3": "runner_final.run_frozen_v2_3",
}


def launch_extension(version, argv=None):
    """Preserve archived entrypoint provenance and its pinned runtime imports."""
    return import_module(EXTENSIONS[version]).main(argv)
