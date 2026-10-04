"""Verify the frozen core and its repository-owned runtime dependencies."""
import hashlib
import json
from pathlib import Path

MANIFEST = Path("runner_final/core_v1_1/manifest.json")


def verify_core(root):
    path = root / MANIFEST
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest["core_id"] != "runner_final_core_v1_1" or manifest["schema_version"] != 1:
        raise RuntimeError("Unsupported frozen core manifest")
    changed = []
    for relative, expected in manifest["sha256"].items():
        file = root / relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError("Frozen core drift; restore this version or create a new core version: " + ", ".join(changed))
    return hashlib.sha256(raw).hexdigest()
