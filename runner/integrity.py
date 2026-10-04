"""Pin canonical package source independently of archived measurement manifests."""

import hashlib
import json
from pathlib import Path

MANIFEST = Path("runner/manifest.json")


def verify_core(root):
    raw = (root / MANIFEST).read_bytes()
    manifest = json.loads(raw)
    if (
        manifest.get("core_id") != "runner_core_v1"
        or manifest.get("schema_version") != 1
    ):
        raise RuntimeError("Unsupported canonical runner manifest")
    changed = [
        name
        for name, digest in manifest["sha256"].items()
        if not (root / name).is_file()
        or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise RuntimeError("Canonical runner source drift: " + ", ".join(changed))
    return hashlib.sha256(raw).hexdigest()
