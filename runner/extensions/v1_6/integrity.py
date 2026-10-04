"""Pin the new policy and verify its unchanged canonical primitive dependency."""

import hashlib
import json
from pathlib import Path
from runner.integrity import verify_core as verify_base


def verify_core(root):
    raw = (root / "runner/extensions/v1_6/manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["core_id"] != "runner_core_v1_6" or manifest["schema_version"] != 1:
        raise RuntimeError("Unsupported v1.6 manifest")
    if verify_base(root) != manifest["canonical_manifest_sha256"]:
        raise RuntimeError("Canonical dependency manifest changed")
    changed = [
        p
        for p, digest in manifest["sha256"].items()
        if not (root / p).is_file()
        or hashlib.sha256((root / p).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise RuntimeError("V1.6 source drift: " + ", ".join(changed))
    return hashlib.sha256(raw).hexdigest()
