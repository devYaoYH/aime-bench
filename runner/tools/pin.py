"""Record a reviewed canonical source snapshot, independently of historical cores."""

import argparse
import hashlib
import json
from runner.lib.common import PACKAGE, ROOT, atomic_json


def build(reason):
    files = [
        p
        for p in PACKAGE.rglob("*.py")
        if "extensions" not in p.relative_to(PACKAGE).parts
        and not {"__pycache__", ".venv", "venv", ".cache"}.intersection(p.parts)
    ]
    files += [
        PACKAGE / "metadata.schema.json",
        PACKAGE / "requirements.txt",
        PACKAGE / "grader/requirements.txt",
        PACKAGE / "grader/requirements-local.txt",
    ]
    return {
        "schema_version": 1,
        "core_id": "runner_core_v1",
        "derived_from": {
            "core_id": "runner_final_core_v1",
            "manifest_sha256": "35d6a06315a0e45441b3ac49bd468a2b94553fb172f378bfebc7ea8cc044070f",
        },
        "policy": "V1 bounded round scheduling, integer candidate extraction, exact-ID continuations and four-request ceiling. Canonical packaging includes a custom integer-dataset adapter.",
        "change_reason": reason,
        "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reason", required=True, help="Describe the reviewed change and validation"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write only the new canonical manifest; never historical manifests",
    )
    args = parser.parse_args()
    if not args.reason.strip():
        parser.error("A nonempty change reason is required")
    value = build(args.reason)
    if args.write:
        atomic_json(PACKAGE / "manifest.json", value)
    else:
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
