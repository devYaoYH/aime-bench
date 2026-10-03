#!/usr/bin/env python3
"""Download, verify, and deploy the pinned model profile; never start inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

MODEL = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
REVISION = "3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b"
ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/vllm" / MODEL / "vllm.yaml"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(destination, files):
    for row in files:
        path = destination / row["name"]
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise RuntimeError(f"Missing or wrong-sized file: {path}")
        if row.get("sha256") and sha256(path) != row["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {path}")
        print(f"Verified {row['name']}", flush=True)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    verified = {row["name"] for row in files if row.get("sha256")}
    if len(shards) != 14 or not shards <= verified:
        raise RuntimeError("Expected all 14 indexed weight shards with SHA256 evidence")
    config = json.loads((destination / "config.json").read_text())
    quantization = config["quantization_config"]
    if config["model_type"] != "qwen3_5_moe" or quantization["quant_method"] != "gptq" or quantization["bits"] != 4:
        raise RuntimeError("Unexpected model architecture or quantization")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path.home() / "models")
    parser.add_argument("--verify-only", action="store_true", help="Offline checks only; no download or deployment")
    args = parser.parse_args()
    destination = args.models_dir.expanduser() / MODEL
    manifest_path = destination / "download_manifest.json"
    profile_hash = sha256(PROFILE)
    if args.verify_only:
        manifest = json.loads(manifest_path.read_text())
        if manifest["model"] != MODEL or manifest["revision"] != REVISION:
            raise RuntimeError("Download manifest does not match the pinned model")
        if manifest["profile_sha256"] != profile_hash or sha256(destination / "vllm.yaml") != profile_hash:
            raise RuntimeError("Deployed profile does not match this checkout; prepare again")
        files = manifest["files"]
    else:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise RuntimeError("Hub revision does not match the pinned commit")
        files = [{"name": f.rfilename, "size": f.size,
                  "sha256": f.lfs.sha256 if f.lfs else None} for f in info.siblings]
        snapshot_download(MODEL, revision=REVISION, local_dir=destination, max_workers=4)
    verify_files(destination, files)
    if not args.verify_only:
        (destination / "vllm.yaml").write_bytes(PROFILE.read_bytes())
        manifest = {"model": MODEL, "revision": REVISION, "files": files,
                    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "profile_sha256": profile_hash,
                    "total_download_bytes": sum(row["size"] for row in files)}
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
    print(f"Prepared and verified {MODEL} @ {REVISION}. No inference was started.")


if __name__ == "__main__":
    main()
