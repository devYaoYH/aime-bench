#!/usr/bin/env python3
"""Download and verify pinned NVFP4 weights and deploy profiles; no inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess

MODEL = "r0b0tlab/VibeThinker-3B-NVFP4"
REVISION = "2fc0013974d1a466e6a5a11839f029d5aff34dc9"
ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("vllm.yaml", "vllm-baseline-16k.yaml")


def digest(path, algorithm="sha256", git_blob=False):
    result = hashlib.new(algorithm)
    if git_blob:
        result.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_files(destination, files):
    for row in files:
        path = destination / row["name"]
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise RuntimeError(f"Missing or wrong-sized file: {path}")
        if row.get("sha256"):
            actual = digest(path)
            expected = row["sha256"]
        else:
            actual = digest(path, "sha1", git_blob=True)
            expected = row["blob_id"]
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {path}")
        print(f"Verified {row['name']}", flush=True)
    weights = [row for row in files if row["name"].endswith(".safetensors")]
    if len(weights) != 1 or weights[0]["name"] != "model.safetensors" or not weights[0].get("sha256"):
        raise RuntimeError("Expected the single SHA256-verified weight file")
    config = json.loads((destination / "config.json").read_text())
    quant = config["quantization_config"]
    exported = json.loads((destination / "hf_quant_config.json").read_text())["quantization"]
    if (config["architectures"] != ["Qwen2ForCausalLM"] or config["model_type"] != "qwen2"
            or quant["quant_method"] != "modelopt" or quant["quant_algo"] != "NVFP4"
            or exported["quant_algo"] != "NVFP4" or exported["group_size"] != 16):
        raise RuntimeError("Unexpected architecture or quantization metadata")
    with (destination / "model.safetensors").open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_size <= 16 * 1024 * 1024:
            raise RuntimeError("Invalid safetensors header length")
        header = json.loads(stream.read(header_size))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not any(value["dtype"] == "U8" for value in tensors.values()) or not any("weight_scale" in key for key in tensors):
        raise RuntimeError("Expected packed FP4 weights and quantization scales")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path.home() / "models")
    parser.add_argument("--verify-only", action="store_true", help="Offline integrity checks; no writes")
    args = parser.parse_args()
    destination = args.models_dir.expanduser() / MODEL
    manifest_path = destination / "download_manifest.json"
    profile_dir = ROOT / "configs/vllm" / MODEL
    profile_hashes = {name: digest(profile_dir / name) for name in PROFILES}
    if args.verify_only:
        manifest = json.loads(manifest_path.read_text())
        if manifest["model"] != MODEL or manifest["revision"] != REVISION:
            raise RuntimeError("Manifest does not match the pinned model")
        if manifest["profile_sha256"] != profile_hashes:
            raise RuntimeError("Manifest profiles do not match this checkout")
        for name, checksum in profile_hashes.items():
            if digest(destination / name) != checksum:
                raise RuntimeError(f"Deployed profile differs: {name}")
        files = manifest["files"]
    else:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise RuntimeError("Hub revision does not match the pinned commit")
        files = [{"name": f.rfilename, "size": f.size, "blob_id": f.blob_id,
                  "sha256": f.lfs.sha256 if f.lfs else None} for f in info.siblings]
        snapshot_download(MODEL, revision=REVISION, local_dir=destination, max_workers=4)
    verify_files(destination, files)
    if not args.verify_only:
        for name in PROFILES:
            (destination / name).write_bytes((profile_dir / name).read_bytes())
        manifest = {"model": MODEL, "revision": REVISION, "files": files,
                    "profile_sha256": profile_hashes,
                    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "total_download_bytes": sum(row["size"] for row in files)}
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
    print(f"Prepared and verified {MODEL} @ {REVISION}. No inference was started.")


if __name__ == "__main__":
    main()
