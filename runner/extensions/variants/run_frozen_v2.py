"""Configuration-only entrypoint for immutable runner core v2."""
import argparse
import asyncio
import hashlib
from importlib import metadata
import json
from pathlib import Path
import signal
import sys

from runner_final.core_v2 import runner
from runner_final.core_v2._runtime import attempt_lock
from runner_final.integrity_v2 import verify_core
from src.common import ROOT

DEFAULT = Path(__file__).with_name("presets") / "math_core_v2.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--preset", type=Path, default=DEFAULT)
    parser.add_argument("--system-prompt-file", type=Path)
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    path = options.preset.expanduser().resolve()
    raw = path.read_bytes()
    preset = json.loads(raw)
    if set(preset) != {"schema_version", "core_id", "prompt_file", "argv"}:
        raise ValueError("Preset must contain schema_version, core_id, prompt_file and argv")
    if preset["schema_version"] != 1 or preset["core_id"] != runner.RUNNER_ID:
        raise ValueError("Preset selects an unsupported core")
    if not isinstance(preset["argv"], list) or not all(isinstance(x, str) for x in preset["argv"]):
        raise ValueError("Preset argv must be a list of argument strings")
    prompt_path = (options.system_prompt_file.expanduser().resolve() if options.system_prompt_file
                   else (path.parent / preset["prompt_file"]).resolve())
    prompt_bytes = prompt_path.read_bytes()
    prompt = prompt_bytes.decode("utf-8")
    if not prompt.strip():
        raise ValueError("System prompt cannot be empty")
    args = runner.parse_args(preset["argv"] + remaining)
    args.system_prompt = prompt
    args.system_prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    args.system_prompt_file = str(prompt_path)
    args.preset_file = str(path)
    args.preset_sha256 = hashlib.sha256(raw).hexdigest()
    args.runtime_package_versions = {}
    for package in ("vllm", "torch", "httpx", "PyYAML", "nvidia-ml-py", "jsonschema"):
        try:
            args.runtime_package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            args.runtime_package_versions[package] = None
    return args


def main(argv=None):
    verify_core(ROOT)
    if "--help" in (sys.argv[1:] if argv is None else argv):
        print("Frozen configuration options: --preset FILE --system-prompt-file FILE\n"
              "Remaining options override preset defaults. No experiment starts with --help.")
    args = parse_args(argv)
    with attempt_lock():
        async def entry():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await runner.run(args)
        asyncio.run(entry())


if __name__ == "__main__":
    main()
