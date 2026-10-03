"""Install and serve the pinned Gemma 3 1B Q8 CPU model for the local-salvage pilot.

Use setup to download/check llama.cpp and the pinned GGUF on Apple Silicon
macOS; use serve to expose CPU-only inference on loopback port 8091 by default.
Binaries, model provenance, checksums, and downloads stay in ignored .local/.
Serving uses configurable CPU threads and parallel slots with 32k context each.
    python -m src.experiments.local_salvage.local_gemma setup
    python -m src.experiments.local_salvage.local_gemma serve
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tarfile

import httpx
from huggingface_hub import get_token

from src.common import ROOT
LOCAL = ROOT / '.local'
TAG = 'b11352'
REPO = 'ggml-org/gemma-3-1b-it-GGUF'
REVISION = 'f9c28bcd85737ffc5aef028638d3341d49869c27'
MODEL_NAME = 'gemma-3-1b-it-Q8_0.gguf'
MODEL_SHA256 = 'b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5'
ARCHIVE_NAME = f'llama-{TAG}-bin-macos-arm64.tar.gz'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(client, url, target, token=None):
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + '.part')
    with client.stream('GET', url, headers={'Authorization': 'Bearer ' + token} if token else {}) as response:
        response.raise_for_status()
        with temporary.open('wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    temporary.replace(target)


def setup():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('This pinned binary setup targets Apple Silicon macOS.')
    for subdir in ['downloads', 'bin', 'models']:
        (LOCAL / subdir).mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=180) as client:
        metadata = client.get(f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{TAG}')
        metadata.raise_for_status()
        asset = next(a for a in metadata.json()['assets'] if a['name'] == ARCHIVE_NAME)
        archive = LOCAL / 'downloads' / ARCHIVE_NAME
        download(client, asset['browser_download_url'], archive)
        expected = asset.get('digest')
        if expected and expected != 'sha256:' + digest(archive):
            raise RuntimeError('llama.cpp archive checksum mismatch')
        if not (LOCAL / 'bin' / f'llama-{TAG}' / 'llama-server').exists():
            with tarfile.open(archive) as tar:
                tar.extractall(LOCAL / 'bin', filter='data')
        model = LOCAL / 'models' / MODEL_NAME
        download(client, f'https://huggingface.co/{REPO}/resolve/{REVISION}/{MODEL_NAME}', model, get_token())
        if digest(model) != MODEL_SHA256:
            raise RuntimeError('Gemma file checksum differs from the pinned experiment artifact')
    (LOCAL / 'llama_source.json').write_text(json.dumps({'tag': TAG, 'asset': asset}, indent=2))
    (LOCAL / 'model_source.json').write_text(json.dumps({'repo': REPO, 'base_model': 'google/gemma-3-1b-it',
                                                       'revision': REVISION, 'file': MODEL_NAME}, indent=2))
    (LOCAL / 'checksums.json').write_text(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in [archive, model]}, indent=2))
    print('Installed original Gemma IT Q8 conversion and llama.cpp under', LOCAL)


def serve(args):
    binary = LOCAL / 'bin' / f'llama-{TAG}' / 'llama-server'
    model = LOCAL / 'models' / MODEL_NAME
    if not binary.exists() or not model.exists():
        raise RuntimeError('Run python3 -m src.experiments.local_salvage.local_gemma setup first')
    command = [str(binary), '-m', str(model), '--alias', 'gemma-3-1b-it',
               '--host', '127.0.0.1', '--port', str(args.port),
               '--device', 'none', '--n-gpu-layers', '0', '--no-kv-offload', '--no-op-offload',
               '--fit', 'off', '--ctx-size', str(32768 * args.parallel), '--parallel', str(args.parallel),
               '--threads', str(args.threads), '--threads-batch', str(args.threads),
               '--no-jinja', '--chat-template', 'gemma']
    # This release's Jinja grammar path fails for Gemma JSON schemas; built-in Gemma works.
    print('CPU-only server; context per slot = 32768; endpoint =', f'http://127.0.0.1:{args.port}', flush=True)
    os.execv(binary, command)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['setup', 'serve'])
    p.add_argument('--port', type=int, default=8091)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--threads', type=int, default=6)
    args = p.parse_args()
    if args.parallel < 1 or args.threads < 1 or not 1 <= args.port <= 65535:
        p.error('parallel/threads must be positive and port between 1 and 65535')
    setup() if args.action == 'setup' else serve(args)
