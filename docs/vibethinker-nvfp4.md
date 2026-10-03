# VibeThinker-3B NVFP4 on Callosum

Model: [r0b0tlab/VibeThinker-3B-NVFP4](https://huggingface.co/r0b0tlab/VibeThinker-3B-NVFP4),
pinned to `2fc0013974d1a466e6a5a11839f029d5aff34dc9`.
The single weight file is 2,183,619,280 bytes, with SHA256
`a683f30a51771dd0223c6073057063a0ccff2de821e40cd7ca8943b2cd41c0ea`.

Callosum has an A100 PCIe 80GB (SM80) and vLLM
`0.30.1rc1.dev623+g4ac0d0eac`, with PyTorch `2.13.0+cu132`.
Its installed `ModelOptNvFp4Config` accepts SM75+, and its
`MarlinNvFp4LinearKernel` supports this GPU. Profiles explicitly select
`modelopt_fp4` and `linear-backend: marlin`: packed FP4 weights with BF16
activations (W4A16). The A100 does not execute native Blackwell W4A4 FP4.
The publisher's GB10 throughput gains therefore do not establish A100 speedups.
See [vLLM ModelOpt support](https://docs.vllm.ai/en/stable/features/quantization/modelopt/).

The default `vllm.yaml` matches the original VibeThinker profile's 65,536-token
context and 95% VRAM budget. `vllm-baseline-16k.yaml` uses 16,384 total context
tokens for the existing naive baseline. Both allow up to 16,384 generated tokens,
enable prefix caching and prompt-token details, and bind to `127.0.0.1:8000`.
KV cache uses the model's BF16 dtype (`auto`). Attention and graph settings
use vLLM defaults. These profiles preserve raw reasoning text for benchmarking;
add `--reasoning-parser deepseek_r1` for separated chat reasoning fields.
No remote custom model code is required for `Qwen2ForCausalLM`.

## Prepare and verify

After pushing locally and pulling the same commit on Callosum:

```sh
cd ~/aime-bench
~/.venvs/vllm/bin/python scripts/prepare_vibethinker_nvfp4.py
~/.venvs/vllm/bin/python scripts/prepare_vibethinker_nvfp4.py --verify-only
```

Preparation resumes the pinned download into
`~/models/r0b0tlab/VibeThinker-3B-NVFP4`, checks every file's size and Hub checksum
(SHA256 for LFS, Git blob SHA1 otherwise), checks quantization metadata and
the packed safetensors header, deploys both tracked profiles, and writes
`download_manifest.json` with the source Git commit and profile hashes.
The offline verification also checks both deployed profiles against this checkout.
Neither command starts inference or modifies the existing vLLM environment.

## Launch when the GPU is free

Inspect `nvidia-smi`, processes, and port 8000 first; leave unrelated jobs alone.

```sh
tmux new-session -s vibethinker-nvfp4 \
  '~/.venvs/vllm/bin/vllm serve --config ~/models/r0b0tlab/VibeThinker-3B-NVFP4/vllm.yaml'
```

For the shorter context, substitute `vllm-baseline-16k.yaml`. To use the managed
coverage runner instead of launching a separate server:

```sh
cd ~/aime-bench
~/.venvs/vllm/bin/python -m src.attempt --model r0b0tlab/VibeThinker-3B-NVFP4
```

Runtime model loading, generation correctness, and A100 throughput remain
unverified until a free-GPU smoke test. Preparation does not launch a benchmark.
