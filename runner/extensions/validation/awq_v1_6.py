"""One AWQ/Marlin v1.6 attempt with matched NVFP4 policy and prompt IDs."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import signal
import sys

import yaml
from runner.extensions.validation import bf16_v1_6 as harness
from runner.lib.common import ROOT
from runner.lib.services import attempt_lock

DRIVER_ID = "runner.extensions.validation.awq_v1_6"
REFERENCE = harness.REFERENCE
MODEL = "AABoyles/VibeThinker-3B-AWQ"
PROFILE = "vllm-v1_6-awq-marlin.yaml"
COMPARISON_SCOPE = (
    "One matched-seed AWQ int4/group128 Marlin versus NVFP4 Marlin deployment. "
    "FlashInfer attention, BF16 activations/KV, 95% memory and v1.6 policy unchanged. "
    "Original tokenizer files retained; exact prompt token IDs checked before launch. "
    "Quantizer calibration/base weight revision not controlled. Not a repeatability estimate."
)
reference_trial = harness.reference_trial
check_controls = harness.check_controls


def trial_args(seed, grader_python):
    return harness.trial_args(seed, grader_python, model=MODEL, profile=PROFILE)


def audit(output, reference, core_hash):
    return harness.audit(output, reference, core_hash, model=MODEL)


def validate_profile(profile, reference):
    current = yaml.safe_load(profile.read_text())
    old = yaml.safe_load(harness.baseline.load(
        ROOT / "attempts" / reference["attempt_id"] / "model_profile.json"
    )["yaml"])
    excluded = {"model", "served-model-name", "quantization", "linear-backend"}
    if {k: v for k, v in current.items() if k not in excluded} != {
        k: v for k, v in old.items() if k not in excluded
    }:
        raise RuntimeError("AWQ profile has unmatched inference controls")
    if (current.get("quantization") != "awq"
            or current.get("linear-backend") != "marlin"
            or current.get("served-model-name") != MODEL):
        raise RuntimeError("Expected AWQ quantization with Marlin linear kernels")
    if profile.read_bytes() != (ROOT / "configs/vllm" / MODEL / PROFILE).read_bytes():
        raise RuntimeError("Remote AWQ profile differs from committed bytes")
    return current


def model_provenance(models_dir, reference):
    # File hashes differ after reserialization by newer Transformers. Compare
    # actual vocabulary, rendering and exact prompt IDs instead of overwriting assets.
    from transformers import AutoTokenizer

    root = Path(models_dir).expanduser()
    old_folder = ROOT / "attempts" / reference["attempt_id"]
    old_model = harness.baseline.load(old_folder / "config.json")["model"]
    evidence, tokenizers = {}, {}
    keys = (
        "architectures", "model_type", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "max_position_embeddings", "bos_token_id", "eos_token_id",
        "tie_word_embeddings",
    )
    for model in (MODEL, old_model):
        folder = root / model
        config = harness.baseline.load(folder / "config.json")
        evidence[model] = {
            "architecture": {k: config.get(k) for k in keys},
            "rope_parameters": config.get("rope_parameters") or {
                "rope_theta": config.get("rope_theta"), "rope_type": "default"
            },
            "asset_sha256": {
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                if (folder / name).exists() else None
                for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                             "special_tokens_map.json", "chat_template.jinja")
            },
            "weight_revisions": {
                p.name: p.read_text().splitlines()[0]
                for p in (folder / ".cache/huggingface/download").glob("*.safetensors.metadata")
            },
            "generation_config": harness.baseline.load(folder / "generation_config.json"),
            "quantization_config": config.get("quantization_config"),
        }
        tokenizers[model] = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    left, right = evidence[MODEL], evidence[old_model]
    quant = left["quantization_config"] or {}
    if (left["architecture"] != right["architecture"]
            or left["rope_parameters"] != right["rope_parameters"]
            or quant.get("quant_method") != "awq" or quant.get("bits") != 4
            or quant.get("group_size") != 128 or quant.get("zero_point") is not True):
        raise RuntimeError("Unmatched architecture/RoPE or unexpected AWQ checkpoint")
    awq, nv = tokenizers[MODEL], tokenizers[old_model]
    if awq.get_vocab() != nv.get_vocab() or awq.all_special_ids != nv.all_special_ids:
        raise RuntimeError("Unmatched tokenizer vocabulary/special token IDs")
    checked = []
    for request_path in sorted(old_folder.glob("trace/*/rollout-01/request.json")):
        body = harness.baseline.load(request_path)
        saved = harness.baseline.load(request_path.with_name("tokens.json"))
        messages = body["messages"]
        ids = awq.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        expected = saved["prompt_token_ids"]
        if ids != expected or ids != nv.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        ):
            raise RuntimeError("AWQ rendered prompt IDs differ: " + str(request_path))
        # Check decoding too, because extraction consumes decoded stream text.
        generated = saved["output_token_ids"]
        if awq.decode(generated) != nv.decode(generated):
            raise RuntimeError("AWQ decoded reference output differs: " + str(request_path))
        checked.append({"question_index": int(request_path.parent.parent.name),
                        "prompt_tokens": len(ids),
                        "prompt_ids_sha256": hashlib.sha256(str(ids).encode()).hexdigest()})
    if len(checked) != 30:
        raise RuntimeError("Expected 30 matched reference prompts")
    evidence["tokenizer_semantic_check"] = {
        "vocabulary_and_special_ids_match": True,
        "exact_reference_prompt_ids_match": True,
        "reference_output_decoding_matches": True,
        "questions": checked,
    }
    evidence["comparison_note"] = COMPARISON_SCOPE
    return evidence


async def entry(options):
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    await harness.execute(options, deployment=sys.modules[__name__])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261011, choices=harness.baseline.SEEDS)
    parser.add_argument("--batch", default="core-v1_6-awq-marlin-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--grader-python", default=str(ROOT / "runner/grader/.venv/bin/python"))
    parser.add_argument("--trial-timeout", type=float, default=600)
    options = parser.parse_args()
    if Path(options.batch).name != options.batch or options.batch in (".", "..") or options.trial_timeout <= 0:
        parser.error("Use a plain batch name and positive timeout")
    with attempt_lock():
        asyncio.run(entry(options))


if __name__ == "__main__":
    main()
