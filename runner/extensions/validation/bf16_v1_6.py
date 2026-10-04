"""One BF16/FlashInfer v1.6 attempt, matched against the scored NVFP4 seed."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

import httpx
import yaml
from runner.extensions.validation import v1_6_batch as baseline
from runner.extensions.v1_6.cli import parse_args
from runner.extensions.v1_6.integrity import verify_core
from runner.lib.common import ROOT, atomic_json, utc_now
from runner.lib.gpu import assert_gpu_idle
from runner.lib.prewarm import reset_cache
from runner.lib.services import Services, attempt_lock, ensure_free, ready

REFERENCE = "core-v1_6-barrier-five-seeds-20261004T224850Z"
MODEL = "WeiboAI/VibeThinker-3B"
PROFILE = "vllm-v1_6-bf16-flashinfer.yaml"
CONTROLS = tuple(
    k for k in baseline.CONTROLS if k not in ("model", "model_profile")
) + (
    "max_fresh_samples_per_question",
    "max_rollout_tokens",
    "max_concurrent_requests",
    "initial_coverage_barrier",
    "continuation_policy",
    "policy_defaults_sha256",
)


def reference_trial(seed):
    rows = baseline.load(ROOT / "runs/experiments" / REFERENCE / "summary.json")[
        "trials"
    ]
    return next(r for r in rows if r["sampling_seed"] == seed)


def trial_args(seed, grader_python, *, model=MODEL, profile=PROFILE):
    return parse_args(
        [
            "--model",
            model,
            "--model-profile",
            profile,
            "--seed",
            str(seed),
            "--max-concurrent-requests",
            "30",
            "--benchmark",
            "--skip-benchmark-prewarm",
            "--reuse-server",
            "--benchmark-year",
            "2025",
            "--grader-python",
            grader_python,
        ]
    )


def check_controls(config, old):
    differences = [k for k in CONTROLS if config[k] != old[k]]
    if differences:
        raise RuntimeError("Unmatched v1.6 controls: " + ", ".join(differences))


def validate_profile(profile, reference):
    current = yaml.safe_load(profile.read_text())
    old = yaml.safe_load(
        baseline.load(
            ROOT / "attempts" / reference["attempt_id"] / "model_profile.json"
        )["yaml"]
    )
    excluded = {"model", "served-model-name", "quantization", "linear-backend"}
    if {k: v for k, v in current.items() if k not in excluded} != {
        k: v for k, v in old.items() if k not in excluded
    }:
        raise RuntimeError("BF16 profile has unmatched non-quantization controls")
    if (
        current.get("quantization") is not None
        or current.get("linear-backend") != "auto"
        or current["served-model-name"] != MODEL
    ):
        raise RuntimeError(
            "Expected unquantized BF16 with automatic native linear kernels"
        )
    committed = ROOT / "configs/vllm" / MODEL / PROFILE
    if profile.read_bytes() != committed.read_bytes():
        raise RuntimeError("Remote BF16 profile differs from committed bytes")
    return current


def model_provenance(models_dir, reference):
    old = baseline.load(ROOT / "attempts" / reference["attempt_id"] / "config.json")
    evidence = {}
    architecture = (
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "bos_token_id",
        "eos_token_id",
        "tie_word_embeddings",
    )
    for name in (MODEL, old["model"]):
        folder = Path(models_dir).expanduser() / name
        config = baseline.load(folder / "config.json")
        rope = config.get("rope_parameters") or {
            "rope_theta": config.get("rope_theta"),
            "rope_type": "default",
        }
        revisions = {
            p.name: p.read_text().splitlines()[0]
            for p in (folder / ".cache/huggingface/download").glob(
                "*.safetensors.metadata"
            )
        }
        evidence[name] = dict(
            config_sha256=hashlib.sha256(
                (folder / "config.json").read_bytes()
            ).hexdigest(),
            architecture={k: config.get(k) for k in architecture},
            rope_parameters=rope,
            tokenizer_sha256={
                f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
                for f in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                )
            },
            weight_revisions=revisions,
            generation_config=baseline.load(folder / "generation_config.json"),
            quantization_config=config.get("quantization_config"),
        )
    left, right = evidence[MODEL], evidence[old["model"]]
    if (
        left["architecture"] != right["architecture"]
        or left["rope_parameters"] != right["rope_parameters"]
        or left["tokenizer_sha256"] != right["tokenizer_sha256"]
        or left["quantization_config"] is not None
    ):
        raise RuntimeError(
            "Unmatched model architecture/RoPE/tokenizer or quantized BF16 checkpoint"
        )
    evidence["comparison_note"] = (
        "Quantized model card declares WeiboAI/VibeThinker-3B as base; it does not pin the original base revision. Architecture, RoPE and tokenizer assets match. Profile overrides both generation ceilings to64K; effective inference math/backend differs."
    )
    return evidence


def audit(output, reference, core_hash, *, model=MODEL):
    config, summary = (
        baseline.load(output / "config.json"),
        baseline.load(output / "summary.json"),
    )
    old_folder = ROOT / "attempts" / reference["attempt_id"]
    old = baseline.load(old_folder / "config.json")
    check_controls(config, old)
    questions = [baseline.load(p) for p in output.glob("trace/*/question.json")]
    mismatches, prefix_errors = [], []
    for q in questions:
        folder = Path("trace") / f"{q['problem_idx']:02d}"
        new_request = baseline.load(output / folder / "rollout-01/request.json")
        old_request = baseline.load(old_folder / folder / "rollout-01/request.json")
        new_request.pop("model")
        old_request.pop("model")
        if new_request != old_request:
            mismatches.append(q["problem_idx"])
        for r in q["rollouts"]:
            if r["segment"] == 1:
                continue
            parent = r["continuation_of_rollout"]
            tokens = baseline.load(
                output / folder / f"rollout-{parent:02d}" / "tokens.json"
            )
            request = baseline.load(
                output / folder / f"rollout-{r['rollout']:02d}" / "request.json"
            )
            prefix = tokens["prompt_token_ids"] + tokens["output_token_ids"]
            if (
                not tokens["complete"]
                or request.get("prompt") != prefix
                or request["max_tokens"]
                != min(
                    config["max_rollout_tokens"] - len(tokens["output_token_ids"]),
                    config["max_context_tokens"] - len(prefix),
                )
            ):
                prefix_errors.append([q["problem_idx"], r["rollout"]])
    allocation = baseline.load(output / "allocation.json")
    coverage = [a for a in allocation["admissions"] if a["phase"] == "coverage"]
    pool = [a for a in allocation["admissions"] if a["phase"] == "pool"]
    release = allocation["barrier_release"]
    barrier_valid = (
        len(coverage) == 30
        and {a["problem_idx"] for a in coverage} == set(range(1, 31))
        and all(a["fresh_sample"] == 1 and a["segment"] == 1 for a in coverage)
        and (
            not pool
            or release is not None
            and all(a["elapsed_s"] >= release["elapsed_s"] for a in pool)
        )
    )
    if release:
        barrier_valid &= all(
            s["coverage_settled"]
            and s["coverage_settled_elapsed_s"] <= release["elapsed_s"]
            and s["initial_generation_retired_elapsed_s"] <= release["elapsed_s"]
            for s in allocation["questions"].values()
        )
    cap_valid = (
        allocation["peak_active_requests"] <= 30
        and all(a["active_requests"] <= 30 for a in allocation["admissions"])
        and all(
            s["fresh"] <= 4 and s["requests"] <= 8
            for s in allocation["questions"].values()
        )
    )
    solved = [q for q in questions if q.get("first_solved")]
    times = sorted(q["first_solved"]["first_solved_elapsed_s"] for q in solved)
    target_valid = (
        len(times) >= 18
        and abs(times[17] - summary["time_to_target_s"]) < 1e-6
        and all(q["winner"]["result"]["verdict"] is True for q in solved)
    )
    valid = (
        summary["target_reached"]
        and summary["status"] == "completed"
        and len(questions) == 30
        and not mismatches
        and not prefix_errors
        and cap_valid
        and barrier_valid
        and target_valid
        and config["core_manifest_sha256"] == core_hash
        and not config["git_dirty"]
        and config["runner_id"] == "runner_core_v1_6"
        and config["model"] == model
        and config["max_context_tokens"] == old["max_context_tokens"]
        and config["served_prompt_tokens"] == old["served_prompt_tokens"]
        and all(
            config["dataset_provenance"][k] == old["dataset_provenance"][k]
            for k in (
                "id",
                "year",
                "role",
                "revision",
                "rows",
                "prompt_sha256",
                "grader_sha256",
            )
        )
    )
    return dict(
        valid=valid,
        attempt_id=output.name,
        time_to_target_s=summary["time_to_target_s"],
        target_reached=summary["target_reached"],
        solved=len(solved),
        status=summary["status"],
        reference_attempt_id=reference["attempt_id"],
        reference_time_to_target_s=reference["time_to_target_s"],
        initial_payload_mismatches=mismatches,
        continuation_errors=prefix_errors,
        barrier_valid=barrier_valid,
        cap_valid=cap_valid,
        peak_active_requests=allocation["peak_active_requests"],
        barrier_release=release,
        generation_requests=sum(len(q["rollouts"]) for q in questions),
        fresh_samples=sum(s["fresh"] for s in allocation["questions"].values()),
        performance=summary["performance"],
        grader_timeline=summary["grader_timeline"],
        trace_storage=summary["trace_storage"],
    )


async def execute(options, *, deployment=None):
    # Deployment adapters share ownership, timing and matched-policy checks.
    if deployment is None:
        import sys

        deployment = sys.modules[__name__]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError(
            "Use a clean pinned worktree; preserve primary checkout changes"
        )
    reference = deployment.reference_trial(options.seed)
    args = deployment.trial_args(options.seed, options.grader_python)
    old = baseline.load(ROOT / "attempts" / reference["attempt_id"] / "config.json")
    check_controls(vars(args), old)
    core_hash = verify_core(ROOT)
    if core_hash != old["core_manifest_sha256"]:
        raise RuntimeError("V1.6 policy differs from NVFP4 control")
    profile = Path(args.models_dir).expanduser() / deployment.MODEL / deployment.PROFILE
    deployment.validate_profile(profile, reference)
    provenance = deployment.model_provenance(args.models_dir, reference)
    batch = ROOT / "runs/experiments" / options.batch
    batch.mkdir(parents=True, exist_ok=False)
    config = dict(
        driver=getattr(deployment, "DRIVER_ID", "runner.extensions.validation.bf16_v1_6"),
        model_provenance=provenance,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        core_manifest_sha256=core_hash,
        seed=options.seed,
        reference_batch=REFERENCE,
        reference_attempt_id=reference["attempt_id"],
        started_at_utc=utc_now(),
        profile_path=str(profile),
        profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        grader_semantic_hashes=baseline.grader_semantics(),
        trial_wall_timeout_s=options.trial_timeout,
        scope=getattr(deployment, "COMPARISON_SCOPE", "One matched-seed deployment comparison: BF16 automatic native linear kernels versus NVFP4 Marlin; FlashInfer attention, BF16 KV/activations and runner policy unchanged. Not a repeatability estimate."),
    )
    atomic_json(batch / "config.json", config)
    result = dict(
        status="initializing", valid=False, seed=options.seed, time_to_target_s=None
    )
    services = Services()
    before = set((ROOT / "attempts").iterdir())
    try:
        ensure_free(args.vllm_port)
        ensure_free(args.grader_port)
        assert_gpu_idle(args.gpu_device)
        command = [
            args.vllm_binary,
            "serve",
            "--config",
            str(profile),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.vllm_port),
            "--enable-prompt-tokens-details",
        ]
        config.update(
            server_command=command, server_env_override={"VLLM_SERVER_DEV_MODE": "1"}
        )
        atomic_json(batch / "config.json", config)
        server = services.launch(
            command, batch / "vllm.log", env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
        )
        async with httpx.AsyncClient(trust_env=False) as client:
            await ready(
                client, args.vllm_url + "/v1/models", args.startup_timeout, server
            )
            result.update(
                status="running", cache_reset=await reset_cache(client, args.vllm_url)
            )
            atomic_json(batch / "summary.json", result)
            async with asyncio.timeout(options.trial_timeout):
                output = await baseline.run(args)
            result.update(
                deployment.audit(output, reference, core_hash), completed_at_utc=utc_now()
            )
    except BaseException as error:
        result.update(
            status="interrupted"
            if isinstance(error, asyncio.CancelledError)
            else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        created = set((ROOT / "attempts").iterdir()) - before
        if len(created) == 1:
            result["attempt_id"] = created.pop().name
        raise
    finally:
        await services.close()
        atomic_json(batch / "summary.json", result)
        print(deployment.MODEL + " RESULT " + json.dumps(result), flush=True)


async def entry(options):
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM, asyncio.current_task().cancel
    )
    await execute(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261011, choices=baseline.SEEDS)
    parser.add_argument(
        "--batch",
        default="core-v1_6-bf16-flashinfer-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--grader-python", default=str(ROOT / "runner/grader/.venv/bin/python")
    )
    parser.add_argument("--trial-timeout", type=float, default=600)
    options = parser.parse_args()
    if (
        Path(options.batch).name != options.batch
        or options.batch in (".", "..")
        or options.trial_timeout <= 0
    ):
        parser.error("Use a plain batch name and positive timeout")
    with attempt_lock():
        asyncio.run(entry(options))


if __name__ == "__main__":
    main()
