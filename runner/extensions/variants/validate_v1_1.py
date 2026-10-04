"""Benchmark immutable core v1.1 for five declared seeds on one server lifetime."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess

import httpx
import yaml

from src.common import ROOT, utc_now
from src.attempt_runners import speedrun_v1_1 as runner
from runner_final.run_v1_1 import parse_args
from runner_final.integrity_v1_1 import verify_core
from runner_final.core_v1._runtime import Services, attempt_lock, ready
from runner_final.core_v1._ports import ensure_free
from runner_final.core_v1.prewarm import reset_cache

PROTOCOL = Path(__file__).with_name("five_seeds_core_v1_1.json")


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


CONTROLS = ('model', 'system_prompt_sha256', 'temperature', 'top_p', 'seed',
            'max_attempts_per_question', 'disable_thinking', 'grader_cost',
            'benchmark', 'buffer_traces', 'no_gpu_telemetry', 'no_overhead_profile',
            'skip_benchmark_prewarm', 'max_context_tokens', 'reuse_server')


def score(output, seed, reference):
    protocol = json.loads(PROTOCOL.read_text())
    config = json.loads((output/'config.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    control_dir = ROOT/'attempts'/reference
    control = json.loads((control_dir/'config.json').read_text())
    questions = [json.loads(p.read_text()) for p in sorted(output.glob('trace/*/question.json'))]
    solved = [q for q in questions if q.get('first_solved')]
    changed_controls = [k for k in CONTROLS if config.get(k) != control.get(k)]
    different, intervals, requests = [], [], []
    caps = (len(questions) == 30 and len({q['problem_idx'] for q in questions}) == 30
            and all(1 <= len(q['rollouts']) <= 4 for q in questions))
    for question in questions:
        index = question['problem_idx']
        relative = Path('trace')/f'{index:02d}'/'rollout-01/request.json'
        original = json.loads((control_dir/relative).read_text())
        last_end = None
        for rollout in question['rollouts']:
            number = rollout['rollout']
            expected = dict(original)
            expected['seed'] = seed + index * 4 + number
            expected['max_tokens'] = 65536 - config['served_prompt_tokens'][str(index)]
            path = output/'trace'/f'{index:02d}'/f'rollout-{number:02d}'/'request.json'
            if not path.exists() or json.loads(path.read_text()) != expected:
                different.append([index, number])
            start = rollout['start_monotonic_s']
            end = rollout['generation_end_monotonic_s']
            caps = (caps and rollout['continuation_of_rollout'] is None
                    and rollout['endpoint'] == '/v1/chat/completions'
                    and rollout['requested_max_tokens'] == expected['max_tokens']
                    and rollout['generated_token_ids_count'] <= expected['max_tokens']
                    and (last_end is None or start >= last_end))
            last_end = end
            intervals.extend([(start, 1), (end, -1)])
            requests.append(rollout)
    active = peak = 0
    for _, change in sorted(intervals):
        active += change
        peak = max(peak, active)
    caps = caps and peak <= 30
    warm = json.loads((output/'inference_warmup.json').read_text())
    warm_control = json.loads((control_dir/'inference_warmup.json').read_text())
    warm_valid = ([r['request'] for r in warm['requests']] == [r['request'] for r in warm_control['requests']]
                  and len(warm['requests']) == 30)
    identity = (config['runner_id'] == runner.RUNNER_ID and config['seed'] == seed
                and config['core_manifest_sha256'] == verify_core(ROOT) == protocol['core_manifest_sha256']
                and config['system_prompt_sha256'] == protocol['system_prompt_sha256']
                and config['model_profile_sha256'] == protocol['profile_sha256']
                and not config['git_dirty'] and config['reuse_server']
                and config['parallelism'] == 30 and config['schedule'] == 'eager'
                and config['rollouts'] == 1 and config['max_attempts_per_question'] == 4
                and config['max_tokens'] == config['first_pass_max_tokens'] == 65536
                and config['no_continuation'] and summary.get('benchmark_prewarm') is None
                and config['grader_health']['queries_so_far'] == 0
                and config['grader_health']['cost_c'] == 3)
    verdicts = all(q['winner'] and q['winner']['result']['verdict'] is True for q in solved)
    times = sorted(q['first_solved']['first_solved_elapsed_s'] for q in solved)
    timing_valid = len(times) >= 18 and times[17] == summary['time_to_target_s']
    valid = (summary['status'] == 'completed' and summary['target_reached']
             and len(solved) >= 18 and verdicts and timing_valid
             and identity and caps and warm_valid and not different and not changed_controls)
    return {'sampling_seed': seed, 'attempt_id': output.name, 'reference_attempt_id': reference,
            'status': summary['status'], 'target_reached': summary['target_reached'],
            'valid': valid, 'time_to_target_s': summary['time_to_target_s'] if valid else None,
            'recorded_time_to_target_s': summary['time_to_target_s'], 'distinct_solved': len(solved),
            'identity_valid': identity, 'caps_valid': caps, 'warmup_valid': warm_valid,
            'verdicts_valid': verdicts, 'timing_valid': timing_valid,
            'different_requests_beyond_declared_output_cap': different,
            'different_control_fields': changed_controls,
            'generation_requests': len(requests),
            'later_fresh_requests': sum(r['rollout'] > 1 for r in requests),
            'continuation_requests': sum(r['continuation_of_rollout'] is not None for r in requests),
            'peak_active_requests': peak,
            'observed_output_token_ids': sum(r['generated_token_ids_count'] for r in requests),
            'grader_timeline': summary['grader_timeline']}


def aggregate(rows, protocol):
    valid = [r for r in rows if r.get('valid') and r.get('time_to_target_s') is not None]
    times = [r['time_to_target_s'] for r in valid]
    paired = [{'sampling_seed': r['sampling_seed'], 'v1_1_s': r['time_to_target_s'],
               'v1_s': protocol['reference_times_s'][str(r['sampling_seed'])],
               'difference_s': r['time_to_target_s'] - protocol['reference_times_s'][str(r['sampling_seed'])]}
              for r in valid]
    return {'successful_trials': len(valid), 'declared_trials': len(protocol['seeds']),
            'all_declared_seeds_retained': [r['sampling_seed'] for r in rows] == protocol['seeds'],
            'median_time_to_target_s': statistics.median(times) if times else None,
            'min_time_to_target_s': min(times) if times else None,
            'max_time_to_target_s': max(times) if times else None,
            'paired_seed_comparisons': paired,
            'faster_than_same_seed_v1_trials': sum(r['difference_s'] < 0 for r in paired),
            'statistics_scope': 'Valid successful trials only; all failures/unmet outcomes retained. Historical seed pairing is not interleaved causal validation.'}


async def execute(options):
    protocol = json.loads(PROTOCOL.read_text())
    seeds = protocol["seeds"]
    argv = protocol["argv"] + ["--reuse-server"]
    if options.grader_python:
        argv += ["--grader-python", options.grader_python]
    args = parse_args(["--preset", str(Path(__file__).with_name("presets") / protocol["preset"])] + argv)
    core_hash = verify_core(ROOT)
    if core_hash != protocol["core_manifest_sha256"]:
        raise RuntimeError("Core v1.1 differs from predeclared protocol")
    if args.system_prompt_sha256 != protocol["system_prompt_sha256"]:
        raise RuntimeError("Prompt differs from predeclared protocol")
    batch = ROOT / "runs/experiments" / options.batch
    batch.mkdir(parents=True, exist_ok=False)
    profile = Path(args.models_dir).expanduser() / args.model / args.model_profile
    if hashlib.sha256(profile.read_bytes()).hexdigest() != protocol["profile_sha256"]:
        raise RuntimeError("Remote profile hash differs from the declared v1.1 profile")
    if yaml.safe_load(profile.read_text()) != yaml.safe_load((ROOT / "configs/vllm/r0b0tlab/VibeThinker-3B-NVFP4/vllm-v1_1-long64k.yaml").read_text()):
        raise RuntimeError("Remote launch profile differs from frozen final configuration")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Validation requires a clean tracked source checkout")
    config = {"driver": "runner_final.validate_v1_1", "started_at_utc": utc_now(),
              "protocol": protocol, "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "argv": argv, "seed_sequence": seeds, "profile_path": str(profile),
              "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
              "core_manifest_sha256": core_hash, "system_prompt_sha256": args.system_prompt_sha256,
              "system_prompt": args.system_prompt, "preset": protocol["preset"],
              "server_reused": True, "prefix_cache_reset_between_trials": True,
              "server_env_override": {"VLLM_SERVER_DEV_MODE": "1"}}
    save(batch / "config.json", config)
    services = Services()
    result = {"status": "initializing", "trials": [], "seed_sequence": seeds}
    save(batch / "summary.json", result)
    try:
        ensure_free(args.vllm_port)
        ensure_free(args.grader_port)
        runner.assert_gpu_idle(args.gpu_device)
        command = [args.vllm_binary, "serve", "--config", str(profile), "--host", "127.0.0.1",
                   "--port", str(args.vllm_port), "--enable-prompt-tokens-details"]
        config["server_command"] = command
        save(batch / "config.json", config)
        server = services.launch(command, batch / "vllm.log", env=dict(os.environ, VLLM_SERVER_DEV_MODE="1"))
        async with httpx.AsyncClient(trust_env=False) as client:
            await ready(client, args.vllm_url + "/v1/models", args.startup_timeout, server)
            result["status"] = "running"
            save(batch / "summary.json", result)
            for trial, seed in enumerate(seeds, 1):
                before = set((ROOT / "attempts").iterdir())
                row = {"trial": trial, "sampling_seed": seed, "valid": False, "time_to_target_s": None}
                try:
                    cache = await reset_cache(client, args.vllm_url)
                    print(f"V1.1 TRIAL {trial}/5 seed={seed}", flush=True)
                    output = await runner.run(parse_args(["--preset", str(Path(__file__).with_name("presets") / protocol["preset"])] + argv + ["--seed", str(seed)]))
                    row.update(score(output, seed, protocol["reference_attempts"][str(seed)]), cache_reset=cache)
                except BaseException as error:
                    row.update(status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed", error=f"{type(error).__name__}: {error}")
                    created = set((ROOT / "attempts").iterdir()) - before
                    if len(created) == 1:
                        row["attempt_id"] = created.pop().name
                    print(f"V1.1 TRIAL FAILURE {trial}: {row['error']}", flush=True)
                    if not isinstance(error, Exception):
                        result["trials"].append(row)
                        save(batch / "summary.json", result)
                        raise
                result["trials"].append(row)
                save(batch / "summary.json", result)
                print("V1.1 RESULT " + json.dumps({k: v for k, v in row.items() if k != "benchmark_prewarm"}), flush=True)
            result.update(status="complete", completed_at_utc=utc_now(),
                          **aggregate(result["trials"], protocol))
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        await services.close()
        save(batch / "summary.json", result)
        print("V1.1 BATCH " + json.dumps({k: v for k, v in result.items() if k != "trials"}), flush=True)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader-python")
    parser.add_argument("--batch", default="core-v1_1-five-seeds-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    options = parser.parse_args(argv)
    if Path(options.batch).name != options.batch or options.batch in (".", ".."):
        parser.error("Use a plain batch directory name")
    with attempt_lock():
        async def entry():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await execute(options)
        asyncio.run(entry())


if __name__ == "__main__":
    main()
