"""One frozen core v1 attempt; externally gated, whole-graph Nsight samples.

Run with administrator counter access on the dedicated A100. The CUDA profiler
limit in this installed vLLM stops after its counter exceeds one, so each window
can include two engine steps. Replay perturbs timing: this is never a scored trial.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import httpx
import yaml

from runner_final.integrity import verify_core
from runner_final.run_frozen import parse_args
from runner_final.validate_frozen import save
from src.common import ROOT, utc_now


def counters(text):
    """Sum per-model Prometheus series, ignoring comments and histogram buckets."""
    wanted = ('vllm:generation_tokens_total', 'vllm:prompt_tokens_total',
              'vllm:num_requests_running', 'vllm:num_requests_waiting',
              'vllm:kv_cache_usage_perc')
    result = dict.fromkeys(wanted, 0.0)
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if name in result:
            result[name] += float(line.rsplit(' ', 1)[1])
    return result


def phase_due(phase, current, start, early):
    generated = current['vllm:generation_tokens_total'] - start['vllm:generation_tokens_total']
    if phase == 'early_decode':
        return generated >= 1500
    if phase == 'long_context_decode':
        return generated >= 100000
    if phase == 'after_continuation_admission':
        return (early is not None and current['vllm:prompt_tokens_total']
                > early['vllm:prompt_tokens_total'] + 1024)
    raise ValueError(phase)


async def execute(options):
    if os.geteuid() != 0:
        raise RuntimeError('This host restricts hardware counters; run this driver with sudo')
    core = verify_core(ROOT)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=ROOT, text=True).strip():
        raise RuntimeError('Use a clean tracked source checkout')
    output = ROOT/'runs/profiling'/options.batch
    output.mkdir(parents=True, exist_ok=False)
    argv = ['--preset', str(ROOT/'runner_final/presets/prompt_adherence.json'),
            '--seed', str(options.seed), '--models-dir', '/home/azureuser/models',
            '--vllm-python', '/home/azureuser/.venvs/vllm/bin/python',
            '--vllm-binary', str(ROOT/'scripts/profile_vllm_ncu.py'),
            '--grader-python', '/home/azureuser/aime-bench/grader/.venv/bin/python']
    args = parse_args(argv)
    profile = Path(args.models_dir)/args.model/args.model_profile
    if yaml.safe_load(profile.read_text()) != yaml.safe_load((ROOT/'runner_final/vllm-flashinfer.yaml').read_text()):
        raise RuntimeError('Serving configuration differs from final core v1')
    ncu_options = ['--target-processes', 'all', '--profile-from-start', 'no',
                   '--replay-mode', 'kernel', '--graph-profiling', 'graph',
                   '--cache-control', 'none', '--clock-control', 'none',
                   '--section', 'SpeedOfLight',
                   '--section', 'SpeedOfLight_HierarchicalTensorRooflineChart',
                   '--metrics', 'lts__t_bytes.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active',
                   '--export', str(output/'decode'), '--force-overwrite']
    config = {'driver': 'src.profiling.core_v1_ncu', 'started_at_utc': utc_now(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'core_manifest_sha256': core, 'seed': options.seed, 'frozen_argv': argv,
              'ncu': str(options.ncu.resolve()), 'ncu_options': ncu_options,
              'ncu_version': subprocess.check_output([str(options.ncu), '--version'], text=True).strip(),
              'vllm_binary': '/home/azureuser/.venvs/vllm/bin/vllm',
              'serving_intervention': {'profiler': 'cuda', 'max_iterations': 1},
              'ranked': False, 'counter_access': 'root workload; driver restriction unchanged',
              'sampling_scope': 'Externally gated decode windows; two worker steps can be captured per window. No application replay.'}
    save(output/'config.json', config)
    subprocess.run(['nvidia-smi', '-q'], stdout=(output/'device.log').open('w'), check=True)
    env = dict(os.environ, CALLOSUM_NCU_CONFIG=str(output/'config.json'), VLLM_SERVER_DEV_MODE='1')
    before = set((ROOT/'attempts').iterdir())
    logfile = (output/'runner.log').open('w')
    process = subprocess.Popen([args.vllm_python, '-m', 'runner_final.run_frozen', *argv],
                               cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT)
    result = {'status': 'running', 'ranked': False, 'windows': [], 'attempt_id': None}
    save(output/'summary.json', result)
    phases = ['early_decode', 'long_context_decode', 'after_continuation_admission']
    baseline = early = None
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=180) as client:
            with (output/'serving_samples.jsonl').open('w') as samples:
                while process.poll() is None:
                    created = set((ROOT/'attempts').iterdir()) - before
                    if len(created) > 1:
                        raise RuntimeError('More than one attempt was created')
                    if created:
                        attempt = next(iter(created))
                        result['attempt_id'] = attempt.name
                        path = attempt/'config.json'
                        if path.exists() and json.loads(path.read_text()).get('official_started_at_utc'):
                            try:
                                response = await client.get(args.vllm_url+'/metrics')
                                response.raise_for_status()
                            except httpx.HTTPError:
                                await asyncio.sleep(.25)
                                continue
                            observed = counters(response.text)
                            row = {'timestamp_utc': utc_now(), 'monotonic_s': time.perf_counter(), **observed}
                            samples.write(json.dumps(row)+'\n'); samples.flush()
                            if baseline is None:
                                # Warmup produces exactly 30*32 generation tokens;
                                # the first poll can already include solving tokens.
                                baseline = dict(observed, **{'vllm:generation_tokens_total': 960.0})
                            if phases and phase_due(phases[0], observed, baseline, early):
                                window = {'phase': phases.pop(0), 'requested_at_utc': utc_now(),
                                          'before': row, 'profiler_limit': 1}
                                if early is None:
                                    early = dict(observed)
                                print('NCU WINDOW '+json.dumps(window), flush=True)
                                response = await client.post(args.vllm_url+'/start_profile')
                                response.raise_for_status()
                                window['start_ack_at_utc'] = utc_now()
                                await asyncio.sleep(1)
                                response = await client.post(args.vllm_url+'/stop_profile')
                                response.raise_for_status()
                                window['stop_ack_at_utc'] = utc_now()
                                result['windows'].append(window)
                                save(output/'summary.json', result)
                    await asyncio.sleep(.25)
        returncode = process.wait()
        result.update(status='complete' if returncode == 0 else 'failed', returncode=returncode,
                      finished_at_utc=utc_now(), missing_phases=phases)
        if result['attempt_id']:
            result['attempt_summary'] = json.loads((ROOT/'attempts'/result['attempt_id']/'summary.json').read_text())
        report = output/'decode.ncu-rep'
        if report.exists():
            for page in ('raw', 'details'):
                with (output/f'ncu_{page}.csv').open('w') as destination:
                    subprocess.run([str(options.ncu), '--import', str(report), '--page', page,
                                    '--csv', '--print-units', 'base'], stdout=destination, check=True)
        else:
            result['profile_error'] = 'No Nsight report was exported'
        if returncode:
            raise RuntimeError(f'Frozen attempt failed with exit {returncode}')
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), 45)
            except asyncio.TimeoutError:
                process.kill(); process.wait()
        raise
    finally:
        logfile.close()
        save(output/'summary.json', result)
        # Return only this job's artifacts to the normal checkout owner.
        owner = ROOT.stat()
        folders = [output]
        if result['attempt_id']:
            folders.append(ROOT/'attempts'/result['attempt_id'])
        for folder in folders:
            subprocess.run(['chown', '-R', f'{owner.st_uid}:{owner.st_gid}', str(folder)], check=True)
    print('NCU RESULT '+json.dumps({k: v for k, v in result.items() if k != 'attempt_summary'}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ncu', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--batch', default='core-v1-ncu-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    options = parser.parse_args()
    if Path(options.batch).name != options.batch or options.batch in ('.', '..'):
        parser.error('Use a plain batch directory name')
    asyncio.run(execute(options))


if __name__ == '__main__':
    main()
