"""Extend the declared core v1.5 series with exactly the remaining four seeds."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess

import yaml

from src.common import ROOT, utc_now
from src.attempt_runners import speedrun_v1_5 as runner
from runner_final.run_v1_5 import parse_args
from runner_final.integrity_v1_5 import verify_core
from runner_final.core_v1._runtime import attempt_lock
from runner_final.validate_frozen import save

PROTOCOL = Path(__file__).with_name('five_seeds_core_v1_5.json')
CONTROLS = ('model', 'model_profile', 'model_profile_sha256', 'system_prompt_sha256',
            'temperature', 'top_p', 'seed', 'first_pass_max_tokens', 'max_tokens',
            'max_attempts_per_question', 'disable_thinking', 'grader_cost',
            'benchmark', 'buffer_traces', 'no_gpu_telemetry', 'no_overhead_profile',
            'skip_benchmark_prewarm', 'max_context_tokens')


def score(output, seed, reference):
    config = json.loads((output/'config.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    control_dir = ROOT/'attempts'/reference
    control = json.loads((control_dir/'config.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted(output.glob('trace/*/question.json'))]
    solved = [r for r in rows if r.get('first_solved')]
    allocation = json.loads((output/'allocation.json').read_text())
    different = []
    for index in range(1, 31):
        path = Path('trace')/f'{index:02d}'/'rollout-01/request.json'
        if not (output/path).exists() or json.loads((output/path).read_text()) != json.loads((control_dir/path).read_text()):
            different.append(index)
    changed_controls = [key for key in CONTROLS if config.get(key) != control.get(key)]
    identity = (config['runner_id'] == runner.RUNNER_ID and config['seed'] == seed
                and config['core_manifest_sha256'] == verify_core(ROOT)
                and not config['git_dirty'] and not config['reuse_server']
                and summary.get('benchmark_prewarm') is None
                and config['inference_warmup']['batch_size'] == 30
                and config['inference_warmup']['tokens_per_request'] == 32)
    caps = (len(rows) == 30 and len({r['problem_idx'] for r in rows}) == 30
            and all(len(r['rollouts']) <= 4 for r in rows)
            and allocation['max_concurrent_requests'] == 30
            and allocation['peak_active_requests'] <= 30
            and all(s['used'] <= 4 for s in allocation['questions'].values()))
    initial = allocation['admissions'][:30]
    coverage = len(initial) == 30 and len({r['problem_idx'] for r in initial}) == 30 and all(r['rollout'] == 1 for r in initial)
    verdicts = all(r['winner'] and r['winner']['result']['verdict'] is True for r in solved)
    valid = (summary['status'] == 'completed' and summary['target_reached']
             and len({r['problem_idx'] for r in solved}) >= 18 and verdicts
             and identity and caps and coverage and not different and not changed_controls)
    rollouts = [r for row in rows for r in row['rollouts']]
    return {'sampling_seed': seed, 'attempt_id': output.name, 'reference_attempt_id': reference,
            'status': summary['status'], 'target_reached': summary['target_reached'],
            'valid': valid, 'time_to_target_s': summary['time_to_target_s'] if valid else None,
            'recorded_time_to_target_s': summary['time_to_target_s'],
            'distinct_solved': len(solved), 'identity_valid': identity, 'caps_valid': caps,
            'initial_coverage_valid': coverage, 'different_initial_questions': different,
            'different_control_fields': changed_controls,
            'generation_requests': len(rollouts),
            'later_fresh_requests': sum(r['rollout'] > 1 and r['continuation_of_rollout'] is None for r in rollouts),
            'continuation_requests': sum(r['continuation_of_rollout'] is not None for r in rollouts),
            'peak_active_requests': allocation['peak_active_requests'],
            'observed_output_token_ids': sum(r['generated_token_ids_count'] for r in rollouts),
            'grader_timeline': summary['grader_timeline']}


def aggregate(rows, protocol):
    valid = [r for r in rows if r.get('valid') and r.get('time_to_target_s') is not None]
    times = [r['time_to_target_s'] for r in valid]
    paired = [{'sampling_seed': r['sampling_seed'], 'v1_5_s': r['time_to_target_s'],
               'v1_s': protocol['reference_times_s'][str(r['sampling_seed'])],
               'difference_s': r['time_to_target_s'] - protocol['reference_times_s'][str(r['sampling_seed'])]}
              for r in valid]
    return {'successful_trials': len(valid), 'declared_trials': len(protocol['seeds']),
            'all_declared_seeds_retained': sorted(r['sampling_seed'] for r in rows) == sorted(protocol['seeds']),
            'median_time_to_target_s': statistics.median(times) if times else None,
            'min_time_to_target_s': min(times) if times else None,
            'max_time_to_target_s': max(times) if times else None,
            'paired_seed_comparisons': paired,
            'faster_than_same_seed_v1_trials': sum(r['difference_s'] < 0 for r in paired),
            'statistics_scope': 'Valid successful trials only; all failures/unmet outcomes retained. Historical seed pairing is not interleaved causal validation.'}


async def execute(options):
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['new_seeds'] != protocol['seeds'][1:]:
        raise RuntimeError('Extension must retain the first seed and run only the remaining four')
    core_hash = verify_core(ROOT)
    if core_hash != protocol['core_manifest_sha256']:
        raise RuntimeError('Core v1.5 differs from the declared version')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('A clean tracked source checkout is required')
    batch = ROOT/'runs/experiments'/options.batch
    batch.mkdir(parents=True, exist_ok=False)
    argv = ['--preset', str(Path(__file__).with_name('presets')/protocol['preset'])]
    if options.grader_python:
        argv += ['--grader-python', options.grader_python]
    args = parse_args(argv)
    profile = Path(args.models_dir).expanduser()/args.model/args.model_profile
    if yaml.safe_load(profile.read_text()) != yaml.safe_load(Path(__file__).with_name('vllm-flashinfer.yaml').read_text()):
        raise RuntimeError('Remote profile differs from the declared standard BF16 KV profile')
    config = {'driver':'runner_final.validate_v1_5','started_at_utc':utc_now(),
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'protocol':protocol,'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'core_manifest_sha256':core_hash,'profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),
              'argv':argv,'fresh_owned_server_and_grader_per_trial':True,
              'completed_seed_not_reexecuted':protocol['seeds'][0]}
    save(batch/'config.json', config)
    prior = score(ROOT/'attempts'/protocol['existing_attempt'], protocol['seeds'][0],
                  protocol['reference_attempts'][str(protocol['seeds'][0])])
    if not prior['valid']:
        raise RuntimeError('Previously completed v1.5 trial failed audit; do not replace it')
    prior.update(trial=1, retained_existing_trial=True)
    result = {'status':'running','seed_sequence':protocol['seeds'],
              'new_seed_sequence':protocol['new_seeds'],'trials':[prior]}
    save(batch/'summary.json', result)
    try:
        for trial, seed in enumerate(protocol['new_seeds'], 2):
            before = set((ROOT/'attempts').iterdir())
            row = {'trial':trial,'sampling_seed':seed,'valid':False,'time_to_target_s':None,
                   'retained_existing_trial':False}
            print(f'V1.5 TRIAL {trial}/5 seed={seed}', flush=True)
            try:
                output = await runner.run(parse_args(argv+['--seed',str(seed)]))
                row.update(score(output, seed, protocol['reference_attempts'][str(seed)]))
            except BaseException as error:
                row.update(status='interrupted' if isinstance(error, asyncio.CancelledError) else 'failed',
                           error=f'{type(error).__name__}: {error}')
                created = set((ROOT/'attempts').iterdir()) - before
                if len(created) == 1:
                    row['attempt_id'] = created.pop().name
                if not isinstance(error, Exception):
                    result['trials'].append(row)
                    raise
            result['trials'].append(row)
            save(batch/'summary.json', result)
            print('V1.5 RESULT '+json.dumps(row), flush=True)
        result.update(status='complete', completed_at_utc=utc_now(), **aggregate(result['trials'],protocol))
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error, asyncio.CancelledError) else 'failed',
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save(batch/'summary.json', result)
        print('V1.5 BATCH '+json.dumps({k:v for k,v in result.items() if k != 'trials'}), flush=True)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',default='core-v1_5-five-seeds-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--grader-python')
    options = parser.parse_args(argv)
    if Path(options.batch).name != options.batch or options.batch in ('.','..'):
        parser.error('Use a plain batch directory name')
    with attempt_lock():
        async def entry():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await execute(options)
        asyncio.run(entry())


if __name__ == '__main__':
    main()
