"""Audit retained v1.5 token evidence and compare the declared five-seed series."""
import argparse
import csv
import statistics
import json
from pathlib import Path

from src.common import ROOT, atomic_json
from runner_final.validate_v1_5 import PROTOCOL, aggregate, score


def audit_attempt(folder, seed, reference):
    row = score(folder, seed, reference)
    config = json.loads((folder/'config.json').read_text())
    allocation = json.loads((folder/'allocation.json').read_text())
    questions = [json.loads(p.read_text()) for p in sorted(folder.glob('trace/*/question.json'))]
    events, winners = [], []
    for question in questions:
        index = question['problem_idx']
        for rollout in question['rollouts']:
            path = folder/'trace'/f'{index:02d}'/f"rollout-{rollout['rollout']:02d}"
            request = json.loads((path/'request.json').read_text())
            assert request['seed'] == seed + index*4 + rollout['rollout'], 'Request seed drift'
            assert request['temperature'] == config['temperature'] and request['top_p'] == config['top_p'], 'Sampling drift'
            parent = rollout['continuation_of_rollout']
            if parent is None:
                assert request['messages'][0]['content'] == config['system_prompt'], 'Fresh prompt drift'
                assert request['max_tokens'] == (8192 if rollout['rollout'] == 1 else 16384), 'Fresh budget drift'
            else:
                tokens = json.loads((folder/'trace'/f'{index:02d}'/f'rollout-{parent:02d}'/'tokens.json').read_text())
                prefix = tokens['prompt_token_ids'] + tokens['output_token_ids']
                assert tokens['complete'] and request['prompt'] == prefix, 'Continuation prefix drift'
                assert request['max_tokens'] == min(16384, config['max_context_tokens']-len(prefix)), 'Additional-token budget drift'
            events += [(rollout['start_monotonic_s'],1),(rollout['generation_end_monotonic_s'],-1)]
            if question['winner'] and question['winner']['rollout'] == rollout['rollout']:
                winners.append({'problem_idx':index,'rollout':rollout['rollout'],
                                'kind':'initial' if rollout['rollout']==1 else 'fresh_retry' if parent is None else 'continuation'})
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    assert active == 0 and peak <= 30, 'Concurrent-request cap drift'
    target = row['recorded_time_to_target_s']
    if target is not None:
        assert all(e['elapsed_s'] <= target for e in allocation['admissions']), 'Admission after target'
    row.update(peak_client_generation_requests=peak,
               max_requests_per_question=max(s['used'] for s in allocation['questions'].values()),
               max_concurrent_per_question=max(s['peak_active'] for s in allocation['questions'].values()),
               winners=winners, artifact_audit_passed=True,
               source_commit=config['git_commit'], core_manifest_sha256=config['core_manifest_sha256'])
    control = json.loads((ROOT/'attempts'/reference/'summary.json').read_text())
    control_questions = [json.loads(p.read_text()) for p in sorted((ROOT/'attempts'/reference).glob('trace/*/question.json'))]
    row['control'] = {'time_to_target_s':control['time_to_target_s'],
                      'generation_requests':control['performance']['generation_requests'],
                      'grader_timeline':control['grader_timeline'],
                      'observed_output_token_ids':sum(r['generated_token_ids_count'] for q in control_questions for r in q['rollouts'])}
    solved = sorted(q['first_solved']['first_solved_elapsed_s'] for q in questions if q.get('first_solved'))
    old_solved = sorted(q['first_solved']['first_solved_elapsed_s'] for q in control_questions if q.get('first_solved'))
    row['milestones'] = [{'solved':n,'v1_5_s':solved[n-1] if len(solved)>=n else None,'core_v1_s':old_solved[n-1]}
                         for n in (1,2,4,6,8,10,12,14,16,18)]
    return row


def analyze(batch):
    summary = json.loads((batch/'summary.json').read_text())
    protocol = json.loads(PROTOCOL.read_text())
    rows = []
    for retained in summary['trials']:
        seed = retained['sampling_seed']
        try:
            row = audit_attempt(ROOT/'attempts'/retained['attempt_id'],seed,protocol['reference_attempts'][str(seed)])
            assert row['valid'] == retained['valid'] and row['time_to_target_s'] == retained['time_to_target_s'], 'Driver/result drift'
        except Exception as error:
            row = {**retained,'valid':False,'time_to_target_s':None,
                   'recorded_time_to_target_s':retained.get('recorded_time_to_target_s',retained.get('time_to_target_s')),
                   'artifact_audit_passed':False,'artifact_audit_error':f'{type(error).__name__}: {error}'}
        rows.append(row)
    result = {'batch':batch.name,'trials':rows,**aggregate(rows,protocol),
              'all_available_artifact_audits_passed':all(r['artifact_audit_passed'] for r in rows),
              'server_scope':'Five fresh owned v1.5 server/grader lifetimes; historical v1 batch used one server lifetime. Initialization excluded from scoring.',
              'validation_scope':'Exact initial payload/control matching, additional-token budgets and parent IDs, seed formula, generation intervals <=30, four-request cap, stop admissions at target.'}
    audited = [r for r in rows if r['artifact_audit_passed']]
    new_requests = sum(r['generation_requests'] for r in audited)
    old_requests = sum(r['control']['generation_requests'] for r in audited)
    new_tokens = sum(r['observed_output_token_ids'] for r in audited)
    old_tokens = sum(r['control']['observed_output_token_ids'] for r in audited)
    result['resource_comparison'] = {
        'audited_trials':len(audited), 'v1_5_generation_requests':new_requests,
        'core_v1_generation_requests':old_requests,
        'request_increase_percent':100*(new_requests/old_requests-1) if old_requests else None,
        'v1_5_observed_output_token_ids':new_tokens,'core_v1_observed_output_token_ids':old_tokens,
        'observed_output_increase_percent':100*(new_tokens/old_tokens-1) if old_tokens else None,
        'v1_5_wrong_checks':sum(r['grader_timeline']['wrong'] for r in audited),
        'core_v1_wrong_checks':sum(r['control']['grader_timeline']['wrong'] for r in audited),
        'median_paired_difference_s':statistics.median(p['difference_s'] for p in result['paired_seed_comparisons']) if result['paired_seed_comparisons'] else None,
        'scope':'Client-observed output IDs; not a complete GPU compute or server-generated token measurement.'}
    with (batch/'comparison.csv').open('w',newline='') as file:
        fields=['seed','core_v1_s','v1_5_s','difference_s','core_v1_requests','v1_5_requests',
                'later_fresh_requests','continuation_requests','core_v1_wrong','v1_5_wrong']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for r in audited:
            writer.writerow(dict(zip(fields,[r['sampling_seed'],r['control']['time_to_target_s'],
                r['recorded_time_to_target_s'],r['recorded_time_to_target_s']-r['control']['time_to_target_s'] if r['recorded_time_to_target_s'] is not None else None,
                r['control']['generation_requests'],r['generation_requests'],r['later_fresh_requests'],
                r['continuation_requests'],r['control']['grader_timeline']['wrong'],r['grader_timeline']['wrong']])))
    atomic_json(batch/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True)
    args=parser.parse_args(argv)
    if Path(args.batch).name != args.batch or args.batch in ('.','..'):
        parser.error('Use a plain batch directory name')
    analyze(ROOT/'runs/experiments'/args.batch)


if __name__ == '__main__':
    main()
