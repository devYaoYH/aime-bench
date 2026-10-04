"""Plan/execute sequential core v2.1 benchmark measurements on AIME and Apex."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import statistics
import subprocess

from src.common import ROOT, atomic_json, utc_now
from src.experiments.post_freeze import with_official_deadline
from runner_final.core_v2_1 import runner
from runner_final.core_v2_1._runtime import attempt_lock
from runner_final.integrity_v2_1 import verify_core
from runner_final.run_frozen_v2_1 import parse_args

PRESETS = ['math_core_v2_1.json', 'apex_core_v2_1.json']


def plan(seed):
    return [{'dataset': name, 'argv':['--preset', str(ROOT/'runner_final/presets'/preset),
                                    '--seed',str(seed),'--benchmark']}
            for name,preset in zip(('AIME 2025','Apex shortlist'),PRESETS)]


def measure(folder):
    config=json.loads((folder/'config.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    rows=[json.loads(line) for p in folder.glob('trace/*/candidate_validation.jsonl')
          for line in p.read_text().splitlines() if line]
    checks=[json.loads(line) for p in folder.glob('trace/*/verification.jsonl')
            for line in p.read_text().splitlines() if line]
    verdicts=[r for r in checks if type(r.get('result',{}).get('verdict')) is bool]
    correct=[r for r in verdicts if r['result']['verdict']]
    questions=[json.loads(p.read_text()) for p in folder.glob('trace/*/question.json')]
    valid=(config['runner_id']=='runner_final_core_v2_1' and
           config['core_manifest_sha256']==verify_core(ROOT) and not config['git_dirty'] and
           config['benchmark'] and all(len(q['rollouts'])<=4 for q in questions))
    values=lambda field:[r[field] for r in rows if r.get(field) is not None]
    return {'attempt_id':folder.name,'status':summary['status'],'identity_valid':valid,
            'target_reached':summary.get('target_reached'), 'solved':len(correct),
            'time_to_18_s':summary.get('time_to_target_s'),
            'official_latency_s':summary.get('official_latency_s'),
            'initialization_and_attempt_latency_s':summary.get('initialization_and_attempt_latency_s'),
            'performance':summary.get('performance'), 'queries_completed':len(verdicts),
            'wrong_checks':len(verdicts)-len(correct),
            'candidate_outcomes':dict(Counter(r['outcome'] for r in rows)),
            'rejection_reasons':dict(Counter(r.get('reason') for r in rows if not r['valid'])),
            'normalization_fallbacks':dict(Counter(r.get('canonical_fallback') for r in rows if r.get('canonical_fallback'))),
            'validation_cpu_s':sum(values('syntax_cpu_s'))+sum(values('canonical_cpu_s')),
            'validation_wall_median_ms':statistics.median(values('validation_wall_s'))*1000 if rows else None,
            'validation_wall_max_ms':max(values('validation_wall_s'))*1000 if rows else None,
            'validation_queue_s_sum':sum(values('queue_s')),
            'scope':'Required CPU/queue/timing traces in benchmark mode; optional GPU/engine/host profiling disabled. Overlapping async waits are not additive latency.'}


async def execute(options):
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Use a clean pinned worktree; preserve unrelated work in the primary checkout')
    batch=ROOT/'runs/experiments'/options.batch
    batch.mkdir(parents=True,exist_ok=False)
    records={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'core_manifest_sha256':verify_core(ROOT),'started_at_utc':utc_now(),
             'seed':options.seed,'official_deadline_s':options.official_deadline_s,
             'jobs':plan(options.seed),'trials':[],'status':'running'}
    atomic_json(batch/'config.json',records)
    for job in records['jobs']:
        trial={'dataset':job['dataset'],'started_at_utc':utc_now(),'compute_mode':'unattended sequential benchmark'}
        records['trials'].append(trial)
        args=parse_args(job['argv'])
        outcome=await with_official_deadline(runner.run,args,ROOT/'attempts',options.official_deadline_s)
        trial.update(outcome)
        trial['finished_at_utc']=utc_now()
        if outcome.get('attempt_id'):
            trial['measurement']=measure(ROOT/'attempts'/outcome['attempt_id'])
        atomic_json(batch/'summary.json',records)
    records.update(status='complete',finished_at_utc=utc_now())
    atomic_json(batch/'summary.json',records)
    print(json.dumps(records,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--seed',type=int,default=20261011)
    parser.add_argument('--batch',default='core-v2_1-benchmarks-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--official-deadline-s',type=float,default=900)
    args=parser.parse_args()
    if args.official_deadline_s<=0:parser.error('Deadline must be positive')
    if not args.execute:
        print(json.dumps({'jobs':plan(args.seed),'deadline_s':args.official_deadline_s,'execute':False},indent=2));return
    with attempt_lock():
        async def entry():
            task=asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM,task.cancel)
            await execute(args)
        asyncio.run(entry())


if __name__=='__main__':main()
