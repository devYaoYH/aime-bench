"""Rebuild a gold-free benchmark timing report from required v2.1 artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from src.common import ROOT, atomic_json
from src.experiments.benchmark_core_v2_1 import measure

REFERENCE='20261004T013249.730690Z'


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row] if path.exists() else []


def distribution(values):
    return {'count':len(values), 'median_ms':statistics.median(values)*1000 if values else None,
            'max_ms':max(values)*1000 if values else None}


def analyze(attempt):
    summary=json.loads((attempt/'summary.json').read_text())
    proposals=[row for path in attempt.glob('trace/*/candidate_validation.jsonl') for row in lines(path)]
    banked=lines(attempt/'solved.jsonl')
    unique={}
    for event in banked:
        index=event['problem_idx']
        unique[index]=min(unique.get(index,float('inf')),event['first_solved_elapsed_s'])
    if len(unique)!=summary['solved']:
        raise ValueError('Distinct first-solved evidence disagrees with summary')
    if any('answer' in q for q in json.loads((attempt/'questions.json').read_text())):
        raise ValueError('Question snapshot must remain gold-free')
    result=measure(attempt)
    generation=[json.loads(p.read_text()) for p in attempt.glob('trace/*/rollout-*/telemetry.json')]
    exact=[json.loads(p.read_text()) for p in attempt.glob('trace/*/rollout-*/tokens.json')]
    result.update(exact_prefix_records=len(exact), exact_prefix_complete=sum(r['complete'] for r in exact),
                  generation_statuses=dict(Counter(r['status'] for r in generation)),
                  generation_censored=sum(r['generation_censored'] for r in generation),
                  observed_generation_latency=distribution([r['generation_latency_s'] for r in generation]),
                  observed_end_to_end_latency=distribution([r['end_to_end_latency_s'] for r in generation]),
                  completion_token_ids=sum(r['generated_token_ids_count'] for r in generation))
    if not result['identity_valid']:
        raise ValueError('Benchmark identity or request cap invalid')
    result.update(cached=distribution([r['validation_wall_s'] for r in proposals if r['cache_hit']]),
                  uncached=distribution([r['validation_wall_s'] for r in proposals if not r['cache_hit']]),
                  cache_hits=sum(r['cache_hit'] for r in proposals),
                  cache_misses=sum(not r['cache_hit'] for r in proposals),
                  grader_timeline=summary.get('grader_timeline'),
                  solve_events=[{'problem_idx':q,'elapsed_s':s} for q,s in sorted(unique.items(),key=lambda pair:pair[1])])
    return result,proposals


def render(batch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np
    records=json.loads((batch/'summary.json').read_text())
    if records['status']!='complete':raise ValueError('Wait for the requested batch to finish')
    rows=[];proposals=[]
    for trial in records['trials']:
        if not trial.get('attempt_id'):raise ValueError('Missing attempt; retain startup failure explicitly')
        row,events=analyze(ROOT/'attempts'/trial['attempt_id'])
        row['dataset']=trial['dataset']; row['deadline_reached']=trial.get('deadline_reached'); rows.append(row);proposals.append(events)
    old=ROOT/'attempts'/REFERENCE
    historical=json.loads((old/'summary.json').read_text())
    checks=[row for path in old.glob('trace/*/verification.jsonl') for row in lines(path)]
    control={'attempt_id':REFERENCE,'time_to_18_s':historical['time_to_target_s'],
             'completed_checks':sum(type(r.get('result',{}).get('verdict')) is bool for r in checks),
             'wrong_checks':sum(r.get('result',{}).get('verdict') is False for r in checks)}
    result={'source_commit':records['source_commit'],'seed':records['seed'],
            'official_deadline_s':records['official_deadline_s'],
            'core_manifest_sha256':records['core_manifest_sha256'],'trials':rows,
            'historical_same_seed_v2_aime_control':control,
            'limitations':['One new trial per dataset; no repeatability estimate.',
                           'Benchmark mode omits optional host/GPU/engine profiles; no measured VRAM peak or KV eviction trace.',
                           'Apex has 47 questions, including two overlapping AIME 2025 questions; target remains 18.',
                           'Validation wall includes queue/IPC/cache work; overlapping async waits cannot be added to official latency.']}
    atomic_json(batch/'profile.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.89,hspace=.40,wspace=.27)
    for column,(row,events) in enumerate(zip(rows,proposals)):
        ax=axes[0,column];events_s=row['solve_events'];xs=[0]+[r['elapsed_s'] for r in events_s]+[row['official_latency_s']]
        ys=[0]+list(range(1,len(events_s)+1))+[len(events_s)]
        ax.step(xs,ys,where='post',color='#176b87',linewidth=2)
        ax.axhline(18,color='#bd493d',linestyle='--',label='18 verified target')
        ax.set(title=row['dataset'],xlabel='Official elapsed seconds',ylabel='Distinct verified questions',ylim=(0,20))
        ax.set_yticks([0,5,10,15,18,20]);ax.grid(alpha=.2);ax.legend(loc='lower right')
        ax=axes[1,column]
        positive=[r['validation_wall_s']*1000 for r in events if r['validation_wall_s']>0]
        bins=np.geomspace(min(positive)*.9,max(positive)*1.1,32)
        for cached,color,label in [(True,'#aaa','Cache hits'),(False,'#176b87','Uncached validation')]:
            data=[r['validation_wall_s']*1000 for r in events if r['cache_hit']==cached and r['validation_wall_s']>0]
            if data:ax.hist(data,bins=bins,weights=np.ones(len(data))*100/len(data),alpha=.7,color=color,label=f'{label} (n={len(data)})')
        ax.set(xscale='log',xlabel='Per-proposal validation wall time (ms, log)',ylabel='Share within cache group (%)')
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Core v2.1: required timings with --benchmark (one seed per dataset)',fontsize=13)
    fig.savefig(batch/'timings.png',dpi=160,bbox_inches='tight');plt.close(fig)
    def fmt(value):return 'unmet' if value is None else f'{value:.3f}'
    table=['| Dataset | Verified | Time to 18 | Official duration | Completed / wrong checks | Validation CPU |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f"| {row['dataset']} | {row['solved']} | {('unmet' if row['time_to_18_s'] is None else fmt(row['time_to_18_s'])+'s')} | {fmt(row['official_latency_s'])}s | {row['queries_completed']} / {row['wrong_checks']} | {row['validation_cpu_s']:.3f}s |")
    detail=[]
    for row in rows:
        outcomes=row['candidate_outcomes']; timing=row['grader_timeline'] or {}; performance=row['performance'] or {}
        fresh=performance.get('fresh_ttft',{}); continued=performance.get('continuation_ttft',{})
        ttft=lambda value: f"{value.get('median_s',0)*1000:.1f}/{value.get('p95_s',0)*1000:.1f}ms (n={value.get('count',0)})"
        detail.append(f"### {row['dataset']} — `{row['attempt_id']}`\n\n"
            f"Status `{row['status']}`; target reached `{row['target_reached']}`; external deadline reached `{row['deadline_reached']}`. "
            f"{outcomes.get('enqueued',0)} unique candidates enqueued; "
            f"{outcomes.get('rejected',0)} syntax/placeholder rejections; "
            f"{outcomes.get('duplicate_expression',0)} expression duplicates suppressed. "
            f"Rejection reasons: `{json.dumps(row['rejection_reasons'],sort_keys=True)}`. "
            f"Normalization fallbacks: `{json.dumps(row['normalization_fallbacks'],sort_keys=True)}`.\n\n"
            f"{row['cache_hits']} cached / {row['cache_misses']} uncached proposals. "
            f"Uncached median/max validation wall: {fmt(row['uncached']['median_ms'])}/{fmt(row['uncached']['max_ms'])}ms; "
            f"cached median: {fmt(row['cached']['median_ms'])}ms. "
            f"Validation queue waits summed to {row['validation_queue_s_sum']:.6f}s (overlapping waits).\n\n"
            f"Exact token records: {row['exact_prefix_complete']}/{row['exact_prefix_records']} complete, including saved cancelled streams. "
            f"Generation statuses: `{json.dumps(row['generation_statuses'])}`; {row['generation_censored']} observations censored by a cap or cancellation. "
            f"Observed generation elapsed median/max: {row['observed_generation_latency']['median_ms']/1000:.3f}/{row['observed_generation_latency']['max_ms']/1000:.3f}s; "
            f"observed end-to-end median/max (through question settlement): {row['observed_end_to_end_latency']['median_ms']/1000:.3f}/{row['observed_end_to_end_latency']['max_ms']/1000:.3f}s. "
            f"{row['completion_token_ids']} completion token IDs recorded. Censored elapsed times are not intrinsic completion latencies.\n\n"
            f"Fresh TTFT median/p95: {ttft(fresh)}. Continuation TTFT median/p95: {ttft(continued)}. "
            f"The first grader job began at {timing.get('first_pick_elapsed_s',0):.3f}s; "
            f"completed jobs occupied {timing.get('actual_service_s',0):.3f}s of grader service, "
            f"with {timing.get('idle_between_queries_s',0):.3f}s idle between completed jobs. "
            "These measurements exclude any service incurred after cancellation.\n\n"
            f"[Config](../../../attempts/{row['attempt_id']}/config.json), "
            f"[summary](../../../attempts/{row['attempt_id']}/summary.json), "
            f"[first-solved events](../../../attempts/{row['attempt_id']}/solved.jsonl).")
    text=("# Core v2.1 benchmark-mode measurements\n\n"
          f"Measured source `{records['source_commit']}`, manifest `{records['core_manifest_sha256']}`, "
          f"seed `{records['seed']}`. Two sequential, dedicated-service runs with NVFP4 VibeThinker-3B, "
          "95% configured GPU memory, cheap 30×32 warmup, unchanged v2 prompt, 30×1 barrier, "
          "8K first output / 16K additional continuations / four total requests per question. "
          f"Each trial has an external {records['official_deadline_s']:g}s official deadline; "
          "unmet/deadline outcomes are retained without replacement. Setup/warmup and final flush are outside official solving time.\n\n"
          + '\n'.join(table) + "\n\n![Solve timeline and CPU validation timings](timings.png)\n\n"
          + '\n\n'.join(detail)
          + f"\n\n## Historical comparison and limits\n\nThe historical same-seed v2 AIME control "
          f"[`{REFERENCE}`](../../../attempts/{REFERENCE}/summary.json) reached 18 in "
          f"{control['time_to_18_s']:.3f}s, with {control['completed_checks']} completed checks and "
          f"{control['wrong_checks']} wrong verdicts. The new AIME result is {rows[0]['time_to_18_s']:.3f}s, "
          f"with {rows[0]['wrong_checks']} wrong verdicts. Recorded model/profile, prompt, sampling, seed, "
          "schedule and budgets match; v2.1 adds the CPU parser dependencies. These runs occurred at different times. "
          "This is one historical comparison, not a replicated isolated effect.\n\n"
          + '\n'.join('- '+s for s in result['limitations'])
          + "\n- Optional profiling is disabled deliberately: CPU validation's required timers remain, but extraction/IO/event-loop overhead and official VRAM/KV series are unavailable.\n"
          + "- CPU cost is recorded worker process time; cache hits record zero worker CPU. Wall timers also include IPC and waiting. Neither total is an estimate of added critical-path latency.\n"
          + "- Completed client checks may differ from all server work: cancelled/queued jobs can still incur grader service. Required client traces remain in Git; raw SSE, grader audits and service logs remain on the pinned remote worktree.\n\n"
          + "## V2.2 status\n\n[Queue-aware runtime correction](../../../runner_final/core_v2_2/README.md) "
          "is separately frozen and pushed as `550d12eb`. It drains queued answers and pending CPU validation, "
          "batches actual wrong verdicts, rechecks after server tokenization, then cancels replaceable live lanes "
          "and forks exact prefixes with feedback. All correction requests consume the four-request cap. "
          "The prompt and CPU parser remain unchanged. All 320 offline tests passed in a clean source copy "
          "with existing ignored legacy fixtures and local socket/macOS sandbox support. No v2.2 GPU speed result is claimed.\n\n"
          + "Rebuild: `.venv/bin/python -m scripts.profile_v2_1_benchmarks runs/experiments/"
          + batch.name + "`. This analysis reads question statements and recorded client verdicts, never reference answers.\n")
    (batch/'README.md').write_text(text)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    args=parser.parse_args();render(args.batch)


if __name__=='__main__':main()
