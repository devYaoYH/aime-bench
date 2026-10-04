"""Rebuild gold-free v2.3 benchmark evidence and solve timelines."""
import argparse
from collections import Counter
import json
from pathlib import Path

from src.common import ROOT, atomic_json
from src.experiments.benchmark_core_v2_3 import measure


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row] if path.exists() else []


def analyze(folder):
    config=json.loads((folder/'config.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    result=measure(folder)
    solved={}
    for event in lines(folder/'solved.jsonl'):
        solved[event['problem_idx']]=min(solved.get(event['problem_idx'],float('inf')),event['first_solved_elapsed_s'])
    if len(solved)!=summary['solved'] or len(solved)!=result['solved']:
        raise ValueError('Distinct first-solved evidence disagrees with completed verdicts')
    if any('answer' in q for q in json.loads((folder/'questions.json').read_text())):
        raise ValueError('Question snapshot must remain gold-free')
    if not result['identity_valid']:
        raise ValueError('Frozen identity, slot budget or request cap mismatch')
    target=config['target_correct']
    times=sorted(solved.values())
    expected=times[target-1] if len(times)>=target else None
    if expected!=summary['time_to_target_s']:
        raise ValueError('Target time disagrees with distinct first-solved events')
    generation=[];exact=[];initial_matches=0;fork_matches=0;continuation_count=0
    for path in sorted(folder.glob('trace/*/rollout-*/request.json')):
        request=json.loads(path.read_text())
        tokens=json.loads(path.with_name('tokens.json').read_text())
        telemetry=json.loads(path.with_name('telemetry.json').read_text())
        generation.append(telemetry);exact.append(tokens)
        if not tokens['prompt_token_ids']:
            raise ValueError('Missing served prompt IDs')
        if len(tokens['prompt_token_ids'])+request['max_tokens']>config['max_context_tokens']:
            raise ValueError('Request exceeds total served context')
        if telemetry['rollout']==1:
            index=str(int(path.parent.parent.name))
            if len(tokens['prompt_token_ids'])!=config['served_prompt_tokens'][index]:
                raise ValueError('Pre-counted template differs from served initial prompt')
            initial_matches+=1
        if 'prompt' in request:
            continuation_count+=1
            if tokens['prompt_token_ids']!=request['prompt']:
                raise ValueError('Served continuation prefix differs from requested IDs')
            parent=path.parent.parent/f"rollout-{telemetry['continuation_of_rollout']:02d}"/'tokens.json'
            saved=json.loads(parent.read_text());prefix=saved['prompt_token_ids']+saved['output_token_ids']
            if not saved['complete'] or request['prompt'][:len(prefix)]!=prefix:
                raise ValueError('Continuation does not preserve exact parent prefix')
            fork_matches+=1
    feedback=[r for path in folder.glob('trace/*/feedback.jsonl') for r in lines(path)]
    feedback_winners=[]
    for path in folder.glob('trace/*/question.json'):
        question=json.loads(path.read_text())
        winner=question.get('winner')
        if winner and any(r['rollout']==winner['rollout'] and r.get('continuation_of_rollout') is not None for r in question['rollouts']):
            feedback_winners.append(question['problem_idx'])
    result.update(solve_events=[{'problem_idx':q,'elapsed_s':s} for q,s in sorted(solved.items(),key=lambda pair:pair[1])],
                  initial_prompt_matches=initial_matches,continuation_prefix_matches=fork_matches,
                  continuation_requests=continuation_count,exact_records=len(exact),
                  exact_complete=sum(r['complete'] for r in exact),
                  generation_statuses=dict(Counter(r['status'] for r in generation)),
                  output_tokens=sum(r['generated_token_ids_count'] for r in generation),
                  verified_continuation_questions=sorted(feedback_winners),
                  not_forked_reasons=dict(Counter(r.get('reason') for r in feedback if r['kind']=='not_forked')),
                  grader_timeline=summary.get('grader_timeline'))
    return result


def render(batch):
    records=json.loads((batch/'summary.json').read_text())
    if records['status']!='complete':raise ValueError('Wait for both declared trials to finish')
    rows=[]
    for trial in records['trials']:
        if trial.get('measurement_error') or not trial.get('attempt_id'):
            raise ValueError('Retain and report failed/partial attempts explicitly before rendering')
        row=analyze(ROOT/'attempts'/trial['attempt_id'])
        row.update(dataset=trial['dataset'],deadline_reached=trial['deadline_reached'],
                   official_deadline_s=trial['official_deadline_s'])
        rows.append(row)
    result={'source_commit':records['source_commit'],'seed':records['seed'],
            'core_manifest_sha256':records['core_manifest_sha256'],'trials':rows,
            'scope':'One trial per dataset; required benchmark evidence, no optional profiles.'}
    reference=ROOT/'runs/experiments/core-v2_1-benchmarks-20261004T161700Z/profile.json'
    comparisons=[]
    if reference.exists():
        historical=json.loads(reference.read_text())
        for row,old in zip(rows,historical['trials']):
            comparisons.append({'dataset':row['dataset'],'reference_attempt_id':old['attempt_id'],
                'reference_time_to_18_s':old['time_to_18_s'],
                'reference_solved_within_current_cutoff':sum(e['elapsed_s']<=row['official_deadline_s'] for e in old['solve_events']),
                'current_cutoff_s':row['official_deadline_s'],
                'scope':'Historical same-seed v2.1 comparison; scheduler, request budgets, feedback and Apex concurrency differ.'})
    result['historical_same_seed_v2_1']=comparisons
    observations=batch/'service_observations.json'
    if observations.exists():
        result['service_observations']=json.loads(observations.read_text())
    atomic_json(batch/'profile.json',result)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,len(rows),figsize=(12,4.2),squeeze=False)
    for ax,row in zip(axes[0],rows):
        events=row['solve_events']
        xs=[0]+[r['elapsed_s'] for r in events]+[row['official_latency_s']]
        ys=[0]+list(range(1,len(events)+1))+[len(events)]
        ax.step(xs,ys,where='post',color='#176b87',linewidth=2)
        ax.axhline(18,color='#bd493d',linestyle='--')
        ax.text(.02,.94,'18 verified target',transform=ax.transAxes,color='#bd493d',fontsize=10)
        ax.set(title=row['dataset'],xlabel='Official elapsed seconds',ylabel='Distinct verified questions',ylim=(0,20))
        ax.set_yticks([0,5,10,15,18,20]);ax.grid(alpha=.2)
        ax.set_xlim(0,row['official_latency_s']*1.03)
    fig.suptitle('Core v2.3 — input-sized slot pool, long rollouts, benchmark mode')
    fig.tight_layout();fig.savefig(batch/'timings.png',dpi=160,bbox_inches='tight');plt.close(fig)
    table=['| Dataset | Verified | Time to 18 | Official duration | Completed / wrong checks | Peak slots |',
           '|---|---:|---:|---:|---:|---:|']
    details=[]
    for row in rows:
        timing='unmet' if row['time_to_18_s'] is None else f"{row['time_to_18_s']:.3f}s"
        pool=row['allocation'];kinds=Counter(a['kind'] for a in pool['admissions'])
        table.append(f"| {row['dataset']} | {row['solved']} | {timing} | {row['official_latency_s']:.3f}s | {row['queries_completed']} / {row['wrong_checks']} | {pool['peak_active_requests']} / {pool['max_concurrent_requests']} |")
        details.append(f"### {row['dataset']} — `{row['attempt_id']}`\n\n"
            f"Status `{row['status']}`; deadline reached `{row['deadline_reached']}`; "
            f"official cutoff {row['official_deadline_s']:g}s. Admissions: `{dict(kinds)}`. "
            f"Every question stayed within four generation requests. Candidate outcomes: `{row['candidate_outcomes']}`. "
            f"Feedback events: `{row['feedback_events']}`; fork skips: `{row['not_forked_reasons']}`.\n\n"
            f"Verified continuation questions: `{row['verified_continuation_questions']}`. "
            "Queued forks are plans; a cutoff can cancel a plan before admission, so queued and executed counts can differ.\n\n"
            f"Initial template counts agree with served IDs for {row['initial_prompt_matches']} questions. "
            f"All {row['continuation_prefix_matches']}/{row['continuation_requests']} continuations preserve exact parent prefixes, "
            f"and every request fits the total context budget. Exact output evidence: {row['exact_complete']}/{row['exact_records']} complete. "
            f"Generation statuses: `{row['generation_statuses']}`; {row['output_tokens']} received output token IDs. "
            f"Required validation CPU: {row['validation_cpu_s']:.3f}s; proposal median/max wall: "
            f"{row['validation_wall_median_ms']:.3f}/{row['validation_wall_max_ms']:.3f}ms.\n\n"
            f"Completed grader jobs occupied {row['grader_timeline']['actual_service_s']:.3f}s of service; "
            f"idle gaps between completed jobs summed to {row['grader_timeline']['idle_between_queries_s']:.3f}s.\n\n"
            f"Fresh / continuation median TTFT: `{row['performance']['fresh_ttft']}` / "
            f"`{row['performance']['continuation_ttft']}` (seconds; different workloads, not a paired cache speed test).\n\n"
            f"[Config](../../../attempts/{row['attempt_id']}/config.json), "
            f"[summary](../../../attempts/{row['attempt_id']}/summary.json), "
            f"[allocation](../../../attempts/{row['attempt_id']}/allocation.json), "
            f"[solved events](../../../attempts/{row['attempt_id']}/solved.jsonl).")
    text=("# Core v2.3 benchmark measurements\n\n"
          f"Measured source `{records['source_commit']}`, manifest `{records['core_manifest_sha256']}`, seed `{records['seed']}`. "
          "Two sequential, independently owned vLLM/grader runs on callosum's A100 80GB, using NVFP4 VibeThinker-3B, "
          "95% configured GPU memory, BF16 KV, FlashInfer, prefix caching and the unchanged v2 prompt. "
          "Cheap warmup is 32 tokens per input-sized slot. AIME uses 30 slots; Apex uses 47. "
          "One initial long request per question, then ready continuations or least-active fresh samples; "
          "65,536 total context, queue-aware batched wrong-verdict forks and four requests per question. "
          "Target remains 18 distinct verified questions. Setup/warmup, cleanup and trace flush are excluded from the official cutoff.\n\n"
          +'\n'.join(table)+"\n\n![Distinct verified solve timelines](timings.png)\n\n"
          +'\n\n'.join(details)+"\n\n## Historical same-seed comparison\n\n"
          +'\n\n'.join(f"{r['dataset']}: v2.1 reference `{r['reference_attempt_id']}` reached "
              +(f"18 in {r['reference_time_to_18_s']:.3f}s" if r['reference_time_to_18_s'] is not None else 'an unmet target')
              +f"; {r['reference_solved_within_current_cutoff']} questions were verified within {r['current_cutoff_s']:g}s."
              for r in comparisons)
          +"\n\nThese historical runs differ in scheduling, long-request budgets and feedback, and Apex changes from 30 to 47 slots. "
          "A single comparison does not isolate any one change.\n\n"
          +("## Default server-log observations\n\n"+'\n\n'.join(
             f"`{o['attempt_id']}`: {o['sample_count']} periodic samples; maximum observed KV-cache occupancy "
             f"{o['max_observed_kv_cache_percent']:.1f}%; maximum waiting requests {o['max_observed_waiting_requests']}; "
             f"{o['preemption_warning_lines']} preemption-warning lines and {o['oom_error_lines']} OOM-error lines."
             for o in result['service_observations']['trials'])
             +"\n\nThese are existing service logs, including warmup, not an exact eviction counter or physical VRAM peak. "
             "[Extracted observations](service_observations.json).\n\n" if 'service_observations' in result else '')
          +"## Limits\n\n"
          "- One seed per dataset establishes execution and observed timings, not repeatability or an isolated optimization effect.\n"
          "- Apex's 300-second cutoff differs from the older v2.1 900-second trial; compare solved counts at equal elapsed time. Two Apex questions overlap AIME 2025.\n"
          "- Benchmark mode disables GPU/engine/optional host profiles; configured 95% memory is not a measured VRAM peak. No KV eviction rate is inferred from timing alone.\n"
          "- Required validation CPU is worker process time; wall/queue waits overlap and cannot be added to official latency. Cancelled generation latencies are censored.\n"
          "- Client verdict counts exclude cancelled checks that may still incur grader service. Full SSE, grader audits and service logs remain in the pinned remote checkout.\n\n"
          f"Rebuild: `.venv/bin/python -m scripts.profile_v2_3_benchmarks runs/experiments/{batch.name}`. "
          "Analysis reads saved candidates and actual grader verdicts, never reference answers.\n")
    (batch/'README.md').write_text(text)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    args=parser.parse_args();render(args.batch)


if __name__=='__main__':main()
