"""Plot saved request lifetimes, budget stages and first-solved verdicts."""
from pathlib import Path
from collections import Counter
import argparse
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(attempt, output):
    records=[json.loads(p.read_text()) for p in attempt.glob('trace/*/rollout-*/telemetry.json')]
    records=[r for r in records if r.get('generation_end_monotonic_s') is not None]
    first_record=min(records,key=lambda r:r['start_monotonic_s'])
    config=json.loads((attempt/'config.json').read_text())
    stamp=lambda x:datetime.fromisoformat(x.replace('Z','+00:00')).timestamp()
    first=first_record['start_monotonic_s']-(stamp(first_record['started_at_utc'])-stamp(config['official_started_at_utc']))
    events=[]
    for r in records:
        lane=1 if r['continuation_of_rollout'] is not None else 0
        events.extend([(r['start_monotonic_s']-first,1,lane),
                       (r['generation_end_monotonic_s']-first,-1,lane)])
    # Ends before starts at equal times; monotonic timestamps avoid UTC rounding.
    times,fresh,continued=[0],[0],[0]
    active=[0,0];peak=0
    for t,delta,lane in sorted(events):
        active[lane]+=delta
        assert min(active)>=0
        times.append(t);fresh.append(active[0]);continued.append(active[1]);peak=max(peak,sum(active))
    assert active==[0,0]
    allocation=json.loads((attempt/'allocation.json').read_text())
    assert peak<=allocation['max_concurrent_requests']
    solved=sorted(json.loads(line)['first_solved_elapsed_s'] for line in (attempt/'solved.jsonl').read_text().splitlines())
    counts=Counter(r['target_generated_tokens'] for r in records)
    fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained',gridspec_kw={'height_ratios':[2,1.2,1.2]})
    axes[0].stackplot(times,fresh,continued,step='post',labels=['Fresh samples','Continuation segments'],colors=['#2588ad','#dc8d39'],alpha=.8)
    axes[0].axhline(allocation['max_concurrent_requests'],color='#485664',ls='--',lw=1)
    trigger=allocation['first_submission']
    if trigger:axes[0].axvline(trigger['elapsed_s'],color='#845bae',ls=':',label='First grader submission')
    axes[0].set_ylabel('Client requests in flight');axes[0].set_ylim(0,allocation['max_concurrent_requests']+7)
    axes[0].legend(loc='upper right',fontsize=9);axes[0].set_xlabel('Official elapsed seconds (client timestamp alignment)')
    axes[1].step([0,*solved],[0,*range(1,len(solved)+1)],where='post',color='#268662',lw=2)
    axes[1].axhline(18,color='#485664',ls='--',lw=1);axes[1].set_ylim(0,20);axes[1].set_ylabel('Verified correct')
    axes[1].set_xlabel('Official elapsed seconds')
    if len(solved)>=18:axes[1].text(solved[17],18.4,f'18 at {solved[17]:.2f}s',ha='right')
    budgets=allocation['token_budgets'];axes[2].bar([str(x//1024)+'K' for x in budgets],[counts[x] for x in budgets],color='#4d7897')
    axes[2].set_xlabel('Cumulative generated-token target per trajectory');axes[2].set_ylabel('Generation requests')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('NVFP4 dynamic admission · maximum 60 streams · RAM-buffered benchmark',fontsize=13)
    output.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(output/f'allocation-timeline.{ext}',dpi=170)
    svg=output/'allocation-timeline.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    result={'observed_peak_client_requests_in_flight':peak,'fresh_requests':sum(r['continuation_of_rollout'] is None for r in records),'continuation_requests':sum(r['continuation_of_rollout'] is not None for r in records),'request_counts_by_cumulative_target':dict(counts), 'scope':'Client lifetimes include HTTP queue/header/decode/cancellation time; these are not sampled engine-running counts. First-solved panel uses official elapsed verdict times.'}
    (output/'timeline-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();render(args.attempt,args.output)
