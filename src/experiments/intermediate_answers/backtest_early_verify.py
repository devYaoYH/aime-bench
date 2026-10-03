"""Back-test early verification on samples 1-4 of the original Qwen run.

Use this offline experiment to compare final, marker, and permissive candidates
with a shared three-second verifier and 120/30/8 generation slots. Source records
and the cached Qwen tokenizer provide estimated intermediate arrival times.
Writes summaries, inventories, CSV statistics, and plots in early_verify_pass4/.
Keys stand in for a perfect verifier; live cancellation and throughput are not
measured. The reusable queue simulator lives in src.verification_replay.
    python -m src.experiments.intermediate_answers.backtest_early_verify
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
import os
from pathlib import Path

from src.answer_extraction import make_rows
from src.common import ROOT, atomic_json
from src.verification_replay import candidate_schedule, simulate
from tokenizers import Tokenizer

GENERATION_SLOTS = (120, 30, 8)


def arrival_fraction(row, event, tokenizer):
    """Wait for a completed line, as the streaming detector does."""
    text = row['_texts'][event['part']]
    boundary = text.find('\n', event['end'])
    boundary = boundary+1 if boundary >= 0 else len(text)
    ends = row['_token_ends'][event['part']]
    part_prefix = row['reasoning_tokens'] if event['part']=='content' else 0
    visible = part_prefix + min(len(ends), bisect_left(ends, boundary)+1)
    return min(1.0, (visible+max(0,row['provider_minus_visible_tokens']))/row['completion_tokens'])


def load_rows(run, tokenizer_path, samples):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = [r for r in make_rows(run, tokenizer) if r['sample_number'] <= samples]
    if len(rows) != 30*samples: raise ValueError('Incomplete selected sample cohort')
    for row in rows:
        record = json.loads((ROOT/row['source_file']).read_text())
        message = record['response']['choices'][0]['message']
        row['_texts'] = {part: message.get(part) or '' for part in ['reasoning','content']}
        row['_token_ends'] = {part:[end for start,end in tokenizer.encode(text, add_special_tokens=False).offsets]
                              for part,text in row['_texts'].items()}
        row['candidates'] = [{**e, 'output_fraction': arrival_fraction(row,e,tokenizer)} for e in row['events']]
        del row['_texts'], row['_token_ends']
    return rows


def render(summary, output):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.local/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(len(GENERATION_SLOTS),1,figsize=(10.8,13.4),sharey=True)
    colors={'final':'#53657d','markers':'#e29b31','permissive':'#008b83'}
    labels={'final':'Completed final answers','markers':'Earlier boxes / Answer: lines',
            'permissive':'Permissive intermediate checking'}
    for ax,slots in zip(axes,GENERATION_SLOTS):
        title = 'All 120 trajectories start together' if slots==120 else f'{slots} generation slots; FIFO sample rounds'
        selected=[r for r in summary['replays'] if r['generation_slots']==slots and r['exponent']==1 and r['first_token_s']==0]
        for r in selected:
            times=[0]+[m['time_s']/60 for m in r['milestones']]
            counts=[0]+list(range(1,len(times)))
            end=max(x['milestones'][-1]['time_s']/60 for x in selected if x['milestones'])+0.3
            ax.step(times+[end],counts+[counts[-1]],where='post',linewidth=2.25,
                    color=colors[r['policy']],label=labels[r['policy']])
            target=r['time_to_18_s']
            if target is not None:
                ax.plot(target/60,18,'o',ms=5,color=colors[r['policy']])
        ax.axhline(18,color='#9aa4ae',ls='--',lw=1)
        final=next(r for r in selected if r['policy']=='final')
        early=next(r for r in selected if r['policy']=='permissive')
        savings=1-early['time_to_18_s']/final['time_to_18_s']
        ax.set_title(title,pad=14,fontsize=12,fontweight='bold')
        ax.set_xlabel('Estimated elapsed time (minutes)')
        ax.grid(alpha=.18)
        ax.set_ylim(0,25)
        ax.set_yticks([0,5,10,15,18,20,22,25])
        ax.text(.97,.08,f"18 correct: {final['time_to_18_s']/60:.2f} → {early['time_to_18_s']/60:.2f} min\n{savings:.1%} less time to threshold",
                transform=ax.transAxes,ha='right',va='bottom',fontsize=11,
                bbox={'facecolor':'white','edgecolor':'#dce2e7','alpha':.95,'boxstyle':'round,pad=.45'})
        ax.set_ylabel('Questions verified correct (of 30)')
    handles,labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels_,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Pass@4: verify candidates while generation continues',fontsize=16,fontweight='bold',y=.98)
    fig.text(.5,.035,'Samples 1–4 only • one shared verifier, 3 s per check • wrong candidates consume slots\n'
             'Oracle exact-answer decisions; intermediate times estimated by token fraction. Fixed saved generation durations.',
             ha='center',fontsize=9,color='#53657d')
    fig.subplots_adjust(top=.92,bottom=.15,hspace=.42)
    for suffix in ['png','svg','pdf']:
        fig.savefig(output/f'time_to_correct.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main(args):
    run=(ROOT/'runs'/args.run).resolve()
    if not run.is_relative_to(ROOT/'runs'): raise ValueError('Source must stay in repository runs')
    output=run/'early_verify_pass4'
    output.mkdir(exist_ok=True)
    rows=load_rows(run,ROOT/'.local/tokenizers/qwen3.json',4)
    replays=[simulate(rows,policy=p,generation_slots=slots,service_s=args.service_s)
             for slots in GENERATION_SLOTS for p in ['final','markers','permissive']]
    sensitivity=[simulate(rows,policy='permissive',generation_slots=slots,service_s=args.service_s,
                          exponent=exponent,first_token_s=ttft)
                 for slots in GENERATION_SLOTS for exponent in [1,1.5,2] for ttft in [0,5,15]]
    summary={'run':args.run,'samples':[1,2,3,4], 'traces':len(rows), 'questions':30,
             'final_pass4':sum(any(r['final_correct'] for r in rows if r['problem_idx']==q) for q in range(1,31)),
             'service_s':args.service_s, 'replays':replays,'timing_sensitivity':sensitivity,
             'assumptions':['Local exact key comparison stands in for a perfectly accurate verifier; no grader invoked.',
                 'One shared serial verifier, each check costs three seconds; wrong answers charged.',
                 'Candidate extraction is permissive and gold-blind, including toy/formatting examples.',
                 'One verification per distinct question-answer pair, across all four trajectories.',
                 'Primary arrival estimate: generation duration × completed-line token fraction.',
                 'Original nonstream responses have no measured intermediate timestamps.',
                 '120-slot replay normalizes all requests to start at t=0; not an observed run.',
                 '30- and 8-slot replays use FIFO sample rounds, reclaim solved generation slots instantly.',
                 'Saved successful-request durations stay fixed despite changed inference concurrency.',
                 'Only a correct completed verification stops or skips generation; negative checks preserve it.',
                 'No throughput, billing, resource-contention or actual cancellation measurement.']}
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'candidate_inventory.json',{'rows':rows})
    with (output/'milestones.csv').open('w',newline='') as f:
        fields=['policy','generation_slots','n_correct','time_s','problem_idx','sample_number','answer','kind','queue_s']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for r in replays:
            for m in r['milestones']:
                writer.writerow({**m,'policy':r['policy'],'generation_slots':r['generation_slots']})
    with (output/'per_question.csv').open('w',newline='') as f:
        fields=['problem_idx','final_pass4','permissive_recoverable']+[f'{p}_{slots}_verified_s' for slots in GENERATION_SLOTS for p in ['final','markers','permissive']]
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        milestones={(r['policy'],r['generation_slots']):{m['problem_idx']:m['time_s'] for m in r['milestones']} for r in replays}
        for q in range(1,31):
            writer.writerow({'problem_idx':q,'final_pass4':q in milestones['final',120],
                'permissive_recoverable':q in milestones['permissive',120],
                **{f'{p}_{slots}_verified_s':milestones[p,slots].get(q) for slots in GENERATION_SLOTS for p in ['final','markers','permissive']}})
    render(summary,output)
    print(json.dumps([{k:r[k] for k in ['policy','generation_slots','correct_questions','time_to_18_s','checks_completed','wrong_checks','checks_before_18','trajectories_started']} for r in replays],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='20260930-155212')
    p.add_argument('--service-s',type=float,default=3)
    args=p.parse_args()
    if args.service_s<=0: p.error('Positive service time required')
    main(args)
