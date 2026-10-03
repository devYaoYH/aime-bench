"""Render token, latency, tool-error, and pass@2 statistics from the saved Python profile.

Use this offline renderer after all 60 optional-tool attempts are saved. Reads
raw multi-round records and summary.json; writes analysis.json, per-question and
time-to-correct plots, and report.md beside the source run. Correct-final-response
arrival times are measured and exclude the separate verifier replay. No API or
worker execution occurs.
    python -m src.experiments.python_tools.report_python_tool_profile
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import statistics

from src.common import ROOT, atomic_json


def median(values):
    return statistics.median(values) if values else None


def render(output):
    output=output.resolve()
    if not output.is_relative_to(ROOT/'runs'): raise ValueError('Results must stay in repository runs')
    summary=json.loads((output/'summary.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted(output.glob('*-sample-*.json'))]
    records=[r for r in records if r['status']!='running']
    if not records: raise ValueError('No completed attempt records yet')
    os.environ['MPLCONFIGDIR']=str(output/'.mplconfig')
    os.environ['XDG_CACHE_HOME']=str(output/'.cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

    statuses=Counter(r['status'] for r in records)
    callers=[r for r in records if r['tool_executions']]
    noncallers=[r for r in records if not r['tool_executions']]
    errors=Counter(e['result'].get('error') for r in records for e in r['tool_executions'] if not e['result']['ok'])
    # Use UTC timestamps for relative duration, not for user-facing calendar times.
    beginning=min(datetime.fromisoformat(r['started_at_utc']) for r in records)
    arrivals={}
    for r in records:
        if r['correct']:
            t=(datetime.fromisoformat(r['finished_at_utc'])-beginning).total_seconds()
            q=r['problem_idx']
            arrivals[q]=min(t,arrivals.get(q,t))
    events=sorted(arrivals.values())
    elapsed=max((datetime.fromisoformat(r['finished_at_utc'])-beginning).total_seconds() for r in records)
    first_correct={r['problem_idx'] for r in records if r['sample_idx']==1 and r['correct']}
    second_correct={r['problem_idx'] for r in records if r['sample_idx']==2 and r['correct']}
    both_capped=[q for q in range(1,31) if len(pair:=[r for r in records if r['problem_idx']==q])==2
                 and all(r['status']=='generation_budget_exhausted' for r in pair)]
    metrics={'status_counts':dict(statuses),'first_sample_correct':sum(r['correct'] for r in records if r['sample_idx']==1),
             'second_sample_correct':len(second_correct),'second_sample_rescued_questions':sorted(second_correct-first_correct),
             'both_samples_capped_questions':both_capped,
             'tool_using_questions':sorted({r['problem_idx'] for r in callers}),
             'sum_python_execution_s':sum(e['result'].get('execution_s',0) for r in records for e in r['tool_executions']),
             'tool_error_counts':dict(errors),'time_to_18_correct_final_answers_s':events[17] if len(events)>=18 else None,
             'median_case_elapsed_s':median([r['elapsed_s'] for r in records]),
             'median_generated_tokens':median([r['accounting']['completion_tokens'] for r in records if r['accounting']['completion_tokens'] is not None]),
             'median_first_tool_arrival_s':median([r['first_tool']['arrival_elapsed_s'] for r in callers]),
             'groups':{name:{'attempts':len(rows),'correct':sum(r['correct'] for r in rows),
                            'median_output_tokens':median([r['accounting']['completion_tokens'] for r in rows if r['accounting']['completion_tokens'] is not None]),
                            'median_elapsed_s':median([r['elapsed_s'] for r in rows])}
                       for name,rows in [('used_tools',callers),('no_tool_call',noncallers)]}}
    atomic_json(output/'analysis.json',metrics)

    colors={'correct':'#13866c','wrong':'#d78b28','capped':'#b0b6bf','error':'#ba4256'}
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
    for sample in (1,2):
        rows=[r for r in records if r['sample_idx']==sample]
        x=[r['problem_idx']+(-.19 if sample==1 else .19) for r in rows]
        c=[colors['correct'] if r['correct'] else colors['capped'] if r['status']=='generation_budget_exhausted'
           else colors['error'] if r['status']=='error' else colors['wrong'] for r in rows]
        axes[0].bar(x,[(r['accounting']['completion_tokens'] or 0)/1000 for r in rows],width=.35,color=c)
        axes[1].bar(x,[r['elapsed_s'] for r in rows],width=.35,color=c)
        for axis,name in [(axes[0],'completion_tokens'),(axes[1],None)]:
            for px,r in zip(x,rows):
                if r['tool_executions']:
                    y=(r['accounting']['completion_tokens'] or 0)/1000 if name else r['elapsed_s']
                    axis.plot(px,y,'ko',markersize=3)
    axes[0].axhline(16.384,color='#777777',ls='--',lw=.8)
    axes[0].set_ylim(0,18.5)
    axes[0].set_ylabel('Generated tokens (thousands)')
    axes[1].set_ylabel('Case elapsed (seconds)')
    axes[1].set_xlabel('Question; sample 1 left, sample 2 right; dots indicate a Python call')
    axes[1].set_xticks(range(1,31))
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=v,label=k.title()) for k,v in colors.items()],ncols=4,loc='upper left')
    for a in axes: a.grid(axis='y',alpha=.2); a.set_axisbelow(True)
    fig.suptitle('Qwen3.5-35B-A3B · optional Python · thinking enabled · pass@2')
    fig.savefig(output/'per_question_profile.png',dpi=180)
    fig.savefig(output/'per_question_profile.svg')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    xs=[0]+[t/60 for t in events]+[elapsed/60]
    ys=[0]+list(range(1,len(events)+1))+[len(events)]
    ax.step(xs,ys,where='post',color='#13866c',lw=2)
    ax.axhline(18,color='#777777',ls='--',lw=.8)
    if len(events)>=18:
        ax.plot(events[17]/60,18,'o',color='#13866c')
        ax.annotate(f'18 correct at {events[17]/60:.2f} min',(events[17]/60,18),xytext=(0,14),
                    textcoords='offset points',ha='center')
    ax.set(xlabel='Minutes since first attempt started',ylabel='Distinct questions with a correct final answer',
           ylim=(0,31),title='Observed hosted run · 8 active trajectories · local key comparison')
    ax.grid(alpha=.2)
    fig.savefig(output/'time_to_correct.png',dpi=180)
    fig.savefig(output/'time_to_correct.svg')
    plt.close(fig)

    lines=['# Optional Python, thinking enabled: all 30 questions, pass@2','',
           f'Observed {len(records)}/60 terminal attempts. Empirical pass@2: **{len(arrivals)}/30**; '
           f'correct attempts: **{sum(r["correct"] for r in records)}/{len(records)}**.',
           '',f'Python used by **{len(callers)} attempts**; '
           f'**{summary["successful_tool_calls"]}/{summary["tool_calls"]}** tool calls succeeded.',
           '',f'Profile wall time: {summary["profile_wall_s"]:.1f} s. '
           f'Generated tokens: {summary["total_completion_tokens"]}; input tokens: {summary["total_prompt_tokens"]}. '
           f'Reported cost: ${summary["reported_cost"]:.6f}.',
           '',f'Model API rounds across all attempts: {sum(r["accounting"]["api_rounds"] for r in records)}.',
           '',f'Status counts: {dict(statuses)}.',
           '',f'Median tokens through first tool call: {summary["median_generated_tokens_before_first_tool"]}; '
           f'median first-call arrival: {metrics["median_first_tool_arrival_s"]} s.',
           '',f'Total isolated Python wall time: {summary["sum_tool_wall_s"]:.3f} s.',
           '',f'Native Python execution time: {metrics["sum_python_execution_s"]:.4f} s. '
           f'First-sample correct questions: {len(first_correct)}; second-sample correct questions: {len(second_correct)}. '
           f'Questions rescued by sample 2: {metrics["second_sample_rescued_questions"]}.',
           '',f'Questions capped in both samples: {both_capped}. '
           f'Questions with any Python call: {metrics["tool_using_questions"]}.',
           '', '![Per-question tokens and latency](per_question_profile.png)',
           '', '![Time to correct final answers](time_to_correct.png)',
           '', 'The step graph uses actual final-response arrival times and local exact-key comparisons. '
           'It does **not** include the hypothetical three-second verifier from the earlier back-test.',
           '', 'Thinking is enabled and tool_choice is auto on every round. There is no forced first call '
           'or separate reasoning-token budget. The existing 16,384 cumulative output ceiling remains, '
           'with broad 16-round/16-tool guards and the existing restricted math worker.',
           '', 'Eight hosted trajectories can be active, with two concurrent CPU workers. This is not a '
           'local A100 measurement. Both samples run even when the first is correct.',
           '', '## Per-question outcomes',
           '', '| Question | Sample 1: answer / outcome | Sample 2: answer / outcome | Python calls (1 / 2) | Pass@2 |',
           '|---|---|---|---|---|']
    for q in range(1,31):
        pair={r['sample_idx']:r for r in records if r['problem_idx']==q}
        def label(s):
            r=pair.get(s)
            if not r: return 'pending'
            return f'{r["candidate"] or "—"} / '+('correct' if r['correct'] else 'wrong' if r['status']=='complete' else r['status'])
        lines.append(f'| {q} | {label(1)} | {label(2)} | '
                     f'{len(pair[1]["tool_executions"]) if 1 in pair else "—"} / '
                     f'{len(pair[2]["tool_executions"]) if 2 in pair else "—"} | '
                     f'{"yes" if q in arrivals else "no"} |')
    lines+=['','## Interpretation limits','',
            'This profiles optional tool use; it has no matched no-tool arm across all 30 questions. '
            'Questions where the model elects to use Python differ from questions where it does not, '
            'so group medians are descriptive and cannot establish a causal speedup. The earlier '
            'historical pass@8 used a different Qwen model. Two samples per question provide limited '
            'precision and no conclusion about the best reasoning budget.',
            '', 'Capped, errored, and unfinished answers are excluded from successful solves. '
            'Any matching extracted integer in a capped response is retained as answer_correct '
            'in its raw record, separately from the strict correct flag.',
            '', 'No SSH, provided grader, local llama inference, or answer key in model inputs.',
            '', '## Tool errors','',f'```json\n{json.dumps(dict(errors),indent=2)}\n```','']
    q26=next((r for r in records if r['problem_idx']==26 and r['sample_idx']==2),None)
    if q26 and len(q26['tool_executions'])==2 and q26['status']=='generation_budget_exhausted':
        first,second=q26['tool_executions']
        if 'Total: 113' in first['result']['stdout'] and second['result']['stdout'].strip()=='113':
            lines+=['## Q26: correct tool output, no final answer','',
                    'Q26 asks for equal-length perfect matchings on the vertices of a regular 24-gon. '
                    'Sample 2 generated a Python program that computed the sum of the matching counts '
                    'for each chord length and printed **Total: 113**, matching the saved key. A second '
                    'Python call summed the counts again and printed **113**. The model then used its '
                    'remaining tokens on reasoning and never emitted a final answer.',
                    '',f'The first result arrived after {first["generated_tokens_at_arrival"]:,} generated tokens '
                    f'and approximately {first["arrival_elapsed_s"]+first["result"]["wall_s"]:.2f} seconds. '
                    f'The second call arrived after {second["generated_tokens_at_arrival"]:,} generated tokens. '
                    'The complete attempt exhausted 16,384 tokens.',
                    '', 'This is a concrete post-hoc recovery opportunity, **excluded from strict pass@2**. '
                    'Accepting arbitrary tool stdout as a final answer would need a separate extraction '
                    'and verification policy; a successful execution alone does not prove a computation '
                    'answers the requested question correctly.',
                    '', '[Raw Q26 trajectory](26-sample-2.json)','']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/python-tool-qwen35-pass2-auto')
    args=p.parse_args()
    render(ROOT/args.out)
