"""Replay permissive candidate verification across the saved Python-enabled pass@2 profile.

Use this offline experiment to adapt 60 multi-round traces, estimate within-round
text delivery, and preserve measured CPU-tool timing. Compares final/marker/prose
candidates with a separate stdout extension using the shared perfect-verifier
simulator, 60/30/8-slot schedules, and observed-start shadow replays. Writes
summary, inventory, plots, and report.md under early_verify_pass2/. Requires local
raw profile records and cached Qwen3.5 tokenizer/provenance; makes no inference calls.
    python -m src.experiments.python_tools.backtest_python_early_verify
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re

from tokenizers import Tokenizer

from src.answer_extraction import ANSWER_LINE, BOX, classify_context, extract_events, numeric_prefix, target_rule
from src.verification_replay import simulate
from src.common import ROOT, atomic_json

POLICIES=('final','markers','permissive','permissive_tools')
LABELS={'final':'Completed final answers','markers':'Boxes / Answer: lines',
        'permissive':'Same permissive text policy','permissive_tools':'Permissive + Python totals'}
COLORS={'final':'#53657d','markers':'#e29b31','permissive':'#008b83','permissive_tools':'#9b54b0'}


def literal_markers(text):
    # Scan independently: the legacy permissive extractor can deduplicate an
    # Answer-line match in favor of a shorter overlapping prose match.
    events=[]
    for pattern,kind in [(BOX,'boxed'),(ANSWER_LINE,'answer_line')]:
        for match in pattern.finditer(text):
            end=match.start()+len(match.group().rstrip())
            events.append({'answer':int(match[1]),'start':match.start(),'end':end,'kind':kind,
                'confidence':classify_context(text,match.start(),kind=kind),'quote':text[match.start():end].strip(),
                'context':text[max(0,match.start()-100):min(len(text),end+100)],
                'marker_only':True,'tool_extension':False})
    return events


def stdout_values(text,problem):
    """Optional wider net, separate from the unchanged text extractor.

    Debug values, toy totals, and incorrect computations stay in the inventory.
    No key is available here. Only literal arithmetic is evaluated, never code.
    """
    events=[]
    offset=0
    rule=target_rule(problem)
    for line in text.splitlines(keepends=True):
        label=re.match(r'^\s*(?:[A-Za-z _()\-]*(?:answer|result|total|remainder)[A-Za-z _()\-]*\s*[:=]\s*)',line,re.I)
        start=label.end() if label else 0
        if label or re.match(r'^\s*[+\-]?\d',line):
            parsed=numeric_prefix(line,start)
            if parsed:
                value,end,expression=parsed
                if not line[end:].strip(' \t\r\n.$'):
                    values=[(value,'identity')]
                    if rule and rule[1]=='mod1000': values.append((value%1000,'mod1000'))
                    elif rule and rule[1]=='minus2025': values.append((value-2025,'minus2025'))
                    seen=set()
                    for answer,transform in values:
                        if 0<=answer<=999 and answer not in seen:
                            seen.add(answer)
                            events.append({'answer':answer,'start':offset,'end':offset+len(line),
                                'kind':'tool_result','confidence':'unverified_tool_value',
                                'expression':expression,'quote':line.strip(),'context':line.strip(),
                                'transform':transform,'tool_extension':True})
        offset+=len(line)
    return events


def event_fraction(text,event,ends,prefix,total,overhead):
    boundary=text.find('\n',event['end'])
    boundary=boundary+1 if boundary>=0 else len(text)
    # Preserve the original conservative rule: expose a completed line and put
    # unaccounted provider overhead before it. Normalize if visible tokens exceed usage.
    visible=prefix+min(len(ends),bisect_left(ends,boundary)+1)
    return min(1.0,(visible+max(0,overhead))/total)


def adapt_record(record,tokenizer,source_file):
    start=datetime.fromisoformat(record['started_at_utc'])
    elapsed=record['elapsed_s']
    candidates=[]
    completion_before=0
    token_audits=[]
    for index,rr in enumerate(record['rounds']):
        message=rr['response']['choices'][0]['message']
        generated=rr['response']['usage']['completion_tokens']
        parts=[('reasoning',message.get('reasoning') or ''),('content',message.get('content') or '')]
        encodings=[tokenizer.encode(text,add_special_tokens=False) for _,text in parts]
        # Function arguments are generated after reasoning/content. They consume
        # decode time even though we do not treat code literals as answer proposals.
        call_text=json.dumps(message.get('tool_calls') or [],ensure_ascii=False) if message.get('tool_calls') else ''
        call_tokens=len(tokenizer.encode(call_text,add_special_tokens=False).ids)
        visible=sum(len(e.ids) for e in encodings)+call_tokens
        denominator=max(generated,visible)
        overhead=generated-visible
        round_start=(datetime.fromisoformat(rr['started_at_utc'])-start).total_seconds()
        token_audits.append({'round_number':rr['number'],'provider_completion_tokens':generated,
                             'visible_tokens':visible,'provider_minus_visible':overhead})
        prefix=0
        for (part,text),encoded in zip(parts,encodings):
            ends=[end for begin,end in encoded.offsets]
            for event in [*extract_events(text,record['problem']),*literal_markers(text)]:
                fraction=event_fraction(text,event,ends,prefix,denominator,overhead)
                offset=min(elapsed,round_start+rr['latency_s']*fraction)
                candidates.append({**event,'round_number':rr['number'],'part':part,'source_file':source_file,
                    'tool_extension':False,'segment_start_s':round_start,'segment_duration_s':rr['latency_s'],
                    'segment_fraction':fraction,'offset_s':offset,'output_fraction':offset/elapsed,
                    'estimated_generated_tokens':completion_before+math.ceil(generated*fraction)})
            prefix+=len(encoded.ids)
        for execution in record['tool_executions']:
            if execution['round_number']!=rr['number']: continue
            # Next API round begins only after all preceding tool results arrive.
            # This is a measured conservative upper bound on their availability.
            available=((datetime.fromisoformat(record['rounds'][index+1]['started_at_utc'])-start).total_seconds()
                       if index+1<len(record['rounds']) else elapsed)
            stdout=execution['result'].get('stdout') or ''
            for event in [*extract_events(stdout,record['problem']),*literal_markers(stdout),*stdout_values(stdout,record['problem'])]:
                candidates.append({**event,'round_number':rr['number'],'part':'tool_stdout','source_file':source_file,
                    'tool_extension':event.get('tool_extension',False),'offset_s':min(elapsed,available),
                    'output_fraction':min(1,available/elapsed),
                    'estimated_generated_tokens':execution['generated_tokens_at_arrival'],
                    'tool_execution_ok':execution['result']['ok']})
        completion_before+=generated
    candidates.sort(key=lambda e:(e['offset_s'],e['end']))
    return {'problem_idx':record['problem_idx'],'sample_number':record['sample_idx'],
            'gold_answer':record['gold_answer'],'api_latency_s':elapsed,
            'final_candidate':int(record['candidate']) if record['status']=='complete' and record['candidate'] is not None else None,
            'final_correct':record['correct'],'source_file':source_file,
            'completion_tokens':completion_before,'started_at_utc':record['started_at_utc'],
            'candidates':candidates,'token_audits':token_audits}


def policy_rows(rows,policy):
    return [{**r,'candidates':[e for e in r['candidates']
        if (policy=='permissive_tools' or not e['tool_extension'])
        and (policy=='markers' or not e.get('marker_only'))]} for r in rows]


def fixed_starts(rows):
    origin=min(datetime.fromisoformat(r['started_at_utc']) for r in rows)
    result=deepcopy(rows)
    for row in result:
        delta=(datetime.fromisoformat(row['started_at_utc'])-origin).total_seconds()
        row['api_latency_s']+=delta
        for e in row['candidates']:
            e['offset_s']+=delta
            if 'segment_start_s' in e: e['segment_start_s']+=delta
            e['output_fraction']=e['offset_s']/row['api_latency_s']
    return result


def render(summary,output):
    os.environ['MPLCONFIGDIR']=str(output/'.mplconfig')
    os.environ['XDG_CACHE_HOME']=str(output/'.cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(10.5,12),layout='constrained')
    for ax,slots in zip(axes,(60,30,8)):
        selected=[r for r in summary['replays'] if r['generation_slots']==slots]
        end=max(r['milestones'][-1]['time_s']/60 for r in selected if r['milestones'])+.2
        for replay in selected:
            p=replay['policy']
            times=[0]+[m['time_s']/60 for m in replay['milestones']]+[end]
            counts=[0]+list(range(1,len(replay['milestones'])+1))+[replay['correct_questions']]
            ax.step(times,counts,where='post',label=LABELS[p],color=COLORS[p],lw=2)
            if replay['time_to_18_s'] is not None:
                ax.plot(replay['time_to_18_s']/60,18,'o',color=COLORS[p],ms=4)
        final=next(r for r in selected if r['policy']=='final')
        early=next(r for r in selected if r['policy']=='permissive')
        tools=next(r for r in selected if r['policy']=='permissive_tools')
        text=f'18 correct: final {final["time_to_18_s"]/60:.2f} min; permissive {early["time_to_18_s"]/60:.2f}; + tools {tools["time_to_18_s"]/60:.2f}'
        ax.set(title='All 60 attempts start together' if slots==60 else f'{slots} trajectory slots; FIFO sample rounds',
               xlabel='Estimated elapsed time (minutes)',ylabel='Questions verified correct',ylim=(0,31))
        ax.axhline(18,color='#999999',ls='--',lw=.8)
        ax.text(.98,.06,text,transform=ax.transAxes,ha='right',fontsize=9,
                bbox={'facecolor':'white','alpha':.95,'edgecolor':'#dddddd'})
        ax.grid(alpha=.18)
    axes[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Qwen3.5 + Python pass@2: permissive early verification',fontsize=15)
    for suffix in ('png','svg','pdf'): fig.savefig(output/f'time_to_correct.{suffix}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    selected=[r for r in summary['replays'] if r['generation_slots']==8]
    end=max(r['milestones'][-1]['time_s']/60 for r in selected)+.2
    for replay in selected:
        p=replay['policy']
        times=[0]+[m['time_s']/60 for m in replay['milestones']]+[end]
        counts=[0]+list(range(1,len(replay['milestones'])+1))+[replay['correct_questions']]
        ax.step(times,counts,where='post',label=LABELS[p],color=COLORS[p],lw=2)
        ax.plot(replay['time_to_18_s']/60,18,'o',color=COLORS[p],ms=4)
    ax.axhline(18,color='#999999',ls='--',lw=.8)
    ax.set(title='Python pass@2 · 8 trajectory slots · 3 seconds per verification',
           xlabel='Estimated elapsed time (minutes)',ylabel='Questions verified correct',ylim=(0,31))
    ax.legend(loc='upper left',fontsize=8)
    ax.grid(alpha=.18)
    for suffix in ('png','svg'): fig.savefig(output/f'eight_slots.{suffix}',dpi=180)
    plt.close(fig)


def main(args):
    run=(ROOT/args.run).resolve()
    if not run.is_relative_to(ROOT/'runs'): raise ValueError('Source must stay within repository runs')
    output=run/'early_verify_pass2'
    output.mkdir(exist_ok=True)
    tokenizer=Tokenizer.from_file(str(ROOT/'.local/tokenizers/qwen35.json'))
    records=[(p,json.loads(p.read_text())) for p in sorted(run.glob('*-sample-*.json'))]
    if {(r['problem_idx'],r['sample_idx']) for p,r in records}!={(q,s) for q in range(1,31) for s in (1,2)}:
        raise ValueError('Expected all 60 fixed pass@2 traces')
    rows=[adapt_record(r,tokenizer,str(p.relative_to(ROOT))) for p,r in records]
    replays=[simulate(policy_rows(rows,p),policy=p,generation_slots=slots,service_s=3)
             for slots in (60,30,8) for p in POLICIES]
    shadow=[simulate(fixed_starts(policy_rows(rows,p)),policy=p,generation_slots=60,service_s=3,stop_on_verified=False)
            for p in POLICIES]
    sensitivity=[simulate(policy_rows(rows,p),policy=p,generation_slots=8,service_s=3,
                          exponent=exponent,first_token_s=delay)
                 for p in ('permissive','permissive_tools') for exponent in (1,1.5,2) for delay in (0,5,15)]
    final_questions={r['problem_idx'] for r in rows if r['final_correct']}
    coverage={p:sorted({r['problem_idx'] for r in policy_rows(rows,p)
                       if r['final_correct'] or any(e['answer']==r['gold_answer'] and
                           (p!='markers' or e['kind'] in ('boxed','answer_line')) for e in r['candidates'])})
              for p in ('markers','permissive','permissive_tools')}
    summary={'run':args.run,'traces':60,'questions':30,'samples':[1,2],
             'final_correct_questions':sorted(final_questions),'coverage':coverage,
             'recovered_questions':{p:sorted(set(qs)-final_questions) for p,qs in coverage.items()},
             'tokenizer':json.loads((ROOT/'.local/tokenizers/qwen35-source.json').read_text()),
             'replays':replays,'observed_start_shadow':shadow,'timing_sensitivity_8_slots':sensitivity,
             'assumptions':['No model requests, remote grader, SSH, or local inference used.',
                'Original permissive extract_events unchanged; all toy/formatting/negated proposals retained.',
                'Marker comparison scans literal boxes/Answer lines independently so overlapping prose deduplication cannot hide them; no change to permissive inventory.',
                'Python-total/bare-number extension reported separately; only literal arithmetic and explicitly requested transforms.',
                'Tool code literals are excluded from extraction; actual printed stdout is scanned.',
                'Intermediate text delivery estimated per API round from completed-line Qwen3.5 token fractions; no original SSE timestamps.',
                'Tool outputs available at the next API-round start, a measured conservative upper bound.',
                'One shared FIFO verifier, 3s per distinct question-answer pair; wrong candidates charged; saved keys are a perfect-verifier stand-in.',
                'Negative checks preserve inference. Only a correct completed check stops/skips a question.',
                'FIFO capacity replays reuse fixed saved full-trajectory durations; actual throughput under changed scheduling is unknown.',
                'Observed-start shadow retains measured start times and every original attempt, without rescheduling after positive checks.',
                'No actual cancellation or billing savings measured.']}
    atomic_json(output/'summary.json',summary)
    # Correctness annotations are added only after extraction and scheduling finish.
    inventory=deepcopy(rows)
    for row in inventory:
        for event in row['candidates']: event['matches_saved_key']=event['answer']==row['gold_answer']
    atomic_json(output/'candidate_inventory.json',{'rows':inventory})
    render(summary,output)
    lines=['# Permissive early extraction: Qwen3.5 + Python pass@2','',
           f'60 fixed traces, two per question. Strict final answers cover **{len(final_questions)}/30**.',
           '', '| Policy | Recoverable questions | Newly recovered |', '|---|---:|---|']
    for p,qs in coverage.items(): lines.append(f'| {LABELS[p]} | {len(qs)} | {summary["recovered_questions"][p]} |')
    lines+=['','## Time to 18 verified correct questions','',
            '| Capacity | Final only | Boxes / Answer: | Same permissive | + Python totals |', '|---|---:|---:|---:|---:|']
    for slots in (60,30,8):
        chosen={r['policy']:r for r in replays if r['generation_slots']==slots}
        lines.append(f'| {slots} | '+' | '.join(f'{chosen[p]["time_to_18_s"]/60:.2f} min' for p in POLICIES)+' |')
    lines+=['','![Step graph](time_to_correct.png)','',
            'All policies use the same three-second verification queue and retire a question only '
            'after a correct completed check. Every wrong candidate costs three seconds. '
            'Repeated (question, integer) pairs are checked once. No candidates are selected by correctness.',
            '', '## Fixed observed starts, without generation rescheduling','',
            '| Policy | Time to 18 | Checks before 18 | Wrong checks overall |', '|---|---:|---:|---:|']
    for r in shadow: lines.append(f'| {LABELS[r["policy"]]} | {r["time_to_18_s"]/60:.2f} min | {r["checks_before_18"]} | {r["wrong_checks"]} |')
    lines+=['','## Eight-slot verification load','',
            '| Policy | Correct questions | Checks before 18 | Wrong checks overall | Peak pending checks |', '|---|---:|---:|---:|---:|']
    for r in replays:
        if r['generation_slots']==8:
            lines.append(f'| {LABELS[r["policy"]]} | {r["correct_questions"]} | {r["checks_before_18"]} | {r["wrong_checks"]} | {r["peak_pending_checks"]} |')
    eight={r['policy']:r for r in replays if r['generation_slots']==8}
    early=eight['permissive']
    before=[c for c in early['checks'] if c['verified_at_s']<=early['time_to_18_s']]
    lines+=['',f'The same permissive policy reduces time to 18 by '
            f'**{1-early["time_to_18_s"]/eight["final"]["time_to_18_s"]:.1%}** in the eight-slot replay. '
            f'It charges {len(before)} checks before the threshold: '
            f'{sum(c["correct"] for c in before)} positive and {sum(not c["correct"] for c in before)} negative.',
            '', '## Recovered source claims','',
            '| Question | Sample | Kind | Earlier proposed answer | Approx. seconds into that trace |',
            '|---|---:|---|---|---:|']
    for q in summary['recovered_questions']['permissive']:
        candidates=[(r,e) for r in policy_rows(rows,'permissive') if r['problem_idx']==q
                    for e in r['candidates'] if e['answer']==r['gold_answer']]
        r,e=min(candidates,key=lambda pair:(pair[0]['sample_number'],pair[1]['offset_s']))
        lines.append(f'| {q} | {r["sample_number"]} | {e["kind"]} | {e["quote"].replace("|","/")} | {e["offset_s"]:.2f} |')
    lines+=['','## Wrong checks and false-positive audit','',
            'The wide net retains hypothetical examples and intermediate quantities. '
            'For instance, Q5 discusses a hypothetical N=2200, yielding a proposed difference of 175; '
            'Q26 says the answer is 1 for the diameter-only subcase. Neither is the requested final '
            'answer. Both consume a negative check and leave inference running. Q24 also contributes '
            'an incorrect 150 and a toy/intermediate 20. No approval is inferred from a claim or '
            'successful Python execution.',
            '', '| Question | Proposed integer | Source quote | Charged result |', '|---|---:|---|---|']
    for c in early['checks']:
        if not c['correct']:
            lines.append(f'| {c["problem_idx"]} | {c["answer"]} | {c["quote"].replace("|","/")} | negative, 3 s |')
    q26=next(r for r in rows if r['problem_idx']==26 and r['sample_number']==2)
    text113=min(e['offset_s'] for e in q26['candidates'] if e['answer']==113 and not e['tool_extension'] and not e.get('marker_only'))
    tool113=min(e['offset_s'] for e in q26['candidates'] if e['answer']==113 and e['tool_extension'])
    lines+=['','## Python-output extension and timing sensitivity','',
            f'Q26 sample 2 has Total: 113 in Python output available by **{tool113:.2f} s**, '
            f'after 11,984 generated tokens. Its later reasoning contains Answer: 113. around '
            f'**{text113:.2f} s**, after approximately 15,225 tokens. Thus both policies recover Q26; '
            f'tool-output checking can expose it about **{text113-tool113:.2f} s earlier** within that trace. '
            'Standalone Answer-line detection also recovers Q26, so literal marker coverage is 20/30.',
            '', 'The marker comparison is independently scanned because the original permissive '
            'extractor deduplicates some overlapping Answer-line matches into prose. This correction '
            'does not change the original permissive candidates or their replay results.',
            '', 'Sensitivity changes only text delivery inside each API round; it does not move a '
            'Python result before execution. This varies the illustrative decode exponent across '
            '1, 1.5, 2 and first-token delay across 0, 5, 15 seconds (clipped for short rounds).']
    for p in ('permissive','permissive_tools'):
        times=[r['time_to_18_s']/60 for r in sensitivity if r['policy']==p]
        lines.append(f'- {LABELS[p]}: **{min(times):.2f}–{max(times):.2f} min** to 18; final-only stays {eight["final"]["time_to_18_s"]/60:.2f} min.')
    lines+=['','## Assumptions and reproducibility','']+[f'- {s}' for s in summary['assumptions']]
    lines+=['','```bash','.venv/bin/python -m unittest test.test_python_early_verify test.test_backtest_early_verify',
            '.venv/bin/python -m src.experiments.python_tools.backtest_python_early_verify','```','',
            'Raw candidates, source quotes, per-round offsets and token audits are in '
            '`candidate_inventory.json`. All charged checks, milestones, generation starts and '
            'timing-sensitivity results are in `summary.json`. Original traces remain unchanged.','']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'coverage':coverage,'recovered':summary['recovered_questions'],
         'eight_slots':[{k:r[k] for k in ('policy','correct_questions','time_to_18_s','checks_before_18','wrong_checks','peak_pending_checks')}
                        for r in replays if r['generation_slots']==8]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/python-tool-qwen35-pass2-auto')
    main(p.parse_args())
