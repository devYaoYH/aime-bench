"""Render saved CPU totals, actual grader timelines, and local replay profiles.

Requires matplotlib. Never launches inference or reconstructs missing CPU spans.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ('BF16 30×1', '20261003T211557.382358Z'),
    ('BF16 30×2', '20261003T211346.861294Z'),
    ('BF16 30×4', '20261003T211054.387701Z'),
    ('NVFP4 30×1 / 16K', '20261003T212812.156610Z'),
]
COLORS = ['#147d92', '#ed9e43', '#8862ad', '#93bcc1', '#dce2e8']


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def load_runs():
    result = []
    for label, attempt_id in RUNS:
        folder = ROOT / 'attempts' / attempt_id
        config = json.loads((folder / 'config.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        overhead = summary['overhead'];timings = overhead['timings']
        cpu = overhead['runner_process']['official_user_s'] + overhead['runner_process']['official_system_s']
        named = ['stream_trace_write_flush', 'sse_json_decode', 'candidate_parse_enqueue']
        totals = [timings[name]['thread_cpu_total_s'] for name in named]
        measured = sum(row['thread_cpu_total_s'] for row in timings.values())
        totals += [measured - sum(totals), cpu - measured]
        started = timestamp(config['official_started_at_utc'])
        solved = sorted(json.loads(line)['first_solved_elapsed_s'] for line in (folder / 'solved.jsonl').read_text().splitlines())
        queries = []
        for path in folder.glob('trace/*/verification.jsonl'):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if 'result' not in event:continue
                verdict = event['result']
                queries.append({'index': int(path.parent.name), 'correct': verdict['verdict'],
                                'candidate': timestamp(event['observed_at_utc'])-started,
                                'start': timestamp(verdict['picked_at'])-started,
                                'end': timestamp(verdict['answered_at'])-started})
        queries.sort(key=lambda q:q['start'])
        result.append({'label':label,'attempt_id':attempt_id,'cpu':cpu,'totals':totals,
                       'solved':solved,'queries':queries,'target_s':solved[17]})
    return result


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.labelcolor':'#34424f',
                         'xtick.color':'#5b6874','ytick.color':'#34424f',
                         'figure.facecolor':'white','axes.facecolor':'white',
                         'svg.fonttype':'none'})


def save(fig, output, name):
    for ext in ('png','svg','pdf'):
        fig.savefig(output/(name+'.'+ext),dpi=180,bbox_inches='tight',facecolor='white')
        if ext == 'svg':
            path = output/(name+'.svg')
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def render(output):
    output.mkdir(parents=True,exist_ok=True);style();runs=load_runs()
    fig,ax=plt.subplots(figsize=(11.5,4.9),layout='constrained')
    y=list(range(len(runs)));left=[0.0]*len(runs)
    labels=['Trace serialization + flush','JSON decode','Candidate parse','Other timed CPU','Outside timed sections']
    for k,(label,color) in enumerate(zip(labels,COLORS)):
        vals=[r['totals'][k] for r in runs]
        ax.barh(y,vals,left=left,color=color,label=label,height=.55,
                hatch='///' if k==4 else None,edgecolor='white',linewidth=.5)
        left=[a+b for a,b in zip(left,vals)]
    for n,r in enumerate(runs):ax.text(r['cpu']+1,n,f"{r['cpu']:.2f}",va='center',weight='bold')
    ax.set_yticks(y,[r['label'] for r in runs]);ax.invert_yaxis()
    ax.set_xlim(0,105);ax.set_xlabel('Official runner CPU-seconds (not elapsed time)')
    ax.set_title('Most runner CPU is outside the detailed timers',loc='left',pad=35)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.20),ncol=3,frameon=False,fontsize=9)
    ax.text(0,1.025,'The remainder includes HTTP/asyncio, profiling bookkeeping, and other unmeasured work.',transform=ax.transAxes,fontsize=10,color='#5b6874')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    save(fig,output,'cpu-breakdown')

    fig,(timeline,curve)=plt.subplots(2,1,figsize=(12,8),height_ratios=[1.2,1],layout='constrained')
    for n,r in enumerate(runs):
        for q in r['queries']:
            timeline.broken_barh([(q['start'],q['end']-q['start'])],(n-.16,.32),facecolors='#147d92' if q['correct'] else '#c75050')
            timeline.plot([q['candidate'],q['start']],[n+.27,n],color='#b4bac2',linewidth=.7)
            timeline.scatter(q['candidate'],n+.27,marker='v',s=20,color='#ed9e43',zorder=3)
        timeline.axvline(r['target_s'],ymin=(n-.4+0.5)/4,ymax=(n+.4+0.5)/4,color='#34424f',linewidth=1,linestyle=':')
        timeline.text(r['target_s']+1,n,f"{r['target_s']:.2f}s",va='center',fontsize=10,weight='bold')
        curve.step([0,*r['solved']],[0,*range(1,len(r['solved'])+1)],where='post',label=r['label'],color=COLORS[n],linewidth=2)
    timeline.set_ylim(-.5,3.5);timeline.set_yticks(range(4),[r['label'] for r in runs])
    timeline.set_xlim(0,128);timeline.set_xlabel('Seconds since official start')
    timeline.set_title('Recorded grader service and candidate arrivals',loc='left',pad=16)
    timeline.legend(handles=[Patch(color='#147d92',label='Correct check'),Patch(color='#c75050',label='Wrong check'),Line2D([],[],marker='v',color='#ed9e43',linestyle='',label='Candidate observed')],loc='upper left',bbox_to_anchor=(0,1.02),ncol=3,frameon=False,fontsize=9)
    timeline.grid(axis='x',alpha=.15);timeline.set_axisbelow(True)
    curve.set_title('Distinct questions first verified correct',loc='left',pad=14)
    curve.set_xlim(0,128);curve.set_ylim(0,19);curve.set_yticks([0,6,12,18]);curve.axhline(18,color='#a8afb8',linestyle='--',linewidth=1)
    curve.set_xlabel('Seconds since official start');curve.set_ylabel('Correct questions')
    curve.grid(alpha=.15);curve.legend(loc='lower right',frameon=False,fontsize=9)
    save(fig,output,'grader-timeline')

    audit=json.loads((ROOT/'runs/experiments/nvfp4-30x1-16k-20261003T212811Z/cpu_replay_audit.json').read_text())
    fig,(bench,profile)=plt.subplots(1,2,figsize=(12,5.6),layout='constrained',width_ratios=[1,1.2])
    variants=audit['results'][:4];names=['Current','Buffer flushes','Defer meter totals','Both changes']
    values=[r['median_cpu_s']*1000 for r in variants]
    bench.barh(range(4),values,color=[COLORS[4],COLORS[0],COLORS[2],COLORS[0]],height=.55)
    for n,value in enumerate(values):bench.text(value+1,n,f'{value:.1f} ms',va='center',fontsize=10)
    bench.set_yticks(range(4),names);bench.invert_yaxis();bench.set_xlim(0,112)
    bench.set_xlabel('Median CPU milliseconds, seven replays');bench.set_title('27.7% less CPU in local replay',loc='left',pad=16)
    rows=[]
    for line in audit['cprofile_current'].splitlines():
        m=re.match(r'^\s*\S+\s+(\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+(.+)$',line)
        if m:rows.append((float(m[1])*1000,m[2]))
    rows=rows[:8]
    def function_label(name):
        for key,label in [('(observe)','Meter.observe'),('(replay)','Replay driver'),("'flush'",'File flush'),('(iterencode)','JSON encode'),('(measure)','Meter.measure'),('(inc)','Meter.inc'),('(feed)','CandidateDetector.feed'),('(raw_decode)','JSON decode')]:
            if key in name:return label
        return name.split('/')[-1]
    profile.barh(range(len(rows)),[r[0] for r in rows],color=COLORS[2],height=.55)
    profile.set_yticks(range(len(rows)),[function_label(r[1]) for r in rows]);profile.invert_yaxis()
    profile.set_xlabel('Self CPU milliseconds in cProfile replay')
    profile.set_title('Profiling its own bookkeeping',loc='left',pad=16)
    for ax in (bench,profile):ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('One saved Q09 stream · 5,987 SSE events · local macOS, no network/GPU',fontsize=11,color='#5b6874')
    save(fig,output,'replay-profile')
    (output/'analysis.json').write_text(json.dumps({'runs':runs,'source_replay_audit':'runs/experiments/nvfp4-30x1-16k-20261003T212811Z/cpu_replay_audit.json'},indent=2)+'\n')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/profiling/20261003-cpu-review')
    render(parser.parse_args().output)
