"""Reconstruct observed decode throughput from prior official vLLM counters.

Post-hoc analysis only; no model or grader requests. Counters exclude warmup by
using differences between official samples after all initial TTFTs have passed.
"""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ATTEMPTS=('20261003T211557.382358Z','20261003T211346.861294Z','20261003T211054.387701Z')
BINS=((1,8),(9,16),(17,32),(33,64),(65,96),(97,128))

def metric(row,name):
    values=[m['value'] for m in row['metrics'] if m['name']==name]
    return sum(values) if values else None

def bucket(value):
    return next(((lo,hi) for lo,hi in BINS if lo<=value<=hi),None)

def analyze(folder):
    config=json.loads((folder/'config.json').read_text())
    initial=[json.loads(p.read_text()) for p in folder.glob('trace/*/rollout-01/telemetry.json')]
    after_prefill=max(t['start_monotonic_s']+t['ttft_s'] for t in initial if t['ttft_s'] is not None)
    rows=[json.loads(line) for line in (folder/'inference_metrics.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['phase']=='official']
    windows=[]
    for prev,cur in zip(rows,rows[1:]):
        if prev['monotonic_s']<after_prefill:continue
        dt=cur['monotonic_s']-prev['monotonic_s']
        n0,n1=(metric(r,'vllm:num_requests_running') for r in (prev,cur))
        g0,g1=(metric(r,'vllm:generation_tokens_total') for r in (prev,cur))
        w0,w1=(metric(r,'vllm:num_requests_waiting') for r in (prev,cur))
        if None in (n0,n1,g0,g1,w0,w1) or dt<=0 or g1<g0 or min(n0,n1)<=0 or max(w0,w1)>0:continue
        # Exclude transitions across bins; running values only observed at endpoints.
        if bucket(n0)!=bucket(n1):continue
        avg=(n0+n1)/2
        windows.append({'elapsed_monotonic_start':prev['monotonic_s'],'duration_s':dt,
                        'running_start':n0,'running_end':n1,'mean_running':avg,
                        'bucket':list(bucket(avg)),'generation_tokens':g1-g0,
                        'aggregate_tokens_s':(g1-g0)/dt,
                        'per_active_request_tokens_s':(g1-g0)/(dt*avg)})
    tokens=sum(x['generation_tokens'] for x in windows);seconds=sum(x['duration_s'] for x in windows)
    active_seconds=sum(x['duration_s']*x['mean_running'] for x in windows)
    return {'attempt_id':folder.name,'model':config['model'],'initial_concurrency':config['parallelism']*config['rollouts'],
            'windows':windows,'included_seconds':seconds,'included_tokens':tokens,
            'weighted_aggregate_tokens_s':tokens/seconds if seconds else None,
            'weighted_per_active_request_tokens_s':tokens/active_seconds if active_seconds else None}

def main():
    out=ROOT/'runs/profiling/20261003-decode-throughput';out.mkdir(parents=True,exist_ok=True)
    cells=[analyze(ROOT/'attempts'/id) for id in ATTEMPTS]
    pooled=[]
    for lo,hi in BINS:
        windows=[x for c in cells for x in c['windows'] if x['bucket']==[lo,hi]]
        if not windows:continue
        tokens=sum(x['generation_tokens'] for x in windows);seconds=sum(x['duration_s'] for x in windows)
        running_seconds=sum(x['duration_s']*x['mean_running'] for x in windows)
        pooled.append({'concurrency_bin':[lo,hi],'windows':len(windows),'observed_seconds':seconds,
                       'mean_running':running_seconds/seconds,'aggregate_tokens_s':tokens/seconds,
                       'per_active_request_tokens_s':tokens/running_seconds})
    result={'schema_version':1,'cells':cells,'pooled_concurrency_bins':pooled,
            'method':'Official cumulative generation counter differences after initial TTFT; positive running endpoints; zero waiting at both endpoints; same running-count bin; duration-weighted rates. Per-active rate uses trapezoidal request-seconds.',
            'limits':'Observational intervals from three BF16/95% trials; contexts and active question mix change with time. Not a controlled fixed-context concurrency benchmark or client arrival-rate measurement. No rates inferred for benchmark-mode attempts without engine samples.'}
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    colors=('#2379ac','#e08b25','#7955a3')
    for c,color in zip(cells,colors):
        for ax,key in zip(axes,('aggregate_tokens_s','per_active_request_tokens_s')):
            ax.scatter([x['mean_running'] for x in c['windows']],[x[key] for x in c['windows']],s=13,alpha=.4,color=color,label=f"30×{c['initial_concurrency']//30}")
    for ax,key in zip(axes,('aggregate_tokens_s','per_active_request_tokens_s')):
        ax.plot([p['mean_running'] for p in pooled],[p[key] for p in pooled],'o-',color='#172731',lw=2,label='Pooled weighted mean')
        ax.set_xlabel('Observed concurrent running requests');ax.grid(alpha=.2);ax.set_xlim(0,125);ax.set_ylim(bottom=0)
    axes[0].set_title('Aggregate decode throughput');axes[0].set_ylabel('Generated tokens / second');axes[0].legend(fontsize=8)
    axes[1].set_title('Decode throughput per active request');axes[1].set_ylabel('Tokens / second / active request')
    fig.suptitle('Prior BF16 sweeps · A100 · 95% VRAM envelope',fontsize=13)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'throughput.{ext}',dpi=170)
    md='''# Observed decoding throughput versus concurrency

![Decode throughput](throughput.png)

Rates below come from prior BF16/95% sweep engine counters, using only official
samples after every initial request's first token, with no waiting requests at
both endpoints. Warmup is excluded. Windows crossing a concurrency bin are
excluded. Aggregate throughput is generated token-counter growth / elapsed time;
per-active throughput divides by approximate request-seconds using endpoint counts.

| Observed concurrency range | Mean active | Aggregate tok/s | Tok/s per active request | Seconds observed |
| --- | ---: | ---: | ---: | ---: |
'''
    for p in pooled:
        lo,hi=p['concurrency_bin'];md+=f"| {lo}–{hi} | {p['mean_running']:.1f} | {p['aggregate_tokens_s']:.0f} | {p['per_active_request_tokens_s']:.1f} | {p['observed_seconds']:.1f} |\n"
    md+='''
| Initial configuration | Included window tok/s | Tok/s per active request | Seconds included |
| --- | ---: | ---: | ---: |
'''
    for c in cells:md+=f"| 30×{c['initial_concurrency']//30} | {c['weighted_aggregate_tokens_s']:.0f} | {c['weighted_per_active_request_tokens_s']:.1f} | {c['included_seconds']:.1f} |\n"
    md+='''
Increasing fan-out can raise aggregate GPU token output while reducing each
trajectory's decoding speed. A speedrun depends on when useful candidate answers
reach the serial grader; aggregate tok/s alone does not predict time to 18.
These are observational measurements: context lengths, active question mix and
cancellations change over each run. They do not establish a fixed-context scaling
curve or a causal concurrency effect. The per-active rate is an approximation
because concurrency is sampled once per interval. Benchmark-mode runs deliberately
omit these engine polls; no throughput counter data is inferred for them.

[Window-level data and source attempts](analysis.json). PNG, SVG and PDF exports
are adjacent. Reproduce with `python scripts/report_decode_throughput.py`.
'''
    (out/'README.md').write_text(md)
    print(json.dumps(pooled,indent=2))

if __name__=='__main__':main()
