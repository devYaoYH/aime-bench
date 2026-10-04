"""Audit and plot this batch using versioned client records; run from repo root."""
from pathlib import Path
import json, statistics, hashlib, collections, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/aime-validation-matplotlib')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='frozen-core-prompt-five-seeds-20261004T005416Z'
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3] if '/runs/experiments/' in str(Path(__file__).resolve()) else Path.cwd()
BATCH=ROOT/'runs/experiments/frozen-core-prompt-five-seeds-20261004T005416Z'
CORE='35d6a06315a0e45441b3ac49bd468a2b94553fb172f378bfebc7ea8cc044070f'
PROMPT='26b591c39bcf55f4c94f5359dcc90d3c5626a524478eee1c5ce44b5038162364'
rows=[]
summary=json.loads((BATCH/'summary.json').read_text())
config=json.loads((BATCH/'config.json').read_text())
assert summary['status']=='complete' and [r['sampling_seed'] for r in summary['trials']]==config['seed_sequence']
assert config['core_manifest_sha256']==CORE and config['system_prompt_sha256']==PROMPT
for trial in summary['trials']:
 p=ROOT/'attempts'/trial['attempt_id']
 c=json.loads((p/'config.json').read_text()); s=json.loads((p/'summary.json').read_text())
 qs=[json.loads(q.read_text()) for q in sorted(p.glob('trace/*/question.json'))]
 solved=sorted([q for q in qs if q.get('first_solved')],key=lambda q:q['first_solved']['first_solved_elapsed_s'])
 assert len(qs)==30 and len(solved)==18 and len({q['problem_idx'] for q in solved})==18
 assert all(q['winner']['result']['verdict'] is True for q in solved)
 assert all(len(q['rollouts'])<=4 for q in qs)
 assert all(q['first_solved']['grader_query_id']==q['winner']['result']['query_id'] and q['first_solved']['grader_answered_at_utc']==q['winner']['result']['answered_at'] for q in solved)
 assert abs(solved[-1]['first_solved']['first_solved_elapsed_s']-s['time_to_target_s'])<0.01
 assert c['core_manifest_sha256']==CORE and c['system_prompt_sha256']==PROMPT and not c['git_dirty']
 assert c['skip_benchmark_prewarm'] and c['benchmark'] and s.get('benchmark_prewarm') is None
 assert c['parallelism']==30 and c['rollouts']==1 and c['first_pass_max_tokens']==8192 and c['max_tokens']==16384 and c['max_attempts_per_question']==4 and c['schedule']=='barrier'
 assert trial['valid'] and trial['matched_requests']['initial_requests']==30 and not trial['matched_requests']['different_questions']
 timeline=s['grader_timeline']
 parts=timeline['first_pick_elapsed_s']+timeline['actual_service_s']+timeline['idle_between_queries_s']
 assert abs(parts-s['time_to_target_s'])<0.03 and not timeline['audit_errors']
 assert timeline['correct']==18 and timeline['completed_queries']==18+timeline['wrong']
 row={'trial':trial['trial'],'seed':trial['sampling_seed'],'attempt_id':p.name,'valid':True,'time_to_target_s':s['time_to_target_s'],'first_pick_s':timeline['first_pick_elapsed_s'],'service_s':timeline['actual_service_s'],'idle_s':timeline['idle_between_queries_s'],'wrong_checks':timeline['wrong'],'generation_requests':s['performance']['generation_requests'],'fresh_ttft_median_s':s['performance']['fresh_ttft']['median_s'],'continuation_ttft_median_s':s['performance']['continuation_ttft']['median_s'],'initialization_and_attempt_s':s['initialization_and_attempt_latency_s'],'tail_question_indices':[q['problem_idx'] for q in solved[-3:]],'winner_candidate_kinds':dict(collections.Counter(q['winner']['kind'] for q in solved)),'winning_continuations':sum(any(r['rollout']==q['winner']['rollout'] and r.get('continuation_of_rollout') is not None for r in q['rollouts']) for q in solved)}
 wrong=[]
 for event_file in sorted(p.glob('trace/*/verification.jsonl')):
  for line in event_file.read_text().splitlines():
   event=json.loads(line)
   if event.get('result',{}).get('verdict') is False:
    wrong.append({'question':int(event_file.parent.name),'candidate':event['candidate'],'kind':event['kind'],'line':event.get('line'),'observed_at_utc':event['observed_at_utc'],'round':event['round']})
 assert len(wrong)==timeline['wrong']
 row['wrong_candidates']=wrong
 rows.append(row)
threshold=config['protocol']['reference_time_to_target_s'];times=[r['time_to_target_s'] for r in rows]
analysis={'source_commit':config['source_commit'],'core_manifest_sha256':CORE,'system_prompt_sha256':PROMPT,'reference_time_s':threshold,'all_five_below_reference':all(t<=threshold for t in times),'median_s':statistics.median(times),'min_s':min(times),'max_s':max(times),'rows':rows,'limits':['All five seeds share one inference-server lifetime; independent restart reproducibility is not established.','AIME 2025 is development data.','No paired original/improved-prompt controls were executed; historical warmed trials changed both prompt and warmup.','Benchmark mode disables optional engine/GPU/CPU profiling; coarse server logs are separate observations.']}
(BATCH/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(9,5.5))
positions=list(range(5)); first=[r['first_pick_s'] for r in rows];service=[r['service_s'] for r in rows];idle=[r['idle_s'] for r in rows]
ax.bar(positions,first,label='Before first grader pickup',color='#849dac')
ax.bar(positions,service,bottom=first,label='Grader service (correct + wrong)',color='#286f8b')
ax.bar(positions,idle,bottom=[a+b for a,b in zip(first,service)],label='Later grader idle',color='#dc9c55')
ax.axhline(threshold,color='#ae4141',linestyle='--',linewidth=1.4,label='71.135s historical reference')
ax.axhline(54,color='#666666',linestyle=':',linewidth=1.2,label='54s serial verification floor')
for i,row in enumerate(rows):ax.text(i,row['time_to_target_s']+1,f"{row['time_to_target_s']:.3f}s\n{row['wrong_checks']} wrong",ha='center',fontsize=9)
ax.set_xticks(positions,[str(r['seed']) for r in rows]);ax.set_ylabel('Seconds to 18 distinct correct verdicts');ax.set_ylim(0,max(times)+16)
ax.set_title('Frozen core v1 · improved prompt · five declared seeds',loc='left',fontweight='bold')
ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.11),ncols=2,fontsize=8)
ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.subplots_adjust(left=0.10,right=0.985,top=0.9,bottom=0.24)
for extension in ('png','svg'):fig.savefig(BATCH/f'five-seed-timing.{extension}',dpi=170,metadata={'Date':None} if extension=='svg' else {})
svg=BATCH/'five-seed-timing.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
print(json.dumps(analysis,indent=2))
