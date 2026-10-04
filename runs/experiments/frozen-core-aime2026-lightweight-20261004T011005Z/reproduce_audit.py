"""Audit the one-seed AIME 2026 transfer from saved required client evidence."""
from pathlib import Path
import json,sys,collections
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.benchmarks import dataset_provenance,load_questions
from runner_final.integrity import verify_core
BATCH=Path(__file__).resolve().parent
batch=json.loads((BATCH/'summary.json').read_text())
protocol=json.loads((BATCH/'config.json').read_text())['protocol']
output=ROOT/'attempts'/batch['attempt_id']
c=json.loads((output/'config.json').read_text());s=json.loads((output/'summary.json').read_text())
assert c['git_commit']==batch['source_commit'] and not c['git_dirty']
assert c['core_manifest_sha256']==protocol['core_manifest_sha256']==verify_core(ROOT)
assert c['system_prompt_sha256']==protocol['system_prompt_sha256']
assert c['dataset_provenance']==protocol['dataset_provenance']==dataset_provenance(2026)
assert c['grader_health']['dataset']['sha256']==c['dataset_provenance']['grader_sha256']
assert c['benchmark_year']==2026 and c['benchmark_role']=='generalization' and c['seed']==20261021
assert c['parallelism']==30 and c['rollouts']==1 and c['schedule']=='barrier'
assert c['first_pass_max_tokens']==8192 and c['max_tokens']==16384 and c['max_attempts_per_question']==4
assert c['benchmark'] and c['skip_benchmark_prewarm'] and not c['reuse_server'] and s.get('benchmark_prewarm') is None
qs=[json.loads(q.read_text()) for q in sorted(output.glob('trace/*/question.json'))]
assert len(qs)==30 and {q['problem_idx'] for q in qs}==set(range(1,31))
solved=sorted([q for q in qs if q.get('first_solved')],key=lambda q:q['first_solved']['first_solved_elapsed_s'])
assert s['status']=='completed' and s['target_reached'] and len(solved)==18
assert all(len(q['rollouts'])<=4 for q in qs)
assert all(q['winner']['result']['verdict'] is True and q['winner']['result']['dataset']['year']==2026 for q in solved)
assert all(q['first_solved']['grader_query_id']==q['winner']['result']['query_id'] and q['first_solved']['grader_answered_at_utc']==q['winner']['result']['answered_at'] for q in solved)
assert abs(solved[-1]['first_solved']['first_solved_elapsed_s']-s['time_to_target_s'])<0.01
prompts={q['problem_idx']:q['problem'] for q in load_questions(year=2026)}
continuations=0
for q in qs:
 folder=output/'trace'/f"{q['problem_idx']:02d}"
 first=json.loads((folder/'rollout-01/request.json').read_text())
 assert first['messages'][0]['content']==c['system_prompt'] and first['messages'][1]['content']==prompts[q['problem_idx']]
 assert first['max_tokens']==8192 and first['seed']==c['seed']+q['problem_idx']*4+1
 for r in q['rollouts']:
  parent=r.get('continuation_of_rollout')
  if parent is not None:
   previous=json.loads((folder/f'rollout-{parent:02d}'/'tokens.json').read_text())
   request=json.loads((folder/f"rollout-{r['rollout']:02d}"/'request.json').read_text())
   assert request['prompt']==previous['prompt_token_ids']+previous['output_token_ids']
   assert request['max_tokens']<=16384 and previous['complete']
   continuations+=1
 t=s['grader_timeline']
assert not t['audit_errors'] and t['correct']==18 and t['completed_queries']==19 and t['wrong']==1
assert abs(t['first_pick_elapsed_s']+t['actual_service_s']+t['idle_between_queries_s']-s['time_to_target_s'])<0.03
analysis={'attempt_id':output.name,'source_commit':c['git_commit'],'benchmark_year':2026,'seed':c['seed'],'core_manifest_sha256':c['core_manifest_sha256'],'system_prompt_sha256':c['system_prompt_sha256'],'dataset_provenance':c['dataset_provenance'],'target_successes':len(solved),'time_to_18_s':s['time_to_target_s'],'initialization_and_attempt_s':s['initialization_and_attempt_latency_s'],'grader_timeline':t,'performance':s['performance'],'exact_id_continuations_verified':continuations,'winning_continuations':sum(any(r['rollout']==q['winner']['rollout'] and r.get('continuation_of_rollout') is not None for r in q['rollouts']) for q in solved),'winner_candidate_kinds':dict(collections.Counter(q['winner']['kind'] for q in solved)),'last_three_questions':[q['problem_idx'] for q in solved[-3:]],'storage_flush_s':s['trace_storage']['latency_s'],'service_cleanup_s':s['service_cleanup_latency_s'],'scope':'One predeclared seed, all 30 questions, stop at 18. No retuning. This is a lightweight transfer check, not a repeatability estimate or full-30 accuracy.'}
(BATCH/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
print(json.dumps(analysis,indent=2))
