"""Replay the selected core v1 five-run batch: exact tokens, solve order and tail.

Requires original SSE traces. Does not run inference or consult answer keys.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import csv

from runner_final.core_v1._streaming import CandidateDetector
from runner_final.integrity import verify_core
from src.common import ROOT, atomic_json

BATCH = 'frozen-core-prompt-five-seeds-20261004T005416Z'
OUTPUT = 'core-v1-five-seeds-answer-tokens'


def ts(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def replay(folder, stream, winner=None, cutoff=None):
    """Count IDs at the exact detector event, including its whole SSE chunk."""
    saved = json.loads((folder/'tokens.json').read_text())
    telemetry = json.loads((folder/'telemetry.json').read_text())
    detector = CandidateDetector()
    if telemetry['continuation_of_rollout']:
        parent = folder.parent/f"rollout-{telemetry['continuation_of_rollout']:02d}/tokens.json"
        detector.feed('content',json.loads(parent.read_text())['visible_text'])
    ids, found, at_cutoff, found_elapsed, found_utc = [], None, 0, None, None
    finish = None

    def consider(events, chunk):
        nonlocal found, found_elapsed, found_utc
        for event in events:
            if winner and found is None and all(event[k]==winner[k] for k in ('answer','part','kind','end')):
                found, found_elapsed, found_utc = len(ids), chunk['elapsed_s'], chunk['timestamp_utc']

    digest = hashlib.sha256()
    with stream.open('rb') as file:
        for line in file:
            digest.update(line)
            chunk=json.loads(line)
            if chunk['data']=='[DONE]':
                if finish != 'length':
                    for part in ('reasoning','content'):
                        consider(detector.feed(part,'',eof=True),chunk)
                continue
            body=json.loads(chunk['data'])
            assert not body.get('error'), 'Recorded inference error'
            for choice in body.get('choices',[]):
                assert choice.get('index',0)==0
                ids.extend(choice.get('token_ids') or [])
                if cutoff is not None and ts(chunk['timestamp_utc']) <= cutoff:
                    at_cutoff=len(ids)
                delta=choice.get('delta') or {}
                if telemetry['endpoint']=='/v1/completions':
                    delta={'content':choice.get('text','')}
                for part,value in (('reasoning',delta.get('reasoning_content') or delta.get('reasoning')),('content',delta.get('content'))):
                    if isinstance(value,str) and value:
                        consider(detector.feed(part,value),chunk)
                if choice.get('finish_reason'):
                    finish=choice['finish_reason']
    assert ids==saved['output_token_ids'], f'Output IDs differ: {folder}'
    assert detector.text['content']==saved['visible_text'], f'Visible text differs: {folder}'
    if winner:
        assert found is not None and found > 0, f'Winner missing: {folder}'
        assert 0 <= ts(winner['observed_at_utc'])-ts(found_utc) <= .050, 'Detector event time differs'
    return {'tokens_at_candidate':found,'segment_candidate_elapsed_s':found_elapsed,
            'candidate_chunk_at_utc':found_utc,'tokens_at_cutoff':at_cutoff,
            'stream_sha256':digest.hexdigest()}


def analyze(repo, streams_root, output):
    core_hash=verify_core(repo)
    batch=repo/'runs/experiments'/BATCH
    trials=json.loads((batch/'summary.json').read_text())['trials']
    rows, source_hashes, tail=[] , {}, []
    for trial in trials:
        seed,aid=trial['sampling_seed'],trial['attempt_id']
        attempt=repo/'attempts'/aid
        config=json.loads((attempt/'config.json').read_text())
        summary=json.loads((attempt/'summary.json').read_text())
        assert config['core_manifest_sha256']==core_hash and trial['valid']
        questions=[json.loads(p.read_text()) for p in sorted(attempt.glob('trace/*/question.json'))]
        solved=sorted((q for q in questions if q.get('first_solved')),key=lambda q:q['first_solved']['first_solved_elapsed_s'])
        assert len(questions)==30 and len(solved)==18
        ranks={q['problem_idx']:i for i,q in enumerate(solved,1)}
        cutoff=ts(solved[-1]['first_solved']['first_solved_at_utc'])
        start=ts(config['official_started_at_utc'])
        for q in questions:
            index=q['problem_idx'];winner=q.get('winner');rank=ranks.get(index)
            if winner:
                assert winner['result']['verdict'] is True
                assert q['first_solved']['grader_query_id']==winner['result']['query_id']
            folder=attempt/'trace'/f'{index:02d}'
            previous_tokens=total_observed=at_target=0
            candidate_tokens=verified_tokens=candidate_elapsed=None
            for roll in q['rollouts']:
                num=roll['rollout'];part=folder/f'rollout-{num:02d}'
                token_record=json.loads((part/'tokens.json').read_text())
                if num > 1:
                    assert roll['continuation_of_rollout']==num-1, 'This batch must contain only linear continuations'
                    parent=json.loads((folder/f'rollout-{num-1:02d}'/'tokens.json').read_text())
                    assert token_record['prompt_token_ids']==parent['prompt_token_ids']+parent['output_token_ids']
                is_winner=winner and winner['rollout']==num
                relative=Path('attempts')/aid/'trace'/f'{index:02d}'/f'rollout-{num:02d}'/'stream.jsonl'
                replayed=replay(part,streams_root/relative,winner if is_winner else None,
                                ts(winner['observed_at_utc']) if winner else cutoff)
                source_hashes[str(relative)]=replayed['stream_sha256']
                if is_winner:
                    candidate_tokens=previous_tokens+replayed['tokens_at_candidate']
                    candidate_elapsed=(ts(replayed['candidate_chunk_at_utc'])-start)
                # For successes, count IDs through the positive client-verdict time separately.
                if winner:
                    verdict_replay=replay(part,streams_root/relative,cutoff=ts(q['first_solved']['first_solved_at_utc']))
                    verified_tokens=(verified_tokens or 0)+verdict_replay['tokens_at_cutoff']
                else:
                    at_target += replayed['tokens_at_cutoff']
                previous_tokens += len(token_record['output_token_ids'])
                total_observed += len(token_record['output_token_ids'])
            row={'seed':seed,'attempt_id':aid,'question':index,'verified':bool(winner),
                 'verification_slot':rank,'tokens_to_winning_candidate':candidate_tokens,
                 'tokens_through_positive_verdict':verified_tokens,
                 'observed_output_tokens':total_observed,
                 'unverified_tokens_at_target':at_target if not winner else None,
                 'winning_rollout':winner['rollout'] if winner else None,
                 'candidate_kind':winner['kind'] if winner else None,
                 'candidate_elapsed_s':candidate_elapsed,
                 'verified_elapsed_s':q['first_solved']['first_solved_elapsed_s'] if winner else None,
                 'grader_queue_wait_s':winner['result']['queue_wait_s'] if winner else None,
                 'grader_toll_s':winner['result']['toll_s'] if winner else None,
                 'candidate_to_verdict_s':q['first_solved']['first_solved_elapsed_s']-candidate_elapsed if winner else None}
            rows.append(row)
            if rank in (17,18):tail.append(row)
    winners=[r for r in rows if r['verified']]
    counts=[r['tokens_to_winning_candidate'] for r in winners]
    distribution={'n':len(counts),'min':min(counts),'p25':statistics.quantiles(counts,n=4,method='inclusive')[0],
                  'median':statistics.median(counts),'p75':statistics.quantiles(counts,n=4,method='inclusive')[2],
                  'p90':statistics.quantiles(counts,n=10,method='inclusive')[8],'max':max(counts)}
    thresholds=[]
    for budget in (2048,4096,6144,8192,10240,12288,16384):
        per_seed={str(t['sampling_seed']):sum(r['seed']==t['sampling_seed'] and r['verified'] and r['tokens_to_winning_candidate']<=budget for r in rows) for t in trials}
        thresholds.append({'cumulative_output_budget':budget,'retained_verified_winners':sum(per_seed.values()),'per_seed':per_seed,
                           'discarded_observed_winners':90-sum(per_seed.values()),
                           'unverified_observed_tokens_beyond_budget':sum(max(0,r['unverified_tokens_at_target']-budget) for r in rows if not r['verified'])})
    by_question=[]
    for q in range(1,31):
        vals=[r['tokens_to_winning_candidate'] for r in winners if r['question']==q]
        by_question.append({'question':q,'verified_trials':len(vals),'unverified_at_stop_trials':5-len(vals),
                            'min_tokens':min(vals) if vals else None,'median_tokens':statistics.median(vals) if vals else None,'max_tokens':max(vals) if vals else None,
                            'slot_17_count':sum(r['question']==q and r['verification_slot']==17 for r in rows),
                            'slot_18_count':sum(r['question']==q and r['verification_slot']==18 for r in rows)})
    result={'source_batch':BATCH,'core_manifest_sha256':core_hash,'seeds':[t['sampling_seed'] for t in trials],
            'method':'Replay frozen core v1 CandidateDetector against original SSE; exact streamed output IDs at winning event, including whole triggering chunk and prior continuation segments. Prompts excluded. Select only saved positive grader verdicts, never answer-key comparisons.',
            'scope':'150 question/seed observations; 90 positive verdicts and 60 unverified at global stop. Unverified rows are missing success-token values, not zero or proven unsolvable. Budget retention is descriptive and does not simulate changed batching, retries, replacement answers, or wall-time savings.',
            'distribution':distribution,'thresholds':thresholds,'by_question':by_question,
            'tail_slot_counts':dict(Counter(r['question'] for r in tail)),
            'tail_continuation_winners':sum(r['winning_rollout']>1 for r in tail),
            'tail':sorted(tail,key=lambda r:(r['seed'],r['verification_slot'])),'rows':rows,
            'source_stream_sha256':source_hashes}
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'analysis.json',result)
    for filename,records in [('questions.csv',rows),('tail.csv',result['tail']),('question_summary.csv',by_question)]:
        with (output/filename).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    print(json.dumps({k:result[k] for k in ('distribution','thresholds','tail_slot_counts','tail_continuation_winners')},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=ROOT)
    parser.add_argument('--streams-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    analyze(args.repo,args.streams_root,args.output or args.repo/'runs/analyses'/OUTPUT)


if __name__=='__main__':main()
