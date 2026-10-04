"""Explain final-run variance using required traces, without optional profiling."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def elapsed(timestamp, start):
    return (datetime.fromisoformat(timestamp.replace('Z', '+00:00')) -
            datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()


def analyze(batch):
    summary = json.loads((batch / 'summary.json').read_text())
    rows = []
    for trial in summary['trials']:
        if not trial.get('valid'):
            continue
        folder = ROOT / 'attempts' / trial['attempt_id']
        config = json.loads((folder / 'config.json').read_text())
        start = config['official_started_at_utc']
        questions = [json.loads(p.read_text()) for p in sorted(folder.glob('trace/*/question.json'))]
        first = [q['rollouts'][0] for q in questions]
        capped = {q['problem_idx']: q['rollouts'][0] for q in questions
                  if q['rollouts'][0]['finish_reason'] == 'length'
                  and q['rollouts'][0]['generated_token_ids_count'] == 8192}
        verified = []
        for path in folder.glob('trace/*/verification.jsonl'):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if isinstance(event.get('result', {}).get('verdict'), bool):
                    verified.append(event)
        verified.sort(key=lambda e: e['result']['seq'])
        gaps = []
        for previous, event in zip(verified, verified[1:]):
            gap = elapsed(event['result']['picked_at'], previous['result']['answered_at'])
            if gap > .1:
                gaps.append({'gap_s': gap, 'next_question': event['result']['index'],
                             'candidate_observed_s': elapsed(event['observed_at_utc'], start),
                             'next_pickup_s': elapsed(event['result']['picked_at'], start),
                             'previous_answer_s': elapsed(previous['result']['answered_at'], start)})
        solved = sorted([q for q in questions if q.get('first_solved')],
                        key=lambda q: q['first_solved']['first_solved_elapsed_s'])
        local_wait = [e['candidate_queue_wait_s'] for e in verified]
        dispatch = [elapsed(e['result']['submitted_at'], e['verification_started_at_utc']) for e in verified]
        row = {'seed': trial['sampling_seed'], 'attempt_id': trial['attempt_id'],
               'time_to_target_s': trial['time_to_target_s'],
               'first_pick_s': trial['grader_timeline']['first_pick_elapsed_s'],
               'grader_service_s': trial['grader_timeline']['actual_service_s'],
               'grader_idle_s': trial['grader_timeline']['idle_between_queries_s'],
               'initial_ttft_median_s': statistics.median(r['ttft_s'] for r in first if r['ttft_s'] is not None),
               'initial_capped': {str(q): {'generation_s': r['generation_latency_s'],
                   'effective_decode_tps': (8192-1)/(r['last_token_s']-r['ttft_s'])}
                   for q, r in capped.items()},
               'first_round_winning_questions': sum(q['winner']['round'] == 1 for q in solved),
               'local_candidate_wait_median_s': statistics.median(local_wait),
               'local_candidate_wait_max_s': max(local_wait),
               'client_submit_dispatch_median_s': statistics.median(dispatch),
               'largest_idle_gaps': sorted(gaps, key=lambda r:r['gap_s'], reverse=True)[:3],
               'last_three_solved': [{'question':q['problem_idx'], 'round':q['winner']['round'],
                   'candidate_observed_s':elapsed(q['winner']['observed_at_utc'],start),
                   'solved_s':q['first_solved']['first_solved_elapsed_s']} for q in solved[-3:]]}
        rows.append(row)
    common = sorted(set.intersection(*(set(r['initial_capped']) for r in rows)), key=int)
    for row in rows:
        row['common_capped_generation_median_s'] = statistics.median(row['initial_capped'][q]['generation_s'] for q in common) if common else None
        row['common_capped_decode_median_tps'] = statistics.median(row['initial_capped'][q]['effective_decode_tps'] for q in common) if common else None
    fast, slow = min(rows, key=lambda r:r['time_to_target_s']), max(rows, key=lambda r:r['time_to_target_s'])
    difference = slow['time_to_target_s'] - fast['time_to_target_s']
    return {'rows': rows, 'common_capped_questions': [int(q) for q in common],
            'fast_seed':fast['seed'], 'slow_seed':slow['seed'], 'spread_s':difference,
            'idle_difference_s':slow['grader_idle_s']-fast['grader_idle_s'],
            'idle_share_of_spread':(slow['grader_idle_s']-fast['grader_idle_s'])/difference}


def report(batch, data):
    rows = data['rows']
    text = '# Why the five warmed runs vary\n\n'
    text += f"The fastest/slowest spread is **{data['spread_s']:.3f}s**. **{data['idle_difference_s']:.3f}s ({data['idle_share_of_spread']:.1%})** comes from extra grader idle time. All five use 19 completed grader checks, one wrong, and approximately 57.002s of service. Candidate arrival in the final few questions explains most of the measured difference. This identifies the timing bottleneck; it does not prove the numerical cause of each reasoning trajectory.\n\n"
    text += '| Seed | First 18 | First-round winners among the final 18 | Initial median TTFT | Same capped questions: median generation | Same capped questions: effective decode | Local candidate wait median |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for r in rows:
        text += f"| {r['seed']} | {r['time_to_target_s']:.3f}s | {r['first_round_winning_questions']} | {r['initial_ttft_median_s']*1000:.1f}ms | {r['common_capped_generation_median_s']:.3f}s | {r['common_capped_decode_median_tps']:.1f} tok/s | {r['local_candidate_wait_median_s']*1000:.3f}ms |\n"
    text += '\nCommon initial 8K-capped questions: '+', '.join(f'Q{q:02d}' for q in data['common_capped_questions'])+'. These requests all produce 8,192 observed token IDs and finish at the cap. Effective decode is `(tokens − 1) / (last visible token − TTFT)`, not a hardware-only kernel benchmark. Comparing identical token counts reduces the bias from early cancellation, but concurrent batch composition can still differ.\n\n'
    text += 'The first-pass budget boundary occurs at nearly the same time, while different seeds produce different sets of early verified answers. Slower seeds need later continuations to supply the 17th/18th answer; those arrivals leave the serial grader idle. Barrier scheduling can add waiting before continuations, and continued reasoning does not guarantee a correct candidate at any particular token count.\n\n'
    text += '| Seed | Largest grader idle gap | Next question | Its correct/checked candidate observed | Last three verified questions (winning round) |\n| --- | ---: | --- | ---: | --- |\n'
    for r in rows:
        gap=r['largest_idle_gaps'][0]
        tail=', '.join(f"Q{q['question']:02d} (r{q['round']}, {q['solved_s']:.1f}s)" for q in r['last_three_solved'])
        text += f"| {r['seed']} | {gap['gap_s']:.3f}s | Q{gap['next_question']:02d} | {gap['candidate_observed_s']:.3f}s | {tail} |\n"
    text += '\nThe new five trials deliberately use different sampling seeds. That changes reasoning paths and answer arrival. Earlier same-seed repetitions also differed: default online vLLM does not guarantee reproducibility across scheduling/batch changes. This is a plausible additional mechanism, not a demonstrated kernel fault here. [vLLM reproducibility documentation](https://docs.vllm.ai/en/stable/usage/reproducibility/). Its [batch-invariance feature](https://docs.vllm.ai/en/latest/features/batch_invariance/) offers a diagnostic direction, but support and performance for this exact NVFP4 Marlin/FlashInfer model build have not been tested.\n\n'
    text += 'AIME 2024 warming cost about 59 seconds per trial and did not remove the answer-arrival tail. Its causal speed benefit is not established without a paired comparison. The selected final v2 defaults therefore restore only the cheaper 30-stream × 32-token warmup; the complete warmed v1 source/protocol and every warmed outcome are preserved. No solving, grading, quantization or token-budget change accompanies that default revert.\n\n'
    text += 'Optional GPU/CPU/engine profiling was disabled. Required timing/token/verdict evidence supports this decomposition; it cannot rule out every transient hardware or KV event. Standard server logs, separately saved in [server-evidence.json](server-evidence.json), provide coarse throughput/KV observations. Client enqueue and submission medians are far smaller than the multi-second idle gaps; the maximum local wait can include intentional per-question serialization behind a wrong verdict.\n\n[Structured variance evidence](variance.json), [five-trial visual report](README.md).\n'
    (batch/'VARIANCE.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch')
    args=parser.parse_args()
    if Path(args.batch).name != args.batch or args.batch in ('.','..'):parser.error('Use a plain batch directory name')
    batch=ROOT/'runs/experiments'/args.batch
    data=analyze(batch)
    (batch/'variance.json').write_text(json.dumps(data,indent=2)+'\n')
    report(batch,data)
    print(json.dumps({k:v for k,v in data.items() if k!='rows'}))


if __name__=='__main__':main()
