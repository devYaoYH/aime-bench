"""Audit conservative answer-marker detection on paired SSE streams and historical traces.

Use this offline experiment to replay the eight saved first-answer streams
and all 240 original responses through src.stream_answer_markers. Writes
guarded_marker_replay.json beside the paired run and guarded_marker_audit.json
in the baseline intermediate_answers/ directory. Requires ignored SSE/raw traces
and derived traces.json; grades only audit marker agreement, not semantic proof.
    python -m src.experiments.streaming.audit_stream_markers
"""
import argparse
import json
from pathlib import Path

from src.common import ROOT
from src.stream_answer_markers import FinalAnswerDetector, delta_text


def replay(output):
    results = []
    for path in sorted((output / 'questions').glob('*.json')):
        if '-progress' in path.name: continue
        record = json.loads(path.read_text())
        detector, first = FinalAnswerDetector(), None
        stream = output / record['api_attempts'][-1]['stream_file']
        for line in stream.read_text().splitlines():
            event = json.loads(line)
            for choice in event['chunk'].get('choices', []):
                if choice.get('index', 0) != 0: continue
                for part, value in delta_text(choice.get('delta') or {}).items():
                    hits = detector.feed(part, value, event['elapsed_s'])
                    if hits and first is None: first = hits[0]
        if record.get('done_received'):
            hits = detector.finish(record['generation_latency_s'])
            if hits and first is None: first = hits[0]
        results.append({'problem_idx': record['problem_idx'], 'arm': record['arm'],
                        'natural_latency_s': record['generation_latency_s'], 'marker': first,
                        'correct': first['answer'] == record['gold_answer'] if first else None,
                        'hypothetical_latency_s': first['observed_at_s'] if first else record['generation_latency_s']})
    summary = {}
    for arm in ('control', 'first_answer'):
        group = [r for r in results if r['arm'] == arm]
        end = sum(r['natural_latency_s'] for r in group)
        prefix = sum(r['hypothetical_latency_s'] for r in group)
        summary[arm] = {'matches': sum(r['marker'] is not None for r in group),
                        'correct_markers': sum(r['correct'] is True for r in group),
                        'sum_original_latency_s': end, 'sum_prefix_latency_s': prefix,
                        'summed_latency_saving_fraction': 1 - prefix/end,
                        'max_original_latency_s': max(r['natural_latency_s'] for r in group),
                        'max_prefix_latency_s': max(r['hypothetical_latency_s'] for r in group)}
    value = {'notes': ['Raw SSE replay; no new inference or actual interruption.',
                       'Syntax and nearby context guard, not semantic verification or a correctness guarantee.',
                       'Require a full standalone line under a Final Answer heading; wait for a line boundary.',
                       'Do not count network chunk EOF as a line/number boundary.'],
             'summary': summary, 'rows': results}
    (output / 'guarded_marker_replay.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def historical(run):
    rows = json.loads((run / 'intermediate_answers/traces.json').read_text())['rows']
    results = []
    for row in rows:
        source = json.loads((ROOT / row['source_file']).read_text())
        message = source['response']['choices'][0]['message']
        detector, first = FinalAnswerDetector(), None
        for part in ('reasoning', 'content'):
            text = message.get(part) or ''
            hits = detector.feed(part, text)
            # Part completion in saved nonstreaming traces is only known post hoc.
            # Do not call finish between parts, as this could disturb pending state.
            if hits and first is None: first = hits[0]
        hits = detector.finish()
        if hits and first is None: first = hits[0]
        if first:
            text = message[first['part']]
            first['context'] = text[max(0, first['start'] - 150):min(len(text), first['end'] + 100)]
        results.append({'problem_idx': row['problem_idx'], 'sample_number': row['sample_number'],
                        'marker': first, 'correct': first['answer'] == row['gold_answer'] if first else None})
    value = {'traces': len(rows), 'with_marker': sum(r['marker'] is not None for r in results),
             'correct_markers': sum(r['correct'] is True for r in results),
             'note': 'Exact-key agreement is not evidence of correct semantic scope. Contexts are retained for audit.',
             'rows': results}
    (run / 'intermediate_answers/guarded_marker_audit.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='runs/first-answer-20261002-paired')
    p.add_argument('--source-run', default='20260930-155212')
    args = p.parse_args()
    replay(ROOT / args.out)
    historical(ROOT / 'runs' / args.source_run)
