"""Replay answer extraction, first-answer stopping, and voting over the original 240 traces.

Use this fixed-cohort offline experiment after eight-sample baseline generation.
It applies shared extraction policies and scope annotations, then writes claim
inventories, trace_map.csv, traces.json, and voting/budget summary statistics in
intermediate_answers/. Requires the local raw attempts and cached Qwen tokenizer;
no inference or remote grading occurs. Token-work savings are retrospective.
    python -m src.experiments.intermediate_answers.analyze_intermediate
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

from tokenizers import Tokenizer
from src.common import ROOT
from src.answer_extraction import BUDGETS, POLICIES, candidate, eligible, make_rows, replay, save_json, trajectory_summary

def main(args):
    started = time.perf_counter()
    run = (ROOT / 'runs' / args.run).resolve()
    if not run.is_relative_to(ROOT / 'runs'):
        raise ValueError('run must be inside the repo')
    output = run / 'intermediate_answers'
    output.mkdir(exist_ok=True)
    tokenizer_path = Path(args.tokenizer_json).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = make_rows(run, tokenizer)
    if len(rows) != 240 or any(sum(r['problem_idx'] == q for r in rows) != 8 for q in range(1, 31)):
        raise ValueError('Expected all 240 saved attempts, eight for each of 30 questions')
    summary = {'run': args.run, 'traces': len(rows), 'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
               'reported_output_tokens': sum(r['completion_tokens'] for r in rows),
               'overhead_distribution': dict(Counter(r['provider_minus_visible_tokens'] for r in rows)),
               'policies': [trajectory_summary(rows, p) for p in POLICIES if p != 'final'],
               'frontier': [replay(rows, p, n) for p in POLICIES for n in range(1, 9)],
               'budget_frontier': [replay(rows, p, n, b) for p in ['final', 'proposal', 'target'] for n in [2, 4, 8] for b in BUDGETS],
               'elapsed_offline_analysis_s': time.perf_counter() - started,
               'limitations': ['Retrospective prefix replay, not a test of a changed generation prompt.',
                               'Audited policies exclude manually annotated toy sections; literal_prose is the uncurated comparison.',
                               'Detected answer clauses are a lower bound: unlabeled intermediate equations and unsupported arithmetic can be missed.',
                               'No streaming timestamps: token positions are measured, early wall times are not.',
                               'Top-2 is candidate coverage requiring an independent selector/verifier, not single-answer accuracy.',
                               'Uniform tie-breaking is evaluated without consulting the key.',
                               'The extraction rule uses syntax and arithmetic, not semantic proof checking.',
                               'Target-equation policy is an aggressive sensitivity analysis, not the primary policy.']}
    save_json(output / 'traces.json', {'rows': rows})
    save_json(output / 'summary.json', summary)
    with (output / 'trace_map.csv').open('w', newline='') as f:
        fields = ['problem_idx', 'sample_number', 'gold_answer', 'first_answer', 'first_correct',
                  'first_answer_tokens', 'first_correct_answer_tokens', 'answer_sequence', 'claims',
                  'final_candidate', 'final_correct', 'finish_reason', 'completion_tokens',
                  'tokens_after_first', 'source_file']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            events = [e for e in row['events'] if eligible(e, 'proposal')]
            first = events[0] if events else {}
            correct_event = next((e for e in events if e['correct']), {})
            sequence = []
            for e in events:
                if not sequence or sequence[-1] != e['answer']: sequence.append(e['answer'])
            writer.writerow({**{k: row[k] for k in fields if k in row},
                             'first_answer': first.get('answer'), 'first_correct': first.get('correct'),
                             'first_answer_tokens': first.get('estimated_stop_tokens'),
                             'first_correct_answer_tokens': correct_event.get('estimated_stop_tokens'),
                             'answer_sequence': ' -> '.join(map(str, sequence)), 'claims': len(events),
                             'tokens_after_first': row['completion_tokens'] - candidate(row, 'proposal')[1]})
    with (output / 'claim_inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['problem_idx', 'sample_number', 'part', 'kind', 'confidence', 'answer', 'correct', 'estimated_stop_tokens', 'quote', 'context'])
        writer.writeheader()
        for row in rows:
            for event in row['events']:
                writer.writerow({k: (row[k] if k in row else event[k]) for k in writer.fieldnames})
    print(json.dumps({'policies': summary['policies'], 'overhead': summary['overhead_distribution'], 'elapsed_s': summary['elapsed_offline_analysis_s']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='20260930-155212')
    parser.add_argument('--tokenizer-json', default=str(ROOT / '.local/tokenizers/qwen3.json'))
    main(parser.parse_args())
