"""Apply the historical intermediate-answer extraction/replay to the no-tool arm.

Reuses the unchanged tool-profile adapter, permissive text extractor, tokenizer,
and three-second FIFO verification simulator. No inference or actual verification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os

from tokenizers import Tokenizer

from src.common import ROOT, atomic_json
from src.experiments.python_tools.backtest_python_early_verify import (
    POLICIES, LABELS, adapt_record, fixed_starts, policy_rows)
from src.experiments.python_tools.no_python_baseline_v1 import baseline_request
from src.verification_replay import simulate


def replay(rows):
    final_questions = sorted({r['problem_idx'] for r in rows if r['final_correct']})
    coverage = {p: sorted({r['problem_idx'] for r in policy_rows(rows, p)
        if r['final_correct'] or any(e['answer'] == r['gold_answer'] and
            (p != 'markers' or e['kind'] in ('boxed', 'answer_line')) for e in r['candidates'])})
        for p in ('markers', 'permissive', 'permissive_tools')}
    return {'final_correct_questions': final_questions, 'coverage': coverage,
            'recovered_questions': {p: sorted(set(qs)-set(final_questions)) for p, qs in coverage.items()},
            'replays': [simulate(policy_rows(rows, p), policy=p, generation_slots=slots, service_s=3)
                        for slots in (60, 30, 8) for p in POLICIES],
            'observed_start_shadow': [simulate(fixed_starts(policy_rows(rows, p)), policy=p,
                generation_slots=60, service_s=3, stop_on_verified=False) for p in POLICIES],
            'timing_sensitivity_8_slots': [simulate(policy_rows(rows, p), policy=p,
                generation_slots=8, service_s=3, exponent=exponent, first_token_s=delay)
                for p in ('permissive', 'permissive_tools') for exponent in (1, 1.5, 2) for delay in (0, 5, 15)]}


def render(run):
    run = run.resolve()
    if not run.is_relative_to(ROOT/'runs'):
        raise ValueError('Run must stay under runs/')
    config = json.loads((run/'config.json').read_text())
    reference_dir = ROOT/config['reference_run']
    tokenizer_path = ROOT/'.local/tokenizers/qwen35.json'
    tokenizer_meta = json.loads((ROOT/'.local/tokenizers/qwen35-source.json').read_text())
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != tokenizer_meta['sha256']:
        raise ValueError('Cached tokenizer differs from historical provenance')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    old_rows, rows = [], []
    for s in (1, 2):
        for q in config['questions']:
            name = f'{q:02d}-sample-{s}.json'
            old_path, path = reference_dir/name, run/name
            if hashlib.sha256(old_path.read_bytes()).hexdigest() != config['reference_record_sha256'][name]:
                raise ValueError(f'Historical record changed: {name}')
            old, record = json.loads(old_path.read_text()), json.loads(path.read_text())
            if record['status'] in ('running', 'error') or record['rounds'][0]['request'] != baseline_request(old):
                raise ValueError(f'Unfinished, errored, or unmatched trace: {name}')
            old_rows.append(adapt_record(old, tokenizer, str(old_path.relative_to(ROOT))))
            rows.append(adapt_record(record, tokenizer, str(path.relative_to(ROOT))))
    historical = replay(old_rows)
    original = json.loads((reference_dir/'early_verify_pass2/summary.json').read_text())
    if tokenizer_meta != original['tokenizer']:
        raise ValueError('Historical and baseline tokenizer metadata differ')
    for key in ['final_correct_questions', 'coverage', 'recovered_questions']:
        if historical[key] != original[key]:
            raise ValueError(f'Historical extraction results changed: {key}')
    for key in ['replays', 'observed_start_shadow', 'timing_sensitivity_8_slots']:
        for new, old in zip(historical[key], original[key], strict=True):
            for metric in ['policy', 'generation_slots', 'correct_questions', 'time_to_18_s',
                           'checks_completed', 'wrong_checks', 'checks_before_18', 'peak_pending_checks']:
                if new[metric] != old[metric]:
                    raise ValueError(f'Historical replay changed: {key}/{metric}')
    result = replay(rows)
    output = run/'early_verify_pass2'
    output.mkdir(exist_ok=True)
    assumptions = deepcopy(original['assumptions'])
    assumptions += ['No-tool inference retains reasoning and content for the same intermediate-answer extractor.',
                    'Historical extraction/replay scalar results were reproduced exactly before comparing.',
                    'The Python-stdout extension is identical to the text policy in this arm because there are no tool outputs.']
    result.update(run=str(run.relative_to(ROOT)), traces=len(rows), questions=len(config['questions']),
                  samples=[1, 2], tokenizer=tokenizer_meta, assumptions=assumptions,
                  reference_run=str(reference_dir.relative_to(ROOT)))
    atomic_json(output/'summary.json', result)
    inventory = deepcopy(rows)
    for row in inventory:
        for event in row['candidates']:
            event['matches_saved_key'] = event['answer'] == row['gold_answer']
    atomic_json(output/'candidate_inventory.json', {'rows': inventory})
    atomic_json(output/'config.json', {'runner_version': 'report_no_python_early_verify_v1',
        'tokenizer': tokenizer_meta, 'generation_slots': [60, 30, 8], 'service_s': 3,
        'policies': list(POLICIES), 'historical_replay_reproduced': True,
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            'src/answer_extraction.py', 'src/verification_replay.py',
            'src/experiments/python_tools/backtest_python_early_verify.py',
            'src/experiments/python_tools/report_no_python_early_verify_v1.py']}})
    def duration(value):
        return 'target unmet' if value is None else f'{value/60:.2f} min'
    lines = ['# Matched intermediate-answer extraction: no Python versus optional Python', '',
             'Same permissive text extraction from reasoning and content, same cached Qwen3.5 tokenizer, and same shared FIFO verifier: three seconds per distinct question-answer pair. All negative checks cost three seconds and leave inference running. The historical replay results were reproduced exactly.', '',
             '| Answer policy | Historical optional Python | No Python |', '|---|---:|---:|',
             f'| Strict final answers | {len(historical["final_correct_questions"])}/30 | {len(result["final_correct_questions"])}/30 |']
    for policy in ('markers', 'permissive', 'permissive_tools'):
        lines.append(f'| {LABELS[policy]} | {len(historical["coverage"][policy])}/30 | {len(result["coverage"][policy])}/30 |')
    lines += ['', '## Estimated time to 18 verified correct questions', '',
              '| Capacity | Policy | Historical optional Python | No Python |', '|---|---|---:|---:|']
    for old, new in zip(historical['replays'], result['replays'], strict=True):
        lines.append(f'| {new["generation_slots"]} | {LABELS[new["policy"]]} | {duration(old["time_to_18_s"])} | {duration(new["time_to_18_s"])} |')
    lines += ['', '![Eight-slot comparison](comparison.png)', '',
              '## Fixed observed starts, with no rescheduling or cancellation', '',
              '| Policy | Historical optional Python | No Python |', '|---|---:|---:|']
    for old, new in zip(historical['observed_start_shadow'], result['observed_start_shadow'], strict=True):
        lines.append(f'| {LABELS[new["policy"]]} | {duration(old["time_to_18_s"])} | {duration(new["time_to_18_s"])} |')
    lines += ['', f'Newly recovered no-tool questions under the same permissive text policy: {result["recovered_questions"]["permissive"]}.', '',
              '## Eight-slot verification load', '',
              '| Policy | Correct questions | Checks before 18 | Wrong checks | Peak pending checks |', '|---|---:|---:|---:|---:|']
    for r in result['replays']:
        if r['generation_slots'] == 8:
            lines.append(f'| {LABELS[r["policy"]]} | {r["correct_questions"]} | {r["checks_before_18"]} | {r["wrong_checks"]} | {r["peak_pending_checks"]} |')
    lines += ['', '## Limits', '',
              'This is an offline replay using saved answers as a perfect-verifier stand-in. Intermediate arrival times are estimated from token fractions in non-streaming responses. No live intermediate grading, early cancellation, or billing savings occurred. Actual hosted durations come from different run dates. A target-unmet policy is not ranked.', '',
              'Python stdout is an additional policy for the historical arm; the baseline has no stdout. The main permissive comparison uses the unchanged text policy in both arms.', '',
              'Candidate inventories retain incorrect and hypothetical proposals. Correctness annotations are added after extraction and simulation; candidates are not selected using the answer key.', '',
              'Reproduce:', '', '```sh',
              f'.venv/bin/python -m src.experiments.python_tools.report_no_python_early_verify_v1 --run {run.relative_to(ROOT)}', '```', '']
    os.environ['MPLCONFIGDIR'] = str(output/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for ax, policy in zip(axes, ('final', 'permissive')):
        for arm, label, color in [(historical, 'Optional Python (historical)', '#227f9b'),
                                  (result, 'No Python', '#b45629')]:
            r = next(r for r in arm['replays'] if r['generation_slots'] == 8 and r['policy'] == policy)
            end = max(max(s['natural_end_s'] for s in r['generation_starts']),
                      max((m['time_s'] for m in r['milestones']), default=0))/60
            xs = [0]+[m['time_s']/60 for m in r['milestones']]+[end]
            ys = [0]+list(range(1, len(r['milestones'])+1))+[r['correct_questions']]
            ax.step(xs, ys, where='post', label=label, color=color, linewidth=2)
        ax.axhline(18, color='#888', linestyle='--', linewidth=.8)
        ax.set(title=LABELS[policy], xlabel='Estimated elapsed minutes', ylabel='Verified correct questions', ylim=(0, 30))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Matched offline replay · 8 trajectory slots · 3 seconds per verification')
    fig.savefig(output/'comparison.png', dpi=170)
    plt.close(fig)
    (output/'report.md').write_text('\n'.join(lines))
    (output/'README.md').write_text('# No-Python intermediate-answer replay\n\nSame extractor, tokenizer, and verifier settings as the historical tool arm. See [paired report](report.md). Timing is an offline estimate; no live cancellation occurred.\n')
    print(json.dumps({'coverage': result['coverage'], 'recovered_questions': result['recovered_questions'],
                     'historical_replay_reproduced': True,
                     'eight_slots': [{k:r[k] for k in ['policy', 'correct_questions', 'time_to_18_s', 'wrong_checks']}
                                    for r in result['replays'] if r['generation_slots'] == 8]}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='runs/no-python-qwen35-pass2-parasail-20261003')
    render(ROOT/p.parse_args().run)
