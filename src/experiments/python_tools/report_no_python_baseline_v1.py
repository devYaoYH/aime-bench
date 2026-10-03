"""Offline comparison of historical optional-Python and matched no-tool arms."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import os
import statistics

from src.common import ROOT, atomic_json
from src.experiments.python_tools.no_python_baseline_v1 import baseline_request


def describe(records):
    first = min(datetime.fromisoformat(r['started_at_utc']) for r in records)
    arrivals = {}
    for r in records:
        if r['correct']:
            elapsed = (datetime.fromisoformat(r['finished_at_utc'])-first).total_seconds()
            q = r['problem_idx']
            arrivals[q] = min(elapsed, arrivals.get(q, elapsed))
    events = sorted(arrivals.values())
    return {'attempts': len(records), 'correct_attempts': sum(r['correct'] for r in records),
            'pass_at_2': len(arrivals),
            'sample_1_correct': sum(r['correct'] for r in records if r['sample_idx'] == 1),
            'sample_2_correct': sum(r['correct'] for r in records if r['sample_idx'] == 2),
            'completion_tokens': sum(r['accounting']['completion_tokens'] or 0 for r in records),
            'prompt_tokens': sum(r['accounting']['prompt_tokens'] or 0 for r in records),
            'reported_cost': sum(r['accounting']['reported_cost'] for r in records),
            'median_case_elapsed_s': statistics.median(r['elapsed_s'] for r in records),
            'time_to_18_correct_s': events[17] if len(events) >= 18 else None,
            'status_counts': dict(Counter(r['status'] for r in records)),
            'response_providers': dict(Counter(round.get('response', {}).get('provider', 'missing')
                                              for r in records for round in r['rounds'])),
            'response_model_counts': dict(Counter(round.get('response', {}).get('model', 'missing')
                                                 for r in records for round in r['rounds'])),
            'system_fingerprints': dict(Counter(str(round.get('response', {}).get('system_fingerprint'))
                                                for r in records for round in r['rounds'])),
            'correct_questions': sorted(arrivals), 'correct_arrival_elapsed_s': events}


def render(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT/'runs'):
        raise ValueError('Output must be under runs/')
    config = json.loads((output/'config.json').read_text())
    reference_dir = ROOT/config['reference_run']
    saved_summary = json.loads((output/'summary.json').read_text())
    if not saved_summary['all_attempts_observed']:
        raise ValueError('Wait for all 60 attempts before comparing')
    control, baseline = {}, {}
    for s in (1, 2):
        for q in config['questions']:
            name = f'{q:02d}-sample-{s}.json'
            if hashlib.sha256((reference_dir/name).read_bytes()).hexdigest() != config['reference_record_sha256'][name]:
                raise ValueError(f'Historical reference changed: {name}')
            old = json.loads((reference_dir/name).read_text())
            new = json.loads((output/name).read_text())
            if new['status'] == 'running' or new['rounds'][0]['request'] != baseline_request(old):
                raise ValueError(f'Unmatched or unfinished baseline: {name}')
            if new['tool_executions'] or new['gold_answer'] != old['gold_answer']:
                raise ValueError(f'No-tool/key invariant failed: {name}')
            control[q, s], baseline[q, s] = old, new
    arms = {'optional_python': describe(list(control.values())), 'no_python': describe(list(baseline.values()))}
    old_summary = json.loads((reference_dir/'summary.json').read_text())
    arms['optional_python']['profile_wall_s'] = old_summary['profile_wall_s']
    arms['no_python']['profile_wall_s'] = saved_summary['profile_wall_s']
    paired = Counter()
    question_pairs = Counter()
    rows = []
    for pair, old in control.items():
        new = baseline[pair]
        label = ('both_correct' if old['correct'] and new['correct'] else 'python_only_correct'
                 if old['correct'] else 'no_python_only_correct' if new['correct'] else 'neither_correct')
        paired[label] += 1
        rows.append({'problem_idx': pair[0], 'sample_idx': pair[1], 'comparison': label,
                     'python_status': old['status'], 'no_python_status': new['status'],
                     'python_answer': old['candidate'], 'no_python_answer': new['candidate'],
                     'python_correct': old['correct'], 'no_python_correct': new['correct'],
                     'python_completion_tokens': old['accounting']['completion_tokens'],
                     'no_python_completion_tokens': new['accounting']['completion_tokens'],
                     'python_elapsed_s': old['elapsed_s'], 'no_python_elapsed_s': new['elapsed_s']})
    for q in config['questions']:
        old = any(control[q, s]['correct'] for s in (1, 2))
        new = any(baseline[q, s]['correct'] for s in (1, 2))
        question_pairs['both_correct' if old and new else 'python_only_correct' if old
                       else 'no_python_only_correct' if new else 'neither_correct'] += 1
    comparison = {'arms': arms, 'provenance': json.loads((output/'provenance.json').read_text()),
                  'paired_attempt_counts': dict(paired),
                  'paired_question_counts': dict(question_pairs), 'paired_attempts': rows,
                  'request_audit': {'matched_requests': len(rows), 'allowed_changes': config['request_changes']},
                  'limits': ['Historical and baseline runs took place at different times; provider load and backend revisions are uncontrolled.',
                             'Same seeds and provider do not guarantee identical stochastic output.',
                             'Only two attempts per question; accuracy differences are descriptive.',
                             'Complete final-answer exact match; capped/error answers excluded in both arms.',
                             'The intervention removes both tool availability and its prompt instructions.']}
    atomic_json(output/'analysis.json', comparison)
    with (output/'paired_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    os.environ['MPLCONFIGDIR'] = str(output/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for name, label, color in [('optional_python', 'Optional Python (historical)', '#227f9b'),
                               ('no_python', 'No Python (matched baseline)', '#b45629')]:
        arm = arms[name]
        events = arm['correct_arrival_elapsed_s']
        xs = [0]+[t/60 for t in events]+[arm['profile_wall_s']/60]
        ys = [0]+list(range(1, len(events)+1))+[len(events)]
        axes[0].step(xs, ys, where='post', label=label, color=color, linewidth=2)
        records = control if name == 'optional_python' else baseline
        shift = -.18 if name == 'optional_python' else .18
        axes[1].bar([q+shift for q in config['questions']],
                    [sum(records[q, s]['correct'] for s in (1, 2)) for q in config['questions']],
                    width=.35, color=color, label=label)
    axes[0].axhline(18, linestyle='--', color='#888', linewidth=.8)
    axes[0].set(xlabel='Minutes since each arm started', ylabel='Distinct correct final answers', ylim=(0, 30),
                title='Hosted arrival times (different run dates)')
    axes[1].set(xlabel='AIME question', ylabel='Correct attempts out of two', ylim=(0, 2.4),
                title='Per-question outcomes', xticks=range(1, 31), yticks=[0, 1, 2])
    axes[1].tick_params(axis='x', labelsize=7)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Qwen3.5-35B-A3B · OpenRouter / Parasail · matched pass@2 controls')
    fig.savefig(output/'comparison.png', dpi=170)
    plt.close(fig)
    metrics = [('Questions solved, pass@2', 'pass_at_2', '/30'),
               ('Correct attempts', 'correct_attempts', '/60'),
               ('Sample 1 correct', 'sample_1_correct', '/30'),
               ('Sample 2 correct', 'sample_2_correct', '/30'),
               ('Generated tokens', 'completion_tokens', ''), ('Input tokens', 'prompt_tokens', ''),
               ('Reported cost, USD', 'reported_cost', ''), ('Total hosted wall time, s', 'profile_wall_s', ''),
               ('Median attempt elapsed, s', 'median_case_elapsed_s', ''),
               ('Time to 18 correct final answers, s', 'time_to_18_correct_s', '')]
    lines = ['# Qwen3.5-35B-A3B: matched no-Python baseline', '',
             'Both arms used OpenRouter with Parasail pinned and fallbacks disabled. The no-tool arm ran from the local client.', '',
             '| Metric | Historical optional Python | No Python |', '|---|---:|---:|']
    def number(value):
        return 'target unmet' if value is None else f'{value:.6f}' if isinstance(value, float) and abs(value) < 1 else f'{value:.2f}' if isinstance(value, float) else f'{value:,}'
    for label, key, suffix in metrics:
        lines.append(f'| {label} | {number(arms["optional_python"][key])}{suffix} | {number(arms["no_python"][key])}{suffix} |')
    lines += ['', '![Hosted timing and per-question accuracy](comparison.png)', '',
              'For the same intermediate-answer extraction and three-second verification replay used by the tool arm, see the [paired intermediate-answer report](early_verify_pass2/report.md).', '', '## Matched controls', '',
              'All 60 saved first requests passed a field-for-field audit against the historical arm, allowing only removal of `tools`, `tool_choice`, and the Python-specific system suffix. The original concise solver prompt remains.', '',
              'Controls: same 30 questions, two samples each, launch order, per-question/sample seeds (`2026100200 + 100*q + sample`), eight active trajectories, thinking enabled with reasoning retained, temperature 0.6, top-p 0.95, top-k 20, and 16,384 cumulative generated tokens. Both samples run regardless of the first result. No automatic retries or early verification. No answer key enters model inputs.', '',
              'Routing: `order=["parasail"]`, `allow_fallbacks=false`, `require_parameters=true`. Every response is checked for the expected provider and model. The tool arm permits follow-up tool rounds within the cumulative budget; the no-tool arm has one request per attempt.', '',
              'The provider control follows [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).', '',
              f'Optional-Python statuses: `{arms["optional_python"]["status_counts"]}`. No-Python statuses: `{arms["no_python"]["status_counts"]}`.', '',
              f'Response providers: optional Python `{arms["optional_python"]["response_providers"]}`; no Python `{arms["no_python"]["response_providers"]}`.', '',
              f'Paired attempt outcomes: `{dict(paired)}`. Paired question pass@2 outcomes: `{dict(question_pairs)}`.', '',
              '## Per-question results', '',
              '| Q | Python sample 1 | No Python sample 1 | Python sample 2 | No Python sample 2 |', '|---|---|---|---|---|']
    def outcome(record):
        answer = record['candidate'] if record['candidate'] is not None else '—'
        status = 'correct' if record['correct'] else 'wrong' if record['status'] == 'complete' else record['status']
        return f'{answer} / {status}'
    for q in config['questions']:
        lines.append(f'| {q} | '+ ' | '.join(outcome(arm[q, s]) for s in (1, 2) for arm in (control, baseline))+' |')
    lines += ['', '## Interpretation', ''] + [f'- {note}' for note in comparison['limits']]
    lines += ['', 'Exact requests/responses and local grading are saved in `NN-sample-S.json`; provenance is in `provenance.json`. Full traces remain local. Summary, request hashes, paired metrics, and this report are versioned.', '',
              'Reproduce the offline comparison:', '',
              '```sh', f'.venv/bin/python -m src.experiments.python_tools.report_no_python_baseline_v1 --out {output.relative_to(ROOT)}', '```', '']
    (output/'report.md').write_text('\n'.join(lines))
    (output/'README.md').write_text('''# Matched no-Python Qwen3.5 pass@2\n\nNo-tool baseline for `../python-tool-qwen35-pass2-auto/`, using the same OpenRouter/Parasail routing and generation controls.\n\n- Producer: `src.experiments.python_tools.no_python_baseline_v1`.\n- Analysis: `src.experiments.python_tools.report_no_python_baseline_v1`.\n- See [comparison report](report.md), [config](config.json), [summary](summary.json), and [paired metrics](analysis.json).\n- [Intermediate-answer comparison](early_verify_pass2/report.md) reuses the historical extractor and three-second verifier simulation.\n- Raw paid requests are saved locally; unfinished records block automatic relaunch.\n''')
    print(json.dumps({'arms': {k: {n: v for n, v in a.items() if n != 'correct_arrival_elapsed_s'} for k, a in arms.items()},
                      'paired_attempt_counts': dict(paired), 'paired_question_counts': dict(question_pairs)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='runs/no-python-qwen35-pass2-parasail-20261003')
    render(ROOT/p.parse_args().out)
