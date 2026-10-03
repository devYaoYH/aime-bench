"""Render paired prompt-pilot outcomes, delivery rates, and hypothetical cancellation savings.

Use this offline renderer after first_answer_pilot completes. It reads saved
question/SSE records and the cached Qwen tokenizer, writes plots and timing
JSON beside the run, and updates docs/reports/first-answer-pilot.md. Retrospective
marker-arrival opportunities are distinguished from actual cancellation savings.
No inference occurs.
    python -m src.experiments.streaming.report_first_answer_pilot
"""
import argparse
from bisect import bisect_right
import json
import os

from src.common import ROOT

os.environ['MPLCONFIGDIR'] = str(ROOT / '.local/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer


def render(directory):
    output = (ROOT / directory).resolve()
    s = json.loads((output / 'summary.json').read_text())
    comparisons = s['comparisons']
    records = [json.loads(p.read_text()) for p in sorted((output / 'questions').glob('*.json')) if '-progress' not in p.name]
    totals = s['arm_totals']
    control, early = totals['control'], totals['first_answer']
    tokenizer = Tokenizer.from_file(str(ROOT / '.local/tokenizers/qwen3.json'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    x = np.arange(len(comparisons))
    colors = {'control': '#64748b', 'first_answer': '#059669'}
    for arm, shift, label in [('control', -.18, 'Original prompt'), ('first_answer', .18, 'First-answer prompt')]:
        for ax, metric in zip(axes, ['completion_tokens', 'generation_latency_s']):
            values = [c[arm][metric] or 0 for c in comparisons]
            bars = ax.bar(x + shift, values, width=.36, color=colors[arm], label=label)
            for bar, c in zip(bars, comparisons):
                val = bar.get_height()
                status = '✓' if c[arm]['correct'] else '×'
                label_value = f'{val:,.0f}' if metric == 'completion_tokens' else f'{val:.1f}s'
                ax.annotate(label_value + '\n' + status, (bar.get_x()+bar.get_width()/2, val),
                            ha='center', xytext=(0, 5), textcoords='offset points', fontsize=7)
    for ax, title in zip(axes, ['Generated tokens (including reasoning)', 'Request-to-natural-end latency']):
        ax.set(xticks=x, xticklabels=[f"Q{c['problem_idx']:02d}" for c in comparisons], title=title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Seconds')
    fig.suptitle('Paired prompt pilot · one sample per arm per question\n✓ correct key match · × wrong/missing · no forced stopping', fontsize=12)
    fig.savefig(output / 'paired_results.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    rates = []
    for ax, c in zip(axes.flat, comparisons):
        for arm in ['control', 'first_answer']:
            r = next(r for r in records if r['problem_idx'] == c['problem_idx'] and r['arm'] == arm)
            if not r.get('response'): continue
            message = r['response']['choices'][0]['message']
            encoded = {p: tokenizer.encode(message[p], add_special_tokens=False) for p in ['reasoning', 'content']}
            ends = {p: [o[1] for o in encoded[p].offsets] for p in encoded}
            times, counts = [0], [0]
            for d in r['timing']['deliveries']:
                times.append(d['elapsed_s'])
                counts.append(sum(bisect_right(ends[p], d[p]) for p in encoded))
            ax.plot(times, counts, color=colors[arm], label=arm.replace('_', ' '))
            if counts[-1] > 0:
                checkpoints = [next(t for t, n in zip(times, counts) if n >= counts[-1] * fraction)
                               for fraction in [.25, .5, .75, 1.0]]
                rates.append({'problem_idx': c['problem_idx'], 'arm': arm, 'visible_tokens': counts[-1],
                              'second_quarter_tokens_per_s': counts[-1] / 4 / (checkpoints[1] - checkpoints[0]),
                              'last_quarter_tokens_per_s': counts[-1] / 4 / (checkpoints[3] - checkpoints[2])})
            first = r['analysis']['first_literal_proposal']
            if first and first['observed_at_s'] is not None:
                ax.scatter([first['observed_at_s']], [first['visible_output_token']], color=colors[arm], marker='o', s=30)
        ax.set(title=f"Q{c['problem_idx']:02d}", xlabel='Seconds after request start', ylabel='Visible Qwen tokens delivered')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Measured SSE delivery curves\nDots: first automatically detected answer clause/marker; delivery includes network effects', fontsize=12)
    fig.savefig(output / 'delivery_curves.png', dpi=170)
    plt.close(fig)
    (output / 'delivery_rates.json').write_text(json.dumps(rates, indent=2) + '\n')

    token_saving = 1 - early['completion_tokens'] / control['completion_tokens']
    latency_saving = 1 - early['sum_latency_s'] / control['sum_latency_s']
    batch_saving = 1 - early['batch_max_latency_s'] / control['batch_max_latency_s']
    hypothetical_cancellation = {}
    for arm in ['control', 'first_answer']:
        group = [r for r in records if r['arm'] == arm]
        prefix_times, prefix_tokens = [], []
        for r in group:
            first = r['analysis']['first_literal_proposal']
            prefix_times.append(first['observed_at_s'] if first else r['generation_latency_s'])
            prefix_tokens.append(first['estimated_stop_tokens'] if first else r['usage']['completion_tokens'])
        hypothetical_cancellation[arm] = {
            'summed_prefix_latency_s': sum(prefix_times), 'max_prefix_latency_s': max(prefix_times),
            'summed_prefix_tokens': sum(prefix_tokens),
            'summed_latency_reduction_fraction': 1-sum(prefix_times)/totals[arm]['sum_latency_s'],
            'correct_first_proposals': sum(r['analysis']['first_literal_proposal'] is not None and
                                          r['analysis']['first_literal_proposal']['correct'] for r in group),
            'note': 'Retrospective measured arrival times; no cancellation was performed, so billing and cancellation overhead are untested.'}
    (output / 'cancellation_replay.json').write_text(json.dumps(hypothetical_cancellation, indent=2) + '\n')
    relative = output.relative_to(ROOT)
    text = [
        '# First-answer system-prompt pilot', '',
        '**The added instruction did not reliably stop Qwen at its first answer.** '
        'All three modified traces that completed still rechecked after a correct requested-answer proposal; '
        'the fourth never produced a detected proposal and hit the cap. '
        'The completed traces contain explicit “let me check” passages after the first candidate. '
        'This small pilot therefore shows a failure of prompt compliance, even though all three completed answers are correct.', '',
        f"**{early['correct']}/4 correct with the modified prompt versus {control['correct']}/4 with the original prompt.** "
        f"Generated output fell from {control['completion_tokens']:,} to {early['completion_tokens']:,} tokens "
        f"({token_saving:.1%} less). Sum of request-end latencies fell {latency_saving:.1%}; "
        f"the longest request in each simultaneous four-question arm fell from {control['batch_max_latency_s']:.2f}s "
        f"to {early['batch_max_latency_s']:.2f}s ({batch_saving:.1%} less). "
        'These are observed measurements on selected questions, not an estimate for the full dataset.', '',
        '## Design', '',
        'Eight requests started together: one original-prompt control and one modified-prompt sample for each of Q1, Q3, Q18, and Q25. '
        'Both used `qwen/qwen3-30b-a3b` at the existing OpenRouter chat-completions endpoint, routed to DeepInfra with fallbacks disabled. '
        'Temperature 0.6, top-p 0.95, top-k 20, reasoning enabled and returned, and the 16,384-token cap match the original run. '
        'Both arms stream; only their system messages differ. The answer key is never included in either request. '
        'Correctness uses the stored key locally; no provided grader, SSH, or local llama inference was used.', '',
        'Q1 and Q3 had consistently correct early proposals with large tails; Q18 tests salvaging an answer from a previously capped problem; '
        'Q25 tests a harder combinatorial problem with repeated checks and toy subproblems. '
        'Selection was made before this pilot from the prior analysis. The 4-question selection favors a measurable effect.', '',
        'The original system prompt was retained verbatim, with this suffix:', '',
        '```text', s['config']['treatment_prompt'][len(s['config']['control_prompt']):].strip(), '```', '',
        '**No stop sequence, smaller cap, answer-triggered cancellation, or continuation was imposed.** '
        f"The modified arm ended naturally (`finish_reason=stop`) on {early['natural_stops']}/4 requests. "
        'A natural endpoint establishes that the model ended its response, while its reasoning trace is needed to assess '
        'whether it ended at the first candidate rather than rechecking first.', '',
        '## Observed paired results', '',
        '| Question | Key | Original answer / finish | Modified answer / finish | Original → modified tokens | Original → modified seconds |',
        '|---|---:|---|---|---:|---:|',
    ]
    for c in comparisons:
        a, b = c['control'], c['first_answer']
        text.append(f"| {c['problem_idx']:02d} | {c['gold_answer']:03d} | {a['candidate']} / {a['finish_reason']} | "
                    f"{b['candidate']} / {b['finish_reason']} | {a['completion_tokens']:,} → {b['completion_tokens']:,} | "
                    f"{a['generation_latency_s']:.2f} → {b['generation_latency_s']:.2f} |")
    text += ['', f'![Paired results](../../{relative}/paired_results.png)', '',
             '## First proposal and residual generation', '',
             'First proposals here use the existing automatic answer-clause/marker extractor without source-specific toy exclusions. '
             'The times are the first received SSE chunks containing the delimited claim; the token position uses the cached official Qwen tokenizer. '
             'These measurements can miss a complete answer computed in an unlabeled equation, and toy proposals must be checked against context.', '',
             '| Question | Arm | First detected proposal | First observed (s) | Further tokens | Further seconds |',
             '|---|---|---:|---:|---:|---:|']
    for c in comparisons:
        for arm in ['control', 'first_answer']:
            r = c[arm]
            first = r.get('first_proposal')
            if first:
                text.append(f"| {c['problem_idx']:02d} | {arm} | {first['answer']} | {first['observed_at_s']:.2f} | "
                            f"{r['tokens_after_first_proposal']:,} | {r['seconds_after_first_proposal']:.2f} |")
            else: text.append(f"| {c['problem_idx']:02d} | {arm} | — | — | — | — |")
    text += ['',
             'Q1 modified: “the answer is 21 + 49 = 70?” is followed immediately by “But let me check if there are other possibilities.” '
             'Q3 modified: “the answer would be 16?” is followed by “Let me check my calculations again.” '
             'Q25 modified: after the requested 907 it says “But just to be thorough, let me check if there is any mistake.” '
             'These are continued checks of the requested answer, not merely the final formatting overhead.', '',
             'Q25 modified also derives `N = 2907` much earlier than its first explicit answer clause, then verifies it with another method '
             'before computing the requested remainder. The clause-based residual table understates that additional rechecking. '
             'Neither Q18 reasoning trace contains a literal 82 or any detected numeric answer proposal.', '',
             f'![Streamed delivery](../../{relative}/delivery_curves.png)', '',
             'The client-observed token-delivery curves are approximately linear over these streams. '
             'The second-quarter and last-quarter delivery rates are tabulated below, excluding the initial startup interval. '
             'This endpoint did not show a pronounced nonlinear slowdown within the measured output lengths. '
             'That observation concerns client delivery under this workload; it does not isolate model compute time or establish '
             'what happens on other hardware, batch sizes, or longer contexts.', '',
             '| Question | Arm | Second-quarter tokens/s | Last-quarter tokens/s |',
             '|---|---|---:|---:|']
    for r in rates:
        text.append(f"| {r['problem_idx']:02d} | {r['arm']} | {r['second_quarter_tokens_per_s']:.1f} | {r['last_quarter_tokens_per_s']:.1f} |")
    text += ['',
             '## Potential client-enforced stopping', '',
             'The streams now measure an opportunity that the prompt did not realize. '
             'In the original-prompt Q25 control, a correct 907 is received at 47.87s, yet the request continues to the cap at 206.50s. '
             'With hypothetical cancellation on the first detected answer clause, and keeping Q18 running to its cap, '
             f"summed request time would be {hypothetical_cancellation['control']['summed_prefix_latency_s']:.2f}s instead of "
             f"{control['sum_latency_s']:.2f}s for the controls ({hypothetical_cancellation['control']['summed_latency_reduction_fraction']:.1%} less), "
             f"and {hypothetical_cancellation['first_answer']['summed_prefix_latency_s']:.2f}s instead of "
             f"{early['sum_latency_s']:.2f}s for the modified arm "
             f"({hypothetical_cancellation['first_answer']['summed_latency_reduction_fraction']:.1%} less). "
             'The first detected proposals in these three answer-producing traces are all correct in each arm. '
             'These are retrospective prefix-arrival calculations; no client cancellation, post-cancellation billing, '
             'or cancellation overhead was tested. Q18 still dominates a wait-for-all batch, so its elapsed wall time scarcely changes.', '',
             'A further experiment should enforce a streamed candidate marker in the application, '
             'validate that the marker refers to the requested quantity, and cancel the request when it arrives. '
             'Parallel sampling and top-two selection can then be tested against the accuracy lost by early stopping. '
             'The current added prompt alone does not provide that stopping guarantee.', '',
             '## Limits and saved evidence', '',
             'This is one independent stochastic draw per arm per question. It is not a shared-prefix continuation experiment, '
             'and it cannot estimate accuracy changes or timing variance reliably. '
             'Fresh controls give a more relevant latency comparison than the historical runs, but queueing, shared server load, '
             'prompt caching, network effects, and backend variation still contribute. '
             'The arm batch maxima are measured request-end maxima within this joint eight-request workload; '
             'they are not timings of two separately executed four-request batches.', '',
             'Natural stopping and a brief tail after a detected answer do not establish that every first complete internal candidate '
             'was immediately emitted. Full reasoning traces remain available for that inspection. '
             'The pilot tests prompt-driven termination; it does not test parallel self-consistency voting, selection between two answers, '
             'or application-driven streaming cancellation.', '',
             f'- [Full measured summary](../../{relative}/summary.json)',
             f'- [Exact original and modified prompts and sampling](../../{relative}/config.json)',
             f'- [Retrospective cancellation replay](../../{relative}/cancellation_replay.json)',
             f'- [Quarter-stream delivery rates](../../{relative}/delivery_rates.json)',
             f'- Full response traces, timestamps for every output delivery, and raw SSE events: `{relative}/questions/`.',
             f'- Eight complete streams: control {control["successful_streams"]}/4, modified {early["successful_streams"]}/4.',
             f'- Reported API cost across both arms: ${control["reported_cost"] + early["reported_cost"]:.6f}.', '',
             'Reproduce the pilot (makes eight paid API calls, reuses completed records at the same path):', '',
             '```bash', '.venv/bin/python -m unittest test.test_first_answer_pilot',
             '.venv/bin/python -m src.experiments.streaming.first_answer_pilot --out runs/first-answer-NEW-LABEL',
             '.venv/bin/python -m src.experiments.streaming.report_first_answer_pilot --out runs/first-answer-NEW-LABEL', '```', '',
             'Render this report from existing records without inference:', '', '```bash',
             f'.venv/bin/python -m src.experiments.streaming.report_first_answer_pilot --out {relative}', '```', '']
    report = ROOT / 'docs/reports/first-answer-pilot.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(text))
    print(json.dumps({'token_saving_fraction': token_saving, 'summed_latency_saving_fraction': latency_saving,
                      'arm_max_latency_saving_fraction': batch_saving}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='runs/first-answer-20261002-paired')
    render(p.parse_args().out)
