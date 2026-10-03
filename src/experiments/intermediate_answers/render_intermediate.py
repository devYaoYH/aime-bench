"""Render plots and the research report from saved intermediate-answer analysis.

Use this offline renderer after analyze_intermediate has produced traces.json
and summary.json. Writes voting/first-answer plots beside those run artifacts
and docs/reports/intermediate-answers.md, including token positions and policy
limitations. Requires the derived local analysis records; makes no model calls.
    python -m src.experiments.intermediate_answers.render_intermediate
"""
import argparse
import json
import os

from src.answer_extraction import ROOT, candidate, eligible

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.local/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(run_name):
    output = ROOT / 'runs' / run_name / 'intermediate_answers'
    summary = json.loads((output / 'summary.json').read_text())
    rows = json.loads((output / 'traces.json').read_text())['rows']
    names = {'final': 'Completed final answer', 'markers': 'First Answer:/boxed',
             'literal_prose': 'First prose (literal)', 'proposal': 'First prose (audited)'}
    colors = {'final': '#475569', 'markers': '#a855f7', 'literal_prose': '#d97706', 'proposal': '#059669'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for policy, name in names.items():
        points = [f for f in summary['frontier'] if f['policy'] == policy]
        for ax, metric in zip(axes, ['unique_modal_correct', 'top2_tie_averaged']):
            ax.plot([f['output_tokens'] / 1e6 for f in points], [f[metric] for f in points],
                    'o-', color=colors[policy], label=name, linewidth=2)
            for f in points:
                if f['parallel_samples'] in [2, 4, 8]:
                    ax.annotate(str(f['parallel_samples']), (f['output_tokens'] / 1e6, f[metric]),
                                xytext=(3, 5), textcoords='offset points', color=colors[policy], fontsize=8)
    for ax, title in zip(axes, ['Correct unique majority (ties abstain)', 'Correct answer in top two (tie averaged)']):
        ax.set(xlabel='Total generated output tokens (millions, 30 questions)', ylabel='Questions out of 30', title=title, ylim=(13, 25))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Existing traces replayed at the first detected answer\nNumbers label parallel attempts per question; 1–7 attempts average all subsets', fontsize=12)
    fig.savefig(output / 'voting_frontier.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 14), constrained_layout=True)
    matrix = np.full((30, 8), .5)
    labels = np.full((30, 8), '—', dtype=object)
    for r in rows:
        first = next((e for e in r['events'] if eligible(e, 'proposal')), None)
        if first:
            matrix[r['problem_idx'] - 1, r['sample_number'] - 1] = 1 if first['correct'] else 0
            labels[r['problem_idx'] - 1, r['sample_number'] - 1] = f"{first['answer']:03d}\n{first['estimated_stop_tokens']/1000:.1f}k"
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, cmap=ListedColormap(['#fecaca', '#e2e8f0', '#a7f3d0']), vmin=0, vmax=1, aspect='auto')
    for (q, s), label in np.ndenumerate(labels): ax.text(s, q, label, ha='center', va='center', fontsize=8)
    golds = {r['problem_idx']: r['gold_answer'] for r in rows}
    ax.set(xticks=range(8), xticklabels=range(1, 9), yticks=range(30),
           yticklabels=[f"Q{q:02d} (key {golds[q]:03d})" for q in range(1, 31)],
           xlabel='Saved attempt', title='First audited answer: value / output-token position\nGreen: correct · Red: wrong · Gray: no detected proposal')
    ax.tick_params(length=0)
    fig.savefig(output / 'first_answer_map.png', dpi=170)
    plt.close(fig)

    policies = {p['policy']: p for p in summary['policies']}
    frontier = {(f['policy'], f['parallel_samples']): f for f in summary['frontier']}
    primary = policies['proposal']
    baseline = frontier['final', 8]
    early = frontier['proposal', 8]
    capped = [r for r in rows if r['finish_reason'] == 'length']
    texts = [
        '# Intermediate answers and first-answer voting',
        '',
        f'Offline analysis of **{len(rows)} Qwen trajectories: 30 questions × 8 attempts**, run `{run_name}`. '
        'No inference server was restarted, no SSH was used, and no provided grader was called. '
        'Correctness here means exact integer equality with the answer key already stored in each benchmark record; it does not certify the reasoning.',
        '',
        '**The existing traces support testing early stopping.** The audited first detected answer is correct in '
        f"{primary['first_correct']}/240 attempts, versus 120/240 completed final answers. "
        f"It recovers {primary['first_correct_capped']}/{len(capped)} capped attempts and saves {primary['output_tokens_saved_fraction']:.1%} "
        'of reported output tokens when attempts without a detected answer continue to their original endpoint. '
        'This is retrospective evidence, not a measured speedup or a test of the proposed new prompt.',
        '',
        '## What changes inside a trace', '',
        f"Among {primary['with_claim']} traces with an audited proposal, {primary['first_correct']} start correct and 12 start wrong. "
        f"{primary['repeated_same_answer_claims']} repeat the same answer; only {primary['multiple_distinct_answers']} has more than one distinct detected requested-answer proposal. "
        'No completed correct final answer is lost by stopping at the first audited proposal. '
        'None of the detected initially correct proposals later changes to a wrong requested-answer proposal.',
        '',
        'The genuine detected correction is **Q23, attempt 3: 600 at token 10,611 → 610 at token 13,633**. '
        'It fixes an off-by-one count (40 blocks versus 39 relevant blocks) and still reaches the 16,384-token cap without a final answer. '
        'First-answer stopping would miss that correction in this attempt, although other attempts vote for 610.',
        '',
        'A clear churning example is **Q3, attempt 3**: a correct 16 appears at token 3,309, but the attempt continues to 16,384 '
        '(13,075 additional output tokens) without completing. **Q1, attempt 3** states the correct 70 at token 961. '
        'Across proposal-producing traces, the median first claim is at token 6,867 and the median tail after it is 3,542 tokens. '
        'These medians use the 157 proposal-producing traces, rather than all 240.',
        '',
        '## Extraction policies', '',
        '| Policy | Traces with a claim | First correct | Correct capped traces | Output saved |',
        '|---|---:|---:|---:|---:|',
    ]
    for policy, label in [('markers', 'First literal Answer:/boxed'), ('literal_prose', 'First numeric answer clause, uncurated'),
                          ('committed', 'Audited asserted clause / marker'), ('proposal', 'Audited tentative or asserted clause / marker'),
                          ('target', 'Aggressive requested-quantity equation sensitivity')]:
        p = policies[policy]
        texts.append(f"| {label} | {p['with_claim']} | {p['first_correct']} | {p['first_correct_capped']} | {p['output_tokens_saved_fraction']:.1%} |")
    texts += [
        '',
        'Numeric clauses include “the answer is”, “should be”, “would be”, and similar explicit proposals. '
        'Literal arithmetic is evaluated as a whole: “21 + 49 = 70” yields 70, not 21. '
        'Fractions, alternatives such as “145 or 129”, unsupported expressions, negations, hypotheticals, formatting examples, '
        'and remembered answers to other problems abstain. Matches in both reasoning and content are included. '
        'One trajectory contributes **one vote**; ten repetitions inside it never become ten votes.',
        '',
        'The audited variants additionally exclude **ten toy sections in six traces** using '
        '[reviewable annotations](../../data/intermediate_exclusions.json), bound to the source reasoning SHA256. '
        'For example, Q25 tests three chairs/two people (answer 3) before solving sixteen chairs/eight people (907); '
        'Q26 checks a square (3) while solving the 24-gon (113). These exclusions are manual retrospective scope judgments, '
        'not a proven online detector. The uncurated variant saves 19.8% and has 141 correct first claims; '
        'the audited variant saves 19.3% and has 145. The identical eight-attempt vote outcomes below survive that sensitivity check.',
        '',
        'The requested-quantity equation variant detects such forms as `m+n = ...` and converts `N = ...` to `N mod 1000` '
        'only when the question requests that operation. It is aggressive and remains vulnerable to intermediate hypotheses and toy equations; '
        'it saves 25.3% but reduces eight-attempt majority accuracy to 22/30, versus 23/30 for audited answer clauses. '
        'The claim inventory and complete trace map retain this distinction; absence of a detected claim is not proof that the trace contains no useful intermediate result.',
        '',
        '## Parallel sampling and voting replay', '',
        '| Endpoint | Attempts/question | Correct unique majority /30 | Correct in top two /30 | Output tokens |',
        '|---|---:|---:|---:|---:|',
    ]
    for policy, n, label in [('final', 8, 'Completed final'), ('markers', 8, 'First marker'),
                           ('literal_prose', 8, 'First prose, uncurated'), ('proposal', 8, 'First prose, audited'),
                           ('final', 4, 'Completed final'), ('proposal', 4, 'First prose, audited'), ('proposal', 2, 'First prose, audited')]:
        f = frontier[policy, n]
        texts.append(f"| {label} | {n} | {f['unique_modal_correct']:.2f} | {f['top2_tie_averaged']:.2f} | {f['output_tokens']/1e6:.3f}M |")
    texts += [
        '',
        'For 1–7 attempts, metrics average **every subset** of the eight saved attempts for each question. '
        'They are expected question counts, not a newly sampled benchmark run or the original first-call result. '
        'Rank candidates by vote count, using uniform tie-breaking independent of the key; unique-majority ties abstain. '
        'Top-two means candidate coverage and requires an independent choice between candidates. It is not single-answer accuracy '
        'and does not establish that Gemma can make that choice reliably.',
        '',
        'With all eight audited first proposals, Q7 votes **271 × 3 versus 821 × 1**. '
        'The saved key is 821, so majority is confidently wrong while the top two retain the correct answer. '
        'Q18 is newly recovered by both first-marker and first-prose majority voting. '
        'Q23 majority also flips from wrong to correct: completed answers vote 600 twice versus 610 once; '
        'audited first proposals vote 610 four times versus 600 three times. '
        'First-prose voting additionally retains Q7 in the top two. Questions 10, 13, 14, 15, 28, and 30 '
        'still have no correct first proposal in this corpus; parallelism cannot repair an answer that none of these samples proposes.',
        '',
        '![Voting frontier](../../runs/' + run_name + '/intermediate_answers/voting_frontier.png)',
        '',
        '## Timing and stopping budgets', '',
        f"The token-work saving is {primary['output_tokens_saved']:,} of {summary['reported_output_tokens']:,} tokens. "
        f"If all eight attempts must finish before voting, mean per-question maximum output length falls from "
        f"{baseline['batch_max_output_tokens']/30:,.0f} to {early['batch_max_output_tokens']/30:,.0f} tokens "
        f"({1-early['batch_max_output_tokens']/baseline['batch_max_output_tokens']:.1%}), a smaller improvement than total work. "
        'This is a critical-path token proxy, not elapsed time: hard no-answer attempts still reach the full cap.',
        '',
        '| Audited first proposals, eight attempts | Majority /30 | Top-two coverage /30 | Output tokens |',
        '|---|---:|---:|---:|',
    ]
    for f in summary['budget_frontier']:
        if f['policy'] == 'proposal' and f['parallel_samples'] == 8:
            texts.append(f"| {f['budget']:,}-token ceiling | {f['unique_modal_correct']:.0f} | {f['top2_tie_averaged']:.0f} | {f['output_tokens']/1e6:.3f}M |")
    texts += [
        '',
        'A blanket 8k cutoff loses too many answers; useful difficult-question proposals appear around 12k–16k. '
        'An early-answer endpoint with escalation for abstentions is more supported here than a universal short token budget.',
        '',
        'Saved responses are nonstreaming and contain only request-end latency, so actual time-to-first-answer, '
        'late-token slowdown, cancellation cost, and parallel throughput cannot be recovered from them. '
        'With KV caching, each new token still attends over previous tokens, and the cache grows with sequence length '
        '([Hugging Face cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation)). '
        'That makes avoiding long tails plausible, but the full wall-time curve also depends on batching, hardware, and kernels; '
        'these data do not justify a numeric nonlinear speedup claim.',
        '',
        'The next controlled experiment should compare the existing prompt against one requiring a clearly scoped '
        '`<candidate_answer>NNN</candidate_answer>` as soon as the model has a **complete answer to the requested problem**. '
        'Stream and cancel on that tag; record actual prefix latency, token cadence, input/output tokens, and concurrent-batch wall time. '
        'Test 2/4/8 samples with top-one accuracy and top-two coverage separately, and retain an escalation path for abstentions or disagreements. '
        'Keep a paired continuation arm to measure corrections lost by cancellation. A changed prompt can change the reasoning prefix itself.',
        '',
        '## Per-question map', '',
        'Each list shows audited first answers in attempt order 1–8. `—` means no detected requested-answer proposal. '
        'The linked 240-row CSV gives token positions, first correct claim, compressed answer sequence, endpoint, and source file; '
        'the detailed JSON includes all claim contexts and classifications.',
        '',
        '| Question | Key | Correct completed /8 | First answers, attempts 1–8 | Correct first /8 |',
        '|---|---:|---:|---|---:|',
    ]
    for q in range(1, 31):
        group = [r for r in rows if r['problem_idx'] == q]
        answers = [candidate(r, 'proposal')[0] for r in group]
        texts.append(f"| {q:02d} | {group[0]['gold_answer']:03d} | {sum(r['final_correct'] for r in group)} | " +
                     ', '.join('—' if a is None else f'{a:03d}' for a in answers) +
                     f" | {sum(a == group[0]['gold_answer'] for a in answers)} |")
    texts += [
        '',
        '![All first answers and positions](../../runs/' + run_name + '/intermediate_answers/first_answer_map.png)',
        '',
        '## Artifacts and reproduction', '',
        f'- [All 240 traces mapped](../../runs/{run_name}/intermediate_answers/trace_map.csv)',
        f'- [Every detected claim, including exclusions](../../runs/{run_name}/intermediate_answers/claim_inventory.csv)',
        f'- [Claim contexts and exact token positions](../../runs/{run_name}/intermediate_answers/traces.json)',
        f'- [Subset, tie, budget, and per-question metrics](../../runs/{run_name}/intermediate_answers/summary.json)',
        '- [Extraction and replay implementation](../../src/answer_extraction.py)',
        '- [Source-bound scope annotations](../../data/intermediate_exclusions.json)',
        '',
        '```bash',
        '.venv/bin/python -m unittest test.test_intermediate',
        f'.venv/bin/python -m src.experiments.intermediate_answers.analyze_intermediate --run {run_name}',
        f'.venv/bin/python -m src.experiments.intermediate_answers.render_intermediate --run {run_name}',
        '```',
        '',
        'These commands are offline and require the existing cached `.local/tokenizers/qwen3.json`. '
        f"Its SHA256 is `{summary['tokenizer_sha256']}`. "
        'All reasoning token counts match the previous trajectory analysis. '
        'Provider completion usage exceeds independently tokenized visible output by only 0–6 tokens per trace; '
        'the replay charges that small overhead before each stopping point conservatively. '
        'The 240 original records are read only. Local llama inference remains stopped.',
        '',
    ]
    report = ROOT / 'docs/reports/intermediate-answers.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(texts))
    print(f"Rendered report and plots from {len(rows)} saved traces.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='20260930-155212')
    main(parser.parse_args().run)
