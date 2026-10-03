"""Extract and summarize answer proposals from saved reasoning without inference.

Use this library when a streaming detector or offline analysis needs literal
answer clauses, requested-quantity transforms, policy selection, or vote replay.
Extraction uses syntax and restricted rational arithmetic without an answer key;
saved-key comparisons and source-bound scope annotations enter retrospective
trace analysis separately. make_rows reads local benchmark/self-consistency JSON.
For the fixed 240-trace experiment, run the intermediate_answers analyzer.
"""
from __future__ import annotations

import ast
from bisect import bisect_left
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
import os
import re
import statistics

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.common import ROOT
POLICIES = ('final', 'markers', 'literal_prose', 'committed', 'proposal', 'target')
BUDGETS = (2000, 4000, 8000, 12000, 16384)

BOX = re.compile(r'\\boxed\s*\{\s*(\d{1,3})\s*\}')
ANSWER_LINE = re.compile(r'(?im)^\s*(?:\*\*)?Answer\s*:\s*\$?\s*(\d{1,3})\s*\$?\s*(?:\*\*)?\s*[.。]?\s*(?=\n|$)')
PROSE = re.compile(r'(?i)\b(?:final\s+)?answer\s*(?P<verb>is|would\s+be|should\s+be|must\s+be|will\s+be|might\s+be|could\s+be|equals|=|:)\s*(?:indeed\s+|therefore\s+|probably\s+|likely\s+|correct\s*:\s*)?')


def evaluate_arithmetic(text):
    """Evaluate only small, literal rational arithmetic, never executable code."""
    text = text.strip().replace('−', '-').replace('×', '*').replace('÷', '/')
    text = re.sub(r'\\(?:times|cdot)', '*', text)
    text = re.sub(r'\\(?:bmod|pmod|mod)\b|\bmod(?:ulo)?\b', '%', text)
    text = text.replace('^', '**')
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    if len(text) > 150:
        return None
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError:
        return None

    def go(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 10**12:
                raise ValueError('large literal')
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = go(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            a, b = go(node.left), go(node.right)
            if isinstance(node.op, ast.Add): result = a + b
            elif isinstance(node.op, ast.Sub): result = a - b
            elif isinstance(node.op, ast.Mult): result = a * b
            elif isinstance(node.op, ast.Div): result = a / b
            elif isinstance(node.op, ast.Mod): result = a % b
            elif isinstance(node.op, ast.Pow) and b.denominator == 1 and 0 <= b <= 12: result = a ** int(b)
            else: raise ValueError('unsupported arithmetic')
            if abs(result.numerator) > 10**15 or result.denominator > 10**12:
                raise ValueError('large result')
            return result
        raise ValueError('nonliteral expression')
    try:
        value = go(tree.body)
        return int(value) if value.denominator == 1 else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def numeric_prefix(text, start):
    """Read an entire numeric expression, avoiding the first operand in a sum.

    Returns the integer and the offset at which its expression is delimited.
    Fractional/ambiguous output expressions abstain; no answer-key-based choices.
    """
    chunk = text[start:start + 220]
    wrapper = re.match(r'\s*(?:\$+|\\\(|\\\[|\*\*)?\s*', chunk)
    skip = wrapper.end()
    boxed = BOX.match(chunk, skip)
    if boxed:
        return int(boxed[1]), start + boxed.end(), boxed[0]
    # LaTeX rational literals, including a terminal integer equality.
    chunk = chunk[skip:]
    leading = re.match(r'[+\-−]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\(\s*\d+)', chunk)
    if not leading:
        return None
    expression = re.match(r'(?:\d|[+\-−*/×÷^%=().\s]|,(?=\d{3}(?:\D|$))|\\(?:times|cdot|bmod|mod)\b|\bmod(?:ulo)?\b)+', chunk)
    if not expression:
        return None
    raw = expression[0].rstrip()
    # Sentence punctuation belongs to the delimiter, not the arithmetic.
    raw = raw.rstrip('.')
    if not raw:
        return None
    tail = chunk[len(raw):]
    if re.match(r'\s*(?:\\sqrt|√|[⁰¹²³⁴⁵⁶⁷⁸⁹]|[a-zA-Z]\s*[+*/^=]|/|,?\s*or\b)', tail):
        return None
    rhs = raw.split('=')[-1].strip()
    # Padded AIME literals such as 016 are decimal, not Python numeric syntax.
    value = int(rhs) if re.fullmatch(r'[+\-]?\d{1,12}', rhs) else evaluate_arithmetic(rhs)
    if value is None:
        return None
    end = start + skip + len(raw)
    # Wait until the next boundary is visible, so "7" does not stop "70".
    if end < len(text):
        end += 1
    return value, end, raw


def classify_context(text, start, verb='', kind='prose'):
    boundary = max(text.rfind('\n', 0, start), text.rfind('. ', 0, start), text.rfind('? ', 0, start))
    prefix = text[max(boundary + 1, start - 180):start]
    lower = prefix.lower()
    if re.search(r'\b(?:not|isn.t|is not)\s+(?:the\s+)?$', lower):
        return 'negated'
    if re.search(r'\b(?:if|for example|for instance|suppose|assuming|say)\b', lower):
        return 'hypothetical'
    if re.search(r'\b(?:format|written|write|entered|instruction|user says|user said)\b', lower):
        return 'format_discussion'
    if re.search(r'\b(?:some similar problems|other problems)\b', lower):
        return 'other_problem'
    if kind == 'target':
        return 'target'
    if re.search(r'\b(?:think|guess|maybe|might|probably|likely|expect|suspect|seems|believe)\b', lower) or verb.lower() in ['would be', 'should be', 'might be', 'could be']:
        return 'tentative'
    return 'asserted'


def target_rule(problem):
    """Optional aggressive sensitivity policy using the requested output label."""
    remainder = re.search(r'remainder when\s+\$([^$]+)\$\s+is divided by\s+\$?1000', problem, re.I)
    if remainder:
        return remainder[1].rstrip('.'), 'mod1000'
    difference = re.search(r'difference between\s+\$?(N)\$?\s+and\s+2025', problem, re.I)
    if difference:
        return difference[1], 'minus2025'
    matches = list(re.finditer(r'\bFind\s+\$([^$]+)\$', problem, re.I))
    if matches:
        label = matches[-1][1].rstrip('.').strip()
        if re.fullmatch(r'[a-zA-Z0-9+^{}\s\\.]+', label):
            return label, 'identity'
    return None


def extract_events(text, problem='', excluded_spans=()):
    events = []
    for pattern, kind in [(BOX, 'boxed'), (ANSWER_LINE, 'answer_line')]:
        for m in pattern.finditer(text):
            events.append({'answer': int(m[1]), 'start': m.start(), 'end': m.end(),
                           'kind': kind, 'confidence': classify_context(text, m.start(), kind=kind),
                           'expression': m[1]})
    for m in PROSE.finditer(text):
        parsed = numeric_prefix(text, m.end())
        if not parsed:
            continue
        answer, end, expression = parsed
        if not 0 <= answer <= 999:
            continue
        confidence = classify_context(text, m.start(), m['verb'])
        if re.search(r'\bnot\b', text[m.end():min(end, m.end() + 20)], re.I):
            confidence = 'negated'
        events.append({'answer': answer, 'start': m.start(), 'end': end, 'kind': 'prose',
                       'confidence': confidence, 'expression': expression})
    rule = target_rule(problem)
    if rule:
        label, transform = rule
        pattern = r'\s*'.join(re.escape(c) for c in re.sub(r'\s+', '', label))
        lhs = re.compile(r'(?<![A-Za-z])' + pattern + r'(?![A-Za-z])\s*(?:=|\bis\b|\bequals\b)\s*')
        for m in lhs.finditer(text):
            parsed = numeric_prefix(text, m.end())
            if not parsed:
                continue
            answer, end, expression = parsed
            if transform == 'mod1000': answer %= 1000
            elif transform == 'minus2025': answer -= 2025
            if 0 <= answer <= 999:
                events.append({'answer': answer, 'start': m.start(), 'end': end, 'kind': 'target_equation',
                               'confidence': classify_context(text, m.start(), kind='target'),
                               'expression': expression, 'target_label': label, 'transform': transform})
    # If a prose clause contains a box, preserve the earlier clause endpoint but
    # do not count the same literal box twice as independent answer claims.
    deduped = []
    priority = {'boxed': 0, 'answer_line': 1, 'prose': 2, 'target_equation': 3}
    for event in sorted(events, key=lambda e: (e['end'], priority[e['kind']], e['start'])):
        if any(e['answer'] == event['answer'] and max(e['start'], event['start']) < min(e['end'], event['end']) for e in deduped[-4:]):
            continue
        event['quote'] = text[event['start']:event['end']].strip()
        event['context'] = text[max(0, event['start'] - 100):min(len(text), event['end'] + 100)]
        if any(span['start'] <= event['start'] < span['end'] for span in excluded_spans):
            event['confidence'] = 'toy_example'
        deduped.append(event)
    return deduped


def eligible(event, policy):
    confidence = event['confidence']
    if event['kind'] in ['boxed', 'answer_line']:
        return True
    if confidence in ['hypothetical', 'negated', 'format_discussion', 'other_problem']:
        return False
    if policy == 'literal_prose': return event['kind'] != 'target_equation'
    if confidence == 'toy_example': return False
    if policy == 'markers': return event['kind'] in ['boxed', 'answer_line']
    if policy == 'committed': return event['kind'] in ['boxed', 'answer_line'] or confidence == 'asserted'
    if policy == 'proposal': return event['kind'] != 'target_equation'
    if policy == 'target': return True
    raise ValueError(policy)


def candidate(row, policy, budget=None):
    if policy == 'final':
        answer, cost = row['final_candidate'], row['completion_tokens']
    else:
        event = next((e for e in row['events'] if eligible(e, policy)), None)
        answer = event['answer'] if event else None
        cost = event['estimated_stop_tokens'] if event else row['completion_tokens']
    if budget is not None:
        if cost > budget: answer = None
        cost = min(cost, budget)
    return answer, cost


def rank_statistics(answers, gold, k):
    counts = Counter(a for a in answers if a is not None)
    if not counts or gold not in counts:
        return {'expected': 0.0, 'guaranteed': False, 'possible': False}
    better = sum(count > counts[gold] for count in counts.values())
    tied = sum(count == counts[gold] for count in counts.values())
    slots = max(0, min(tied, k - better))
    return {'expected': slots / tied, 'guaranteed': slots == tied, 'possible': slots > 0}


def replay(rows, policy, n, budget=None):
    by_question = defaultdict(list)
    for row in rows: by_question[row['problem_idx']].append(row)
    totals = Counter()
    question_rows = []
    for index, group in sorted(by_question.items()):
        group.sort(key=lambda r: r['sample_number'])
        local = Counter()
        count = 0
        for subset in combinations(group, n):
            count += 1
            choices = [candidate(r, policy, budget) for r in subset]
            answers = [c[0] for c in choices]
            votes = Counter(a for a in answers if a is not None)
            leaders = [a for a, v in votes.items() if v == max(votes.values())] if votes else []
            gold = group[0]['gold_answer']
            local['unique_modal_correct'] += len(leaders) == 1 and leaders[0] == gold
            local['top1_tie_averaged'] += rank_statistics(answers, gold, 1)['expected']
            top2 = rank_statistics(answers, gold, 2)
            local['top2_tie_averaged'] += top2['expected']
            local['top2_guaranteed'] += top2['guaranteed']
            local['top2_possible'] += top2['possible']
            local['oracle_any_correct'] += gold in answers
            local['questions_with_votes'] += bool(votes)
            local['modal_tie'] += len(leaders) > 1
            local['output_tokens'] += sum(c[1] for c in choices)
            local['prompt_tokens'] += sum(r['prompt_tokens'] for r in subset)
            local['batch_max_output_tokens'] += max(c[1] for c in choices)
        averaged = {key: value / count for key, value in local.items()}
        averaged['problem_idx'] = index
        question_rows.append(averaged)
        totals.update({key: value for key, value in averaged.items() if key != 'problem_idx'})
    return {'policy': policy, 'parallel_samples': n, 'budget': budget, 'questions': len(by_question),
            'model_requests': len(by_question) * n, **dict(totals), 'by_question': question_rows}


def make_rows(run, tokenizer):
    paths = sorted((run / 'questions').glob('*.json')) + sorted((run / 'self_consistency/questions').glob('*/*.json'))
    previous_path = run / 'analysis/trajectory_metrics.json'
    previous = {(r['problem_idx'], r['sample_number']): r for r in json.loads(previous_path.read_text())['rows']} if previous_path.exists() else {}
    rows = []
    annotation_path = ROOT / 'data/intermediate_exclusions.json'
    annotations = json.loads(annotation_path.read_text())['traces'] if annotation_path.exists() else {}
    for path in paths:
        record = json.loads(path.read_text())
        if not record.get('response'):
            continue
        message = record['response']['choices'][0]['message']
        parts = [('reasoning', message.get('reasoning') or ''), ('content', message.get('content') or '')]
        annotation = annotations.get(f"{record['problem_idx']:02d}/{record.get('sample_number', 1):02d}", {})
        if annotation and hashlib.sha256(parts[0][1].encode()).hexdigest() != annotation['reasoning_sha256']:
            raise ValueError('Audited exclusions do not match the source trace')
        total, events, reasoning_tokens = 0, [], 0
        for part, text in parts:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if part == 'reasoning': reasoning_tokens = len(encoded.ids)
            ends = [offset[1] for offset in encoded.offsets]
            for event in extract_events(text, record['problem'], annotation.get('spans', []) if part == 'reasoning' else ()):
                event['part'] = part
                event['visible_output_token'] = total + bisect_left(ends, event['end']) + 1
                events.append(event)
            total += len(encoded.ids)
        usage = record.get('usage') or record['response'].get('usage') or {}
        completion = usage['completion_tokens']
        overhead = completion - total
        index, sample = record['problem_idx'], record.get('sample_number', 1)
        if (index, sample) in previous and previous[index, sample]['reasoning_tokens'] != reasoning_tokens:
            raise ValueError('Reasoning token count changed from existing analysis')
        for event in events:
            event['estimated_stop_tokens'] = min(completion, event['visible_output_token'] + max(overhead, 0))
            # Evaluation occurs after the extraction function returns.
            event['correct'] = event['answer'] == int(record['gold_answer'])
        row = {'problem_idx': index, 'sample_number': sample, 'gold_answer': int(record['gold_answer']),
               'final_candidate': int(record['candidate']) if record.get('candidate') is not None else None,
               'final_correct': record.get('correct') is True, 'finish_reason': record.get('finish_reason'),
               'completion_tokens': completion, 'prompt_tokens': usage['prompt_tokens'],
               'visible_tokens': total, 'provider_minus_visible_tokens': overhead,
               'reasoning_tokens': reasoning_tokens, 'api_latency_s': record.get('generation_latency_s'),
               'source_file': str(path.relative_to(ROOT)), 'events': events}
        rows.append(row)
    rows.sort(key=lambda r: (r['problem_idx'], r['sample_number']))
    return rows


def trajectory_summary(rows, policy):
    counts = Counter()
    first_tokens, correct_first_tokens, tails, distinct_counts = [], [], [], []
    for row in rows:
        events = [e for e in row['events'] if eligible(e, policy)]
        counts['traces'] += 1
        if not events:
            counts['no_claim'] += 1
            continue
        first, last = events[0], events[-1]
        distinct = set(e['answer'] for e in events)
        distinct_counts.append(len(distinct))
        counts['with_claim'] += 1
        counts['first_correct'] += first['correct']
        counts['last_claim_correct'] += last['correct']
        counts['ever_correct'] += any(e['correct'] for e in events)
        counts['first_wrong_then_correct_claim'] += not first['correct'] and any(e['correct'] for e in events[1:])
        counts['first_correct_then_wrong_claim'] += first['correct'] and any(not e['correct'] for e in events[1:])
        counts['first_wrong_final_correct'] += not first['correct'] and row['final_correct']
        counts['first_correct_final_wrong'] += first['correct'] and row['final_candidate'] is not None and not row['final_correct']
        counts['first_correct_final_missing'] += first['correct'] and row['final_candidate'] is None
        counts['first_correct_capped'] += first['correct'] and row['finish_reason'] == 'length'
        counts['multiple_distinct_answers'] += len(distinct) > 1
        counts['repeated_same_answer_claims'] += len(events) > len(distinct)
        counts['claims'] += len(events)
        counts['reasoning_first_claims'] += first['part'] == 'reasoning'
        first_tokens.append(first['estimated_stop_tokens'])
        if first['correct']: correct_first_tokens.append(first['estimated_stop_tokens'])
        tails.append(row['completion_tokens'] - first['estimated_stop_tokens'])
    saved = sum(row['completion_tokens'] - candidate(row, policy)[1] for row in rows)
    return {'policy': policy, **dict(counts), 'median_first_claim_tokens': statistics.median(first_tokens) if first_tokens else None,
            'median_first_correct_claim_tokens': statistics.median(correct_first_tokens) if correct_first_tokens else None,
            'median_post_first_tokens': statistics.median(tails) if tails else None,
            'median_distinct_answers': statistics.median(distinct_counts) if distinct_counts else None,
            'output_tokens_saved': saved, 'output_tokens_saved_fraction': saved / sum(r['completion_tokens'] for r in rows)}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
