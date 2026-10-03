"""Detect explicit final-answer blocks safely across streamed chunk boundaries.

Use FinalAnswerDetector when an application needs conservative Answer:/boxed
markers in reasoning or content deltas. Feed channels separately and call finish
only at genuine stream completion. Requiring a Final Answer heading reduces
formatting-example matches at the cost of coverage; detection does not verify
scope or correctness. delta_text normalizes OpenRouter reasoning fields.
"""
import re

HEADING = re.compile(r'(?i)^\s*(?:#{1,6}\s*)?(?:\*\*)?Final Answer\s*:?\s*(?:\*\*)?\s*$')
ANSWER = re.compile(r'(?i)^\s*(?:\*\*)?Answer\s*:\s*\$?\s*(\d{1,3})\s*\$?\s*(?:\*\*)?\s*[.。]?\s*$')
BOX = re.compile(r'^\s*(?:\$\$?|\\\[|\\\()?\s*\\boxed\s*\{\s*(\d{1,3})\s*\}\s*(?:\$\$?|\\\]|\\\))?\s*[.。]?\s*$')
MATH_WRAPPERS = {'$', '$$', r'\[', r'\]', r'\(', r'\)'}
META_CONTEXT = re.compile(r'(?i)\b(?:for example|for instance|hypothetical|suppose|quoted|format|instructions?)\b|^```')


class FinalAnswerDetector:
    """Feed each received delta separately; never treat a chunk boundary as EOF.

    A marker only becomes eligible after its complete line is received. Call
    finish() only at genuine stream EOF/DONE, never after each network chunk.
    Once returned, a candidate remains a provisional vote, not a verified result.
    """
    def __init__(self):
        self.pending = {'reasoning': '', 'content': ''}
        self.offsets = {'reasoning': 0, 'content': 0}
        self.final_block = {'reasoning': False, 'content': False}
        self.previous_lines = {'reasoning': [], 'content': []}
        self.matches = []

    def _line(self, part, line, elapsed_s):
        text = line.strip()
        start = self.offsets[part]
        self.offsets[part] += len(line)
        if not text or text in MATH_WRAPPERS:
            return None
        if HEADING.fullmatch(text):
            self.final_block[part] = not any(META_CONTEXT.search(p) for p in self.previous_lines[part][-1:])
            self.previous_lines[part].append(text)
            return None
        self.previous_lines[part].append(text)
        self.previous_lines[part] = self.previous_lines[part][-3:]
        boxed = BOX.fullmatch(text)
        answer = ANSWER.fullmatch(text)
        if self.final_block[part] and (boxed or answer):
            match = boxed or answer
            result = {'answer': int(match[1]), 'kind': 'boxed' if boxed else 'answer_line',
                      'part': part, 'observed_at_s': elapsed_s, 'line': text,
                      'start': start, 'end': self.offsets[part]}
            self.matches.append(result)
            self.final_block[part] = False
            return result
        self.final_block[part] = False
        return None

    def feed(self, part, text, elapsed_s=None):
        if part not in self.pending: raise ValueError('part must be reasoning or content')
        self.pending[part] += text or ''
        found = []
        while '\n' in self.pending[part]:
            line, rest = self.pending[part].split('\n', 1)
            self.pending[part] = rest
            result = self._line(part, line + '\n', elapsed_s)
            if result: found.append(result)
        return found

    def finish(self, elapsed_s=None):
        found = []
        for part in self.pending:
            if self.pending[part]:
                result = self._line(part, self.pending[part], elapsed_s)
                self.pending[part] = ''
                if result: found.append(result)
        return found


def delta_text(delta):
    """Match OpenRouter reasoning fields without duplicating reasoning_details."""
    reasoning = delta.get('reasoning')
    if not reasoning:
        reasoning = ''.join(d.get('text', '') for d in delta.get('reasoning_details', [])
                            if d.get('type') == 'reasoning.text')
    return {'reasoning': reasoning or '', 'content': delta.get('content') or ''}
