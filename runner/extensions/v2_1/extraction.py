"""Incremental extraction of explicit, complete mathematical answer markers.

No mathematical interpretation or gold comparison takes place here. A balanced
box is a prospective answer only; the vendored grader decides equivalence.
"""
import re


class CandidateDetector:
    """Balanced boxed/fbox expressions and complete standalone Answer lines.

    Each channel has an independent scan cursor and brace state. Token boundaries
    cannot complete a box or a line; natural EOF can complete an Answer line.
    Explicit prose such as 'the answer is' is deliberately not interpreted.
    """

    marker = re.compile(r"\\(?:boxed|fbox)\s*\{")
    line = re.compile(r"(?i)^\s*(?:\*\*)?Answer\s*:\s*(.*?)\s*$")
    max_answer_chars = 4096

    def __init__(self):
        self.text = {"content": "", "reasoning": ""}
        self.cursor = {"content": 0, "reasoning": 0}
        self.box_start = {"content": None, "reasoning": None}
        self.depth = {"content": 0, "reasoning": 0}
        self.escaped = {"content": False, "reasoning": False}
        self.line_start = {"content": 0, "reasoning": 0}

    @classmethod
    def _line_answer(cls, value):
        value = value.strip()
        if value.endswith("**"):
            value = value[:-2].strip()
        for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
                value = value[len(left):-len(right)].strip()
                break
        return value

    def feed(self, part, delta, eof=False):
        self.text[part] += delta
        text, found = self.text[part], []
        cursor = self.cursor[part]
        while cursor < len(text):
            if self.box_start[part] is None:
                match = self.marker.search(text, cursor)
                if match is None:
                    # Retain an incomplete marker (including arbitrarily spaced
                    # opening brace) without rescanning previous complete output.
                    tail = text.rfind("\\", cursor)
                    if tail >= 0:
                        pending = text[tail:]
                        if any(marker.startswith(pending) for marker in (r"\boxed", r"\fbox")) or re.fullmatch(r"\\(?:boxed|fbox)\s*", pending):
                            cursor = tail
                            break
                    cursor = len(text)
                    break
                self.box_start[part] = match.end()
                self.depth[part] = 1
                self.escaped[part] = False
                cursor = match.end()
            else:
                char = text[cursor]
                if char == "\\":
                    self.escaped[part] = not self.escaped[part]
                else:
                    if not self.escaped[part]:
                        if char == "{":
                            self.depth[part] += 1
                        elif char == "}":
                            self.depth[part] -= 1
                    self.escaped[part] = False
                    if self.depth[part] == 0:
                        value = text[self.box_start[part]:cursor].strip()
                        if value and len(value) <= self.max_answer_chars:
                            found.append({"answer": value, "part": part, "kind": "boxed", "end": cursor + 1})
                        self.box_start[part] = None
                cursor += 1
        self.cursor[part] = cursor
        start = self.line_start[part]
        while "\n" in text[start:] or (eof and start < len(text)):
            end = text.find("\n", start)
            end = len(text) if end < 0 else end + 1
            match = self.line.fullmatch(text[start:end].strip())
            if match:
                value = self._line_answer(match[1])
                if value and len(value) <= self.max_answer_chars and not re.search(r"\\(?:boxed|fbox)\b", value):
                    found.append({"answer": value, "part": part, "kind": "answer_line", "end": end})
            start = end
        self.line_start[part] = start
        return found
