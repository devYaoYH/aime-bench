"""CPU-only full-input grammar checking and bounded expression canonicalization.

SymPy's pinned ANTLR grammar is used without evaluation for syntax. Symbolic
normalization runs in an owned spawned process, never on the stream event loop.
No model code is executed, no answer key is read, and no grader verdict is inferred.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from functools import wraps
import hashlib
from importlib import metadata
import multiprocessing
import re
import time

POLICY = {
    "id": "latex_syntax_expression_keys_v2_1",
    "backend": "SymPy 1.14.0 ANTLR grammar; antlr4-python3-runtime 4.11.0",
    "device": "cpu",
    "syntax_timeout_s": 1.0,
    "canonical_timeout_s": 0.25,
    "max_chars": 4096,
    "max_tokens": 1024,
    "max_delimiter_depth": 64,
    "placeholder_filter": "EXPRESSION, ANSWER, YOUR_ANSWER, ellipsis, question mark",
    "equivalence": "SymPy simplify key plus original symbolic-denominator constraints; tuples ordered, finite sets unordered",
    "fallback": "After syntax success, normalization errors/timeouts retain a raw-string key. Syntax errors/timeouts reject with evidence.",
}


class InvalidMath(ValueError):
    pass


def _parts(text, separator=','):
    """Split a math list only at top level, rejecting mismatched delimiters."""
    stack, parts, start = [], [], 0
    pairs = {')': '(', ']': '[', '}': '{'}
    for i, char in enumerate(text):
        if char in '([{':
            stack.append(char)
            if len(stack) > POLICY['max_delimiter_depth']:
                raise InvalidMath('delimiter_depth_limit')
        elif char in ')]}':
            if not stack or stack.pop() != pairs[char]:
                raise InvalidMath('mismatched_delimiter')
        elif char == separator and not stack:
            parts.append(text[start:i].strip())
            start = i + 1
    if stack:
        raise InvalidMath('unclosed_delimiter')
    parts.append(text[start:].strip())
    if any(not p for p in parts):
        raise InvalidMath('empty_expression')
    return parts


def _unwrap(text):
    text = text.strip()
    for left, right in (('$$', '$$'), ('$', '$'), (r'\(', r'\)'), (r'\[', r'\]')):
        if text.startswith(left) and text.endswith(right):
            return text[len(left):-len(right)].strip()
    return text


def _encloses(text, left, right):
    if not text.startswith(left) or not text.endswith(right):
        return False
    depth = 0
    for index in range(len(left)-1, len(text)):
        if text[index] == left[-1]:
            depth += 1
        elif text[index] == right[-1]:
            depth -= 1
            if depth == 0:
                return index == len(text)-1
    return False


def _grammar_tree(text, lexer_type, parser_type, antlr, listener):
    text = _unwrap(text)
    if not text:
        raise InvalidMath('empty_expression')
    # Typesetting wrappers cannot make an instruction placeholder an answer.
    probe = re.sub(r'\\(?:mathrm|mathit|text)\s*\{([^{}]*)\}', r'\1', text)
    if re.search(r'(?i)(?<![A-Za-z])(?:EXPRESSION|ANSWER|YOUR_ANSWER)(?![A-Za-z])|\.\.\.|…|\?', probe):
        raise InvalidMath('placeholder')
    # English prose is otherwise legal implicit letter multiplication in ANTLR.
    # Reject multi-word prose, while retaining abc, x y and TeX named symbols.
    uncommanded = re.sub(r'\\[A-Za-z]+(?:\{[^{}]*\})?', '', text)
    if re.search(r'\b[A-Za-z]{3,}\s+[A-Za-z]{2,}\b', uncommanded):
        raise InvalidMath('prose')
    if text in (r'\emptyset', r'\varnothing'):
        return ('set', [])
    # Finite sets and tuples are compositions of grammar-checked expressions.
    if _encloses(text, r'\{', r'\}'):
        if not text[2:-2].strip():
            return ('set', [])
        return ('set', [_grammar_tree(p, lexer_type, parser_type, antlr, listener)
                        for p in _parts(text[2:-2])])
    parts = _parts(text)
    if len(parts) > 1:
        return ('tuple', [_grammar_tree(p, lexer_type, parser_type, antlr, listener) for p in parts])
    if _encloses(text, '(', ')'):
        parts = _parts(text[1:-1])
        if len(parts) > 1:
            return ('tuple', [_grammar_tree(p, lexer_type, parser_type, antlr, listener) for p in parts])
    lexer = lexer_type(antlr.InputStream(text))
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)
    tokens = antlr.CommonTokenStream(lexer)
    tokens.fill()  # Force lexical errors even in unconsumed suffixes.
    if len(tokens.tokens) > POLICY['max_tokens']:
        raise InvalidMath('token_limit')
    parser = parser_type(tokens)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)
    parser.math()
    if parser.getCurrentToken().type != antlr.Token.EOF:
        raise InvalidMath('unconsumed_suffix')
    return ('scalar', text)


def _canonical(tree, sympy, parse_latex):
    kind, body = tree
    if kind != 'scalar':
        keys = [_canonical(item, sympy, parse_latex) for item in body]
        if kind == 'set':
            keys = sorted(set(keys))
        return kind + ':' + repr(keys)
    expr = parse_latex(body, strict=True, backend='antlr')
    # Preserve excluded points: x/x and 1 must not collapse without an x != 0
    # assumption. Equivalence of sets of constraints is conservative, not solved.
    constraints = sorted({sympy.srepr(sympy.simplify(node.base))
                          for node in sympy.preorder_traversal(expr)
                          if isinstance(node, sympy.Pow) and node.exp.is_negative is True
                          and node.base.free_symbols})
    return kind + ':' + sympy.srepr(sympy.simplify(expr)) + ':nonzero=' + repr(constraints)


def _worker(connection):
    """Two-stage reply permits safe raw-key fallback after syntax succeeds."""
    try:
        if metadata.version('sympy') != '1.14.0' or metadata.version('antlr4-python3-runtime') != '4.11.0':
            raise RuntimeError('Install the pinned core_v2_1/requirements-syntax.txt')
        import antlr4
        import sympy
        from antlr4.error.ErrorListener import ErrorListener
        from sympy.parsing.latex._antlr.latexlexer import LaTeXLexer
        from sympy.parsing.latex._antlr.latexparser import LaTeXParser
        from sympy.parsing.latex import parse_latex

        class StrictListener(ErrorListener):
            def syntaxError(self, recognizer, offendingSymbol, line, column, msg, exc):
                raise InvalidMath('invalid_latex')

        listener = StrictListener()
        for example in ('1/2', r'\frac{1}{2}', r'\sqrt{2}', 'x+x', '(x^2-1)/(x-1)'):
            tree = _grammar_tree(example, LaTeXLexer, LaTeXParser, antlr4, listener)
            _canonical(tree, sympy, parse_latex)  # Warm imports/grammar outside timing.
        connection.send({'ready': True})
        while True:
            answer = connection.recv()
            began = time.process_time()
            try:
                tree = _grammar_tree(answer, LaTeXLexer, LaTeXParser, antlr4, listener)
            except (InvalidMath, RecursionError) as exc:
                connection.send({'stage': 'syntax', 'valid': False, 'reason': str(exc),
                                 'syntax_cpu_s': time.process_time() - began})
                continue
            connection.send({'stage': 'syntax', 'valid': True, 'syntax_cpu_s': time.process_time() - began})
            began = time.process_time()
            try:
                canonical = _canonical(tree, sympy, parse_latex)
                result = {'canonical_key': 'expr:' + hashlib.sha256(canonical.encode()).hexdigest(),
                          'canonical_form': canonical, 'canonical_fallback': None}
            except Exception as exc:
                result = {'canonical_key': 'raw:' + hashlib.sha256(answer.encode()).hexdigest(),
                          'canonical_form': None, 'canonical_fallback': type(exc).__name__}
            connection.send({'stage': 'canonical', **result,
                             'canonical_cpu_s': time.process_time() - began})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({'ready': False, 'error': f'{type(exc).__name__}: {exc}'})
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ExpressionValidator:
    """One shared warm CPU worker per attempt, serialized and cached."""
    def __init__(self):
        self.lock = asyncio.Lock()
        self.process = self.connection = None
        self.cache = OrderedDict()

    async def _receive(self, timeout):
        if not await asyncio.to_thread(self.connection.poll, timeout):
            raise TimeoutError
        return self.connection.recv()

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None
        if self.connection:
            self.connection.close()
            self.connection = None

    async def start(self):
        if self.process is not None:
            return
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        try:
            message = await self._receive(10)
            if not message.get('ready'):
                raise RuntimeError(message.get('error', 'Syntax worker startup failed'))
        except BaseException:
            self.close()
            raise

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        self.close()

    async def validate(self, answer):
        began = time.perf_counter()
        async with self.lock:
            queue_s = time.perf_counter() - began
            if answer in self.cache:
                value = dict(self.cache[answer], cache_hit=True, queue_s=queue_s)
                self.cache.move_to_end(answer)
                value.update(syntax_cpu_s=0, canonical_cpu_s=0, validation_wall_s=time.perf_counter()-began)
                return value
            if len(answer) > POLICY['max_chars']:
                return {'valid': False, 'reason': 'character_limit', 'queue_s': queue_s,
                        'validation_wall_s': time.perf_counter()-began, 'cache_hit': False}
            await self.start()
            self.connection.send(answer)
            try:
                result = await self._receive(POLICY['syntax_timeout_s'])
                if result.get('stage') != 'syntax':
                    raise RuntimeError('Syntax worker protocol failure')
                if result['valid']:
                    try:
                        canonical = await self._receive(POLICY['canonical_timeout_s'])
                        if canonical.get('stage') != 'canonical':
                            raise RuntimeError('Canonical worker protocol failure')
                        result.update(canonical)
                    except (TimeoutError, EOFError):
                        self.close()
                        result.update(canonical_key='raw:' + hashlib.sha256(answer.encode()).hexdigest(),
                                      canonical_form=None, canonical_fallback='canonical_timeout_or_crash')
            except (TimeoutError, EOFError):
                self.close()
                result = {'valid': False, 'reason': 'syntax_timeout_or_crash'}
            except BaseException:
                self.close()
                raise
            result.update(cache_hit=False, queue_s=queue_s, validation_wall_s=time.perf_counter()-began)
            self.cache[answer] = dict(result)
            if len(self.cache) > 4096:
                self.cache.popitem(last=False)
            return result


def with_validator(function):
    """Standalone scheduling/stream tests also own and clean their validator."""
    @wraps(function)
    async def wrapped(*args, syntax_validator=None, **kwargs):
        if syntax_validator is not None:
            return await function(*args, syntax_validator=syntax_validator, **kwargs)
        async with ExpressionValidator() as validator:
            return await function(*args, syntax_validator=validator, **kwargs)
    return wrapped
