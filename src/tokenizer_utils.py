"""Recover exact source-preserving tokenizer prefixes for reasoning analysis.

Use prefix_for when a judging experiment must show exactly N original reasoning
tokens without decoding changes. It validates token count and source alignment,
falling back to tokenizer offsets when needed. QWEN_TOKENIZER_REPO identifies the
original baseline tokenizer for both Jev prefixes and escalation analysis.
This library performs no downloads or inference; callers supply a tokenizer.
"""
from tokenizers import Tokenizer

QWEN_TOKENIZER_REPO = "Qwen/Qwen3-30B-A3B"


def prefix_for(text: str, tokenizer: Tokenizer, limit: int) -> tuple[str, int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) < limit:
        raise ValueError(f"Reasoning trace has only {len(encoded.ids)} tokens")
    prefix = tokenizer.decode(encoded.ids[:limit], skip_special_tokens=False)
    if not text.startswith(prefix):
        # Preserve original bytes, using the tokenizer's final offset.
        prefix = text[: encoded.offsets[limit - 1][1]]
    if not text.startswith(prefix) or len(tokenizer.encode(prefix, add_special_tokens=False).ids) != limit:
        raise ValueError(f"Unable to recover an exact {limit}-token prefix")
    return prefix, len(encoded.ids)
