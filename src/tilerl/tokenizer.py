"""Tokenizer facade: the `tokenizers` package when a checkpoint is configured,
a byte-level fallback otherwise. No torch, no web stack — training and eval
need this without the server extra installed."""

from __future__ import annotations

import os
import sys
from typing import Any, Protocol

QWEN38_SOURCE = os.environ.get("TILERL_QWEN38_SOURCE", "Qwen/Qwen3-27B")

NO_WEIGHTS = (
    "hint: download the checkpoint (or set TILERL_QWEN38_SOURCE to a\n"
    "      local safetensors directory), or use --model tiny."
)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """utf-8 bytes, vocab 256: serves any model with ``vocab_size >= 256`` without a checkpoint."""

    vocab_size = 256
    stop_token_ids: tuple[int, ...] = ()

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


class _HfTokenizerAdapter:
    def __init__(self, tok: Any) -> None:
        self._tok = tok
        self.stop_token_ids = tuple(
            token_id
            for token in ("<|im_end|>", "<|endoftext|>")
            if (token_id := tok.token_to_id(token)) is not None
        )

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def get_tokenizer(source: str | None = None) -> Tokenizer:
    """HF tokenizer from a hub id or local directory; ``None`` is the byte fallback."""
    if source:
        from tokenizers import Tokenizer as HfTokenizer

        if os.path.isdir(source):
            tok = HfTokenizer.from_file(os.path.join(source, "tokenizer.json"))
        else:
            tok = HfTokenizer.from_pretrained(source)
        return _HfTokenizerAdapter(tok)
    return ByteTokenizer()


def qwen38_tokenizer() -> Tokenizer:
    """The 27B tokenizer, with the same hint as its weights: a bare hub id 401s.

    One copy for cli and train; a bare `TILERL_QWEN38_SOURCE` that fails to load
    (401, or a bad path) prints the same checkpoint hint and exits.
    """
    try:
        return get_tokenizer(QWEN38_SOURCE)
    except Exception as exc:
        # HF's 401 body is a dozen lines of auth advice; the first names the cause.
        # Some exceptions (MemoryError) stringify empty, so splitlines() can be [].
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        print(f"error: could not load the Qwen3-27B tokenizer from {QWEN38_SOURCE!r}: "
              f"{first}\n{NO_WEIGHTS}", file=sys.stderr)
        sys.exit(1)
