"""Tokenizer facade: the `tokenizers` package when a checkpoint is configured,
a byte-level fallback otherwise. No torch, no web stack — training and eval
need this without the server extra installed."""

from __future__ import annotations

import os
from typing import Any, Protocol

__all__ = ["ByteTokenizer", "Tokenizer", "get_tokenizer"]


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """Fallback tokenizer: utf-8 bytes, 256 vocab. Lossless, no files.

    Token ids are 0..255, so any model with ``vocab_size >= 256`` (tiny is
    320) can be served with no checkpoint present.
    """

    vocab_size = 256
    stop_token_ids: tuple[int, ...] = ()

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


class _HfTokenizerAdapter:
    """Adapts a `tokenizers.Tokenizer` to the facade contract."""

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
    """Load a HF tokenizer from a hub id or local directory.

    A random-weight tiny model needs no checkpoint, so ``source=None`` uses
    :class:`ByteTokenizer`. A configured checkpoint fails closed.
    """
    if source:
        from tokenizers import Tokenizer as HfTokenizer

        if os.path.isdir(source):
            tok = HfTokenizer.from_file(os.path.join(source, "tokenizer.json"))
        else:
            tok = HfTokenizer.from_pretrained(source)
        return _HfTokenizerAdapter(tok)
    return ByteTokenizer()


def render_chat(messages: list[tuple[str, str]], thinking: bool | None = None) -> str:
    """ChatML — the format Qwen3.x was trained on and what the stop set
    assumes. ``messages`` are ``(role, text)`` pairs; the assistant turn is
    left open for the model.

    ``thinking`` follows the 27B checkpoint's own template: True opens the
    reasoning block (``<think>\n``) for the model to fill, False closes an
    empty one in the prompt (``<think>\n\n</think>\n\n``) — thinking is
    switched in the prompt, not by forcing tokens at decode time. None leaves
    the turn at ``assistant\n`` (the tiny/dev path)."""
    rendered = "".join(f"<|im_start|>{r}\n{t}<|im_end|>\n" for r, t in messages)
    tail = {None: "", True: "<think>\n", False: "<think>\n\n</think>\n\n"}[thinking]
    return f"{rendered}<|im_start|>assistant\n{tail}"
