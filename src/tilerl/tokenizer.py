"""Tokenizer facade: the `tokenizers` package when a checkpoint is configured,
a byte-level fallback otherwise. No torch, no web stack — training and eval
need this without the server extra installed."""

from __future__ import annotations

import os
from typing import Any, Protocol

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


# Qwen3.8-27B model card sampling per thinking mode (non-thinking also wants
# presence_penalty 1.5, which the engine does not have).
SAMPLING = {True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
            False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20}}


def render_chat(messages: list[tuple[str, str]], thinking: bool | None = None) -> str:
    """ChatML from ``(role, text)`` pairs with the assistant turn left open.
    ``thinking`` follows the 27B template: True opens ``<think>``, False closes
    an empty one in the prompt, None leaves the bare turn (tiny/dev path)."""
    rendered = "".join(f"<|im_start|>{r}\n{t}<|im_end|>\n" for r, t in messages)
    tail = {None: "", True: "<think>\n", False: "<think>\n\n</think>\n\n"}[thinking]
    return f"{rendered}<|im_start|>assistant\n{tail}"
