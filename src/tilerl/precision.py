"""Precision policy: the one place a dtype is chosen. Served weights keep the
checkpoint format, compute precision belongs to the kernel cell, accumulating
state defaults to fp32; narrowing one lands here with a gate, never at a call site."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_TABLE = {
    "optimizer_state": torch.float32,
    # ISO frames: fp32 is 200 GiB on the 27B; bf16 flips only once Newton-Schulz
    # is measured to keep them orthonormal there.
    "frame": torch.float32,
    "adapter": torch.bfloat16,
}


#: element size in bytes of the scale dtype named in a Format (docs/design-cost-model.md)
_SCALE_ITEMSIZE = {"e4m3": 1, "f32": 4, "bf16": 2}


@dataclass(frozen=True)
class Format:
    """A tensor's storage: element bits plus zero or more scale planes.

    A scale plane is ``(group, dtype)``: one scale of ``dtype`` per ``group``
    elements along the last axis; ``group=None`` is one scale for the whole
    tensor. SHIM pending cc's task-A PR (same names/formula as
    docs/design-cost-model.md); rebase deletes this copy.
    """

    bits: int
    scales: tuple[tuple[int | None, str], ...] = ()


#: Formats the cost model names. The fp8 KV scale plane is one f32 per head per
#: token, i.e. a scale per ``head_dim`` elements — so its group is the model's
#: head_dim, not a fixed constant: :func:`kv_format` builds it per model. At the
#: 27B's 16 planes x 4 heads x 256, kv_format(256) gives 32 KiB + 512 B/token,
#: matching PagedKvPool.bytes_per_token.
bf16 = Format(bits=16)
f32 = Format(bits=32)
nvfp4 = Format(bits=4, scales=((16, "e4m3"), (None, "f32")))


def kv_format(head_dim: int) -> Format:
    """The fp8 KV plane format for a model: 8-bit values plus one f32 scale per
    ``head_dim`` elements (one per plane x head x token)."""
    return Format(bits=8, scales=((head_dim, "f32"),))


def nbytes(fmt: Format, shape: tuple[int, ...]) -> int:
    """Bytes ``shape`` occupies under ``fmt``: payload plus every scale plane.

    A scale plane is one scale per ``group`` elements (``numel // group``), or a
    single scalar for the whole tensor when ``group is None`` (the nvfp4 global
    reciprocal scale).
    """
    numel = 1
    for d in shape:
        numel *= d
    total = numel * fmt.bits // 8
    for group, dtype in fmt.scales:
        itemsize = _SCALE_ITEMSIZE[dtype]
        total += itemsize if group is None else (numel // group) * itemsize
    return total


def dtype(role: str, device: Any = None) -> torch.dtype:
    if role == "recurrent_state":
        # sm90's fused GDN kernel is f32-IO (a bf16 pool cost two casts per layer per tick);
        # CPU and metal cast at the boundary, so bf16 storage is free there.
        return torch.float32 if getattr(device, "type", device) == "cuda" else torch.bfloat16
    return _TABLE[role]


def roles() -> tuple[str, ...]:
    return tuple(_TABLE) + ("recurrent_state",)
