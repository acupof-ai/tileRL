"""Precision policy: the one place a dtype is chosen.

Three rules, in order. (1) Served weights are the checkpoint's own format —
fp4, fp8 or bf16 — and are never converted; the loaders own that, not this
table. (2) Compute precision belongs to the kernel: the registry cell for an
arch picks the tensor-core input, and memory-bound ops keep the weight
format. (3) State that accumulates across steps defaults to fp32. Narrowing
one is a measured decision: it lands with a parity or gradcheck gate and a
wins entry, and it lands here, never at a call site.
"""

from __future__ import annotations

from typing import Any

import torch

#: Roles whose dtype is a choice. Anything not listed is not a choice.
_TABLE = {
    "optimizer_state": torch.float32,
    # ISO singular frames. fp32 is 200 GiB on the 27B; bf16 is the only fit
    # there, and whether Newton-Schulz keeps the frames orthonormal in bf16 is
    # measured on the pod before it flips (docs/design-rl-stack.md §1).
    "frame": torch.float32,
    # LoRA A/B: served alongside bf16 masters, so the adapter is bf16.
    "adapter": torch.bfloat16,
}


def dtype(role: str, device: Any = None) -> torch.dtype:
    """The dtype for ``role``; ``device`` only matters for ``recurrent_state``."""
    if role == "recurrent_state":
        # sm90's fused GDN kernel is f32-IO: a bf16 pool cost two 1.5 MB casts
        # per layer per tick (+1.2 GiB at 16 slots on the 27B). CPU and metal
        # compute in f32 and cast at the boundary, so bf16 storage is free.
        return torch.float32 if getattr(device, "type", device) == "cuda" else torch.bfloat16
    return _TABLE[role]


def roles() -> tuple[str, ...]:
    return tuple(_TABLE) + ("recurrent_state",)
