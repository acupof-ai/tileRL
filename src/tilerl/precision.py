"""Precision policy: the one place a dtype is chosen. Served weights keep the
checkpoint format, compute precision belongs to the kernel cell, accumulating
state defaults to fp32; narrowing one lands here with a gate, never at a call site."""

from __future__ import annotations

from typing import Any

import torch

_TABLE = {
    "optimizer_state": torch.float32,
    # ISO frames: fp32 is 200 GiB on the 27B; bf16 flips only once Newton-Schulz
    # is measured to keep them orthonormal there.
    "frame": torch.float32,
    "adapter": torch.bfloat16,
}


def dtype(role: str, device: Any = None) -> torch.dtype:
    if role == "recurrent_state":
        # sm90's fused GDN kernel is f32-IO (a bf16 pool cost two casts per layer per tick);
        # CPU and metal cast at the boundary, so bf16 storage is free there.
        return torch.float32 if getattr(device, "type", device) == "cuda" else torch.bfloat16
    return _TABLE[role]


def roles() -> tuple[str, ...]:
    return tuple(_TABLE) + ("recurrent_state",)
