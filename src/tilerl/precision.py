"""Precision policy: the one place a dtype is chosen. Served weights keep the
checkpoint format, compute precision belongs to the kernel cell, accumulating
state defaults to fp32; narrowing one lands here with a gate, never at a call site."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
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

#: A scale group: an int groups along the last axis; a tuple groups over TRAILING
#: axes (one entry per axis, None = that whole axis).
Group = int | None | tuple[int | None, ...]


@dataclass(frozen=True)
class Format:
    """A tensor's storage: element bits plus zero or more scale planes.

    A scale plane is ``(group, dtype)``. An int ``group`` puts one scale per
    ``group`` elements along the last axis; a tuple groups over the trailing axes
    (``None`` = that whole axis); ``group=None`` is one scale for the whole tensor.
    """

    bits: int
    scales: tuple[tuple[Group, str], ...] = ()


def _scale_count(shape: tuple[int, ...], group: Group) -> int:
    if group is None:
        return 1
    groups = (group,) if isinstance(group, int) else group
    # groups cover the trailing axes; the leading axes each carry one set.
    leading = list(shape[: len(shape) - len(groups)])
    per = [1 if g is None else -(-d // g) for g, d in zip(groups, shape[-len(groups) :])]
    n = 1
    for d in leading + per:
        n *= d
    return n


#: Plain dtypes a cost names.
bf16 = Format(bits=16)
f32 = Format(bits=32)
#: Disk face: ModelOpt NVFP4 — e4m3 block scale per 16 along K, one f32 per tensor.
nvfp4 = Format(bits=4, scales=((16, "e4m3"), (None, "f32")))
#: Device face of an NVFP4 linear after renorm_fp4_scale: f32 scale per 16 along K
#: and one f32 per output row (the global scale split into a per-row epilogue).
nvfp4_dev = Format(bits=4, scales=((16, "f32"), ((None,), "f32")))
#: Device face of an fp8-e4m3 linear: 1 B/elem plus an f32 [N/128,K/128] block grid
#: and one f32 per output row.
fp8_dev = Format(bits=8, scales=(((128, 128), "f32"), ((None,), "f32")))


def kv_format(head_dim: int) -> Format:
    """The fp8 KV plane format for a model: 8-bit values plus one f32 scale per
    ``head_dim`` elements (one per plane x head x token)."""
    return Format(bits=8, scales=((head_dim, "f32"),))


def nbytes(fmt: Format, shape: tuple[int, ...]) -> int:
    """Bytes ``shape`` occupies under ``fmt``: payload plus every scale plane."""
    numel = 1
    for d in shape:
        numel *= d
    total = numel * fmt.bits // 8
    for group, dtype in fmt.scales:
        total += _scale_count(shape, group) * _SCALE_ITEMSIZE[dtype]
    return total


#: Safetensors suffixes that are a weight's scale sidecars (never standalone rows).
_SCALE_SUFFIX = (".weight_scale", ".weight_global_scale", ".weight_scale_inv", ".weight_scale_2")


def weight_specs(header: dict) -> list[tuple[str, tuple[int, ...], Format]]:
    """Classify a safetensors shard header into the served weights and their device faces.

    ``header`` maps tensor name -> {"shape", ...}; no weight bytes are read. The
    fp4/fp8 split is checkpoint-specific -- a config cannot derive it -- so the
    classification follows the loader's (model.py) dispatch on the tensor names:

    - ``X.weight_packed`` (ModelOpt) or ``X.weight`` + ``X.weight_scale_2``
      (official NVFP4): :data:`nvfp4_dev`, logical shape [N, K] from the [N, K/2]
      packed nibbles.
    - ``X.weight`` + ``X.weight_scale_inv`` (ModelOpt FP8): :data:`fp8_dev`.
    - any other ``X.weight`` / plain tensor: its stored dtype (bf16 on 27B).

    The scale sidecars are priced by the Format and are not separate rows.
    """
    names = set(header) - {"__metadata__"}
    out: list[tuple[str, tuple[int, ...], Format]] = []

    def plain(name: str) -> tuple[str, tuple[int, ...], Format]:
        dt = str(header[name].get("dtype", "BF16")).upper()
        bits = {
            "F64": 64,
            "F32": 32,
            "F16": 16,
            "BF16": 16,
            "F8_E4M3FN": 8,
            "I64": 64,
            "I32": 32,
            "I16": 16,
            "I8": 8,
            "U8": 8,
            "BOOL": 1,
        }.get(dt, 16)
        return name, tuple(header[name]["shape"]), Format(bits)

    for name in sorted(names):
        stem = name.removesuffix(".weight") if name.endswith(".weight") else name
        if name.endswith(_SCALE_SUFFIX):
            continue
        if name.endswith(".weight_packed"):
            n, k2 = header[name]["shape"]
            out.append((name, (n, k2 * 2), nvfp4_dev))
        elif name.endswith(".weight"):
            if f"{stem}.weight_scale_2" in names:
                out.append((name, tuple(header[name]["shape"]), nvfp4_dev))
            elif f"{stem}.weight_scale_inv" in names:
                out.append((name, tuple(header[name]["shape"]), fp8_dev))
            else:
                out.append(plain(name))
        else:
            out.append(plain(name))
    return out


def _safetensors_header(path: Path) -> dict:
    """A shard's {name: {shape, dtype}} header — the first 8 bytes give its JSON length,
    so shapes are read without touching any weight bytes."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def checkpoint_weight_specs(ckpt_dir: str | Path) -> list[tuple[str, tuple[int, ...], Format]]:
    """Every served weight in a checkpoint with its device face, from headers only.

    Uses the index's weight_map when present, else the single/sharded safetensors
    files. The fp4/fp8 population (config cannot derive it) is read off the actual
    tensor names; bytes are derived through :func:`nbytes`, not read.
    """
    ckpt_dir = Path(ckpt_dir)
    index = ckpt_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = sorted(p.name for p in ckpt_dir.glob("model-*.safetensors")) or [
            "model.safetensors"
        ]
    rows: list[tuple[str, tuple[int, ...], Format]] = []
    for shard in shards:
        rows.extend(weight_specs(_safetensors_header(ckpt_dir / shard)))
    return rows


def dtype(role: str, device: Any = None) -> torch.dtype:
    if role == "recurrent_state":
        # sm90's fused GDN kernel is f32-IO (a bf16 pool cost two casts per layer per tick);
        # CPU and metal cast at the boundary, so bf16 storage is free there.
        return torch.float32 if getattr(device, "type", device) == "cuda" else torch.bfloat16
    return _TABLE[role]


def roles() -> tuple[str, ...]:
    return tuple(_TABLE) + ("recurrent_state",)
