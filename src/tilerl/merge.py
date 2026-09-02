"""ISO-Merger: compose specialists that share a base without data (arXiv 2607.19331).

RLVR moves a linear's singular frames and leaves its spectrum alone, so each
2D weight is merged as ``W* = U* Σ₀ V*ᵀ``: the base spectrum verbatim, the
frames composed from the specialists' displacements in the Stiefel tangent
space at ``(U₀, V₀)``. Offline, checkpoint-only; ``tilerl merge`` is the CLI.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from torch import Tensor

# Appendix E clip on the retention coefficients: no sign flips, no over-amplification.
_C_CLIP = (0.0, 1.5)


def _tangent(x0: Tensor, y: Tensor) -> Tensor:
    """Π_X(Y) = Y − X sym(XᵀY), the projection onto the Stiefel tangent space at X."""
    return y - x0 @ ((x0.T @ y + y.T @ x0) / 2)


def _polar(x: Tensor) -> Tensor:
    # ponytail: torch.linalg SVD retraction; Newton-Schulz matmuls when this runs per step
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    return u @ vh


def iso_merge_weight(w0: Tensor, ws: list[Tensor], rho_keep: float = 0.9, ridge: float = 1e-3):
    """Merge one 2D weight. Computed in float64, returned in ``w0.dtype``."""
    u0, s0, vh0 = torch.linalg.svd(w0.double(), full_matrices=False)
    v0 = vh0.T
    keep = max(1, round(rho_keep * s0.numel()))
    xu, xv, g = [], [], []
    for w in ws:
        u, _, vh = torch.linalg.svd(w.double(), full_matrices=False)
        # An SVD fixes each (u_k, v_k) pair only up to a joint sign; align to the base.
        sign = torch.where((u0 * u).sum(0) < 0, -1.0, 1.0).double()
        du, dv = _tangent(u0, u * sign - u0), _tangent(v0, vh.T * sign - v0)
        du[:, keep:] = 0
        dv[:, keep:] = 0
        xu.append(du)
        xv.append(dv)
        g.append((du * s0) @ vh0 + (u0 * s0) @ dv.T)  # first-order effect on W
    # Unit retention: (Γ + ridge·I) c = diag(Γ) keeps every specialist's own
    # first-order effect at strength 1 in the sum. Ridge is relative to Γ's scale.
    gram = torch.stack([torch.stack([(a * b).sum() for b in g]) for a in g])
    d = gram.diagonal()
    eye = torch.eye(len(ws), dtype=gram.dtype)
    c = torch.linalg.solve(gram + (ridge * d.mean() + 1e-30) * eye, d).clamp(*_C_CLIP)
    # Masking broke the tangent constraint; re-project the sum, then retract.
    cu = _tangent(u0, sum(ci * x for ci, x in zip(c, xu)))
    cv = _tangent(v0, sum(ci * x for ci, x in zip(c, xv)))
    return ((_polar(u0 + cu) * s0) @ _polar(v0 + cv).T).to(w0.dtype)


def average_merge(base: dict[str, Tensor], specialists: list[dict[str, Tensor]]):
    """W₀ + mean(W_i − W₀): the task-vector baseline, and ISO's rule for non-matrix params."""
    return {
        k: torch.stack([s[k].float() for s in specialists]).mean(0).to(v.dtype)
        for k, v in base.items()
    }


def iso_merge(
    base: dict[str, Tensor],
    specialists: list[dict[str, Tensor]],
    rho_keep: float = 0.9,
    ridge: float = 1e-3,
) -> dict[str, Tensor]:
    """Merge ``model.params``-style dicts: every float 2D tensor is a linear weight."""
    out = average_merge(base, specialists)
    for k, w0 in base.items():
        if w0.dim() == 2 and w0.is_floating_point():
            out[k] = iso_merge_weight(w0, [s[k] for s in specialists], rho_keep, ridge)
    return out


if __name__ == "__main__":  # runnable check: one specialist comes back, spectrum kept
    torch.manual_seed(0)
    w0 = torch.randn(96, 64)
    w1 = w0 + 1e-3 * torch.randn(96, 64)
    m = iso_merge_weight(w0, [w1])
    assert (m - w1).norm() / (w1 - w0).norm() < 0.5, "K=1 does not return the specialist"
    s0, s = torch.linalg.svdvals(w0), torch.linalg.svdvals(m)
    assert torch.allclose(s, s0, rtol=1e-3), "merged weight lost the base spectrum"
    print("merge: K=1 and spectrum OK")


def merge_checkpoints(
    base: str | Path,
    specialists: list[str | Path],
    out: str | Path,
    method: str = "iso",
    shard_bytes: int = 2 << 30,
    **kw,
) -> int:
    """Merge safetensors checkpoints one tensor at a time and write sharded
    output: peak memory is one tensor from each input plus one output shard,
    not K+1 checkpoints. Inputs must hold bf16 masters (what ``save_hf`` writes
    for a trained model); fp4 byte checkpoints are refused. Returns the tensor
    count."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    def index(d: str | Path) -> dict[str, Path]:
        d = Path(d)
        files = sorted(d.glob("model-*.safetensors")) or [d / "model.safetensors"]
        # a safe_open handle is not a dict: .keys() is its only listing
        return {k: f for f in files for k in safe_open(str(f), "pt").keys()}  # noqa: SIM118

    srcs = [index(base)] + [index(s) for s in specialists]
    keys = list(srcs[0])
    if any(k.endswith(".wq") for k in keys):
        raise ValueError("merge needs bf16 masters; this checkpoint holds fp4 bytes (.wq)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(base) / "config.json", out / "config.json")
    handles: dict[Path, object] = {}

    def get(src: dict[str, Path], k: str) -> Tensor:
        f = src[k]
        if f not in handles:
            handles[f] = safe_open(str(f), "pt")
        return handles[f].get_tensor(k)

    shard: dict[str, Tensor] = {}
    weight_map: dict[str, str] = {}
    size = 0

    def flush() -> None:
        nonlocal shard, size
        if shard:
            name = f"model-{len(set(weight_map.values())):05d}.safetensors"
            save_file(shard, str(out / name))
            weight_map.update(dict.fromkeys(shard, name))
            shard, size = {}, 0

    for k in keys:
        w0, ws = get(srcs[0], k), [get(s, k) for s in srcs[1:]]
        if method == "iso" and w0.dim() == 2 and all(w.shape == w0.shape for w in ws):
            m = iso_merge_weight(w0, ws, **kw)
        else:
            m = torch.stack([w.float() for w in ws]).mean(0).to(w0.dtype)
        shard[k] = m.contiguous()
        size += m.numel() * m.element_size()
        if size >= shard_bytes:
            flush()
    flush()
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}))
    return len(keys)
