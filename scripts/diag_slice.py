"""Where does the signal die in a real-slice forward: weight stats, pack/unpack
roundtrip, then a short forward printing every rmsnorm output norm and logits std.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import torch

from tilerl.config import qwen36_27b
from tilerl.kv_cache import LinearStatePool
from tilerl.model import load_hf
from tilerl_kernels.backend import get_backend
from tilerl_kernels.reference import pack_fp4, unpack_fp4


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "/host/tc27-nvfp4-slice2"
    cfg = replace(qwen36_27b(), num_layers=2, full_attn_layers=())
    model = load_hf(cfg, source, keep_master=True)  # inspects the bf16 masters

    print("== weight stats (bf16 masters) ==")
    for key in ("layers.0.in_proj_qkv", "layers.0.gate_proj", "layers.1.out_proj", "lm_head"):
        w = model.params[key].float()
        print(
            f"{key}: absmean={w.abs().mean():.5f} absmax={w.abs().max():.3f} nan={w.isnan().any().item()}"
        )

    print("== pack/unpack roundtrip ==")
    for key in ("layers.0.gate_proj", "lm_head"):
        w = model.params[key].float()
        back = unpack_fp4(*pack_fp4(model.params[key])).float()
        print(f"{key}: rel_err={((back - w).abs().mean() / w.abs().mean().clamp_min(1e-9)):.4f}")

    backend = get_backend()
    print(f"== forward on {backend.target} ==")
    model.params = {k: v.to(backend.device) for k, v in model.params.items()}
    kv = SimpleNamespace()
    kv.dense = True
    kv.state_pool = LinearStatePool(
        1,
        cfg.num_linear_layers,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        device=backend.device,
    )
    kv.state_slot = torch.zeros(1, dtype=torch.long)

    orig_rmsnorm = backend.rmsnorm
    counter = [0]

    def traced_rmsnorm(x, w, eps):
        y = orig_rmsnorm(x, w, eps)
        counter[0] += 1
        print(f"  rmsnorm #{counter[0]}: out_norm={y.float().norm():.4f}")
        return y

    backend.rmsnorm = traced_rmsnorm
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    pos = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ids, pos, kv, backend)
    lf = logits.float()
    print(f"logits: std={lf.std():.6f} min={lf.min():.4f} max={lf.max():.4f}")
    print(f"logits[0,0,:8]={[round(v, 4) for v in lf[0, 0, :8].tolist()]}")


if __name__ == "__main__":
    main()
