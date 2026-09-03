"""Native-fp8 projection fusion parity (serving-only, fuse_projections=True).

The fp4 case lives in test_weights.py::test_fused_projections_parity; this
covers the native-fp8 qkvz group: _fuse_projections concats .w8/.wscale along
N, _gdn splits the fused output back at the qkv boundary. sm90 serves the fp8
bytes; every other cell has Backend.materialize rebuild a bf16 weight from
them. Either way the concat is lossless, so the logits match.
"""

from dataclasses import replace

import numpy as np
import torch
from tilerl_kernels.backend import get_backend

from tilerl.config import tiny
from tilerl.model import Model, _fuse_projections, build_random
from tilerl.train import _training_kv


def _quant_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-128-block e4m3 weight quant (the checkpoint format load_hf keeps
    native): w8 [N,K] f8_e4m3, wscale [ceil(N/128), ceil(K/128)] f32."""
    n, k = w.shape
    ns, ks = (n + 127) // 128, (k + 127) // 128
    wp = w.float().new_zeros((ns * 128, ks * 128))
    wp[:n, :k] = w.float()
    scale = (
        wp.reshape(ns, 128, ks, 128).permute(0, 2, 1, 3).reshape(ns, ks, -1).abs().amax(-1) / 448.0
    ).clamp_min(1e-12)
    w8 = (wp / scale.repeat_interleave(128, 0).repeat_interleave(128, 1)).clamp(-448, 448)
    return w8[:n, :k].to(torch.float8_e4m3fn), scale.contiguous()


def test_fused_fp8_qkvz_parity():
    # GDN N dims must be 128-aligned or the per-128-block wscale grid doesn't
    # concat (the real 27B satisfies this: 10240 = 80 blocks, 6144 = 48).
    cfg = replace(
        tiny(),
        linear_num_key_heads=1,
        linear_key_head_dim=128,
        linear_num_value_heads=1,
        linear_value_head_dim=128,
    )
    model = build_random(cfg, seed=7)
    gdn = "layers.1"
    for key in (f"{gdn}.in_proj_qkv", f"{gdn}.in_proj_z"):
        w8, wscale = _quant_fp8_block(model.params.pop(key))  # serving keeps no master
        model.params[f"{key}.w8"] = w8
        model.params[f"{key}.wscale"] = wscale
    fused = Model(cfg, dict(model.params))
    _fuse_projections(cfg, fused.params)
    assert f"{gdn}.qkvz.w8" in fused.params and f"{gdn}.qkvz" not in fused.params

    batch = np.random.default_rng(3).integers(3, cfg.vocab_size, size=(2, 16)).astype(np.int64)
    positions = np.arange(16, dtype=np.int64)
    backend = get_backend()
    d = backend.device
    model.params = backend.materialize(model.params)
    fused.params = backend.materialize(fused.params)
    # sm90 keeps the fp8 bytes, every other cell got one bf16 weight -- never both
    assert (f"{gdn}.qkvz" in fused.params) ^ (f"{gdn}.qkvz.w8" in fused.params)
    with torch.no_grad():
        y0 = model.forward(batch, positions, _training_kv(model, 2, 16, device=d), backend)
        y1 = fused.forward(batch, positions, _training_kv(fused, 2, 16, device=d), backend)
    assert torch.allclose(y0, y1, rtol=1e-2, atol=1e-2), (y0 - y1).abs().max()
