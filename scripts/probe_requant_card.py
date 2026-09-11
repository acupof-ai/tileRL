"""On-card requant gate for --served-fp4 (#483): the sm90 materialize twiddles
every .wq once and tags it tw-bf16; requantize_fp4 must re-pack THROUGH that
twiddle and the sm90 decode kernel must still read sane bytes. Loads the real
27B checkpoint (keep_master), materializes to cuda, times a full re-pack, and
checks served vs bf16-master on sampled linears.

CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda TILERL_27B_CKPT=/work/Qwen3.8-27B-NVFP4 \
  /work/tl013/bin/python3 scripts/probe_requant_card.py
"""

import os
import time

import torch
from tilerl_kernels.backend import get_backend
from tilerl_kernels.reference import (
    linear_fp4 as ref_linear_fp4,
)
from tilerl_kernels.reference import (
    pack_fp4,
    twiddle_fp4,
)

from tilerl import config as config_mod
from tilerl.engine import card_guard
from tilerl.model import fp4_param_keys, load_hf, requantize_fp4


def main():
    ck = os.environ.get("TILERL_27B_CKPT")
    assert ck, "set TILERL_27B_CKPT"
    card_guard()  # materialize touches the card: refuse before touching an ungranted one
    backend = get_backend()
    print("arch", backend.arch, "device", torch.cuda.get_device_name(0))
    cfg = config_mod.qwen38_27b()
    t0 = time.time()
    model = load_hf(cfg, ck, fuse_projections=False, keep_master=True)
    print("load_hf %.1fs" % (time.time() - t0))

    keys = sorted(k for k in fp4_param_keys(cfg) if f"{k}.wq" in model.params)
    print("fp4 served keys:", len(keys))

    t0 = time.time()
    model.params = backend.materialize(model.params)
    torch.cuda.synchronize()
    print("materialize %.1fs" % (time.time() - t0))

    # Premise: materialize left twiddled, tagged slots (the thing CPU is blind to).
    tagged = [k for k in keys if getattr(model.params[k + ".wq"], "_tl_layout", "natural") != "natural"]
    print("tagged slots:", len(tagged), "of", len(keys))
    assert len(tagged) == len(keys), "sm90 must twiddle+tag every fp4 slot"
    k0 = keys[0]
    wq = model.params[k0 + ".wq"]
    print("sample tag", k0, getattr(wq, "_tl_layout", None))

    # A full re-pack after a simulated step: perturb every bf16 master slightly.
    with torch.no_grad():
        for k in keys:
            model.params[k].add_(torch.randn_like(model.params[k]) * 1e-4)

    torch.cuda.synchronize()
    t0 = time.time()
    n = requantize_fp4(model)
    torch.cuda.synchronize()
    print("requant keys", n, "per-step repack %.3f ms" % ((time.time() - t0) * 1e3))

    # Layout correctness on cuda: slot == twiddle(pack(master)), tag preserved.
    bad_layout = 0
    for k in keys[:16]:
        master = model.params[k]
        scale0 = model.params[k + ".scale"]
        nat, _ = pack_fp4(master, block=master.shape[1] // scale0.shape[1])
        want = twiddle_fp4(nat)
        if not torch.equal(model.params[k + ".wq"], want.to(model.params[k + ".wq"].device)):
            bad_layout += 1
    print("layout mismatches (of 16 sampled):", bad_layout)
    assert bad_layout == 0
    assert all(getattr(model.params[k + ".wq"], "_tl_layout", None) == "tw-bf16" for k in keys)

    # Served sm90 kernel vs the natural f32 reference on sampled linears.
    torch.manual_seed(0)
    rels, argmax_agree = [], 0
    for k in keys[:12]:
        wq = model.params[k + ".wq"]
        scale = model.params[k + ".scale"]
        osc = model.params[k + ".oscale"]
        N, K2 = wq.shape
        K = K2 * 2
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        y = backend.linear_fp4(x, wq, scale, oscale=osc).float()
        # reference wants NATURAL nibbles + f32 on host
        from tilerl_kernels.reference import untwiddle_fp4

        nat = untwiddle_fp4(wq).cpu()
        yref = ref_linear_fp4(x.cpu().float(), nat, scale.cpu().float(), osc.cpu().float())
        rel = ((y.cpu() - yref).abs().max() / yref.abs().max()).item()
        rels.append(rel)
        argmax_agree += int((y[0].argmax() == yref[0].argmax().cuda()).all().item())
    print("served-vs-ref max rel per linear:", [f"{r:.4f}" for r in rels])
    print(f"worst max rel {max(rels):.4f}, argmax rows agree {argmax_agree}/12")
    print("PROBE_OK")


if __name__ == "__main__":
    main()
