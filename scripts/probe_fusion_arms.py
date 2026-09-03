"""Which arm is right when projection fusion moves a logit: the fused GEMM or the three?

`fuse_projections=True` changes 53 of 1000 MMLU answers with |delta logit| up to
4.46, where an arm change is worth ~0.153. Fusion is documented lossless, so one
of the two arms is wrong and serving ships the fused one.

**Three stages, because a dense reference is a third arithmetic and not ground
truth.** On a question where all three disagree, comparing kernels to a reference
settles nothing -- and "the reference agrees with fuse 0" could mean the two share
a bias rather than that fuse 1 is wrong.

1. **Is fusion weight-preserving at all?** Dequantize the fp4 packs to bf16 once,
   then run the SAME dense matmul twice: on the three separate weights, and on
   their concat. `renorm_fp4_scale` takes a per-row amax (`reference.py:249`,
   `keepdim=True`), so every row carries its own scale and its own oscale and
   stacking rows should change no row's numbers -- bit-for-bit, not allclose. If
   this fails, the bug is in the concat and the kernels are innocent.
2. **Where does the kernel diverge?** Per-layer: feed one prompt through both
   engines and record each layer's hidden state, so the FIRST layer whose output
   differs is named rather than inferred from the logits 64 layers later.
3. **Which arm does the reference agree with?** Only meaningful once (1) holds.
   Per-question, on the largest-|delta| flips -- a mechanism worth 4.46 is visible
   per layer, so the biggest are the informative ones, not a random sample.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_fusion_arms.py \
        --source /work/Qwen3.8-27B-NVFP4 --flips /work/mmlucc2/concurrency.json \
        --top 4 --out /work/fusionarms
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.eval import LETTERS, mmlu_questions
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import _projection_groups, load_hf
from tilerl.tokenizer import get_tokenizer
from tilerl_kernels.backend import get_backend
from tilerl_kernels import reference


def stage1_concat_is_lossless(cfg, source, backend, layers=(3, 4)) -> int:
    """Dequantize to bf16, then one dense matmul on the separate weights and on
    their concat. Per-row fp4 scales mean this must hold bit-for-bit.

    Layers 3 and 4 cover a full-attn layer (qkv, gate_up) and a GDN one (ab,
    qkvz, gate_up). qkvz is the ONLY group the existing parity test covers, so
    the others are where an untested claim would live."""
    print("=== 1. is the fp4 concat weight-preserving? (dequantize, then dense twice) ===")
    model = load_hf(cfg, source)
    bad = 0
    x = torch.randn(8, cfg.hidden_size, dtype=torch.bfloat16, device=backend.device)
    for i in layers:
        for fused_key, group in _projection_groups(cfg, i):
            if not all(f"{k}.wq" in model.params for k in group):
                continue
            ws = []
            for k in group:
                w = reference.unpack_fp4(model.params[f"{k}.wq"], model.params[f"{k}.scale"],
                                         model.params.get(f"{k}.oscale"))
                ws.append(w.to(backend.device, torch.bfloat16))
            # the three, each against x, then stacked; and the concat against x
            sep = torch.cat([x.float() @ w.float().t() for w in ws], dim=1)
            cat = x.float() @ torch.cat(ws, dim=0).float().t()
            same = torch.equal(sep, cat)
            mx = (sep - cat).abs().max().item()
            bad += not same
            print(f"  L{i:<3} {fused_key.split('.')[-1]:<6} N={[w.shape[0] for w in ws]} "
                  f"bitwise {'EQUAL' if same else 'DIFFER'}  max|d| {mx:.3e}"
                  f"{'' if same else '   <-- the concat is the bug'}")
    del model
    torch.cuda.empty_cache()
    return bad


def hidden_trace(cfg, source, backend, tok, prompt, fuse, allowed):
    """Every layer's output for one prompt, plus the final logits."""
    model = load_hf(cfg, source, fuse_projections=fuse)
    engine = build_engine(cfg, model, backend, num_blocks=256, num_slots=4, max_batch=1,
                          max_total_tokens=8192, prefix_store=NoPrefixStore(),
                          decode_graph=False)
    hid: list = []
    aux = tuple(range(cfg.num_layers))
    fwd = model.forward

    def w_forward(ids, positions, kv, be, hidden_out=None, last_only=False, aux_layers=()):
        return fwd(ids, positions, kv, be, hidden_out=hid, last_only=last_only,
                   aux_layers=aux)

    model.forward = w_forward
    rid = engine.submit(tok.encode(prompt),
                        SamplingParams(temperature=0.0, max_new_tokens=1, seed=0,
                                       allowed_ids=allowed))
    for _ in range(64):
        engine.step()
        if rid in engine.poll():
            break
    layers = [h.detach().float().cpu() for h in hid[:cfg.num_layers]]
    model.forward = fwd
    del engine, model
    torch.cuda.empty_cache()
    return layers


def stage2_first_divergent_layer(cfg, source, backend, tok, prompt, allowed) -> None:
    print("\n=== 2. which layer diverges first? ===")
    a = hidden_trace(cfg, source, backend, tok, prompt, False, allowed)
    b = hidden_trace(cfg, source, backend, tok, prompt, True, allowed)
    n = min(len(a), len(b))
    print(f"  {'layer':>5} {'max|d|':>11} {'rel':>10}  kind")
    first = None
    for i in range(n):
        d = (a[i] - b[i]).abs().max().item()
        scale = a[i].abs().max().clamp_min(1e-30).item()
        if d > 0 and first is None:
            first = i
        if i < 6 or d / scale > 1e-2 or i == n - 1:
            kind = cfg.is_full_attn(i) and "full-attn" or "gdn"
            print(f"  {i:>5} {d:>11.3e} {d / scale:>10.3e}  {kind}")
    print(f"  first layer with ANY difference: {first} "
          f"({'full-attn' if first is not None and cfg.is_full_attn(first) else 'gdn'})"
          if first is not None else "  no layer differs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--flips", required=True, help="concurrency.json from probe_mmlu_concurrency")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    backend = get_backend()
    assert backend.device.type == "cuda", backend.device
    cfg = qwen38_27b()
    tok = get_tokenizer(a.source)
    prompts, golds, _ = mmlu_questions(a.n, a.seed)
    allowed = tuple(sorted({tok.encode(f" {c}")[-1] for c in LETTERS}
                           | {tok.encode(c)[-1] for c in LETTERS}))

    bad = stage1_concat_is_lossless(cfg, a.source, backend)
    if bad:
        print("\nSTOP: the concat is not weight-preserving, so the kernels are not the "
              "suspect. Stages 2 and 3 would measure the wrong thing.")
        raise SystemExit(1)
    print("  the concat preserves every row -> any kernel disagreement is the kernel's")

    flips = json.loads(Path(a.flips).read_text())["flips"]
    fusion = [r for r in flips if r["knob"] == "fusion" and r["pair"].startswith("conc8")]
    fusion.sort(key=lambda r: -r["arm_delta"])
    picks = fusion[: a.top]
    print(f"\nlargest-|delta| fusion flips: "
          f"{[(r['q'], round(r['arm_delta'], 2)) for r in picks]}")

    stage2_first_divergent_layer(cfg, a.source, backend, tok, prompts[picks[0]["q"]], allowed)

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "arms.json").write_text(json.dumps(
        {"concat_lossless": bad == 0, "picks": picks}, indent=1))
    print(f"\nwrote {o / 'arms.json'}")


if __name__ == "__main__":
    main()
