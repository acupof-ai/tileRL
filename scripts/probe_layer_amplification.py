"""Does a 1.4e-04 prelude divergence reach 4.46 by layer 63, or stay at 1e-04?

Three probes narrowed the fusion flips (53 of 1000 MMLU answers, |delta logit| to
4.46) to a seed and left the amplification unmeasured:

- `probe_fusion_weights.py`: `_fuse_projections` is weight-preserving bit-for-bit.
- `probe_fusion_kernels.py`: fused and unfused GEMMs are bitwise identical at
  M=1/8/64, negative control 6.007e-02.
- `probe_attn_prep.py`: the branch `fuse_projections` actually gates -- `attn_prep`
  vs discrete `rmsnorm`/`rope`/`write_tokens` -- differs on the K plane at
  **1.562e-02 max, 1.44e-04 relative, 580 of 262144 elements**, with gate and V
  bitwise identical.

A 1.4e-04 relative seed and a 4.46 logit move are four orders apart. Either 64
layers amplify it or the flips have another source. **This probe measures the
amplification curve instead of asserting it**: the divergence per layer, both
arms, one real prompt.

`Backend._rows` casts activations to bf16 once per layer on cuda
(`backend.py:187`), so an f32 reorder that lands on a rounding boundary is
re-quantized every layer -- the recorded mechanism for this kind of growth. What
is unrecorded is the rate, and the rate is what decides whether the chain closes.

Reading two curves at once, because "it grew" is not the same as "it grew because
of this":

* **arm curve** -- |fuse0 hidden - fuse1 hidden| per layer. What fusion does.
* **noise floor** -- the same arm against ITSELF on a second identical run. A
  deterministic kernel gives exactly 0 here; anything above 0 is nondeterminism
  and the arm curve is only readable above it.

Without the floor, a curve that reaches O(1) by layer 63 could be run-to-run
variance in either arm and would read as amplification.

    CUDA_VISIBLE_DEVICES=6 TILERL_TARGET=cuda TILELANG_CACHE_DIR=/work/tilelang_cache \
    PYTHONPATH=src:packages/tilerl-kernels/src \
    python3 scripts/probe_layer_amplification.py --source /work/Qwen3.8-27B-NVFP4 \
        --flips /work/mmlucc2/concurrency.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tilerl_kernels.backend import get_backend

from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.eval import LETTERS, mmlu_questions
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.tokenizer import get_tokenizer


class CopyOnAppend(list):
    """Copy each hidden state as it arrives.

    The first version of this probe kept the references and copied after the
    forward returned, which read garbage: the tensors handed to `hidden_out` are
    views into buffers later kernels reuse, so by the end all 64 entries held
    whatever was written last. It showed up as a noise floor equal to the signal
    (2.767e+02 comparing an arm against itself) while the logits were bit-identical
    -- the tell that the capture was broken and not the model."""

    def append(self, t):
        super().append(t.detach().float().cpu().clone())


def trace(cfg, source, backend, tok, prompt, fuse, allowed):
    """Every layer's output for one prompt, plus the final letter logits.

    `aux_layers` is the model's own tap: passing all 64 makes `hidden_out`
    collect each layer rather than reimplementing the loop."""
    model = load_hf(cfg, source, fuse_projections=fuse)
    engine = build_engine(cfg, model, backend, num_blocks=1024, num_slots=4, max_batch=1,
                          max_total_tokens=8192, prefix_store=NoPrefixStore(),
                          decode_graph=False)
    hid = CopyOnAppend()
    fwd = model.forward
    aux = tuple(range(cfg.num_layers))

    def w_forward(ids, positions, kv, be, hidden_out=None, last_only=False, aux_layers=()):
        return fwd(ids, positions, kv, be, hidden_out=hid, last_only=last_only,
                   aux_layers=aux)

    model.forward = w_forward
    logits: list = []
    sample = engine._sample_batch

    def w_sample(rows):
        for r, lg, _ in rows:
            logits.append(lg.detach().float().cpu().clone())
        return sample(rows)

    engine._sample_batch = w_sample
    rid = engine.submit(tok.encode(prompt),
                        SamplingParams(temperature=0.0, max_new_tokens=1, seed=0,
                                       allowed_ids=allowed))
    for _ in range(64):
        engine.step()
        if rid in engine.poll():
            break
    layers = list(hid[:cfg.num_layers])
    model.forward, engine._sample_batch = fwd, sample
    del engine, model
    torch.cuda.empty_cache()
    assert layers, "no hidden states captured: the aux tap did not fire"
    return layers, (logits[0] if logits else None)


def curve(a, b, n_tok: int) -> list[dict]:
    """Per-layer divergence over the REAL tokens only.

    The prefill tick is padded to a bucket -- question 492 is 76 tokens run at
    T=128 -- and rows 76..127 are never written by the model, so they hold
    whatever was last in the buffer. Comparing them measured allocator state:
    the floor (one arm against itself) came out at 1.173e+03 while the logits
    were bit-identical, which is impossible for a real divergence and is the tell
    that the compared region was not model output."""
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape:
            break
        xs, ys = x[:, :n_tok], y[:, :n_tok]
        d = (xs - ys).abs()
        scale = xs.abs().mean().clamp_min(1e-30)
        out.append({"layer": i, "max": d.max().item(),
                    "rel": (d.mean() / scale).item(),
                    "differing": int((d > 0).sum().item()), "total": d.numel()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--flips", default="", help="concurrency.json; picks the largest flip")
    ap.add_argument("--q", type=int, default=-1, help="question index, overrides --flips")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    be = get_backend()
    assert be.device.type == "cuda", be.device
    cfg = qwen38_27b()
    tok = get_tokenizer(a.source)
    prompts, golds, subjects = mmlu_questions(a.n, a.seed)
    allowed = tuple(sorted({tok.encode(f" {c}")[-1] for c in LETTERS}
                           | {tok.encode(c)[-1] for c in LETTERS}))

    qi = a.q
    if qi < 0:
        assert a.flips, "need --flips or --q"
        fl = [r for r in json.loads(Path(a.flips).read_text())["flips"]
              if r["knob"] == "fusion"]
        assert fl, "no fusion flips in that file"
        top = max(fl, key=lambda r: r["arm_delta"])
        qi = top["q"]
        print(f"question {qi} ({top['subject']}), the largest fusion flip: "
              f"{top['a']}->{top['b']}, arm |delta| {top['arm_delta']:.3f}")

    n_tok = len(tok.encode(prompts[qi]))
    h0, lg0 = trace(cfg, a.source, be, tok, prompts[qi], False, allowed)
    h1, lg1 = trace(cfg, a.source, be, tok, prompts[qi], True, allowed)
    # the floor: fuse 0 twice. A deterministic kernel gives exactly 0 at every layer.
    h0b, lg0b = trace(cfg, a.source, be, tok, prompts[qi], False, allowed)

    print(f"prompt {n_tok} tokens; the tick runs T={h0[0].shape[1]}, so rows "
          f"{n_tok}..{h0[0].shape[1] - 1} are padding and are excluded")
    arm, floor = curve(h0, h1, n_tok), curve(h0, h0b, n_tok)
    print(f"\n{'layer':>5} {'arm max|d|':>12} {'arm rel':>10} {'floor max':>11} "
          f"{'arm/floor':>10}  kind")
    for r, f in zip(arm, floor):
        i = r["layer"]
        if i < 4 or i % 8 == 0 or i >= len(arm) - 2:
            ratio = r["max"] / f["max"] if f["max"] > 0 else float("inf")
            print(f"{i:>5} {r['max']:>12.3e} {r['rel']:>10.2e} {f['max']:>11.3e} "
                  f"{ratio:>10.1f}  {'full-attn' if cfg.is_full_attn(i) else 'gdn'}")

    first = next((r["layer"] for r in arm if r["max"] > 0), None)
    fl_max = max((f["max"] for f in floor), default=0.0)
    print(f"\nfirst layer where the arms differ: {first}")
    print(f"noise floor (same arm twice), max over all layers: {fl_max:.3e}")
    # the floor is a validity gate, not a caveat: both runs are fuse=0 on one card
    # with temperature 0, so any nonzero value means the capture is wrong (the first
    # version kept buffer views and read 2.767e+02 here) and the arm curve is unreadable.
    assert fl_max == 0.0, (
        f"the same arm run twice differs by {fl_max:.3e}: the capture is broken, "
        "not the model -- fix it before reading the arm curve")
    if arm:
        g = arm[-1]["max"] / arm[first]["max"] if first is not None and arm[first]["max"] else 0
        print(f"layer {arm[first]['layer'] if first is not None else 0} -> "
              f"{arm[-1]['layer']}: "
              f"{arm[first]['max'] if first is not None else 0:.3e} -> {arm[-1]['max']:.3e} "
              f"(x{g:.0f})")
    if lg0 is not None and lg1 is not None:
        sel = list(allowed)
        d = (lg0[sel] - lg1[sel]).abs().max().item()
        fld = (lg0[sel] - lg0b[sel]).abs().max().item() if lg0b is not None else 0.0
        print(f"letter logits: arm |delta| {d:.3f}, floor {fld:.3f}")
        print("  the 53-flip population had median 1.071 / max 4.462 -- this single "
              "question's value is what the curve has to explain")
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
        (o / "amplification.json").write_text(json.dumps(
            {"q": qi, "subject": subjects[qi], "gold": golds[qi],
             "arm": arm, "floor": floor}, indent=1))
        print(f"wrote {o / 'amplification.json'}")


if __name__ == "__main__":
    main()
