"""Base-vs-spec logits at the first divergence position — is the 38/200
completion gap tile rounding or a verify bug?

The engine's ``_verify`` docstring states the guarantee: every committed spec
token is the trunk's own draw at that chain position, and a W>1 tile does not
agree bit-for-bit with a W=1 tile off the CPU reference. So base and spec
completions can diverge without any bug — where the two arms' logits differ by
tile rounding and the argmax flips on a near-tie. The bug-shaped alternative
is the spec arm committing a token its own logits did not argmax.

For the first N questions the two arms finish differently, this script dumps
both arms' trunk logits and committed token at the first diverging position:
both tokens equal their own logits' argmax -> rounding, exonerated; the spec
token != its argmax -> verify bug.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/acc_spec_divergence_logits.py \
        --source /work/Qwen3.8-27B-NVFP4 --draft /work/Qwen3.8-27B-DFlash2 \
        --gsm8k /work/gsm8k_test.jsonl --diff /work/accspec_b1/gsm8k-diff.json \
        --out /work/accspec_div
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from tilerl import engine as engine_mod
from tilerl.config import qwen38_27b
from tilerl.engine import build_engine
from tilerl.eval import generate
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.prompt import render_chat, sampling
from tilerl.tokenizer import get_tokenizer

# req_id -> position -> (logits row on CPU, sampled token). The spec arm may
# sample a position more than once (a rejected chain re-draws it); the last
# draw is the committed one, so later captures overwrite.
_caps: dict[int, dict[int, tuple[torch.Tensor, int]]] = {}
_orig_sample_batch = engine_mod.Engine._sample_batch


def _capture_sample_batch(self, rows):
    toks = _orig_sample_batch(self, rows)
    for (r, lg, pos), tok in zip(rows, toks):
        _caps.setdefault(r.req_id, {})[int(pos)] = (lg.detach().float().cpu(), int(tok))
    return toks


engine_mod.Engine._sample_batch = _capture_sample_batch


def _run_arm(cfg, model, backend, tok, prompt, sp, draft_path, width):
    from tilerl.spec import load_draft

    draft = load_draft(model, draft_path) if draft_path else None
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=1, max_batch=1,
                          draft=draft, spec_depth=max(1, width - 1), decode_graph=True,
                          prefix_store=NoPrefixStore())
    _caps.clear()
    generate(engine, tok, [prompt], sp, 1)
    caps = dict(_caps)
    engine = draft = None
    torch.cuda.empty_cache()
    return caps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--diff", required=True, help="gsm8k-diff.json from an acc_spec_arms run")
    p.add_argument("--n", type=int, default=3, help="how many diverging questions to probe")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = qwen38_27b()
    tok = get_tokenizer(args.source)
    model = load_hf(cfg, args.source)
    rows = [json.loads(ln) for ln in Path(args.gsm8k).read_text().splitlines() if ln.strip()]
    diff = json.loads(Path(args.diff).read_text())
    params = sampling(tok, False, 512, temperature=0.0, max_think_tokens=0, seed=0)
    sp = replace(params, temperature=0.0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for qi in diff["differing"][: args.n]:
        r = rows[qi]
        prompt = render_chat([("user", r["prompt"])], False)
        base = _run_arm(cfg, model, backend, tok, prompt, sp, None, 1)
        spec = _run_arm(cfg, model, backend, tok, prompt, sp, args.draft, 8)
        rid = next(iter(base))
        bp, sp2 = base[rid], spec[next(iter(spec))]
        pos = next((k for k in sorted(bp) if k in sp2 and bp[k][1] != sp2[k][1]), None)
        if pos is None:
            print(f"q{qi}: no divergence in this re-run", flush=True)
            reports.append({"q": qi, "diverged": False})
            continue
        bl, bt = bp[pos]
        sl, st = sp2[pos]
        ba, sa = int(bl.argmax()), int(sl.argmax())
        d = (bl - sl).abs()
        bgap = bl.topk(2).values.diff().item()
        sgap = sl.topk(2).values.diff().item()
        rep = {"q": qi, "diverged": True, "position": pos,
               "base_tok": bt, "spec_tok": st,
               "base_argmax": ba, "spec_argmax": sa,
               "base_tok_is_argmax": bt == ba, "spec_tok_is_argmax": st == sa,
               "logits_abs_max": float(d.max()), "logits_abs_mean": float(d.mean()),
               "base_top2_gap": bgap, "spec_top2_gap": sgap,
               "base_logit_at_spec_tok": float(bl[st]), "base_logit_at_base_tok": float(bl[bt]),
               "spec_logit_at_base_tok": float(sl[bt]), "spec_logit_at_spec_tok": float(sl[st])}
        reports.append(rep)
        print(f"q{qi} pos {pos}: base tok {bt} (argmax {ba}, match {bt == ba})  "
              f"spec tok {st} (argmax {sa}, match {st == sa})  "
              f"|dlogits| max {rep['logits_abs_max']:.2e} mean {rep['logits_abs_mean']:.2e}  "
              f"top2 gap base {bgap:.2e} spec {sgap:.2e}", flush=True)

    (out / "divergence.json").write_text(json.dumps(reports))
    n_div = sum(1 for r in reports if r.get("diverged"))
    n_benign = sum(1 for r in reports if r.get("diverged")
                   and r["base_tok_is_argmax"] and r["spec_tok_is_argmax"])
    print(f"\n{n_div}/{len(reports)} diverged; {n_benign}/{n_div} benign (both tokens are "
          f"their own logits' argmax)", flush=True)


if __name__ == "__main__":
    main()
