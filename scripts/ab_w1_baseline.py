"""The W=1 no-speculation baseline at the depth sweep's exact config.

`ab_draft_depth.py` cannot produce this row: `--depths` refuses 0, and
`engine.py:405` rejects width <= 1 whenever a draft is attached, so W=1 needs an
engine built WITHOUT one. Everything else here is copied from that script rather
than re-chosen -- same model build, same `wikitext_ids` passages, same ctx,
tokens, batch and `measure`, so `tick(W=1)` is subtractable from its rows. A
baseline measured at a different context or on different text is not a baseline.

  scripts/pod_sync.sh run w1 'python3 -u scripts/ab_w1_baseline.py \
      --source /work/Qwen3.8-27B-NVFP4 --ctx 2048 --tokens 128 --prompts 8 --batch 8'
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src")
)

from ab_draft_depth import (  # noqa: E402
    _build_model,
    _engine_sha,
    _sha,
    bucket,
    measure,
)
from corpus import wikitext_ids  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.engine import build_engine  # noqa: E402
from tilerl.kv_cache import BLOCK_TOKENS  # noqa: E402
from tilerl.server import get_tokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    if a.prompts % a.batch:
        raise SystemExit(f"--prompts {a.prompts} is not a multiple of --batch {a.batch}")

    # _build_model reads TILERL_QWEN38_SOURCE, not --source: without it, cli.py:20
    # falls back to the Hub and the run dies on a network error 40 s in.
    os.environ.setdefault("TILERL_QWEN38_SOURCE", a.source)

    be = get_backend()
    arch = getattr(be, "arch", "") or "sm70"
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    prompts = wikitext_ids(get_tokenizer(a.source), a.prompts, a.ctx)

    print(f"# probe {_sha(__file__)}, engine tree {_engine_sha()}, arch {arch}")
    print(f"# W=1 NO-DRAFT baseline: ctx={a.ctx}, wikitext x{a.prompts}, "
          f"max_new_tokens={a.tokens}, batch={a.batch}. M={a.batch} -> "
          f"{bucket(arch, a.batch)}")

    need = -(-(a.ctx + a.tokens) // BLOCK_TOKENS) * a.batch + 8
    e = build_engine(cfg, model, be, num_blocks=need, num_slots=a.batch + 1,
                     max_batch=a.batch, max_total_tokens=a.ctx + a.tokens + 64)
    assert e._draft is None, "a draft attached: this is not the W=1 arm"
    assert e._width == 1, f"width {e._width}, expected 1"

    groups = [prompts[i * a.batch:(i + 1) * a.batch] for i in range(a.prompts // a.batch)]
    measure(e, groups[0], a.tokens, arch)  # warm: JIT and this (batch, width) capture
    got = [measure(e, g, a.tokens, arch) for g in groups]

    print(f"# {'ms/tick':>8} {'tok/fwd':>8} {'tok/s':>7}  per-M: KERNELxCOUNT:MEAN_MS")
    for i, g in enumerate(got):
        ms, tpf, width = g[0], g[1], g[2]
        per = "  ".join(f"{k}{w}x{len(v)}:{sum(v) / len(v):.1f}"
                        for (k, w), v in sorted(g[3].items()))
        # tok/fwd from `measure` is ALREADY aggregate over the batch: multiplying by
        # batch again printed 2364.1 tok/s for a 27B on one card. Verified against
        # ab_draft_depth's own seven rows, which this reproduces to +-0.1.
        print(f"  {ms:8.2f} {tpf:8.2f} {tpf / ms * 1000:7.1f}  {per}"
              + (f"   (group {i})" if len(got) > 1 else ""))
        assert abs(width - 1.0) < 1e-9, f"mean width {width}, expected exactly 1"
    print(f"\n# subtract from ab_draft_depth's rows at the same ctx/batch: this is "
          f"tick(W=1) on {bucket(arch, a.batch)[0]}.")


if __name__ == "__main__":
    main()
