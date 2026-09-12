"""#545 pending-remote V100 point: sm70 decode-graph capture poisons the caching
allocator. Two arms, run as SEPARATE processes:

  --arm auto : decode_graph=None (the default). On sm70 the engine must run with
               graph OFF and a one-time warning; dense -> empty_cache -> sparse in
               one process must survive every empty_cache (the fidelity harness
               crash shape). Prints AUTO_ARM_PASS.
  --arm force: decode_graph=True still honours capture on sm70; capture fails,
               eager fallback runs, and the post-run empty_cache is expected to
               INTERNAL-assert (captures_underway). Prints FORCE_POISON_CONFIRMED.

Small context on purpose — the failure is the capture, not the context length.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl import cli  # noqa: E402
from tilerl.cli import _build_model  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402

TOKENS = 2048
NEW = 8


def _run_engine(be, cfg, model, *, decode_graph, sparse: bool) -> None:
    kw = dict(num_blocks=0, num_slots=1, max_batch=1,
              max_total_tokens=TOKENS + 512, max_blocks=(TOKENS // 16) + 32,
              decode_graph=decode_graph)
    if sparse:
        kw.update(sparse_k=128, scorer="bounds", kv_cold_bytes=1 << 30)
    e = build_engine(cfg, model, be, **kw)
    print(f"# sparse={sparse} decode_graph_on={e._decode_graph_on}", flush=True)
    ids = [(t % 31000) + 7 for t in range(TOKENS)]
    rid = e.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=NEW, seed=0))
    out: list = []
    while len(out) < NEW:
        e.step()
        out = e.poll().get(rid, out)
    print(f"# sparse={sparse} generated {len(out)} tokens {out[:4]}", flush=True)
    e.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["auto", "force"])
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    cli._QWEN38_SOURCE = args.source
    be = get_backend()
    print(f"# arm={args.arm} arch={be.arch} torch={torch.__version__}", flush=True)
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)

    dg = None if args.arm == "auto" else True
    _run_engine(be, cfg, model, decode_graph=dg, sparse=False)
    print("# dense arm done; calling empty_cache", flush=True)
    try:
        torch.cuda.empty_cache()
        print("EMPTY_CACHE_AFTER_DENSE OK", flush=True)
    except RuntimeError as exc:
        print(f"EMPTY_CACHE_AFTER_DENSE RAISES {type(exc).__name__}: "
              f"{str(exc).splitlines()[0]}", flush=True)
        if args.arm == "force" and "captures_underway" in str(exc):
            print("FORCE_POISON_CONFIRMED", flush=True)
            return
        raise

    if args.arm == "force":
        print("# force arm: empty_cache survived, poison NOT reproduced", flush=True)
        return

    _run_engine(be, cfg, model, decode_graph=None, sparse=True)
    try:
        torch.cuda.empty_cache()
        print("EMPTY_CACHE_AFTER_SPARSE OK", flush=True)
    except RuntimeError:
        traceback.print_exc()
        print("AUTO_ARM_FAIL", flush=True)
        sys.exit(1)
    print("AUTO_ARM_PASS", flush=True)


if __name__ == "__main__":
    main()
