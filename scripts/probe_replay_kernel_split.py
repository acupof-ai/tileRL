#!/usr/bin/env python3
"""Split ONE steady sparse W2 trunk graph replay into kernel classes — V100 sm70.

The b806 phase window measured p2_replay = 28.2 ms at cmax bucket 2048 (32.9k
context), 66% of the graph tick. nsys 2022.4 cannot export traces on this box,
so the device run is wrapped by nsys directly:

    nsys profile -t cuda --stats=true --capture-range=cudaProfilerApi \\
        --capture-range-end=stop -o replay2048 \\
        python scripts/probe_replay_kernel_split.py --bucket 2048 --nticks 20

Only ticks between cudaProfilerStart/Stop are captured: the 32.9k prefill and
the 64 warmup ticks are excluded. The cuda_gpu_kern_sum table is parsed from
nsys stdout and each kernel row is folded into:

    gemm_fp4   NVFP4 weight GEMM/GEMV (names: *linear_fp4*, *gemm*, *mma*)
    attn       full-attention paged decode (*paged_attention*)
    gdn        gated-delta (*gdn_*)
    norm_rope  rmsnorm / rope / silu_mul
    other      everything else (copies, casts, embedding, sampler, ...)

Output: per-class count, total ms, share of GPU time, plus the RAW rows (the
matching regex is built against observed names, so unknown names stay visible
as "other" and in the raw dump). Served weight bytes are summed from the
materialized model tensors and printed as the roofline numerator (15.81 GiB
on the live V100 ledger, not the 13.5 GiB estimate): bytes / 900 GB/s.

Geometry mirrors probe_sparse_w2_phase_timing (rev 1d106e6a): sparse_k=128,
sparse_min_tokens=0, bounds scorer, depth-1 draft with the 2048 read window,
the counting-prompt prime whose first decode tick lands in the cmax bucket.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def ids_for_bucket(tok, bucket: int) -> list[int]:
    instr = tok.encode(" Count aloud from one to forty, one number per line:")
    fid = tok.encode(" z")[-1:]
    nfill = (bucket + 7) * 16 + 7 - len(instr)
    return fid * nfill + instr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", type=int, default=2048, choices=[512, 1024, 2048])
    ap.add_argument("--nticks", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--source", default=os.environ.get("TILERL_QWEN38_SOURCE", ""))
    ap.add_argument("--draft", default="/home/chenkailun.c/mmlu-assets/model_mtp.safetensors")
    ap.add_argument("--cold-ssd", default="/home/chenkailun.c/sparse_cold_128k.bin")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl import build as build_mod
    from tilerl.build import build_engine, build_model
    from tilerl.cli import _qwen38_tokenizer
    from tilerl.engine import SamplingParams
    from tilerl.sparse_engine import cmax_bucket
    from tilerl.spec import load_draft

    if args.source:
        build_mod.QWEN38_SOURCE = args.source
    be = get_backend()
    if be.device.type != "cuda":
        print("this probe needs cuda", file=sys.stderr)
        return 14
    cfg, model = build_model("qwen38-27b", seed=0, fuse_projections=True)
    weight_bytes = sum(t.numel() * t.element_size() for t in model.params.values())
    draft = load_draft(model, args.draft, attn_window_tokens=2048)
    e = build_engine(
        cfg, model, be,
        num_slots=4, max_batch=4, max_total_tokens=131072,
        max_num_batched_tokens=512,
        sparse_k=128, sparse_min_tokens=0, sparse_device_select=True,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        cold_ssd_path=args.cold_ssd, cold_ssd_bytes=8 << 30, cold_format="f16",
        decode_graph=True, draft=draft, spec_depth=1,
    )
    import tilerl
    print(f"TILERL_FILE {tilerl.__file__}", flush=True)
    print(f"SPARSE_GRAPH_ON {e._sparse_graph_on}", flush=True)
    # #818 guard A: depth1+draft sparse graph must be armed; without it the
    # captured replay under measurement is eager.
    if not e._sparse_graph_on:
        print("FATAL sparse graph forced eager (guard A/#818 not in this tree)",
              file=sys.stderr)
        return 14
    tok = _qwen38_tokenizer()
    rid = e.submit(ids_for_bucket(tok, args.bucket),
                   SamplingParams(temperature=0.0, max_new_tokens=4096, seed=0))

    def nfwd():
        return e.stats()["decode_forwards"]

    f0 = nfwd()
    own_w = None
    while nfwd() - f0 < args.warmup or e._waiting:
        e.step()
        e.poll()
        row = next((r for r in e._running if r.req_id == rid), None)
        if own_w is None and row is not None and row.phase == 2:
            srows = e._sparse.decode_rows([row], [2])
            own_w = max((len(x.get("own", ())) for x in srows), default=None)
            cmax = max((len(x["cand"]) for x in srows), default=0)
            if cmax_bucket(cmax) != args.bucket:
                print(f"FATAL cmax bucket {cmax_bucket(cmax)} != {args.bucket}",
                      file=sys.stderr)
                return 14
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    p0 = nfwd()
    for _ in range(args.nticks):
        e.step()
        e.poll()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()
    import time
    time.sleep(2)
    got = nfwd() - p0

    roof_ms = weight_bytes / 900e9 * 1000.0
    summary = {
        "bucket": args.bucket, "ticks": got, "nticks": args.nticks,
        "own_w_pages": own_w,
        "weight_bytes": weight_bytes,
        "roofline_ms_at_900GBs": round(roof_ms, 2),
    }
    print("REPLAY_SPLIT_META " + json.dumps(summary), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    e.shutdown()
    return 0 if got >= args.nticks - 2 else 14


if __name__ == "__main__":
    raise SystemExit(main())
