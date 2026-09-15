"""The ``tilerl bench`` commands (architecture L5; moved out of cli.py).

cli.py keeps only command registration/dispatch. Heavy imports stay inside the
handlers so importing this module (and thus ``tilerl --help``) stays light.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def cmd_bench_calibrate(args: argparse.Namespace) -> None:
    """Measure one card's HBM bandwidth and bf16 peak and append BOTH rows to the bench
    ledger. Cuda-only: off a card there is nothing to measure, and the roofline then
    stays pending-remote (a datasheet number is refused, not substituted)."""
    from . import calibration as cal

    if args.card is None:
        sys.exit("error: --calibrate needs --card N (the physical GPU to measure)")
    import torch

    if not torch.cuda.is_available():
        sys.exit(
            "error: --calibrate is cuda-only (large D2D copy + bf16 GEMM, CUDA events); "
            "run on the card: TILERL_TARGET=cuda tilerl bench --calibrate --card N. "
            "Off cuda the roofline prints pending-remote, never a datasheet number.")
    rows = cal.calibrate_rows(args.card)
    cal.append_rows(rows)
    for r in rows:
        print(f"appended {r['metric']:<18} {r['value']:10.2f} {r['unit']:<8} "
              f"{r['device']['name']} card {r['device']['card']} -> {cal.store_path()}")

def cmd_bench_kernels(args: argparse.Namespace) -> None:
    """Print the per-kernel roofline table for one decode tick, no GPU required.

    Bytes/flops render from the declarations at every batch; when a calibration row for
    this device is in the ledger, ``bound`` is the measured roofline lower bound
    (max(bytes/bw, flops/peak)) and the actual registry kernel's ms fills on cuda.
    Without a row the measured columns print pending-remote — bandwidth and peak are
    measured, never datasheet.
    """
    from . import calibration as cal
    from . import config as config_mod
    from . import kernel_cost
    from .model import checkpoint_weight_faces, require_checkpoint_matches
    from .precision import fp8_block_dev, fp8_dev, kv_format, nvfp4, nvfp4_dev, nvfp4_dev_b32

    cfg = config_mod.qwen38_27b() if args.model == "qwen38-27b" else config_mod.tiny()
    if args.checkpoint:
        require_checkpoint_matches(cfg, args.model, args.checkpoint)
    faces = checkpoint_weight_faces(cfg, args.checkpoint) if args.checkpoint else None
    face_label = {
        nvfp4: "nvfp4", nvfp4_dev: "nvfp4", nvfp4_dev_b32: "nvfp4",
        fp8_block_dev: "fp8blk", fp8_dev: "fp8",
    }
    src = f"checkpoint {args.checkpoint}" if faces is not None else "nvfp4 weights (config face)"

    def face_cell(r):
        return face_label.get(r.get("face"), "bf16")

    # The device name the floors are keyed on. Cuda reads the real card; off cuda there
    # is no measured floor (a --device-name override exists only for debugging the join).
    device_name = args.device_name
    on_cuda = False
    uuid = None
    if device_name is None:
        import torch

        on_cuda = torch.cuda.is_available()
        if on_cuda:
            device_name = torch.cuda.get_device_name(0)
            uuid = str(torch.cuda.get_device_properties(0).uuid)
        else:
            device_name = f"{cfg.name}-cpu"
    floors = cal.calibration(cal.load_rows(), device_name, uuid)
    # On cuda, time the actual registry GEMM kernel per row so %bound is measured; the
    # backend is built once. Off cuda (or for non-GEMM rows) ms/%bound stay pending.
    # Spec dims come from the config (quant face does not change a linear's dims); the
    # timed kernel itself is resolved per row from the row's face in time_row_ms, so a
    # collapsed name split by face (some layers fp8, some nvfp4) times the right kernel.
    backend = None
    spec_by_name: dict[str, tuple] = {}
    if on_cuda and floors is not None:
        from tilerl_kernels.backend import get_backend

        from .model import param_specs

        backend = get_backend()
        spec_by_name = {k.split(".")[-1]: tuple(v) for k, v in param_specs(cfg).items()}

    def render(rows: list[dict], label: str, b: int, s: int, timed_s: int):
        """``b,s`` name the PRICED tick (bytes/flops rows); ``timed_s`` is the query
        rows per launch the timed GEMM runs on — s on prefill, 1 per request on a
        decode tick (decode streams one new token per row, not s; timing b*s timed a
        fat prefill GEMM and printed a fictional ~10x decode tick). lm_head runs once
        per request, so it times at M=b even when timed_s=1."""
        print(f"# {cfg.name} {label}, fp8 KV, {src}, floor device={device_name}")
        if floors is None:
            print("# (no calibration row for this device: ms/bound/%bound pending-remote)")
        tb_sum = tf_sum = 0
        # measured tick totals: Σ count × per-launch ms/bound. Only timed rows enter.
        ms_sum = bound_sum = 0.0
        timed_launches = 0
        for r in rows:
            by_one, fl_one = r["bytes"], r["flops"]          # ONE launch
            by, fl = by_one * r["count"], fl_one * r["count"]  # whole tick
            tb_sum += by
            tf_sum += fl
            face = f"{face_cell(r):>7}"
            if floors is None:
                print(f"{r['name']:<26} {r['count']:>5} {r['shape']:>22} {face} "
                      f"{by:>12,} {fl:>10,} {'pending':>11} {'pending':>11} {'pending':>11}")
                continue
            peak = cal.row_peak_tflops(floors, r, b, timed_s)
            # Per-launch bound: the timed kernel is ONE launch at the row's exact shape,
            # so its roofline floor must be for one launch too. count scales the TOTAL
            # columns below, never the per-row ratio (bound and ms are the same workload).
            bound_one = (cal.bound_seconds(by_one, fl_one, floors["bw_gbs"], peak)
                         if peak is not None else None)
            ms = None
            if backend is not None and r["name"] in spec_by_name:
                # lm_head samples once per request (M=b); other rows at timed_s.
                mm = b if r["name"] == "lm_head" else timed_s
                ms = cal.time_row_ms(
                    {**r, "_spec": spec_by_name[r["name"]]}, backend, mm)
            if bound_one is None:
                bnd_col = f"{'pending':>9}ms"
            else:
                bnd_col = f"{bound_one * 1e3:9.3f}ms"
            if ms is None:
                print(f"{r['name']:<26} {r['count']:>5} {r['shape']:>22} {face} "
                      f"{by:>12,} {fl:>10,} {'pending':>11} {bnd_col:>11} {'pending':>11}")
            else:

                pct = (bound_one / (ms / 1000.0) * 100.0) if bound_one is not None else float("nan")
                print(f"{r['name']:<26} {r['count']:>5} {r['shape']:>22} {face} "
                      f"{by:>12,} {fl:>10,} {ms:>9.3f}ms {bnd_col:>11} {pct:>10.1f}%")
                if bound_one is not None:
                    # A kernel cannot beat its own measured ceiling: bound/ms > 100 with
                    # a non-cuda async/short-circuit bug is an instrument error, not data.
                    if pct > 100.0 + 1.0:
                        raise SystemExit(
                            f"bench --kernels: {r['name']} {face_cell(r).strip()} achieved "
                            f"{pct:.1f}% of its roofline floor (>100) — the timed kernel "
                            f"({ms:.3f} ms) did less work than the priced row "
                            f"({bound_one*1e3:.3f} ms/launch); instrument error, not a result.")
                    ms_sum += ms * r["count"]
                    bound_sum += bound_one * 1e3 * r["count"]
                    timed_launches += r["count"]
        if floors is not None and timed_launches:
            print(f"{'TIMED TOTAL':<26} {timed_launches:>5} {'':>22} {'':>7} "
                  f"{'':>12} {'':>10} {ms_sum:>8.1f}ms {bound_sum:>8.1f}ms "
                  f"{bound_sum / ms_sum * 100.0:>10.1f}%")

        return tb_sum, tf_sum

    print(f"{'kernel':<26} {'count':>5} {'shape':>22} {'face':>7} {'bytes':>12} "
          f"{'flops':>10} {'ms':>11} {'bound':>11} {'%bound':>11}")
    if args.prefill:
        pre = kernel_cost.TickShape(b=1, s=args.prefill, kv=kv_format(cfg.head_dim),
                                    weight=nvfp4, faces=faces)
        tb, tf = render(kernel_cost.prefill_rows(cfg, pre),
                        f"prefill S={args.prefill}", 1, args.prefill, args.prefill)
        print(f"{'PREFILL TOTAL':<26} {'':>5} {'':>22} {'':>7} {tb:>12,} {tf:>10,}")
        return
    batches = tuple(int(x) for x in args.batches.split(",")) if args.batches else (1, 8)
    for b in batches:
        tick = kernel_cost.TickShape(b=b, s=args.context, kv=kv_format(cfg.head_dim),
                                     weight=nvfp4, faces=faces)
        tb, tf = render(kernel_cost.tick_rows(cfg, tick),
                        f"decode tick B={b} s={args.context}", b, args.context, 1)
        print(f"{'TICK TOTAL':<26} {'':>5} {'':>22} {'':>7} {tb:>12,} {tf:>10,}")

    k_pages = getattr(args, "sparse_k", 0)
    if k_pages:
        # Sparse selection kernels (unit A derived rows): the scorer reads index keys
        # from HBM; the cold-page fetch rides PCIe. ms is pending until a sparse timing
        # fixture exists; the bounds are derived here. Rendered per B so the scorer read
        # scales with rows.
        import torch

        kv_fp8 = getattr(args, "kv_fp8", None)
        for b in batches:
            st = kernel_cost.TickShape(b=b, s=args.context, kv=kv_format(cfg.head_dim),
                                       weight=nvfp4, faces=faces)
            srows = kernel_cost.sparse_indexer_rows(
                cfg, st, k_pages=k_pages,
                kv_fp8=torch.float8_e4m3fn if kv_fp8 == "e4m3" else None)
            print(f"# sparse selection k_pages={k_pages} scorer={args.scorer} "
                  f"B={b}, floor device={device_name}")
            print(f"{'kernel':<26} {'count':>5} {'shape':>30} "
                  f"{'HBM bytes':>14} {'HBM bound':>11} {'PCIe bytes':>12} {'PCIe bound':>11}")
            for r in srows:
                hbm_b = r["bytes"] * r["count"]
                pcie_b = r.get("pcie_bytes", 0) * r["count"]
                hbm_bnd = (cal.bound_seconds(hbm_b, r["flops"] * r["count"],
                                             floors["bw_gbs"], floors["peak_tflops"]) * 1e3
                           if floors is not None else None)
                pcie_gbs = floors.get("pcie_gbs") if floors else None
                pcie_bnd = (pcie_b / (pcie_gbs * 1e9) * 1e3 if pcie_gbs and pcie_b else None)
                print(f"{r['name']:<26} {r['count']:>5} {r['shape']:>30} "
                      f"{hbm_b:>14,} "
                      f"{(f'{hbm_bnd:8.3f}ms' if hbm_bnd is not None else 'pending'):>11} "
                      f"{pcie_b:>12,} "
                      f"{(f'{pcie_bnd:8.3f}ms' if pcie_bnd is not None else 'pending'):>11}")

def cmd_bench(args: argparse.Namespace) -> None:
    if getattr(args, "calibrate", False):
        cmd_bench_calibrate(args)
        return
    if getattr(args, "kernels", False):
        cmd_bench_kernels(args)
        return
    views = [f"--{v}" for v in ("table", "readme", "regress", "questions", "collectors")
             if getattr(args, v, False)]
    if views or args.suite:
        import subprocess

        script = Path(__file__).resolve().parent.parent.parent / "scripts/bench_harness.py"
        cmd = [sys.executable, str(script), *views]
        if args.suite:
            cmd += ["--suite", args.suite]
        if args.source:
            cmd += ["--source", args.source]
        if args.gpu is not None:
            cmd += ["--gpu", str(args.gpu)]
        if args.batches:
            cmd += ["--batches", args.batches]
        sys.exit(subprocess.call(cmd))

    if args.name:
        import json
        import subprocess

        root = Path(__file__).resolve().parent.parent.parent
        reg = json.loads((root / "docs" / "bench-metrics.json").read_text())
        m = reg["metrics"].get(args.name)
        c = (m or {}).get("collector")
        if not c:
            why = (m or {}).get("why_no_collector", "no collector registered")
            raise SystemExit(f"no collector for {args.name!r}: {why}")
        sys.exit(subprocess.call(
            [sys.executable, str(root / c["script"]), *args.collector_args]))

    import torch
    from tilerl_kernels.backend import get_backend

    from . import engine as engine_mod
    from .build import build_model, build_serving_engine

    backend = get_backend()
    cfg, model = build_model(args.model, seed=0)
    engine = build_serving_engine(cfg, model, backend)
    gen = torch.Generator().manual_seed(0)

    def rand_ids(n: int) -> list[int]:
        return torch.randint(0, cfg.vocab_size, (n,), generator=gen).tolist()

    def run_to_done(req_id: int, max_ticks: int) -> None:
        for _ in range(max_ticks):
            engine.step()
            if req_id in engine.poll():
                return

    # Warmup at the timed prompt_len: the JIT specializes per shape; gen=2 compiles decode too.
    warmup_id = engine.submit(
        rand_ids(args.prompt_len),
        engine_mod.SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=2, seed=0),
    )
    run_to_done(warmup_id, max_ticks=16)

    req_id = engine.submit(
        rand_ids(args.prompt_len),
        engine_mod.SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=args.gen, seed=0),
    )
    t0 = time.perf_counter()
    engine.step()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    decode_ms: list[float] = []
    for _ in range(args.gen):
        t0 = time.perf_counter()
        engine.step()
        decode_ms.append((time.perf_counter() - t0) * 1000.0)
    assert req_id in engine.poll(), "bench request did not complete"

    decode_avg = sum(decode_ms) / len(decode_ms)
    print(
        f"tilerl bench: model={cfg.name} target={backend.target} "
        f"prompt_len={args.prompt_len} gen={args.gen}"
    )
    print(f"{'phase':<10} {'ms/tok':>12} {'tok/s':>12}")
    print(
        f"{'prefill':<10} {prefill_ms / args.prompt_len:>12.3f} "
        f"{1000.0 * args.prompt_len / prefill_ms:>12.1f}"
    )
    print(f"{'decode':<10} {decode_avg:>12.3f} {1000.0 / decode_avg:>12.1f}")
    print(f"prefill total: {prefill_ms:.1f} ms | decode total: {sum(decode_ms):.1f} ms")
