"""tilerl CLI. Heavy imports live inside the handlers so ``--help`` stays instant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import ledger as _ledger
from .bench import cmd_bench
from .build import (
    DEFAULT_SPARSE_K,
    MODEL_NAMES,
    build_model,
    build_serving_engine,
)
from .build import (
    NO_WEIGHTS as _NO_WEIGHTS,
)
from .build import (
    QWEN38_SOURCE as _QWEN38_SOURCE,
)
from .build import (
    kv_fp8_dtype as _kv_fp8,
)
from .recipes import RECIPES, flags
from .train import cmd_train


def _progress(as_json: bool):
    """The run's progress printer: stdout normally, STDERR under --json.

    Not a no-op under --json, which is what it used to be. A `--json` eval-only run then
    printed nothing at all until the manifest at the end, and with the eval arm writing
    `eval-{tag}.jsonl` only on completion and the run directory holding just the
    pre-written manifest, the log was the sole progress signal -- so a healthy 67-minute
    27B eval was indistinguishable from a hung one for its whole duration, and two
    sessions began treating one as dead (errors/2026-09-08-a-silence-with-no-writer.md).

    stderr, not stdout, because `--json` exists so a caller can parse the manifest:
    `tests/test_recipes.py:47` reads `out[out.index("{"):]`, so anything containing a
    brace ahead of the manifest breaks it. stdout is already not a lone object -- the
    same test records that TileLang writes kernel-cache warnings there from C++ -- which
    is why the manifest is found by first brace rather than by parsing the whole stream,
    and why moving OUR lines off stdout is what keeps that workable.
    """
    if not as_json:
        return print
    return lambda *a, **k: print(*a, **{**k, "file": sys.stderr, "flush": True})

def _qwen38_tokenizer():
    """The 27B tokenizer, with the same hint as its weights: a bare hub id 401s."""
    from .tokenizer import get_tokenizer

    try:
        return get_tokenizer(_QWEN38_SOURCE)
    except Exception as exc:
        # HF's 401 body is a dozen lines of auth advice; the first names the cause.
        # Some exceptions (MemoryError) stringify empty, so splitlines() can be [].
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        print(f"error: could not load the Qwen3-27B tokenizer from {_QWEN38_SOURCE!r}: "
              f"{first}\n{_NO_WEIGHTS}", file=sys.stderr)
        sys.exit(1)

def _device_free(args, backend) -> int:
    """--device-free bytes, or the CUDA card's free; off cuda without the flag refuse."""
    import torch

    if args.device_free is not None:
        return args.device_free
    if backend.device.type == "cuda":
        return int(torch.cuda.mem_get_info()[0])
    sys.exit("error: --dry-run budget rows need --device-free BYTES off CUDA "
             "(there is no card to read mem_get_info from)")

def _sparse_spec(args, cfg) -> dict | None:
    """--sparse-k K: price the sparse-KV ledger. Concurrent rows = --max-batch,
    slots and chunk budget come from the same args build_engine uses (a 0 chunk
    budget means the engine default 512); context per row = --max-ctx or the
    model ceiling; scorer picks learned index keys vs Quest bounds."""
    k = getattr(args, "sparse_k", 0)
    if not k:
        return None
    return {
        "num_rows": args.max_batch,
        "num_slots": args.slots,
        "context": int(args.max_ctx or cfg.max_position_embeddings),
        "k_pages": k,
        "max_num_batched_tokens": args.max_batched_tokens or 512,
        "scorer": getattr(args, "scorer", "index"),
    }

def _dry_run_checkpoint(args, backend) -> None:
    """--dry-run --checkpoint DIR: price the ledger from safetensors HEADERS alone
    (the served faces model.checkpoint_weight_faces derives), blocks fitted
    arithmetically after the fixed weights + state-pool bytes. Same table as the built
    --dry-run; peak is None so transient/totals are suppressed."""
    import torch as _torch

    from . import config as config_mod
    from .engine import _graph_on
    from .memory import (
        _state_bytes,
        fit_num_blocks,
        format_memory_table,
        memory_table,
        plan,
        weight_row_faces,
    )
    from .model import checkpoint_weight_faces, require_checkpoint_matches
    from .precision import f32

    cfg = {"tiny": config_mod.tiny, "tiny-agent": lambda: config_mod.tiny(65536),
           "qwen38-27b": config_mod.qwen38_27b}[args.model]()
    require_checkpoint_matches(cfg, args.model, args.checkpoint)
    faces = checkpoint_weight_faces(cfg, args.checkpoint)
    device_free = _device_free(args, backend)
    # state slots add the decode-graph replay row on cuda (auto-on); the fit happens
    # after weights and that pool are resident, so subtract the same fixed bytes.
    state_slots = args.slots + int(_graph_on(backend, None))
    free_after_fixed = max(
        0, device_free - weight_row_faces(faces).n - _state_bytes(cfg, state_slots, f32))
    kv_fp8 = _kv_fp8(args.kv_fp8)
    num_blocks = args.blocks or fit_num_blocks(cfg, free_after_fixed, _torch.bfloat16, kv_fp8)
    rows = plan(cfg, None, free_after_fixed, num_slots=state_slots, num_blocks=num_blocks,
                state_dtype=_torch.float32, kv_io=_torch.bfloat16, kv_fp8=kv_fp8,
                explicit_state_budget=args.state_bytes, dram_budget=args.dram_bytes,
                ckpt_faces=faces, sparse=_sparse_spec(args, cfg))
    table = memory_table(rows, {}, None)
    if args.json:
        print(json.dumps(table, indent=1))
    else:
        print(f"tilerl serve --dry-run: model={cfg.name} checkpoint={args.checkpoint} "
              f"target={backend.target} device_free {device_free/1e6:.0f} MiB")
        print(format_memory_table(table))

def _apply_device_reserve(args, backend) -> None:
    """Set the held-VRAM fence before weights load (the first allocator cudaMalloc).

    --device-reserve-mib is one knob with one path: the process memory fraction
    HOLDS the floor (an over-fence cudaMalloc raises catchable OOM that the
    supervisor restarts), and it caps mem_get_info, so build's peak-live block
    trim must not run too (that would double-charge). 0 is a full-card no-op and
    skips the call entirely. Off cuda there is no fraction to set."""
    reserve = int(getattr(args, "device_reserve_mib", 0) or 0) * 1024 * 1024
    if not reserve or backend.device.type != "cuda":
        return
    import torch

    from .memory import reserve_memory_fraction

    total = torch.cuda.get_device_properties(backend.device).total_memory
    frac = reserve_memory_fraction(total, reserve)
    torch.cuda.set_per_process_memory_fraction(frac, backend.device)
    print(f"serve: device memory fraction {frac:.5f} "
          f"(reserve {reserve / 2**20:.0f} MiB of {total / 2**30:.2f} GiB held)", flush=True)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    from tilerl_kernels.backend import get_backend

    from .server import create_app, get_tokenizer

    backend = get_backend()
    _apply_device_reserve(args, backend)
    if args.checkpoint:
        if not args.dry_run:
            sys.exit("error: --checkpoint is a --dry-run header-only query; "
                     "drop --checkpoint to serve a built-in model")
        _dry_run_checkpoint(args, backend)
        return
    cfg, model = build_model(args.model, seed=0, fuse_projections=True)
    draft = None
    if args.draft:
        from .spec import load_draft

        draft = load_draft(model, args.draft)
    # Before the engine: it takes the decode for stop sequences.
    tokenizer = _qwen38_tokenizer() if args.model == "qwen38-27b" else get_tokenizer(None)
    try:
        engine = build_serving_engine(cfg, model, backend,
                           draft=draft, depth=args.depth, slots=args.slots,
                           blocks=args.blocks, max_ctx=args.max_ctx,
                           max_batch=args.max_batch, dram_bytes=args.dram_bytes,
                           state_bytes=args.state_bytes, kv_fp8=args.kv_fp8,
                           cold_format=getattr(args, "cold_format", ""),
                           kv_store=getattr(args, "kv_store", ""),
                           cold_ssd_path=getattr(args, "cold_ssd_path", ""),
                           cold_ssd_bytes=getattr(args, "cold_ssd_bytes", 0),
                           decode=tokenizer.decode,
                           max_batched_tokens=args.max_batched_tokens,
                           sparse_k=getattr(args, "sparse_k", 0),
                           scorer=getattr(args, "scorer", "bounds"),
                           kv_cold_bytes=getattr(args, "kv_cold_bytes", 0),
                           decode_graph=getattr(args, "decode_graph", None),
                           sparse_min_tokens=getattr(args, "sparse_min_tokens", 0),
                           sparse_prefill_tokens=getattr(args, "sparse_prefill_tokens", 0),
                           device_reserve_mib=getattr(args, "device_reserve_mib", 0))
    except Exception as build_exc:
        # Startup OOM (e.g. a fraction too small for weights): exit fatally instead
        # of half-starting, so the supervisor's marker path restarts cleanly.
        import torch as _torch_build

        if isinstance(build_exc, _torch_build.cuda.OutOfMemoryError):
            from .engine import fatal_device_exit

            fatal_device_exit(build_exc)
        raise
    app = create_app(engine, tokenizer, model_name=cfg.name)
    # --dry-run: build (which materializes and fits) then print the memory ledger and stop,
    # never bind the HTTP port. --json prints the rows for the cost-model tooling. The budget
    # rows need device_free: the card's free on CUDA, else --device-free (bytes) is required.
    if args.dry_run:
        from .memory import format_memory_table, measured_peak_bytes, memory_table, plan

        if getattr(args, "sparse_k", 0) and getattr(args, "checkpoint", ""):
            # Header-only --checkpoint prices the derived ledger; a built sparse engine
            # shows its LIVE rows instead, so the two are mutually exclusive.
            sys.exit("error: --sparse-k with a built engine prints the live sparse ledger; "
                     "the derived --checkpoint table is a separate path — drop --checkpoint "
                     "to run the sparse engine, or drop --sparse-k for the header-only table")
        device_free = _device_free(args, backend)
        kv, sp = engine._kv, engine._states
        if getattr(engine, "_sparse", None) is not None:
            # A built sparse engine reports its LIVE rows (bounds/hot/cold from actual
            # pages); the dense plan() below would price a kv_pool that isn't allocated.
            table = engine.stats()["memory"]
        else:
            draft_layers = (engine._draft.cfg.num_layers
                            if getattr(engine._draft, "kv", None) is not None else 0)
            rows = plan(cfg, model.params, device_free, num_slots=sp.num_slots,
                        num_blocks=kv.num_blocks, state_dtype=sp.states.dtype,
                        kv_io=kv.dtype, kv_fp8=kv.kv_fp8, draft_layers=draft_layers)
            # Same table (incl. transient + totals) /health serves from engine.stats()["memory"].
            measured = {r["owner"]: r.get("measured") for r in engine.stats()["memory"]
                        if r.get("measured") is not None and r["kind"] == "allocation"}
            peak = measured_peak_bytes(backend)
            table = memory_table(rows, measured, peak)
        if args.json:
            print(json.dumps(table, indent=1))
        else:
            print(f"tilerl serve --dry-run: model={cfg.name} target={backend.target} "
                  f"device_free {device_free/1e6:.0f} MiB")
            print(format_memory_table(table))
        if getattr(args, "record_residency", False):
            # Append peak + its static/transient split to the same ledger the kernel
            # roofline (%bound) lives in, through benchrec's single schema writer. A
            # card-less sm* row is rejected by benchrec, so refuse off cuda instead of
            # recording a residency row with a fabricated target/card.
            import torch as _torch2

            if not _torch2.cuda.is_available():
                sys.exit(
                    "error: --record-residency is cuda-only (device residency belongs to "
                    "the card; the CPU tiny cell has no measured peak). Run on the card: "
                    "TILERL_TARGET=cuda tilerl serve --dry-run --record-residency")
            from .ledger import append_residency, residency_row

            by = {r["owner"]: r["derived"] for r in table if r["kind"] == "allocation"}
            static = sum(v for k, v in by.items() if k != "transient")
            transient = by["transient"]
            card = _torch2.cuda.current_device()
            name = _torch2.cuda.get_device_name(card)
            uuid = str(_torch2.cuda.get_device_properties(card).uuid)
            rid = append_residency(
                residency_row(name, card, peak, static, transient, backend.arch,
                              model=cfg.name, uuid=uuid))
            print(f"appended device_resident_bytes peak {peak:,} = static {static:,} + "
                  f"transient {transient:,} ({name}) -> {rid}")
        return
    # Print the pool: with --blocks 0 it is fitted to the card, so this is the served
    # context ceiling and the one number a 32 GB card gets wrong silently.
    from .kv_cache import BLOCK_TOKENS

    kv = engine.stats()["blocks_total"]
    print(f"tilerl serve: model={cfg.name} target={backend.target} "
          f"kv={kv} blocks = {kv * BLOCK_TOKENS} tokens")
    if args.warmup:
        # One capture per (bucket x chain width); generating tokens instead is a
        # lottery with no floor, since the draft's confidence sets each width.
        t0 = time.perf_counter()
        n = engine.precapture()
        print(f"tilerl serve: {n} decode graphs in {time.perf_counter() - t0:.0f}s")
    engine.run()
    print(f"tilerl serve: http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        engine.shutdown()


def cmd_generate(args: argparse.Namespace) -> None:
    # One process per device: an in-process wrapper serialises every tick on the GIL.
    from .generate import generate

    devices: list[int] = []
    for part in args.devices.split(","):  # "0-7" or "0,1,2" or "0-3,6"
        if "-" in part:
            lo, hi = part.split("-")
            devices.extend(range(int(lo), int(hi) + 1))
        else:
            devices.append(int(part))
    stats = generate(
        prompts=args.prompts, out=args.out, devices=devices,
        source=args.source, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        max_batch=args.max_batch,
    )
    print(json.dumps(stats))


def cmd_merge(args: argparse.Namespace) -> None:
    from .ledger import (
        commit,
        find_run_for_artifact,
        new_manifest,
        now,
        read_manifest,
        runs_root,
        write_manifest,
    )
    from .merge import merge_checkpoints

    # `out` is deliberately not in the id: merging the same inputs is the same run
    # whether the bytes land here or there, so a repeat returns the recorded run.
    m = new_manifest("merge", {"base": args.base, "specialists": list(args.specialists),
                               "method": args.method, "commit": commit()})
    prev = read_manifest(runs_root(), m["id"])
    if prev and prev["finished"] and not args.force:
        print(json.dumps(prev, indent=1) if args.json
              else f"run {prev['id']} already finished; --force reruns")
        return
    # Link lineage through the run that WROTE each checkpoint: the dir carries no run
    # id, only the producing manifest's artifacts. An unlinked dir simply contributes no
    # parent; ids are never guessed from a path component.
    paths = [args.base, *args.specialists]
    m["parents"] = [pid for pid in (find_run_for_artifact(runs_root(), p) for p in paths)
                    if pid is not None]
    n = merge_checkpoints(args.base, args.specialists, args.out, method=args.method)
    m["metrics"], m["artifacts"], m["finished"] = {"tensors": n}, {"out": args.out}, now()
    write_manifest(runs_root(), m)
    if args.json:
        print(json.dumps(m, indent=1))
    else:
        print(f"merged {len(args.specialists)} specialists ({args.method}) -> {args.out}  run {m['id']}")

def _device_ledger_rows():
    """One section per device from the bench store: newest calibration pair + newest
    residency. Reads calibration.load_rows (the store's one reader); no parser here."""
    from . import calibration as cal

    return cal.device_sections(cal.load_rows())

def _format_device_ledger(sections: list[dict]) -> str:
    lines = ["devices (newest measured floor + residency per card):"]
    for s in sections:
        bw, pk, res = s["hbm_bw_gbs"], s["bf16_peak_tflops"], s["residency"]
        pk_name = "bf16_peak_tflops"
        if pk is None:  # pre-Ampere cards are calibrated at f16 (no bf16 tensor path)
            pk, pk_name = s["f16_peak_tflops"], "f16_peak_tflops"
        lines.append(f"  {s['device']}")
        if bw is None or pk is None:
            lines.append("    hbm_bw / tensor peak: pending-remote (run bench --calibrate)")
        else:
            lines.append(
                f"    hbm_bw_gbs {bw['value']:.1f} (commit {str(bw['commit'])[:8]}, "
                f"{bw['date']})  {pk_name} {pk['value']:.1f} "
                f"(commit {str(pk['commit'])[:8]}, {pk['date']})")
        if res is None:
            lines.append("    residency: pending-remote (run serve --dry-run --record-residency)")
        else:
            mib = 2**20
            lines.append(
                f"    resident {res['peak'] / mib:10.2f} MiB = static "
                f"{res['static'] / mib:.2f} + transient {res['transient'] / mib:.2f} "
                f"(commit {str(res['commit'])[:8]}, {res['date']})")
    if len(lines) == 1:
        lines.append("  (no calibration or residency rows in the bench store)")
    return "\n".join(lines)

def cmd_ledger(args: argparse.Namespace) -> None:
    from .ledger import lineage, list_runs, runs_root, time_to_score

    if args.devices:
        sections = _device_ledger_rows()
        print(json.dumps(sections, indent=1) if args.json
              else _format_device_ledger(sections))
        return
    runs = lineage(runs_root(), args.lineage) if args.lineage else list_runs(runs_root())
    if args.time_to_score is not None:
        # A row per run, because the answer is per run: the target's step, the interval it
        # was crossed in, and the cumulative training seconds at that point.
        rows = [(m, time_to_score(m, args.time_to_score)) for m in runs]
        if args.json:
            print(json.dumps([{"id": m["id"], **(r or {"reached": None})}
                              for m, r in rows], indent=1))
            return
        for m, r in rows:
            if r is None:
                print(f"{m['id']}  no eval_curve (run with --eval-every N to record one)")
            elif r["reached"]:
                # The interval, not an interpolated step: the target was crossed somewhere
                # in (after_step, step] and only the right end was measured. `correct/total`
                # and the SE come along because the curve scores a SUBSET -- 0.61 on 20 rows
                # is not 0.61 on 500, and the SE is the point's own rate, not p=0.5's.
                # The dip note is printed, not folded into `reached`: a transient crossing
                # and a run that held are different facts, and suppressing the step would
                # turn one noisy point into "never reached".
                dip = ("" if r.get("held", True) else
                       f"  [fell below {args.time_to_score:.3g} again at step "
                       f"{r['dipped_at']}: a transient crossing, not arrival]")
                print(f"{m['id']}  score {args.time_to_score:.3g} reached at step "
                      f"{r['step']} (in ({r['after_step']}, {r['step']}]), "
                      f"{r['secs']:.1f}s cumulative, scored {r['score']:.3g} "
                      f"({r['correct']}/{r['total']}){dip}{_ledger.se_note(r)}")
            else:
                print(f"{m['id']}  score {args.time_to_score:.3g} NOT reached in "
                      f"{r['steps_run']} steps / {r['secs']:.1f}s; best "
                      f"{r['best']:.3g} on {r['n']} rows{_ledger.se_note(r)}")
        return
    text = "\n".join(_format_ledger_run(m) for m in runs)
    print(json.dumps(runs, indent=1) if args.json else text)

def _format_ledger_run(m: dict) -> str:
    """The one-line run, then its engine occupancy table when the manifest carries
    one (#475), indented under the row; older/manifest-less runs print only the line."""
    from .ledger import format_run
    from .memory import format_memory_table

    line = format_run(m)
    mem = (m.get("engine") or {}).get("memory")
    if not mem:
        return line
    block = "\n".join("    " + ln for ln in format_memory_table(mem).splitlines())
    return f"{line}\n{block}"

def _build_parser(recipe: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tilerl",
        description="tileRL: TileLang inference + training (CPU/CUDA/Metal).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="start the OpenAI-compatible HTTP server")
    p_serve.add_argument("--model", choices=MODEL_NAMES, default="tiny")
    p_serve.add_argument("--dry-run", action="store_true",
                         help="build the engine, print the memory ledger (derived vs measured "
                              "occupancy + budget rows), and exit without starting the HTTP server")
    p_serve.add_argument("--device-free", type=int, default=None,
                         help="free device bytes for --dry-run budget rows; required off CUDA "
                              "(defaults to mem_get_info on CUDA)")
    p_serve.add_argument("--checkpoint", default="", metavar="DIR",
                         help="with --dry-run: price the ledger from this checkpoint's "
                              "safetensors headers only (no load, no card); weights come "
                              "from the real nvfp4/fp8 tensor names, not config")
    p_serve.add_argument("--json", action="store_true",
                         help="with --dry-run, print the memory rows as JSON")
    p_serve.add_argument("--record-residency", action="store_true",
                         help="with --dry-run, append the measured resident peak and its "
                              "static/transient split to the bench ledger (benchrec)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--draft", help="MTP/NextN head safetensors: speculative decode. For "
                                        "Qwen3.8-27B-NVFP4 the mtp.* keys live in "
                                        "model_mtp.safetensors, so pass that file.")
    p_serve.add_argument("--depth", type=int, default=3,
                         help="drafts per row per tick; 3 fills the sm70 verify ladder's "
                              "4-row rung exactly (spec.LADDER_WIDTHS) — 4 spills to the "
                              "8-row rung and measured slower than no speculation")
    p_serve.add_argument("--slots", type=int, default=8,
                         help="GDN state slots. A slot is held from submit to finish, so "
                              "this must be >= --max-batch or that concurrency is "
                              "unreachable; above it, each slot is one more queued "
                              "request instead of a 503. They are expensive: measured on "
                              "a 32GB V100 with a draft, slots 8/16 fit 42384/12112 "
                              "tokens of context (step_states scales slots x width)")
    p_serve.add_argument("--blocks", type=int, default=0,
                         help="KV blocks (16 tokens each); 0 = fit the pool to the card, "
                              "capped by --max-ctx. Measured 2026-09-04: 3927 blocks = "
                              "62832 tokens on a 32GB V100 at --slots 3 with a draft")
    p_serve.add_argument("--max-ctx", type=int, default=0,
                         help="cap served context (0 = the model's own limit); pairs with "
                              "--blocks so a request cannot outgrow the pool")
    p_serve.add_argument("--state-bytes", type=int, default=0,
                         help="HBM budget in bytes for resident GDN snapshots (0 = a quarter "
                              "of free memory, the default). Without it the tiers' pressure "
                              "regime is unreachable: a quarter of free on an H20 is 17.9 GiB, "
                              "116 snapshots at 157 MiB each, so DRAM pressure starts near 115 "
                              "concurrent agent sessions and no benchable load reaches it")
    p_serve.add_argument("--dram-bytes", type=int, default=0,
                         help="host budget in bytes for demoted GDN snapshots (0 = off, the "
                              "default). Turn it on only when concurrent sessions outnumber "
                              "the snapshots HBM holds (free/4, which on a 32GB V100 is 9 at "
                              "144 MiB each): measured 2 sessions -> 0 promotions, 9 -> 17, "
                              "12 -> 24. Below that threshold it is 1.51x WORSE on wall "
                              "clock -- one conversation re-reads only its newest entry, so "
                              "the LRU snapshot a demotion picks is never asked for again "
                              "(43 demotions, 0 promotions). Read /health's dram_promotions "
                              "to see whether the workload crossed it, and dram_budget to "
                              "see the tier is on at all")
    p_serve.add_argument("--kv-cold-bytes", type=int, default=0,
                         help="pinned-host budget for sparse-KV cold pages. Auto-sized to the "
                              "whole context when --sparse-k is set and this is 0")
    p_serve.add_argument("--device-reserve-mib", type=int,
                         default=int(os.environ.get("TILERL_DEVICE_RESERVE_MIB", "0")),
                         metavar="MIB",
                         help="hold this many MiB of device VRAM free via a process memory "
                              "fraction set before weights load; an over-fence cudaMalloc then "
                              "raises catchable OOM (supervisor restart) instead of blocking at "
                              "the edge (sm70 wedge headroom). 0 (default) = full card, no-op.")
    p_serve.add_argument("--kv-fp8", choices=["e4m3", "e5m2"], default="",
                         help="store the KV planes in fp8: 65536 -> 33280 bytes per token at the "
                              "27B's 16 planes x 4 heads x 256, a 1.969x saving, the 0.031 being "
                              "one f32 scale per (plane, block, kv_head, token). Off by default: "
                              "the pool and the writers are fp8 but the attention kernels still "
                              "read a dequantized plane, so this is correctness-complete and not "
                              "yet a bandwidth win, and no decode tok/s figure exists "
                              "(docs/design-fp8-kv.md). Refused on sm70, whose fused write_tokens "
                              "has no fp8 twin and would scatter into a dequantized copy. e4m3 is "
                              "the default choice on measurement, not analogy: it beats e5m2 "
                              "1.89x on the worst element with nothing underflowing on tiny")
    p_serve.add_argument("--kv-store", default="",
                         help="directory of the cold-start KV boot store. A request whose "
                              "block-aligned prefix matches a context saved there "
                              "(Engine.save_boot) loads its pages and recurrent state "
                              "from disk and skips the prefill. Keyed by the prefix hash, "
                              "fingerprinted by weights, per-page checksummed.")
    p_serve.add_argument("--cold-format", choices=["f16", "native"], default="",
                         help="dtype of a demoted KV page in the host/SSD tier. Default "
                              "(unset): f16 on an f32 pool (sm70, whose host has half the "
                              "room and has no f16 attention path), native elsewhere. f16 "
                              "narrows K/V on the D2H copy and widens back on promote; "
                              "native keeps the pool dtype. fp8 scale planes stay f32.")
    p_serve.add_argument("--cold-ssd-path", default="", metavar="FILE",
                         help="cold KV pages past the --kv-cold-bytes host budget spill to "
                              "this one mmap'd file (block-id keyed, no index); promote "
                              "reads them back through the same path. Serving spill for one "
                              "process (sparse cold tier; the old dense --ssd-path tier was "
                              "removed 2026-09-14).")
    p_serve.add_argument("--cold-ssd-bytes", type=int, default=0, metavar="BYTES",
                         help="countable SSD spill capacity for admission (0 with "
                              "--cold-ssd-path = free space on the spill filesystem).")

    p_serve.add_argument("--sparse-k", type=int, default=DEFAULT_SPARSE_K, metavar="PAGES",
                         help=f"sparse-KV pages selected per row (default {DEFAULT_SPARSE_K} "
                              "= sparse OFF, dense engine; pass N>0 to opt in to Quest "
                              "selection with N pages + the always-on 8-page window). "
                              "Sparse is off by default pending the sm90 continuity gate. "
                              "With --dry-run --checkpoint it instead prices the derived "
                              "ledger. docs/design-sparse-kv.md")
    p_serve.add_argument("--sparse-min-tokens", type=int, default=0, metavar="N",
                         help="hybrid with --sparse-k: prompts up to N tokens run DENSE on "
                              "the captured graph and pin their whole context (no sparse "
                              "sharing); longer prompts run sparse. 0 = all sparse. A dense "
                              "prompt that does not fit the device KV pool routes sparse.")
    p_serve.add_argument("--sparse-prefill-tokens", type=int, default=0, metavar="N",
                         help="hybrid (--sparse-min-tokens): cap one sparse prefill tick "
                              "to N tokens so it stays ~1 s and dense waits are bounded "
                              "(default 192 ~= 1 s on the V100 sparse prefill rate)")
    p_serve.add_argument("--scorer", choices=["index", "bounds"], default="bounds",
                         help="sparse-KV page scorer: training-free Quest page bounds "
                              "(default) or the learned V4.1 indexer keys")
    p_serve.add_argument("--max-batch", type=int, default=8,
                         help="concurrent rows; drop to 2 for a single-user endpoint (a decode "
                              "graph is captured per bucket x chain width, so a lower "
                              "ceiling is fewer captures)")
    p_serve.add_argument("--max-batched-tokens", type=int, default=0,
                         help="token budget for one tick's chunked prefill; raise for "
                              "faster prefill at decode's expense (default: engine's 512)")
    p_serve.add_argument("--decode-graph", action="store_const", const=True, default=None,
                         help="force the captured decode tick on an arch whose AUTO path "
                              "disables it (sm70: dense capture fails there and poisons the "
                              "allocator; with --sparse-k the sparse graph is captured "
                              "instead). Informed opt-in for capture measurement.")
    p_serve.add_argument("--no-warmup", dest="warmup", action="store_false",
                         help="skip precapturing the decode graphs; the first real messages "
                              "then pay for them (1088 ms/token falling to 26 over six "
                              "requests on sm70)")
    p_serve.set_defaults(func=cmd_serve)

    p_train = sub.add_parser("train", help="SFT, --rl (GRPO) or --opd; --recipe for a gated flag set")
    p_train.add_argument("--model", choices=MODEL_NAMES, default="tiny")
    p_train.add_argument("--steps", type=int, default=20)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--opd", action="store_true",
                         help="on-policy distillation: the engine rolls out, LoRA adapters train")
    p_train.add_argument("--indexer-warmup", action="store_true",
                         help="learned-indexer KL warm-up on the frozen sparse-KV base (unit D)")
    p_train.add_argument("--indexer-corpus",
                         help="prepared span dir (scripts/prepare_indexer_corpus): 27B recall run")
    p_train.add_argument("--k-pages", type=int, default=128,
                         help="indexer recall: pages selected per row/source layer")
    p_train.add_argument("--indexer-di", type=int, default=128,
                         help="indexer head dim (V4.1 releases 128)")
    p_train.add_argument("--recall-threshold", type=float, default=0.9,
                         help="27B indexer recall acceptance: mean recall@k_pages after warm-up")
    p_train.add_argument("--indexer-control-corpus",
                         help="held-only span dir: cross-corpus control recall under the trained weights")
    p_train.add_argument("--q-samples", type=int, default=0,
                         help="27B recall: evaluate the dense teacher at this many seeded query "
                              "positions per span (256 -> O(256*T) instead of O(T^2)); 0 = all")
    p_train.add_argument("--q-min-pos", type=int, default=2048,
                         help="27B recall: sampled query positions are >= this (leaves missable pages)")
    p_train.add_argument("--held-spans", type=int, default=0,
                         help="27B recall: use a balanced round-robin subset of this many "
                              "held spans across lengths (16 over 8k/16k/32k -> 6/5/5); 0 = all")
    p_train.add_argument("--rl", action="store_true",
                         help="GRPO: the engine samples a group per prompt, a reward scores "
                              "them, the group mean is the baseline (no critic)")
    p_train.add_argument("--group", type=int, default=8, help="rollouts per prompt (--rl)")
    p_train.add_argument("--prompts-per-step", type=int, default=1,
                         help="--rl: prompts in one step, each normalised within its own "
                              "--group. A step is gradient-free only when EVERY group "
                              "ties, so 2x4 beats 1x8 at the same 8 rows: measured 1.21x "
                              "more gradient-bearing steps on MATH level 5, 1.34x on "
                              "GSM8K's 0.81 tie fraction. The step is "
                              "--prompts-per-step x --group rows, which is what the "
                              "engine is sized for.")
    p_train.add_argument("--micro", type=int, default=1,
                         help="--rl: rows per backward, gradients accumulated to one "
                              "update (0 = the whole group). The normalizer stays the "
                              "batch's, so this is the same update, not a smaller one. "
                              "Default 1 because 0 does not fit the production shape: at "
                              "group 8 / cap 2048 on one H20, 0 peaks 88.21 GiB and OOMs "
                              "on step 2, against 44.55 GiB and 3/3 steps at 1")
    p_train.add_argument("--max-new-tokens", type=int, default=32, help="rollout length")
    p_train.add_argument("--data", help="JSONL {prompt, answer}: real prompts, exact-match "
                         "reward on the last number (scripts/gsm8k_jsonl.py)")
    p_train.add_argument("--reward", choices=["number", "boxed"], default="number",
                         help="how --data's answer is matched: last number (GSM8K) or the "
                              "last \\boxed{} (MATH, scripts/math_jsonl.py)")
    p_train.add_argument("--length-penalty", type=float, default=0.1,
                         help="RL reward subtracts this times completion/cap, so an all-right "
                              "group is ordered by length instead of tying at zero advantage. "
                              "A switch, not a dial: it CANCELS exactly in an all-right group "
                              "(the advantage divides by the group std), and above "
                              "cap/(cap-1) a short wrong answer can outrank a long right one. "
                              "Set 0 to disable")
    p_train.add_argument("--temperature", type=float, default=None,
                         help="rollout temperature (default: the model card's, per thinking mode)")
    p_train.add_argument("--max-think-tokens", type=int, default=0,
                         help="cap on <think> per rollout, forced closed past it; 0 = thinking off")
    p_train.add_argument("--eval-mmlu", type=int, default=0,
                         help="score N MMLU questions before and after (needs `datasets`)")
    p_train.add_argument("--load-adapter",
                         help="adapter.safetensors to copy into the LoRA tensors before "
                         "training/eval; refuses a file whose keys do not match. With "
                         "--steps 0 this re-scores a finished run's adapter")
    p_train.add_argument("--allow-short-rollouts", action="store_true",
                         help="bypass the before-eval and periodic rollout-length guards; "
                              "the smoke recipes want the "
                              "truncation, a real run almost never does")
    p_train.add_argument("--deterministic", action="store_true",
                         help="decode eager (no captured graph): bitwise reproducible "
                              "rollouts across processes, at a rollout throughput cost")
    p_train.add_argument("--eval-max-new-tokens", type=int, default=2048,
                         help="eval generation length; independent of --max-new-tokens, "
                         "which caps the ROLLOUTS. Scoring at the training cap measures "
                         "the cap, not the policy")
    p_train.add_argument("--eval-gsm8k", help="JSONL {prompt, answer}: greedy exact-match "
                         "accuracy before and after")
    p_train.add_argument("--eval-n", type=int, default=100, help="rows of --eval-gsm8k to score")
    # 0 = off, so no existing invocation changes. The subset is a fixed-seed shuffle
    # (_curve_rows): the file is ordered, so a prefix would run 5 pt low -- a bias the
    # size of the effect the curve measures. One seed for every run, so a curve point is
    # paired across steps and comparable across runs to the historical anchor.
    p_train.add_argument("--eval-every", type=int, default=0,
                         help="score the held-out curve subset every N steps (0 = off)")
    p_train.add_argument("--patience", type=int, default=0,
                         help="early-stop after N curve POINTS without a significant score "
                              "gain -- a point is --eval-every steps, not one -- or "
                              "immediately on a significant decline; 0 = never stop")
    p_train.add_argument("--eval-curve-n", type=int, default=20,
                         help="rows of --eval-gsm8k in the curve subset; keep the scoring "
                              "under 5%% of a step. The run refuses when the subset's "
                              "worst-case binomial SE does not resolve --curve-target-pt")
    p_train.add_argument("--curve-target-pt", type=float, default=5.0,
                         help="effect in points the --eval-every curve exists to locate; "
                              "the run refuses when 50/sqrt(--eval-curve-n) >= this")
    p_train.add_argument("--eval-curve-seed", type=int, default=0,
                         help="shuffle seed selecting the curve subset; fixed across runs "
                              "so points stay paired and comparable")
    p_train.add_argument("--judge", action="store_true",
                         help="let the policy rank rollouts the binary reward ties "
                              "(judge.py: tests decide first, order only)")
    p_train.add_argument("--force", action="store_true",
                         help="retrain even if this run's manifest is already finished")
    p_train.add_argument("--json", action="store_true",
                         help="print the run manifest as JSON instead of step lines")
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--optim", choices=["adafactor", "iso"], default="adafactor",
                         help="full-parameter SFT optimizer; --rl/--opd train LoRA and ignore it")
    p_train.add_argument("--save-model", action="store_true",
                         help="full-parameter SFT only: save the trained bf16 model under the "
                              "run dir and record it as artifacts.out, so it can be a "
                              "`tilerl merge` specialist (merge needs bf16 masters, not fp4)")
    p_train.add_argument("--served-fp4", action="store_true",
                         help="full-parameter SFT only: keep the served fp4 faces beside the "
                              "bf16 masters and re-pack them after every optimizer step, so a "
                              "shared serving engine decodes trained weights without a reload.")
    p_train.add_argument("--lora-rank", type=int, default=16)
    p_train.add_argument("--tp", type=int, default=1,
                         help="tensor-parallel width; dp is WORLD_SIZE//tp, cp is 1. "
                              "Launch under torchrun --nproc_per_node=WORLD_SIZE.")
    p_train.add_argument("--draft", help="draft head safetensors: speculative rollout (--opd)")
    p_train.add_argument("--depth", type=int, help="drafts per row per tick; default is the "
                         "head's own (chain: 2; block: its checkpoint's block minus the anchor)")
    p_train.add_argument("--recipe", choices=sorted(RECIPES),
                         help="a flag set that passed a gate (recipes.py); flags override it")
    p_train.add_argument("--dry-run", action="store_true",
                         help="print the training memory rows (adapter/optimizer/frame/tape) "
                              "and stop; no model, no engine, no card")
    p_train.add_argument("--batch", type=int, default=0,
                         help="--dry-run: micro-batch rows B for the tape row (default: "
                              "--micro, else --group)")
    p_train.add_argument("--train-seq-len", type=int, default=0, metavar="S",
                         help="--dry-run: token rows per batch for the layer-segment tape "
                              "(default: --max-new-tokens)")
    # The recipe is the subparser's defaults, so anything typed still wins.
    p_train.set_defaults(func=cmd_train, **(flags(recipe) if recipe else {}))

    p_bench = sub.add_parser("bench", help="benchmark prefill/decode throughput")
    p_bench.add_argument("--model", choices=MODEL_NAMES, default="tiny")
    p_bench.add_argument("--prompt-len", type=int, default=128)
    p_bench.add_argument("--gen", type=int, default=32)
    p_bench.add_argument(
        "--suite",
        default=None,
        help="run the full harness (scripts/bench_harness.py) instead of the quick "
        "tiny timer: comma list of decode-kv,prefill,kv-reuse,train,micro",
    )
    p_bench.add_argument("--source", default=None, help="27B checkpoint dir (harness GPU suites)")
    p_bench.add_argument("--gpu", type=int, default=None, help="GPU index (harness)")
    p_bench.add_argument("--batches", default=None, help="harness decode batch sizes, e.g. 1,8")
    p_bench.add_argument("--kernels", action="store_true",
                         help="print the per-kernel roofline table (bytes/flops), no GPU")
    p_bench.add_argument("--context", type=int, default=4096,
                         help="pooled context tokens the --kernels decode reads against")
    p_bench.add_argument("--prefill", type=int, default=0, metavar="S",
                         help="print the prefill roofline table for S tokens instead of decode")
    p_bench.add_argument("--checkpoint", default=None,
                         help="--kernels: price weights from this checkpoint dir's actual "
                              "nvfp4/fp8 device faces instead of the all-nvfp4 config face")
    p_bench.add_argument("--calibrate", action="store_true",
                         help="measure this card's HBM bandwidth, bf16 peak and PCIe H2D "
                              "bandwidth and append three rows to the bench ledger (cuda-only)")
    p_bench.add_argument("--card", type=int, default=None,
                         help="physical GPU card for --calibrate")
    p_bench.add_argument("--device-name", default=None,
                         help="device name the --kernels floor lookup keys on (default: "
                              "the cuda card; off cuda there is no floor)")
    p_bench.add_argument("--sparse-k", type=int, default=0, metavar="PAGES",
                         help="--kernels: also render the sparse-KV selection table with "
                              "this many indexed pages per row (derived HBM/PCIe bounds)")
    p_bench.add_argument("--scorer", choices=["index", "bounds"], default="index",
                         help="sparse scorer for --sparse-k (default index)")
    p_bench.add_argument("--kv-fp8", default="", choices=["", "e4m3", "e5m2"],
                         help="--sparse-k: price the cold/hot pages in fp8 KV (e4m3)")
    for v in ("table", "readme", "regress", "questions", "collectors"):
        p_bench.add_argument(f"--{v}", action="store_true",
                             help=f"bench view: {v} from the bench store, no GPU")
    p_bench.add_argument("name", nargs="?",
                         help="metric name: run its registered collector "
                              "(docs/bench-metrics.json), remaining args pass through")
    p_bench.add_argument("collector_args", nargs=argparse.REMAINDER)
    p_bench.set_defaults(func=cmd_bench)

    p_gen = sub.add_parser(
        "generate", help="offline batch generation, one process per device"
    )
    p_gen.add_argument("prompts", help="JSONL, one object per line with token_ids")
    p_gen.add_argument("--out", required=True, help="JSONL to write")
    p_gen.add_argument("--devices", default="0", help="CUDA indices, e.g. 0-7 or 0,1,2")
    p_gen.add_argument("--source", default=None, help="27B checkpoint dir (omit for tiny)")
    p_gen.add_argument("--max-new-tokens", type=int, default=128)
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_gen.add_argument("--top-p", type=float, default=1.0)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--max-batch", type=int, default=32,
                       help="concurrent requests per device")
    p_gen.set_defaults(func=cmd_generate)

    p_merge = sub.add_parser("merge", help="merge specialist checkpoints that share a base")
    p_merge.add_argument("--base", required=True, help="base checkpoint dir")
    p_merge.add_argument(
        "--specialists",
        required=True,
        type=lambda v: v.split(","),
        help="comma-separated specialist checkpoint dirs",
    )
    p_merge.add_argument("--out", required=True, help="merged checkpoint dir")
    p_merge.add_argument("--method", choices=["iso", "average", "ties", "dare"],
                         default="iso")
    p_merge.add_argument("--force", action="store_true",
                         help="re-merge even if a finished run with these inputs exists")
    p_merge.add_argument("--json", action="store_true", help="print the manifest as JSON")
    p_merge.set_defaults(func=cmd_merge)

    p_ledger = sub.add_parser("ledger", help="list runs ($TILERL_RUNS, default ./runs)")
    p_ledger.add_argument("--lineage", metavar="ID", help="this run and what it descends from")
    p_ledger.add_argument("--time-to-score", type=float, metavar="SCORE",
                          help="read time_to_score off each run's eval_curve: the step that "
                               "first scored >= SCORE, the interval it was crossed in, and "
                               "the cumulative training seconds there")
    p_ledger.add_argument("--devices", action="store_true",
                          help="per-device measured calibration (HBM GB/s, bf16 TFLOPS) and "
                               "resident peak from the bench store, instead of listing runs")
    p_ledger.add_argument("--json", action="store_true")
    p_ledger.set_defaults(func=cmd_ledger)

    return parser

def main() -> None:
    recipe = getattr(_build_parser().parse_known_args()[0], "recipe", None)
    if recipe:
        # stderr: --json is not known until the parser is built, and stdout is the JSON stream.
        print(f"recipe {recipe}: {RECIPES[recipe]['status']}", file=sys.stderr)
    args = _build_parser(recipe).parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
