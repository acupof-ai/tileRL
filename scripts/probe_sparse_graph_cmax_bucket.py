"""H2 probe v4 — does a lazy sparse-graph capture at a cmax BUCKET boundary contaminate?

v4 (2026-09-17) fixes two v3 (md5 202dcf3) defects that produced NO data on
V100 and burned the window on test bugs, not an H2 result:

1. v3 priming raised bare StopIteration: it submitted a row with
   max_new_tokens=4, and after driving prefill a single decode tick finished
   (or otherwise removed) the short row, so `next(r for r in _running ...)`
   failed before a graph/eager first-token was captured; W=2 failed earlier
   (cmax=None). v4 submits 32 tokens, waits for the DECODE phase by polling
   the phase (never an extra fixed step), and every row lookup goes through
   `_row`, which dumps phase/_finished/_failed/poll/generated count instead
   of a bare StopIteration.

2. v3 built three 27B engines in ONE process (graph -> eager -> B=4).
   gc.collect does not return the caching allocator pool, so the third
   build_engine OOMed in LinearStatePool with ~304 MiB free. v4 runs every
   engine arm in its OWN subprocess (fresh CUDA context; exit returns all
   device memory). The parent builds no engine; it only spawns and aggregates
   one JSON line per child.

Safety (in every child): ONE engine alive at a time; on a failed sm70 capture
the allocator is poisoned and empty_cache asserts, so NO-CAPTURE hard-exits
with os._exit(12) without another CUDA call. Exit 12 propagates as
H2_CAPTURE_FAILED and the poisoned context dies with its child.

Run (parent):

  H2_COLD_BYTES=1073741824 H2_COLD_SSD=$HOME/sparse_cold_128k.bin \\
  H2_COLD_SSD_BYTES=8589934592 H2_COLD_FORMAT=f16 \\
  TILERL_TARGET=cuda TILERL_QWEN38_SOURCE=<ckpt> PYTHONPATH=src:... \\
    python scripts/probe_sparse_graph_cmax_bucket.py <ckpt> --draft <mtp>

Modes:
  (default)        all bucket/W comparisons + B=4, one subprocess per engine
  --only-b4        just the B=4 child
  --skip-b4        only the bucket/W comparisons
  --buckets …      sparse cmax buckets (n<=8192 dense buckets auto-skip)
  --worker arm <source> <draft> <bucket> <depth> <graph|eager>
  --worker b4  <source> <draft>      (internal; spawned by the parent)

Parent exit codes:
  0 H2_REFUTED   10 H2_BAD   11 H2_ILLEGAL
 12 H2_CAPTURE_FAILED   13 H2_PROBE_ERROR (a harness bug, not an H2 result)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys

CMAX_BUCKETS = [512, 1024, 2048]  # sparse-only; 64/128/256 land dense (n<=8192)
SPARSE_MIN = 8192
MAX_NEW = 32  # v4: row survives many decode ticks so the first one is caught
_PHASE_DECODE = 2
_PHASE_DONE = 3


class ProbeError(RuntimeError):
    """A harness/state-machine bug, distinct from any H2 verdict."""


def tokens_for_bucket(bucket: int) -> int:
    """First decode candidate count at the bucket top.

    own_first = q_lo//16 - (WINDOW_PAGES-1); own_first == bucket needs
    q_lo//16 == bucket + 7; the row's last token is n = q_lo + 1.
    """
    return (bucket + 8 - 1) * 16 + 15 + 1


def _row(engine, rid: int, what: str):
    """Fetch a live row or raise ProbeError with a full dump (no bare
    StopIteration — the v3 defect)."""
    live = [r for r in engine._running if r.req_id == rid]
    if live:
        return live[0]
    finished = dict(getattr(engine, "_finished", {}))
    failed = dict(getattr(engine, "_failed", {}))
    polled = engine.poll().get(rid, "<absent>")
    raise ProbeError(
        f"{what}: row {rid} not in _running; poll={polled}; "
        f"in_finished={rid in finished}; in_failed={rid in failed}: "
        f"failed_note={failed.get(rid)}"
    )


def prime_to_decode(engine, n_tokens: int, tag: str):
    """Submit one row, advance ONLY to its first DECODE tick while it is live."""
    from tilerl.engine import SamplingParams

    ids = [(t % 31000) + 7 for t in range(n_tokens)]
    rid = engine.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW, seed=0))
    engine.step()  # admit
    for guard in range(40000):
        row = next((r for r in engine._running if r.req_id == rid), None)
        if row is None:
            _row(engine, rid, f"{tag} after prefill")  # raises with dump
        if row.phase == _PHASE_DECODE:
            return rid
        if row.phase == _PHASE_DONE:
            raise ProbeError(f"{tag}: DONE before decode (output={len(row.output)})")
        engine.step()
    raise ProbeError(f"{tag}: prefill guard exhausted")


def one_decode_tick(engine, rid: int, tag: str):
    """Run decode steps until the row appends at least one NEW output token
    (sparse rows do not surface tokens via poll() per tick and their live slot
    win_parity does not flip — both v3 signals are inert here). Compare the
    first decode-produced token id, which is deterministic at temperature 0 and
    is exactly what a contaminated capture would change. Returns
    (new_token, n_steps, out_len_before)."""
    import torch

    row = _row(engine, rid, f"{tag} pre-tick")
    before = len(row.output)
    for steps in range(1, 65):
        engine.step()
        if engine._backend.device.type == "cuda":
            torch.cuda.synchronize()
        row = _row(engine, rid, f"{tag} step{steps}")
        if len(row.output) > before:
            return row.output[before], steps, before
    raise ProbeError(
        f"{tag}: no new output token after 64 decode steps (out_len stuck at {before})"
    )


def build_engine_arm(source, draft_path, bucket, depth, decode_graph, model_name="qwen38-27b"):
    """Build ONE engine (graph XOR eager), prime, one decode tick, shutdown.
    Returns the JSON-serializable arm result. model_name="tiny" is the CPU
    dry-run seam (random weights, no draft file, device need not be cuda)."""
    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl import build
    from tilerl.build import build_engine, build_model
    from tilerl.sparse_engine import cmax_bucket
    from tilerl.sparse_index import WINDOW_PAGES
    from tilerl.spec import load_draft

    dry = model_name == "tiny"
    if not dry:
        build.QWEN38_SOURCE = source
    be = get_backend()
    if not dry and be.device.type != "cuda":
        raise ProbeError("this arm needs the CUDA cell (or --model tiny to dry-run)")
    cfg, model = build_model(model_name, seed=0, fuse_projections=not dry)
    draft = None if dry else load_draft(model, draft_path)
    n = tokens_for_bucket(bucket)
    if n <= SPARSE_MIN:
        return {"arm": "dense-skip", "bucket": bucket, "W": depth + 1}

    if not dry:
        torch.cuda.reset_peak_memory_stats()
    res = {
        "bucket": bucket,
        "W": depth + 1,
        "graph": decode_graph,
        "n_tokens": n,
        "token": None,
        "decode_steps_to_token": None,
        "out_len_at_token": None,
        "observed_cmax": None,
        "captured": None,
        "produced_token": False,
        "peak_mib": None,
    }
    e = build_engine(
        cfg,
        model,
        be,
        num_slots=4,
        max_batch=4,
        max_total_tokens=131072,
        max_num_batched_tokens=512,
        sparse_k=128,
        sparse_min_tokens=SPARSE_MIN,
        scorer="bounds",
        kv_cold_bytes=0 if dry else int(os.environ.get("H2_COLD_BYTES", str(1 << 30))),
        cold_ssd_path="" if dry else os.environ.get("H2_COLD_SSD", ""),
        cold_ssd_bytes=0 if dry else int(os.environ.get("H2_COLD_SSD_BYTES", "0")),
        cold_format="f16",
        decode_graph=decode_graph,
        draft=draft if depth else None,
        spec_depth=depth if depth else None,
    )
    rid = prime_to_decode(e, n, tag=f"b{bucket}W{depth + 1}")
    rows = e._sparse.decode_rows([_row(e, rid, "decode_rows")], [1 + depth])
    res["observed_cmax"] = max((len(r["cand"]) for r in rows), default=0)
    own_w = WINDOW_PAGES + (1 if depth >= 1 else 0)
    key = (1, 1 + depth, cmax_bucket(res["observed_cmax"]), own_w)
    token, steps, out_len = one_decode_tick(e, rid, f"b{bucket}W{depth + 1}")
    captured = key in e._sparse_graphs if decode_graph else None
    res.update(
        token=[token],
        decode_steps_to_token=steps,
        out_len_at_token=out_len + 1,
        captured=bool(captured) if decode_graph else None,
        produced_token=True,
        peak_mib=(torch.cuda.max_memory_reserved() // (1 << 20)) if not dry else None,
    )
    # The NO-CAPTURE hard exit is a real-card sm70 safety rule; a CPU tiny
    # dry-run never captures, so do not treat that as a poisoned context.
    if decode_graph and not captured and not dry:
        print(
            f"[FATAL b{bucket} W{depth + 1}] expected key {key} absent; "
            f"keys={list(e._sparse_graphs)}",
            file=sys.stderr,
            flush=True,
        )
        os._exit(12)
    e.shutdown()
    del e, model, be
    if draft is not None:
        del draft
    gc.collect()  # child is about to exit; never empty_cache on sm70
    return res


def _sparse_keys_after(engine):  # pragma: no cover - only used pre-exit
    try:
        return list(engine._sparse_graphs)
    except Exception:
        return "?"


def build_b4(source, draft_path):
    """B=4 concurrent sparse+d1 graph rows, a single child. H2_MODEL=tiny gives
    a depth=0 four-row sparse graph structural dry-run (no 27B draft, cannot
    exercise the d1 illegal-access risk but proves admit/lockstep/slot-free)."""
    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl import build
    from tilerl.build import build_engine, build_model
    from tilerl.engine import SamplingParams
    from tilerl.spec import load_draft

    dry = os.environ.get("H2_MODEL", "qwen38-27b") == "tiny"
    if not dry:
        build.QWEN38_SOURCE = source
    be = get_backend()
    cfg, model = build_model("tiny" if dry else "qwen38-27b", seed=0, fuse_projections=not dry)
    draft = None if dry else load_draft(model, draft_path)
    n = tokens_for_bucket(512)
    res = {"arm": "b4", "dry": dry, "ok": False, "finished": 0, "leaked_slots": None, "note": ""}
    e = build_engine(
        cfg,
        model,
        be,
        num_slots=4,
        max_batch=4,
        max_total_tokens=131072,
        max_num_batched_tokens=512,
        sparse_k=128,
        sparse_min_tokens=SPARSE_MIN,
        scorer="bounds",
        kv_cold_bytes=0 if dry else int(os.environ.get("H2_COLD_BYTES", str(1 << 30))),
        cold_ssd_path="" if dry else os.environ.get("H2_COLD_SSD", ""),
        cold_ssd_bytes=0 if dry else int(os.environ.get("H2_COLD_SSD_BYTES", "0")),
        cold_format="f16",
        decode_graph=True,
        draft=draft,
        spec_depth=0 if dry else 1,
    )
    try:
        rids = []
        for k in range(4):
            ids = [((t + 7919 * (k + 1)) % 31000) + 7 for t in range(n)]
            rids.append(
                e.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW, seed=100 + k))
            )
        e.step()
        for _ in range(40000):
            if not any(r.req_id in rids for r in e._running):
                break
            e.step()
        if not dry:
            torch.cuda.synchronize()
        res["finished"] = sum(len(e.poll().get(r, ())) for r in rids)
        res["leaked_slots"] = e.stats()["slots_used"]
        res["ok"] = res["leaked_slots"] == 0
        res["note"] = "clean" if res["ok"] else f"slots_used={res['leaked_slots']}"
    except Exception as exc:
        res["note"] = f"EXC {type(exc).__name__}: {exc}"
    finally:
        e.shutdown()
        del e, model, draft, be
        gc.collect()
    return res


def spawn_worker(*worker_argv) -> dict:
    """Fresh subprocess running this script --worker ...; parse its JSON line."""
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker", *map(str, worker_argv)]
    proc = subprocess.run(
        cmd, env=dict(os.environ), capture_output=True, text=True, cwd=_repo_root()
    )
    out = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode == 12:
        return {"arm": "capture-failed", "child_rc": 12, "stderr": proc.stderr[-2000:]}
    if proc.returncode != 0:
        return {
            "arm": "worker-error",
            "child_rc": proc.returncode,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-3000:],
        }
    if not out:
        return {
            "arm": "worker-error",
            "child_rc": 0,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-3000:],
        }
    return json.loads(out[-1])


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compare_bucket(source, draft, bucket, depth):
    g = spawn_worker("arm", source, draft, bucket, depth, "graph")
    e = spawn_worker("arm", source, draft, bucket, depth, "eager")
    if g.get("arm") in ("capture-failed", "worker-error") or e.get("arm") == "worker-error":
        return {
            "bucket": bucket,
            "W": depth + 1,
            "verdict": "PROBE/CAPTURE",
            "graph": g,
            "eager": e,
            "token_ok": False,
            "produced_ok": False,
        }
    token_ok = g.get("token") is not None and g.get("token") == e.get("token")
    produced_ok = bool(g.get("produced_token")) and bool(e.get("produced_token"))
    return {
        "bucket": bucket,
        "W": depth + 1,
        "observed_cmax": g.get("observed_cmax"),
        "captured": g.get("captured"),
        "graph_token": g.get("token"),
        "eager_token": e.get("token"),
        "graph_steps": g.get("decode_steps_to_token"),
        "eager_steps": e.get("decode_steps_to_token"),
        "peak_graph_mib": g.get("peak_mib"),
        "token_ok": token_ok,
        "produced_ok": produced_ok,
        "verdict": "match" if (token_ok and produced_ok) else "CONTAMINATED",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?")
    ap.add_argument("--draft", default="")
    ap.add_argument("--buckets", type=int, nargs="*", default=CMAX_BUCKETS)
    ap.add_argument("--only-b4", action="store_true")
    ap.add_argument("--skip-b4", action="store_true")
    ap.add_argument("--worker", nargs="*", default=None)
    args = ap.parse_args()

    # --- in-process worker (one fresh CUDA context per process) ---
    if args.worker is not None:
        try:
            kind = args.worker[0]
            if kind == "arm":
                _, source, draft, bucket, depth, mode = args.worker
                model_name = os.environ.get("H2_MODEL", "qwen38-27b")
                print(
                    json.dumps(
                        build_engine_arm(
                            source,
                            draft,
                            int(bucket),
                            int(depth),
                            mode == "graph",
                            model_name=model_name,
                        )
                    ),
                    flush=True,
                )
                return 0
            if kind == "b4":
                _, source, draft = args.worker
                print(json.dumps(build_b4(source, draft)), flush=True)
                return 0
            raise ProbeError(f"unknown worker kind {kind}")
        except ProbeError as exc:
            print(json.dumps({"arm": "probe-error", "note": str(exc)}), flush=True)
            return 13

    # --- B=4 only ---
    if args.only_b4:
        r = spawn_worker("b4", args.source, args.draft)
        print(json.dumps(r, indent=2))
        if r.get("arm") == "capture-failed":
            return 12
        if "illegal" in str(r.get("note", "")).lower():
            return 11
        return 0 if r.get("ok") else 11

    if not args.source or not args.draft:
        print("source and --draft required", file=sys.stderr)
        return 13

    results, bad, capture_fail = [], False, False
    for depth in (0, 1):
        for b in args.buckets:
            if tokens_for_bucket(b) <= SPARSE_MIN:
                print(f"[bucket {b:5d} W={depth + 1}] DENSE-SKIP", flush=True)
                continue
            r = compare_bucket(args.source, args.draft, b, depth)
            results.append(r)
            print(
                f"[bucket {b:5d} W={depth + 1}] {r['verdict']} "
                f"cmax={r.get('observed_cmax')} captured={r.get('captured')} "
                f"g_tok={r.get('graph_token')} e_tok={r.get('eager_token')} "
                f"g_steps={r.get('graph_steps')} e_steps={r.get('eager_steps')} "
                f"produced_ok={r.get('produced_ok')}",
                flush=True,
            )
            if r["verdict"] == "PROBE/CAPTURE":
                capture_fail = True
            elif not (r["token_ok"] and r["produced_ok"]):
                bad.append(r)

    illegal = False
    if not args.skip_b4 and not capture_fail:
        b4 = spawn_worker("b4", args.source, args.draft)
        print(
            f"[b4 d1] ok={b4.get('ok')} finished={b4.get('finished')} "
            f"leaked={b4.get('leaked_slots')} :: {b4.get('note')}",
            flush=True,
        )
        if b4.get("arm") == "capture-failed":
            capture_fail = True
        elif "illegal" in str(b4.get("note", "")).lower():
            illegal = True
        elif not b4.get("ok"):
            bad.append({"bucket": "b4", **b4})

    print("=" * 60)
    if capture_fail:
        print("H2_CAPTURE_FAILED")
        print(json.dumps(results, indent=2))
        return 12
    if illegal:
        print("H2_ILLEGAL")
        return 11
    if bad:
        print(f"H2_BAD: {[(x.get('bucket'), x.get('W')) for x in bad]}")
        print(json.dumps(results, indent=2))
        return 10
    print(
        f"H2_REFUTED: {len(results)} bucket/W comparisons match; "
        f"{'B=4 clean' if not args.skip_b4 else 'B=4 skipped'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
