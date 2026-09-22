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
  --parity         #804: per-tick token-ID + inputs parity, 6 cells (see below)
  --only-b4        just the B=4 child
  --skip-b4        only the bucket/W comparisons
  --buckets …      sparse cmax buckets (default 512/1024/2048)
  --worker arm <source> <draft> <bucket> <depth> <graph|eager>
  --worker parity <source> <draft> <graph|eager> <depth>   (internal)
  --worker b4  <source> <draft>      (internal; spawned by the parent)

--parity (#804) four jobs in ONE window (one 27B model load per arm process,
shared by its three buckets):
  1. first-token question: the LAST prefill-position logits are sha1-hashed
     elementwise and must match between arms before any decode is compared;
  2. per committed token: graph vs eager token ids, stop at the first divergent
     position (A arbitration);
  3. all 6 cells (cmax 512/1024/2048 x W=1/2) must show full-sequence parity
     (the v5 acceptance gate);
  4. at the divergence tick the inputs are compared and the verdict follows
     fixmisc's table:
        inputs equal + token differ  -> TOKEN_DIVERGE_INPUT_EQUAL_H1_AT_SKETCH_PRECISION
        (sketch-level equality; a fired H1 is arbitrated byte-exactly)
        inputs differ + token differ -> TOKEN_DIVERGE_INPUT_DIFF_H3:<field>
     The fingerprint sketch is keyed by LOGICAL page (physical block ids and
     state slots differ across the arm processes); the recurrent state/conv is
     read per layer at FORWARD ENTRY, own-window K/V at the previous boundary,
     selected earlier pages from an immutable-page table. A fired divergence is
     arbitrated byte-exactly by rerunning with H2_DUMP_CELL=b:W and
     H2_DUMP_STEPS=<out_len>, which dumps states/conv/parity/per-page K/V of
     both arms to parity_dump_*.pt.
  Parent exit codes add: PARITY 0 match / 10 bad / 12 capture / 13 probe bug.

Both arms are built with sparse_min_tokens=0 and sparse_device_select=True so
the graph gate can pass and graph/eager differ ONLY in decode_graph. The
worker measures the first-decode cmax and corrects n by 16 tokens per unit
cmax (resubmitting) until the target bucket is observed.

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

CMAX_BUCKETS = [512, 1024, 2048]  # all sparse; n = 8311 / 16503 / 32887
MAX_NEW = 32  # v4: row survives many decode ticks so the first one is caught
_PHASE_PREFILL = 1
_PHASE_DECODE = 2
_PHASE_DONE = 3
# 27B load + up to 32.9k-token chunked prefill + 32 decodes per arm; the bucket
# 2048 arm is the slow one. TimeoutExpired is a harness fault (rc 13), never
# collapsed into capture-failed (12) or an H2 verdict.
WORKER_TIMEOUT_S = 3000


class ProbeError(RuntimeError):
    """A harness/state-machine bug, distinct from any H2 verdict."""


def tokens_for_bucket(bucket: int) -> int:
    """Prompt length whose FIRST decode tick sees candidate count == bucket.

    First-decode cmax = n//16 - (WINDOW_PAGES-1) = n//16 - 7 (verified on CPU
    tiny over n=8192..8400: cmax 512 holds for n in [8304,8319], 513 from
    8320; linear 1 cmax per 16 tokens to n=16503->1024, 32887->2048). Take the
    midpoint of the 16-wide window so a one-token scheduling slip cannot push
    the tick into the next bucket; the worker asserts the observed bucket.
    """
    return (bucket + 7) * 16 + 7


def _row(engine, rid: int, what: str):
    """Fetch a live row or raise ProbeError with a full dump (no bare
    StopIteration — the v3 defect). poll() can itself raise RequestFailed on a
    cold-spill error; that must not erase the _finished/_failed dump."""
    live = [r for r in engine._running if r.req_id == rid]
    if live:
        return live[0]
    finished = dict(getattr(engine, "_finished", {}))
    failed = dict(getattr(engine, "_failed", {}))
    try:
        polled = engine.poll().get(rid, "<absent>")
    except Exception as exc:  # cold-spill RequestFailed, etc.
        polled = f"<poll raised {type(exc).__name__}: {exc}>"
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


def _cancel_and_drain(engine, rid: int, tag: str) -> None:
    if not engine.cancel(rid):
        raise ProbeError(f"{tag}: cancel({rid}) returned False")
    for _ in range(1000):
        if not any(r.req_id == rid for r in engine._running):
            return
        engine.step()
    raise ProbeError(f"{tag}: cancelled row {rid} still live after 1000 steps")


def prime_at_bucket(engine, n: int, bucket: int, depth: int, tag: str):
    """Prime a row whose first-decode cmax lands in `bucket`, measuring the real
    cmax and correcting n by 16 tokens per cmax before resubmitting. The closed
    form n=(bucket+7)*16+7 lands mid-window on CPU tiny; this rescues the run if
    a real-card scheduling detail shifts it instead of recording a next-bucket
    verdict. No decode tick runs here, so the graph arm captures nothing during
    the probes. Returns (rid, cmax, n_used)."""
    from tilerl.sparse_engine import cmax_bucket

    for attempt in range(4):
        rid = prime_to_decode(engine, n, f"{tag} attempt{attempt}")
        rows = engine._sparse.decode_rows([_row(engine, rid, "decode_rows")], [1 + depth])
        cmax = max((len(r["cand"]) for r in rows), default=0)
        if cmax_bucket(cmax) == bucket:
            return rid, cmax, n
        n += (bucket - cmax) * 16  # 16 tokens per candidate page/unit cmax
        if n <= 0:
            raise ProbeError(f"{tag}: cmax correction overshot to n={n}")
        _cancel_and_drain(engine, rid, f"{tag} attempt{attempt}")
    raise ProbeError(f"{tag}: cmax never landed in bucket {bucket}")


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
        "observed_bucket": None,
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
        # 0, not the serve's 8192: the sparse graph gate is `not
        # _sparse_min_tokens`; 8192 would force every sparse tick eager and the
        # graph arm would hard-exit after a full 33k prefill.
        sparse_min_tokens=0,
        # ON in BOTH arms: graph vs eager must differ ONLY in decode_graph, or
        # a token mismatch is a different candidate set, not capture
        # contamination.
        sparse_device_select=True,
        scorer="bounds",
        kv_cold_bytes=0 if dry else int(os.environ.get("H2_COLD_BYTES", str(1 << 30))),
        cold_ssd_path="" if dry else os.environ.get("H2_COLD_SSD", ""),
        cold_ssd_bytes=0 if dry else int(os.environ.get("H2_COLD_SSD_BYTES", "0")),
        cold_format="f16",
        decode_graph=decode_graph,
        draft=draft if depth else None,
        spec_depth=depth if depth else None,
    )
    rid, cmax, n = prime_at_bucket(e, n, bucket, depth, f"b{bucket}W{depth + 1}")
    res["n_tokens"] = n
    res["observed_cmax"] = cmax
    res["observed_bucket"] = cmax_bucket(cmax)
    own_w = WINDOW_PAGES + (1 if depth >= 1 else 0)
    key = (1, 1 + depth, bucket, own_w)
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
            f"keys={_sparse_keys_after(e)}",
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
    res = {
        "arm": "b4",
        "dry": dry,
        "ok": False,
        "finished": 0,
        "leaked_slots": None,
        "captured_keys": None,
        "note": "",
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
        sparse_min_tokens=0,
        sparse_device_select=True,
        scorer="bounds",
        kv_cold_bytes=0 if dry else int(os.environ.get("H2_COLD_BYTES", str(1 << 30))),
        cold_ssd_path="" if dry else os.environ.get("H2_COLD_SSD", ""),
        cold_ssd_bytes=0 if dry else int(os.environ.get("H2_COLD_SSD_BYTES", "0")),
        cold_format="f16",
        decode_graph=True,
        draft=draft,
        spec_depth=0 if dry else 1,
    )
    expected_tokens = 4 * MAX_NEW
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
        polled = e.poll()  # destructive: bind once, do not re-call per rid
        res["finished"] = sum(len(polled.get(r, ())) for r in rids)
        res["leaked_slots"] = e.stats()["slots_used"]
        try:
            res["captured_keys"] = [list(map(str, k)) for k in e._sparse_graphs]
        except Exception:
            res["captured_keys"] = "?"
        # slots==0 alone passed v3 even when rows died early. A real d1 B=4 arm
        # must have captured the (4, 2, 512, 9) graph AND generated every token;
        # CPU tiny cannot capture CUDA graphs, so only the token count gates it.
        if dry:
            res["ok"] = res["finished"] == expected_tokens and res["leaked_slots"] == 0
        else:
            captured_b4 = any(k[0] == 4 for k in e._sparse_graphs)
            res["ok"] = (
                captured_b4 and res["finished"] == expected_tokens and res["leaked_slots"] == 0
            )
            if not captured_b4:
                print(
                    f"[FATAL b4] no B=4 sparse graph captured; keys={res['captured_keys']}",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(12)
        if not res["ok"]:
            res["note"] = (
                f"finished={res['finished']} expected={expected_tokens} "
                f"slots_used={res['leaked_slots']}"
            )
        else:
            res["note"] = "clean"
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
    try:
        proc = subprocess.run(
            cmd,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            timeout=WORKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        tail = exc.stderr or b""
        tail = tail.decode()[-2000:] if isinstance(tail, bytes) else str(tail)[-2000:]
        return {"arm": "worker-timeout", "timeout_s": WORKER_TIMEOUT_S, "stderr": tail}
    out = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode == 12:
        return {"arm": "capture-failed", "child_rc": 12, "stderr": proc.stderr[-2000:]}
    if proc.returncode == 11:
        return {
            "arm": "illegal-access",
            "child_rc": 11,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-3000:],
        }
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


PARITY_BUCKETS = [512, 1024, 2048]  # 3 cmax buckets x W=1,2 = the 6 v5 cells
PARITY_GEN = 48  # counting sequence length per cell; crosses a cmax doubling mid-run


def _sig(x) -> float:
    """6-significant-fingerprint of a device reduction. Bit-equal inputs run the
    same torch reduction over the same layout and come out identical; the 1e-6
    relative rounding only absorbs reorder noise. A fired mismatch is confirmed
    at byte precision on the H2_DUMP_* rerun, never trusted from the sketch."""
    return float(f"{float(x):.6e}")


def _state_fp(engine, slot) -> dict:
    """Per-layer (sum, sumsq) of the recurrent state and conv window, plus the
    window parity. The cross-process state SLOT number is not comparable (each
    arm allocates slots independently), only the slot the live row names is."""
    sp = engine._states
    s = sp.states[slot].float().reshape(sp.states.shape[1], -1)
    out = {
        "st": [
            v
            for pair in zip(s.sum(1).tolist(), (s * s).sum(1).tolist())
            for v in (_sig(pair[0]), _sig(pair[1]))
        ]
    }
    if sp.conv_windows is not None:
        c = sp.conv_windows[slot].float().reshape(sp.conv_windows.shape[1], -1)
        out["cv"] = [
            v
            for pair in zip(c.sum(1).tolist(), (c * c).sum(1).tolist())
            for v in (_sig(pair[0]), _sig(pair[1]))
        ]
    out["parity"] = int(sp.win_parity[slot].item())
    return out


def _pages_fp(engine, pages: list[int], rid: int) -> dict:
    """K/V fingerprints keyed by LOGICAL page (physical block ids differ across
    the two arm processes). Per page four aggregates over ALL planes
    (k-sum, k-sumsq, v-sum, v-sumsq): enough to place a mismatch on a page; the
    H2_DUMP_* rerun supplies per-layer raw bytes. A sum+sumsq pair cannot
    self-cancel on a real perturbation."""
    if not pages:
        return {}
    tr = engine._sparse.tracker
    pool = engine._kv
    phys = [tr.resident[rid][p] for p in pages]  # all selected+own resident pre-finalize
    out = {}
    for name, src in (("k", pool.k_pool), ("v", pool.v_pool)):
        t = src[:, phys].float()  # [L, P, H, T, D]
        f = t.reshape(t.shape[0], t.shape[1], -1)
        sm = f.sum((0, 2))
        sq = (f * f).sum((0, 2))
        for j, p in enumerate(pages):
            out.setdefault(str(p), {})[name] = [_sig(sm[j]), _sig(sq[j])]
    return out


def _install_parity_hooks(engine, job):
    """Two hooks per forward, both instance attributes (tree untouched):

    * model.forward ENTRY: the true inputs of this tick -- ids/positions, the
      recurrent state, conv window and parity for the row's slot, BEFORE the
      forward runs. This is exactly what "inputs equal" in the H1/H3 table
      means; taking it at finalize would miss a W=2 verify rewrite (verify
      runs after finalize). The last prefill chunk's logits are hashed on
      return for the prefill-parity precondition.
    * SparseRuntime.finalize: the one boundary both sparse forwards share
      (eager after model.forward; graph replay right after g.run), where the
      tick's selected logical pages are known. Earlier complete pages are
      immutable, so per-page K/V fingerprints accumulate across ticks; a
      selected page is byte-comparable to every earlier sighting.

    Physical block ids and state slot numbers differ across the two arm
    processes, so every K/V fingerprint is keyed by LOGICAL page and the
    state is read through the row's own slot."""
    rt = engine._sparse
    orig_finalize = rt.finalize
    orig_graph = rt.run_decode_graph
    orig_fwd = engine._model.forward
    orig_step = engine.step
    orig_commit = engine._commit

    def commit(req, toks, lps=None):
        # Both commit paths funnel through _commit: plain decode via
        # _sample_commit AND the W=2 spec path via _verify -> _commit (which
        # _sample_commit would miss). Record one entry per actual commit with
        # its pre-commit lengths, so the parent can attribute EVERY token to
        # its own tick even when a verify accepts 2 in one step (B1). Judge
        # phase PRE-call: a stop-hit token is appended then _finish flips the
        # row to DONE inside the call, and it must still be recorded.
        out_before, seq_before = len(req.output), int(req.seq_len)
        was_decode = int(req.phase) == _PHASE_DECODE
        rc = orig_commit(req, toks, lps)
        if job["rid"] is not None and req.req_id == job["rid"] and was_decode:
            job["commits"].append(
                {
                    "out_before": out_before,
                    "seq_before": seq_before,
                    "toks": list(toks)[: len(req.output) - out_before],
                }
            )
        return rc

    def fwd(input_ids, positions, kv, backend, **kw):
        # Input state MUST be read before the forward (the forward writes the
        # new recurrent state); graph arm reads it pre-replay in run_graph.
        # sf.rows are srow DICTS in the build_rows shape on BOTH arms: keys
        # req_id/own/cand/decoding/... and NO req object (decode_rows rows carry
        # req=r but never reach sf.rows). Read only dict keys and the BatchKv
        # tensors (state_slot/seq_q_lens) -- never a .req attribute.
        sf0 = getattr(kv, "sparse", None)
        pre = {}
        if sf0 is not None and job["rid"] is not None:
            for bi0, rw0 in enumerate(getattr(sf0, "rows", []) or []):
                if rw0["req_id"] == job["rid"] and rw0.get("decoding"):
                    pre[bi0] = {
                        "ids": [int(x) for x in input_ids[bi0]],
                        "pos": [int(x) for x in positions[bi0]],
                        "state": _state_fp(engine, int(kv.state_slot[bi0])),
                    }
        out = orig_fwd(input_ids, positions, kv, backend, **kw)
        if sf0 is not None and job["rid"] is not None:
            for bi, rw in enumerate(getattr(sf0, "rows", []) or []):
                if rw["req_id"] != job["rid"] or rw.get("decoding"):
                    continue
                # Requirement 1: the two arms' LAST prefill-position logits
                # must be bit-identical. Cryptographic hash of the exact
                # float32 bytes (one vocab-sized vector ~151k values), not a
                # sketch -- a sketch could not certify per-element equality.
                import hashlib

                last = int(kv.seq_q_lens[bi]) - 1
                vec = out[bi, last].detach().float().contiguous()
                b = vec.cpu().numpy().tobytes()
                job["prefill_logits"].append(
                    {
                        "sha1": hashlib.sha1(b).hexdigest(),
                        "argmax": int(vec.argmax()),
                        "n": int(vec.numel()),
                    }
                )
                if bi in pre:  # decode entry on a mixed tick (defensive)
                    job["cur"] = pre[bi]
        return out

    def finalize(sf, rows, hidden=None):
        # Run finalize FIRST, then observe the selection: selected_pages is the
        # one point both arms share (eager host _chosen and captured device
        # _dchosen both surface through it) and it is the exact pin set
        # finalize just computed. Reading sf.selected per group at the forward
        # boundary forced a replay-time D2H only on the graph arm (fill()
        # clears its cache every tick) -- two mechanisms, a fabricated H3.
        # r.output/seq_len are still pre-commit here: commit runs AFTER
        # finalize on both paths.
        dropped = orig_finalize(sf, rows, hidden)
        if job["rid"] is None:
            return dropped
        for bi, r in enumerate(rows):  # rows are _Req on both paths
            if r.req_id != job["rid"]:
                continue
            srow = sf.rows[bi]  # build_rows-shaped dict
            own = list(srow["own"])
            kept = sorted(sf.selected_pages(bi))  # post-finalize pin set
            earlier = [p for p in kept if p not in own]
            # Earlier complete pages are immutable once written (the forward
            # writes only the own window's trailing page, and candidates
            # exclude the own span). Fingerprint first sightings; on refresh
            # ticks (device_select=False, one per SPARSE_REFRESH_TICKS in BOTH
            # arms) re-fingerprint the whole selection so immutability is
            # verified by a guard that can fire, not trusted.
            is_refresh = not sf.device_select
            target = (
                earlier if is_refresh else [p for p in earlier if str(p) not in job["immutable"]]
            )
            fp_sel = _pages_fp(engine, target, job["rid"]) if target else {}
            if is_refresh:
                for p in earlier:
                    v = fp_sel[str(p)]
                    old = job["immutable"].get(str(p))
                    if old is not None and old != v:
                        job["imm_conflict"].append(str(p))
                    job["immutable"][str(p)] = v
            else:
                job["immutable"].update(fp_sel)
            boundary = {
                "out_before": len(r.output),
                "seq_before": int(r.seq_len),
                "cmax": len(srow["cand"]),
                "phase": int(r.phase),
                "path": job["path"],
                "device_select": bool(sf.device_select),
                "own": own,
                "selected_pages": kept,
                # forward-POST state of the own window: the trailing partial
                # page here already contains THIS tick's write, so the
                # parent takes tick k's own fingerprints from boundary k-1.
                "own_fp": {str(p): fp_ for p, fp_ in _pages_fp(engine, own, job["rid"]).items()},
            }
            if int(r.phase) == _PHASE_PREFILL:
                job["prefill_boundary"] = boundary
                if (
                    os.environ.get("H2_DUMP_CELL") == f"{job['bucket']}:{job['W']}"
                    and os.environ.get("H2_DUMP_WHEN", "decode") == "prefill"
                ):
                    _raw_dump(engine, kept, job)
                continue
            cur = job["cur"] or {"ids": None, "pos": None, "state": None}
            boundary.update({"ids": cur["ids"], "pos": cur["pos"], "state": cur["state"]})
            job["ticks"].append(boundary)
            job["cur"] = None
            if os.environ.get("H2_DUMP_CELL") == f"{job['bucket']}:{job['W']}" and str(
                len(r.output)
            ) in set(os.environ.get("H2_DUMP_STEPS", "").split(",")):
                _raw_dump(engine, kept, job)
        return dropped

    def run_graph(reqs, chains=None):
        job["path"] = "graph"
        r0 = next((r for r in reqs if r.req_id == job["rid"]), None)
        if r0 is not None:
            ch = chains[reqs.index(r0)] if chains is not None else [r0.output[-1]]
            job["cur"] = {
                "ids": [int(x) for x in ch],
                "pos": [int(r0.seq_len) - 1 + j for j in range(len(ch))],
                "state": _state_fp(engine, int(r0.state_slot)),
            }
        ok = orig_graph(reqs, chains)
        if not ok:
            # Refresh/failed-capture tick: the engine now runs the EAGER sparse
            # forward, whose finalize owns this tick's record. Drop the stale
            # graph entry and relabel so the refresh is not counted as replay.
            job["path"] = "eager"
            job["cur"] = None
            if not getattr(rt, "graph_on", True):
                # Failed sm70 capture poisons the allocator: hard-exit 12 NOW,
                # before another CUDA call (same rule as the arm worker).
                print(
                    "[FATAL parity] sparse graph capture failed; poisoned context",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(12)
        return ok

    def step():
        job["path"] = "eager"
        job["cur"] = None
        return orig_step()

    engine._model.forward = fwd
    rt.finalize = finalize
    rt.run_decode_graph = run_graph
    engine.step = step
    engine._commit = commit


def _raw_dump(engine, pages, job):
    """Byte-exact rerun artifact at the divergence step (H2_DUMP_CELL=b:W and
    H2_DUMP_STEPS=comma pre-commit out_len): the row's state/conv/parity and
    every selected+own page's K/V. This is the inputs_for-equivalent arbiter
    when the sketch fingerprints classify a tick."""
    import torch

    rid = job["rid"]
    r = next(x for x in engine._running if x.req_id == rid)
    slot = int(r.state_slot)
    sp, tr, kv = engine._states, engine._sparse.tracker, engine._kv
    blob = {
        "out_len": len(r.output),
        "seq_len": int(r.seq_len),
        "slot": slot,
        "states": sp.states[slot].cpu().clone(),
        "win_parity": sp.win_parity[slot].cpu().clone(),
        "output": list(r.output),
        "pages": {},
    }
    for p in pages:
        ph = tr.resident[rid].get(p)
        if ph is not None:
            blob["pages"][str(p)] = (kv.k_pool[:, ph].cpu().clone(), kv.v_pool[:, ph].cpu().clone())
    if sp.conv_windows is not None:
        blob["conv_windows"] = sp.conv_windows[slot].cpu().clone()
    path = os.path.abspath(
        f"parity_dump_{job['arm']}_{job['bucket']}_w{job['W']}_o{len(r.output)}.pt"
    )
    torch.save(blob, path)
    print(f"RAW_DUMP {path}", flush=True)


def prime_counting(engine, tok, bucket, depth, tag, job):
    """Prime the counting prompt whose FIRST decode tick lands in `bucket`.
    Filler is repeated single ' z' tokens (exact length, unlike a join whose BPE
    length drifts); eager produces the correct 1,2,3 sequence after it on device.
    Same measure-and-correct loop as prime_at_bucket. Returns (rid, cmax, n).

    job["rid"] is bound at submit so the forward hook hashes the successful
    attempt's prefill logits; per-attempt state is reset before each submit."""
    from tilerl.engine import SamplingParams
    from tilerl.sparse_engine import cmax_bucket

    instr = tok.encode(" Count aloud from one to forty, one number per line:")
    fid = tok.encode(" z")[-1:]
    nfill = tokens_for_bucket(bucket) - len(instr)
    for attempt in range(6):
        ids = fid * nfill + instr
        # N1: bind rid BEFORE every probe step and reset ALL per-attempt
        # collections here. A sparse-graph capture during a priming step then
        # still runs under a live rid and the run_graph os._exit(12) guard, so a
        # capture failure cannot poison the allocator and surface as rc13.
        job["rid"] = engine.submit(
            ids, SamplingParams(temperature=0.0, max_new_tokens=PARITY_GEN, seed=0)
        )
        job["ticks"] = []
        job["commits"] = []
        job["immutable"] = {}
        job["imm_conflict"] = []
        job["prefill_logits"] = []
        job["prefill_boundary"] = None
        job["cur"] = None
        engine.step()
        for _ in range(40000):
            row = next((r for r in engine._running if r.req_id == job["rid"]), None)
            if row is None:
                _row(engine, job["rid"], f"{tag} attempt{attempt} after prefill")
            if row.phase == _PHASE_DECODE:
                break
            if row.phase == _PHASE_DONE:
                raise ProbeError(f"{tag}: DONE before decode")
            engine.step()
        srows = engine._sparse.decode_rows([_row(engine, job["rid"], "cmax")], [1 + depth])
        cmax = max((len(x["cand"]) for x in srows), default=0)
        if cmax_bucket(cmax) == bucket:
            # Discard any records the probe steps beyond prefill appended; the
            # decode measurement starts from the prefill boundary only.
            job["ticks"] = []
            job["commits"] = []
            return job["rid"], cmax, len(ids)
        nfill += (bucket - cmax) * 16
        _cancel_and_drain(engine, job["rid"], f"{tag} attempt{attempt}")
    raise ProbeError(f"{tag}: counting prompt never landed in bucket {bucket}")


def build_parity_worker(source, draft_path, graph, depth, model_name="qwen38-27b"):
    """One arm process, ONE model load for all three buckets at this W. Per
    decode tick records committed token ids and a finalize-boundary fingerprint
    (selection by logical page, per-page K/V, recurrent state, conv, parity)."""
    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl import build
    from tilerl.build import build_engine, build_model
    from tilerl.cli import _qwen38_tokenizer
    from tilerl.sparse_engine import cmax_bucket
    from tilerl.spec import load_draft

    if model_name != "qwen38-27b":
        raise ProbeError("parity mode has no tiny dry-run")
    build.QWEN38_SOURCE = source
    be = get_backend()
    if be.device.type != "cuda":
        raise ProbeError("parity mode needs CUDA")
    cfg, model = build_model(model_name, seed=0, fuse_projections=True)
    draft = load_draft(model, draft_path) if depth else None
    e = build_engine(
        cfg,
        model,
        be,
        num_slots=4,
        max_batch=4,
        max_total_tokens=131072,
        max_num_batched_tokens=512,
        sparse_k=128,
        sparse_min_tokens=0,
        sparse_device_select=True,
        scorer="bounds",
        kv_cold_bytes=int(os.environ.get("H2_COLD_BYTES", str(1 << 30))),
        cold_ssd_path=os.environ.get("H2_COLD_SSD", ""),
        cold_ssd_bytes=int(os.environ.get("H2_COLD_SSD_BYTES", "0")),
        cold_format="f16",
        decode_graph=graph,  # sole arm difference
        draft=draft,
        spec_depth=depth if depth else None,
    )
    tok = _qwen38_tokenizer()
    job = {
        "rid": None,
        "ticks": [],
        "commits": [],
        "prefill_logits": [],
        "prefill_boundary": None,
        "immutable": {},
        "imm_conflict": [],
        "cur": None,
        "path": "eager",
        "arm": "graph" if graph else "eager",
        "bucket": None,
        "W": depth + 1,
    }
    _install_parity_hooks(e, job)

    cells = []
    for bucket in PARITY_BUCKETS:
        job["bucket"] = bucket
        rid, cmax, n_tokens = prime_counting(e, tok, bucket, depth, f"parity b{bucket}", job)
        for _ in range(PARITY_GEN * 12 + 200):
            e.step()
            torch.cuda.synchronize()
            n_tok = sum(len(c["toks"]) for c in job["commits"])
            live = next((r for r in e._running if r.req_id == rid), None)
            if live is None or n_tok >= PARITY_GEN:
                break
        # One decode forward = one finalize boundary AND exactly one _commit
        # (plain: 1 token; W=2 verify: 1-2 tokens). Equal-length index
        # alignment is the B1 invariant the parent cross-checks.
        ticks, commits = job["ticks"], job["commits"]
        if len(ticks) != len(commits):
            raise ProbeError(
                f"parity {job['arm']} b{bucket} W{depth + 1}: "
                f"{len(ticks)} tick boundaries vs {len(commits)} commits"
            )
        for t, c in zip(ticks, commits):
            if t["out_before"] != c["out_before"] or t["seq_before"] != c["seq_before"]:
                raise ProbeError(
                    f"parity {job['arm']} b{bucket} W{depth + 1}: tick/commit "
                    f"misaligned ({t['out_before']},{t['seq_before']}) vs "
                    f"({c['out_before']},{c['seq_before']})"
                )
            t["tokens"] = c["toks"]
        n_tok = sum(len(c["toks"]) for c in commits)
        if n_tok < PARITY_GEN:
            raise ProbeError(
                f"parity {job['arm']} b{bucket} W{depth + 1}: only {n_tok}/{PARITY_GEN} tokens"
            )
        n_graph = sum(1 for t in ticks if t["path"] == "graph")
        final_out = list(e.poll()[rid]) if not any(r.req_id == rid for r in e._running) else None
        cells.append(
            {
                "bucket": bucket,
                "W": depth + 1,
                "observed_cmax": cmax,
                "observed_bucket": cmax_bucket(cmax),
                "n_tokens": n_tokens,
                "ticks": ticks,
                "commits": commits,
                "n_graph": n_graph,
                "prefill_logits": job["prefill_logits"][-1] if job["prefill_logits"] else None,
                "prefill_own_fp": (job["prefill_boundary"] or {}).get("own_fp"),
                "immutable": job["immutable"],
                "imm_conflict": job["imm_conflict"],
                "head": [t for c in commits for t in c["toks"]][:16],
            }
        )
        if final_out is not None and len(final_out) < PARITY_GEN:
            raise ProbeError(f"parity {job['arm']} b{bucket}: finished with {len(final_out)}")

    e.shutdown()
    gc.collect()  # never empty_cache on sm70 after capture
    return {"arm": job["arm"], "W": depth + 1, "cells": cells}


def _input_view(cell, tick_idx):
    """Tick k's INPUTS assembled from boundary k-1: ids/pos/state captured at
    forward entry, geometry from boundary k, own-window K/V as boundary k-1 left
    it (boundary k already contains tick k's write in the trailing page), and
    every selected EARLIER page (the kept set minus own) from the immutable
    table -- those pages never change after write. Tick 0's own window is the
    prefill boundary."""
    t = cell["ticks"][tick_idx]
    prev = cell["ticks"][tick_idx - 1]["own_fp"] if tick_idx else cell.get("prefill_own_fp")
    earlier = [p for p in t["selected_pages"] if p not in t["own"]]
    return {
        "ids": t["ids"],
        "pos": t["pos"],
        "state": t["state"],
        "seq_before": t["seq_before"],
        "cmax": t["cmax"],
        "own": t["own"],
        "selected_pages": t["selected_pages"],
        "own_fp": prev,
        "sel_fp": {str(p): cell["immutable"].get(str(p)) for p in earlier},
    }


def _input_diff(a, b):
    """First unequal component of two tick input views, or None."""
    if a is None or b is None:
        return "missing-input-view"
    for k in (
        "ids",
        "pos",
        "state",
        "seq_before",
        "cmax",
        "own",
        "selected_pages",
        "own_fp",
        "sel_fp",
    ):
        if a.get(k) != b.get(k):
            return k
    return None


def compare_parity(source, draft, depth):
    """Spawn graph first (a capture failure ends the run before the eager 27B
    load pays), then eager. Per cell: prefill-logits parity gate, then walk the
    committed token streams; at the first divergent token classify per fixmisc:
      inputs equal + token differ  -> ..._H1_AT_SKETCH_PRECISION (byte dump
        arbitrates); inputs differ + token differ -> ..._H3:<field>
      Before either, an elementwise alignment gate compares per-tick
      seq_before and commit counts; any mismatch is ALIGNMENT_UNMATCHED
      (harness fault), since the two arms are independent autoregressions.
    """
    g = spawn_worker("parity", source, draft, "graph", depth)
    if g.get("arm") in ("capture-failed", "illegal-access", "worker-error", "worker-timeout"):
        return {"W": depth + 1, "verdict": "PROBE/CAPTURE", "graph": g, "rows": []}
    e = spawn_worker("parity", source, draft, "eager", depth)
    if e.get("arm") in ("illegal-access", "worker-error", "worker-timeout", "capture-failed"):
        return {"W": depth + 1, "verdict": "PROBE/CAPTURE", "eager": e, "rows": []}

    rows, harness, bad = [], [], []
    for gc_, ec in zip(g.get("cells", []), e.get("cells", [])):
        cell = {
            "bucket": gc_["bucket"],
            "W": depth + 1,
            "observed_bucket": gc_.get("observed_bucket"),
            "n_tokens": (gc_.get("n_tokens"), ec.get("n_tokens")),
            "graph_replays": gc_.get("n_graph"),
            "graph_head": gc_.get("head"),
            "eager_head": ec.get("head"),
        }
        if gc_.get("n_tokens") != ec.get("n_tokens"):
            cell["verdict"] = "PROMPT_GEOMETRY_MISMATCH"
            harness.append(cell)
            rows.append(cell)
            continue
        if gc_.get("n_graph", 0) == 0:
            cell["verdict"] = "NO_GRAPH_COVERAGE"
            harness.append(cell)
            rows.append(cell)
            continue
        pl_g, pl_e = gc_.get("prefill_logits"), ec.get("prefill_logits")
        if pl_g is None or pl_e is None:
            cell["verdict"] = "NO_PREFILL_LOGITS"
            harness.append(cell)
            rows.append(cell)
            continue
        if pl_g["sha1"] != pl_e["sha1"] or pl_g["n"] != pl_e["n"]:
            cell["verdict"] = "PREFILL_LOGITS_DIFFER"
            cell["prefill"] = (pl_g, pl_e)
            harness.append(cell)
            rows.append(cell)
            continue
        if gc_.get("imm_conflict") or ec.get("imm_conflict"):
            cell["verdict"] = "IMMUTABLE_PAGE_CHANGED"
            cell["conflicts"] = (gc_.get("imm_conflict"), ec.get("imm_conflict"))
            harness.append(cell)
            rows.append(cell)
            continue

        # B1 alignment gate: the two arms are independent autoregressions, so a
        # token comparison is only meaningful tick-for-tick. Their finalize
        # boundary count, per-tick seq_before and per-tick commit counts must
        # agree elementwise. A W=2 verify accepts 1 or 2 tokens; if the arms'
        # acceptance patterns diverged, the streams cannot be aligned -- that is
        # a harness fault (ALIGNMENT_UNMATCHED), never an H1/H3 verdict.
        g_seq = [t["seq_before"] for t in gc_["ticks"]]
        e_seq = [t["seq_before"] for t in ec["ticks"]]
        g_nc = [len(c["toks"]) for c in gc_["commits"]]
        e_nc = [len(c["toks"]) for c in ec["commits"]]
        if g_seq != e_seq or g_nc != e_nc:
            cell["verdict"] = "ALIGNMENT_UNMATCHED"
            cell["align"] = {
                "tick_count": (len(g_seq), len(e_seq)),
                "first_seq_diff": next(
                    (i for i in range(min(len(g_seq), len(e_seq))) if g_seq[i] != e_seq[i]),
                    None,
                ),
                "commit_counts_g": g_nc,
                "commit_counts_e": e_nc,
            }
            harness.append(cell)
            rows.append(cell)
            continue
        g_flat, g_owner, e_flat, e_owner = [], [], [], []
        for ti, t in enumerate(gc_["ticks"]):
            for tok in t["tokens"]:
                g_flat.append(tok)
                g_owner.append(ti)
        for ti, t in enumerate(ec["ticks"]):
            for tok in t["tokens"]:
                e_flat.append(tok)
                e_owner.append(ti)
        cell["produced"] = (len(g_flat), len(e_flat))
        k = next((i for i in range(min(len(g_flat), len(e_flat))) if g_flat[i] != e_flat[i]), None)
        if k is None and len(g_flat) == len(e_flat):
            cell["verdict"] = "MATCH"
            rows.append(cell)
            continue
        if k is None:
            cell["verdict"] = f"LENGTH_DIVERGE({len(g_flat)} vs {len(e_flat)})"
            harness.append(cell)
            rows.append(cell)
            continue

        gti, eti = g_owner[k], e_owner[k]
        gt, et = gc_["ticks"][gti], ec["ticks"][eti]
        field = _input_diff(_input_view(gc_, gti), _input_view(ec, eti))
        cell["verdict"] = (
            "TOKEN_DIVERGE_INPUT_EQUAL_H1_AT_SKETCH_PRECISION"
            if field is None
            else f"TOKEN_DIVERGE_INPUT_DIFF_H3:{field}"
        )
        cell["first"] = {
            "token_pos": k,
            "graph_tick": gti,
            "eager_tick": eti,
            "graph_token": g_flat[k],
            "eager_token": e_flat[k],
            "graph_path": gt["path"],
            "graph_device_select": gt["device_select"],
            "input_field": field,
            "graph_tick_ids": gt["ids"],
            "eager_tick_ids": et["ids"],
        }
        bad.append(cell)
        rows.append(cell)

    if harness:
        return {
            "W": depth + 1,
            "verdict": "PROBE",
            "rows": rows,
            "faults": [c["verdict"] for c in rows if c["verdict"] != "MATCH"],
        }
    if bad:
        return {
            "W": depth + 1,
            "verdict": "BAD",
            "rows": rows,
            "faults": [c["verdict"] for c in rows if c["verdict"] != "MATCH"],
        }
    return {"W": depth + 1, "verdict": "MATCH", "rows": rows, "faults": []}


def compare_bucket(source, draft, bucket, depth):
    g = spawn_worker("arm", source, draft, bucket, depth, "graph")
    # First capture failure ends the sweep in main(): do not pay a 27B eager
    # build here (v3 burned most of its window on engines past the first fault).
    if g.get("arm") in ("capture-failed", "illegal-access", "worker-error", "worker-timeout"):
        return {
            "bucket": bucket,
            "W": depth + 1,
            "verdict": "ILLEGAL" if g.get("arm") == "illegal-access" else "PROBE/CAPTURE",
            "graph": g,
            "eager": None,
            "token_ok": False,
            "produced_ok": False,
        }
    e = spawn_worker("arm", source, draft, bucket, depth, "eager")
    if e.get("arm") in ("illegal-access", "worker-error", "worker-timeout"):
        return {
            "bucket": bucket,
            "W": depth + 1,
            "verdict": "ILLEGAL" if e.get("arm") == "illegal-access" else "PROBE/CAPTURE",
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
        "observed_bucket": g.get("observed_bucket"),
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
    ap.add_argument(
        "--parity",
        action="store_true",
        help="#804 per-tick graph-vs-eager token+inputs parity over 6 cells",
    )
    ap.add_argument("--parity-out", default="parity_result.json")
    ap.add_argument("--worker", nargs="*", default=None)
    args = ap.parse_args()

    # --- in-process worker (one fresh CUDA context per process) ---
    if args.worker is not None:
        try:
            kind = args.worker[0]
            if kind == "arm":
                _, source, draft, bucket, depth, mode = args.worker
                model_name = os.environ.get("H2_MODEL", "qwen38-27b")
                out = build_engine_arm(
                    source,
                    draft,
                    int(bucket),
                    int(depth),
                    mode == "graph",
                    model_name=model_name,
                )
            elif kind == "parity":
                _, source, draft, mode, depth = args.worker
                out = build_parity_worker(source, draft, mode == "graph", int(depth))
            elif kind == "b4":
                _, source, draft = args.worker
                out = build_b4(source, draft)
            else:
                raise ProbeError(f"unknown worker kind {kind}")
            print(json.dumps(out), flush=True)
            return 0
        except ProbeError as exc:
            print(json.dumps({"arm": "probe-error", "note": str(exc)}), flush=True)
            return 13
        except Exception as exc:
            note = f"EXC {type(exc).__name__}: {exc}"
            print(json.dumps({"arm": "worker-exc", "note": note}), flush=True)
            # CUDA illegal memory access surfaces as a runtime error mid-tick;
            # keep it distinct (11) from harness faults (13) and capture (12).
            if "illegal" in note.lower():
                return 11
            return 13

    # --- #804 parity: 6 cells (3 cmax buckets x W=1,2), 2 arm processes/W ---
    if args.parity:
        if not args.source or not args.draft:
            print("source and --draft required", file=sys.stderr)
            return 13
        out = []
        verdict = "MATCH"
        for depth in (0, 1):
            r = compare_parity(args.source, args.draft, depth)
            out.append(r)
            for c in r.get("rows", []):
                print(
                    f"[parity b{c['bucket']:5d} W={c['W']}] {c['verdict']} "
                    f"cmax={c.get('observed_bucket')} graph_replays={c.get('graph_replays')} "
                    f"g_head={c.get('graph_head')} e_head={c.get('eager_head')}",
                    flush=True,
                )
                if c.get("first"):
                    print(json.dumps(c["first"], indent=1), flush=True)
            if r["verdict"] == "PROBE/CAPTURE":
                verdict = "CAPTURE_OR_PROBE"
            elif r["verdict"] == "PROBE":
                verdict = "PROBE"
            elif r["verdict"] == "BAD" and verdict == "MATCH":
                verdict = "BAD"
        with open(args.parity_out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("=" * 60)
        print(
            {
                "MATCH": "PARITY_MATCH",
                "BAD": "PARITY_BAD",
                "PROBE": "PARITY_PROBE_ERROR",
                "CAPTURE_OR_PROBE": "PARITY_CAPTURE_OR_PROBE",
            }[verdict]
        )
        return {"MATCH": 0, "BAD": 10, "PROBE": 13, "CAPTURE_OR_PROBE": 12}[verdict]

    # --- B=4 only ---
    if args.only_b4:
        r = spawn_worker("b4", args.source, args.draft)
        print(json.dumps(r, indent=2))
        if r.get("arm") == "capture-failed":
            return 12
        if r.get("arm") in ("worker-error", "worker-timeout"):
            return 13
        if r.get("arm") == "illegal-access" or "illegal" in str(r.get("note", "")).lower():
            return 11
        return 0 if r.get("ok") else 11

    if not args.source or not args.draft:
        print("source and --draft required", file=sys.stderr)
        return 13

    results: list = []
    bad: list = []
    capture_fail = illegal = False
    probe_fault = None
    for depth in (0, 1):
        for b in args.buckets:
            r = compare_bucket(args.source, args.draft, b, depth)
            results.append(r)
            print(
                f"[bucket {b:5d} W={depth + 1}] {r['verdict']} "
                f"cmax={r.get('observed_cmax')} bucket={r.get('observed_bucket')} "
                f"captured={r.get('captured')} "
                f"g_tok={r.get('graph_token')} e_tok={r.get('eager_token')} "
                f"g_steps={r.get('graph_steps')} e_steps={r.get('eager_steps')} "
                f"produced_ok={r.get('produced_ok')}",
                flush=True,
            )
            if r["verdict"] == "ILLEGAL":
                illegal = True
                break
            if r["verdict"] == "PROBE/CAPTURE":
                fault = r["graph"] if r.get("graph") else r.get("eager")
                if fault and fault.get("arm") == "capture-failed":
                    capture_fail = True
                else:
                    probe_fault = r
                break
            if not (r["token_ok"] and r["produced_ok"]):
                bad.append(r)
        if illegal or capture_fail or probe_fault:
            break

    if not (illegal or capture_fail or probe_fault) and not args.skip_b4:
        b4 = spawn_worker("b4", args.source, args.draft)
        print(
            f"[b4 d1] ok={b4.get('ok')} finished={b4.get('finished')} "
            f"leaked={b4.get('leaked_slots')} :: {b4.get('note')}",
            flush=True,
        )
        if b4.get("arm") == "capture-failed":
            capture_fail = True
        elif b4.get("arm") in ("worker-error", "worker-timeout"):
            probe_fault = {"b4": b4}
        elif b4.get("arm") == "illegal-access" or "illegal" in str(b4.get("note", "")).lower():
            illegal = True
        elif not b4.get("ok"):
            bad.append({"bucket": "b4", **b4})

    print("=" * 60)
    if probe_fault:
        print("H2_PROBE_ERROR")
        print(json.dumps(probe_fault, indent=2))
        return 13
    if capture_fail:
        print("H2_CAPTURE_FAILED")
        print(json.dumps(results, indent=2))
        return 12
    if illegal:
        print("H2_ILLEGAL")
        print(json.dumps(results, indent=2))
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
