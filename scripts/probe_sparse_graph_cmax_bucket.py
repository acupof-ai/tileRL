"""H2 probe — does a lazy sparse-graph capture at a cmax BUCKET boundary contaminate?

v5 (2026-09-17) fixes the B=4 arm. v4's B=4 child used 8311-token rows under
the serve's 512-token prefill cap; the four rows staggered into decode, only
B=1/B=2 graphs formed and the missing-B=4 guard exited rc12, so the
sparse+d1 B=4 illegal-access question went untested. v5 admits four SHORT
equal rows (H2_B4_N=135) under a cap that holds them in one prefill tick
(H2_B4_PREFILL_CAP=1024, asserted 4*n <= cap); their first decode tick is a
genuine B=4 wave capturing (B=4,W,64,own_w). CPU tiny now gates wave
formation too. The bucket/W comparison arms are unchanged from v4, which
measured H2_BAD on V100 (6/6 first-token contamination).

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
  --buckets …      sparse cmax buckets (default 512/1024/2048)
  --worker arm <source> <draft> <bucket> <depth> <graph|eager>
  --worker b4  <source> <draft>      (internal; spawned by the parent)

Both arms are built with sparse_min_tokens=0 and sparse_device_select=True so
the graph gate can pass and graph/eager differ ONLY in decode_graph. The
worker measures the first-decode cmax and corrects n by 16 tokens per unit
cmax (resubmitting) until the target bucket is observed.

The B=4 child uses SHORT equal rows (H2_B4_N=135) under a prefill cap that
holds all four in one tick (H2_B4_PREFILL_CAP=1024, asserted 4*n <= cap) so
the four rows enter decode together and a genuine B=4 sparse graph is captured
(cmax clamps to floor bucket 64). The v4 B=4 arm used 8311-token rows under
the serve's 512 cap; those staggered and never formed B=4 (rc12).

Mechanism fork — H2_ID_OFFSET (device, sm70/sm90; needs no src change). The
measured H2_BAD graph token is the last INPUT id, which at offset 0 also equals
n+6, so "input/embedding echo" and "baked position/seq buffer" coincide.
Separate them by holding positions/n fixed and shifting every token id, two
short graph sweeps at the same bucket (no B=4, one engine per arm as usual):

  H2_ID_OFFSET=0    ... <ckpt> --draft <mtp> --buckets 512 1024 --skip-b4
  H2_ID_OFFSET=5000 ... <ckpt> --draft <mtp> --buckets 512 1024 --skip-b4

Predicted last-input id is ((n-1+K) % 31000) + 7: bucket 512 (n=8311) gives
8317 at K=0 -> 13317 at K=5000; bucket 1024 (n=16503) gives 16509 -> 21509.
Read the GRAPH token per (bucket, K); the eager token is only the control that
the shifted inputs changed the model (it is a genuine next-token, not n+6, and
moves with K — already true on CPU tiny, 133 -> 188).

  graph token at K=5000 == 13317 / 21509 (tracks the shifted last-input id)
      -> the replay echoes the INPUT/embedding: a last_only / copy_ artifact.
  graph token stays 8317 / 16509 (moves only with n, not the id offset)
      -> a POSITION/seq static buffer is baked into the captured forward.
  graph token equals the K=5000 EAGER token
      -> contamination absent at that offset (content/bucket dependent); sweep
         more K before concluding — do not assign either branch.

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


#: Constant id OFFSET added to every prompt token (mod vocab), positions and n
#: unchanged. Mechanism discriminator for a future device arm: the default
#: prompt has ids[t]=(t%31000)+7, so its last input id at bucket 512 (n=8311)
#: is 8317 and at 1024 (n=16503) is 16509 -- EXACTLY the corrupted graph
#: argmax on V100. Rerun the graph arm with H2_ID_OFFSET != 0:
#:   graph token == new last input id  -> the replay echoes the INPUT/embedding
#:                                        token (a last-only / copy_ artifact);
#:   graph token == n+6 regardless     -> it echoes a POSITION/seq static buffer
#:                                        (capture baked seq_len into logits).
#: Both arms use the same offset, so CPU graph=eager parity is unaffected.
ID_OFFSET = int(os.environ.get("H2_ID_OFFSET", "0"))
_ID_MOD = 31000
_ID_BASE = 7


def _prompt_id(t: int) -> int:
    return ((t + ID_OFFSET) % _ID_MOD) + _ID_BASE


def prime_to_decode(engine, n_tokens: int, tag: str):
    """Submit one row, advance ONLY to its first DECODE tick while it is live."""
    from tilerl.engine import SamplingParams

    ids = [_prompt_id(t) for t in range(n_tokens)]
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


#: B=4 needs all four rows to co-prefill in ONE tick and enter decode together.
#: The serve's 512-token prefill cap staggers four long rows (the v4 rc12: only
#: B=1/B=2 graphs formed). Size the rows to fit one tick instead of raising the
#: cap to swallow four 8k prefills: n=135 -> 4*135=540 tokens in one ~1024 cap,
#: cmax clamps to the floor bucket 64, and the illegal-access question is the
#: B=4 concurrency, not the bucket depth. H2_B4_N / H2_B4_PREFILL_CAP override.
B4_N = int(os.environ.get("H2_B4_N", "135"))
B4_PREFILL_CAP = int(os.environ.get("H2_B4_PREFILL_CAP", "1024"))


def build_b4(source, draft_path):
    """B=4 concurrent sparse+d1 graph rows, a single child. The four rows are
    short and equal so a single prefill tick admits them together and their first
    decode tick is a genuine B=4 wave (key B=4 captured) — the v4 shape only ever
    reached B=2. H2_MODEL=tiny is the depth=0 CPU structural dry-run (no 27B
    draft, cannot exercise the d1 illegal risk but proves the B=4 wave forms and
    frees its slots)."""
    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl import build
    from tilerl.build import build_engine, build_model
    from tilerl.engine import SamplingParams
    from tilerl.spec import load_draft

    dry = os.environ.get("H2_MODEL", "qwen38-27b") == "tiny"
    if 4 * B4_N > B4_PREFILL_CAP:
        raise ProbeError(
            f"B=4 prefill cap {B4_PREFILL_CAP} < 4*{B4_N}={4 * B4_N}; "
            "the rows would stagger and no B=4 wave forms (the v4 rc12)"
        )
    if not dry:
        build.QWEN38_SOURCE = source
    be = get_backend()
    cfg, model = build_model("tiny" if dry else "qwen38-27b", seed=0, fuse_projections=not dry)
    draft = None if dry else load_draft(model, draft_path)
    n = B4_N
    res = {
        "arm": "b4",
        "dry": dry,
        "n_tokens": n,
        "prefill_cap": B4_PREFILL_CAP,
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
        max_num_batched_tokens=B4_PREFILL_CAP,
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
    b4_width = 1 if dry else 2  # depth 0 -> W=1; real d1 -> W=2
    try:
        rids = []
        for k in range(4):
            ids = [(((t + 7919 * (k + 1) + ID_OFFSET) % _ID_MOD) + _ID_BASE) for t in range(n)]
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
        # The gate is the WAVE, not the tokens. n=135 clamps cmax to the floor
        # bucket 64; the key must be (B=4, W=dry1/real2, 64, own_w 8/9). CPU tiny
        # captures the W=1 graph (the dry seam has no draft, so no W=2), which
        # still proves the four rows co-decoded. Require the exact B=4 key in
        # both modes: token==128 + slots==0 alone passed the v3 staggered shape.
        own_w = 8 if dry else 9
        expected_b4_key = (4, b4_width, 64, own_w)
        captured_b4 = expected_b4_key in e._sparse_graphs
        res["ok"] = captured_b4 and res["finished"] == expected_tokens and res["leaked_slots"] == 0
        if not captured_b4:
            # Not OOM poisoning (a B<=3 key may have captured) and not an illegal
            # access: the harness failed to form a B=4 wave. Hard-exit 12 so the
            # parent cannot read a token-only pass as the B=4 concurrency test.
            print(
                f"[FATAL b4] expected key {expected_b4_key} absent; keys={res['captured_keys']}",
                file=sys.stderr,
                flush=True,
            )
            os._exit(12)
        if not res["ok"]:
            res["note"] = (
                f"finished={res['finished']} expected={expected_tokens} "
                f"slots_used={res['leaked_slots']} key={expected_b4_key}"
            )
        else:
            res["note"] = f"clean key={expected_b4_key}"
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
