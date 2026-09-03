"""Steady-state decode rate vs context length — no prefill in the window.

The two-point slope in bench_workloads.py cannot measure long context: a 4K
prompt takes 8 chunked-prefill ticks at max_num_batched_tokens=512, speculation
is off on every mixed tick (engine.py:790), and at lo=32 those ticks dominate
the lo point instead of cancelling. That is why 4K read as 14.8 tok/s spec and
18.0 dense — both prefill, neither decode.

This drives the engine directly and times only ticks where the request is in
DECODE phase, so the window contains no prefill at all. Reports tok/s and
tokens per trunk forward against context length.

  scripts/v100.sh run lc 'CKPT=...; /usr/bin/python3 -u scripts/bench_ctx_decode.py \
      --source $CKPT [--draft $CKPT/model-00018-of-00018.safetensors]'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS
from tilerl.spec import load_draft

#: Every point costs THREE full prefills (two warmups + the measure) at ~31 ms per
#: prompt token, so 32768 is ~51 min on its own — pair --min-ctx/--max-ctx to walk
#: the long end one point per run rather than sweeping into a multi-hour job.
CTXS = [32, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def _sync() -> None:
    """Drain the device before reading the clock. Conditional so the control flow
    below is testable on a CPU-only box, which is where the batch=4 slot leak that
    killed a 20-minute pod run would have been caught."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prompt(ctx: int, i: int, vocab: int) -> list[int]:
    """A ctx-token prompt drawn from ONE fixed distribution over the whole vocab.

    Was `range(10 + i*ctx, 10 + (i+1)*ctx)`, which makes the prompt's CONTENT a
    function of ctx: ctx=1024 read ids 10..1033 (0.4% of the vocab, mostly
    byte-level and special tokens) while ctx=16384 read 10..16393 (6.6%, real
    word pieces). Acceptance is a property of what the draft is predicting, so
    every tok/forward in the sweep mixed "the context is longer" with "the prompt
    is different" and the two cannot be separated after the fact -- measured
    2.86 -> 2.03 across that pair, which is 1.41x of a 1.80x rate drop.

    Same generator seed for every ctx, so a shorter prompt is a PREFIX of a
    longer one in distribution: length is then the only variable.
    """
    g = torch.Generator().manual_seed(1000 + i)
    # Not a default-able argument: vocab=0 would make randint raise from inside a
    # 20-minute pod run rather than here, and vocab=<too small> would quietly
    # narrow the distribution -- which is the exact confound this function removes.
    if vocab < 1024:
        raise SystemExit(f"_prompt: vocab={vocab} is not this model's vocabulary size")
    return torch.randint(0, vocab, (ctx,), generator=g).tolist()


def measure(e, ctx: int, tokens: int, batch: int = 1, vocab: int = 0) -> tuple[float, float, int]:
    """tok/s and tok/forward over DECODE ticks only.

    ``batch`` submits that many concurrent requests, which is the only way a tick
    reaches B*W rows: the engine's max_batch is an upper bound, so one request
    speculates at W rows and never touches the rungs serving actually compiles
    (B=4 W=4 is 16 rows -> rung 32, where ncols=2 turns on; B=1 W=4 is 4 rows and
    does not). A B=1 run silently measured the same kernel in both arms of the
    spec-ncols A/B -- errors/2026-09-03-the-spec-ncols-ab-ran-at-b1.md.

    Only _run_forward sees the batch a tick really ran: tok/forward cannot stand in
    for it, since a fully-rejected batch still yields one token per request per tick
    and so reads ~batch no matter how wide the tick was.
    """
    rows: list[int] = []
    batches: list[int] = []
    orig = type(e)._run_forward

    def spy(self, decodes, prefills, chunks):
        if decodes and not prefills:
            rows.append(sum(1 + len(r.drafts) for r in decodes))
            batches.append(len(decodes))
        return orig(self, decodes, prefills, chunks)

    rids = [e.submit(_prompt(ctx, i, vocab),
                     SamplingParams(temperature=0.0, max_new_tokens=tokens, seed=0))
            for i in range(batch)]
    # poll() DRAINS _finished for every request, so anything it returns must be kept:
    # discarding it here (the old `and e.poll()` truth test) threw away an output the
    # window below was waiting for, and that loop then spun its whole tick budget with
    # the GPU at 0% -- six minutes of a live-looking run. Every loop below is bounded in
    # WALL CLOCK as well as ticks, because an empty tick costs nothing and a tick cap
    # cannot tell "spinning" from "working".
    done: dict = {}
    # Prefill is ~31 ms per PROMPT token on this card (see the prefill-roofline entry), so
    # the budget has to scale with batch*ctx or it fires on work that is merely long:
    # B=8 ctx=1024 is 8192 prompt tokens = ~254 s of legitimate prefill, and a flat 300 s
    # killed it at 8/8 admitted -- reported as a stall when nothing was stuck. 4x that
    # estimate, floored at the old 300 s.
    prefill_budget = max(300.0, 4 * batch * ctx * 0.031)
    deadline = time.perf_counter() + prefill_budget
    for _ in range(4096):  # burn the prefill chunks for every request
        done.update(e.poll())
        if any(rid in done for rid in rids):
            raise SystemExit(f"ctx={ctx}: a request finished during prefill; lower --tokens")
        reqs = [next((r for r in e._running if r.req_id == rid), None) for rid in rids]
        if all(r is not None and r.phase == _PHASE_DECODE for r in reqs):
            break
        if time.perf_counter() > deadline:
            raise SystemExit(f"ctx={ctx}: prefill stalled {prefill_budget:.0f} s with "
                             f"{sum(r is not None for r in reqs)}/{batch} admitted")
        e.step()
    else:
        raise SystemExit(f"ctx={ctx}: prefill did not finish in 4096 ticks")
    _sync()
    s0, t0 = e.stats(), time.perf_counter()
    # Close the window at the FIRST completion: speculation accepts different numbers
    # of drafts per request, so past that point ticks run at B-1, B-2, ... and dilute
    # the measurement with the narrow ticks this batch flag exists to avoid.
    type(e)._run_forward = spy
    try:
        deadline = time.perf_counter() + 600.0
        for _ in range(8 * tokens + 64):
            e.step()
            done.update(e.poll())
            if any(rid in done for rid in rids):
                break
            if time.perf_counter() > deadline:
                raise SystemExit(f"ctx={ctx}: window stalled 600 s, {len(done)} finished")
        else:
            raise SystemExit(f"ctx={ctx}: no request completed in the tick budget")
        _sync()
        wall, s1 = time.perf_counter() - t0, e.stats()
    finally:
        type(e)._run_forward = orig
    # Retire everything before returning: submit() takes a state slot at admission and
    # only _finish returns it, so a request left in EITHER queue holds one and the next
    # call raises "LinearStatePool exhausted". _running alone is not the idle condition
    # (engine.py checks _waiting too) -- draining only it is why the first fix still died.
    deadline = time.perf_counter() + 600.0
    for _ in range(16 * tokens + 256):
        if not e._running and not e._waiting:
            break
        e.step()
        e.poll()
        if time.perf_counter() > deadline:
            raise SystemExit(f"ctx={ctx}: drain stalled 600 s with {len(e._running)} running")
    else:
        raise SystemExit(f"ctx={ctx}: {len(e._running)} running + {len(e._waiting)} "
                         f"waiting would not retire")
    n = s1["tokens_generated"] - s0["tokens_generated"]
    fwd = s1["decode_forwards"] - s0["decode_forwards"]
    mixed = s1["mixed_forwards"] - s0["mixed_forwards"]
    if mixed:  # a mixed tick never speculates; it would dilute tok/forward
        raise SystemExit(f"ctx={ctx}: {mixed} mixed ticks inside the window")
    # Guard the BATCH, not the row count. An earlier version asserted
    # max(rows) >= batch * width, which is wrong for a reason that only shows up as the
    # batch grows: verify_lens trims each request's chain independently
    # (engine.py:991-997), so requiring the full B*W demands all B requests keep every
    # draft in the SAME tick -- B independent events, so the false-positive rate rises
    # with B. It killed a healthy B=8 run at 28 rows (8 requests, 4 drafts trimmed).
    # The claim worth checking is "did B requests actually decode together", which the
    # trim cannot affect.
    # Named `batches`, not `reqs`: the admission check below rebinds `reqs` to a list of
    # _Req objects, and max() over those raises TypeError instead of comparing counts.
    if batches and max(batches) < batch:
        raise SystemExit(f"ctx={ctx}: widest tick had {max(batches)} requests, expected "
                         f"{batch}; this is not the batch it claims to be")
    if rows and max(rows) < batch:  # a chain trimmed to nothing still submits 1 row each
        raise SystemExit(f"ctx={ctx}: widest tick was {max(rows)} rows for {batch} "
                         f"requests; the spy is not seeing verify ticks")
    return n / wall, n / max(fwd, 1), n


def timed(e, ctx: int, tokens: int, batch: int = 1, vocab: int = 0) -> tuple[float, float, str]:
    """Warm this context, then measure it, and flag an unwarmed reading.

    A speculative run captures a CUDA graph per (batch, chain width), and a
    width first seen inside a timed window puts its multi-second compile in the
    measurement — the third time that has silently ruined a number here. Two
    warmups, not one: the first also absorbs the kernel JIT that fires on the
    very first call at a new context. The ratio check then catches any capture
    that still slipped through, since a capture is seconds against a tick of
    tens of ms and a clean pair agrees closely.

    Flags rather than raises: a SystemExit here leaves the engine holding the
    whole card, and the orphan is invisible until the next run OOMs.
    """
    measure(e, ctx, tokens, batch, vocab)
    warm, _, _ = measure(e, ctx, tokens, batch, vocab)
    tps, per_fwd, _ = measure(e, ctx, tokens, batch, vocab)
    return tps, per_fwd, " UNWARMED" if tps > 2 * warm else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1,
                    help="concurrent requests per tick; the rung a verify tick compiles "
                         "keys on B*W, so B=1 never reaches the 32 rung serving uses")
    ap.add_argument("--max-ctx", type=int, default=max(CTXS),
                    help="skip contexts above this. B=4 OOMs at ctx>=512 on a 32 GB card: "
                         "paged_attention asked for 1.50 GiB with 0.69 free, and the block "
                         "pool is only 0.03 GB of that, so it is transient kernel work "
                         "scaling with B*S*history, not something a pool size fixes")
    ap.add_argument("--min-ctx", type=int, default=0,
                    help="skip contexts below this. Raising --max-ctx grows the KV pool "
                         "(blocks come from max(ctxs)), so a ceiling probe re-runs every "
                         "shorter context with less free memory than its own run had: "
                         "--max-ctx 1024 costs 236 MB more pool than 512 and stalled the "
                         "ctx=512 step that reads 88.5 tok/s on its own. Pair the two to "
                         "measure ONE context at its own pool size")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    # cli binds _QWEN38_SOURCE from the env at import, which already happened.
    cli._QWEN38_SOURCE = args.source

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.draft else None
    # max_batch tracks the submissions. It used to be floored at 4, which quadrupled a
    # B=1 run's block pool and its graph count for rows that could never be admitted --
    # noise in exactly the single-stream config a serve default has to be chosen from.
    # Floored at 2 only because num_slots = b + 2 needs headroom for the padding slot
    # engine.py:827 reserves.
    b = max(2, args.batch)
    # num_slots > max_batch on purpose: a tick with fewer rows than its graph bucket
    # permanently reserves one slot for padding rows (engine.py:827), taken from this
    # same pool and never returned. With num_slots == max_batch that leaves b-1 for
    # requests and the next submit() raises "LinearStatePool exhausted" -- which it did,
    # twice, and the engine hides it by falling back to an exact-size graph.
    slots = b + 2
    # Blocks for max_batch concurrent requests at the longest context -- NOT num_slots,
    # which is 2 higher and bought 0.5 GB of pool that OOMed a 32 GB card at B=4. A block
    # costs 2.125 MiB on sm70: 2.000 trunk (16 full-attn planes x 4 kv heads x 16 tokens
    # x 256 head_dim x 4 B, k and v) + 0.125 for the draft's plane, which mirrors
    # num_blocks. f32 because sm70's attention IO is f32 (engine.py:1202); a bf16 card
    # pays half. Measured by scripts/probe_block_bytes.py -- the 0.92 MB this comment
    # used to claim was 2.42x low, bf16 and without the draft plane.
    # The +32 is slack for block-boundary rounding, not for one more request.
    ctxs = [c for c in CTXS if args.min_ctx <= c <= args.max_ctx]
    if not ctxs:
        raise SystemExit(
            f"--min-ctx {args.min_ctx} --max-ctx {args.max_ctx} excludes every context in {CTXS}"
        )
    blocks = b * (-(-(max(ctxs) + args.tokens + 2 * (1 + args.depth)) // BLOCK_TOKENS)) + 32
    # Print it: the pool is a function of max(ctxs), so two runs that both report a
    # "ctx=512" row can be holding different amounts of free memory. Comparing those
    # rows across runs stalled one at a context the other measured at 88.5 tok/s.
    print(f"pool: {blocks} blocks ({blocks * 2.125:.0f} MiB) sized for ctx={max(ctxs)}, "
          f"sweeping {ctxs}")
    e = build_engine(cfg, model, backend, num_blocks=blocks, num_slots=slots, max_batch=b,
                     # Follows the sweep, not a constant: a hardcoded 8192 rejected the
                     # ctx=8192 row itself, since a request is ctx + tokens + drafts.
                     max_total_tokens=max(ctxs) + args.tokens + 2 * (1 + args.depth),
                     draft=draft, spec_depth=args.depth if draft else 1)
    # Capture every (bucket, width) up front. The trim varies W per tick, so waiting for
    # warmup to happen to hit each one is a lottery: at B=1 there are 4 graphs and two
    # warmups absorbed them, but B=4 needs 12 (~14 s each) and the first row came back
    # flagged UNWARMED. cli.py and prof_serve_ramp.py both do this; this bench did not.
    t0 = time.perf_counter()
    n_graphs = e.precapture()
    if n_graphs:
        print(f"precapture: {n_graphs} graphs in {time.perf_counter() - t0:.0f}s")
    label = f"spec d{args.depth}" if draft else "dense"
    w = 1 + args.depth if draft else 1
    rows = args.batch * w
    print(f"\n{label} B={args.batch} ({rows} rows/tick): "
          f"{'ctx':>6} {'tok/s':>8} {'ms/tok':>8} {'tok/fwd':>8}")
    for ctx in ctxs:
        tps, per_fwd, flag = timed(e, ctx, args.tokens, args.batch, cfg.vocab_size)
        print(f"{ctx:>6} {tps:>8.1f} {1000 / tps:>8.1f} {per_fwd:>8.2f}{flag}")


def _self_check() -> None:
    """The prompt must depend on ctx only through its LENGTH.

    Runs on the GPU-less box, before a 20-minute pod job: the old prompt was
    `range(10 + i*ctx, ...)`, so ctx also chose which slice of the vocabulary the
    draft had to predict, and the acceptance column mixed two causes with no way
    to separate them afterwards. errors/2026-09-03-the-context-sweep-changed-the-prompt.md
    """
    V = 248320
    short, long = _prompt(1024, 0, V), _prompt(16384, 0, V)
    assert short == long[:1024], "a shorter prompt must be a prefix of a longer one"
    # The distributions must agree, which the old prompt's did not: mean id 522 vs 8202.
    m_s, m_l = sum(short) / len(short), sum(long) / len(long)
    assert abs(m_s / m_l - 1) < 0.10, f"prompt distribution shifts with ctx: {m_s} vs {m_l}"
    assert _prompt(64, 0, V) != _prompt(64, 1, V), "each request needs its own prompt"
    try:
        _prompt(64, 0, 0)
    except SystemExit:
        pass
    else:
        raise AssertionError("vocab=0 must be refused here, not inside a pod run")
    print("bench_ctx_decode: prompt depends on ctx only through length")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
