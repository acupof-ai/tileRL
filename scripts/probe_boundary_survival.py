"""Between the boundary publish and DONE, does the entry survive spillable?

Candidate 3 of `errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md` publishes at
the interior boundary and spills that entry later, at DONE. Its headline -- 15 unspillable
prompts against today's 2448 -- was computed from the chunk walk alone, with NO store
pressure: it counts prompts whose boundary is predictable, not prompts whose entry is still
there when the spill happens.

This measures the second thing. The window is real: `_finish_prefills` publishes the first
interior boundary (engine.py:1069) and the prompt-complete entry lands at DONE, and between
them sit every publish from every other row in the batch.

Two outcomes, and they answer DIFFERENT questions -- conflating them is the trap:

  spilled   entry present, snapshot resident -> reaches DISK, survives a restart
  demoted   entry present, snapshot on the host tier -> ruled: no promote at DONE, so it
            never reaches disk; but `lookup` DOES promote it, so it still SERVES a hit
  no-entry  evicted -> lost to both

So there are two yields and the tier moves only one of them:

  spill_yield = clamp(hbm_slots - batch, 0, batch) / batch    -- the tier does not change it
  match_yield -- the tier takes it from 0% to 100%, because `_entries_capacity`
                 (kv_cache.py:1444) adds the host budget to the same denominator and a
                 demoted entry stays in the index

`demoted` and `no-entry` look alike only from the disk's side. From the second turn's side
they are 100 percentage points apart: one is a 12.7 ms H2D, the other is a re-prefill.

Which makes the disk half the smaller one: `match_yield` is what serves a second turn and
the DRAM tier already takes it to 100%, while `spill_yield` feeds `KvTier`, which carries a
REJECT on the serve path (1.65x worse per turn, 0 hits at 12 sessions,
errors/2026-09-06-the-ssd-tier-is-165x-worse-at-12-sessions.md) and is off by default.

Run: python3 scripts/probe_boundary_survival.py
Result: `spill_yield` is LINEAR in the snapshot budget, not a threshold --
`survivors = clamp(hbm_slots - batch, 0, batch)`, zero mismatches over batches 2..32 and
every slot count from 0 to 2*batch+2. Each row holds TWO snapshots between its boundary
publish and its DONE (the boundary entry and the prompt-complete one), so each row's DONE
displaces one slot and the boundary entries are the LRU victims, having been published
first. At the V100 config `build_engine` describes (9 resident snapshots,
`engine.py:1668-1682`) and the default `max_batch=8`, that is 1 of 8 -- 12.5%.

An earlier version of this probe sampled slots 9/16/17/32 only, which are the ramp's two
ends plus a midpoint, and reported a step function with no transition. A sweep that samples
only endpoints cannot tell a ramp from a step.

The tier is off by default on every card (`engine.py:1556 dram_bytes=0`,
`cli.py:1049 --dram-bytes default 0`).

Run: python3 scripts/probe_boundary_survival.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages",
                                "tilerl-kernels", "src"))
os.environ.setdefault("TILERL_TARGET", "cpu")

import torch  # noqa: E402

from tilerl.kv_cache import BLOCK_TOKENS, DramSnapshots, PagedKvPool, PrefixStore  # noqa: E402


def _snap(nbytes: int) -> tuple:
    """A snapshot of the given size, in the shape the store expects: (states, windows)."""
    n = max(1, nbytes // 4)
    return (torch.zeros(n, dtype=torch.float32), None)


def _classify(store: PrefixStore, tokens: tuple[int, ...]) -> str:
    """What a DONE-time spill would find, read off the store's own state."""
    h = store._hash_all(tokens)
    for e in store._entries.get(h, ()):
        if e.tokens == tokens:
            if e.demoted or e.state is None:
                return "demoted"
            return "spilled"
    return "no-entry"


def run(batch: int, snapshot_bytes: int, budget_slots: int, dram_slots: int,
        prompt_blocks: int) -> dict:
    """One tick-ordered pass: every row publishes its boundary, then every row DONEs.

    The interleave is what the window is made of -- a row's boundary entry ages by every
    OTHER row's publish before its own DONE arrives.
    """
    pool = PagedKvPool(num_blocks=batch * prompt_blocks * 4 + 64, num_layers=1,
                       num_kv_heads=1, head_dim=8, device=torch.device("cpu"))
    kw = {"capacity": 4096, "state_bytes": snapshot_bytes * budget_slots}
    if dram_slots:
        kw["dram"] = DramSnapshots(budget_bytes=snapshot_bytes * dram_slots)
    store = PrefixStore(pool, **kw)

    boundary_blocks = max(1, prompt_blocks // 2)
    boundaries = []
    for r in range(batch):
        toks = tuple(range(r * 100000, r * 100000 + boundary_blocks * BLOCK_TOKENS))
        blocks = [pool.alloc_block() for _ in range(boundary_blocks)]
        store.insert(toks, blocks, _snap(snapshot_bytes), spill=False)
        boundaries.append(toks)

    # Every row finishes: the prompt-complete entry is published, which is the pressure the
    # boundary entry has to survive.
    for r in range(batch):
        toks = tuple(range(r * 100000, r * 100000 + prompt_blocks * BLOCK_TOKENS))
        blocks = [pool.alloc_block() for _ in range(prompt_blocks)]
        store.insert(toks, blocks, _snap(snapshot_bytes), spill=False)

    out = {"spilled": 0, "demoted": 0, "no-entry": 0}
    for toks in boundaries:
        out[_classify(store, toks)] += 1
    # Classify BEFORE looking up: a lookup promotes a demoted entry, which would rewrite
    # the cell it is meant to measure.
    out["matched"] = sum(1 for toks in boundaries if store.lookup(toks) is not None)
    out["entries_capacity"] = store._entries_capacity()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-bytes", type=int, default=144 << 20,
                    help="27B GDN snapshot; kv_cache.py:377")
    ap.add_argument("--prompt-blocks", type=int, default=8)
    args = ap.parse_args()

    # `budget_slots` is what the byte budget holds: build_engine gives the store a quarter
    # of free HBM, and its own comment prices the V100 at 9 resident snapshots.
    print(f"{'batch':>6} {'HBM':>5} {'DRAM':>5} {'ecap':>5} "
          f"{'spilled':>8} {'demoted':>8} {'no-entry':>9} {'spill':>7} {'match':>7}")
    for budget_slots, dram_slots in ((9, 0), (9, 8), (9, 64), (96, 0)):
        for batch in (2, 4, 8, 16, 32):
            r = run(batch, args.snapshot_bytes, budget_slots, dram_slots,
                    args.prompt_blocks)
            print(f"{batch:>6} {budget_slots:>5} {dram_slots:>5} {r['entries_capacity']:>5} "
                  f"{r['spilled']:>8} {r['demoted']:>8} {r['no-entry']:>9} "
                  f"{r['spilled'] / batch:>6.1%} {r['matched'] / batch:>6.1%}")

    # The closed form, over every slot count rather than the four this probe first sampled.
    for b in (2, 4, 8, 16, 32):
        for s in range(2 * b + 3):
            got = run(b, args.snapshot_bytes, s, 0, args.prompt_blocks)["spilled"]
            assert got == max(0, min(s - b, b)), (b, s, got)
    print("\nclosed form ok: spilled == clamp(slots - batch, 0, batch), "
          "batches 2..32 x every slot count 0..2*batch+2")

    # Two controls, because a probe that cannot fail measures nothing.
    starved = run(8, args.snapshot_bytes, 1, 0, args.prompt_blocks)
    assert (starved["spilled"], starved["no-entry"]) == (0, 8), starved
    # The threshold counts SNAPSHOTS, and a snapshot is constant size at any prefix length
    # (kv_cache.py:1434), so prompt length must not move it.
    for pb in (2, 4, 8, 16, 32):
        edge = run(8, args.snapshot_bytes, 16, 0, pb)
        assert edge["spilled"] == 8, (pb, edge)
    # The tier does not move `spilled` but does move `matched` -- the two yields differ.
    with_tier = run(16, args.snapshot_bytes, 9, 64, args.prompt_blocks)
    assert with_tier["spilled"] == 0 and with_tier["matched"] == 16, with_tier
    print("\ncontrols ok: a 1-slot budget spills nothing; the ramp does not move with "
          "prompt length (2..32 blocks, all 8/8 at slots == 2*batch); and the tier leaves "
          "spill at 0/16 while taking match to 16/16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
