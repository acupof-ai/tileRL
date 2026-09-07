"""Why does a recovered SSD entry never get read? Print the key and the hashes.

The card-1 restart bench came back INVALID: the faulted arm recovered 1 entry and
took 0 SSD hits, 0 prefetches, 0 tick loads. The write side works (321 MiB spilled),
so the miss is on the read side and upstream of the break-even -- n* was 0, which
admits every length.

Two candidates, and the point of this script is to tell them apart rather than guess:

  (a) the recovered key matches no hash the turn-2 request computes;
  (b) the probe never runs at all -- `submit` wraps it in `contextlib.suppress`,
      so anything raised inside `prefetch_if_worth_it` is silent.

Runs the real classes on the real tier directory, no server: this is about hashing
and dict lookups, not about the model.

    scripts/pod_run.sh ssdprobe 1 -- /work/tl013/bin/python -u scripts/probe_ssd_read_miss.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tilerl.kv_cache import BLOCK_TOKENS, KvTier, PagedKvPool, PrefixStore  # noqa: E402

SPILL = os.environ.get("PROBE_SPILL", "/work/ssd_probe_tier")
FP = "probe-fingerprint-v1"


def _pool() -> PagedKvPool:
    # Tiny but real: the tier stores whatever the pool holds, and this asks about keys.
    return PagedKvPool(64, 2, 8, num_layers=1, device=torch.device("cuda"),
                       dtype=torch.bfloat16)


def _state(pool):
    return (torch.zeros(1, 2, 8, 8, device=pool.device),
            torch.zeros(1, 2, 4, 8, device=pool.device))


def _publish(store, pool, tokens):
    """Insert one block-aligned prefix the way the engine does, with spill on."""
    n = len(tokens) // BLOCK_TOKENS
    blocks = [pool.alloc_block() for _ in range(n)]
    ok = store.insert(tuple(tokens), blocks, _state(pool), spill=True)
    for b in blocks:
        pool.free_block(b)
    return ok


def main() -> int:
    turn1 = list(range(1000, 1000 + 8 * BLOCK_TOKENS))     # 128 tokens
    turn2 = turn1 + list(range(2000, 2000 + 4 * BLOCK_TOKENS))  # turn1 + 64 more

    # --- write side: publish turn 1 and let the tier spill it -------------------
    pool = _pool()
    tier = KvTier(SPILL, FP, min_tokens=BLOCK_TOKENS)
    store = PrefixStore(pool, ssd=tier)
    print(f"published turn1: {_publish(store, pool, turn1)}")
    for _ in range(200):
        if tier.stats()["ssd_pending"] == 0:
            break
        import time
        time.sleep(0.05)
    print(f"after spill: {tier.stats()['ssd_entries']} entries on disk")
    key_written = store._hash_all(tuple(turn1))
    print(f"key WRITTEN for turn1  = {key_written:016x}")

    # --- restart: a fresh tier over the same directory ---------------------------
    pool2 = _pool()
    tier2 = KvTier(SPILL, FP, min_tokens=BLOCK_TOKENS)
    store2 = PrefixStore(pool2, ssd=tier2)
    recovered = sorted(tier2._lru.keys())
    print(f"recovered {len(recovered)} key(s): {[f'{k:016x}' for k in recovered]}")

    # --- the probe's view: what does prefetch_if_worth_it actually ask for? ------
    n_star = store2.break_even_tokens(1533.3)
    print(f"break_even_tokens(1533.3) = {n_star}  (len(turn2)={len(turn2)})")

    h, hashes = 0, []
    for t in turn2:
        h = store2._roll(h, int(t))
        hashes.append(h)
    asked = []
    for i in range(len(turn2) - len(turn2) % BLOCK_TOKENS, 0, -BLOCK_TOKENS):
        asked.append((i, hashes[i - 1]))
    print(f"probe would ask {len(asked)} lengths, longest first: "
          f"{[(i, f'{k:016x}') for i, k in asked[:4]]} ...")

    hit = [(i, k) for i, k in asked if k in tier2._lru]
    print(f"MATCH against recovered index: {[(i, f'{k:016x}') for i, k in hit]}")

    # --- and does the real call fire? -------------------------------------------
    before = tier2.stats()["ssd_prefetches"]
    try:
        fired = store2.prefetch_if_worth_it(turn2, 1533.3)
        err = None
    except Exception as e:  # noqa: BLE001 - the suppressed one; print it
        fired, err = None, f"{type(e).__name__}: {e}"
    after = tier2.stats()["ssd_prefetches"]
    print(f"prefetch_if_worth_it -> {fired}  raised={err}  "
          f"prefetches {before} -> {after}")

    # --- the SERVER's ordering: submit prefetches, the very next tick looks up, so
    # --- lookup sees the fetch still in flight and declines it (kv_cache.py:1127).
    immediate = store2.lookup(turn2)
    print(f"lookup WITHOUT waiting -> {None if immediate is None else immediate.length}"
          f"  fetch_waits={store2.fetch_waits}  ssd_hits={store2.stats()['ssd_hits']}")

    # --- and does lookup serve it once the read lands? ---------------------------
    for _ in range(200):
        if not tier2.fetch_pending(key_written):
            break
        import time
        time.sleep(0.05)
    got = store2.lookup(turn2)
    st = store2.stats()
    print(f"lookup(turn2) -> {None if got is None else got.length} tokens; "
          f"ssd_hits={st['ssd_hits']} tick_loads={st.get('ssd_tick_loads')}")

    print("\nVERDICT:")
    if not recovered:
        print("  nothing recovered -- the write side, not the read side")
    elif not hit:
        print("  (a) the recovered key matches NO length the probe asks for")
        print(f"      written {key_written:016x}, recovered {[f'{k:016x}' for k in recovered]}")
    elif err:
        print(f"  (b) the probe RAISED and submit would swallow it: {err}")
    elif after == before:
        print("  (c) the probe ran, the key matched, and it still queued nothing")
    else:
        print("  the probe fires here -- the miss is in the server path, not this one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
