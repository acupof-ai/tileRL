"""Does hit/miss alternate by session parity under a small snapshot budget?

The V100 tier cell read turn-0 hits on every other conversation: 3364-token prompts hit
at 5.6 s, 3344-token ones missed at 19.1 s, strictly alternating over 12 sessions. Two
candidate causes are confounded by construction -- `_fillers` assigns `_TOPICS[i % 2]`, and
the two topics differ in length, so LENGTH and PARITY alternate together and neither can be
read off the pattern.

This runs the real `PrefixStore` (not a simulation) over the interleave the bench produces:
12 sessions, each publishing its prefill chunks then its decode entry, into a budget of a
few snapshots. If an LRU rhythm alone reproduces alternation, length is the coincidence and
no card time is needed.

Arm L varies length with parity as the bench does; arm U gives every session the SAME
length. If both alternate, length is not the variable. If only L does, it is.

  PYTHONPATH=src:packages/tilerl-kernels/src TILERL_TARGET=cpu \
      python3 scripts/probe_session_parity_lru.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tilerl.kv_cache import BLOCK_TOKENS, PagedKvPool, PrefixStore

HEAD = 4 * BLOCK_TOKENS  # the shared system prefix every session opens with
SNAP = (torch.randn(3, 4, 8, 8), None)


def _store(entries: int) -> tuple[PagedKvPool, PrefixStore]:
    """A store whose state budget holds exactly `entries` snapshots."""
    pool = PagedKvPool(4096, 2, 8, device=torch.device("cpu"), layer_map=(0,))
    probe = PrefixStore(pool)
    probe.insert(list(range(HEAD)), [pool.alloc_block() for _ in range(HEAD // BLOCK_TOKENS)],
                 SNAP)
    per = probe.stats()["state_bytes"]
    probe.clear()
    return pool, PrefixStore(pool, state_bytes=per * entries)


def _publish(pool: PagedKvPool, store: PrefixStore, toks: list[int]) -> None:
    blocks = [pool.alloc_block() for _ in range(len(toks) // BLOCK_TOKENS)]
    store.insert(toks, blocks, SNAP)
    for b in blocks:
        pool.free_block(b)


def run(entries: int, sessions: int, chunks: int, vary_length: bool) -> list[bool]:
    """One turn 0 over `sessions` interleaved conversations; True where the head hit.

    Each session looks the head up (what the engine does before prefilling), then publishes
    `chunks` nested prefill boundaries plus one decode entry -- the publish pattern
    `_finish_prefills` and the decode path produce together.
    """
    pool, store = _store(entries)
    # The head is published once, by the fixture's first session, and shared by all.
    _publish(pool, store, list(range(HEAD)))
    hits = []
    for s in range(sessions):
        body = list(range(1000 + s * 100000, 1000 + s * 100000 + 40 * BLOCK_TOKENS))
        toks = list(range(HEAD)) + body
        hit = store.lookup(toks)
        hits.append(bool(hit) and hit.length >= HEAD)
        # odd sessions are one chunk shorter when length varies with parity, as _TOPICS does
        n = chunks - (s % 2 if vary_length else 0)
        for c in range(1, n + 1):
            _publish(pool, store, toks[: (HEAD // BLOCK_TOKENS + c) * BLOCK_TOKENS])
    return hits


def _fmt(hits: list[bool]) -> str:
    return "".join("H" if h else "." for h in hits)


def main() -> int:
    print(f"12 sessions, shared head {HEAD} tokens, real PrefixStore\n")
    print(f"{'budget':>7} {'chunks':>7} {'length varies':>14}  turn-0 head hits (H=hit)")
    alternating = []
    for entries in (4, 6, 11, 13):
        for chunks in (2, 6):
            for vary in (True, False):
                hits = run(entries, 12, chunks, vary)
                pat = _fmt(hits)
                # alternating = hits sit on one parity class and misses on the other
                even = {hits[i] for i in range(0, 12, 2)}
                odd = {hits[i] for i in range(1, 12, 2)}
                alt = len(even) == 1 and len(odd) == 1 and even != odd
                if alt:
                    alternating.append((entries, chunks, vary))
                print(f"{entries:>7} {chunks:>7} {str(vary):>14}  {pat}"
                      f"{'   <- ALTERNATES' if alt else ''}")
    print()
    if not alternating:
        print("No arm alternates by parity: an LRU rhythm over this publish pattern does not")
        print("reproduce the V100 reading, so the cause is elsewhere and needs card time.")
    else:
        varies = {v for _, _, v in alternating}
        if varies == {True}:
            print("Only the length-varying arms alternate: LENGTH is the variable, not parity.")
        elif varies == {False} or len(varies) == 2:
            print("Alternation appears with a FIXED length too, so it is the LRU rhythm and")
            print("length is a coincidence of _TOPICS[i % 2].")
    return 0


if __name__ == "__main__":
    # One runnable check: a budget of 13 with 12 sessions cannot evict the head at all, so
    # every session must hit. If this fails the harness is wrong, not the store.
    assert all(run(13, 12, 2, False)), "an unpressured budget must hit on every session"
    raise SystemExit(main())
