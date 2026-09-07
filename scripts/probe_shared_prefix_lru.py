"""Does a hit on the shared prefix protect it from the publisher's own tail?

`errors/2026-09-07-a-prompts-own-publishes-evict-its-shared-prefix.md` says an LRU over a
nested family evicts the shared head first, and prices a fix at "score by shareability".
But `kv_cache.py:1120` moves an entry to the MRU end on every matched lookup, so the head
is only the LRU victim while nothing is hitting it. That changes the fix's size: if one hit
protects it, the defect is a cold-start window, not a policy error.

Three arms, budget for 3 entries, one conversation publishing 8 boundaries:
  A  no hit on the head        -- the entry's own measurement, reproduced as the control
  B  one hit before the tail   -- does that single refresh survive 5 more publishes?
  C  a hit between publishes   -- the steady state a second session actually produces

  PYTHONPATH=src:packages/tilerl-kernels/src TILERL_TARGET=cpu python3 scripts/probe_shared_prefix_lru.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tilerl.kv_cache import BLOCK_TOKENS, PagedKvPool, PrefixStore

HEAD = 2 * BLOCK_TOKENS  # the shared system prompt, 32 tokens
SNAP = (torch.randn(3, 4, 8, 8), None)


def _store(entries: int) -> tuple[PagedKvPool, PrefixStore]:
    pool = PagedKvPool(256, 2, 8, device=torch.device("cpu"), layer_map=(0,))
    probe = PrefixStore(pool)
    probe.insert(list(range(HEAD)), [pool.alloc_block() for _ in range(HEAD // BLOCK_TOKENS)], SNAP)
    per = probe.stats()["state_bytes"]
    probe.clear()
    return pool, PrefixStore(pool, state_bytes=per * entries)


def _publish(pool: PagedKvPool, store: PrefixStore, toks: list[int]) -> None:
    blocks = [pool.alloc_block() for _ in range(len(toks) // BLOCK_TOKENS)]
    store.insert(toks, blocks, SNAP)
    for b in blocks:
        pool.free_block(b)


def run(hit_at: set[int], entries: int = 3, publishes: int = 8) -> tuple[bool, list[int]]:
    """Publish `publishes` nested boundaries, looking up the head after the ones in `hit_at`.

    Returns whether a second session sharing only the head can still hit it.
    """
    pool, store = _store(entries)
    convo = list(range(1000, 1000 + publishes * BLOCK_TOKENS))
    head = convo[:HEAD]
    for n in range(1, publishes + 1):
        _publish(pool, store, convo[: n * BLOCK_TOKENS])
        if n in hit_at:
            store.lookup(head)  # a second session arriving with only the shared header
    hit = store.lookup(head)
    lengths = sorted(len(e.tokens) for e in store._by_id.values())
    return hit is not None and hit.length >= HEAD, lengths


def main() -> int:
    arms = [
        ("A  no hit on the head", set()),
        ("B  one hit, right after the head is published", {2}),
        ("C  a hit after every publish", set(range(2, 9))),
    ]
    print(f"budget 3 entries, 8 nested publishes, shared head = {HEAD} tokens\n")
    print(f"{'arm':>46} {'head still hits':>16}  resident lengths")
    out = {}
    for label, hit_at in arms:
        hit, lengths = run(hit_at)
        out[label[0]] = hit
        print(f"{label:>46} {str(hit):>16}  {lengths}")

    # A must fail or the entry's own measurement does not reproduce and nothing below means
    # anything; C must pass or move_to_end is not reaching this path at all.
    assert not out["A"], "the control passed: the head survived with no hit, so this probe " \
                         "cannot see the eviction the entry measured"
    assert out["C"], "a hit after every publish still lost the head, so the MRU refresh at " \
                     "kv_cache.py:1120 does not protect a matched entry"
    print(f"\nA (control) evicts the head, C (refreshed every publish) keeps it, "
          f"B (one early hit) {'keeps' if out['B'] else 'loses'} it.")

    # A single hit anywhere cannot save it, and the reason is worth more than the sweep:
    # a lookup only refreshes an entry that is STILL RESIDENT. Once the head is gone the
    # lookup is a miss, and a miss restores nothing. So the quantity is not "how recent was
    # the last hit" but "how large is the gap BETWEEN hits" -- the head has to be re-touched
    # before the tail's next `budget` publishes push it out.
    print(f"\n{'hit every k publishes':>24} {'survives':>9}  resident lengths")
    worst = None
    for k in range(1, 8):
        hit, lengths = run(set(range(2, 9, k)))
        print(f"{k:>24} {str(hit):>9}  {lengths}")
        if hit:
            worst = k
    assert worst is not None, "no hit interval kept the head, so the MRU refresh at " \
                              "kv_cache.py:1120 never protects it and this is a pure policy bug"
    single, _ = run({8})
    assert not single, "a single hit after the last publish kept the head, so the head was " \
                       "still resident then and the eviction order is not what the entry says"
    print(f"\nThe head survives iff it is re-hit at least every {worst} publishes, at a budget "
          f"of 3 entries.\nA single hit at ANY position fails -- including after the last "
          f"publish -- because a lookup refreshes\nonly a resident entry, and a miss restores "
          f"nothing.")

    # k=2 at budget 3 is one point, and a single point extrapolates to nothing: the V100's
    # budget is 11. Sweep the budget to find whether the tolerated interval tracks it, so
    # the number quoted for the live card is read off a relation rather than assumed flat.
    print(f"\n{'budget':>8} {'max interval that survives':>28}")
    rel = []
    for entries in (2, 3, 4, 6, 8, 11):
        ok = [k for k in range(1, 20) if run(set(range(2, 21, k)), entries, 20)[0]]
        rel.append((entries, max(ok) if ok else 0))
        print(f"{entries:>8} {max(ok) if ok else 0:>28}")
    assert all(k > 0 for _, k in rel), f"some budget tolerated no interval at all: {rel}"
    assert rel[-1][1] > rel[0][1], (
        f"the tolerated interval did not grow with the budget ({rel}), so it cannot be read "
        "off the budget and the V100's 11 needs its own measurement"
    )
    # The six points are exactly budget-1, so say so and let it fail rather than eyeballing a
    # pattern: an unasserted regularity is a coincidence until the next budget breaks it.
    off = [(b, k) for b, k in rel if k != b - 1]
    assert not off, f"the interval is not budget-1 at {off}; the relation quoted below is wrong"
    slack = rel[-1][1]
    print(f"\nThe tolerated interval is exactly budget-1 at all six budgets, so at the V100's 11 "
          f"entries a shared\nhead survives a gap of {slack} publishes. One conversation emits 70 "
          f"at gen 1024, so a second session\nmust arrive inside every {slack} of them -- about "
          f"{70 // slack} arrivals per conversation. That is the operand:\nnot the sharing RATE "
          f"but the sharing INTERVAL against the publish rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
