"""Score a prefix-eviction policy by reuse in tokens, over shared-head x budget.

Six sessions of 2048 tokens, two turns, driven through the ENGINE so the publishes come from
its own chunk boundaries -- a hand-written `insert` cannot reproduce the defect this measures
(errors/2026-09-07-a-miss-self-reinforces.md).

The score is read at the FIRST lookup of each request. Reading it after the request finishes
is vacuous: the row's own publishes are in the store by then, so every policy scores a hit the
row just created for itself, and pure LRU appears to beat the fix.

Blind spot: one model, one prompt shape, tokens-not-seconds. It ranks policies; the wall-clock
number has to come from a card. Results in wins/2026-09-08-evict-by-length-times-sharers.md.
"""

import sys

sys.path.insert(0, "src")
from tilerl_kernels.backend import get_backend

from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import build_random


def run(budget_snapshots, sessions=6, plen=2048, shared=0):
    cfg = tiny()
    eng = build_engine(cfg, build_random(cfg, seed=15), get_backend(), num_blocks=4096,
                       num_slots=8, max_batch=1, max_total_tokens=32768)
    real = eng._prefix.lookup
    seen = []
    def spy(toks):
        m = real(toks)
        seen.append(0 if m is None else m.length)
        return m
    eng._prefix.lookup = spy
    head = [5] * shared
    turn2 = []
    for turn in (1, 2):
        for i in range(sessions):
            toks = head + [100 + i] * (plen - shared) + ([7] * 20 if turn == 2 else [])
            seen.clear()
            rid = eng.submit(toks, SamplingParams(max_new_tokens=4, temperature=0.0))
            t = 0
            while rid not in eng.poll() and t < 900:
                eng.step(); t += 1
                one = eng._prefix._snapshot_bytes
                if one:
                    eng._prefix.state_bytes = budget_snapshots * one
            if turn == 2:
                turn2.append(seen[0] if seen else -1)
    st = eng._prefix.stats()
    print(f"shared {shared} budget {budget_snapshots}: turn2 reuse {turn2} "
          f"sum {sum(turn2)} entries {st['entries']} evict {st['evictions']}")

for shared in (0, 512, 1024, 1536):
    for b in (4, 6, 8, 12):
        run(b, shared=shared)
