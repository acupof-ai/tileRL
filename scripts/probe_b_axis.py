"""Is B a shape axis the way S and Mb are, and how many values does serving reach?

#73 bucketed the draft's prefill width (spec.py:405) and its block-table width
(spec.py:412) because `kernels_mma.py:22` bakes them into the kernel. That same
line reads `B, S, H, D = T.const("B, S, H, D")` -- **B is in the same tuple**, and
`params.pkl` showed the whole tuple is what varies per compile ([1, 37, 4, 256]).
Nothing rounds B: engine.py:666 sizes rows by `len(reqs)`, spec.py:412 by
`len(plan)`, and spec.py:484's chain loop by `len(live)`, which shrinks mid-chain
as rows hit block boundaries.

Counts the distinct (B, S) pairs that reach the two seq_q_lens kernels, by
wrapping the bound kernel rather than `Backend._kernel` -- the getter's cache key
holds factory args only (backend.py:249), so the per-shape dispatch is inside
`tilelang.jit` and invisible there.

**NEGATIVE RESULT, kept for it**: the CPU target cannot answer this. `backend.py:889`
falls back to the pool's torch loop when `write_tokens` is absent from the arch's
registry, and CPU dispatches `paged_attention`, never `paged_attention_split` -- so
neither B-baking kernel runs on the twin and the assert below fires. The axis was
measured instead by reading the pod's tilelang cache
(wins/2026-09-04-b-is-a-shape-axis-and-bucketing-it-is-rejected.md): 76 distinct
(B, S) pairs over 35 S values, and bucketing B is REJECTED because a padded batch row
costs 3.3x a useful one where a padded prefill row is discarded.

    uv run python scripts/probe_b_axis.py    # asserts: records nothing on cpu
"""

from __future__ import annotations

import collections
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.config import tiny  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.model import build_random  # noqa: E402

SHAPES: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
SEEN: set[str] = set()


def _spy(backend) -> None:
    """Record (B, S) per call of the two kernels that bake it."""
    inner = backend._kernel

    def outer(name, *a, **kw):
        SEEN.add(name)
        k = inner(name, *a, **kw)
        if name not in ("write_tokens", "paged_attention_split"):
            return k

        def wrapped(*args, **kwargs):
            q = args[0]  # both kernels take the [B, S, ...] tensor first
            SHAPES[name][tuple(q.shape[:2])] += 1
            return k(*args, **kwargs)

        return wrapped

    backend._kernel = outer


def main() -> None:
    cfg = tiny()
    backend = get_backend()
    _spy(backend)
    eng = build_engine(
        cfg, build_random(cfg, seed=0), backend,
        num_blocks=64, num_slots=8, max_batch=4, max_total_tokens=1024,
    )
    sp = SamplingParams(max_new_tokens=6, temperature=0.0)
    # Stagger the arrivals: a batch that forms and drains is what varies B.
    for n in (11, 23, 7, 19):
        eng.submit(list(range(2, 2 + n)), sp)
        eng.step()
    for _ in range(48):
        eng.poll()
        eng.step()

    for name, c in SHAPES.items():
        bs = sorted({b for b, _ in c})
        ss = sorted({s for _, s in c})
        print(f"\n{name}: {len(c)} distinct (B, S) pairs, {sum(c.values())} calls")
        print(f"  B values: {bs}")
        print(f"  S values: {ss}")
        for k, v in sorted(c.items()):
            print(f"    B={k[0]:>2} S={k[1]:>4}  x{v}")
    if not SHAPES:
        # The expected outcome on cpu, and the reason this script exists: naming the
        # kernels the target does NOT dispatch is what sent the measurement to the pod.
        called = sorted(SEEN)
        print("no (B, S) recorded: this target dispatches neither B-baking kernel.")
        print(f"  called instead: {called}")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
