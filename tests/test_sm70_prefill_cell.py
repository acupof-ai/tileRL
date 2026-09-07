"""sm70 gates for the query-tiled prefill cell.

Skipped everywhere but a Volta card: the cell is registered on sm70 only, and on
the CPU target the same call resolves to the twin
(`kernels.make_paged_attention_prefill`), whose parity is gated in
`test_ops_parity.py`. These arms run in the acceptance window on the V100.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "auto")

import numpy as np
import pytest
from tilerl_kernels import backend as backend_mod
from tilerl_kernels.backend import get_backend

from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import build_random

pytestmark = pytest.mark.skipif(
    get_backend().arch != "sm70", reason="the tiled prefill cell is sm70-only"
)

_PROMPT = 64  # > _MAX_VERIFY_W, so the routing predicate sends it to the new cell


def _engine(seed: int = 1234):
    cfg = tiny()
    return build_engine(
        cfg,
        build_random(cfg, seed=seed),
        get_backend(),
        num_blocks=16,
        num_slots=4,
        max_batch=4,
        max_total_tokens=512,
    )


def _run(prompt, max_new_tokens: int = 8, expect: str | None = None) -> list[int]:
    """Run one prompt; if `expect` is given, assert that kernel actually ran.

    Without this the arms pass vacuously wherever the sm70 branch is not taken:
    on cpu both sides of the monkeypatch resolve to the same `paged_attention`,
    so a broken cell compares equal to itself. Measured -- three mutations of the
    twin left all four tests green until this assertion existed.
    """
    backend = get_backend()
    seen: list[str] = []
    real = backend._kernel

    def spy(name, **kw):
        seen.append(name)
        return real(name, **kw)

    engine = _engine()
    backend._kernel = spy
    try:
        rid = engine.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens))
        done: dict = {}
        for _ in range(512):
            done.update(engine.poll())
            if rid in done and len(done[rid]) >= max_new_tokens:
                return done[rid]
            engine.step()
        raise TimeoutError("engine did not finish")
    finally:
        backend._kernel = real
        engine.shutdown()
        if expect is not None:
            assert expect in seen, (
                f"{expect} never ran; the tick used {sorted(set(seen))}. This arm "
                "would have passed vacuously."
            )


def test_the_tiled_cell_and_split_agree_on_the_same_prompt(monkeypatch):
    """Greedy completions must not change when the prefill chunk switches kernels.

    Forcing `is_prefill_width` to False sends the same chunk down
    `paged_attention_split`, which is what shipped before this cell. **That arm
    exercises a configuration nothing ships** -- after the routing change no
    prefill-width chunk reaches split -- and it is deliberate: it isolates the
    kernel swap from everything else in the tick. The shipped path on the other
    side of the predicate is covered by the T=1 and T=8 arms below, which patch
    nothing.
    """
    prompt = np.random.default_rng(0).integers(3, 320, size=_PROMPT).astype(np.int64)
    assert backend_mod.is_prefill_width(_PROMPT), "the arm must straddle the predicate"

    tiled = _run(prompt, expect="paged_attention_prefill")
    monkeypatch.setattr(backend_mod, "is_prefill_width", lambda s: False)
    split = _run(prompt, expect="paged_attention_split")

    assert tiled == split, (
        f"the tiled cell and split disagree on a {_PROMPT}-token prompt: "
        f"{tiled} vs {split}. Greedy decoding makes this exact, not approximate."
    )


@pytest.mark.parametrize("width", [1, 8])
def test_the_split_path_is_untouched_at_decode_and_verify_widths(width):
    """No-regression on the shipped side of the predicate.

    s <= _MAX_VERIFY_W still routes to `paged_attention_split`; this cell must not
    have moved decode or a speculative verify. No patching: this is the served
    configuration.
    """
    assert not backend_mod.is_prefill_width(width), "these widths must stay on split"
    prompt = np.random.default_rng(1).integers(3, 320, size=width).astype(np.int64)
    out = _run(prompt, max_new_tokens=4, expect="paged_attention_split")
    assert len(out) == 4, f"width {width} produced {len(out)} tokens"


def test_the_cell_is_registered_and_takes_the_tile_as_a_parameter():
    """`kv_dtype` must reach the maker, or the fp16 rung is a second kernel."""
    import inspect

    from tilerl_kernels.registry import _SM70_KERNELS

    sig = inspect.signature(_SM70_KERNELS["paged_attention_prefill"])
    for p in ("block_M", "block_N", "kv_dtype"):
        assert p in sig.parameters, f"the maker must take {p} from the call site"
