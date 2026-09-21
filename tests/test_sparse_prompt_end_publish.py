"""D796-GATE: a same-head follower cannot adopt a prompt whose pages never leave
the resident union.

Device evidence (#796): 45328 demotions, 45328 `offer_drop` calls, and then
`keys_nonempty_calls=0`, `keys_total=0`, `xfer_calls=0`, `peak_shared_pages=0`.
The publish chain stops *inside* `publish_dropped`, which returns `{}` on every
call -- the transfer branches are never entered, so this is NOT a blob-loss bug.

Mechanism (read, not guessed): `publish_dropped` closes the frontier only to a
length `m` with `{0..m-1} ⊆ pend` (pages that LEFT the union) AND `m ∈ snaps`
(an aligned prefill-chunk boundary snapshot, capped at `{lowest, newest}`). When
decode is short relative to the context, the low pages are still resident when
the request ends, so the contiguous offered prefix never reaches the lowest
surviving snapshot and `m` walks back to `old_len` -> `{}`.

This gate drives that shape on CPU with no stub: k=128 over a 2002-page prompt
keeps the early pages inside the selection+window union, and a short decode ends
the request before they are offered. Measured deterministic 3/3.

RED today (d60a679c): `published=0`, no entry, no shared blob, follower miss.
It goes GREEN only when a prompt-end remedial closure names the still-resident
pages -- the fix this issue is about. Both red lines must hold in one run:
adoption works AND the close tick still moves zero bytes.
"""

from __future__ import annotations

import numpy as np

from tilerl.build import build_engine
from tilerl.config import tiny
from tilerl.engine import SamplingParams
from tilerl.kv_cache import BLOCK_TOKENS
from tilerl.model import build_random
from tilerl.testing import RefBackend

#: The device shape: k=128 (the serve default), a 32k-class prompt, and a
#: 2048-token prefill chunk (=128 pages, so the lowest snapshot sits at m=128).
_PAGES = 2002
_K = 128
_CHUNK = 2048
#: Generated tokens for the publisher. SHORT on purpose: this is the whole
#: point -- decode does not run long enough to offer the low pages.
_SHORT_DECODE = 8
#: ... and the same publisher decoding well past one chunk, which DOES close the
#: frontier naturally. The pair is the discriminator: it is decode length, not
#: the engine shape, that decides whether the natural chain fires at all.
_LONG_DECODE = 32


def _publisher(decode_tokens: int, prefix_store=None, prompt_tokens: int | None = None):
    kw = {} if prefix_store is None else {"prefix_store": prefix_store}
    e = build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=4096, num_slots=4, max_batch=1, max_total_tokens=65536,
        max_num_batched_tokens=_CHUNK, sparse_k=_K, scorer="bounds",
        kv_cold_bytes=1 << 30, decode_graph=True, **kw)
    n_tok = _PAGES * BLOCK_TOKENS if prompt_tokens is None else prompt_tokens
    prompt = (np.arange(n_tok, dtype=np.int64) % 300) + 7
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=decode_tokens, seed=0))
    saw_resident = False
    for _ in range(40000):
        d = e.poll()
        if rid in d and len(d[rid]) >= decode_tokens:
            break
        e.step()
        if e._sparse.tracker.resident.get(rid):
            saw_resident = True
    return e, prompt, saw_resident


def _shared_bytes(cold) -> int:
    return sum(v[0] for v in cold._shared.values())


def _adopt_after(e, prompt) -> int:
    """Tokens a same-head follower adopts, measured on its first step."""
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    e.step()
    r = next((x for x in e._running if x.req_id == rid), None)
    return 0 if r is None else int(r.sparse_matched)


def test_a_short_decode_prompt_is_adoptable_by_a_same_head_follower():
    """THE GATE. A publisher whose prefill completed and which decoded briefly
    must still leave its prompt adoptable by an immediate same-head follower.

    RED today. The natural chain is healthy but structurally cannot fire here:
    with k=128 the low prompt pages stay inside the selection+window union, so
    the contiguous offered prefix never reaches the lowest surviving chunk
    snapshot and `publish_dropped` returns {} forever (#796: 45328 offer_drop
    calls, keys_total=0, no transfer branch entered). Fixkv's prompt-end
    remedial closure is what turns this green; it must do so WITHOUT moving
    bytes on the request-close tick (guarded separately below).

    Prefill completion and residency are asserted first, so this cannot be
    satisfied by a build that merely never got as far as publishing.
    """
    e, prompt, saw_resident = _publisher(_SHORT_DECODE)
    try:
        pfx = e._sparse.prefix
        assert pfx is not None, "this gate needs a real sparse prefix cache"
        assert saw_resident, "no residency was ever tracked; the geometry is wrong"

        # --- the contract under test -------------------------------------
        assert _shared_bytes(e._kv.cold) > 0, (
            "no shared blob bytes landed for a completed, briefly-decoded prompt: "
            "an immediate same-head follower has nothing to adopt (#796)")
        entry = pfx.lookup(prompt)
        assert entry is not None, (
            "the prompt is not findable in the sparse prefix index after a short "
            "decode; the prompt-end frontier never closed (#796)")
        adopted = _adopt_after(e, prompt)
        assert adopted > 0, (
            "a same-head follower adopted 0 tokens from a completed prompt whose "
            "pages never left the union (#796)")
    finally:
        e.shutdown()


def test_the_same_publisher_decoding_past_a_chunk_does_publish_and_adopt():
    """Control for the assertion above: the ONLY thing changed is decode length.

    Without this, the first test would also pass on a build where the whole
    publish path was broken -- it could not tell 'natural chain correctly found
    nothing' from 'publish never works'. Here it must work, end to end.
    """
    e, prompt, _res = _publisher(_LONG_DECODE)
    try:
        sp = e._sparse
        pfx = sp.prefix
        assert pfx.published > 0, (
            "decoding past one chunk did not close the frontier; the natural chain "
            "is broken, so the short-decode gate above proves nothing")
        assert len(pfx._by_id) > 0, "published count moved with no entry"
        assert _shared_bytes(e._kv.cold) > 0, "entries exist but no shared blob bytes landed"
        entry = pfx.lookup(prompt)
        assert entry is not None, "a published prefix is not findable by its own prompt"
        assert _adopt_after(e, prompt) > 0, "follower did not adopt a published prefix"
    finally:
        e.shutdown()


def test_the_close_path_still_carries_no_forced_publish():
    """The other red line (M3, `wins/2026-09-21-close-zero-bytes`): a request end
    must not force a frontier closure. Asserted structurally rather than by a
    counter, because the counter that matters -- close-tick `pub_*` / `ssd_mmap`
    -- only exists under TILERL_STEP_TIMING, and the thing being guarded is that
    the call is gone from the release path at all.

    `SparsePrefixCache.close_request` is the deleted forced-closure entry point;
    a remediation fix must not reintroduce a call to it on the close path. The
    prompt-end trigger belongs off the request-close tick.
    """
    import inspect

    from tilerl import sparse_engine
    from tilerl.engine import Engine

    assert not hasattr(sparse_engine.SparsePrefixCache, "close_request"), (
        "close_request is back on SparsePrefixCache -- the M3 close-time forced "
        "closure this issue is re-scoping around")
    src = inspect.getsource(Engine._release)
    assert "close_request" not in src, "the request-release path calls close_request again"


def test_an_unaligned_prompt_closes_to_its_actual_end_with_zero_tail_recompute():
    """b'2 + perf2 F1: the real M6 prompt is 32028 tokens = 2001 pages + 12
    tokens, so no note_boundary snapshot exists at the prompt end. Without a
    synthesized end snapshot the closure stops at m=1920 and the follower
    recomputes the 81-page / 1308-token (4.1%) tail. finish must add the
    prompt-end snapshot from live recurrent state and close through it.

    Asserts the EXACT adopted length (not >0) and that shared BYTES land
    (a partial/skipped dead entry would also satisfy a >0 entry check)."""
    n_tok = _PAGES * BLOCK_TOKENS + 12  # 32044 here; same shape class as 32028
    prompt_pages = n_tok // BLOCK_TOKENS
    e, prompt, _res = _publisher(_SHORT_DECODE, prompt_tokens=n_tok)
    try:
        pfx = e._sparse.prefix
        entry = pfx.lookup(prompt)
        assert entry is not None, "unaligned prompt end did not close (#796 F1)"
        assert len(entry["keys"]) == prompt_pages, (
            f"closed to {len(entry['keys'])} pages, expected the full "
            f"{prompt_pages} (tail would be recomputed)")
        assert _shared_bytes(e._kv.cold) > 0, "unaligned closure landed no blob bytes"
        adopted = _adopt_after(e, prompt)
        assert adopted == prompt_pages * BLOCK_TOKENS, (
            f"follower adopted {adopted} tokens, expected {prompt_pages * BLOCK_TOKENS}")
    finally:
        e.shutdown()
