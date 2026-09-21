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


def _build_engine():
    return build_engine(
        cfg=tiny(), model=build_random(tiny(), seed=11), backend=RefBackend(),
        num_blocks=4096, num_slots=4, max_batch=1, max_total_tokens=65536,
        max_num_batched_tokens=_CHUNK, sparse_k=_K, scorer="bounds",
        kv_cold_bytes=1 << 30, decode_graph=True)


def _run(e, decode_tokens, prompt=None, n_tok=None):
    if prompt is None:
        n_tok = _PAGES * BLOCK_TOKENS if n_tok is None else n_tok
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
    return prompt, saw_resident


def _publisher(decode_tokens: int, prefix_store=None, prompt_tokens: int | None = None):
    kw = {} if prefix_store is None else {"prefix_store": prefix_store}
    e = _build_engine(**kw)
    prompt, saw_resident = _run(e, decode_tokens, n_tok=prompt_tokens)
    return e, prompt, saw_resident


def _shared_bytes(cold) -> int:
    return sum(v[0] for v in cold._shared.values())


def _adopt_after(e, prompt) -> int:
    """Tokens a same-head follower adopts, measured on its first step."""
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    e.step()
    r = next((x for x in e._running if x.req_id == rid), None)
    return 0 if r is None else int(r.sparse_matched)


def _publisher_with_failing_finish_transfer(mode: str):
    """A short-decode publisher whose finish publish dies on its 3rd page:
    'raise' = a SpillWriteError out of the transfer; 'soft' = the transfer
    places no shared record (a key absent from share_keys). Armed on the
    finish close_prompt so natural decode offers are untouched."""
    from tilerl.kv_tiers import SpillWriteError

    e = _build_engine()
    sp = e._sparse
    real_xfer = sp.transfer_to_shared
    real_close = sp.prefix.close_prompt
    state = {"calls": 0}

    def arm_close(*a, **k):
        state["armed"] = True  # finish is this publisher's last publish
        return real_close(*a, **k)

    def fault_xfer(r, page, content_key, draft_block=None):
        if getattr(state, "armed", False) or state.get("armed"):
            state["calls"] += 1
            if state["calls"] == 3:
                if mode == "raise":
                    raise SpillWriteError("injected spill failure")
                return  # soft: no record placed
        return real_xfer(r, page, content_key, draft_block)

    sp.transfer_to_shared = fault_xfer
    sp.prefix.close_prompt = arm_close
    prompt, _saw = _run(e, _SHORT_DECODE)
    return e, prompt, state


def test_a_failed_finish_publish_leaves_no_dead_entry_and_no_dirty_follower():
    """P2-1 (#796): close_prompt attaches the index entry BEFORE the per-page
    transfers. A spill failure on page 3 must roll the whole close back:
    no lookup entry, no content key without a blob, and a same-head follower
    misses instead of adopting dead keys. The publisher itself succeeded, so it
    still completes normally — the abandoned publish is not a client error."""
    for mode in ("raise", "soft"):
        e, prompt, st = _publisher_with_failing_finish_transfer(mode)
        try:
            assert st["calls"] >= 3, f"{mode}: the injected fault never fired (vacuous gate)"
            pfx = e._sparse.prefix
            assert pfx.lookup(prompt) is None, f"{mode}: dead entry survived the failed publish"
            orphans = [k for en in pfx._by_id.values() for k in en["keys"]
                       if k not in e._kv.cold.share_keys()]
            assert not orphans, f"{mode}: entries name {len(orphans)} keys with no blob"
            assert _adopt_after(e, prompt) == 0, (
                f"{mode}: follower dirty-adopted from a half-published prefix")
        finally:
            e.shutdown()


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
        assert adopted == _PAGES * BLOCK_TOKENS, (
            f"a same-head follower adopted {adopted} tokens, expected the full "
            f"{_PAGES * BLOCK_TOKENS} closure (#796)")
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


def test_an_unaligned_prompt_closes_to_its_actual_end_with_zero_tail_recompute():
    """The real M6 prompt is 32028 tokens = 2001 pages + 12 tokens. The
    production prefill chunker cuts the unaligned tail back to a block
    boundary (engine.py chunk loop), so a snapshot exists at floor page 2001 =
    32016 tokens and only the 12-token tail is recomputed. finish must close
    through that naturally-present deepest snapshot -- no synthesized end
    snapshot from post-decode live state.

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


def test_a_row_that_adopted_does_not_republish_at_finish():
    """A follower that adopted a prefix must not re-close it at its own finish:
    re-publishing adds a redundant frozen entry (extra refs on the #793 capacity
    path) for zero new bytes. Only an origin publisher (sparse_matched==0)
    publishes at finish. The adopted blobs it already pins survive its drop via
    the request-pin release; the original publisher's entry stays the source."""
    e, prompt, _res = _publisher(_SHORT_DECODE)
    try:
        rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=8, seed=0))
        e.step()
        r = next(x for x in e._running if x.req_id == rid)
        assert r.sparse_matched > 0, "geometry failed to adopt"
        pfx = e._sparse.prefix
        n_entries_before = len(pfx._by_id)
        n_shared_bytes_before = _shared_bytes(e._kv.cold)
        for _ in range(40000):
            d = e.poll()
            if rid in d and len(d[rid]) >= 8:
                break
            e.step()
        assert len(pfx._by_id) == n_entries_before, "adopted row added a re-published entry"
        assert _shared_bytes(e._kv.cold) == n_shared_bytes_before, "adopted finish moved bytes"
    finally:
        e.shutdown()
