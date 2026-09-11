"""Unit C: boot-from-SSD KV store (KvBootStore, --kv-store).

A fully prefilled context saved once is bulk-reloaded into fresh blocks on a
later engine whose request prefix matches, skipping the prefill. The load goes
through the same cold-dtype widening a host promote uses, restores the recurrent
GDN snapshot, and rejects a corrupted page by checksum.
"""

from __future__ import annotations

import torch

from tilerl.kv_cache import BLOCK_TOKENS, KvBootStore, NoPrefixStore
from tilerl.testing import RefBackend


def _drain(engine, rid, n):
    for _ in range(512):
        done = engine.poll()
        if rid in done and len(done[rid]) >= n:
            return done[rid][:n]
        engine.step()
    raise TimeoutError("engine did not finish in 512 ticks")


def _engine(cfg, tmp, store: bool):
    from tilerl.engine import build_engine
    from tilerl.model import build_random

    return build_engine(
        cfg, build_random(cfg, seed=11), RefBackend(), num_blocks=64,
        num_slots=4, max_batch=1, max_total_tokens=2048,
        prefix_store=NoPrefixStore(),
        **({"kv_store": str(tmp)} if store else {}))


def _decode_req(engine, rid):
    req = None
    for _ in range(64):
        for r in engine._running:
            if r.req_id == rid and r.phase == 2:
                req = r
        if req is not None:
            return req
        engine.step()
    raise AssertionError("request never reached decode")


def test_saved_context_boots_a_fresh_engine_and_continues_identically(tmp_path):
    """write -> cold start -> continuation: the booted engine must emit the same
    greedy tokens as an in-process dense continuation, and counts a boot hit rather
    than a prefill."""
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams

    cfg = tiny()
    import numpy as np

    prompt = np.arange(7, 7 + 5 * BLOCK_TOKENS + 3, dtype=np.int64)  # 6 whole pages
    params = SamplingParams(temperature=0.0, max_new_tokens=6, seed=0)

    writer = _engine(cfg, tmp_path, store=True)
    rid = writer.submit(prompt, params)
    req = _decode_req(writer, rid)
    written = writer.save_boot(req)
    assert written > 0
    tok_inproc = _drain(writer, rid, 6)
    writer_prefills = writer.stats()["prefill_forwards"]
    writer.shutdown()

    # A fresh engine (empty HBM, empty prefix store) boots the saved prefix.
    booter = _engine(cfg, tmp_path, store=True)
    rid2 = booter.submit(prompt, params)
    tok_boot = _drain(booter, rid2, 6)
    st = booter.stats()
    assert st["boot_hits"] == 1, st
    # The boot covers the 5 whole pages; only the <16-token tail forwards once,
    # versus the writer's multi-forward full prefill.
    assert st["prefill_forwards"] == 1, st
    assert writer_prefills > 1
    assert tok_boot == tok_inproc, f"booted continuation {tok_boot} != in-process {tok_inproc}"
    booter.shutdown()


def test_store_byte_count_matches_the_priced_cold_rows(tmp_path):
    """The on-disk K/V+scale bytes equal the ledger's per-cold-page price x pages."""
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams
    from tilerl.memory import per_cold_kv_block_bytes

    cfg = tiny()
    import numpy as np

    prompt = np.arange(3, 3 + 4 * BLOCK_TOKENS, dtype=np.int64)  # exactly 4 pages
    e = _engine(cfg, tmp_path, store=True)
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=2, seed=0))
    req = _decode_req(e, rid)
    e.save_boot(req)
    entry = next(p for p in (tmp_path / "tilerl_kvboot").iterdir() if p.is_dir())
    kv_bytes = sum((entry / f).stat().st_size for f in ("k.bin", "v.bin"))
    # CPU tiny pool is f32; the auto cold rule narrows an f32 pool to f16 (the sm70
    # rule the CPU cell mirrors), so the stored width is half the f32 pool's.
    expect = 4 * per_cold_kv_block_bytes(cfg, torch.float32, None, torch.float16)
    assert kv_bytes == expect, (kv_bytes, expect)
    # the plan/health ledger carries the store as one tier=ssd kv_cold allocation,
    # derived == measured bytes.
    row = [r for r in e.stats()["memory"]
           if r["owner"] == "kv_cold" and r["tier"] == "ssd"]
    assert len(row) == 1 and row[0]["derived"] == row[0]["measured"]
    assert row[0]["derived"] >= kv_bytes, row[0]
    e.shutdown()


def test_a_corrupted_page_file_fails_loudly(tmp_path):
    """One flipped byte in k.bin must fail the page checksum, not load plausible KV."""
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams

    cfg = tiny()
    import numpy as np

    prompt = np.arange(3, 3 + 3 * BLOCK_TOKENS, dtype=np.int64)
    e = _engine(cfg, tmp_path, store=True)
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    e.save_boot(_decode_req(e, rid))
    entry = next(p for p in (tmp_path / "tilerl_kvboot").iterdir() if p.is_dir())
    kf = entry / "k.bin"
    b = bytearray(kf.read_bytes())
    b[0] ^= 0xFF  # page 0's K
    kf.write_bytes(b)

    # direct store load raises (checksum), independent of the engine's admit handling
    store = KvBootStore(str(tmp_path), e._boot._fingerprint)
    fresh = _engine(cfg, tmp_path / "other", store=False)
    try:
        store.boot(list(prompt[: 3 * BLOCK_TOKENS]), fresh._kv)
        raise AssertionError("a corrupted page loaded without error")
    except RuntimeError as exc:
        assert "checksum" in str(exc) or "corrupt" in str(exc)
    finally:
        e.shutdown(); fresh.shutdown()


def test_dense_bulk_boot_is_refused_with_sparse_k(tmp_path):
    """A bulk boot allocates EVERY context block against the device pool up front;
    the sparse hot pool holds only k+window+chunk per slot, so the combination must
    raise at build time rather than admit a row that cannot fit."""
    import pytest

    from tilerl.config import tiny
    from tilerl.engine import build_engine
    from tilerl.model import build_random

    cfg = tiny()
    with pytest.raises(NotImplementedError, match="sparse_k"):
        build_engine(
            cfg, build_random(cfg, seed=11), RefBackend(), num_blocks=64,
            num_slots=4, max_batch=1, max_total_tokens=2048,
            prefix_store=NoPrefixStore(), sparse_k=2, kv_cold_bytes=1 << 30,
            kv_store=str(tmp_path))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
