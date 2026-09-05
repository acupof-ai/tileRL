"""TP-2 training gate: sharded gradients must equal the unsharded ones.

Runs two gloo ranks on the CPU target, so it gates on this machine and in CI.
The failure this exists for is silent: a missing backward collective leaves the
loss correct and every gradient short by a factor of ``world``, and the loss
recovers by the next step.

    TILERL_TARGET=cpu python3 tests/tp_world2.py            # the gate
    TILERL_TARGET=cpu python3 tests/tp_world2.py --no-fork  # the negative control
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]


def _run(rank: int, world: int, no_fork: bool, tie: bool, out: dict) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ["TILERL_TARGET"] = "cpu"

    from dataclasses import replace

    from tilerl import model as model_mod
    from tilerl.autograd import RecordingBackend, Tape
    from tilerl.config import tiny
    from tilerl.tensor_parallel import shard_params, tp_config
    from tilerl.testing import RefBackend
    from tilerl.train import _training_kv

    if no_fork:  # the control: delete the backward collective, keep everything else
        import tilerl.model as m

        m._tp_fork = lambda backend, x: x

    # tiny() ties the head, so tie=False is the ONLY arm that runs the
    # vocab-parallel lm_head -- the branch the 27B takes.
    cfg = tiny() if tie else replace(tiny(), tie_word_embeddings=False)
    full = model_mod.build_random(cfg, seed=0, keep_master=True)
    backend = RefBackend()

    if world > 1:
        backend.init_tp(world, rank)
        params = shard_params(full.params, cfg, rank, world)
        local = model_mod.Model(tp_config(cfg, world), params)
    else:
        local = full

    ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    tape = Tape()
    with torch.no_grad(), tape:
        logits = local.forward(ids, np.arange(4, dtype=np.int64),
                               _training_kv(local, 1, 4, device=backend.device),
                               RecordingBackend(backend))
    grads = tape.backward(torch.ones_like(logits))
    # Every param by NAME, replicated and sharded alike. Comparing only the
    # replicated ones would pass with every sharded gradient wrong; the caller
    # shards the world=1 gradients through the same rule to compare the rest.
    # A torch tensor through a Manager dict deadlocks (measured: identical probe
    # returns in 3 s with .tolist(), hangs past 60 s with a tensor). Ship lists.
    out[rank] = {k: (g.float().tolist(), tuple(g.shape))
                 for k, v in local.params.items()
                 if (g := grads.get(id(v))) is not None}


def _one(no_fork: bool, tie: bool) -> tuple[bool, int]:
    from dataclasses import replace

    from tilerl.config import tiny
    from tilerl.tensor_parallel import shard_params

    mgr = mp.Manager()

    one: dict = {}
    _run(0, 1, no_fork, tie, one)
    ref = {k: torch.tensor(v).reshape(s) for k, (v, s) in one[0].items()}
    assert ref, "world=1 produced no gradients at all"

    two = mgr.dict()
    mp.spawn(_run, args=(2, no_fork, tie, two), nprocs=2, join=True)

    cfg = tiny() if tie else replace(tiny(), tie_word_embeddings=False)
    ok, compared = True, 0
    for r in (0, 1):
        got = {k: torch.tensor(v).reshape(s) for k, (v, s) in two[r].items()}
        assert got, f"rank {r} produced no gradients at all"
        # The same rule that shards the weights shards their gradients, so a
        # sharded weight is comparable too: rank r's gradient must equal slice r
        # of the unsharded one.
        want = shard_params({k: v for k, v in ref.items() if k in got}, cfg, r, 2)
        missing = sorted(set(ref) - set(got))
        if missing:
            print(f"rank {r}: {len(missing)} params lost their gradient: {missing[:4]}")
            ok = False
        for k in sorted(got):
            w = want.get(k)
            if w is None or w.shape != got[k].shape:
                print(f"rank {r}: {k} shape {tuple(got[k].shape)} vs expected "
                      f"{None if w is None else tuple(w.shape)}")
                ok = False
                continue
            compared += 1
            # bf16 activations: the two orders of summation differ in the last
            # bits, measured max|d| 1.4e-4 on gradients of order 1e2. A missing
            # collective is a factor of `world`, so this tolerance still catches it
            # by four orders of magnitude -- the --no-fork control proves that.
            if not torch.allclose(w.float(), got[k].float(), rtol=2e-3, atol=1e-3):
                ratio = (got[k].float().abs().sum() / w.float().abs().sum().clamp_min(1e-12)).item()
                print(f"rank {r}: {k} MISMATCH max|d|={(got[k] - w).abs().max().item():.3e} "
                      f"sum-ratio={ratio:.4f}")
                ok = False

    return ok, compared


def main() -> int:
    no_fork = "--no-fork" in sys.argv
    rc = 0
    for tie in (True, False):
        head = "tied head" if tie else "vocab-parallel head"
        ok, compared = _one(no_fork, tie)
        print(f"[{head}] compared {compared} gradient tensors across 2 ranks: "
              + ("match" if ok else "DIFFER"))
        if no_fork:
            # The control must FAIL on both arms. A control that passes is not one.
            if ok:
                print(f"[{head}] negative control PASSED -- vacuous gate")
                rc = 1
        elif not ok:
            rc = 1
    if no_fork and rc == 0:
        print("negative control: correctly FAILED on both head layouts")
    elif rc == 0:
        print("TP-2 gradients match TP-1 on both head layouts")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
