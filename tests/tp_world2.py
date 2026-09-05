"""TP-2 training gate: sharded gradients must equal the unsharded ones.

Runs two gloo ranks on the CPU target, so it gates on this machine and in CI.
The failure this exists for is silent: a missing backward collective leaves the
loss correct and every gradient short by a factor of ``world``, and the loss
recovers by the next step.

    TILERL_TARGET=cpu python3 tests/tp_world2.py            # the gate
    TILERL_TARGET=cpu python3 tests/tp_world2.py --no-fork     # control: no backward collective
    TILERL_TARGET=cpu python3 tests/tp_world2.py --local-clip  # control: clip on the local shard
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


def _step_run(rank: int, world: int, local_clip: bool, out: dict) -> None:
    """One real ``train_step`` (forward, backward, clip, AdamW) and the weights
    it produced. The gradient arms above stop before the update, so nothing there
    sees the clip norm -- which is global, and was this shard's own.

    The model comes from ``cli._shard``, the path ``--tp`` takes, so this gates
    the selection too: a shard built only here would leave the CLI untested."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    os.environ["TILERL_TARGET"] = "cpu"
    os.environ["WORLD_SIZE"], os.environ["RANK"] = str(world), str(rank)

    from tilerl import model as model_mod
    from tilerl.autograd import AdamW
    from tilerl.cli import _shard
    from tilerl.config import tiny
    from tilerl.testing import RefBackend
    from tilerl.train import train_step

    if local_clip:  # the control: clip on this rank's shard, the pre-fix behaviour
        import tilerl.train as t

        real = t.clip_grad_norm
        t.clip_grad_norm = lambda g, m, sharded=None, backend=None: real(g, m)

    cfg = tiny()
    full = model_mod.build_random(cfg, seed=0, keep_master=True)
    backend = RefBackend()
    _, local = _shard(cfg, full, world, backend, model_mod)

    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    loss = train_step(local, ids, backend, AdamW(lr=1e-2))
    out[rank] = (loss, {k: (v.float().tolist(), tuple(v.shape))
                        for k, v in local.params.items()})


def _one_step(local_clip: bool) -> tuple[bool, int, float, float]:
    from tilerl.config import tiny
    from tilerl.tensor_parallel import shard_params

    one: dict = {}
    _step_run(0, 1, local_clip, one)
    loss1, w1 = one[0]
    ref = {k: torch.tensor(v).reshape(s) for k, (v, s) in w1.items()}

    mgr = mp.Manager()
    two = mgr.dict()
    mp.spawn(_step_run, args=(2, local_clip, two), nprocs=2, join=True)

    cfg = tiny()
    ok, compared = True, 0
    for r in (0, 1):
        loss2, w2 = two[r]
        if abs(loss2 - loss1) > 1e-3 * max(1.0, abs(loss1)):
            print(f"rank {r}: loss {loss2:.6f} vs tp=1 {loss1:.6f}")
            ok = False
        got = {k: torch.tensor(v).reshape(s) for k, (v, s) in w2.items()}
        # The updated WEIGHTS, not the gradients: a per-shard clip norm scales
        # each rank differently and only shows up after the optimizer applies it.
        want = shard_params({k: v for k, v in ref.items() if k in got}, cfg, r, 2)
        for k in sorted(got):
            w = want.get(k)
            if w is None or w.shape != got[k].shape:
                continue
            compared += 1
            # In ULPS, not absolutely. These are bf16 weights, so the two orders
            # of summation can only ever land one rounding step apart, and one
            # step is 7.8e-3 at magnitude 1 -- an absolute tolerance tight enough
            # to catch a wrong clip scale flags that rounding as a failure, and
            # one loose enough to allow it is 4 bf16 steps wide.
            d = (got[k].float() - w.float()).abs()
            ulp = torch.exp2(torch.floor(torch.log2(w.float().abs().clamp_min(1e-30))) - 7)
            off = (d > ulp * 1.01).sum().item()
            if off:
                print(f"rank {r}: {k} POST-UPDATE MISMATCH {off}/{d.numel()} weights off by "
                      f">1 bf16 ulp, max {(d / ulp).max().item():.1f} ulp")
                ok = False
    return ok, compared, loss1, two[0][0]


def _refusals() -> bool:
    """``--tp`` must refuse the two layouts it cannot train correctly.

    The refusal has to happen BEFORE ``init_tp``: this runs in the parent, where
    the spawned arms have already left MASTER_ADDR set, so a ``_shard`` that
    accepted dp=2 would block in ``init_process_group(world_size=4)`` waiting for
    three ranks that do not exist. A control that hangs reports nothing, so the
    backend here raises instead of joining -- reaching it at all is the failure.
    """
    from tilerl import model as model_mod
    from tilerl.cli import _shard
    from tilerl.config import tiny

    class _NoJoin:
        def init_tp(self, *a, **kw):
            raise AssertionError("reached init_tp: the layout was accepted")

    cfg = tiny()
    m = model_mod.build_random(cfg, seed=0)
    keep = os.environ.get("WORLD_SIZE"), os.environ.get("RANK")
    os.environ["WORLD_SIZE"], os.environ["RANK"] = "4", "0"
    ok = True
    for tp, want in ((3, "does not divide"), (2, "not reduced across dp replicas")):
        try:
            _shard(cfg, m, tp, _NoJoin(), model_mod)
        except SystemExit as e:
            if want not in str(e):
                print(f"--tp {tp}: refused with the wrong reason: {e}")
                ok = False
        except AssertionError as e:
            print(f"--tp {tp} under WORLD_SIZE=4: {e}; it must be refused")
            ok = False
        else:
            print(f"--tp {tp} under WORLD_SIZE=4 was ACCEPTED; it must be refused")
            ok = False
    for k, v in zip(("WORLD_SIZE", "RANK"), keep):
        os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    print("--tp refuses a non-dividing world and an unreduced dp>1" if ok else "--tp: FAILED")
    return ok


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
    if no_fork:
        print("negative control: correctly FAILED on both head layouts"
              if rc == 0 else "negative control PASSED somewhere -- vacuous gate")
        return rc
    if rc == 0:
        print("TP-2 gradients match TP-1 on both head layouts")

    # End to end: a whole train_step, weights compared AFTER the update.
    local_clip = "--local-clip" in sys.argv
    ok, compared, loss1, loss2 = _one_step(local_clip)
    print(f"[train_step] loss tp=1 {loss1:.6f} vs tp=2 {loss2:.6f}; "
          f"{compared} updated tensors compared: " + ("match" if ok else "DIFFER"))
    if local_clip:
        print("local-clip control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    if not _refusals():
        rc = 1
    return rc or (0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
