"""ISO-Merger on the tiny model: the frame procedure is exact enough on one
specialist, keeps the base spectrum, and composing two SFT specialists beats
both the base and plain task-vector averaging on their own batches.
"""

from __future__ import annotations

import numpy as np
import torch

from tilerl.autograd import AdamW
from tilerl.cli import _build_model
from tilerl.merge import average_merge, iso_merge
from tilerl.model import Model
from tilerl.testing import RefBackend
from tilerl.train import train_step

_RNG = np.random.default_rng(0)
BATCH_A = _RNG.integers(1, 320, size=(4, 16))
BATCH_B = _RNG.integers(1, 320, size=(4, 16))


def _loss(model, ids, backend):
    return train_step(model, ids, backend, AdamW(lr=0.0))  # lr=0: forward + loss, no update


def _sft(ids, backend, steps=15):
    _, model = _build_model("tiny", seed=0, keep_master=True)
    opt = AdamW(lr=1e-3)
    for _ in range(steps):
        train_step(model, ids, backend, opt)
    return model


def _f32(params):
    return {k: v.float() for k, v in params.items()}


def test_iso_merge_one_specialist_and_spectrum():
    """K=1 returns the specialist (up to the masked trailing modes), and every
    merged matrix carries the base's singular values."""
    backend = RefBackend()
    _, base = _build_model("tiny", seed=0, keep_master=True)
    spec = _sft(BATCH_A, backend)
    merged = iso_merge(_f32(base.params), [_f32(spec.params)])
    for k, w in merged.items():
        if w.dim() != 2:
            continue
        err = (w - spec.params[k].float()).norm() / spec.params[k].float().norm()
        assert err < 1e-2, f"{k}: K=1 merge is {err:.2e} from the specialist"
        s0, s = torch.linalg.svdvals(base.params[k].double()), torch.linalg.svdvals(w.double())
        assert torch.allclose(s, s0, rtol=1e-3), f"{k}: spectrum moved"


def test_the_averaging_control_is_balanced_across_both_tasks():
    """`average_merge` is the control ISO is judged against, so it needs its own gate.

    Every verdict in `test_iso_merge_two_specialists` is relative to this arm, and
    the reason to gate it is NOT that a broken control flatters ISO -- measured, it
    does not. A control collapsed to one specialist is easier to beat on the task it
    dropped and *harder* on the task it kept, because that arm moves all the way to
    the specialist, which beats any merge on its own task:

        control            A        B    iso <= A?  iso <= B?
        avg(A,B) correct   18.460   17.550   yes       yes
        avg(A) only        14.945   21.941   NO        yes
        avg(B) only        22.291   13.565   yes       NO

    So a degenerate control turns ISO's gate red on a merge that is fine. The gate
    here is what tells those two apart, and without it the failure reads as an ISO
    regression.

    The obvious formulation does not work: "average beats the base on both tasks" is
    TRUE for the A-only average (B=21.941 against the base's own 21.990, ahead by
    0.049) -- it passes the same way #299's `or` passed, one layer down. What
    separates them is the BALANCE of the two gains, not their sign:

        arm            gain A   gain B   ratio
        avg(A,B)        3.875    4.440    1.1x
        avg(A) only     7.390    0.049  150.1x
        avg(B) only     0.044    8.425  190.4x

    Two decades between the real average and either degenerate one, so the 3x
    threshold is a wide band rather than a value fitted to these numbers.
    """
    backend = RefBackend()
    cfg, base = _build_model("tiny", seed=0, keep_master=True)
    a, b = _sft(BATCH_A, backend), _sft(BATCH_B, backend)

    def gains(specialists):
        m = Model(cfg, average_merge(base.params, specialists))
        return (_loss(base, BATCH_A, backend) - _loss(m, BATCH_A, backend),
                _loss(base, BATCH_B, backend) - _loss(m, BATCH_B, backend))

    def ratio(g):
        lo, hi = sorted(g)
        return hi / lo if lo > 1e-9 else float("inf")

    ga, gb = gains([a.params, b.params])
    assert ga > 0 and gb > 0, f"averaging lost to the base: A {ga:.3f} B {gb:.3f}"
    assert ratio((ga, gb)) < 3, f"averaging is lopsided: A {ga:.3f} B {gb:.3f}"
    # Negative controls: an average that saw one specialist must fail, and it is
    # the ratio that fails it -- both of these still beat the base on both tasks.
    for name, specs in (("A only", [a.params]), ("B only", [b.params])):
        one = gains(specs)
        assert ratio(one) >= 3, f"{name} passed the balance gate: {one}"
        assert min(one) > 0, f"{name} no longer beats the base, so the gate is not the ratio"


def test_iso_merge_two_specialists():
    """The P3 merger gate: two specialists EACH keep their own task better than
    plain averaging (`roadmap.md:120-123`), so this is a conjunction.

    It was an `or`, which a merge that ignores one specialist passes on the
    strength of the other: dropping specialist B scores A=15.660 B=21.954 --
    0.036 below the base's own 21.990, and 4.4 worse than averaging -- and the
    `or` admitted it.

    What the `and` does and does not catch, swept rather than assumed. #284 says
    this loss comparison covers the merge-math constants; it covers them past
    their knee. `ridge` is monotone with a knee at 1.0 (mean |dW|/|W| 0.0137 at
    1e-3, 0.0125 at 1e-1, 0.0069 at 1.0, 0.0000 at 1e4), so 1e-1 passing is the
    parameter still working, not a blind spot -- the gate fails from 1.0 up.
    `rho_keep` 0.9 -> 0.1 fails at A=21.646 B=21.110.
    """
    backend = RefBackend()
    cfg, base = _build_model("tiny", seed=0, keep_master=True)
    a, b = _sft(BATCH_A, backend), _sft(BATCH_B, backend)
    iso = Model(cfg, iso_merge(base.params, [a.params, b.params]))
    avg = Model(cfg, average_merge(base.params, [a.params, b.params]))
    out = {
        n: (_loss(m, BATCH_A, backend), _loss(m, BATCH_B, backend))
        for n, m in (("base", base), ("avg", avg), ("iso", iso))
    }
    print({n: f"A={la:.3f} B={lb:.3f}" for n, (la, lb) in out.items()})
    assert out["iso"][0] < out["base"][0] and out["iso"][1] < out["base"][1], out
    # EACH task, not either: an `or` here is passed by a merge that lost one specialist.
    assert out["iso"][0] <= out["avg"][0], f"A regressed against averaging: {out}"
    assert out["iso"][1] <= out["avg"][1], f"B regressed against averaging: {out}"


def test_merge_checkpoints_streams_shards_and_records(tmp_path, monkeypatch):
    """The file-level merge equals the dict-level one, writes shards load_hf
    reads back, and leaves a manifest."""
    import json
    import sys

    from tilerl.cli import main
    from tilerl.merge import merge_checkpoints
    from tilerl.model import load_hf, save_hf

    backend = RefBackend()
    dirs, params = [], []
    for seed in (0, 1, 2):
        cfg, model = _build_model("tiny", seed=0, keep_master=True)
        if seed:
            ids = torch.randint(1, cfg.vocab_size, (2, 16), generator=torch.Generator().manual_seed(seed))
            for _ in range(3):
                train_step(model, ids.numpy(), backend, AdamW(lr=1e-3))
        save_hf(model, tmp_path / f"ck{seed}")
        dirs.append(str(tmp_path / f"ck{seed}"))
        params.append(dict(model.params))
    n = merge_checkpoints(dirs[0], dirs[1:], tmp_path / "out", shard_bytes=1 << 14)
    assert n and len(list((tmp_path / "out").glob("model-*.safetensors"))) > 1
    got = load_hf(cfg, tmp_path / "out", keep_master=True).params
    want = iso_merge(params[0], params[1:])
    for k, w in want.items():
        if w.dim() == 2:
            # Exact, not a tolerance: both sides reach iso_merge_weight (merge.py:79 and
            # :144), so the merge math cancels and only the shard write plus load_hf round
            # trip is under test. That path has no dtype conversion, so the old atol=rtol=2e-2
            # guarded a difference measured at 0 over 17 tensors. The detection floor is
            # bf16's mantissa step, not this assertion: scaling the shard by 1.001 still
            # rounds to the same value, 1.01 is caught. The merge math itself is covered by
            # test_iso_merge_two_specialists' loss comparison, not here.
            assert torch.equal(got[k], w), k
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(sys, "argv", ["tilerl", "merge", "--base", dirs[0], "--specialists",
                                      ",".join(dirs[1:]), "--out", str(tmp_path / "out2")])
    main()
    m = json.loads(next((tmp_path / "runs").glob("*/manifest.json")).read_text())
    assert m["command"] == "merge" and m["metrics"]["tensors"] == n
    # `cmd_merge` writes its manifest directly, never through `_finish`, so it defines no
    # gates -- and `gates_pass([])` is `all([])` = True, which made every merge row read
    # `pass` over zero checks. The verdict for a finished run with no gates is `none`.
    from tilerl.ledger import format_run

    assert m["gates"] == [] and m["finished"], m
    assert format_run(m).split()[3] == "none", format_run(m)


if __name__ == "__main__":  # runnable check
    test_iso_merge_one_specialist_and_spectrum()
    test_iso_merge_two_specialists()
    print("merge: K=1, spectrum, two specialists OK")
