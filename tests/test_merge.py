"""ISO-Merger on the tiny model: the frame procedure is exact enough on one
specialist, keeps the base spectrum, and composing two SFT specialists beats
both the base and plain task-vector averaging on their own batches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tilerl.autograd import AdamW
from tilerl.cli import _build_model
from tilerl.merge import (
    average_merge,
    dare_merge,
    dare_merge_weight,
    iso_merge,
    ties_merge,
    ties_merge_weight,
)
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


def test_ties_arithmetic_pinned_on_hand_matrices():
    """TIES: per-row top-keep trim, sign election from the trimmed sum, disjoint
    mean of agreeing specialists. Numbers hand-computed, not re-derived."""
    w0 = torch.zeros(2, 3)
    # Row 0 task vectors: s1 [-5,1,0], s2 [4,1,0]. Row 1: s1 [0,3,10], s2 [0,1,10].
    w1 = torch.tensor([[-5.0, 1.0, 0.0], [0.0, 3.0, 10.0]])
    w2 = torch.tensor([[4.0, 1.0, 0.0], [0.0, 1.0, 10.0]])
    # keep=1/3 keeps one entry per row per specialist: col 0 both rows' top in the
    # column each specialist dominates, so row0 col1's +1 votes trim away (without
    # the trim they survive and col1 would add +1), and row1 col1's 3/1 trim away.
    # Row 0 col0: trimmed sum -1 elects -, only s1 agrees -> -5 (not a -1 vote sum).
    # Row 1 col2: both agree +, disjoint MEAN (10+10)/2 = 10, not the 20 sum.
    out = ties_merge_weight(w0, [w1, w2], keep=1 / 3)
    assert torch.allclose(out, torch.tensor([[-5.0, 0.0, 0.0], [0.0, 0.0, 10.0]]))
    # K=1 with keep=1 trims nothing and returns the specialist (f32 roundoff only).
    w0s, ws = torch.randn(3, 4), torch.randn(3, 4)
    assert torch.allclose(ties_merge_weight(w0s, [ws], keep=1.0), ws, atol=1e-6)
    # Non-float tensors pass through untouched (task vectors need float math).
    wi = torch.tensor([[1, 2], [3, 4]])
    assert torch.equal(ties_merge({"k": wi}, [{"k": wi}, {"k": wi}])["k"], wi)


def test_dare_arithmetic_pinned_on_hand_matrices():
    """DARE: Bernoulli drop per specialist, survivors rescaled 1/(1-p), averaged.
    The seed masks are explicit torch.Generator draws, pinned here."""
    z = torch.zeros(4)
    s1, s2 = torch.full((4,), 10.0), torch.full((4,), 20.0)
    # seed 0: mask [0,1,0,0]; seed 1 draws [0.7576,0.2793,0.4031,0.7347] -> [1,0,0,1].
    # survivors rescaled x2 then averaged: col0 s2 40/2=20, col1 s1 20/2=10,
    # col2 neither -> 0, col3 s2 40/2=20.
    out = dare_merge_weight(z, [s1, s2], drop=0.5, seed=0)
    assert torch.allclose(out, torch.tensor([20.0, 10.0, 0.0, 20.0]))
    # drop=0 is plain averaging; the same seed reproduces the same mask, a different
    # seed changes it.
    assert torch.allclose(dare_merge_weight(z, [s1, s2], drop=0.0, seed=0),
                          torch.full((4,), 15.0))
    a = dare_merge_weight(z, [s1, s2], drop=0.5, seed=7)
    assert torch.equal(a, dare_merge_weight(z, [s1, s2], drop=0.5, seed=7))
    assert not torch.equal(a, dare_merge_weight(z, [s1, s2], drop=0.5, seed=8))
    wi = torch.tensor([1, 2, 3])
    assert torch.equal(dare_merge({"k": wi}, [{"k": wi}, {"k": wi}])["k"], wi)


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


def test_ties_and_dare_keep_both_tasks_and_iso_beats_them():
    """The two task-vector baselines each keep BOTH tasks (beat the base), and on
    the tiny harness ISO already ranks ahead of both — the shape of the pending
    27B verdict that ISO must beat TIES/DARE. Measured at this tree:
    avg 18.46/17.55, ties 18.02/16.45, dare 18.45/17.55, iso 16.00/14.73, base 22.34/21.99."""
    backend = RefBackend()
    cfg, base = _build_model("tiny", seed=0, keep_master=True)
    a, b = _sft(BATCH_A, backend), _sft(BATCH_B, backend)
    arms = {
        name: Model(cfg, fn(base.params, [a.params, b.params]))
        for name, fn in (("ties", ties_merge), ("dare", dare_merge))
    }
    iso = Model(cfg, iso_merge(base.params, [a.params, b.params]))
    loss = {n: (_loss(m, BATCH_A, backend), _loss(m, BATCH_B, backend))
            for n, m in arms.items()}
    base_l = (_loss(base, BATCH_A, backend), _loss(base, BATCH_B, backend))
    iso_l = (_loss(iso, BATCH_A, backend), _loss(iso, BATCH_B, backend))
    print({n: f"A={la:.3f} B={lb:.3f}" for n, (la, lb) in loss.items()},
          f"iso A={iso_l[0]:.3f} B={iso_l[1]:.3f}", f"base A={base_l[0]:.3f} B={base_l[1]:.3f}")
    for n, (la, lb) in loss.items():
        assert la < base_l[0] and lb < base_l[1], f"{n} dropped a task: {n} {la}/{lb} vs {base_l}"
        assert iso_l[0] <= la and iso_l[1] <= lb, f"ISO did not beat {n} on tiny: {iso_l} vs {la}/{lb}"


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


def test_merge_checkpoints_ties_and_dare_equal_dict_level(tmp_path):
    """The streaming path dispatches ties/dare to the per-weight math over the raw
    checkpoint keys (HF names), exactly like iso does. The DARE salt is the raw
    key, so the reference here merges the raw tensors under those same keys —
    internal load_hf keys would salt a different mask."""
    import zlib

    from safetensors import safe_open

    from tilerl.merge import dare_merge_weight, merge_checkpoints, ties_merge_weight
    from tilerl.model import save_hf

    backend = RefBackend()
    dirs = []
    for seed in (0, 1):
        cfg, model = _build_model("tiny", seed=0, keep_master=True)
        if seed:
            for _ in range(3):
                train_step(model, torch.randint(1, cfg.vocab_size, (2, 16)).numpy(),
                           backend, AdamW(lr=1e-3))
        save_hf(model, tmp_path / f"mc{seed}")
        dirs.append(str(tmp_path / f"mc{seed}"))

    def raw(d):
        f = next((tmp_path / d).glob("model-*.safetensors"), None) \
            or (tmp_path / d / "model.safetensors")
        h = safe_open(str(f), "pt")
        keys = h.keys()  # noqa: SIM118 — safe_open handle is not directly iterable
        return {k: h.get_tensor(k) for k in keys}

    r0, r1 = raw("mc0"), raw("mc1")
    for method, weight_fn in (("ties", ties_merge_weight), ("dare", dare_merge_weight)):
        merge_checkpoints(dirs[0], dirs[1:], tmp_path / f"o-{method}", method=method)
        got = raw(f"o-{method}")
        for k, w0 in r0.items():
            if w0.is_floating_point():
                seed = zlib.crc32(k.encode()) if method == "dare" else None
                want = weight_fn(w0, [r1[k]], seed=seed) if method == "dare" \
                    else weight_fn(w0, [r1[k]])
                assert torch.equal(got[k], want.contiguous()), (method, k)
    # A re-run of DARE is byte-identical: the mask comes from the key, not clock randomness.
    merge_checkpoints(dirs[0], dirs[1:], tmp_path / "o-dare2", method="dare")
    a, b = raw("o-dare"), raw("o-dare2")
    assert all(torch.equal(a[k], b[k]) for k in a)


def _sft_run(model_dir_arg: str, seed: int):
    """One full-SFT tiny run via the CLI with --save-model; returns (run_id, model dir).

    The saved model is the producer artifact a merge specialist links back through.
    The run is selected by its seed, not directory order: two runs coexist and APFS
    readdir order made an unfiltered scan return seed=0 twice (an id_a==id_b failure
    that CI's runner order hid).
    """
    import json as _json

    from tilerl import cli
    from tilerl.ledger import runs_root

    argv = ["train", "--model", "tiny", "--steps", "1", "--seed", str(seed),
            "--save-model"]
    if model_dir_arg == "json":
        argv.append("--json")
    import contextlib

    with contextlib.suppress(SystemExit):  # steps=1 cannot satisfy ce_falls; manifest still lands
        cli.cmd_train(cli._build_parser().parse_args(argv))
    for d in Path(runs_root()).iterdir():
        m = _json.loads((d / "manifest.json").read_text())
        if m.get("artifacts", {}).get("out") and m["inputs"].get("seed") == seed:
            return m["id"], m["artifacts"]["out"]
    raise AssertionError(f"seed={seed} SFT run wrote no artifacts.out")


def test_merge_lineage_idempotency_and_json(tmp_path, monkeypatch, capsys):
    """A merge records the runs that wrote its inputs, a repeat is a no-op, and
    --json prints the manifest — the P4 lineage chain end to end through the CLI."""
    import json as _json

    from tilerl import cli
    from tilerl import merge as merge_mod
    from tilerl.ledger import lineage, list_runs, read_manifest, runs_root

    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    id_a, dir_a = _sft_run("json", 0)
    id_b, dir_b = _sft_run("", 1)
    assert id_a != id_b
    # The SFT artifact is a mergeable bf16 checkpoint dir, not an adapter.
    assert list(Path(dir_a).glob("*.safetensors"))

    out = str(tmp_path / "merged")
    args = ["merge", "--base", dir_a, "--specialists", dir_b, "--out", out, "--json"]
    capsys.readouterr()  # discard the two SFT runs' output
    cli.cmd_merge(cli._build_parser().parse_args(args))
    (mm,) = [r for r in list_runs(runs_root()) if r["command"] == "merge"]
    assert mm["parents"] == [id_a, id_b], mm["parents"]
    assert _json.loads(capsys.readouterr().out)["id"] == mm["id"]  # --json prints the manifest
    # The two-node (here three-node) walk reaches both producers.
    assert [r["id"] for r in lineage(runs_root(), mm["id"])] == [mm["id"], id_a, id_b]

    # A repeat with the same inputs is a no-op: merge_checkpoints is never called again
    # and no new manifest appears.
    calls = 0
    real = merge_mod.merge_checkpoints

    def _count(*a, **k):
        nonlocal calls
        calls += 1
        return real(*a, **k)

    monkeypatch.setattr(merge_mod, "merge_checkpoints", _count)
    cli.cmd_merge(cli._build_parser().parse_args(
        ["merge", "--base", dir_a, "--specialists", dir_b, "--out", out]))
    assert calls == 0
    assert len(list(Path(runs_root()).iterdir())) == 3
    assert read_manifest(runs_root(), mm["id"])["finished"]


if __name__ == "__main__":  # runnable check
    test_iso_merge_one_specialist_and_spectrum()
    test_iso_merge_two_specialists()
    print("merge: K=1, spectrum, two specialists OK")
