"""The precision table is total over its roles, and the call sites read it."""

import torch

from tilerl import precision


def test_policy_is_total_and_device_aware():
    for r in precision.roles():
        assert isinstance(precision.dtype(r, "cpu"), torch.dtype)
    assert precision.dtype("recurrent_state", "cuda") == torch.float32
    assert precision.dtype("recurrent_state", "cpu") == torch.bfloat16
    assert precision.dtype("optimizer_state") == torch.float32


def test_iso_frames_follow_the_policy():
    from tilerl.iso import ISO

    u, _, _ = ISO().frames(torch.randn(6, 4, dtype=torch.bfloat16))
    assert u.dtype == precision.dtype("frame")


def test_on_policy_guard_refuses_cached_engines():
    """Both arms, because only one of them was reachable from this test.

    The guard refuses a captured decode graph OR a live prefix store. Measured by
    mutation: with the `_decode_graph_on` half deleted this test still passed — on cpu
    `_graph_on` returns False, so the engine it builds trips the prefix arm and the
    graph arm never fires. That arm is the one that matters on the pod, where
    `_graph_on` defaults to True for CUDA and a `grpo_loop` call that forgot
    `decode_graph=False` lands on exactly the untested half.

    Deleting either half now fails: prefix arm CAUGHT before, graph arm CAUGHT after.
    """
    import pytest

    from tilerl.cli import _build_model
    from tilerl.engine import build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.testing import RefBackend
    from tilerl.train import grpo_loop

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    run = lambda e: list(grpo_loop(e, model, [[1, 2, 3]], lambda p, c: 0.0, 1, RefBackend()))

    # prefix cache on, graph off
    cached = build_engine(cfg, model, RefBackend(), num_blocks=32, num_slots=4)
    with pytest.raises(ValueError, match="on-policy"):
        run(cached)

    # graph on, prefix off — decode_graph=True is honoured on cpu, so this is testable
    # here and not a CUDA-only path.
    graphed = build_engine(cfg, model, RefBackend(), num_blocks=32, num_slots=4,
                           decode_graph=True, prefix_store=NoPrefixStore())
    assert graphed._decode_graph_on is True, "decode_graph=True was not honoured"
    with pytest.raises(ValueError, match="on-policy"):
        run(graphed)



def test_opd_refuses_a_cached_engine_with_no_adapters_too():
    """The guard must be unconditional, not conditional on `trainable`.

    `opd_loop` used to skip it when `trainable is None`, reasoning that "a frozen teacher
    with no adapters cannot go stale". The teacher is not frozen: with no `trainable`,
    `train_step` updates `model.params` (train.py:81) and the engine samples from that same
    object (engine.py:320). Measured on tiny, 27 of the teacher's parameters changed within
    two steps -- the exempt path was the one that went stale fastest.

    Asserts the no-adapter call is refused, because the tree has no caller passing None
    today and "nobody passes it" is a fact that a future caller silently reverses. The
    other arm (with adapters) is covered above.
    """
    import pytest

    from tilerl.cli import _build_model
    from tilerl.engine import build_engine
    from tilerl.testing import RefBackend
    from tilerl.train import opd_loop

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    # Prefix store on: the same cached-engine condition the adapter arm is refused for.
    cached = build_engine(cfg, model, RefBackend(), num_blocks=32, num_slots=4)
    with pytest.raises(ValueError, match="on-policy"):
        opd_loop(cached, model, [[1, 2, 3]], 1, RefBackend(), trainable=None)


def test_kernel_io_is_keyed_on_arch_not_on_being_cuda():
    """No dtype decision in backend.py may read `target.startswith("cuda")`.

    sm70 has no bf16 load, so its cells are compiled f32 and `Backend.io` says so
    per-arch. Three call sites re-derived it as `bf16 if cuda else f32`, which is
    right on sm90 and wrong on every other CUDA arch: gdn_prep then got f16 tensors
    against an f32 signature and died with "input Q dtype mismatch, expected
    float32" -- mid-run, after the weights were resident, pointing at the kernel
    rather than at the marshalling.

    Structural because there is no sm70 in CI: the arch that breaks cannot run the
    numeric gate, so the gate has to read the source. Negative control: restore any
    of the three `io = torch.bfloat16 if self.target.startswith("cuda")` lines and
    this fails naming its line number.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels"
           / "src" / "tilerl_kernels" / "backend.py").read_text()
    bad = []
    for node in ast.walk(ast.parse(src)):
        # An `x = <bf16/f16> if <...cuda...> else <...>` anywhere: the dtype is the
        # tell, since target.startswith("cuda") is legitimate for control flow.
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp)):
            continue
        whole, cond = ast.dump(node.value), ast.dump(node.value.test)
        if "cuda" in cond and ("bfloat16" in whole or "float16" in whole):
            bad.append(node.lineno)
    assert not bad, (
        f"backend.py:{bad}: a dtype chosen by `target.startswith('cuda')`. Use "
        f"self.io / self.gemv_io / self.scale_io, which are keyed on arch -- sm70 "
        f"is CUDA and takes f32."
    )


if __name__ == "__main__":  # runnable check
    test_policy_is_total_and_device_aware()
    test_kernel_io_is_keyed_on_arch_not_on_being_cuda()
    print("precision: policy OK")
