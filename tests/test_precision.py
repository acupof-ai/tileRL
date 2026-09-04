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
    import pytest

    from tilerl.cli import _build_model
    from tilerl.engine import build_engine
    from tilerl.testing import RefBackend
    from tilerl.train import grpo_loop

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    engine = build_engine(cfg, model, RefBackend(), num_blocks=32, num_slots=4)  # prefix cache on
    with pytest.raises(ValueError, match="on-policy"):
        list(grpo_loop(engine, model, [[1, 2, 3]], lambda p, c: 0.0, 1, RefBackend()))


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
