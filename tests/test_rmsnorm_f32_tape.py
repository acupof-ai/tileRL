"""`rmsnorm_f32` is a new tape op, and the tape's failure mode for one is silence.

The q/k norm fix routes those two call sites to `backend.rmsnorm_f32` so the
output reaches rope in f32 and rounds once at the KV pool instead of twice
(errors/2026-09-03-unfused-prelude-double-rounds.md, ratio 2.0306 on the elements
the change moves).

That makes it a **new backend op**, not an edit to an existing one, and
`_TapeBackend.__getattr__` opens with `if name not in _BWD: return attr` — so an
op nobody registers is returned raw and records nothing. Measured before writing
this: a `rmsnorm_f32` absent from `_BWD` produced **0 tape entries** where
`rmsnorm` produced 1, with no error and no warning. Forward parity, the CPU twin
and the sm90 oracle all still pass in that state, and training silently stops
learning `q_norm` and `k_norm`.

This is the per-op gate: the entry is recorded, and the gradient the tape resolves
for it matches finite differences. The population gate — every forward op is in
`_BWD` or declared gradient-free, which catches the *next* op added this way — is
tilerl-45's `#48`; the two catch different failures (the op you forget vs the
backward you get wrong) and neither retires the other.
"""

import torch

from tilerl import autograd
from tilerl.testing import RefBackend


def test_rmsnorm_f32_records_a_tape_entry_and_its_gradient_is_right():
    be = RefBackend()
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8)
    w = torch.randn(8)
    eps = 1e-6

    with autograd.Tape() as tape:
        rb = autograd.RecordingBackend(be)
        y = rb.rmsnorm_f32(x, w, eps)
    assert len(tape._entries) == 1, (
        f"rmsnorm_f32 recorded {len(tape._entries)} tape entries, not 1: it is missing "
        "from autograd._BWD, so the q/k norm gradient is silently dropped"
    )
    assert y.dtype == torch.float32, f"the whole point is an f32 output, got {y.dtype}"

    # _BWD maps it to rmsnorm_bwd -- _default's argument names the backward, not
    # the key. Check that resolved backward against central differences.
    handler = autograd._BWD["rmsnorm_f32"]
    grads = dict(handler(be, torch.ones_like(y), (x, w, eps), {}))
    gx_a = grads[0]
    assert gx_a is not None and gx_a.shape == x.shape

    step, worst = 1e-3, 0.0
    flat = x.reshape(-1)
    for j in torch.randperm(flat.numel())[:16]:
        old = flat[j].item()
        flat[j] = old + step
        fp = be.rmsnorm_f32(x, w, eps).sum().item()
        flat[j] = old - step
        fm = be.rmsnorm_f32(x, w, eps).sum().item()
        flat[j] = old
        num = (fp - fm) / (2 * step)
        ana = gx_a.reshape(-1)[j].item()
        worst = max(worst, abs(ana - num) / max(abs(ana), abs(num), 1e-6))
    assert worst < 5e-2, f"rmsnorm_f32 dx finite-diff rel err {worst:.3e}"


def test_rmsnorm_f32_matches_rmsnorm_up_to_the_output_dtype():
    """The fix must change only where the value is rounded, not the value.

    On CPU both go through the same f32 reference, so this is exact there; on sm90
    `rmsnorm` stores bf16 and `rmsnorm_f32` stores f32, and the sm90 arm of
    test_attn_prelude_oracle is what measures that difference."""
    be = RefBackend()
    torch.manual_seed(1)
    x, w = torch.randn(3, 5, 16), torch.randn(16)
    a = be.rmsnorm(x, w, 1e-6)
    b = be.rmsnorm_f32(x, w, 1e-6)
    assert torch.equal(a.float(), b.float()), (a - b).abs().max()
