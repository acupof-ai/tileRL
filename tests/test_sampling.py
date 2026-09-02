"""top_k is a threshold and a mask on the device: no sync, per-row k."""

import torch

from tilerl.engine import SamplingParams, _restrict


def test_top_k_keeps_k_logits():
    logits = torch.tensor([0.1, 3.0, 2.0, -1.0, 2.5])
    out = _restrict(logits, SamplingParams(top_k=2))
    assert torch.isfinite(out).sum() == 2 and out[1] == 3.0 and out[4] == 2.5
    assert torch.equal(_restrict(logits, SamplingParams(top_k=0)), logits)
    assert torch.isfinite(_restrict(logits, SamplingParams(top_k=1))).nonzero().item() == 1
