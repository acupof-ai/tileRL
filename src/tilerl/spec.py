"""Speculative decoding: the draft head and the verify-length policy.

Two pieces, separable:

``verify_lens`` decides HOW MANY drafted tokens per request are worth verifying
this tick — DSpark §3.2.2 (sglang's ``compute_verify_token_budget``). Verifying
a draft costs a row in the trunk forward whether or not it is accepted, so the
question is goodput, not acceptance rate: maximize
``(R + Σ top-B survival) / (bias + row·(R + B))`` over the admission cut. B=0
is one of the arms, so the policy never chooses to speculate when speculating
loses.

``survival[j]`` is P(the first j+1 drafts all accept) — monotone decreasing, the
cumulative product of per-position confidence. A checkpoint with a
``confidence_head`` supplies that confidence directly; a DFlash-style head has
none, and the draft's own softmax probability for the token it emitted is the
fallback.
"""

from __future__ import annotations

__all__ = ["verify_lens", "survival"]

#: Measured cost of one trunk verify forward: a fixed cost plus a per-row cost,
#: in ms. The defaults are agent-infer's H20 numbers; re-measure per target.
BIAS_MS = 211.0
ROW_MS = 0.53


def survival(confidences: list[float]) -> list[float]:
    """Per-position confidence -> P(first j+1 drafts all accept)."""
    out, p = [], 1.0
    for c in confidences:
        p *= float(c)
        out.append(p)
    return out


def verify_lens(
    survivals: list[list[float]], bias_ms: float = BIAS_MS, row_ms: float = ROW_MS
) -> list[int]:
    """Per-request draft-keep lengths maximizing verify goodput.

    ``survivals[r]`` must be monotone decreasing (it is a cumulative product),
    which is what makes a single global admission cut yield a PREFIX per
    request rather than an arbitrary subset.
    """
    eps = 1e-6
    r = len(survivals)
    flat = sorted((p for s in survivals for p in s if p >= eps), reverse=True)
    best, cut, total = r / (bias_ms + row_ms * r), float("inf"), 0.0
    for i, p in enumerate(flat, 1):
        total += p
        theta = (r + total) / (bias_ms + row_ms * (r + i))
        if theta > best:
            best, cut = theta, p
    out = []
    for s in survivals:
        n = 0
        while n < len(s) and s[n] >= cut:
            n += 1
        out.append(n)
    return out


if __name__ == "__main__":  # runnable check
    assert survival([0.9, 0.8, 0.5]) == [0.9, 0.9 * 0.8, 0.9 * 0.8 * 0.5]
    # A confident draft is worth verifying; a hopeless one is not.
    assert verify_lens([[0.99, 0.98, 0.97]], bias_ms=1.0, row_ms=0.1) == [3]
    assert verify_lens([[1e-9, 1e-9]]) == [0]
    # Cheap rows -> keep more; the cut is global, the kept span is a prefix.
    lens = verify_lens([[0.99, 0.9, 0.2], [0.3, 0.05, 0.01]], bias_ms=1.0, row_ms=0.1)
    assert lens[0] >= lens[1], lens
    print("spec: verify_lens OK", lens)
