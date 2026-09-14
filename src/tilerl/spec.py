"""Speculative decoding: the draft head and the verify-length policy.

``verify_lens`` decides how many drafted tokens per request are worth verifying
this tick (DSpark §3.2.2, sglang's ``compute_verify_token_budget``): a draft
costs a trunk row whether or not it is accepted, so maximize goodput
``(R + Σ top-B survival) / (bias + row·(R + B))`` over the admission cut. B=0
is one of the arms. ``survival[j]`` = P(the first j+1 drafts all accept).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .kv_cache import BLOCK_TOKENS, BatchKv
from .model import Model

#: One trunk verify forward = fixed + per-row cost, ms. agent-infer's H20 numbers.
#: sm70 is a staircase, not a line: the GEMV ladder rounds verify width up to a
#: rung, so at ctx 1024 verify costs w<=2 29.38, w<=4 45.49, w<=8 80.31 ms, one
#: draft forward 5.75 (errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md,
#: errors/2026-09-03-block-parallel-drafting-is-1.016x-on-sm70.md). Both measured
#: DIRECTLY -- CUDA events on the draft, verify = that rung's tick minus that rung's
#: own draft -- because a cross-depth subtraction amplifies tick noise 12.9x
#: (wins/2026-09-04-a-difference-amplifies-its-operands-noise.md). Those components
#: rebuild the end-to-end tick to within 14% at W=2/4/8, so depth moves TWO terms:
#: one more draft forward (5.75 ms, flat) plus a wider verify -- do not price it as one.
#: H20 constants, and repricing them for sm70 is NOT the fix -- the cost's SHAPE is
#: wrong here, not its scale. engine.py pads every chain to max(len) and the ladder
#: rounds B*W up, so a trim between two widths sharing a rung saves nothing (W=3 and
#: W=4 collide at B=1 and at B=4 alike); only W<=2 is a cheaper rung. The measured
#: price would cut W=4 at acceptance p~=0.92, just below the recorded 84.4%, exactly
#: where the end-to-end numbers say W=4 earns 1.157-1.228x
#: (errors/2026-09-03-repricing-verify-lens-was-the-wrong-fix.md).
#: ponytail: H20 line on a staircase cost. The sm70 line 0.670 + 0.5265*W is fitted
#: at B=1 (bench_ctx_decode.py submits one request), so its W is a chain width and
#: not a launched-row count -- re-measure the slope at B=4 before pricing a trim with
#: it, or the rung and the line are on different axes.
BIAS_MS = 211.0
ROW_MS = 0.53

#: Prefill widths are padded to this so kernel shapes stay bounded. Lives here, not
#: in engine.py, because engine imports spec and both paths must round identically --
#: the draft skipping the bucket cost a served first visit 14 compiles inline.
_PREFILL_BUCKET = 64

#: Verify widths the sm70 M-ladder serves without padding waste. A width
#: between rungs pays the next rung's full price: depth 5 (W=6) costs the same
#: 8-row launch as depth 7 (W=8), which measured 10% SLOWER than depth 3 on the
#: one workload where every draft is accepted. 32 is the top rung, and it is
#: no longer a cliff — X is pre-packed f16 there too, 29-36 us/row against the
#: 122-128 it cost when the flag stopped at 8.
LADDER_WIDTHS = (1, 2, 4, 8, 32)


def survival(confidences: list[float]) -> list[float]:
    out, p = [], 1.0
    for c in confidences:
        p *= float(c)
        out.append(p)
    return out


def verify_lens(
    survivals: list[list[float]], bias_ms: float = BIAS_MS, row_ms: float = ROW_MS
) -> list[int]:
    """Per-request draft-keep lengths maximizing verify goodput. ``survivals[r]``
    is monotone decreasing, so one global cut yields a prefix per request.

    Prices a tick as ``bias_ms + row_ms * rows`` -- the H20 shape. sm70 instead pays
    a staircase in the WIDEST chain, so the two disagree about the optimal cut; see
    the module constants.
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


# This block is NOT the tail of the file: a class and three functions are defined below
# it, so they do not exist while it runs. Anything added here can only use what precedes
# this block. Gated by tests/test_main_selfchecks.py since 2026-09-08 -- before that, the
# #22 block-parallel reject and the staircase constants below ran only when typed.
if __name__ == "__main__":  # runnable check
    assert survival([0.9, 0.8, 0.5]) == [0.9, 0.9 * 0.8, 0.9 * 0.8 * 0.5]
    assert verify_lens([[0.99, 0.98, 0.97]], bias_ms=1.0, row_ms=0.1) == [3]
    assert verify_lens([[1e-9, 1e-9]]) == [0]
    lens = verify_lens([[0.99, 0.9, 0.2], [0.3, 0.05, 0.01]], bias_ms=1.0, row_ms=0.1)
    assert lens[0] >= lens[1], lens

    # A trim only pays on sm70 if it changes the RUNG, not the width: engine.py pads to
    # max(len) and B*W rounds up, so a trim between two widths sharing a rung buys
    # nothing. True at B=1 (W=3 and W=4 both launch 4 rows) and at B=4 (both 32), which
    # is what makes repricing the constants the wrong fix. Fails if LADDER_WIDTHS changes.
    for B, collide, cheap in ((1, 4, 2), (4, 32, 8)):
        rung = {w: next(x for x in LADDER_WIDTHS if x >= B * w) for w in (1, 2, 3, 4)}
        assert rung[3] == rung[4] == collide, f"B={B}: W=3 and W=4 must share a rung: {rung}"
        assert rung[2] == cheap, f"B={B}: W=2 must be a cheaper rung: {rung}"

    # The profiled components must still rebuild the end-to-end tick, or the two-term
    # story above is stale. verify(W) + (W-1) draft forwards, against the measured line
    # 0.670 + 0.5265*W dense ticks of 23.7 ms (ctx 1024). 6.8-10.1% on the current
    # staircase; it was 3-7% against the withdrawn one.
    # Measured staircase, rung -> ms, and one draft forward. Both are now measured
    # DIRECTLY -- CUDA events around `_draft.step`, verify = that rung's tick minus
    # that rung's own draft (scripts/ab_draft_depth.py --time-draft, ds17). No
    # cross-depth subtraction anywhere, which is what the previous values had:
    # {2: 32.79, 4: 49.52, 8: 86.24} with DRAFT 3.93 came from differencing two
    # depths' rung-matched ticks, and that difference amplifies tick noise 12.9x
    # (wins/2026-09-04-a-difference-amplifies-its-operands-noise.md).
    #
    # The two instruments AGREE on the marginal forward within one run -- 5.21 ms
    # subtracted against 5.29 direct, 1.6% -- so 3.93 was one draw of an amplified
    # difference, not a different quantity. What it cost: the draft was 1.35x
    # under-priced and the whole 3.93 gap sat in VERIFY_MS instead, which is why
    # every rung there reads 7-10% high.
    #
    # Verify per rung now agrees across depths to 0.15-0.31% (rung 2 over four
    # depths, rung 4 over three), against the 0% the subtraction asserted by
    # construction and could not show.
    VERIFY_MS = {2: 29.38, 4: 45.49, 8: 80.31}
    DRAFT_MS = 5.75
    # The staircase sits 4.2-14.0% off the 0.670 + 0.5265W dense-tick line. That
    # line was fitted with ncols=1 at B=1 and serving runs ncols=2 at B=4 (task
    # #40, open), so a per-rung disagreement of this size is the fit's, not the
    # staircase's -- the shape is what this gate is for and it holds. Tolerance is
    # 0.15 rather than 0.10 for that reason; tighten it when the line is refitted,
    # do not retune the staircase to a line measured on a different kernel.
    for w, verify_ms in VERIFY_MS.items():
        parts = (verify_ms + (w - 1) * DRAFT_MS) / 23.7
        line = 0.670 + 0.5265 * w
        assert abs(parts / line - 1) < 0.15, f"W={w}: parts {parts:.3f} vs line {line:.3f}"

    # Block-parallel drafting is rejected because WIDER is worse here, and that is
    # arithmetic over this ladder plus the staircase above -- so it belongs next to
    # them and breaks if either moves. A block head pays ONE draft forward at any k,
    # which is the whole mechanism; the suffix decays 0.79 per position (DSpark's own
    # measured [72,57,45]) off p=0.654.
    #
    # p=0.654 inverts 2.36 tok/fwd, measured on wikitext-103 at ctx=1024 -- the
    # corpus we actually serve prose from, not the uniform-random ids that read 2.99
    # (wins/2026-09-04-depth-default-is-wrong-on-text.md). Acceptance is a property
    # of the PROMPT, so no single p is "the" right one -- but the verdict does not
    # depend on choosing: written as a ratio the prompt cancels, `verdict =
    # (yield / tok_fwd) x ceiling`, and every arm from tpf 2.03 to 3.34 lands in
    # 0.97-1.06x. Held at both the random p (0.722) and this one before the switch.
    # k=3 is the optimum, and k=7 -- the width block-parallel makes cheap -- falls
    # far below it, because rung 4 -> 8 costs 36.72 ms for +0.06 tok/forward.
    # Negative control run: price rung 8 at rung 4's 49.52 ms (the "wider is free"
    # world the mechanism assumes). TWO asserts catch it independently -- the
    # staircase check above trips first (W=8 parts 3.250 vs line 4.882), and with
    # that one relaxed the optimum moves 3 -> 7 and trips the next. Checked
    # separately, because an assert that only fires after an earlier one has
    # already failed has not been shown to do anything.
    # errors/2026-09-03-block-parallel-drafting-is-1.016x-on-sm70.md
    def _tok_per_fwd(k, p=0.6536, decay=0.79):
        total, carry = 1.0, 1.0
        for i in range(k):
            carry *= p * decay**i
            total += carry
        return total

    rates = {}
    for k in (2, 3, 4, 5, 6, 7):
        rung = next(x for x in LADDER_WIDTHS if x >= 1 + k)
        if rung in VERIFY_MS:
            rates[k] = _tok_per_fwd(k) * 1000 / (VERIFY_MS[rung] + DRAFT_MS)
    assert max(rates, key=rates.get) == 3, f"k=3 must stay the optimum: {rates}"
    assert rates[7] < rates[3], (
        f"the go-wider gift must stay negative: k=7 {rates[7]:.1f} vs k=3 {rates[3]:.1f} tok/s"
    )
    # Our own head at k=3, derived from the SAME staircase and the SAME p, so both
    # sides of the ratio move together. It used to be a hardcoded 42.7 tok/s
    # measured on random ids while p came from text -- one prompt on each side.
    # Both sides are rung-4 ticks; a mean-tick rate would put a rung mixture on one
    # side only.
    #
    # The margin used to be asserted here against a 1.06 bound. That bound is GONE,
    # and this is the re-derivation the old comment demanded rather than a retune:
    # pricing the draft directly (5.75 against a differenced 3.93) moves the margin
    # to 1.104x, past 1.06. But the margin was never the load-bearing quantity --
    # it is a decay MODEL over a measurement, and it grows whenever the draft is
    # re-measured upward, so any bound on it needs retuning every time.
    #
    # What the reject rests on is one comparison, and it needs neither the decay
    # model nor the head's parameter count: **rung 8's verify alone (80.31 ms)
    # exceeds our ENTIRE k=3 tick (62.74 ms).** A block head exists to make width
    # cheap, so the arm it proposes is k=7 -- which lands on rung 8. Grant it a FREE
    # forward and zero accuracy loss and it still reads 0.862x, because the width it
    # buys is unaffordable before its own cost is priced at all.
    #
    # Falsifiable, not structural: both asserts fire under a control that prices rung
    # 8 at 50.00 ms, and they were checked SEPARATELY (an assert that only fires
    # after an earlier one has already failed has not been shown to do anything).
    # An sm70 GEMV improvement at M=8 is what would flip this -- task #21.
    ours = _tok_per_fwd(3, decay=1.0) * 1000 / (VERIFY_MS[4] + 3 * DRAFT_MS)
    assert VERIFY_MS[8] > VERIFY_MS[4] + 3 * DRAFT_MS, (
        f"rung 8 verify {VERIFY_MS[8]:.2f} ms no longer exceeds our whole k=3 tick "
        f"{VERIFY_MS[4] + 3 * DRAFT_MS:.2f} ms -- the width a block head buys has become "
        "affordable and Task #22 needs re-deriving, not reopening on a margin"
    )
    free_wide = _tok_per_fwd(7, decay=1.0) * 1000 / (VERIFY_MS[8] + DRAFT_MS)
    assert free_wide < ours, (
        f"a block head with a FREE forward and zero accuracy loss reads {free_wide:.1f} "
        f"against our {ours:.1f} tok/s at k=7; a win here reopens #22"
    )
    print(f"spec: verify_lens OK {lens}; rung 8 verify {VERIFY_MS[8]:.1f} ms > our whole "
          f"k=3 tick {VERIFY_MS[4] + 3 * DRAFT_MS:.1f} ms, so a free-forward block head "
          f"at k=7 reads {free_wide / ours:.3f}x")

    # The reject above has ONE free variable, and it is acceptance -- which this repo
    # keeps measuring 1.4x apart between prompts. At decay=1 the two sides share every
    # tick constant, so `free_wide < ours` reduces to
    #   (1 + p^4) * (VERIFY_MS[4] + 3*DRAFT_MS) / (VERIFY_MS[8] + DRAFT_MS) < 1
    # -- the ladder, the decay model and DSpark's parameter count all cancel. Solve it
    # and the reject holds only while p < 0.781. That bound sat in a scratch calculation
    # and nowhere in the tree, so a future prompt reading tok/fwd 2.9 would pass both
    # asserts above while the sentence they encode ("the width is unaffordable") had
    # stopped being true. Asserted here, at the p the constants were derived from.
    #
    # Measured, same harness, same randint(vocab, seed=1000) prompt: ctx 1024-4096 read
    # tok/fwd 2.03-2.10 (p 0.576-0.593) and ctx 8192 read 2.89 -- p 0.786, PAST the
    # bound, where a free-forward k=7 reads 1.007x. Whether that is context or the
    # draft-KV fix in b9af605 is task #67 and is not settled here; what is settled is
    # that the margin is one prompt wide.
    p_bound = ((VERIFY_MS[8] + DRAFT_MS) / (VERIFY_MS[4] + 3 * DRAFT_MS) - 1) ** 0.25
    assert p_bound > 0.6536, (
        f"the p={0.6536} these constants were derived from is at or past the reject's own "
        f"bound {p_bound:.4f} -- (1+p^4)*tick_ratio has reached 1, so a free-forward block "
        "head at k=7 no longer loses and #22 must be re-derived at the measured acceptance"
    )
    print(f"spec: the reject holds while acceptance p < {p_bound:.4f}; derived at "
          f"p=0.6536 (tok/fwd 2.36, wikitext), and ctx 8192 measured p=0.786 (#67)")

    # The same arithmetic at B=4, the SERVING batch, where it comes out worse. Measured
    # on ticks bucketed by their own M (scripts/ab_draft_depth.py --batch 4): rung 32
    # verify derives to 170.03 ms independently at depth 2 (52 ticks) and depth 3 (46),
    # and 169.47 at depth 4 (47) -- three depths with 12/16/20 useful rows agreeing to
    # 0.33%, which is the rung thesis with no cross-batch subtraction in it. One draft
    # forward there is 10.36 ms, so drafting is 15% of a rung-32 tick against 19-24% of
    # a rung-4 tick at B=1.
    #
    # A block head replaces k forwards with one, so a SMALLER draft share is a LOWER
    # ceiling: 1.115x here against 1.16-1.21x at B=1, and the acceptance it must retain
    # rises from 82.7% to 89.7%. Batching makes this arm harder, not easier, because the
    # verify launch it cannot shrink grows as a share of the tick. Asserted so a future
    # change that makes drafting cheaper cannot quietly revive the arm without moving
    # this number too. wins/2026-09-04-rung-cost-not-useful-rows.md
    B4_VERIFY_32_MS, B4_DRAFT_MS, B4_K = 170.03, 10.36, 3
    b4_tick = B4_VERIFY_32_MS + B4_K * B4_DRAFT_MS
    b4_ceiling = b4_tick / (B4_VERIFY_32_MS + B4_DRAFT_MS)
    assert b4_ceiling < 1.16, (
        f"B=4's block-parallel ceiling {b4_ceiling:.3f}x must stay below B=1's 1.16x: "
        "a bigger batch spends more of the tick on the verify launch a block head "
        "cannot remove"
    )
    assert 1 / b4_ceiling > 0.86, (
        f"break-even retention {1 / b4_ceiling:.3f} -- the arm needs the parallel head "
        "to keep this share of tok/forward, and at B=4 it is stricter than B=1's 0.827"
    )
    print(f"spec: B=4 block-parallel ceiling {b4_ceiling:.3f}x, "
          f"break-even retention {1 / b4_ceiling:.1%}")


# Chain MTP head. It loses to the block drafter on GPU-bound serving builds
# (H20 d1-d3: 0.986x/0.863x/0.572x, wins/2026-09-09-spec-decode-h20-b1-loses-on-the-serving-build.md)
# but wins 1.76-1.82x on the eager build, where a ~56 ms/tick fixed host cost
# dominates and the chain amortizes it. sm70 is host-bound the same way; do not
# delete without sm70 numbers showing it loses there too.
class DraftHead:
    """NextN / DSpark draft head: ``fc([norm(embed(t)), norm(h_trunk)])`` into a
    short full-attention stack, read out through the trunk's lm_head. The layers
    are a ``Model`` with a 1-layer config, not a second transformer block."""

    #: Drafter contract, shared with ``dflash2.DFlash2Head`` and read by the engine.
    #: ``aux_layers`` are trunk layers whose output the head taps ( () = none, so the
    #: head serves behind a prefix cache); ``width`` is the verify tick's width,
    #: 1 committed token + width-1 drafts; ``no_quant`` stays out of the fp8 serve.
    aux_layers: tuple[int, ...] = ()
    no_quant: tuple[str, ...] = ()

    def set_depth(self, depth: int | None) -> None:
        """Apply the caller's ``spec_depth``; None keeps the head's own. Idempotent."""
        if depth is not None:
            self.width = depth + 1

    def attach(self, backend, num_blocks: int, dtype=None) -> None:
        """The draft KV plane spans the trunk's whole block space, so the head attends
        over the same prefix the trunk does (a chain-local block dropped acceptance
        from 84.4% to 55.8%). ``dtype`` mirrors the trunk pool: the pool dtype IS the
        attention kernel's ABI, and sm70 runs f32 IO."""
        from .kv_cache import PagedKvPool

        self.backend = backend
        kw = {} if dtype is None else {"dtype": dtype}
        self.kv = PagedKvPool(num_blocks, self.cfg.num_kv_heads, self.cfg.head_dim,
                              num_layers=self.cfg.num_layers, device=backend.device,
                              layer_map=tuple(range(self.cfg.num_layers)), **kw)

    def __init__(self, trunk: Any, params: dict[str, torch.Tensor], num_layers: int = 1) -> None:
        from .model import Model

        self.trunk = trunk
        self.params = params
        cfg = replace(
            trunk.cfg, num_layers=num_layers, full_attn_layers=tuple(range(num_layers)), fp4=False
        )
        self.cfg = cfg
        self.layers = Model(cfg, params)
        self.has_confidence = "confidence.weight" in params
        self.width = 3  # 2 drafts; ``set_depth`` overrides
        self.forwards = 0  # cumulative draft forwards; a probe divides its own timing by this

    def forward(self, hidden, ids, positions, kv, backend, hidden_out=None,
                last_only=False) -> torch.Tensor:
        """hidden [B,T,H] (trunk's pre-final-norm state), ids [B,T] (the token
        each position predicts FROM) -> draft logits [B,T,vocab], or [B,1,vocab]
        when ``last_only`` selects one position per row. ``hidden_out`` receives
        the head's own hidden at FULL width, appended before the reduction."""
        self.forwards += 1  # a tick runs 1..depth of these: the chain loop can break early
        eps = self.cfg.rms_eps
        ids = torch.as_tensor(ids, dtype=torch.long, device=backend.device)
        positions = torch.as_tensor(positions, dtype=torch.long, device=backend.device)
        e = backend.embedding(ids, self.trunk.params["embed_tokens"])
        if "pre_fc_norm_embedding" in self.params:  # Qwen NextN: both sides normed
            e = backend.rmsnorm(e, self.params["pre_fc_norm_embedding"], eps)
        hidden = backend.rmsnorm(hidden, self.params["pre_fc_norm_hidden"], eps)
        # embed first (agent-infer qwen35_spec.rs:40-55); the other order does not predict
        x = self.layers._linear(backend, torch.cat([e, hidden], dim=-1), "fc")
        for i in range(self.cfg.num_layers):
            x = self.layers._full_attn(i, x, positions, kv, backend)
            x = self.layers._mlp(i, x, kv, backend)
        if hidden_out is not None:
            hidden_out.append(x)
        # Same trade the trunk makes (model.py:371): a vocab-wide readout over every
        # prefill position is thrown away one line later. Here it OOMed a 32 GB card --
        # 1.41 GiB at B=8 ctx=512, of which 8 rows (7.6 MiB) were read.
        if last_only is not False and x.shape[1] > 1:
            idx = (torch.full((x.shape[0],), x.shape[1] - 1, device=backend.device)
                   if last_only is True
                   else torch.as_tensor([n - 1 for n in last_only], device=backend.device))
            x = x[torch.arange(x.shape[0], device=backend.device), idx].unsqueeze(1)
        x = backend.rmsnorm(x, self.params["norm"], eps)
        head = self.trunk.cfg.head_key
        return self.trunk._linear(backend, x, head)

    def confidence(self, hidden, probs, backend) -> torch.Tensor:
        """Per-position P(accept), [B,T]: the checkpoint's head, else ``probs``."""
        if not self.has_confidence:
            return probs
        y = backend.linear(hidden, self.params["confidence.weight"],
                           bias=self.params.get("confidence.bias"))
        return torch.sigmoid(y).reshape(y.shape[:-1])

    def step(self, rows) -> None:
        """Contract: leave next tick's chain in ``r.drafts``.

        Draft over every position a row materialized but has not drafted yet:
        position q consumes the trunk hidden at q-1 and the token at q, so the
        run spans ``[draft_pos+1 .. seq_len-1]`` and its last position drafts
        the next token. Leaves next tick's chain in ``r.drafts``."""
        backend = self.backend
        dev = backend.device
        plan = []
        for r in rows:
            if r.hidden is None or r.done:
                continue
            lo, hi = max(1, r.draft_pos + 1), r.seq_len - 1
            # Position q needs the trunk hidden at q-1, and the engine keeps only the
            # LAST forward's hidden (engine.py:747) plus one previous position. A row
            # that advanced several chunked-prefill ticks without drafting therefore has
            # no hidden for its early positions: at ctx=2048 one reached seq_len 1536
            # with draft_pos 0, asked for 1535 positions, and got the 512 it actually
            # had -- `pad` then added w-q=0 and the fc concat died on 1535 vs 511, three
            # frames from the cause. Drop the unbacked positions instead of slicing past
            # the buffer: the last position is what leaves a chain in r.drafts, and it is
            # always inside the newest hidden.
            base = r.hidden_from - (r.hidden_prev is not None)
            lo = max(lo, base + 1)
            if hi < lo:
                continue
            # Sparse+spec uses the draft's own DENSE blocks; dense mode reuses trunk.
            dblocks = r.draft_blocks if r.draft_blocks else r.blocks
            # A block shortfall here is silent: the write lands on the wrong page and the
            # next position attends over garbage. Both engine paths reason to this
            # separately; this is the one place that knows `hi`.
            assert len(dblocks) * BLOCK_TOKENS > hi, (
                f"draft would write position {hi} but the row owns {len(dblocks)} "
                f"blocks x {BLOCK_TOKENS} = {len(dblocks) * BLOCK_TOKENS} positions "
                f"(seq_len={r.seq_len}, draft_pos={r.draft_pos})")
            plan.append((r, lo, hi, dblocks))
        if not plan:
            return
        # Bucket the draft's prefill width the way the trunk does (engine.py:741).
        # Unbucketed, every distinct prompt length gave the draft a new (n, w) and
        # recompiled the two seq_q_lens kernels: a served first visit at a new prompt
        # length paid 14 compiles / 15.5 s inline, 4.4 tok/s against 45.0 on the
        # repeat. The padding rows are free of correctness risk because the kernels
        # already gate on SeqQLens (kernels_mma.py:71), and `sq` below stays exact.
        w = max(hi - lo + 1 for _, lo, hi, _ in plan)
        if w > 1:
            w = -(-w // _PREFILL_BUCKET) * _PREFILL_BUCKET
        # Table width = pool size, for the same reason as engine.py:666 -- the kernels
        # compile Mb in, so a per-tick width recompiles. `max(len(r.blocks))` grows one
        # column per BLOCK_TOKENS of context, so it was a second shape axis after the
        # width: measured on the served path, bucketing w alone cut a new prompt length
        # from 14 compiles to 4-8, all at S=64 with block tables [1,1]/[1,3]/[1,4]/[1,5]/[1,6].
        nb = self.kv.num_blocks
        n = len(plan)
        ids = np.zeros((n, w), dtype=np.int64)
        pos = np.zeros((n, w), dtype=np.int64)
        bt = torch.zeros(n, nb, dtype=torch.long)
        hs, sl, sq = [], [], []
        for i, (r, lo, hi, dblocks) in enumerate(plan):
            q = hi - lo + 1
            ids[i, :q] = r.tokens[lo : hi + 1]
            pos[i, :q] = np.arange(lo, hi + 1)
            bt[i, : len(dblocks)] = torch.tensor(dblocks, dtype=torch.long)
            sl.append(hi + 1)
            sq.append(q)
            # hidden at [lo-1 .. hi-1]; hidden_prev supplies the previous forward's position
            h, base = r.hidden, r.hidden_from
            if r.hidden_prev is not None:
                h, base = torch.cat([r.hidden_prev, r.hidden], dim=1), base - 1
            off = (lo - 1) - base
            hq = h[:, off : off + q]
            # The clamp above makes this exact; assert it rather than let a short slice
            # reach the fc concat, where the shapes name neither the row nor the cause.
            assert hq.shape[1] == q, (
                f"draft hidden for [{lo},{hi}] is {hq.shape[1]} of {q} positions "
                f"(hidden_from={r.hidden_from}, width={h.shape[1]}, off={off})"
            )
            hs.append(torch.nn.functional.pad(hq, (0, 0, 0, w - q)))
        kv = BatchKv(
            block_table=bt.to(dev), seq_len=torch.tensor(sl, device=dev),
            state_slot=torch.zeros(n, dtype=torch.long, device=dev),
            kv_pool=self.kv, state_pool=None,
            seq_q_lens=torch.tensor(sq, device=dev),
        )
        dh: list = []
        # last_only, or the vocab readout runs over every position of a prefill chunk and
        # one row is read: 512 x 248320 f32 = 485 MiB, which is the allocation that OOMed
        # ctx=8192 at a 4146-block pool. hidden_out is appended at full width, so `last`
        # still indexes dh.
        logits = self.forward(torch.cat(hs, dim=0), ids, pos, kv, backend,
                              hidden_out=dh, last_only=sq)
        last = torch.tensor([q - 1 for q in sq], device=dev)
        rng = torch.arange(n, device=dev)
        tok, prob = backend.greedy(logits)
        h = dh[-1][rng, last].unsqueeze(1)
        confs: list[list[float]] = [[] for _ in plan]
        if (self.width - 1) > 1:
            conf = self.confidence(h, prob, backend)
            for i, c in enumerate(conf[:, -1].tolist()):
                confs[i].append(float(c))
        chains = [[int(t)] for t in tok[:, -1].tolist()]
        for i, (r, _, hi, dblocks) in enumerate(plan):
            if r.draft_pos == 0:
                # Position 0 is never drafted but attention still reads its page,
                # which a recycled block leaves holding another request's.
                b = dblocks[0]
                self.kv.k_pool[:, b, :, 0, :] = 0
                self.kv.v_pool[:, b, :, 0, :] = 0
            r.draft_pos = hi

        # Remaining chain steps, one position each, bounded by the blocks the row owns.
        # ponytail: clamps the chain instead of allocating; a row at a block boundary drafts shorter.
        for j in range(1, (self.width - 1)):
            live = [i for i, (r, _, hi, dblocks) in enumerate(plan)
                    if hi + j < len(dblocks) * BLOCK_TOKENS]
            if not live:
                break
            li = torch.tensor(live, device=dev)
            kv = BatchKv(
                block_table=bt[live].to(dev),
                seq_len=torch.tensor([plan[i][2] + 1 + j for i in live], device=dev),
                state_slot=torch.zeros(len(live), dtype=torch.long, device=dev),
                kv_pool=self.kv, state_pool=None,
                seq_q_lens=torch.ones(len(live), dtype=torch.long, device=dev),
            )
            dh = []
            logits = self.forward(
                h[li], np.array([[chains[i][-1]] for i in live], dtype=np.int64),
                np.array([[plan[i][2] + j] for i in live], dtype=np.int64),
                kv, backend, hidden_out=dh,
            )
            tok, prob = backend.greedy(logits)
            conf = self.confidence(dh[-1], prob, backend)
            for k, c in enumerate(conf[:, -1].tolist()):
                confs[live[k]].append(float(c))
            for k, t in enumerate(tok[:, -1].tolist()):
                chains[live[k]].append(int(t))
            h = h.index_copy(0, li, dh[-1])

        keep = verify_lens([survival(c) for c in confs]) if (self.width - 1) > 1 \
            else [1] * len(plan)
        for i, (r, *_) in enumerate(plan):
            p = r.params
            if p.max_think_tokens is not None and p.end_think_ids and not r.thought_closed:
                keep[i] = 0  # a forced end-think token is not the sampler's
            r.drafts = chains[i][: keep[i]]


#: Draft tensor stems -> param keys, matched after any ``layers.N.`` prefix.
#: Qwen NextN prefixes ``mtp.``; DSpark drops it and carries one ``hidden_norm``
#: instead of the two pre-fc norms.
_DRAFT_TOP = {
    "fc": "fc",
    "norm": "norm",
    "hidden_norm": "pre_fc_norm_hidden",
    "pre_fc_norm_hidden": "pre_fc_norm_hidden",
    "pre_fc_norm_embedding": "pre_fc_norm_embedding",
    "confidence_head.proj": "confidence",
}


def _split_layer(stem: str) -> tuple[str, str]:
    """``layers.3.mlp_conv.base_kernel`` -> ``("layers.3.", "mlp_conv.base_kernel")``."""
    if stem.startswith("layers."):
        idx, sep, tail = stem[len("layers.") :].partition(".")
        if sep and idx.isdigit():
            return f"layers.{int(idx)}.", tail
    return "", stem


def read_head_params(path: str | Path, stems: dict[str, str]) -> dict[str, torch.Tensor]:
    """One draft-head safetensors -> param keys: ``stems`` names the head's own
    tensors, ``_param_key_for`` the ordinary Qwen3 layer ones."""
    from safetensors import safe_open

    from .model import _is_lm_head, _param_key_for

    params: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    unknown: list[str] = []
    nextn = False
    with safe_open(str(path), "pt", device="cpu") as f:
        names = list(f.keys())
        for name in names:
            bare = name.removeprefix("mtp.").removeprefix("model.")
            stem = bare.removesuffix(".weight").removesuffix(".bias")
            nextn |= stem == "pre_fc_norm_hidden"
            prefix, tail = _split_layer(stem)
            key = stems.get(tail)
            if key is not None:
                if key == "confidence":  # the only head tensor with a bias
                    key += ".bias" if bare.endswith(".bias") else ".weight"
                params[prefix + key] = f.get_tensor(name)
                continue
            mapped = _param_key_for(bare)
            # forward reads the embedding and the readout off the TRUNK, so a head
            # shipping its own is dead weight — and engine._quantize_draft packs
            # anything 2-D, which at 248320x5120 is 2.5 GB on a card that has OOMed.
            # _is_lm_head, not `mapped == "lm_head"`: a QUANTIZED readout arrives as
            # three tensors (lm_head.wq/scale/oscale) that _param_key_for cannot name,
            # so matching on the mapped key sent all three to `unknown` and made the
            # 27B NVFP4 draft shard unloadable -- the one shard we actually serve.
            if _is_lm_head(bare) or mapped in ("embed_tokens", "final_norm"):
                skipped.append(bare)
            elif mapped is not None:
                params[mapped] = f.get_tensor(name)
            else:
                unknown.append(bare)
    if skipped:
        warnings.warn(
            f"draft head {path}: ignoring {sorted(skipped)} — the trunk's are shared",
            stacklevel=2,
        )
    # A tensor this map does not name is the wrong reader for this checkpoint, not
    # dead weight: loading a DFlash2 head through _DRAFT_TOP drops all 11 of its
    # conv and selector weights, and the first draft then dies on a KeyError far
    # from the cause.
    if unknown:
        raise RuntimeError(
            f"draft head {path}: {len(unknown)} tensor(s) map to no parameter — "
            f"{sorted(unknown)[:8]}{'...' if len(unknown) > 8 else ''}. Wrong head "
            "format for this reader, or a key this port does not implement."
        )
    # Zero-centered Qwen3_5RMSNorm (y = x*(1+w)): load_hf folds the +1 in for the
    # trunk, and only a Qwen NextN head is built that way. DSpark and DFlash norms
    # are plain w*x — agent-infer's dspark.rs:580,726, and vLLM/sglang build every
    # DFlash norm from their stock RMSNorm. Keying the fold on the one format that
    # needs it makes no-fold the default, which is the safe way round: the missing
    # fold is loud (the head's argmax ranked 248191/248320), the spurious one is not.
    if nextn:
        for k, v in params.items():
            if k.endswith(("norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding")):
                params[k] = (v.float() + 1.0).to(v.dtype)
    return params


def load_draft(trunk: Any, path: str | Path) -> Any:
    """Load a draft head from one safetensors file beside the trunk: a Qwen
    NextN / DSpark chain head, or the DFlash2 block drafter. A checkpoint
    directory resolves to its ``model.safetensors``; mmapping the directory
    itself raises a bare ``OSError: No such device``, which names nothing."""
    from safetensors import safe_open

    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"draft head: {path.parent} holds no model.safetensors")
    with safe_open(str(path), "pt", device="cpu") as f:
        if any(n.startswith("candidate_selector.") for n in list(f.keys())):

            return load_dflash2(trunk, path)
    params = read_head_params(path, _DRAFT_TOP)
    missing = {"fc", "norm", "pre_fc_norm_hidden"} - set(params)
    if missing:
        raise RuntimeError(f"draft head {path}: missing {sorted(missing)}")
    # Indices must be 0..n-1: an absolute-index convention (DeepSeek numbers its MTP
    # layer by its position in the trunk) would otherwise infer a depth of index+1
    # and fail later on a missing layers.0, pointing at the wrong thing.
    idx = sorted({int(k.split(".")[1]) for k in params if k.startswith("layers.")})
    if idx and idx != list(range(len(idx))):
        raise RuntimeError(f"draft head {path}: layers indexed {idx}, expected 0..{len(idx) - 1}")
    return DraftHead(trunk, params, num_layers=len(idx) or 1)


# ---------------------------------------------------------------------------
# DFlash2 block drafter (merged from dflash2.py): one pass proposes a block.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DFlash2Config:
    """``config.json``'s ``dflash_config``, plus the window the draft attends over."""

    block_size: int
    taps: int
    group_size: int
    mask_token_id: int
    rank: int
    top_k: int
    #: trunk layers whose hidden states ``fc`` consumes, in concat order
    target_layers: tuple[int, ...]
    sliding_window: int


#: DFlash2 tensor stems -> param keys (see :func:`tilerl.spec.read_head_params`).
_DFLASH2_TOP = {
    "fc": "fc",
    "norm": "norm",
    "hidden_norm": "hidden_norm",
    "candidate_selector.hidden_projection": "selector.proj",
    "candidate_selector.predecessor_codebook": "selector.pred",
    "candidate_selector.successor_codebook": "selector.succ",
    "attention_conv.base_kernel": "attn_conv.base",
    "attention_conv.kernel_projection": "attn_conv.proj",
    "mlp_conv.base_kernel": "mlp_conv.base",
    "mlp_conv.kernel_projection": "mlp_conv.proj",
}

#: Knobs this port does not implement; each would change the draft with no error.
#: Qwen3.8-27B-DFlash2 sets none, so the raise is the whole implementation.
_UNSUPPORTED = (
    "input_embedding_scale",
    "output_multiplier",
    "final_logit_softcapping",
    "attention_sink_bias",
    "add_swa_attention_sink_bias",
)


def _attend(q, k, v, q_pos, k_pos, window):
    """Non-causal GQA: every block slot sees every other and every context key
    inside the window. ponytail: torch-eager, tilelang once the block draft is
    on the hot path."""
    d, group = q.shape[-1], q.shape[2] // k.shape[2]
    k, v = k.repeat_interleave(group, 2).float(), v.repeat_interleave(group, 2).float()
    att = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) / math.sqrt(d)
    att = att.masked_fill(q_pos[:, None] - k_pos[None, :] >= window, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(att, -1), v)


class DFlash2Head:
    """``params`` holds the head's own weights only: the embedding and the
    readout are the trunk's, and the block's K/V never outlives one draft."""

    #: codebooks the selector GATHERS rows from, not linears: block-quantizing
    #: them would replace [248320,256] with a .w8 the walk cannot index.
    no_quant = ("selector.pred", "selector.succ")

    def __init__(self, trunk: Any, params: dict[str, torch.Tensor], cfg, dcfg) -> None:
        self.trunk, self.params, self.cfg, self.dcfg = trunk, params, cfg, dcfg
        self.groups = cfg.hidden_size // dcfg.group_size
        #: the engine re-serves the head fp8; a Model gives the head the same
        #: .w8/.wscale dispatch the trunk has, over THIS params dict.
        self.layers = Model(cfg, params)
        #: the block is fixed by the checkpoint: anchor + block_size-1 drafts
        self.width = dcfg.block_size  # verify tick: anchor + block_size-1 drafts
        self.aux_layers = dcfg.target_layers

    def set_depth(self, depth: int | None) -> None:
        """The block is the checkpoint's, so ``spec_depth`` is not the caller's to choose."""
        if depth is not None and depth != self.width - 1:
            raise ValueError(
                f"spec_depth={depth} with a block drafter: the checkpoint's block is "
                f"{self.width} slots (anchor + {self.width - 1} drafts), so the verify "
                f"width is fixed. Pass spec_depth=None."
            )

    def attach(self, backend, num_blocks: int, dtype=None) -> None:
        """No KV plane of its own: the context K/V is projected from the trunk's aux taps,
        so ``dtype`` (the trunk pool's, part of the contract) has nothing to size here."""
        self.backend = backend

    def step(self, rows) -> None:
        """Leave next tick's chain in ``r.drafts``. A context position's K/V is a
        pure projection of the trunk's aux taps, so it is projected the tick it
        commits and kept — re-projecting the whole context every tick is ~10 ms at
        T=512 B=8. The walk runs once for the tick; ``block_hidden`` is still per
        row, which is the rest of the batching ceiling."""
        backend = self.backend
        dev = backend.device
        live, hidden, anchors = [], [], []
        for r in rows:
            if r.aux is None or r.done:
                continue
            # The context is every position but the anchor; a row still prefilling has none.
            end = r.seq_len if r.prefilling else r.seq_len - 1
            lo, base = r.ctx_len, r.hidden_from
            if lo < base:
                raise RuntimeError(
                    f"req {r.req_id}: context K/V has a hole at {lo}..{base} — the trunk "
                    f"never forwarded those positions this process"
                )
            if end > lo:
                pos = torch.arange(lo, end, device=dev)
                new = self.context_kv(r.aux[:, lo - base : end - base], pos, backend)
                r.ctx = new if r.ctx is None else [
                    (torch.cat([k, nk], 1), torch.cat([v, nv], 1))
                    for (k, v), (nk, nv) in zip(r.ctx, new)
                ]
                w = self.dcfg.sliding_window
                if r.ctx[0][0].shape[1] > w:
                    r.ctx = [(k[:, -w:], v[:, -w:]) for k, v in r.ctx]
                r.ctx_len = end
            if not r.decoding or r.ctx is None:
                continue
            n = r.ctx[0][0].shape[1]
            ctx_pos = torch.arange(r.ctx_len - n, r.ctx_len, device=dev)
            anchor = r.tokens[-1]
            live.append(r)
            anchors.append(anchor)
            hidden.append(self.block_hidden(r.ctx, ctx_pos, anchor, r.ctx_len, backend))
        if live:
            walked = self.paths(torch.cat(hidden)[:, 1:], anchors, backend)
            for r, drafts in zip(live, walked):
                r.drafts = drafts


    def context_kv(self, aux_hidden, positions, backend) -> list[tuple]:
        """Per-layer (k, v) for the context, straight off the trunk's stacked taps."""
        cfg = self.cfg
        h = backend.rmsnorm(
            self._lin(backend, aux_hidden, "fc"), self.params["hidden_norm"], cfg.rms_eps
        )
        out = []
        for i in range(cfg.num_layers):
            k = self._heads(backend, h, f"layers.{i}.k_proj", cfg.num_kv_heads)
            k = backend.rmsnorm(k, self.params[f"layers.{i}.k_norm"], cfg.rms_eps)
            v = self._heads(backend, h, f"layers.{i}.v_proj", cfg.num_kv_heads)
            out.append((self._rope(backend, k, positions), v))
        return out

    def block_hidden(self, ctx, ctx_pos, anchor, start, backend) -> torch.Tensor:
        """The block's post-norm hidden ``[1, block_size, H]``: slot 0 is the
        anchor the trunk already committed, the rest are mask slots."""
        cfg, dc = self.cfg, self.dcfg
        dev = backend.device
        ids = torch.full((1, dc.block_size), dc.mask_token_id, dtype=torch.long, device=dev)
        ids[0, 0] = anchor
        pos = torch.arange(start, start + dc.block_size, device=dev)
        x = backend.embedding(ids, self.trunk.params["embed_tokens"])
        for i in range(cfg.num_layers):
            p = f"layers.{i}"
            h, c = self._conv_in(backend, x, f"{p}.input_norm", f"{p}.attn_conv")
            h = self._attn(backend, i, h, pos, ctx[i], ctx_pos)
            x = backend.add(x, self._conv(h, c, self.params[f"{p}.attn_conv.base"][1]))
            h, c = self._conv_in(backend, x, f"{p}.post_attn_norm", f"{p}.mlp_conv")
            h = backend.silu_mul(
                self._lin(backend, h, f"{p}.gate_proj"),
                self._lin(backend, h, f"{p}.up_proj"),
            )
            h = self._lin(backend, h, f"{p}.down_proj")
            x = backend.add(x, self._conv(h, c, self.params[f"{p}.mlp_conv.base"][1]))
        return backend.rmsnorm(x, self.params["norm"], cfg.rms_eps)

    def path(self, hidden, anchor, backend) -> list[int]:
        return self.paths(hidden, [anchor], backend)[0]

    def paths(self, hidden, anchors, backend) -> list[list[int]]:
        """Top-k per slot, then one greedy walk per row. ``prev`` feeds the next
        slot so the block is sequential; the rows are not, so they walk together
        and the batch costs one host sync, not two per slot per row."""
        dc = self.dcfg
        head = self.trunk.cfg.head_key
        # topk before the widening cast: it is order-preserving, and an f32 copy of
        # the full [B, W, 248320] readout is 55 MB at B=8
        unary, cand = torch.topk(self.trunk._linear(backend, hidden, head), dc.top_k, dim=-1)
        proj = self._lin(backend, hidden, "selector.proj").float()
        # gathered rows only: an f32 copy of either [248320,256] codebook is 254 MB
        pred, succ = self.params["selector.pred"], self.params["selector.succ"]
        prev = torch.as_tensor(anchors, dtype=torch.long, device=hidden.device)
        rows = torch.arange(len(anchors), device=hidden.device)
        out = []
        for j in range(hidden.shape[1]):
            score = unary[:, j].float() + torch.einsum(
                "bkr,br->bk", succ[cand[:, j]].float(), pred[prev].float() * proj[:, j]
            )
            prev = cand[rows, j, score.argmax(-1)]
            out.append(prev)
        return torch.stack(out, 1).tolist()

    def draft(self, aux_hidden, positions, anchor, backend) -> list[int]:
        """``aux_hidden`` [1,T,len(target_layers)*H] concatenated in
        ``target_layers`` order and ``positions`` [T] describe the context;
        returns ``block_size - 1`` tokens continuing from ``anchor``, the token
        the trunk committed at ``positions[-1] + 1``."""
        pos = torch.as_tensor(positions, dtype=torch.long, device=backend.device)
        ctx = self.context_kv(aux_hidden, pos, backend)
        h = self.block_hidden(ctx, pos, anchor, int(pos[-1]) + 1, backend)
        return self.path(h[:, 1:], anchor, backend)

    # --- pieces -------------------------------------------------------------
    def _lin(self, backend, x, key):
        return self.layers._linear(backend, x, key)

    def _heads(self, backend, h, key, heads):
        return self._lin(backend, h, key).reshape(*h.shape[:2], heads, self.cfg.head_dim)

    def _rope(self, backend, x, positions):
        cfg = self.cfg
        return backend.rope(x, positions, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)

    def _attn(self, backend, i, h, pos, ctx, ctx_pos):
        cfg, p = self.cfg, f"layers.{i}"
        q = self._heads(backend, h, f"{p}.q_proj", cfg.num_attention_heads)
        k = self._heads(backend, h, f"{p}.k_proj", cfg.num_kv_heads)
        v = self._heads(backend, h, f"{p}.v_proj", cfg.num_kv_heads)
        q = self._rope(backend, backend.rmsnorm(q, self.params[f"{p}.q_norm"], cfg.rms_eps), pos)
        k = self._rope(backend, backend.rmsnorm(k, self.params[f"{p}.k_norm"], cfg.rms_eps), pos)
        k, v = torch.cat([ctx[0], k], 1), torch.cat([ctx[1], v], 1)
        out = _attend(q, k, v, pos, torch.cat([ctx_pos, pos]), self.dcfg.sliding_window)
        return self._lin(backend, out.reshape(*h.shape[:2], -1), f"{p}.o_proj")

    def _conv_in(self, backend, x, norm_key, conv_key):
        """Norm, then the conv before the op — and the coefficients its partner
        after the op consumes. One projection yields both sides' taps."""
        h = backend.rmsnorm(x, self.params[norm_key], self.cfg.rms_eps)
        c = self._lin(backend, h, f"{conv_key}.proj").reshape(
            *h.shape[:2], 2, self.dcfg.taps, self.groups
        )
        return self._conv(h, c[:, :, 0], self.params[f"{conv_key}.base"][0]), c[:, :, 1]

    def _conv(self, x, delta, base):
        """Two-tap grouped depthwise conv along the block. Coefficients are
        per-token; a tap reaching before the block start is zero-padded, so slot
        0 carries only its own term and no earlier block leaks in."""
        dc, g = self.dcfg, self.groups
        blocks = x.reshape(*x.shape[:2], g, dc.group_size).float()
        coef = base.float().reshape(1, 1, dc.taps, g, dc.group_size) + delta.float().unsqueeze(-1)
        out = coef[:, :, 0] * blocks
        for tap in range(1, dc.taps):
            out[:, tap:] += coef[:, tap:, tap] * blocks[:, :-tap]
        return out.reshape(*x.shape[:2], -1)


def load_dflash2(trunk: Any, path: str | Path) -> DFlash2Head:
    """Load the block drafter from its safetensors and the config.json beside
    it. Its shapes are not the trunk's: 32/8 heads of 128, no attention gate,
    full RoPE."""
    path = Path(path)
    hf = json.loads((path.parent / "config.json").read_text())
    d = hf["dflash_config"]
    knobs = {**hf, **d}
    unsupported = [k for k in _UNSUPPORTED if knobs.get(k)]
    if unsupported:
        raise RuntimeError(f"DFlash2 head {path}: unimplemented {sorted(unsupported)}")
    if hf.get("is_causal") is not False:
        raise RuntimeError(f"DFlash2 head {path}: only non-causal block attention is implemented")
    n = hf["num_hidden_layers"]
    cfg = ModelConfig(
        name="dflash2",
        hidden_size=hf["hidden_size"],
        intermediate_size=hf["intermediate_size"],
        num_layers=n,
        num_attention_heads=hf["num_attention_heads"],
        num_kv_heads=hf["num_key_value_heads"],
        head_dim=hf["head_dim"],
        vocab_size=hf["vocab_size"],
        full_attn_layers=tuple(range(n)),
        rope_theta=hf["rope_parameters"]["rope_theta"],
        max_position_embeddings=hf["max_position_embeddings"],
        rms_eps=hf["rms_norm_eps"],
        tie_word_embeddings=hf["tie_word_embeddings"],
        fp4=False,
        full_attn_gated=False,
    )
    dcfg = DFlash2Config(
        block_size=d["block_size"],
        taps=d["conv_kernel_size"],
        group_size=d["conv_group_size"],
        mask_token_id=d["mask_token_id"],
        rank=d["selector_rank"],
        top_k=d["selector_top_k"],
        target_layers=tuple(d["target_layer_ids"]),
        sliding_window=hf["sliding_window"],
    )
    return DFlash2Head(trunk, read_head_params(path, _DFLASH2_TOP), cfg, dcfg)
