"""Speculative decoding: the draft head and the verify-length policy.

``verify_lens`` decides how many drafted tokens per request are worth verifying
this tick (DSpark §3.2.2, sglang's ``compute_verify_token_budget``): a draft
costs a trunk row whether or not it is accepted, so maximize goodput
``(R + Σ top-B survival) / (bias + row·(R + B))`` over the admission cut. B=0
is one of the arms. ``survival[j]`` = P(the first j+1 drafts all accept).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .kv_cache import BLOCK_TOKENS, BatchKv

#: One trunk verify forward = fixed + per-row cost, ms. agent-infer's H20 numbers.
#: sm70 is a staircase, not a line: the GEMV ladder rounds verify width up to a
#: rung, so at ctx 1024 verify costs w<=2 32.79, w<=4 49.52, w<=8 86.24 ms, one
#: draft forward 3.93 (errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md,
#: errors/2026-09-03-block-parallel-drafting-is-1.016x-on-sm70.md). Measured on
#: ticks bucketed by their OWN rung, because verify_lens trims per tick and a
#: configured depth runs a mixture. Those components rebuild the end-to-end tick to
#: within 10% at W=2/4/8, so depth moves TWO terms: one more draft forward (3.93 ms,
#: flat) plus a wider verify -- do not price it as one.
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
    # Measured staircase, rung -> ms, and one draft forward. Both come from ticks
    # BUCKETED BY THEIR OWN RUNG at ctx=1024, B=1 (scripts/ab_draft_depth.py):
    # verify_lens trims per tick, so a configured depth runs a MIXTURE of rungs,
    # and subtracting two depths' mean ticks moves part of that mixture across a
    # rung step and charges it to the draft. The rung-4 verify cost derives to
    # 49.52 ms independently at depth 2 (56 ticks) and depth 3 (55) -- same rung,
    # same cost, which is what the subtraction assumes.
    # The previous values (VERIFY {2: 36.58, 4: 49.87, 8: 68.46}, DRAFT 5.53) were
    # taken from each depth's MEAN tick, which mixes rungs: the mean-tick
    # subtraction reads 4.54 ms against this 3.93, and the 0.61 gap is 16% of all
    # ticks crossing the 16.73 ms rung-2 -> rung-4 step.
    VERIFY_MS = {2: 32.79, 4: 49.52, 8: 86.24}
    DRAFT_MS = 3.93
    # The staircase sits 6.8-10.1% BELOW the 0.670 + 0.5265W dense-tick line. That
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
    # The bound is 1.06, not the 1.16% noise floor, because the verdict is NOT
    # prompt-independent in value -- only in form. Lower acceptance favours the
    # parallel head: 0.972x at p=0.881, 1.016x at 0.722, 1.035x at 0.654 (the
    # wikitext p, which is the one set above). The decay model shrinks the suffix
    # geometrically, so as p falls a larger share of the yield sits in positions the
    # parallel head keeps. Below ~1.06 the arm stays inside the ceiling's own
    # assumption that a block head's single forward costs what one of ours does,
    # against a DSpark head of 4.08x the parameters and a 2.36x budget -- so the
    # reject is carried by that gap, not by this margin. An arm ABOVE 1.06 would
    # mean the decay model no longer bounds it and the verdict needs re-deriving,
    # not retuning. errors/2026-09-03-block-parallel-drafting-is-1.016x-on-sm70.md
    ours = _tok_per_fwd(3, decay=1.0) * 1000 / (VERIFY_MS[4] + 3 * DRAFT_MS)
    assert rates[3] / ours < 1.06, (
        f"block-parallel's margin outgrew the decay model's bound: "
        f"{rates[3] / ours:.3f}x of our own {ours:.1f} tok/s at k=3"
    )
    print("spec: verify_lens OK", lens)

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
            # A block shortfall here is silent: the write lands on the wrong page and the
            # next position attends over garbage. Both engine paths reason to this
            # separately; this is the one place that knows `hi`.
            assert len(r.blocks) * BLOCK_TOKENS > hi, (
                f"draft would write position {hi} but the row owns {len(r.blocks)} "
                f"blocks x {BLOCK_TOKENS} = {len(r.blocks) * BLOCK_TOKENS} positions "
                f"(seq_len={r.seq_len}, draft_pos={r.draft_pos})")
            plan.append((r, lo, hi))
        if not plan:
            return
        w = max(hi - lo + 1 for _, lo, hi in plan)
        nb = max(len(r.blocks) for r, _, _ in plan)
        n = len(plan)
        ids = np.zeros((n, w), dtype=np.int64)
        pos = np.zeros((n, w), dtype=np.int64)
        bt = torch.zeros(n, nb, dtype=torch.long)
        hs, sl, sq = [], [], []
        for i, (r, lo, hi) in enumerate(plan):
            q = hi - lo + 1
            ids[i, :q] = r.tokens[lo : hi + 1]
            pos[i, :q] = np.arange(lo, hi + 1)
            bt[i, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.long)
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
        logits = self.forward(torch.cat(hs, dim=0), ids, pos, kv, backend,
                                     hidden_out=dh)
        last = torch.tensor([q - 1 for q in sq], device=dev)
        rng = torch.arange(n, device=dev)
        tok, prob = backend.greedy(logits[rng, last].unsqueeze(1))
        h = dh[-1][rng, last].unsqueeze(1)
        confs: list[list[float]] = [[] for _ in plan]
        if (self.width - 1) > 1:
            conf = self.confidence(h, prob, backend)
            for i, c in enumerate(conf[:, -1].tolist()):
                confs[i].append(float(c))
        chains = [[int(t)] for t in tok[:, -1].tolist()]
        for i, (r, _, hi) in enumerate(plan):
            if r.draft_pos == 0:
                # Position 0 is never drafted but attention still reads its page,
                # which a recycled block leaves holding another request's.
                b = r.blocks[0]
                self.kv.k_pool[:, b, :, 0, :] = 0
                self.kv.v_pool[:, b, :, 0, :] = 0
            r.draft_pos = hi

        # Remaining chain steps, one position each, bounded by the blocks the row owns.
        # ponytail: clamps the chain instead of allocating; a row at a block boundary drafts shorter.
        for j in range(1, (self.width - 1)):
            live = [i for i, (r, _, hi) in enumerate(plan)
                    if hi + j < len(plan[i][0].blocks) * BLOCK_TOKENS]
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
        for i, (r, _, _) in enumerate(plan):
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
            from .dflash2 import load_dflash2

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
