"""PROBE-ONLY #805/#536002: shared teacher-forcing + per-position logits probe.

One instrument for three users: the v2 lag-1 quality gate, fixmisc's
graph-vs-eager (⑤) diagnostic, and impl's R x window sweep. It is installed on
an already-built Engine from a probe script (monkeypatch of engine methods);
the production src is unchanged and nothing here is imported by main.

What it does
------------
`install_teacher_force(engine, anchor)` feeds every sampled position the
ANCHOR token instead of the engine's own draw, while leaving the whole decode
mechanism (sparse graph replay / eager refresh, draft, verify, select_step,
commit) exactly as production runs. The substitution happens on the TOKEN
OUTPUT of Engine._sample_batch — the single chokepoint both the plain greedy
commit and the speculative verify (got[0..W-1]) consume — so:

  * a verify tick compares drafts against anchor next-tokens and accepts a
    draft only when it equals the anchor; n_ok/select_step/GDN adoption are the
    normal code path, so the committed prefix is always a prefix of the anchor;
  * no token is fed through a prefill-style forward — the measured path is the
    real decode graph (plain or refresh) path.

At the same time it records, for each request and each GENERATED position that
is actually committed, the trunk logits the draw came from. For a W>1 verify
tick only the accepted slots got[0..n_ok] are recorded (the rejected draft
slot's logits are dropped). Logits are recorded for every generated position;
the consumer can split the first 16 from the rest and choose its own warm window.

Anchor
------
A dict {request_index: [tokens...]}. The token committed at generated position
p is anchor[p]. The anchor is normally control's free-running greedy stream; a
non-teacher-forced (free-running) arm simply is not installed and records
logits of its own draws. With greedy sampling, teacher-forcing an arm with its
OWN free-run output must be byte-identical (top1 = 1.0) — the CPU self-anchor
gate. Offsetting the anchor by one must go red.

The recorded row: {"i", "logits": full-vocab f32 (cpu), "top1", "anchor",
"committed": actual token}. Consumers compute argmax agreement and symmetric
KL; this module intentionally does NOT set thresholds.
"""

from __future__ import annotations

import torch


class TeacherForceRecorder:
    def __init__(self, engine, anchors, record_full_logits: bool = True):
        # anchors: {request_index: [anchor token per generated position]}.
        # Pass None to RECORD ONLY (the free-running control): logits are
        # captured and no token is forced; its own greedy output is the anchor.
        self.e = engine
        self.anchors = None if anchors is None else \
            {int(k): list(v) for k, v in anchors.items()}
        self.record_full_logits = record_full_logits
        # per request: list of recorded position rows, in commit order
        if anchors is None:
            self.rows = {}
        else:
            self.rows = {int(k): [] for k in self.anchors}
        # pending records from the most recent _sample_batch, keyed by id(req):
        # list of (record, chain_next_token_or_None). _verify marks acceptance.
        self._pending = {}
        self._orig_verify = None
        self._orig_sample_batch = None
        # map an id(_Req)->request index for the rows _sample_batch sees
        self._rid_to_idx = {}

    # ------------------------------------------------------------ install
    def install(self):
        e = self.e
        self._orig_sample_batch = e._sample_batch

        rec = self

        def _patched_sample_batch(rows):
            # Record logits for every request first (record-only control arm
            # creates its bucket on demand), then force the anchor token only
            # when an anchor stream is supplied.
            orig_toks = rec._orig_sample_batch(rows)
            out = list(orig_toks)
            # ensure buckets for record-only mode
            if rec.anchors is None:
                for _r, _l, _g in rows:
                    rec._idx_for(_r)
            # group this batch's rows per request for verify acceptance marking
            batch_by_req = {}
            for slot, (r, lg, gen_idx) in enumerate(rows):
                idx = rec._idx_for(r)
                if idx is None:
                    continue
                anc = rec.anchors[idx] if rec.anchors is not None else None
                forced = None
                if anc is not None and 0 <= gen_idx < len(anc):
                    forced = int(anc[gen_idx])
                    out[slot] = forced
                row = {"gen_idx": int(gen_idx),
                       "out_len_before": len(r.output),
                       "top1": int(lg.argmax().item()) if lg.numel() else None,
                       "drawn": int(orig_toks[slot]),
                       "anchor": (int(anc[gen_idx])
                                  if anc is not None and gen_idx < len(anc)
                                  else None),
                       "committed": int(out[slot]),
                       "accepted": True}  # plain path: one slot, accepted
                if self.record_full_logits:
                    row["logits"] = lg.detach().to("cpu", dtype=torch.float32).clone()
                self.rows[idx].append(row)
                batch_by_req.setdefault(id(r), []).append(row)
            rec._pending = batch_by_req
            return out

        e._sample_batch = _patched_sample_batch

        # Wrap verify so rejected verify-chain slots are marked accepted=False.
        # n_ok is exactly the production rule: got[n_ok] == chains[n_ok+1], with
        # got the (anchor-forced) sampled tokens; committed slots are 0..n_ok.
        self._orig_verify = e._verify

        def _patched_verify(rows, chains, logits, hidden):
            rec_self = self
            orig = rec_self._orig_verify
            # mark all pending rows rejected first, then accept per n_ok below
            for rr in rec_self._pending.values():
                for row in rr:
                    row["accepted"] = False
            orig(rows, chains, logits, hidden)
            # After the real verify, derive n_ok per row from the anchor-forced
            # output the same way production does, using recorded gen_idx and
            # the draft chains. Accepted = positions that advanced output.
            for i, r in enumerate(rows):
                plist = rec_self._pending.get(id(r))
                if not plist:
                    continue
                chain = chains[i]
                # produced chain tokens are rows whose gen_idx is in-chain;
                # production accepted leading run against chain[j+1].
                got = [row["committed"] for row in plist]
                n_ok = 0
                while n_ok < len(got) - 1 and got[n_ok] == chain[n_ok + 1]:
                    n_ok += 1
                for j in range(n_ok + 1):
                    if j < len(plist):
                        plist[j]["accepted"] = True
            rec_self._pending = {}

        e._verify = _patched_verify
        return self

    def bind_request(self, idx: int, req) -> None:
        """Associate a submitted request object with its anchor index. Optional:
        if callers step one prompt at a time (the common probe) request index
        is auto-assigned on first encounter. Explicit bind wins."""
        self._rid_to_idx[id(req)] = idx

    def _idx_for(self, req):
        rid = id(req)
        idx = self._rid_to_idx.get(rid)
        if idx is None:
            # Auto-assign by first encounter. When anchors are supplied, only
            # assign within their range (explicit bind otherwise); record-only
            # mode assigns an unbounded index per new request.
            used = set(self._rid_to_idx.values())
            if self.anchors is None:
                idx = len(used)
            else:
                free = [i for i in range(len(self.anchors)) if i not in used]
                idx = free[0] if free else None
            if idx is not None:
                self._rid_to_idx[rid] = idx
                self.rows.setdefault(idx, [])
        return idx

    # ------------------------------------------------------------ export
    def accepted_positions(self, idx: int) -> list[dict]:
        """Records that actually advanced the output (verify accepted slots
        got[0..n_ok] plus every plain-greedy slot), one per generated position.
        The _verify wrapper sets accepted=False on rejected draft slots."""
        return [r for r in self.rows.get(idx, []) if r.get("accepted", True)]

    def uninstall(self):
        if self._orig_sample_batch is not None:
            self.e._sample_batch = self._orig_sample_batch
            self._orig_sample_batch = None
        if self._orig_verify is not None:
            self.e._verify = self._orig_verify
            self._orig_verify = None
        self._pending = {}


# --------------------------------------------------------------------------- #
# Comparators used by the quality gate; pure functions, no torch dependency for
# the discrete agreement, torch only for KL on full logits.
# --------------------------------------------------------------------------- #
def top1_agreement(recs_a, recs_b) -> float:
    """Fraction of positions present in both (matched in order) whose top1
    argmax agrees. Records are aligned by sequence order; callers pass equal
    lengths (same anchor => equal committed prefix length)."""
    n = min(len(recs_a), len(recs_b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if recs_a[i]["top1"] == recs_b[i]["top1"])
    return same / n


def first_top1_divergence(recs_a, recs_b):
    n = min(len(recs_a), len(recs_b))
    for i in range(n):
        if recs_a[i]["top1"] != recs_b[i]["top1"]:
            return i
    return None


def kl_from_logits(la: torch.Tensor, lb: torch.Tensor) -> float:
    """KL(P_a || P_b) from raw logits (f32), numerically stable."""
    pa = torch.softmax(la.float(), dim=-1)
    return float(torch.sum(pa * (torch.log_softmax(la.float(), -1)
                                 - torch.log_softmax(lb.float(), -1))).item())


def margins(recs_a, recs_b):
    """At each disagreeing position, both arms' top1 and top2 logits, so the
    consumer can describe near-tie vs real difference without a fixed epsilon.
    Requires full logits. Returns list of {pos, a_top1, a_margin, b_*}."""
    out = []
    n = min(len(recs_a), len(recs_b))
    for i in range(n):
        ra, rb = recs_a[i], recs_b[i]
        if ra["top1"] == rb["top1"] or "logits" not in ra:
            continue
        def m(rec):
            t = rec["logits"].topk(2)
            return int(t.indices[0]), float(t.values[0] - t.values[1])
        a1, am = m(ra)
        b1, bm = m(rb)
        out.append({"pos": i, "a_top1": a1, "a_top1_minus_top2": am,
                    "b_top1": b1, "b_top1_minus_top2": bm})
    return out
