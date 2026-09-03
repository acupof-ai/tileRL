"""Real-text prompts, so a measurement of acceptance is about the model.

Acceptance is a property of the prompt DISTRIBUTION, not of the model alone: at
ctx=1024 the same config reads 2.62 tok/forward on ids drawn uniformly from the
vocabulary and 3.34 on `range(10, 10+ctx)`, a 1.27x spread that straddles every
break-even we compute. Neither is text. A depth or width default settled on
either one is settled on an artifact
(wins/2026-09-03-long-context-decode-is-all-tick-cost.md).

Wikitext-103's test split is in the pod's HF cache, which is why it is the corpus
here rather than something better matched to serving traffic. That is a real
limitation: wikitext is encyclopedic prose, and a chat or code workload will
accept differently.
"""

from __future__ import annotations

import glob
import os

WIKITEXT_GLOB = (
    "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/"
    "wikitext-103-raw-v1/test-*.parquet"
)


def spans(ids: list[int], n: int, ctx: int, skip: int) -> list[list[int]]:
    """`n` disjoint spans of exactly `ctx` ids, starting at `skip`.

    `skip` drops the head of the stream: wikitext's first rows are short headers and
    blank lines that tokenize into runs of newlines -- trivially predictable, so a
    prompt starting there measures what the `range(10, 10+ctx)` prompt measured.
    """
    if len(ids) < skip + n * ctx:
        raise SystemExit(
            f"corpus has {len(ids)} tokens, need {skip + n * ctx} for {n} x {ctx}")
    return [ids[skip + i * ctx : skip + (i + 1) * ctx] for i in range(n)]


def wikitext_ids(tok, n: int, ctx: int, skip: int = 512) -> list[list[int]]:
    """`n` prompts of `ctx` tokens each from wikitext-103's test split."""
    paths = glob.glob(os.path.expanduser(WIKITEXT_GLOB))
    if not paths:
        raise SystemExit(f"wikitext-103 test parquet not in the HF cache: {WIKITEXT_GLOB}")
    import pyarrow.parquet as pq

    text = "\n".join(pq.read_table(sorted(paths)[0]).column("text").to_pylist())
    return spans(tok.encode(text), n, ctx, skip)


def _self_check() -> None:
    """`spans` must return disjoint, exactly-ctx slices past `skip`, or refuse.

    Runs on the GPU-less box. The failure this guards is a prompt shorter than
    asked for: `measure` divides by its length, so a short prompt reads as a
    cheaper tick rather than as an error.
    """
    ids = list(range(1000))
    got = spans(ids, 3, 100, 512)
    assert [len(g) for g in got] == [100, 100, 100], [len(g) for g in got]
    assert got[0][0] == 512 and got[2][-1] == 811, (got[0][0], got[2][-1])
    assert got[0][-1] + 1 == got[1][0], "spans must be contiguous and disjoint"
    try:
        spans(ids, 3, 200, 512)  # 512 + 600 > 1000
    except SystemExit:
        pass
    else:
        raise AssertionError("spans must refuse rather than return a short prompt")
    print("corpus: spans OK")


if __name__ == "__main__":
    _self_check()
