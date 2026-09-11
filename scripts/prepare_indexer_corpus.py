#!/usr/bin/env python3
"""Prepare 8k/16k/32k indexer-warm-up prompts from the pod's wiki parquet.

The corpus has almost no single 8k-32k documents, so consecutive ARTICLES are
concatenated into a token stream and cut into disjoint fixed-length spans (the
scripts/corpus.long_doc_spans construction). Articles are split by a seeded
hash BEFORE tokenising: the held-out stream is built only from held-out
articles, so no held-out article's token appears in any training span.

Outputs (one jsonl per split/length, rows {"ctx","ids"}), a histogram json, and
a manifest recording the seed, source files and held-out article ids so the
recall number is reproducible.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq

CONTEXTS = (8192, 16384, 32768)
HELD_OUT_FRAC = 0.1
SEED = 20260911
SKIP = 512
# spans per split per length: enough held-out for a stable recall mean, enough
# train prompts for a few hundred steps with resampling off the held-out set.
N_HELD = {"8192": 32, "16384": 16, "32768": 8}
N_TRAIN = {"8192": 160, "16384": 80, "32768": 40}


def _bucket(article_id: str) -> str:
    h = int(hashlib.sha256(f"{SEED}:{article_id}".encode()).hexdigest(), 16)
    return "held" if h % 100 < HELD_OUT_FRAC * 100 else "train"


def main(wiki_glob: str, out_dir: str, tok_path: str) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    files = sorted(glob.glob(wiki_glob))
    if not files:
        sys.exit(f"no wiki parquet under {wiki_glob}")
    # Only ~4.7M tokens are needed for the full train+held span demand; stop
    # reading files once each split's stream has that much (plus the skip head),
    # so prep does not tokenize all ~1.6M corpus articles.
    needed = SKIP + sum(N_HELD[str(c)] * c for c in CONTEXTS)
    needed_train = SKIP + sum(N_TRAIN[str(c)] * c for c in CONTEXTS)
    streams = {"held": [], "train": []}
    held_ids: list[str] = []
    n_articles = {"held": 0, "train": 0}
    char_hist: dict[str, int] = {}
    done = False
    for f in files:
        table = pq.read_table(f, columns=["id", "text"])
        ids = table.column("id").to_pylist()
        texts = table.column("text").to_pylist()
        for aid, text in zip(ids, texts):
            split = _bucket(str(aid))
            n_articles[split] += 1
            if split == "held":
                held_ids.append(str(aid))
            b = len(text) // 1000
            char_hist[b] = char_hist.get(b, 0) + 1
            streams[split].extend(tok.encode(text).ids)
        if len(streams["held"]) >= needed and len(streams["train"]) >= needed_train:
            done = True
            break
    if not done:
        sys.exit("not enough corpus tokens for the requested spans")
    os.makedirs(out_dir, exist_ok=True)
    summary = {"seed": SEED, "files_used": files[: 1 + files.index(f)],
               "contexts": list(CONTEXTS), "n_articles": n_articles, "held_ids": held_ids,
               "stream_tokens": {k: len(v) for k, v in streams.items()}}
    for split, want in (("held", N_HELD), ("train", N_TRAIN)):
        pos = SKIP
        for ctx in CONTEXTS:
            spans = []
            while len(spans) < want[str(ctx)] and pos + ctx <= len(streams[split]):
                spans.append(streams[split][pos : pos + ctx])
                pos += ctx
            if len(spans) < want[str(ctx)]:
                sys.exit(f"{split} ctx={ctx}: only {len(spans)} spans, need {want[str(ctx)]}")
            path = os.path.join(out_dir, f"{split}_{ctx}.jsonl")
            with open(path, "w") as fh:
                for s in spans:
                    fh.write(json.dumps({"ctx": ctx, "ids": s}) + "\n")
            summary[f"{split}_{ctx}"] = len(spans)
            print(f"{split} {ctx}: {len(spans)} spans")
    # token-length histogram over a sample of raw articles (character buckets in k).
    summary["char_len_k_histogram"] = dict(sorted(char_hist.items())[:12])
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
