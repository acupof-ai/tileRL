"""Build the cutover window's mixed-length prompt file from real corpora.

The window needs 8 prompts: 6 real 37.6k, 1 short (<8k), 1 ~16k. The short ones
are the interesting ones, because `sparse_min_tokens` changes their ROUTING, not
just their decode path:

    engine.py:1142  sparse_on = min_tokens == 0 or len(tokens) > min_tokens

  * 4096 tokens — baseline (8192) serves it DENSE; min0 serves it SPARSE. The two
    configs do different work on it, so its wall times are not a regression
    reading; its OUTPUT is the check (item 4 of the cutover: min0 sends short
    prompts through sparse now, confirm the text is sane).
  * 16000 tokens — above either threshold, so both configs go sparse. A genuine
    like-for-like prefill comparison.

Order in the file is the order the probe runs them (`load_prompts` takes the
first N in range), and that order is the prompt index everywhere downstream,
including the prefill-tick parser's run indexing. Long prompts first so they are
contiguous, short pair last.

Sources are pre-existing real-token corpora on the device, not synthesized:
  * $HOME/serve805_prompts.jsonl — the 30 real 37.6k wikitext prompts used by
    every earlier #805 window.
  * fidelity-corpus/held_32768.jsonl / held_8192.jsonl — held wikitext docs as
    `{"ctx", "ids"}`; sliced, so the ids stay real contiguous text.

    python3 scripts/make_cutover_prompts.py --home $HOME --out cutover_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# (label, source, count, tokens) — tokens None takes the record whole.
PLAN = [
    ("long37k", "serve805_prompts.jsonl", 6, None),
    ("short4k", "fidelity-corpus/held_8192.jsonl", 1, 4096),
    ("mid16k", "fidelity-corpus/held_32768.jsonl", 1, 16000),
]


def _ids_of(obj):
    for key in ("input_ids", "ids", "tokens"):
        v = obj.get(key)
        if isinstance(v, list) and v and isinstance(v[0], int):
            return list(v)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=os.path.expanduser("~"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, manifest = [], []
    for label, rel, count, want in PLAN:
        path = os.path.join(args.home, rel)
        if not os.path.exists(path):
            print(f"FATAL: missing source {path}", file=sys.stderr)
            return 90
        taken = 0
        with open(path) as f:
            for line in f:
                if taken >= count:
                    break
                line = line.strip()
                if not line:
                    continue
                ids = _ids_of(json.loads(line))
                if ids is None:
                    continue
                if want is not None:
                    if len(ids) < want:
                        continue
                    ids = ids[:want]
                rows.append({"input_ids": ids})
                manifest.append({"i": len(rows) - 1, "label": label,
                                 "source": rel, "n_tokens": len(ids)})
                taken += 1
        if taken < count:
            print(f"FATAL: {rel} yielded {taken} of {count} {label} prompts",
                  file=sys.stderr)
            return 90

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    man = {"prompts": manifest, "n": len(rows),
           "note": "order here is the probe's prompt index; the prefill parser "
                   "indexes runs the same way"}
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)

    print(f"wrote {args.out}: {len(rows)} prompts")
    for m in manifest:
        print(f"  {m['i']}: {m['label']:9s} {m['n_tokens']:6d} tok  {m['source']}")
    print(f"probe args for this file: --min-tokens "
          f"{min(m['n_tokens'] for m in manifest)} "
          f"--max-tokens {max(m['n_tokens'] for m in manifest)} --n-prompts {len(rows)} "
          f"--min-prompts {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
