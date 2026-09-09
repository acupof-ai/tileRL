"""MATH-L5 boxed-matcher eye check (2026-09-09, 27's kill-rule replacement).

The run's per-problem rows store only {i, correct, tokens, answer} -- the completion
is discarded, so the matcher cannot be audited from the run's own files. This probe
runs the same eval path the before-arm uses (render_chat, greedy, cap 6144, boxed
matcher) on the base model and prints the three things the score cannot distinguish:

  1. does the model emit \boxed{} at all
  2. what extract_boxed returns (None / sane)
  3. whether extracted and gold normalize to the same form

Kill rule per 27: model emitted \boxed{} and the matcher cannot extract it -> kill.
The score itself is never the kill criterion; MATH L5 is allowed to be low.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TILERL_TARGET", "cuda")

from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.cli import _build_model, _qwen38_tokenizer  # noqa: E402
from tilerl.engine import build_engine  # noqa: E402
from tilerl.eval import generate_ids  # noqa: E402
from tilerl.kv_cache import NoPrefixStore  # noqa: E402
from tilerl.math_answer import extract_boxed, normalize  # noqa: E402
from tilerl.model import drop_quantized  # noqa: E402
from tilerl.prompt import render_chat, sampling  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/work/math_test_l5_500.jsonl")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--cap", type=int, default=6144)
ap.add_argument("--blocks", type=int, default=2048)
a = ap.parse_args()

cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
drop_quantized(model)
engine = build_engine(cfg, model, get_backend(), num_blocks=a.blocks, num_slots=8,
                      decode_graph=False, prefix_store=NoPrefixStore())
tok = _qwen38_tokenizer()

rows = [json.loads(ln) for ln in Path(a.data).read_text().splitlines() if ln.strip()][:a.n]
prompts = [render_chat([("user", r["prompt"])], False) for r in rows]
ids = generate_ids(engine, tok, prompts, sampling(tok, False, a.cap, temperature=0.0), 8)

n_boxed = n_extract_none = n_match = 0
for i, (r, seq) in enumerate(zip(rows, ids)):
    text = tok.decode(list(seq))
    gold = r["answer"]
    has_boxed = r"\boxed{" in text
    ext = extract_boxed(text)
    got, want = normalize(ext), normalize(gold)
    match = got is not None and got == want
    n_boxed += has_boxed
    n_extract_none += has_boxed and ext is None
    n_match += match
    verdict = "MATCH" if match else ("EXTRACT-FAILED" if has_boxed and ext is None else "wrong")
    print(f"\n=== q{i}: {verdict}  tokens={len(seq)}", flush=True)
    print(f"  gold raw:       {gold!r}")
    print(f"  gold norm:      {want!r}")
    print(f"  boxed in text:  {has_boxed}")
    print(f"  extracted raw:  {ext!r}")
    print(f"  extracted norm: {got!r}")
    print(f"  tail: ...{text[-300:]!r}")

print(f"\nsummary: {n_boxed}/{len(rows)} emitted \\boxed{{}}, "
      f"{n_extract_none} emitted-but-unextractable, {n_match}/{len(rows)} matched", flush=True)
print("kill rule: model emitted \\boxed{} and matcher could not extract -> kill the run", flush=True)
