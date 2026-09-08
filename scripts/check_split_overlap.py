"""Are the MATH level-5 train and eval files disjoint, and how long are level-5 answers?

Two preconditions for the steps_to_score curve, both read rather than assumed:

  1. OVERLAP. Training on rows the curve scores makes the curve measure memorisation.
     A peer wrote that failure up today from the other direction (a contamination
     correlated with the treatment), so this is a read, not a trust.
  2. The prompt length distribution, which sets the pool and bounds the cap.

Reports counts and the intersection size. Prints nothing about the answers themselves.
"""
import json
import sys

tr_path, ev_path = sys.argv[1], sys.argv[2]


def rows(p):
    return [json.loads(ln) for ln in open(p) if ln.strip()]


tr, ev = rows(tr_path), rows(ev_path)
tr_p = {r["prompt"] for r in tr}
ev_p = {r["prompt"] for r in ev}
both = tr_p & ev_p
print(f"train {len(tr)} rows ({len(tr_p)} distinct prompts)  {tr_path}")
print(f"eval  {len(ev)} rows ({len(ev_p)} distinct prompts)  {ev_path}")
print(f"OVERLAP: {len(both)} prompts in both "
      f"({100 * len(both) / max(len(ev_p), 1):.1f}% of the eval set)")
if both:
    print("  -> the curve would be scoring rows the run trained on")

lv = {}
for r in tr:
    lv[r.get("level", "?")] = lv.get(r.get("level", "?"), 0) + 1
print(f"train levels: {lv}")
lv = {}
for r in ev:
    lv[r.get("level", "?")] = lv.get(r.get("level", "?"), 0) + 1
print(f"eval levels:  {lv}")

for name, rs in (("train", tr), ("eval", ev)):
    ch = sorted(len(r["prompt"]) for r in rs)
    n = len(ch)
    print(f"{name} prompt chars: median {ch[n // 2]}  p90 {ch[int(0.9 * n)]}  max {ch[-1]}")
    boxed = sum("boxed" in r["prompt"] for r in rs)
    print(f"  {boxed}/{n} prompts ask for \\boxed{{}} -> --reward boxed is the matcher")
