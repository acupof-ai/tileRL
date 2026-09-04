# The thinking cap: train under a tight token budget, keep the accuracy, spend 23% fewer tokens

**Date:** 2026-09-04
**Run:** `52b8251e79bc` — grpo-gsm8k-27b, 100 steps, group 8, LoRA rank 16, lr 1e-4,
`max_new_tokens=256`, thinking off. Adapter 170.8M params, 341 MB.

## Context

A GRPO run trained at a 256-token cap reported GSM8K 39.0% → 94.2%. The 39.0%
control was measuring the cap rather than the model
([the audit](../errors/2026-09-04-the-eval-cap-measured-itself.md)), so the
headline was worthless. The question left over was what, if anything, the run had
actually taught the policy.

The answer is that it taught brevity, brevity is worth money, and it is not paid
for in accuracy.

## What worked

Score the trained adapter and the base under the **uncapped** protocol — 2048
cap, thinking on, greedy, the same 500 test questions, one engine, the adapter
loaded into the same weights (`add_lora` starts B at zero, so the first pass *is*
the base):

| GSM8K, uncapped, n=500 | accuracy | 95% CI | total tokens | tokens / correct |
|---|---:|---:|---:|---:|
| base | 448/500 = **89.6%** | 86.6 – 92.0 | 157,601 | 351.8 |
| trained | 474/500 = **94.8%** | 92.5 – 96.4 | 121,642 | **256.6** |
| | **+5.2 pts** | no overlap | **−22.8%** | **−27.1%** |

z = 3.07, p = 0.0022 on the unpaired two-proportion test; the arms are paired on
the same questions, so the real test is sharper. **The pre-registered threshold
was 4.79 points** — written down when the baseline was first measured and before
the trained arm ran — so this clears a bar set in advance, not one chosen after.

Both directions moved. This is not a speed/quality trade.

## It transfers out of domain

The adapter saw GSM8K and nothing else. Free-form generation, n=100 each:

| | accuracy | tokens |
|---|---:|---:|
| MMLU | 72.0% → 68.0% | 493 → 385 (**−22.0%**) |
| ARC-Easy | 80.0% → 84.0% | 251 → 204 (**−18.9%**) |
| PIQA | 84.0% → 88.0% | 210 → 162 (**−22.9%**) |

Tokens fall 19–23% on all three. Accuracy moves −4, +4, +4 — at n=100 the 95%
interval on a difference is about ±12.7 points, so **none of the three is
distinguishable from no change**, in either direction. The honest claim is
"tokens down, accuracy not measurably harmed", not "tokens down, accuracy up".

MMLU had to be re-run free-form to say anything here: `mmlu_score` runs at
`max_new_tokens=1`, so as wired it structurally cannot see output length.

## The mechanism

Same question, base against trained, at the 256 cap that produced the adapter:

```
BASE, 256 tokens, hit the cap
  ### Step 1: Calculate the volume of a single raised bed
  *   Width = 2 feet ...
  $$V_{bed} = \text{Width} \times \text{Length} \times \text{Height}$$
  $$V_{bed} = 2 \, \text{ft} \times 8 \, \text{ft} \times 2 \, \text{ft} = 32 \, \text{cubic feet}$$
  ### Step 2: ... $$V_{total} = 10 \times 32 = 320 \, \text{cubic feet}$$
                                          <- cut here, 2 of 4 steps done
  answer_match read 320.0, gold 1920 -> WRONG

TRAINED, 109 tokens
  **Volume of one raised bed:**  2 ft × 8 ft × 2 ft = 32 cubic feet
  **Total volume needed:**       10 beds × 32 = 320 cubic feet
  **Number of bags needed:**     320 ÷ 2 = 160 bags
  **Total cost:**                160 × $12 = **$1,920**
  answer_match read 1920.0 -> CORRECT
```

Every intermediate survives — 32, 320, 160, 1920, all four steps. What went is
the LaTeX ceremony and the section headers. The policy did not learn to reason
differently; it stopped spending most of its budget on formatting.

**This mechanism claim is n = 1, read by eye.** The accuracy and length numbers
above are n = 500 and n = 100; the step-preservation reading is one completion
pair and is labelled as such until someone scores "what fraction of the gold
derivation's intermediates still appear" across the set.

## What it cost to get

72% of the training steps carried **zero gradient** — 44 of 61 sampled, 42 of
those because all eight rollouts were right. `groups_untied` failed at 0.76 and
the run is marked FAIL for it. The result above came from roughly a quarter of
the intended signal, which says how fast a binary reward saturates against an
89.6% base and is the argument for `--judge` (landed default-off in #64).

Reproduction is exact: `52b8251e79bc` matches the original `a4332cbca4fa` on
every metric — gsm8k 195/471, reward 0.705/0.94, ce 3.269, tied 0.76, mmlu
0.752/0.750 — differing only in `peak_gib` by 0.01.

**The two token columns were not reproducible from the tree when this was
written.** `gsm8k_accuracy` returned `(correct, total)` and discarded the
completions on decode, so 157,601 and 121,642 came from a probe script and no
`uv run tilerl train` could produce them — the headline number of a merged entry,
resting on something outside the repo. #70 makes it `(correct, total, tokens)`
and records `gsm8k_before_tokens` / `gsm8k_after_tokens` in the manifest, so a
rerun prints these two columns itself.

## Rule

**A tight budget during training is a lever, not just a constraint.** Cap the
rollout well below what the model would freely spend, score correctness only, and
the policy finds the shorter path to the same answer; the saving then follows the
weights to tasks the cap never saw. Output tokens are the serving bill, so a
23% cut at equal accuracy is a real win in the same units the product is sold in.

Do not read a *capped* score as evidence for this. The 39.0% → 94.2% that started
it was an artifact; the claim rests on the uncapped arms, where the cap the
policy was trained under is absent from the measurement entirely.
