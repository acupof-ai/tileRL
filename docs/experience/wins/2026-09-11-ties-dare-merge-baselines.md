# TIES and DARE merge baselines land next to average — tiny/CPU, 2026-09-11

> Status: CPU-verified on tiny; the 27B ISO-vs-TIES/DARE verdict stays `pending-remote`

## Context

The ISO merger's only in-tree control was plain task-vector averaging. P3's
claim is that ISO beats the *literature* baselines, so `src/tilerl/merge.py`
now implements the two the merger is compared against:

- **TIES** (`--method ties`): per specialist, each row keeps its top-`keep`
  (default 0.8) fraction of task-vector entries by magnitude; the elected sign
  at each position is `sign(Σ trimmed)`; positions average only the
  specialists agreeing with it (the disjoint mean); `W = W₀ + merged`.
- **DARE** (`--method dare`): each specialist's task vector has a `drop`
  (default 0.5) fraction of entries zeroed by an independent Bernoulli mask,
  survivors rescaled by `1/(1-drop)`, then averaged; `W = W₀ + mean`.

Both are pure tensor math over the existing one-tensor-at-a-time
`merge_checkpoints` path — torch is only the container, no optimizer or
autograd. Non-float tensors pass through. The checkpoint path salts DARE's
mask seed with the tensor name, so a rerun over the same inputs is
reproducible.

## What Worked

Same tiny harness as the ISO entry (two 15-step AdamW SFT specialists,
`train_step` causal CE):

| method | loss on A | loss on B |
|---|---:|---:|
| base | 22.335 | 21.990 |
| average | 18.460 | 17.550 |
| **ties** (keep 0.8) | 18.020 | 16.452 |
| dare (drop 0.5) | 18.445 | 17.546 |
| iso | **15.995** | **14.726** |

Both baselines keep BOTH tasks (beat the base conjunctively), and ISO already
ranks ahead of both on tiny — the shape of the 27B claim, not a substitute for
the 27B run. TIES trims toward the dominant per-row movements and gains over
average on B; DARE at this scale barely moves from averaging (random drops are
unbiased after rescaling).

Arithmetic is pinned on hand-computed matrices in `tests/test_merge.py`, not
the formula: the TIES case discriminates trim (trimmed-away votes would
otherwise survive) and disjoint mean (an agreeing pair must return the mean,
not the 2x sum); the DARE case pins the explicit `torch.Generator` masks and
the rescale. Four mutants were red before the gate shipped: trim disabled,
mean→sum, rescale deleted, one shared mask instead of per-specialist.

## Rule

A baseline in the paper is a baseline in the tree, pinned by its arithmetic: a
comparison against an imagined TIES/DARE cannot be audited. Tiny loss
ordering licenses the procedure comparison; the 27B verdict that ISO beats
them on real RL specialists remains the pod run.

## Results

| date | commit | machine | target | model | avg A/B | ties A/B | dare A/B | iso A/B |
|---|---|---|---|---|---|---|---|---|
| 2026-09-11 | this | Mac (no GPU) | cpu | tiny | 18.46/17.55 | 18.02/16.45 | 18.45/17.55 | 16.00/14.73 |
| pending-remote | | pod | cuda | qwen38-27b | | | ISO must beat both | |

27B command (bf16 SFT specialists; fp4 checkpoints are refused by merge):

```bash
for m in ties dare iso average; do
  TILERL_TARGET=cuda uv run tilerl merge \
    --base  <27b-bf16-base> --specialists <s1>,<s2> --method $m --out runs/merge-$m
  # evaluate each out dir on both tasks' held-out sets; ISO wins only if it
  # beats ties AND dare conjunctively.
done
```

Raw artifacts: `TILERL_TARGET=cpu uv run pytest -q -s tests/test_merge.py`.
