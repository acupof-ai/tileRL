# `serve` on a 32 GB card: two allocations nothing had ever summed — V100 (sm70), 2026-09-02

> Status: both fixed. `serve --model qwen38-27b` asked for **275 GB** of KV
> (caught by arithmetic, before it ran) and then kept the **embedding table in
> two dtypes at once**, 7.11 GiB for one tensor — which is what actually OOMed,
> on the first token, after `/health` had already answered OK.

## Context

Everything measured on this card — decode 37.6 tok/s, prefill 7.89 ms/token, the
speculation ladder — came from scripts that build the engine directly:

```python
e = build_engine(cfg, model, be, num_blocks=1024, num_slots=4, max_batch=4, ...)
```

`tilerl serve` does not take that route. It calls `cli._build_engine`, which sizes
the pool from the model's own config:

```python
ctx = int(cfg.max_position_embeddings)
kw = dict(num_blocks=max(256, (ctx * 8) // BLOCK_TOKENS), ...)
```

The 27B declares `max_position_embeddings = 262144`. That is 131072 blocks of 16
tokens, and each block on sm70 holds f32 KV for 16 full-attention layers × 4 KV
heads × head_dim 256 × (K and V) = 2.10 MB.

**131072 × 2.10 MB = 275 GB, on a card with 32 of which 19 are already weights.**

The default is not wrong — it was written for a 96 GB H20 where the arithmetic
lands differently, and the flag it needed simply did not exist yet.

## Root Cause

A default that is only reachable through a path nothing benchmarks. Eleven bench
entries, three profilers, and a compile gate all exercise the engine; none of them
call `_build_engine`. The one caller that does is the product surface.

The tell was available for free and I did not look: the ratio between what the
benches allocate (1024 blocks) and what serving would allocate (131072) is 128×,
and both numbers were in the tree the whole time.

## Fix

`serve` gains `--blocks` and `--max-ctx`, both defaulting to the old behaviour so
no other target changes:

```
--blocks 2048 --max-ctx 4096      # 32768 tokens of KV = 4.3 GB, 8 rows at 4K
--blocks 4096 --max-ctx 8192      # 8.6 GB
```

`--max-ctx` matters as much as `--blocks`: it caps what a request may ask for, so
a prompt cannot outgrow the pool that was sized for it. Sizing the pool without
capping admission just moves the failure from startup to the first long prompt.

## The second wall, thirty seconds later

With the flags in, the probe still died:

```
/usr/bin/python3: No module named tilerl.__main__;
'tilerl' is a package and cannot be directly executed
```

`python3 -m tilerl` does not work: the package ships no `__main__.py`, and the
`tilerl` console script `pyproject.toml` declares is installed into the venv,
not into the pod's `/usr/bin/python3` (which is the interpreter that has to be
used — its torch 2.5.1+cu121 matches the 535 driver, the venv's does not).

Working form: `/usr/bin/python3 -c 'from tilerl.cli import main; main()' serve ...`

Worth recording because it is invisible from a Mac. Locally `uv run tilerl serve`
works, which is exactly the reassurance that stops you checking.

## The third wall

```
ERROR: [Errno 13] error while attempting to bind on address ('127.0.0.1', 811):
       permission denied
```

Ports below 1024 need root. Mine, entirely — I picked 811 for the probe to avoid
colliding with anything on 8000. 8811 works.

Not interesting in itself, but the sequence is: **sizing bug → entry point →
port**, three failures before the first token, none of which any test or benchmark
could have caught, all of which would have been discovered live in front of the
user. The 19 GB weight load takes ~5 minutes, so each one cost a full round trip.

The run that got past all three confirmed the engine itself was never the problem:
the model loaded, the engine started, and the pool allocated without OOM — the
error came from `uvicorn` binding, i.e. after everything this change was actually
about.

## The fourth wall — and the one that was actually going to kill the demo

Past the port, the server started, `/health` answered, `nvidia-smi` read
**30082 MiB of 32510**, and the first chat completion died:

```
CUDA out of memory. Tried to allocate 4.74 GiB.
  4.14 GiB is free ... File "backend.py", line 1015, in embedding
```

4.74 GiB is exactly `248320 × 5120 × 4` — the f32 embedding table. `materialize`
had already moved the bf16 table to the card (2.37 GiB); the gather then cast it
to f32 and cached that, so **one table occupied 7.11 GiB in two dtypes**.

sm70 cannot codegen a bf16 load, and `Backend.io` is f32 for exactly that reason
— but it loads **f16** natively. The narrow copy is the whole fix: cast during
`materialize`'s device move, add an `embedding_f16` body, and the f32 copy never
exists. 4.74 → 2.37 GiB, i.e. **2.37 GiB back**, with the gather still widening
to f32 on read. Untied heads only: a tied table is also the `lm_head` weight and
that linear wants f32.

Note the shape of this one against the first three. The KV sizing bug was found
by arithmetic before it ran; this one needed the process to actually reach its
first token, because it is a *sum* — the table is fine, the pool is fine, the
weights are fine, and 14.17 + 0.74 + 4.74 + 4.20 + activations is not.

## Rule

**A default is only tested on the paths that reach it.** "The engine is
thoroughly benchmarked" said nothing about `serve`, because every benchmark
constructs the engine itself and passes the sizes it wants. Before demoing a
product surface, run *that surface*, not the thing underneath it.

Corollary, and the reason this cost minutes instead of a demo: the arithmetic was
doable in one line from numbers already in the tree. When two callers of the same
constructor disagree by 128× on a memory parameter, one of them has never run.

Third: **rehearse the demo path end to end, on the target, before the demo.** All
four failures sat in front of the first token and behind a 5-minute weight load.
None of them are in the code this branch spent two days optimizing; all of them
would have happened live.

Fourth, from the embedding table: **a memory budget is a sum, so audit it as
one.** Each allocation here was individually defensible; the failure only exists
in the total. The one-line audit that would have caught it — body 14.17 + head
0.74 + table 4.74 + pool 4.20 = 23.85 GiB before a single activation, on a card
with 31.74 — is arithmetic over numbers the config already knows, and it now
lives in the entry above.
