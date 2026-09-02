# The KV pool's dtype was not the attention kernel's — 4.71 ms/token — 2026-09-02

## Context

After the attention thread-redundancy fix, decode at 4096 ctx sat at 30.0
tok/s. Attention was down to 2.92 ms of a 33 ms token, so the remaining 30 ms
had never been attributed. A per-kernel profile of the decode window answered
it in one run.

## What Worked

Profiling by op class, with the profile reconciled against the clock:

| class | ms/tok | % GPU |
|---|---:|---:|
| fp4 GEMV | 21.04 | 61.4% |
| **elementwise** | **7.69** | **22.4%** |
| attention | 2.92 | 8.5% |
| rmsnorm | 1.19 | 3.5% |
| GDN | 1.10 | 3.2% |

The largest elementwise entry was **4.71 ms/token over 32 calls** — exactly
2 × 16 full-attn layers, i.e. `backend.py`'s `self._f32(k_cache)` and
`self._f32(v_cache)`. The pool is bf16 (`kv_cache.py:100`), the sm70 attention
kernel is f32, so every layer of every token materialized an f32 copy of the
**whole plane** — all 1024 blocks, not the live ones, which is why the cost was
independent of context.

Arithmetic: a plane is 1024 × 4 × 16 × 256 = 16.8M elements; read bf16 33.5 MB
+ write f32 67 MB = 100 MB, × 2 (K and V) × 16 layers = 3.2 GB/token = 3.6 ms
at 900 GB/s, against 4.71 measured.

Fix: allocate the pool at `backend.io`. **The identical fix already existed one
struct over** — `LinearStatePool` takes f32 on CUDA with a comment naming the
same cast cost ("a bf16 pool cost two 1.5MB casts per layer per tick"). The KV
pool simply never got it. Cost: +1 GB of device memory.

| ctx | dense after | before |
|---:|---:|---:|
| 32 | 38.8 | 32.4 |
| 1024 | 37.8 | 31.8 |
| 4096 | **35.3** | **30.0** |

Elementwise fell 7.69 → 2.75 ms/token; every context gained ~18%.

**The roofline was wrong the whole time.** *(And this correction was itself wrong
— see the amendment below.)* Measuring the checkpoint instead of citing a
remembered number: 20.35 GB, not 14 — packed nibbles 12.81, block scales 3.20,
norms/embed/lm_head 4.32. The bound read 900/20.35 = 44.2 tok/s, so dense 35.3
looked like 80% of roofline rather than 55%.

**Amendment, 2026-09-02:** 20.35 GB is the checkpoint, not the decode stream. A
dense token streams **16.04 GB** — trunk 15.24 + lm_head 0.80; `embed_tokens`
(2.54, one row gathered) and the visual tower (0.92, never run on a text tick)
are resident but not streamed. The real bound is 900/16.04 = **56.1 tok/s**, so
dense 35.3 is **63%** of roofline. Full write-up:
`errors/2026-09-02-roofline-is-the-streamed-subset.md`. Speculation exceeding a
dense roofline (50.8) is still expected: a verify forward emits several tokens
for one weight read.

Two defects surfaced on the way, both invisible to any timing run:

- `write_tokens` hardcoded a bf16 pool. Three attempts to parameterize the
  dtype failed — tilelang's eager builder re-executes the kernel body with only
  its own kwargs bound, so a closure variable is not in scope inside a
  `T.Tensor` annotation, whatever it is named and whether it is a string or a
  conditional. Settled with a literal f32 twin factory.
- `_draft_step` asked for hidden it no longer had. A chunked prefill overwrites
  `r.hidden` per chunk while `draft_pos` stays behind them, so a 1024 prompt at
  512/chunk requested 1535 positions and held 511. `F.pad` accepted the
  truncation silently and `torch.cat` failed downstream with a shape nobody
  could source. An assert printing base/off/q named it immediately (`off=-1024`).
  Every speculative number measured before this — including the 46.5 tok/s peak
  reported earlier the same day — was produced with misaligned hidden.

Final, with both fixed:

| ctx | dense | spec d3 | spec at session start |
|---:|---:|---:|---:|
| 512 | 38.2 | 48.4 | 37.7 |
| 1024 | 37.8 | **50.8** | 34.5 |
| 4096 | 35.3 | **40.3** | 16.7 |

4096 ctx speculation is **2.41×** what it was. tok/fwd is unchanged by the
draft fix (2.95-3.34 before and after), so that bug cost correctness, not
throughput — the draft read zeros rather than a neighbour's KV.

## Rule

A pool's dtype is part of the kernel's ABI. When a kernel declares f32 IO and
its backing store is bf16, the cast does not disappear — it becomes a
per-call copy of the entire store, and it is invisible in every end-to-end
number and every kernel microbenchmark. The profile that finds it must be by op
class over the real decode window.

Second: reconcile a profile against the clock before reading it. The first run
of this profile reported 8217 ms/token against a 33 ms token — it had spanned
the prefill chunks while dividing by the decode's token count. `prof_decode_budget.py`
now prints GPU-vs-wall and refuses to present numbers when they disagree by 2×.

Third: measure the checkpoint, do not cite it. A remembered 14 GB became the
denominator of every "% of roofline" claim in this project until `du` on the
shards said 20.35.
