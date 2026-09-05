# `--max-ctx 4096` was a flag, not a ceiling — V100 sm70, 2026-09-05

> Status: **Shipped and measured on the card.** `blocks_total` 256 → 2048, and the
> 11019-token prompt that returned 400 now returns 200.

## Context

ckl's report: "V100 上开的上下文长度有点太短了,输出到很短就截断了" — the served
context is too short, output truncates.

**The symptom was misattributed, and that changes the fix.** Output was not being
truncated. A 4000-`max_tokens` request came back `finish_reason: stop` with 1141
completion tokens — the model chose to stop. What actually failed was the *prompt*
side: an 11017-token request got

```
HTTP 400 {"message":"request (11017 tokens) exceeds max_total_tokens (4096)"}
```

So the ceiling was `--max-ctx 4096`, hard-coded at `scripts/serve_v100.sh:51` and
therefore **in git** — not, as first reported to the peer, "only a launch parameter,
no PR needed." I had read the running process's `/proc/<pid>/cmdline` and stopped
before reading where that value came from.

## What Worked

`--max-ctx 4096` → `32768`, an **8x** raise with no code change, no memory lever,
and nothing traded.

The reason it was free is the shape of `_fit_blocks` (`engine.py:1170`):

```python
return min(fit, cap) if cap else fit
```

`cap` is `--max-ctx × --max-batch / 16` (`cli.py:89`); `fit` is two thirds of the
free memory at build time. At `--max-ctx 4096 --max-batch 1` the cap is 256 blocks,
and the live server reported `blocks_total: 256` — **numerically identical to the cap,
so the number alone could not say which operand bound it.** That ambiguity is the
whole reason 4096 survived: it looks like a measured ceiling.

It was the cap. The fit at `--slots 8` with a draft was measured at **2649 blocks =
42384 tokens** (#65, `2026-09-04-slots-default-8-not-16.md`) in a *costlier*
configuration than serve runs — `--depth 3`, whose per-step verify states scale with
`slots × width`. 32768 tokens is 2048 blocks. The cap still binds at 32K, with the fit
above it.

| `--max-ctx` | blocks (= cap at `--max-batch 1`) | pool @ 2.125 MiB |
|---:|---:|---:|
| 4096 (was) | 256 | 544 MiB |
| **32768 (now)** | **2048** | **4352 MiB** |
| 42384 | 2649 | 5629 MiB — the measured fit, `--depth 3` |

Block size is 2.125 MiB, not the 2.00 MiB `serve-v100.md` carried: 2.000 MiB of trunk
K+V (16 full-attn layers × 4 KV heads × 16 tokens × head_dim 256 × f32 × 2 planes)
**plus the draft's mirrored 1/16** at `--depth 1`, which `_fit_blocks` charges for and
the doc's table did not.

**A derivation I published and then withdrew, in this entry because the retraction
should travel with the number.** I first put the ceiling at 2453 blocks / 39248
tokens, from `25492 / 32768 MiB` resident read off the live server, backing out
≈ 7820 MiB free at startup. That is wrong twice over. `nvidia-smi` later read
**22136 MiB**, not 25492 — and more importantly the read happens *after* the pool is
allocated, whereas `_fit_blocks` runs *after* `torch.cuda.empty_cache()`
(`engine.py:1211`), whose own comment records that call as the difference between a
1024-token and a 62832-token context. A post-hoc `nvidia-smi` cannot see the memory
state the fit saw. The measured 2649 supersedes the derived 2453; the doc now cites
the measurement.

## Measured on the card

Restarted at this change (`boot 0 at 09:39:13 sha 6a4e737`) and measured before
anything else touched the server, because the first request is the only one that can
carry a cold JIT cost.

| | 4096 (was) | 32768 (now) |
|---|---:|---:|
| `blocks_total` | 256 | **2048** |
| 11019-token prompt | `400 exceeds max_total_tokens` | **200**, `prompt_tokens 11019` |
| idle VRAM | 22136 MiB | 25980 MiB |
| peak during the long prefill | — | **27036 MiB** of 32768 |

**The KV arithmetic checks out independently.** The idle delta is
25980 − 22136 = **3844 MiB** against a predicted 4352 − 544 = **3808 MiB** — 1.009x.
That is the 2.125 MiB block confirmed by a second route, and it would have been
1.062x had the draft's 1/16 plane been left out.

**Peak is 27036 MiB, 5732 MiB spare**, and the transient above idle is only
1056 MiB — the prefill partials at 11019 tokens, which is what the third held back
from the fit exists to cover.

**First-request cost: 19.4 s, not 182 s.** The first request took 182.5 s wall and an
identical second took 163.1 s (1.119x). Only the 19.4 s difference is the one-time fp4
JIT for that prefill shape; the remaining 163 s is what an 11019-token prompt costs
every time. Reporting 182 s as "the JIT cost" would have been wrong by 9.4x, and the
second request is the only thing that separates them.

That warm 163.1 s is **14.68 ms per prompt token** (163.1 s minus ~1.3 s for 64
decoded tokens at the measured 50.0 tok/s, over 11019 tokens), against the 8.92
ms/token the bench table records at ctx 4096. Prefill cost per token grows with
context; this entry does not claim the shape of that curve, only the two endpoints.

**19.4 s is the `kv_tier` cold-start baseline**, and it is small — which matters for
`kv_tier`: an offload tier has to beat re-prefilling, and re-prefilling this prompt
costs 163 s while the JIT it avoids is worth 19.

## Rule

When a limit and a derived cap are numerically equal, the number cannot say which one
binds — read the `min()`, not the reported total. And never infer build-time free
memory from a later `nvidia-smi`: `empty_cache()` runs in between and is worth 6x here.

## Results

| date | commit | machine | target | model | max ctx | blocks | idle VRAM | peak |
|---|---|---|---|---|---:|---:|---:|---:|
| 2026-09-05 | 6a4e737 | V100-SXM2-32GB | cuda sm70 | 27B NVFP4, draft d1 | 4096 | 256 | 22136 MiB | — |
| 2026-09-05 | 6a4e737 | V100-SXM2-32GB | cuda sm70 | 27B NVFP4, draft d1 | **32768** | **2048** | **25980 MiB** | **27036 MiB** |

**Two claims withdrawn from this entry, both recorded rather than deleted.**

1. **"The resident server is mid-session, so this is unmeasured."** Assumed, never
   read. `/health` gave `finished: 0`, `tokens_generated: 0`, `running: 0` against
   31.5 minutes of uptime, and the log had zero request lines — nothing was in
   session. The numbers were unmeasured because I had not measured them, and the
   premise that excused it needed no defending, which is why it went unchecked. They
   are measured above now.
2. **The 2453-block / 39248-token ceiling**, derived from a `25492 / 32768 MiB` read.
   Superseded by the measured 2048 at cap and #65's 2649 at fit; see the section
   above.

Raw artifacts: `~/serve70c.log` on the pod (`=== boot 0 at 2026-09-05T09:39:13
+08:00 sha 6a4e737 ===`); `/health` `blocks_total` before and after.

That JIT number is also the **`kv_tier` cold-start baseline**, and `kv_tier`'s
before-arm must be **32768**, not 4096 — 8x of the headroom is a flag, and folding it
into offload's credit would overstate offload.

Raw artifacts: `~/serve70c.log` on the pod (`=== boot N ===` lines carry the sha);
`/health` `blocks_total` before and after.
