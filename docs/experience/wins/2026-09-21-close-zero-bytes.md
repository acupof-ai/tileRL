# close 零字节：请求结束不再强制发布 KV — sm70/V100, 2026-09-22 设备验证（finish 有界回退见 #800）

> Status: **device-verdicted 2026-09-22 (V100 sm70, serve 4dd944b4 M3-only;
> the finish-publish restoration is #800 / a38c3bc4).** Scope narrowed after
> M6: a **cancelled/failed** row still moves zero bytes, but a **successful**
> origin finish was given a bounded one-shot publish in #800 because pure
> natural-leave never closed the frontier on the serving geometry (followers
> recomputed, see
> [2026-09-21-finish-publish-restores-serving-adoption.md](../errors/2026-09-21-finish-publish-restores-serving-adoption.md)).
> This win records the deletion of the forced close; the restoration entry
> records the one re-added bounded path. Read both.

## Context

Publish-once refactor M3 (#782): a page reaches the shared prefix only when it
leaves the resident union (`offer_drop`). `_release` used to force the
prompt-end frontier closure at request finish, snapshotting every page the hot
pool never dropped from its live device frame. Under training's disjoint-span
workload those pages had zero followers and each 32k close paid ~1.8 s
ssd_mmap + ~1.8 s host→device transfer for them (that ~1.8s+~1.8s is the
2026-09-19 CLOSETAIL n=6/7 geometry; historical background vendored in
`errors/kv-once-before-close-2026-09-21/close-ticks.tsv`, not this run). The
metric that matters: request-end-tick `ssd_mmap` and the five `pub_*`
segments.

## What Worked

`engine._release` no longer calls the deleted forced `close_request` on every
finish; the unconditional live-frame transfer loop is gone. M6 device
measurement on the M3-only stack (serve 4dd944b4, V100, W2048/4slot/sparse128,
32k): across **n=17 request-end ticks** (identified post-M5 by
`release_cold_forget`, since the `release_close_request`/busyidle bracket was
deleted in #791) every tick carried `ssd_mmap=0` and all five
`pub_bounds_d2h/pub_frame_d2h/pub_draft_clone/pub_share_hold/pub_cold_transfer`
keys absent — zero close-byte violations (artifact `~/closewin-m6/baseline/serve.log`;
#785 comment). This is the "close stops forcing a publish" half, confirmed.

The same M6 window exposed the cost of removing it entirely: serving followers
could not adopt, which #800 later fixed with the bounded successful-finish
publish (cancelled/failed rows remain zero-byte — verified on a38c3bc4).
Natural-leave freeze (`at_first`/`at_prompt_end`) is unchanged.


## Rule

A content-addressed cache publishes at the moment content leaves the owning
pool, never at a request lifecycle event; forcing publication at close bills
every publisher for followers that do not exist.

## Results

M6 device, V100 sm70, Qwen3.8-27B NVFP4, W2048/4slot/sparse128/1GiB-RAM/8GiB-SSD f16.
Two rounds; numbers below are the M3-only close-byte measurement (the
adoption/finish-publish round is the linked errors entry — do not merge the two).

| date | serve sha | round | end ticks | end-tick ssd_mmap / five pub_* | steady model p50 (n) |
|---|---|---|---:|---|---:|
| 2026-09-22 | 4dd944b4 (M3-only) | 32k close window | 17 | 0 on every tick (all 5 keys absent) | 170 ms (n=45, 69d60c77 before-baseline) |
| 2026-09-22 | a38c3bc4 (#800) | post-fix serving | — | cancel row 0; successful finish publishes (see errors entry) | 158 ms (n=36) |

"end tick" = a step-tick carrying `release_cold_forget` (post-#791 marker; the
old `release_close_request` bracket no longer exists). Sources:
`~/closewin-m6/baseline/serve.log` (M3-only), `~/closewin-m6/r800/serve.log`
(#800); before-baseline steady `~/closewin/baseline/steady.json`
(`steady_model_p50_ms`). model p50 is the steady-comparable key (total folds
draft/spec and publish/first ticks). Raw artifacts are V100-local for this
round; the close medians and components are recomputable from
`errors/kv-once-before-close-2026-09-21/close-ticks.tsv` for the 09-21 before data.

