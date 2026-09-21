# M3 删掉 close 发布后，32k+短进场 follower 全量重算；finish 有界回退恢复领养 — V100, 2026-09-22

> Status: **device-verdicted 2026-09-22 on V100, serve a38c3bc4 (#800).** M6 #785
> confirms both halves: a short-advance follower adopts (no long decode), and a
> cancelled/failed row still publishes zero. Measurement provenance and the
> different-geometry wall numbers are recorded below.

## Context

Publish-once M3 (#782, [2026-09-21-close-zero-bytes.md](2026-09-21-close-zero-bytes.md))
deleted the forced prompt-end frontier closure at request finish: a page
reaches the shared prefix only when it leaves the k+window resident union
(`offer_drop`). That is correct for the training disjoint-span workload that
motivated it, but device follow-up on the M6 serving geometry (32028-token
prompt, short follower advance, pool with no pressure) found same-head followers
recomputing the whole prompt.

**Before vs after are DIFFERENT runs/geometries, not an A/B on the same arm**
(both V100 sm70, W2048/4slot/sparse128/1GiB-RAM/8GiB-SSD f16):
- *Before arm* (serve 4dd944b4/e7de6cc1, the broken M3-only stack, full 32k
  prefill each time): **189–246 s follower wall**, zero adoption,
  `sparse_prefix_warm_adoptions=0`, `kv_cold_shared_bytes=0`.
- *After arm* (serve a38c3bc4, `scripts/repro_adopt_796.py`): follower
  **8.6 s** (artifact `d796_adopt.json` key `follower_wall_s`), adopted, not
  re-run. The two wall numbers must not be placed side by side as one arm's
  delta; they are the pre-fix and post-fix rounds.

Device counters over the *before* run: 45328 demotions and 45328 `offer_drop`
calls, then `keys_nonempty_calls=0`, `keys_total=0`, `xfer_calls=0`,
`peak_shared_pages=0` (PROBE796 atexit line, frozen in #796).


## Root Cause

`publish_dropped` closes the grow frontier only to a length m where
`{0..m-1} ⊆ pend` (pages that LEFT the union, with bounds) AND `m ∈ snaps`
(an aligned prefill-chunk boundary GDN snapshot; the snap set is capped to
`{lowest, newest}`). Closure is contiguous from page 0. In the M6 geometry
the prompt is 2001 pages + 12 tokens and sparse_k=128 plus the forced own
window (8 pages) keep the low pages resident; with the pool at 199/2213
blocks nothing evicts them (`evict_victim=0`). After 8 decode tokens the low
pages have never been offered, so no contiguous pending prefix ever reaches
even the lowest surviving snapshot and the closure returns `{}` on every
call. The natural-leave trigger is healthy; the geometry never fires it.

A proposed remediation (b'2) synthesized a prompt-end GDN snapshot from live
recurrent state at finish. A production-path probe showed it was both
unneeded and wrong: the production prefill chunker cuts the unaligned tail
(last-chunk cut to a block boundary, `engine.py:1549-1561`) so the <=15-token
remainder ships as its own forward and a real snapshot exists at the floor
page — for the 32028 prompt, page 2001 = 32016 tokens, tail recompute 12
tokens, not the 1308 a ragged-end analysis predicted. A snapshot written at
finish would instead capture post-decode recurrent state at the prompt
boundary.

## Fix

A scoped revert of one piece of M3, with none of the machinery M3 deleted:

- An origin publisher that finishes successfully closes and publishes its
  prompt prefix synchronously at finish, while its frames, private blobs and
  aligned snapshots are still live, reusing the exact per-page transfer an
  offer uses (resident frames are D2H'd inline). No #741 batch D2H context,
  no #743 background thread/futures/queue, no SSD-lift worker.
- The origin gate is `not failed and sparse_matched == 0`: a cancel, a failed
  row, or an adopted follower never publishes (an adopted row's blobs are
  already shared under the same content keys; re-closing adds a redundant
  entry and refs for zero bytes).
- `close_prompt` takes a `publishable(p)` predicate naming only pages with a
  live source (held private blob or resident frame), so the closure can never
  name a dead content key. A source-less page in the middle does not kill the
  aligned suffix: the closure falls back to the highest snapshot boundary
  below the first gap instead of dropping the whole suffix.
- The index entry attaches BEFORE the per-page transfers, so a spill failure
  partway (a raised SpillWriteError, or a lift that placed no record) would
  otherwise leave a dead entry on the lookup chain. `abort_close` rolls the
  close back on any such failure: the pre-close grow entry is restored, the
  entries the close added are unlinked, one ref per actually-landed key is
  released, and the follower misses instead of dirty-reading; the request
  itself still finishes successfully (the abandoned publish is not a client
  error).
- The prompt-end synthesized snapshot was deleted; the deepest closure is the
  natural chunk-aligned boundary the production chunker already records, and
  the follower re-forwards only the <16-token tail.

CPU gates (`tests/test_sparse_prompt_end_publish.py`, k=128, 2002-page
prompt): short-decode publisher is adopted by an immediate same-head follower
at the exact full closure length (2002 pages, 0-token tail for the aligned
fixture); unaligned 32044-token prompt closes to 2002 pages and the follower
adopts 32032 tokens (12-token recompute); decoding past one chunk still
publishes via the natural chain (the discriminator control); an adopted row
publishes nothing at its own finish; an injected failure on the 3rd finish
transfer (both raise and silent-no-record shapes) leaves no dead entry and a
miss instead of a dirty adopt; plus the hole-fallback unit gate in
test_sparse_engine.py. Full suite 1096 passed.

## Device verdict (2026-09-22, V100, serve a38c3bc4)

`scripts/repro_adopt_796.py` (#799), geometry 16000-word head = 32028 tokens /
2001 full pages + 12-token tail publisher → request-end finish → 128-token-suffix / 48-gen follower
→ streaming cancel. Artifacts on V100 `~/closewin-m6/r800/d796/`
(`d796_adopt.json`, `d796_health_poll.csv` 200 ms rows, `serve.log`,
finish/follower tick extracts). Serve identity: sha a38c3bc4, python pgrep
3713537 (same-minute CSV `serve_pgrep` column; the JSON `serve.sha` field came
through empty — known harness scratch, sha taken from the boot line and
`.synced_commit`).

- **Adoption restored.** After the publisher finish: `kv_cold_shared_pages=1286`,
  `sparse_prefix_entries=2`, `sparse_prefix_published=1`
  (`d796_adopt.json` cold_after_publisher / sparse_after; CSV row t=137.741 is
  the first with `sparse_prefix_warm_adoptions=1`). Follower:
  `sparse_prefix_hits +1`, `sparse_prefix_warm_adoptions +1`,
  `tokens_equal_oracle=true` (temperature 0), follower wall **8.6 s** vs the
  before-round 190–246 s recompute (different rounds, see Context).
- **Cold-tier fill state at verdict:** shared RAM **1.491 GiB / 1286 pages**,
  shared SSD **0 B** (`cold_after_publisher`: shared_gib 1.491,
  shared_ssd_gib 0.0; `kv_cold_shared_ssd_bytes=0`) — under the
  1 GiB-RAM/8 GiB-SSD budget the shared set landed in RAM, no SSD this run;
  private/private-SSD 0. Block pool was not under pressure (sparse residency
  small at k=128, consistent with the #785 finding that evict_victim is
  device-unreachable here).
- **Finish-tick byte cost (the bounded revert, by design at successful finish):**
  finish tick total 1559 ms with `pub_bounds_d2h=149`,
  `pub_draft_clone=208`, `pub_frame_d2h=770`, `pub_share_hold=7`,
  `pub_cold_transfer=22` ms and `ssd_mmap=0` (serve.log finish tick extract).
  This is a one-time per-publisher finish publish, not the old per-close cost on
  every request; the device-copy envelope is the same order as the pre-fix
  static estimate (~2.2 GiB), observed shared fill 1.49 GiB.
- **Unaligned tail recomputed:** 32028 tokens = 2001 full pages + 12 tokens;
  bounds exist only on whole pages, so the follower re-forwards the 12-token
  tail by design (the ~2 s follower prefill ticks 155/156 are the 128-token
  suffix + this tail, not a head recompute — consistent with 8.6 s total).
- **No steady-state regression:** sparse eager decode **model p50 158 ms**
  (n=36 model-only ticks) vs the 69d60c77 before-baseline steady
  `steady_model_p50_ms=170` (n=45) — same band. The current run's total p50
  ~237 ms is NOT comparable: it folds draft/spec activity and the
  publish/first ticks; model-only is the steady-comparable key. Graph-path
  decode p50 44 ms.
- **Cancel still zero publish:** the streaming-disconnect row's end ticks
  (159/164) carry `release_cold_forget` 21/10 ms and zero `pub_*`/`ssd_mmap`;
  `close_zero_on_cancel=true`, zero matching publish ticks. The origin gate
  (`not failed and sparse_matched==0`) holds.

A harness self-report bug surfaced and is fixed separately in #801: the gate
read a log `sparse_matched=N` field the engine never prints, so it self-reported
FAIL despite counters +1; the gate now uses the `sparse_prefix_*` counter
deltas. Adoption itself was never in doubt (counters + wall prove it).

## Rule

Delete a forced lifecycle publish when the workload has no followers, but
keep the geometry where a request ends with resident prefix that an immediate
same-head follower needs: a content cache may publish at finish while the
content is still live, gated on origin identity and on a live source per page
— it must not synthesize state the forward path did not record, and must not
name a key it cannot land.
