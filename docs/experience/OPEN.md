# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| [sm70 32k per-request cold SSD close blocks the close path ~3.1 s](errors/2026-09-19-sm70-32k-decode-physical-wall-and-ssd-close-tail.md) | `HostKvPages` cold shared-page transfer at request close (`release_close_request` / `ssd_mmap` / `pub_cold_transfer`, `src/tilerl/kv_tiers.py`, `src/tilerl/sparse_engine.py`) | V100 1 GiB-RAM/8 GiB-SSD cold tier: steady 32k decode is a ~168 ms model trunk (physical wall, ~9.5–10.7 effective tok/s), but once per request the close pays ssd_mmap 1.81–2.03 s + pub_cold_transfer 1.71–1.87 s (release_close_request ≈3.1 s; >300 ms ticks are 5.9% of warm d1 ticks and 7.1–15.4% of d3). Fix (open): move the cold shared-page SSD transfer off the request-close path — amortize, background, or defer it so close is not gated on mmap writeback |
| [shared prefix SSD spill ignores --cold-ssd-bytes and the extent file never shrinks](errors/2026-09-19-sm70-32k-decode-physical-wall-and-ssd-close-tail.md) | `ColdSsdFile` shared `.prefix.bin` spill (`src/tilerl/kv_tiers.py`) | the shared prefix cache publishes pages nothing serves; the shared spill file is not bounded by `--cold-ssd-bytes` (13.6 GiB logical / 25 GiB physical high-water after 2×2 37.6k fills vs an 8 GiB cap), and #735 extent growth frees slots for reuse but never hole-punches, so disk is only reclaimed by deleting the file with the serve stopped. **Cap + trailing-extent truncation LANDED env-gated/default-off (#740, `TILERL_COLD_PREFIX_SSD_CAP=1`; over-cap pages stay in host RAM, freed fully-empty trailing extents ftruncate back). Remaining: mid-file hole-punch (`FALLOC_FL_PUNCH_HOLE`/`F_PUNCHHOLE`) for non-trailing free extents, device confirmation of the disk reclaim/RAM trade, and whether to flip the default on** |
| [draft read window W=2048 end-to-end not significant](errors/2026-09-19-w2048-window-end-to-end-not-significant.md) | `DraftHead` trailing read window vs sparse d1 decode wall time (`spec.py` / `scripts/probe_draft_window_sweep.py`) | n=30 V100 sm70: the high-position decode draft step is 6–10× cheaper at W=1024/2048 (32k 118→10/12 ms, 16k 59→10/12 ms) and acceptance cost is inside the registered 3%/0.04 gate, but end-to-end vs a repeated W=0 bracket is only +15.6% (32k) / +10.4% (16k; bracket drifts +55%), below the +20% bar. Default stays W=0. Named-not-landed fix: (1) the cold-tier O(1) finalize + per-tick lock work (see 2026-09-17/18 cold-tier and long-step-tick rows) so the per-step saving clears the jitter band; (2) a sweep that records per-prompt tok_s (IQR/p10–p90), then re-measure W |
| [a full cold tier makes sm70 sparse 32k decode long-tail](errors/2026-09-17-cold-tier-full-finalize-relocation-long-tail.md) | KV cold tier at 8 GB / device_free ~200 MB; `sparse_finalize` page moves + unattributed hollow forward | warm 32k serves 7.86 tok/s with an empty cold tier but 2.69–6.56 full (1.4–2.95 worst); p50 tick healthy ~175–190 ms but 40–58% of full-tier ticks >300 ms. Dual mechanism: a ~120–144-page finalize batch costs 81→629–667 ms (7–8x) at full tier; plus 1.1–1.4 s hollow forward ticks (offers_pages=0, inner segments unaccounted). Isolated by cold-tier fill at fixed W=2048; independent of #698/#695/#697. Fix (open, undecided): revisit kv-cold-bytes / the #654/#656 shrink-realloc floor and/or cap 32k concurrency; add a per-tick alloc/sync probe to attribute the hollow ticks |
| [sm70 dense+d1 occasionally holds engine._lock for 1–5.6 s in one step tick](errors/2026-09-15-sm70-long-step-tick-holds-engine-lock.md) | `src/tilerl/engine.py` step tick / `engine._lock` (suspected CUDA allocator at ~350 MiB free) | ~12% of late-tick forwards hold the lock 1–5.6 s (V100 sm70 dense+d1, 72df5351), freezing synchronous engine callers (the SSE cancel caller was moved off the loop; the tick is unfixed). Self-limiting (returns); the permanent-hang sibling is the row above. No JIT/eager-fallback/sparse/demote evidence; allocator suspected, unproven. Fix: in-engine segmented phase timing + memory_stats at >500 ms ticks to localize allocator vs kernel vs plan |
| [sm70 sparse decode graph + MTP d1 corrupts multi-token output](errors/2026-09-14-sm70-sparse-graph-mtp-corrupts-output.md) | sparse lazy graph capture keyed on `(B,W,cmax,own_w)` | sparse+d1+`--decode-graph` deterministically corrupts decode and hit illegal access at B=4; eager and dense+graph exact. H1 (warmup scribbled live block 0) disproved after #585. **H2 (capture at a cmax-bucket transition) TESTED BAD on device 2026-09-17 (#700, 220b3c95): first replay corrupts the first token at 6/6 sparse bucket×W** — see [2026-09-17 first-token entry](errors/2026-09-17-sm70-sparse-decode-graph-replay-corrupts-first-token.md); the B=4 illegal-access arm did not form (prefill-cap stagger) and is moot while the graph stays off. Hybrid #586 works around it by forcing sparse ticks eager. Open fix: make a new-bucket capture's first replay match eager on live shapes |
| [a short request during a long sparse prefill runs at 5.59 tok/s](errors/2026-09-14-short-request-fill-bound-during-sparse-prefill.md) | device queue shared by sparse prefill and dense decode | a short dense request during a unique 128k sparse fill: 5.59 tok/s / 3.2 s TTFT vs ~52 solo (32k point was 9.9 tok/s / 2.5 s); sparse tick ~1 s median on sm70. Fairness and slots are fine; a dense tick syncs behind the in-flight prefill kernel. cap 96 shortens waits at +124% prefill time, rejected. Fix: sparse prefill kernel or chunking that lets dense decode pass the fill |
| [an sm90 B=8 spec wave is not reproducible across identical cold waves](errors/2026-09-13-sm90-b8-spec-wave-not-reproducible.md) | owner cc; sm90 B>1 sparse spec tick (packed prefill/verify or batched draft) | two identical fresh COLD B=8 waves agree only **3/8** (rows 0/1/5/6/7 diverge at tokens 27/30/31/5/30) under the ACTIVE #563 unfused guard; B=1 sequential warm-vs-cold is 8/8 exact (logits max_abs 0.0) and CPU tiny B=8 is 8/8. Not the fused-routing bug (#567, guarded off here): reproducibility defect, sparse-vs-sparse, needs a served-shape per-row first-logit dump. **pending-remote: `scripts/pod_run.sh warmctl <h20> -- bash scripts/probe_warm_control_run.sh`** — deferred (H20 unavailable by decision 2026-09-14). |
| [the training rollout tick is 2.6x serving](errors/2026-09-08-the-training-rollout-tick-is-2.6x-serving.md) | `src/tilerl/train.py` `grpo_loop` rollout (`rollout_secs`) vs `build_engine` config | two full-27B B=8 decode measurements disagree **2.64x** with the slower one on the *shorter* context — 55.21 and 61.68 ms/tick training (two shas, two cards, agreeing to 1.12x) against 23.34 serving, all graph-on, all `wall / decode_forwards`. Worth **45.9% of a GRPO step** if it closed and **93% unattributed**: LoRA is ≤4% of the gap (0.537 GFLOP/tick is 0.054 ms even at 10 TFLOP/s; 768 launches inside a captured graph is 1.54 ms) and `fuse_projections` 2.9%. The reason on record was false — `_training_kv` is at `train.py:160` in the forward/backward, not the rollout, and the serving arm was itself W=1 no-draft. **Arm ran on V100 sm70 (main d1b56684, 2026-09-14): NO GAP** — train 10.26 vs serve 19.44 ms/tok, train *faster* (gap 0.528x, recorded was 2.643x train-slower); fwd/tok identical, residual ~0, device time moves with pool blocks the wrong way. The config bundle does not reproduce it on sm70; recorded pair was graph-on on sm90 and this box auto-disables capture. **Next arm needs sm90: graph-on vs eager across both configs (one process/one weights); LoRA held out as its own follow-up.** Deferred (H20/sm90 unavailable by decision 2026-09-14). |
| [a miss self-reinforces](errors/2026-09-07-a-miss-self-reinforces.md) | `src/tilerl/engine.py` `_finish_prefills` | **Publisher half fixed** — the first interior boundary plus the last, a constant 2 publishes at any prompt length ([wins/2026-09-08](wins/2026-09-08-cut-the-prefill-publish-flood.md)). **Open half: V100 alternation** the cell was run to check: turn-0 hits on every other conversation, which did not reproduce on the H20 and needs the V100 grid with the per-row instrument |
| [entries-per-row against the snapshot budget](errors/2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md) | `src/tilerl/engine.py` `_finish_prefills` publish gate | #271 traded 62 nested entries for 2 and costs the DRAM-tier cell **1.81–2.12x** on the parent-child pair `45acd87`→`a43a379` (16 lines). Measured: TTFT is **linear in the tokens a hit did not match**, R² 0.998, predicting the unfitted miss row to −6.8%; and depth is a **512×k ratchet in submission order**, so each session's match is set by its queue position, earliest worst at 1.7%. Sign flips with turn depth (+259 s at turns 0–1, −54 s at turn 2), so **both endpoints of the count axis lose**. The two sizing constraints **conflict**: `K × snapshot_bytes ≤ budget` gives K ≈ 1 here while the depth threshold admits K ≈ 4–11. Arm: same pair at a budget where K ≥ 2 fits, in TTFT — after the fixed term is pinned |
| [a hit carries 1.2–2.5 s of depth-independent cost](errors/2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md) | unattributed; `src/tilerl/engine.py` `_admit` hit path | the fixed term of the hit-cost model, paid at any match length. **A range, not a number**: the data's unmatched fraction bottoms out at 0.437, so the constant is extrapolated 44% past the nearest point — the three per-turn lines agree to 6.1% mid-range and to 75.9% at zero. Extra forwards explain 0.15–0.39 s (+22 to +58 forwards over the arithmetic minimum); the state restore has the right shape (a snapshot is constant size at any prefix length, `kv_cache.py:1434`) and is 0.17 ms at HBM bandwidth against ~1200–2500 ms. Fix: one run with turns held constant (one turn, more sessions) pins it without the covariate. It caps how closely a publisher ladder can space rungs, so the ladder cannot be priced until it is |
| GDN state differs when prefix-hit and fresh-prefill use different chunk sizes | `src/tilerl/engine.py` `_build_plan` first-chunk cut; GDN parallel scan | When a prefix hit restores at a boundary the warm-up reached with chunk size A and the fresh prefill reaches with chunk size B ≠ A, the GDN states differ by ~3e-4 (2.5% relative on near-zero elements). Source: the parallel scan rounds per chunk length. **CPU/metal reference path: bounded.** `test_gdn_chunk_rounding_bound` (PR #445) measures the worst per-element violation ratio against the consumer's `allclose(rtol=1e-2, atol=1e-5)` over a 4×5 chunk×seq grid, three seeds: ≤1.20e-2 on CPU (bound 2.4e-2). **CUDA: still unbounded** — the recorded defect is H20 card 1 ratio ~2.0, 118× above the CPU bound, there is no scheduled CUDA runner by design (CI CPU-only, GPU manual via scripts/pod_run.sh), so it waits for a named card window: `TILERL_TARGET=cuda uv run pytest tests/test_ops_parity.py::test_gdn_chunk_rounding_bound -k cuda` on the pod. Measured 2026-09-10, H20 card 1, see [errors/2026-09-10-gdn-state-chunk-size-rounding.md](errors/2026-09-10-gdn-state-chunk-size-rounding.md). The gate bounds the torch reference, not the sm90 kernel — same structure, different arithmetic |
| [a non-stream 128k request 504s on the fixed 30-min completion timeout](errors/2026-09-16-nonstream-128k-hits-fixed-completion-timeout-504.md) | `_COMPLETION_TIMEOUT_S` (`src/tilerl/messages.py`), `prompt.await_completion` / `server.await_or_cancel` | #658 device gate (V100 0cc82a36): the streaming 128k cold sparse fill completed 200/stop in 1037 s (~118 prefill tok/s, 333 MiB SSD spill); the **non-stream** arm hit the fixed 30-min `await_completion` deadline and 504'd, though the cold sparse prefill legitimately runs ~17.3 min on sm70 and a longer/colder fill crosses by design. Not the wedge — the loop is healthy and the row progresses; only the fixed await deadline fires. Fix LANDED: configurable `--completion-timeout-s` / `TILERL_COMPLETION_TIMEOUT_S` (default 1800 unchanged; 0 = no deadline, disconnect still interrupts) on the three non-stream routes, stream/ws unaffected. Closes when the long-context serve raises/zeros the cap and a non-stream 128k fill is confirmed 200 on device (or callers use `stream=true`) |

| [a result cache without a code version makes two same-config runs start from different states](errors/2026-09-09-a-result-cache-without-a-code-version.md) | before-arm eval cache — `_before_eval_key` and the cache read in `src/tilerl/train.py` (`runs/eval-cache/<key>.json`) | a run's pre-step-1 engine state is decided by the tree's cache file, not by any recorded config field: a reused tree hits (clean engine at step 1, 0 eval requests) and a fresh tree always misses (1500 requests, pre-warmed engine), so worktree age decides the training trajectory and the only witness is one `eval_before_cache.cache_hit` line. The key has no code-version dimension and the cache lives in the tree. The entry's named fix is a policy decision and neither half landed: (a) an unconditional fixed warmup before training, same shape and count on both paths, or (b) separate engine instances for eval and training. Mitigations 1 (refuse to train on a hit whose paying sha differs) and 2 (sha in the key) are also unshipped, and sha-in-key alone would not fix same-sha reruns. Confounded, not refuted, by [the decode graph's run-to-run nondeterminism](errors/2026-09-09-decode-graph-run-to-run-nondeterminism.md) |
| [the collapse passes through the tool-call modality](errors/2026-09-09-the-collapse-passes-through-tool-call-modality.md) | `src/tilerl/train.py` `grpo_loop` / the `live=[len(c)>0]` mask; step-32 onset unattributed | the absorption half is explained (the mask drops empty completions), the onset half is not: first empty completions land at step 32 and the run stays empty to step 100, and the empty collapse **replicated** on the eval-every-25 control at the same step-32 length trajectory (`prereg.jsonl` `check3-eval25-ctl`) — so the same-day sibling non-replication (a step-75 score dip in a different run) does not close it. Both named experiments still owe their run: the 3-step before-arm A/B and the queued 100-step graph-off run. Not touched by the later `lam>0` length-term fix — this run is `lam=0` |

## Closed by triage 2026-09-14

**think-off 50 tok/s target withdrawn** — ckl closed it 2026-09-14: think-off solo
48.1 tok/s, the gap is 4%, the request path works, and the target is not worth a
bimodal-timing chase while serving is stable. See
[errors/2026-09-14-think-off-single-request-below-target.md](errors/2026-09-14-think-off-single-request-below-target.md).


**`--eval-curve-n` 20 cannot resolve the effect** — closed: the run now refuses at start when the subset's worst-case binomial SE exceeds `--curve-target-pt` (default 5.0 pt), instead of warning after the run. See [errors/2026-09-08-a-default-that-cannot-resolve-its-own-effect.md](errors/2026-09-08-a-default-that-cannot-resolve-its-own-effect.md).

**CUDA tests have no regular runner** — closed as a recorded design decision: `.github/workflows/ci.yml` gates CPU-only (GPU tests auto-skip on the GPU-less CI box) and device runs are manual on the pod through `scripts/pod_run.sh` (CLAUDE.md "CI"). The four failures named when the row was written (2026-09-10, commit `7ce53e77`) are all disposed: the submit-rollback and ragged-prompt-state failures are struck below (a test seam the CUDA graph never crosses; a chunk-rounding artifact), the fp8-pool failure was the test hardcoding the pre-pad block count, and the prefetched-hit "failure" was the test correctly refusing a vacuous pass. The one live defect needing a CUDA run — the GDN chunk-rounding bound on the sm90 kernel — stays on the board as its own row above. A scheduled pod GPU cron is a future decision, not work in flight.

## Struck, not fixed — claims that were measured and turned out not to be defects

A row deleted with no record invites the same reasoning back. These were on the list above
and came off because driving the code refuted them, not because a fix landed.

**`_demote_one` is unreachable under count pressure** — `src/tilerl/kv_cache.py:1225`,
struck 2026-09-08. The reading was that the `len(self._by_id) <= self.capacity` guard sits
inside a `while len > capacity or bytes > budget` loop, so the guard can never be true and
the store evicts where it could demote. That holds only on the branch where the loop is
entered on the **count** term; under byte-only pressure the count term is not over, the
guard is true, and demote runs. Driving the real `PrefixStore` with a `DramSnapshots` tier,
512-byte snapshots, three pressures:

| pressure | entries | demoted | evictions |
|---|---:|---:|---:|
| byte only — `capacity=64`, `state_bytes=1200`, 6 published | 6 | **4** | **0** |
| count only — `capacity=3`, `state_bytes=1 GiB`, 6 published | 3 | 0 | 3 |
| both — `capacity=4`, `state_bytes=1200`, 8 published | 4 | **2** | 4 |
| boundary — `capacity=9`, `state_bytes=2500`, 9 published | 9 | **6** | **0** |

The first row is the refutation: if the guard made the path unreachable, 4 demotions and 0
evictions is impossible. The guard is also not conservative — `_demote_one` leaves the entry
in `_by_id`, so a demote cannot shrink the count term and only eviction can satisfy it.
Removing the guard would demote every entry to the host tier under count pressure, find the
count unchanged, and evict them anyway: one wasted tier write plus a `forget` per entry.

The last row is `len == capacity` exactly, at different constants (800-byte snapshots,
16-token prefixes, a 4096-block pool so the block term cannot participate) — the boundary
where "the guard is always false" should hold if it holds anywhere, and it does not.

**The block axis is a third term, struck 2026-09-08 on the same argument.**
`evict_until_free` (`kv_cache.py:1330`) has no demote branch, which reads as the same
omission on another axis. It is not: `_demote_one` touches neither the block pool nor
`_by_id`, so a demote frees **zero** blocks and there is nothing the branch could do.
Confirmed: a forced `_demote_one` moved `dram.demotions` 3→4 and `free_blocks` 58→58. Why,
at length, in
[a missing branch that cannot satisfy its own loop](errors/2026-09-08-a-missing-branch-that-cannot-satisfy-its-loop.md).

The reasoning was self-consistent and no step in it was wrong. It described one branch while
claiming something about all of them, and nobody ran the store. Two sessions read the shape;
three rows answered it in minutes.

**`test_submit_rollback_and_terminal_failure` DID NOT RAISE on CUDA** —
`tests/test_e2e.py`, struck 2026-09-10. The reading was that a path which should fail did
not: the test patches `engine._model.forward` to raise, and the pod run saw
`DID NOT RAISE RuntimeError` where CPU raised — an engine error-path defect, CUDA-specific.
Driving the test with entry counters on card 1, tree `73e095cb`:

| | `_model.forward` entered | `_run_decode_graph` entered | `step()` | `take(1)` | blocks/slots |
|---|---:|---:|---|---|---:|
| CUDA | 0 | 1 | no raise | None | 1/1 |
| CPU | 1 | 0 | raise | RequestFailed | 0/0 |

The patched seam is entered 0 times on the card: a CUDA pure-decode tick replays a captured
graph through `_run_decode_graph` and never calls `_model.forward`, so the failure was never
injected. The engine's handler was never reached — no engine line changed; the fix landed in
the test, which now patches both seams. The red was loud by construction: `pytest.raises`
fails when the seam it patches stops being called, unlike an "output changed" assertion,
which goes green in the same situation — 9b's fp8 seam, the same root, was silent. Why, at
length, in
[a test patched a seam the CUDA tick never crosses](errors/2026-09-10-a-test-patched-a-seam-the-cuda-tick-never-crosses.md).

**`test_a_ragged_prompt_publishes_and_its_state_matches_no_store` failed on CUDA with
`max|delta| 3.586e-04, norms 43.6672 vs 43.6672`** — struck 2026-09-10. The reading was a
silent correctness issue on the GDN recursive-state restore path. It is not: the restore is
an exact `clone()` → `copy_()`, and both arms are individually bit-identical across reruns.
The difference is between the arms, not within them. The warm-up prompt (100 tokens)
produces a first prefill chunk of 64; the reference (164 tokens) produces 128. The GDN
kernel's parallel scan rounds differently per chunk length, so the state at the restore
point (token 64) is the **end** of a 64-token chunk in one arm and an **intermediate point**
in a 128-token chunk in the other. The deterministic ~3e-04 delta propagates to token 164.
Two near-zero elements fail `allclose(rtol=1e-2, atol=1e-5)`:

| element | ref | hit | delta | tolerance |
|---|---|---|---|---|
| (0,0,2,5) | -1.184e-02 | -1.154e-02 | 2.99e-04 | 1.28e-04 |
| (0,0,3,5) | -9.560e-03 | -9.773e-03 | 2.13e-04 | 1.06e-04 |

The max-delta element (ref=5.05, delta=3.59e-04) passes with tolerance 0.0505 — the failure
is entirely at near-zero elements where `rtol` gives a tight bound. Fix: match chunk sizes
(`short, long = 66, 100`), making the states bit-identical. The chunk-size-dependent
rounding itself is real and uncovered — see the new OPEN row above. Full analysis in
[errors/2026-09-10-gdn-state-chunk-size-rounding.md](errors/2026-09-10-gdn-state-chunk-size-rounding.md).

