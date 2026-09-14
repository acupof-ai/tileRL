# Quality audit 2026-09-14

Ten-dimension adversarial audit at baseline `6094e41b` (some findings already
disposed by later PRs; state column marks those). 54 agents: one finder per
dimension, every finding independently re-derived by a skeptic told to refute.
**35 confirmed, 11 refuted.** Confirmed findings only; refuted claims are not
recorded here. Line numbers were accurate at 6094e41b and must be re-read
before a fix.

Legend: 🔴 correctness in production · 🟠 gate/test that cannot see the defect ·
🔵 API surface a client hits · ⚪ resource/data · ⚫ docs/scripts/debt.

## 🔴 Production correctness

1. **Sparse full-prefix hit without a draft hangs forever** —
   `engine.py` zero-residual adoption (`prefill_from == len(tokens)`) is gated
   on `self._draft is not None`; a sparse engine with no draft (`--sparse-k`
   without `--draft`) takes `prefill_from = matched`, no chunk ever forwards,
   the row spins until the 1800 s completion timeout (504), and the slot, hot
   blocks and shared-prefix refs leak. Reachable by resending any
   block-aligned prompt (retries, repeated first turn, n>1). Fix: the
   re-forward-last-page path must run for the no-draft sparse follower too.
2. **Phantom block demand off-by-one at aligned decode crossings** —
   `engine.py _build_plan` growth formula covers through position
   `seq_len+q-1`, but the last physical write is `seq_len+q-2` (position
   `seq_len-1` is a rewrite). At a saturated pool it demands one block that is
   never written (~1/16 of aligned-ending final ticks, trigger
   `(prompt+max_new)%16 == 1`), evicts live prefix entries and finishes the
   request with RequestFailed one tick before completion. The draft loop uses
   the correct `<= seq_len-1` bound. Dense-only.
3. **Completion timeout/error routes return without `engine.cancel`** —
   504/RequestFailed paths in the non-stream routes release the HTTP response
   but the engine row runs to max_new_tokens, holding slot and blocks.
   Distinct from the client-disconnect path fixed in #598: here the client is
   still attached and the server gives up by timeout.
4. **SSE GeneratorExit cancel has no end-to-end gate** — the handler calls
   `engine.cancel` on GeneratorExit; no test drives a real mid-stream socket
   close, so it is unverified behavior (source-read only).

## 🟠 Gates that cannot see the defect

5. The old non-stream disconnect gate cancelled the ASGI task instead of
   feeding `http.disconnect` (mechanism gap fixed by #598, with new
   delayed/at-start/spin gates) — the residual coverage gaps are #3 and #4
   (timeout path, SSE/ws), keep this class in mind.
6. **sm90 fused-prelude "gate" prints instead of asserting** and skips on
   every CI runner: it can never go red.
7. **`/health` lock gates are wall-clock pass/fail with no skip** on a live
   round trip — contradicts the flaky-test inventory.
8. **Distributed `*_world*` gates never run their negative-control flags**;
   the controls proving each gate can fail are manual-only.
9. **`test_layering` does not resolve absolute `from tilerl...` imports**
   although its docstring says it does.
10. **`/ws/chat` has no behavioral disconnect gate**;
    `test_the_routes_cancel_when_the_client_hangs_up` greps source only.

## 🔵 API surface

11. **Tool round-trip transcript destroyed**: `assistant.tool_calls` and
    `role:'tool'` results are not rendered into ChatML.
12. **Responses API silently drops typed `input_text` parts**: documented list
    input renders an empty user turn.
13. **`tool_choice: 'none'` accepted but ignored**: tools still rendered,
    model can still emit a tool call.
14. **reasoning effort caps only on the OpenAI chat route**; the messages and
    responses routes map effort into prompt text with no engine token cap.
15. **Streaming never parses tool calls**: raw `<tool_call>` XML lands in
    content and finish_reason `length`, where the non-stream route returns
    structured `tool_calls`.
16. **Chat route accepts provider-hosted tool types (web_search…) the Responses
    route rejects**, rendering a null-name tool definition.
17. **`submit()` admission is unbounded**: `_waiting` grows without limit and
    waits past the client deadline; no backpressure.

## ⚪ Resource / data

18. **`calibration._row` ignores its target and hardcodes `sm90`**: a V100
    re-calibration writes sm70 measurements into the sm90 population.
19. **Corrupt `KvBootStore` entry raises RuntimeError instead of returning
    None** and fails every running request; blocks leak if aux.pt load fails
    after the copy try-block.
20. **`select_pages` imported from `tilerl_kernels.reference` on the sparse
    hot path**, bypassing the Backend op seam (placement decision owed in
    architecture steps 6/11).
21. **KV fp8 quant/dequant lives in the framework with no sm90 parity gate**
    for its kernel twin.
22. **Quest `page_bounds`/`page_bound_scores` are sm70-only registry cells**
    with no target-neutral CPU twins registered or parity-gated, and the
    Backend methods are unused by production.
23. **Framework reaches into backend privates** `_MAX_VERIFY_W` and
    `Backend._dp_pg`.
24. **`_benchrec` loads `scripts/benchrec.py` by filesystem path**: a wheel or
    sdist install (scripts/ omitted) crashes bench/ledger. Already decided:
    bridge lives in ledger.py with a ponytail marker; package benchrec.py
    after the scripts sweep (architecture "C").
25. **Dead `HostKvPages._evict_to_ssd`**, uncalled since #525 and stale vs
    live spill accounting.

## ⚫ Docs / scripts / tooling

26. Docs index entry counts ~2.3x stale (263 / 127 wins / 136 errors claimed
    vs 592 / 275 / 317 present).
27. README Status says 14 open defects; OPEN.md has 9.
28. design-rl-stack.md says `serve --devices` ships; deleted in #399.
29. design-parallel.md "what already exists" table cites line numbers that no
    longer point at the named symbols.
30. serve-v100.md documents the /health event-loop stall as live after #577
    fixed it.
31. `pod_session_selftest.sh` is red on the clean tree and no gate runs any
    `scripts/*_selftest.sh`.
32. `audit_scripts_entrypoints.py` false-DEAD: its scanner misses
    `spec_from_file_location` path loads.
33. `bench_gil_yield_compare.sh` rewrites `src/tilerl/engine.py` in place;
    both arms are identical after #568 and its restore mutates a tracked file.
34. `verify_h20_fp4.py` / `probe_kv_fp8_27b.py` default `--source` to the
    checkpoint deleted 2026-09-10 and ignore `TILERL_QWEN38_SOURCE`.
35. Completion wait poll interval `0.02` duplicated as independent literals
    across the routes (fold into the 8b unification).

## Refuted (11) — recorded so they are not re-raised

Card-ownership regex drift; BLOCK_TOKENS hardcode in generate/train; wait-loop
"four copies" (three, and 8b already owns them); sparse eager promotions leak;
pad-row half-allocation leak; Engine.shutdown cold-tier leak; AnyIO limiter
starves /health; framework replays `_tl_layout`; pod.sh/k8s/Dockerfile glue
dead; and two review-process meta findings.

## Fix order proposed

- **Now (correctness, before any serving deploy beyond 62adb8c2):** 1, 2, 3.
  Each needs a failing behavioral gate first (1 and 2 CPU-engine gates; 3 a
  route-level timeout-cancels gate), then V100 real-curl verification for 1/2
  alongside the #598 probe.
- Next: API surface 11–16 (client-visible correctness), gates 6–10.
- Fold-ins: 20/23/24 ride the architecture steps; 25 and 33 ride the scripts
  cleanup; 26–30 one docs PR; 35 rides step 8b.
