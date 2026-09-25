# docs/

Design notes, the roadmap, and the measurement archive. Every perf or accuracy
number in this repo traces to a dated entry under `experience/`; nothing here
states a number it did not measure.

## Design — how it works and why

| Doc | Answers |
|---|---|
| [quality-audit-2026-09-14.md](quality-audit-2026-09-14.md) | Ten-dimension adversarial audit, 35 confirmed findings ranked for the wrap-up: two production defects, gate gaps, API surface, resource/data, docs/scripts. |
| [design-architecture.md](design-architecture.md) | Target module layout for the wrap-up refactor: seven layers, imports point down only, one assembler, the hybrid `SparseRuntime` seam, and the PR sequence with its gates. |
| [design-engine.md](design-engine.md) | The four layers and the one seam each — frontend, scheduling, model, storage. Why the decode tick is a captured kernel sequence, why prefix sharing is read-only, and the physics the design has to satisfy. |
| [design-kernels.md](design-kernels.md) | The kernel tree's file contracts, the registry rule (arch cell = CPU floor + overrides), the SOTA-copy provenance header, and the precision-before-tiles order a perf campaign follows. |
| [design-rl-stack.md](design-rl-stack.md) | The three pieces of the RL product: the ISO optimizer and merger, the DFlash2 draft head and what keeps it on-policy, and the ledger CLI an agent drives. Marks what is settled and what is not. |
| [design-rl-architecture.md](design-rl-architecture.md) | Why the RL training stack is shaped this way: what wall clock a given score costs and which architecture minimises it, under the objective set 2026-09-08. |
| [design-cost-model.md](design-cost-model.md) | The one primitive that prices every byte and kernel, from which card, host and SSD occupancy derive, under the invariant `peak = sum(static rows, derived) + transient`. |
| [design-sparse-kv.md](design-sparse-kv.md) | Converting a dense checkpoint so its full-attention layers attend to a selected page subset, with a per-token index key on device and the rest of the KV in host RAM or on SSD. |
| [design-sparse-offgraph-staging.md](design-sparse-offgraph-staging.md) | Device-resident off-graph staging for the sparse captured decode (D-0 → D-3): D-0 shipped under CPU byte-equal coverage, D-1..D-3 pending device acceptance. |
| [design-parallel.md](design-parallel.md) | Design only, reviewed before implementation: tensor parallelism first, context/sequence parallelism after, and what the tree already provides. |
| [design-idempotency-key.md](design-idempotency-key.md) | A read-only, unscheduled proposal: a client-generated idempotency key that reattaches a retry to an in-flight request, dependent on the post-#672 server shape. |
| [support-matrix.md](support-matrix.md) | Per-op, per-target status — cpu, sm90, sm100, metal — for bf16, fp4 and fp8. A cell is `done` only if it ran, never because it compiled. `registry.py` is the source of truth; this mirrors it. |

## Comparisons and assessments

| Doc | Answers |
|---|---|
| [rl-sota-parity.md](rl-sota-parity.md) | Our RL loop read against TRL and AReaL at source level, default by default, each difference marked chosen or missed. The on-policy discipline is stricter than both; memory is where they did engineering and we did none. |
| [design-pd-afd.md](design-pd-afd.md) | Design only. PD and attention/FFN disaggregation as deployment topologies over the existing seams — what each would need, and why PD comes first. |
| [design-gdn2.md](design-gdn2.md) | Adoption assessment for GatedDeltaNet-2. Algorithm copyable, code not (NC licence, Triton). No GDN2 checkpoint in scope, so YAGNI until one appears. |

## Where it is going

[roadmap.md](roadmap.md) — the north star, a dated "where we are" table with
evidence links, and phases P1–P6. Phases exit on a named measurable event,
never a date; a gate needing a GPU not in hand ships `pending-remote` and does
not claim the number. `CHANGELOG.md` at the repo root is the running record —
phase exits, default flips, accept-or-reject verdicts.

## The archive

[experience/](experience/) — 639 dated entries, one measurement each, wins and
rejections both. Start at [experience/README.md](experience/README.md), which
picks the ~27 that carry the findings the rest of the repo rests on.

[analysis/](analysis/) — the cross-cutting write-ups, where a question is asked
of the whole system rather than one change: the [sglang
comparison](analysis/2026-08-28-vs-sglang-h20.md), [is the gap to Arle the price
of TileLang?](analysis/2026-08-27-tilelang-vs-native.md), the [adversarial
defect audit](analysis/2026-08-27-defect-audit.md) (historical), the [method record behind
decode 52.6 → 90.9](analysis/2026-08-28-decode-52-to-84.md) (historical), the [pod
verification](analysis/2026-08-27-pod-verification.md) (historical), and [what closing the
prefill gap would actually require](analysis/2026-08-29-what-sota-would-require.md).

[bench-schema.md](bench-schema.md) — one measurement is one record: the
append-only store `docs/experience/bench/measurements.jsonl`, one JSON object
per line, and why a record missing a required field is rejected at write time.
[bench-inventory.md](bench-inventory.md) — the collector map and the
Keep/Delete triage, now consumed (its residue: what the registry cannot hold);
`docs/bench-metrics.json` is the live collector map.

## History

Design docs and work-allocation tables that the live tree no longer needs but
that later docs and CHANGELOG lines cite as provenance.

| Doc | Answers |
|---|---|
| [arch-review-2026-09-09.md](history/arch-review-2026-09-09.md) | An architecture review against `fa1bcec` (2026-09-09): what to delete, what to build, and what "AI friendly" means, with line counts for scripts, src+kernels and tests. |
| [design-ssd-read-path.md](history/design-ssd-read-path.md) | The SSD read path as designed (async-with-deadline) — SUPERSEDED 2026-09-09 (a daemon reader thread shipped instead) and REMOVED 2026-09-14 with the dense KV tier. |
| [ownership-tables.md](history/ownership-tables.md) | Work-allocation tables that sat in the live design docs while the units were open; all units landed, kept for attribution. |

**Docs cited by CHANGELOG cannot be deleted.** CHANGELOG is the central record;
each line points to its evidence. Deleting the evidence leaves the verdict as a
bare assertion. Mark stale docs SUPERSEDED at the top of the file instead — the
mark says what replaced it, why it still exists, and where the current answer
lives. Checkable: `git grep -c "<doc-path>" origin/main -- CHANGELOG.md` —
non-zero means SUPERSEDED, not delete.

## Operations

| Doc | Answers |
|---|---|
| [serve-v100.md](serve-v100.md) | Running the 27B on the pod with the chat UI on a laptop — the SSH tunnel, the exact server command, and why warmup captures the decode graphs up front. |
| [run-close-window-v100.md](run-close-window-v100.md) | The one-key V100 close-tail window harness: the arm table, the health-body gate that refuses a wrong server, the instrumentation preset, and the recovery checklist. |
| [serve-h20.md](serve-h20.md) | The sm90 sparse+d1+decode-graph supervisor run through `pod_run.sh` — the `/work/tl013` cu129 interpreter (not `uv run`), the in-checkpoint draft, and the parameterized cold-spill path. Frozen with H20 (stopped 2026-09-16). |
| [serve-cold-prefill-cap.md](serve-cold-prefill-cap.md) | When to raise the sparse prefill chunk from the shared-server default 192 to 512 (dedicated/offline long-context cold fill), the two zero-code ways to do it, and why concurrent serving must keep 192. First-token latency only. |
| [measurement-window-v100.md](measurement-window-v100.md) | How to run one V100 measurement window: arm order with the control at both ends, the env-only arm rule (and which knobs are import-time so cannot flip mid-process), silencing the liveness watchdog, boot and stop gates, the four-key cold-tier gate, and the vendoring discipline. Procedure only — results live in `experience/`. |
| [run-h20-arms.md](run-h20-arms.md) | The H20 (sm90) 32k arm matrix: six arms with the serve env each pins, the two client protocols (the wikitext-train corpus pass for A1–A5, the cold-fill V100 wall protocol for A6), and the reading rules — the standard steady set, the two rates, the four-key cold gate, and why an arm's graph mode is read from `/health` rather than the boot banner. |
| [api-compat-surface.md](api-compat-surface.md) | The completion routes sharing one engine — OpenAI Chat Completions, Anthropic Messages, OpenAI Responses and the playground's own WS transport — the vendor shape each presents, and the named deviations. |
| [lessons/](lessons/) | Notes on driving Claude Code against tileRL's own server (the Messages shim, now `src/tilerl/messages.py`; the rollout launcher's sandbox, deleted with `tests/test_rollout.py` in #594) and cross-cutting test/audit guidance such as why a name-based dead-code scan false-positives on framework-reflected symbols and registry/accounting names. No longer only the two measured Q&A notes; kept as CHANGELOG evidence. |
| [tick-anatomy.html](tick-anatomy.html) | A rendered page: every layer of one V100 speculative decode tick against its byte floor. |
