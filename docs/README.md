# docs/

Design notes, the roadmap, and the measurement archive. Every perf or accuracy
number in this repo traces to a dated entry under `experience/`; nothing here
states a number it did not measure.

## Design — how it works and why

| Doc | Answers |
|---|---|
| [design-engine.md](design-engine.md) | The four layers and the one seam each — frontend, scheduling, model, storage. Why the decode tick is a captured kernel sequence, why prefix sharing is COW, and the physics the design has to satisfy. |
| [design-kernels.md](design-kernels.md) | The kernel tree's file contracts, the registry rule (arch cell = CPU floor + overrides), the SOTA-copy provenance header, and the precision-before-tiles order a perf campaign follows. |
| [design-rl-stack.md](design-rl-stack.md) | The three pieces of the RL product: the ISO optimizer and merger, the DFlash2 draft head and what keeps it on-policy, and the ledger CLI an agent drives. Marks what is settled and what is not. |
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

[experience/](experience/) — 263 dated entries, one measurement each, wins and
rejections both. Start at [experience/README.md](experience/README.md), which
picks the ~24 that carry the findings the rest of the repo rests on.

[analysis/](analysis/) — the cross-cutting write-ups, where a question is asked
of the whole system rather than one change: the [sglang
comparison](analysis/2026-08-28-vs-sglang-h20.md), [is the gap to Arle the price
of TileLang?](analysis/2026-08-27-tilelang-vs-native.md), the [adversarial
defect audit](analysis/2026-08-27-defect-audit.md), the [method record behind
decode 52.6 → 90.9](analysis/2026-08-28-decode-52-to-84.md), the [pod
verification](analysis/2026-08-27-pod-verification.md), and [what closing the
prefill gap would actually require](analysis/2026-08-29-what-sota-would-require.md).

## Operations

| Doc | Answers |
|---|---|
| [serve-v100.md](serve-v100.md) | Running the 27B on the pod with the chat UI on a laptop — the SSH tunnel, the exact server command, and why warmup captures the decode graphs up front. |
| [lessons/](lessons/) | Two measured Q&A notes on driving Claude Code against tileRL's own server: the Messages shim, and the rollout launcher's sandbox. |
| [tick-anatomy.html](tick-anatomy.html) | A rendered page: every layer of one V100 speculative decode tick against its byte floor. |
