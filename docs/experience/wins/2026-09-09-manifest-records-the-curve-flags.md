# The manifest records the curve and patience flags — 2026-09-09

## Context

9b needed to confirm a running real-run's flags and could not: the manifest records
`eval_curve_seed` but not `--eval-every`, `--eval-curve-n`, `--patience` or
`--patience-mode`, so the only evidence was `/proc/<pid>/cmdline` — which dies with the
process. The flags decide which problems the curve scores, how dense it is, and where the
run stops.

## What changed

All four join the manifest inputs dict — and therefore the run id, since the id is the
sha256 of that dict. The reuse rule ("same inputs = same run; a finished one is returned
instead of retrained") makes this a safety property, not bookkeeping: a patience on/off
comparison pair sharing an id would hand the second run the first's finished manifest,
silently, and that pair is the evidence for the `--patience` default-flip decision. The
same rule already put `tp`, `reward`, `length_penalty` and `eval_curve_seed` in the id.

**Cost, bounded and accepted:** every existing run's id stops matching its config, so an
identical-config rerun trains from scratch instead of reusing. Old runs stay on disk,
`list_runs` still lists them, and lineage by explicit id still resolves — only the
identical-config reuse path changes, once per old config, and `--force` exists anyway.

**Boundary:** run `3276b687898d` (in flight at this change) has an id under the OLD rule —
the same command after this change gets a different id. Its config is backed by 9b's
`provenance.txt` (cmdline verbatim + commit sha `2b55eb6` + driver + torch), not by its
id. Do not reverse-derive that run's id with the new rule and read the mismatch as a
broken record.

## Evidence

`run_id` differs when each flag differs (asserted). `ruff check` clean; CPU suite
517 passed / 15 skipped / 6 xfailed.
