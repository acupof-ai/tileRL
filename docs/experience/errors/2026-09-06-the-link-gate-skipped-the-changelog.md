# The link gate skipped the file with the most links — 2026-09-06

## Context

`test_docs_links.py` resolves every markdown reference against `git ls-files`. It
scans `p.startswith("docs/")`. `CHANGELOG.md` is at the repo root, so its **321
`.md` links were never checked** — the file AGENTS.md calls the central progress
record, where every phase exit, default flip and verdict lands with a link to its
entry.

## Root Cause

**The gap already let one through, and I caught it by hand rather than by gate.**
`CHANGELOG.md` carries `merge=union`, so cherry-picking my own commit onto
`origin/main` copied a *different* PR's CHANGELOG line in verbatim — a citation of
`errors/2026-09-06-a-spec-rate-over-a-dense-roofline.md`, which was on #157's
branch and not on #158's. The gate ran and passed. `git ls-tree HEAD` showed the
file absent while the link pointed at it.

Measured, two arms, the same injected dead link in each:

| arm | file | gate |
|---|---|---|
| control | `docs/experience/OPEN.md` | **rc=1**, and the failure names the injected link |
| test | `CHANGELOG.md` | **rc=0**, silent |

Baseline green first, so neither arm is attributable to a pre-existing failure.

## Two mistakes while fixing it

**Widening the scan turned the gate red on a template, not a defect.** AGENTS.md
documents the entry skeleton as `` `errors/YYYY-MM-DD-slug.md` ``, a literal
placeholder. Once root-level files were scanned it read as two dead links —
AGENTS.md and the `CLAUDE.md` symlink that points at it.

**My first repair silently weakened the gate.** I required `20\d\d` in `_TICK`,
matching the year guard `_BARE` already had. Measured before committing: old
pattern **130** tick-shaped hits across 349 files, new pattern **125** — and three
of the five lost were real, all on `TEMPLATE-bench.md`, a file the gate should
keep checking. A tightening that fixes a false positive by dropping true ones is
not a fix. The exemption is now the placeholder string itself, which loses nothing.

## Fix

`_scanned()` returns `docs/**` plus root-level `.md`; `_PLACEHOLDER` skips
`YYYY-MM-DD`. Both halves get a test, and **both tests were made to fail on
purpose** — a green test that has never gone red measures nothing:

| mutation | test that went red | assertion that fired |
|---|---|---|
| drop the `"/" not in p` half of the `or` | `test_a_root_level_md_is_scanned` | "a root-level .md is not in the scanned set" |
| drop the placeholder exemption | `test_every_docs_reference_resolves` | "unresolved markdown references" |

In each arm the other tests stayed green, `__pycache__` was cleared between arms
(a restored file is not a restored import), and the restored tree is green at 6
passed. The root probe lives at the repo root rather than under `docs/`, so it is
the `or`'s second operand it exercises and not the first — the pre-existing
`test_the_resolver_reports_a_reference_that_does_not_exist` covers the first and
stays green under that mutation, which is why both exist.

## Rule

**A scan's filter is a claim about coverage, and the file most likely to break is
the one the filter's shape excludes.** `startswith("docs/")` reads as "the docs",
and the densest source of doc links sat outside it for as long as the gate has
existed.

Second: **when a widened check goes red, separate the false positive from the
defect before narrowing anything.** My narrowing was in the same shape as the
guard it copied and still cost three real hits — so count what a pattern change
loses, on the real tree, before committing it.

## Results

Dev-only tooling: a test file. No runtime change, so no bench entry.

| date | commit | measurement | value |
|---|---|---|---|
| 2026-09-06 | (this) | `.md` links in CHANGELOG.md, previously unscanned | **321** |
| 2026-09-06 | (this) | dead link injected into CHANGELOG.md, before the fix | **not reported** (control in docs/: reported) |
| 2026-09-06 | (this) | `_TICK` hits lost by the rejected `20\d\d` narrowing | **5, of which 3 real** |
| 2026-09-06 | (this) | mutation controls, one per guard | **2/2 red on the right assertion** |
