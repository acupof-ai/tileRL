"""Does every markdown reference in docs/ resolve to a file that exists?

main carried 35 dead cross-references on 2026-09-05; a docs reorg plus two sessions took
it to 0. Nothing was checking, so they accumulated one at a time -- each written by
someone recalling a filename rather than reading it.

Two of the three that needed a human to resolve were unresolvable by name at all: no such
file had ever existed, and the intended target had to be found by a content signature in
the citing sentence. That is the cost this gate exists to avoid paying again.

Entries reference siblings as `wins/x.md` / `errors/x.md`, relative to docs/experience/
rather than to the citing file, so the resolver tries several bases. A first version of
this scan omitted that one and reported 103 dead links against a real 3 -- the tool was
the defect, which is why the base list is spelled out rather than inferred.

`docs/analysis/` is deliberately NOT in that list. A bare-name reference should resolve
only near where it was written; when a file moves to another directory the citing text has
to be updated, and the point of this gate is to say so. Adding the new directory would
have made the one real dead link on main resolve itself -- weakening the check rather than
fixing the tool.

Existence is asked of `git ls-files`, not of the filesystem, because the filesystem
answers for the developer's machine and CI checks out git. Measured: this gate passed
locally and failed on ubuntu because a file that #73 moved to docs/analysis/ was still
sitting untracked in my docs/experience/ from before the move. Verified the other way too,
in a throwaway clone of origin/main: the git-based version reports the 4 dead links CI
sees, including the one the filesystem version could not.
"""

import os
import re
import subprocess
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"
ROOT = DOCS.parent

#: `[text](path.md)`, then the two bare shapes entries use inside backticks.
_LINK = re.compile(r"\[[^\]]*\]\(([^)#]+\.md)[^)]*\)")
_TICK = re.compile(r"`((?:docs/|\.\./)?(?:experience/)?(?:wins|errors)/[\w./-]+\.md)`")
_BARE = re.compile(r"`(20\d\d-\d\d-\d\d-[\w.-]+\.md)`")

#: AGENTS.md documents the entry skeleton as `errors/YYYY-MM-DD-slug.md`, a literal
#: template. Once root-level .md files came into the scan it read as a dead link, twice
#: (AGENTS.md and the CLAUDE.md symlink). Exempt the placeholder itself, not every
#: undated name: requiring `20\d\d` in _TICK also dropped three real hits on
#: `TEMPLATE-bench.md`, which the gate should keep checking.
_PLACEHOLDER = re.compile(r"\bYYYY-MM-DD\b")


def _tracked() -> set[str]:
    """Repo-relative paths git knows about, which is what CI checks out.

    Existence is asked of git, not of the filesystem. A stale untracked copy of a moved
    file makes the filesystem answer yes where CI answers no -- measured: this gate passed
    locally and failed on ubuntu for exactly that reason, because a file #73 moved to
    docs/analysis/ was still sitting in my docs/experience/ from before the move.
    """
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", "docs", "*.md"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def _scanned(tracked: set[str]) -> list[str]:
    """Markdown files whose links this gate checks.

    `docs/` plus the root-level `.md` files, which is where CHANGELOG.md lives: it
    carries 321 `.md` links and AGENTS.md calls it the central progress record, and it
    was outside the scan until 2026-09-06. It has `merge=union`, so a cherry-pick copies
    another branch's line in verbatim -- a citation of an entry that is not on this
    branch, which is exactly the dead link this gate exists to catch.
    """
    return sorted(
        p for p in tracked
        if p.endswith(".md") and (p.startswith("docs/") or "/" not in p)
    )


def _dead() -> list[str]:
    tracked = _tracked()
    out = []
    for rel in _scanned(tracked):
        md = ROOT / rel
        if not md.exists():  # tracked but not checked out (sparse checkout)
            continue
        text = md.read_text(errors="ignore")
        for m in (*_LINK.finditer(text), *_TICK.finditer(text), *_BARE.finditer(text)):
            raw = m.group(1)
            if _PLACEHOLDER.search(raw):
                continue
            bases = (
                Path(rel).parent,
                Path("docs"),
                Path("docs/experience"),
                Path("docs/experience/wins"),
                Path("docs/experience/errors"),
                Path("."),
            )
            # os.path.normpath collapses `..`, which a plain string join does not: a
            # `../errors/x.md` from wins/ must become docs/experience/errors/x.md before it
            # can be compared against git's paths. Without this the scan reported ~40
            # false positives, all of them `../` shaped.
            cands = {os.path.normpath(str(b / raw)) for b in bases}
            if not (cands & tracked):
                line = text[: m.start()].count("\n") + 1
                out.append(f"{rel}:{line} -> {raw}")
    return out


def test_every_docs_reference_resolves():
    dead = _dead()
    assert not dead, "unresolved markdown references:\n  " + "\n  ".join(dead)


def test_the_resolver_reports_a_reference_that_does_not_exist():
    """The base list tries six directories, so it could match almost anything.

    The probe has to be `git add`ed, since existence is now asked of git: an untracked
    file is invisible to the scan, so writing it alone would make this test pass for the
    wrong reason -- the probe would be skipped rather than reported.
    """
    probe = DOCS / "_link_probe.md"
    rel = str(probe.relative_to(ROOT))
    probe.write_text("[x](2026-01-01-no-such-entry.md)\n")
    subprocess.run(["git", "-C", str(ROOT), "add", "--intent-to-add", rel], check=True)
    try:
        assert rel in _tracked(), "the probe was not staged, so the scan cannot see it"
        dead = _dead()
    finally:
        subprocess.run(["git", "-C", str(ROOT), "rm", "--cached", "-q", "--", rel], check=False)
        probe.unlink()
    assert any("_link_probe.md" in d for d in dead), (
        "a reference to a file that does not exist was not reported: the resolver's base "
        f"list matches too broadly. Got {dead}"
    )


def test_a_root_level_md_is_scanned():
    """CHANGELOG.md was outside the scan, and it is the file most likely to carry one.

    `_scanned` is an `or` -- `docs/` prefix, or no slash at all -- and only the first half
    had a test. It carries 321 `.md` links, and `merge=union` means a cherry-pick copies a
    peer's line in verbatim, citing an entry that is not on this branch: measured on
    2026-09-06, that happened and the gate stayed green.

    The probe goes at the repo root, not under docs/, so deleting the `"/" not in p` half
    turns this red while `test_the_resolver_reports_a_reference_that_does_not_exist` stays
    green -- which is the point of having both.
    """
    probe = ROOT / "_ROOT_LINK_PROBE.md"
    rel = probe.name
    probe.write_text("[x](2026-01-01-no-such-entry.md)\n")
    subprocess.run(["git", "-C", str(ROOT), "add", "--intent-to-add", rel], check=True)
    try:
        assert rel in _tracked(), "the probe was not staged, so the scan cannot see it"
        assert rel in _scanned(_tracked()), "a root-level .md is not in the scanned set"
        dead = _dead()
    finally:
        subprocess.run(["git", "-C", str(ROOT), "rm", "--cached", "-q", "--", rel], check=False)
        probe.unlink()
    assert any(rel in d for d in dead), f"a dead link in a root-level .md was not reported: {dead}"


def test_the_placeholder_exemption_does_not_swallow_a_real_dead_link():
    """`YYYY-MM-DD` is exempt because AGENTS.md documents the entry skeleton with it.

    An exemption is a second guard and needs its own negative: a dated name must still be
    reported from the same file shape. Requiring `20\\d\\d` in `_TICK` instead was tried and
    rejected -- it dropped three real hits on `TEMPLATE-bench.md`.
    """
    probe = DOCS / "_exempt_probe.md"
    rel = str(probe.relative_to(ROOT))
    probe.write_text(
        "skeleton `errors/YYYY-MM-DD-slug.md` is a template\n"
        "`errors/2026-01-01-no-such-entry.md` is not\n"
    )
    subprocess.run(["git", "-C", str(ROOT), "add", "--intent-to-add", rel], check=True)
    try:
        assert rel in _tracked(), "the probe was not staged, so the scan cannot see it"
        dead = _dead()
    finally:
        subprocess.run(["git", "-C", str(ROOT), "rm", "--cached", "-q", "--", rel], check=False)
        probe.unlink()
    hits = [d for d in dead if "_exempt_probe.md" in d]
    assert len(hits) == 1, f"expected exactly the dated link reported, got {hits}"
    assert "2026-01-01" in hits[0], f"the reported link is not the dated one: {hits[0]}"


def test_no_doc_invokes_a_flag_the_cli_does_not_have():
    """A doc that writes `tilerl serve --x` must mean a flag `serve` accepts.

    `docs/serve-v100.md:163` recommended `--kv-tier <dir>` as one of three ways past the
    KV capacity wall, alongside two real ones, and added "the engine path is tested" —
    `tilerl serve --kv-tier /tmp/x` answers `unrecognized arguments`. The code was written
    on another branch and only the prose describing it was ported. That is invisible to
    every other check: the flag name resolves as English, the file it lives in is tracked,
    and nothing executes the line.

    Scoped to flags on the same line AFTER a `tilerl <sub>` invocation. A first version
    compared every `--flag` in docs/ against the CLI and reported 18 misses, of which the
    real count was 1: `--query-compute-apps` is nvidia-smi's, `--system-site-packages` is
    venv's, `--tokens` belongs to a script. Most flags in these docs are scripts' flags,
    so the wider question has a different answer than the one being asked.

    Reads the parser in-process rather than shelling out to `--help` seven times: measured
    0.01s against 0.55s, and verified the two agree on all seven subcommands (12/24/10/8/
    9/5/3 flags, exact match), including under each of the four recipes, since
    `_build_parser` takes one.

    **This gate does not catch the `--kv-tier` line that motivated it.** That line is prose
    naming a flag, with no `tilerl serve` on it, and the negative control confirms it passes
    — deliberately, because a scan wide enough to catch prose reported 18 misses for 1 real
    one. What the gate covers is the copy-pasteable form: a command someone will run. Prose
    recommending a flag stays a reading problem.
    """
    accepted = _cli_flags()

    invoke = re.compile(r"\btilerl\s+(" + "|".join(sorted(accepted)) + r")\b")
    flag = re.compile(r"(--[a-z][a-z0-9-]{2,})")
    bad = []
    for rel in sorted(_tracked()):
        p = ROOT / rel
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "NOT IMPLEMENTED" in line:
                continue
            # Segment per invocation: one line can name two subcommands, and taking only
            # the first attributes the second's flags to it. CHANGELOG.md:235 has
            # `tilerl train --rl` and `tilerl ledger [--lineage ID]` -- both real, but a
            # first-match-only scan reported `train --lineage` as a phantom.
            hits = list(invoke.finditer(line))
            for n, m in enumerate(hits):
                end = hits[n + 1].start() if n + 1 < len(hits) else len(line)
                for f in flag.findall(line[m.end():end]):
                    if f not in accepted[m.group(1)]:
                        bad.append(f"{rel}:{i} -> tilerl {m.group(1)} {f}")
    assert not bad, (
        "docs invoke flags the CLI does not accept (mark a deliberate proposal with "
        "NOT IMPLEMENTED on the same line):\n  " + "\n  ".join(bad))


def _cli_flags() -> dict[str, set[str]]:
    """Every --flag each subcommand accepts, read off the parser in-process.

    Shelling out to --help seven times measured 0.55s against 0.01s for this, and
    the two agree exactly on all seven subcommands.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from tilerl import cli

    parser = cli._build_parser()
    accepted: dict[str, set[str]] = {}
    for action in parser._actions:
        if action.dest == "cmd" and getattr(action, "choices", None):
            for name, sp in action.choices.items():
                accepted[name] = {o for a in sp._actions for o in a.option_strings
                                  if o.startswith("--")}
    assert accepted, "could not read subcommands off the parser"
    return accepted


#: The CLI surface as of 2026-09-09. Any change to it fails test_cli_surface_is_frozen
#: until this set is edited in the same PR, so a surface change is deliberate and visible.
_EXPECTED_CLI_FLAGS: dict[str, frozenset[str]] = {
    'serve': frozenset({
        '--blocks',
        '--depth',
        '--draft',
        '--dram-bytes',
        '--help',
        '--host',
        '--kv-fp8',
        '--max-batch',
        '--max-batched-tokens',
        '--max-ctx',
        '--model',
        '--no-warmup',
        '--port',
        '--slots',
        '--ssd-min-tokens',
        '--ssd-path',
        '--state-bytes',
    }),
    'train': frozenset({
        '--allow-short-rollouts',
        '--data',
        '--depth',
        '--deterministic',
        '--draft',
        '--eval-curve-n',
        '--eval-curve-seed',
        '--eval-every',
        '--eval-gsm8k',
        '--eval-max-new-tokens',
        '--eval-mmlu',
        '--eval-n',
        '--force',
        '--group',
        '--help',
        '--json',
        '--judge',
        '--length-penalty',
        '--load-adapter',
        '--lora-rank',
        '--lr',
        '--max-new-tokens',
        '--max-think-tokens',
        '--micro',
        '--model',
        '--opd',
        '--optim',
        '--patience',
        '--prompts-per-step',
        '--recipe',
        '--reward',
        '--rl',
        '--seed',
        '--steps',
        '--temperature',
        '--tp',
    }),
    'pretrain': frozenset({
        '--ckpt-dir',
        '--ckpt-every',
        '--data',
        '--help',
        '--lr',
        '--model',
        '--seed',
        '--seq-len',
        '--steps',
        '--warmup',
    }),
    'bench': frozenset({
        '--batches',
        '--gen',
        '--gpu',
        '--help',
        '--model',
        '--prompt-len',
        '--questions',
        '--readme',
        '--regress',
        '--source',
        '--suite',
        '--table',
    }),
    'generate': frozenset({
        '--devices',
        '--help',
        '--max-batch',
        '--max-new-tokens',
        '--out',
        '--seed',
        '--source',
        '--temperature',
        '--top-p',
    }),
    'merge': frozenset({
        '--base',
        '--help',
        '--method',
        '--out',
        '--specialists',
    }),
    'ledger': frozenset({
        '--help',
        '--json',
        '--lineage',
        '--time-to-score',
    }),}


def test_cli_surface_is_frozen():
    """A flag must not disappear from the CLI silently.

    2026-09-09: a PR deleting --deterministic landed green, CLEAN, conflict-free --
    the deletion was plain in the diff and the merger read only the green checks.
    "Read the diff before merging" is discipline and fails on a night of 20 PRs;
    this makes the deletion itself fail CI.

    Boundary: a wiring test proves a flag does something; this test proves a flag
    does not vanish silently; neither catches semantic drift behind a flag that
    stays present -- that is the wiring test's job.

    Self-checked: with --no-warmup removed from p_serve this fails naming the lost
    flag; restored, it passes. A guard never seen red is not a guard.
    """
    actual = _cli_flags()
    assert set(actual) == set(_EXPECTED_CLI_FLAGS), (
        "subcommands changed: "
        f"new {sorted(set(actual) - set(_EXPECTED_CLI_FLAGS))}, "
        f"removed {sorted(set(_EXPECTED_CLI_FLAGS) - set(actual))}; "
        "edit _EXPECTED_CLI_FLAGS in the same PR")
    for sub, expected in _EXPECTED_CLI_FLAGS.items():
        lost = sorted(expected - actual[sub])
        gained = sorted(actual[sub] - expected)
        assert not lost and not gained, (
            f"{sub} surface changed: lost {lost}, gained {gained}; "
            "edit _EXPECTED_CLI_FLAGS in the same PR")


def test_no_readme_number_is_absent_from_every_dated_entry():
    """README:89 says every number above it sits in a dated entry. Hold it to that.

    `43.2 tok/s wall over the network` sat on README:22 with no entry anywhere: its only
    source was the body of the commit that added it, and that commit touched README.md
    alone — so it passed the every-runtime-change-gets-an-entry rule by not being a
    runtime change, and the number had no measurement record at all. Measured: of the 23
    distinct decimals in README, exactly that one was absent from all 285 dated entries.

    **This gate falsifies, it does not verify.** A number found in some entry is not
    thereby sourced *by* that entry — README:57's "1.6 points apart" (MMLU significance)
    matches "~1.6% of f32 peak" in the chunkwise-WY entry, two unrelated quantities
    sharing a digit string. What the gate can prove is the other direction: a number
    present in NO dated entry has no source, and that is a fact about the README rather
    than a guess about provenance. Signal is good at this scope — 1 finding, 0 false
    positives across 23 numbers — because the range is small and the claim is absolute.

    Integers are excluded deliberately: `8`, `27`, `4096` are configuration, not
    measurements, and scanning them reported mostly version numbers and tensor shapes.
    """
    import re as _re

    dated = [p for p in _tracked() if _re.search(r"docs/.*/20\d\d-\d\d-\d\d-", p)]
    assert len(dated) > 100, f"only {len(dated)} dated entries found — the scan is broken"

    readme = ROOT / "README.md"
    text = readme.read_text()
    # A decimal, not bounded by word chars or another dot: `1.6` must not match `21.65`
    # or a version like `0.1.8`.
    num = _re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")
    corpus = "\n".join((ROOT / p).read_text(errors="ignore")
                       for p in dated if (ROOT / p).is_file())

    missing = []
    for n in dict.fromkeys(num.findall(text)):
        if not _re.search(rf"(?<![\w.]){_re.escape(n)}(?![\w.])", corpus):
            line = next(i for i, ln in enumerate(text.splitlines(), 1) if n in num.findall(ln))
            missing.append(f"README.md:{line} -> {n}")
    assert not missing, (
        "README numbers that appear in no dated entry under docs/ — either the entry is "
        "missing or the number was never measured:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    print(f"{len(_dead())} dead references across {len(_tracked())} tracked docs")
