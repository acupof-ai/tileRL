"""One measurement is one record: validate, append, load.

The bench system's spine: collectors append records to
``docs/experience/bench/measurements.jsonl`` (append-only, one JSON object per
line); every view (``bench_harness --table/--readme/--regress/--questions``)
reads that file and nothing else. A record missing a required field is
REJECTED here, at write time -- the schema exists because a day was burned on
populations nobody wrote down (build, cap, compiles, sha).

Schema: ``docs/bench-schema.md``. Registry (unit/direction/weight/required
shape keys): ``docs/bench-metrics.json``.

A rerun of the same population appends a new row with ``supersedes`` (the old
row's id) and ``note`` (why); both rows stay, views take the newest
non-superseded. Reruns are visible -- that is the point.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
STORE = _ROOT / "docs/experience/bench/measurements.jsonl"
REGISTRY = _ROOT / "docs/bench-metrics.json"

TARGETS = ("cpu", "metal", "sm90", "sm70")
BUILDS = ("eager", "fused", "fused+graph", "fused+graph+draft")
FLOOR_KINDS = ("bandwidth", "compute", "roofline", "measured-best", "baseline")
#: Floor kinds that state a physical limit. --questions ranks headroom against
#: these; measured-best is a regression quantity (vs our own best) and lives in
#: --regress. The two must not share a sorted column: 4.99x of headroom and a
#: 1.02x regression are not the same kind of number.
PHYSICAL_FLOOR_KINDS = ("bandwidth", "compute", "roofline", "baseline")
#: Hard physical limits. A value that beats one is a measurement error, not a
#: result — 135.5 tok/s against a 129 roofline sailed through every "is it good
#: enough" gate for three days. Baseline is excluded: the null is meant to be
#: beaten.
HARD_FLOOR_KINDS = ("bandwidth", "compute", "roofline")
#: A baseline floor must name the null it is measured against.
_BASELINE_NULL = re.compile(r"=")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

REQUIRED = ("metric", "value", "unit", "target", "build", "model", "shape",
            "warm", "n", "spread", "device", "commit", "dirty", "cmd", "floor")


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text())


def git_commit() -> str:
    """Full 40-char sha of the tree under test, self-collected. A record's
    commit is never hand-filled: a written-down sha is the nib of main at
    entry-writing time, not the tree that produced the number
    (wins/2026-09-03-batched-selector-walk.md:80 — 40bc83c cannot produce its
    own row; B=1 landed in #58). The pod is a tarball, not a clone: pod_sync
    stamps ``.synced_commit`` at the repo root, and a record without either is
    rejected."""
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        stamp = _ROOT / ".synced_commit"
        return stamp.read_text().strip() if stamp.exists() else "unknown"


def git_dirty() -> bool:
    """True when the tree has uncommitted changes: a sha cannot fully identify
    a dirty tree (9b's 267-line analysis tool lived in untracked files). The
    pod tarball carries ``.synced_dirty``, stamped by pod_sync. No fallback:
    unknown must not render as clean — git_commit() rejects the same state, so
    this branch is unreachable through append(); a default here is the one
    place that would stay silent if that check were ever loosened."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "-C", str(_ROOT), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        marker = _ROOT / ".synced_dirty"
        if marker.exists():
            return marker.read_text().strip() == "1"
        raise RuntimeError("git_dirty: no git repo and no .synced_dirty marker")


def _commit_exists(commit: str) -> bool | None:
    """Existence check catches typos, not wrong-tree shas (a real sha from
    another tree passes). The real guard is git_commit() self-collection.
    None when not in a git repo (pod tarball): nothing to check against."""
    import subprocess

    try:
        subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "--git-dir"],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    try:
        subprocess.check_call(
            ["git", "-C", str(_ROOT), "cat-file", "-e", commit + "^{commit}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def previous_best(record: dict, lower_is_better: bool) -> float | None:
    """Best value among non-superseded rows with this record's population key,
    excluding the record itself; None on first sight. The other half of the
    symmetric gate: a new best that beats the old one by a wide margin is
    implausible until explained — a 135.5 with a prior 78.4."""
    rid = record.get("id")
    k = key(record)
    superseded = {r["supersedes"] for r in load_all() if r.get("supersedes")}
    vals = [r["value"] for r in load_all()
            if key(r) == k and r.get("id") != rid and r["id"] not in superseded]
    if not vals:
        return None
    return min(vals) if lower_is_better else max(vals)


def _canon(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def new_id(record: dict) -> str:
    """Content-addressed: the same record resubmitted is the same id."""
    return hashlib.sha1(_canon(record).encode()).hexdigest()[:12]


def key(record: dict) -> tuple:
    """The population: two rows with the same key measure the same thing."""
    return (record["metric"], record["target"], record["build"], record["model"],
            tuple(sorted(record["shape"].items())),
            (record["device"]["name"], record["device"].get("card")))


def is_regressable(record: dict) -> bool:
    """A point estimate with no dispersion cannot enter the regression view."""
    return record["n"] >= 2


def validate(record: dict, registry: dict, existing_ids: set) -> list[str]:
    """Every reason the record is rejected; empty list means accept."""
    errs = []
    for f in REQUIRED:
        if f not in record:
            errs.append(f"missing field {f!r}")
    if errs:
        return errs  # nothing below is safe to check

    metric = record["metric"]
    if metric not in registry["metrics"]:
        errs.append(f"metric {metric!r} not in registry")
    else:
        reg = registry["metrics"][metric]
        if record["unit"] != reg["unit"]:
            errs.append(f"unit {record['unit']!r} != registry {reg['unit']!r}")
        for k in reg["shape"]:
            if k not in record["shape"]:
                errs.append(f"shape missing required key {k!r} for metric {metric!r}")

    if record["target"] not in TARGETS:
        errs.append(f"target {record['target']!r} not in {TARGETS}")
    if record["build"] not in BUILDS:
        errs.append(f"build {record['build']!r} not in {BUILDS}")

    if not isinstance(record["value"], (int, float)) or isinstance(record["value"], bool):
        errs.append("value must be a number")
    if not record["model"] or not isinstance(record["model"], str):
        errs.append("model must be a non-empty string")
    if not isinstance(record["shape"], dict) or not record["shape"]:
        errs.append("shape must be a non-empty object")
    if not record["cmd"] or not isinstance(record["cmd"], str):
        errs.append("cmd must be a non-empty string")

    warm = record["warm"]
    if not isinstance(warm, dict) or "compiles" not in warm:
        errs.append("warm.compiles missing (0 is an assertion; absent is unmeasured)")
    elif not isinstance(warm["compiles"], int) or warm["compiles"] < 0:
        errs.append("warm.compiles must be an int >= 0")
    if not isinstance(warm, dict) or warm.get("state") not in ("cold", "warm"):
        errs.append("warm.state must be 'cold' or 'warm'")

    if not isinstance(record["n"], int) or record["n"] < 1:
        errs.append("n must be an int >= 1")
    if not isinstance(record["spread"], (int, float)) or record["spread"] < 0:
        errs.append("spread must be a number >= 0")
    if record["n"] == 1 and record["spread"] != 0:
        errs.append("n=1 rows carry no dispersion: spread must be 0")

    dev = record["device"]
    if not isinstance(dev, dict) or not dev.get("name"):
        errs.append("device.name required")
    if record["target"] in ("sm90", "sm70") and dev.get("card") is None:
        errs.append(f"device.card required on target {record['target']}")

    commit = record.get("commit", "")
    if not _COMMIT_SHA.match(str(commit)):
        errs.append(f"commit {commit!r} must be the full 40-hex sha, self-collected "
                    "(git rev-parse HEAD in the tree that produced the number)")
    elif _commit_exists(commit) is False:
        errs.append(f"commit {commit[:12]} not in this repo — a typo or a sha from another tree")
    if not isinstance(record.get("dirty"), bool):
        errs.append("dirty must be a bool (git status --porcelain non-empty)")

    floor = record["floor"]
    if not isinstance(floor, dict):
        errs.append("floor must be an object")
    else:
        if not isinstance(floor.get("value"), (int, float)) or floor["value"] <= 0:
            errs.append("floor.value must be a number > 0")
        if floor.get("unit") != record.get("unit"):
            errs.append("floor.unit must equal the record's unit "
                        "(a floor in other units is a forged floor)")
        if floor.get("kind") not in FLOOR_KINDS:
            errs.append(f"floor.kind not in {FLOOR_KINDS}")
        deriv = floor.get("derivation")
        if not deriv or not isinstance(deriv, str):
            errs.append("floor.derivation required")
        elif "no known floor" in deriv.lower():
            errs.append("floor.derivation refuses to name the floor ('no known floor')")
        elif floor.get("kind") == "baseline" and not _BASELINE_NULL.search(deriv):
            errs.append("baseline floor must name the null it is measured against (e.g. '0.25 = 1/4 choices')")

    if record.get("supersedes") is not None:
        if not record.get("note"):
            errs.append("supersedes requires note (why the rerun)")
        if record["supersedes"] not in existing_ids:
            errs.append(f"supersedes id {record['supersedes']!r} not in store")

    if new_id(record) in existing_ids:
        errs.append("duplicate: an identical record is already in the store "
                    "(a rerun with a new value gets a new id + supersedes)")
    return errs


def append(record: dict) -> str:
    """Validate and append one line; raises ValueError with every reason."""
    registry = load_registry()
    existing = {json.loads(line)["id"] for line in _lines() if line.strip()}
    errs = validate(record, registry, existing)
    if errs:
        raise ValueError("record rejected:\n  - " + "\n  - ".join(errs))
    row = dict(record)
    row["id"] = new_id(record)
    row["date"] = time.strftime("%Y-%m-%d")
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with STORE.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row["id"]


def _lines() -> list[str]:
    return STORE.read_text().splitlines() if STORE.exists() else []


def load_all() -> list[dict]:
    """Parse the store. A torn final line (a writer mid-append) is skipped with a
    count — 1 skipped line is a torn tail the next read completes, many mean the
    file is corrupted — so a concurrent reader never takes a view down."""
    rows, skipped = [], 0
    for line in _lines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        print(f"benchrec: WARNING skipped {skipped} unparseable line(s) "
              f"(1 = a torn append, re-read; many = the store is corrupted)", file=sys.stderr)
    return rows


def current(records: list[dict]) -> dict:
    """Newest non-superseded row per population key."""
    superseded = {r["supersedes"] for r in records if r.get("supersedes")}
    out: dict = {}
    for r in records:  # append order = time order
        if r["id"] in superseded:
            continue
        out[key(r)] = r
    return out


def add_record_args(
    ap, *, default_target: str = "sm90", default_device: str | None = None,
    client_side: bool = False,
):
    """The population flags every collector carries. --build stays argparse-optional
    here but record_common raises without it: a server client cannot see the server's
    build (eager vs fused+graph is 6.4x on decode), while an engine-direct script
    derives the build from its own flags and passes it to ``record_common``.

    client_side: the script measures a remote server over HTTP and cannot see its
    attributes. Rule: a client-side collector must not default any field describing
    the server -- the default would be a value the client cannot know, and it would
    look correctly set. --device-name then has no default and record_common raises
    without it. Defaults are only for things the collector itself knows.

    default_device is None, not a card name: a collector that knows its device passes
    the name explicitly; everything else falls through to torch.cuda.get_device_name
    in record_common. A hardcoded "H20" default made that fallback dead code and
    mislabeled every non-H20 run (2026-09-09, eight rows)."""
    ap.add_argument("--build", choices=list(BUILDS),
                    help="the build under test (required when the script cannot see it)")
    ap.add_argument("--target", default=default_target, choices=list(TARGETS))
    ap.add_argument("--device-name", default=None if client_side else default_device,
                    help="GPU model of the server under test; required for client-side "
                         "collectors, which cannot see it. Engine scripts default to "
                         "torch.cuda.get_device_name")
    ap.add_argument("--card", type=int, help="physical GPU card; required on sm90/sm70")
    ap.add_argument("--model-name", default="27B-nvfp4")
    if client_side:
        ap.set_defaults(_benchrec_client_side=True)


def record_common(args, *, build: str | None = None) -> dict:
    """The fields every record shares, from add_record_args + the build."""
    build = build or args.build
    if not build:
        raise SystemExit("--build required: a client cannot see the server's build, "
                         "and eager vs fused+graph is 6.4x on decode")
    if build not in BUILDS:
        raise SystemExit(f"build {build!r} not in {BUILDS}")
    if args.target in ("sm90", "sm70") and args.card is None:
        raise SystemExit(f"--card required on target {args.target}")
    device_name = args.device_name
    if not device_name and getattr(args, "_benchrec_client_side", False):
        raise SystemExit("--device-name required: a client cannot see the server's "
                         "device, and a default here is a population lie -- a cpu run "
                         "labeled H20 enters every device-grouped view and every "
                         "measured-best comparison")
    if not device_name:
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass
    if not device_name:
        device_name = args.target
    return {"target": args.target, "build": build, "model": args.model_name,
            "device": {"name": device_name, "card": args.card},
            "commit": git_commit(), "dirty": git_dirty(), "cmd": " ".join(sys.argv)}


def measured_best(record: dict, lower_is_better: bool) -> tuple[float | None, str | None]:
    """Best accepted value for this record's population, across the whole
    non-superseded history; (None, None) on first sight. Superseded rows were
    bad runs (a cold run that warmed the engine) — their values never anchor a floor."""
    records = load_all()
    superseded = {r["supersedes"] for r in records if r.get("supersedes")}
    k = key(record)
    rows = [r for r in records if key(r) == k and r["id"] not in superseded]
    if not rows:
        return None, None
    best = (min(rows, key=lambda r: r["value"]) if lower_is_better
            else max(rows, key=lambda r: r["value"]))
    return best["value"], best["id"]


def measured_best_floor(record: dict, lower_is_better: bool) -> dict:
    """A measured-best floor for a record about to be appended: the population's
    best accepted value, or this measurement on first sight."""
    best, best_id = measured_best(record, lower_is_better)
    if best is None:
        v, deriv = record["value"], "first accepted row for this population; floor = this measurement"
    else:
        v = min(best, record["value"]) if lower_is_better else max(best, record["value"])
        deriv = f"best accepted value for this population (row {best_id})"
    return {"value": v, "unit": record["unit"], "kind": "measured-best", "derivation": deriv}


if __name__ == "__main__":
    # Selftest: the system must prove it goes red. A good world accepts; four
    # bad worlds each reject (or, for n=1, accept but stay out of regression).
    reg = load_registry()
    good = {
        "metric": "decode_tok_s", "value": 94.6, "unit": "tok/s",
        "target": "sm90", "build": "fused+graph", "model": "27B-nvfp4",
        "shape": {"batch": 1, "ctx": 1024},
        "warm": {"state": "warm", "compiles": 0},
        "n": 30, "spread": 0.017,
        "device": {"name": "H20", "card": 6},
        "commit": git_commit(), "dirty": git_dirty(),
        "cmd": "python3 scripts/bench_b1_decode.py --build fused+graph",
        "floor": {"value": 129.0, "unit": "tok/s", "kind": "roofline",
                  "derivation": "129 tok/s = 30.9 GB weights / 4 TB/s H20 HBM (wins/2026-08-24-sota-all-levers.md)"},
    }
    assert validate(good, reg, set()) == [], validate(good, reg, set())

    bad = dict(good)
    del bad["commit"]
    assert validate(bad, reg, set()), "missing commit must reject"

    bad = json.loads(json.dumps(good))
    bad["commit"] = "0" * 40
    assert validate(bad, reg, set()), "commit not in this repo must reject (typo or wrong tree)"

    bad = json.loads(json.dumps(good))
    del bad["dirty"]
    assert validate(bad, reg, set()), "missing dirty must reject"

    bad = json.loads(json.dumps(good))
    del bad["warm"]["compiles"]
    assert validate(bad, reg, set()), "warm without compiles must reject"

    bad = json.loads(json.dumps(good))
    bad["floor"] = {"value": 1.0, "unit": "%", "kind": "baseline", "derivation": "no known floor"}
    assert validate(bad, reg, set()), "forged floor must reject"
    # A baseline floor is legal on the metric whose null it names.
    mmlu = json.loads(json.dumps(good))
    mmlu.update({"metric": "mmlu_pct", "value": 74.6, "unit": "%", "shape": {"n": 200}})
    mmlu["floor"] = {"value": 25.0, "unit": "%", "kind": "baseline",
                     "derivation": "0.25 = 1/4 choices"}
    assert validate(mmlu, reg, set()) == [], validate(mmlu, reg, set())

    one = json.loads(json.dumps(good))
    one["n"], one["spread"] = 1, 0.0
    assert validate(one, reg, set()) == [], validate(one, reg, set())
    assert not is_regressable(one), "n=1 must stay out of the regression view"
    assert is_regressable(good)

    # Full append path, against a throwaway store: the system must reject, not
    # just the validator. A bad row must not reach the file; a rerun with
    # supersedes must replace the old row in current() while keeping both lines.
    import tempfile

    old_store = STORE
    STORE = Path(tempfile.mkdtemp()) / "measurements.jsonl"
    try:
        rid = append(good)
        assert rid in STORE.read_text()
        bad = dict(good)
        del bad["commit"]
        try:
            append(bad)
            raise AssertionError("append must reject a record missing sha")
        except ValueError:
            pass
        assert len(STORE.read_text().splitlines()) == 1, "rejected row must not reach the file"

        rerun = dict(good)
        rerun["value"] = 95.0
        rerun["supersedes"] = rid
        rerun["note"] = "rerun: cold run warmed the engine"
        append(rerun)
        assert len(STORE.read_text().splitlines()) == 2, "superseded rows stay in the file"
        cur = current(load_all())
        assert cur[key(good)]["value"] == 95.0, "current() must take the rerun"

        # measured_best_floor: first sight floors at the measurement; a later
        # better row moves the floor; a superseded row never anchors it.
        fresh = dict(good)
        fresh["shape"] = {"batch": 1, "ctx": 2048}
        assert measured_best_floor(fresh, lower_is_better=False)["value"] == fresh["value"]
        better = dict(good)
        better["value"] = 120.0
        append(better)
        assert measured_best_floor(dict(good), lower_is_better=False)["value"] == 120.0
        # previous_best: None on first sight, the prior best once rows exist
        assert previous_best(fresh, lower_is_better=False) is None
        assert previous_best(current(load_all())[key(good)], lower_is_better=False) == 95.0
    finally:
        STORE = old_store

    print("benchrec: schema selftest OK (good accepts, bad worlds reject (missing commit, unknown commit, missing dirty, previous_best, n=1 fenced out of regression, "
          "append path rejects and a supersedes rerun replaces)")
