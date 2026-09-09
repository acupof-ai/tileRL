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
#: A baseline floor must name the null it is measured against.
_BASELINE_NULL = re.compile(r"=")
_SHA = re.compile(r"^[0-9a-f]{7,40}$")

REQUIRED = ("metric", "value", "unit", "target", "build", "model", "shape",
            "warm", "n", "spread", "device", "sha", "cmd", "floor")


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text())


def git_sha() -> str:
    """Code sha under test. The pod is a tarball, not a clone: pod_sync stamps
    ``.synced_commit`` at the repo root, and a record without either is rejected."""
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        stamp = _ROOT / ".synced_commit"
        return stamp.read_text().strip() if stamp.exists() else "unknown"


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

    if not _SHA.match(str(record["sha"])):
        errs.append(f"sha {record['sha']!r} not a git sha")

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
    return [json.loads(line) for line in _lines() if line.strip()]


def current(records: list[dict]) -> dict:
    """Newest non-superseded row per population key."""
    superseded = {r["supersedes"] for r in records if r.get("supersedes")}
    out: dict = {}
    for r in records:  # append order = time order
        if r["id"] in superseded:
            continue
        out[key(r)] = r
    return out


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
        "sha": "90f308a", "cmd": "python3 scripts/bench_b1_decode.py --build fused+graph",
        "floor": {"value": 129.0, "unit": "tok/s", "kind": "roofline",
                  "derivation": "129 tok/s = 30.9 GB weights / 4 TB/s H20 HBM (wins/2026-08-24-sota-all-levers.md)"},
    }
    assert validate(good, reg, set()) == [], validate(good, reg, set())

    bad = dict(good)
    del bad["sha"]
    assert validate(bad, reg, set()), "missing sha must reject"

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
        del bad["sha"]
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
    finally:
        STORE = old_store

    print("benchrec: schema selftest OK (good accepts, 3 bad worlds reject, n=1 fenced out of regression, "
          "append path rejects and a supersedes rerun replaces)")
