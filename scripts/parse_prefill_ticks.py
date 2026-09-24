"""Prefill tick wall per prompt, parsed from an engine stderr log.

The cutover window compares two configs that differ in `sparse_min_tokens`
(8192 baseline vs 0 armed). That flag also changes PREFILL routing, and prefill
is the majority of end-to-end time, so the decode-side gates say nothing about
it. This parser reads the number out of what the engine already prints.

Source of the number, stated exactly: `_StepTiming.tick_end` emits one line per
tick (with `TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0`):

    [step-timing] tick 12 total=1815ms dec=0 pre=1 plan=... model=1518ms ...

`pre` is that tick's prefill row count and `dec` its decode row count. A prompt
is served as a run of `dec=0 pre>=1` ticks followed by `dec>=1` ticks; the
boundary is exact in practice (measured on a 37.6k floor run: 74 prefill ticks
per prompt and zero ticks with both dec>=1 and pre>=1).

Measured quantity, and its limits:
  * `prefill_tick_wall_ms` is the SUM of step walls over that prompt's prefill
    ticks. It is not submit->first-token: it excludes tokenization, HTTP, queue
    wait before the first tick, and the first decode tick itself. Those are not
    in this log. Name it "prefill tick wall" in any report; do not call it TTFT.
  * `ttft_upper_ms` = prefill_tick_wall + the first decode tick's total. An
    upper bound, since it still omits queue wait and tokenize.
  * Tick totals are wall clock, so a slow tick's IO (ssd_mmap, cold promotion)
    lands inside the number. That is intended: it is the wall the request pays.

Prompt segmentation: prompts are emitted in submission order and one prompt's
ticks are contiguous, so runs are delimited by prefill runs. Run index i is
prompt i. If the log does not start at tick 1 (an appended run), only the runs
present are returned, indexed from 0 — pass the log of a single run.

    python3 scripts/parse_prefill_ticks.py LOG [LOG ...] [--json OUT]

Exit codes: 0 parsed at least one prompt; 14 nothing parseable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Every field is optional in the pattern except tick/total/dec/pre: a log from an
# older revision may lack the tail, and a missing tail must not drop the tick.
_LINE = re.compile(r"\[step-timing\] tick (\d+) total=(\d+)ms dec=(\d+) pre=(\d+)")


def parse(path):
    """Return the list of prompt rows for one log. Never raises on a malformed
    line: a line that does not match the header is skipped, and the count of
    skipped step-timing lines is reported so silence is not read as absence."""
    rows, skipped = [], 0
    with open(path, errors="replace") as f:
        for line in f:
            if "step-timing" not in line:
                continue
            if line.startswith("[step-timing]") and " avg:" in line:
                continue  # the run-end averages line, not a tick
            m = _LINE.search(line)
            if not m:
                skipped += 1
                continue
            rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                         int(m.group(4))))

    out, cur_pre, cur_n, started = [], 0, 0, False
    leading_decode = 0
    mixed = 0
    for tick, total, dec, pre in rows:
        if dec >= 1 and pre >= 1:
            # A tick carrying BOTH decode and prefill rows. The segmentation below
            # cannot place it: it is neither "a prefill tick" nor "a decode tick
            # closing a run", so its wall lands in neither bucket and the prompt
            # split is ambiguous. Measured 0 on the 37.6k floor run, but a
            # different prompt mix (a short prompt admitted while a long one
            # decodes) can produce them, and silently under-counting prefill is
            # exactly the failure this parser exists to avoid. Counted, and the
            # caller turns a nonzero count into rc14 rather than reporting numbers
            # that quietly omit these ticks.
            mixed += 1
            continue
        if dec == 0 and pre >= 1:
            cur_pre += total
            cur_n += 1
            started = True
        elif dec >= 1 and started:
            out.append({"prefill_ticks": cur_n,
                        "prefill_tick_wall_ms": cur_pre,
                        "first_decode_tick_ms": total,
                        "ttft_upper_ms": cur_pre + total})
            cur_pre = cur_n = 0
            started = False
        elif dec >= 1 and not out:
            # Decode ticks before the first prefill run seen: the log starts
            # mid-stream. That prompt's prefill is not in the file, its run is
            # dropped, and every later index is one low against a prompt
            # manifest. Counted so the shift is visible.
            #
            # There is deliberately NO tail field: decode ticks after the last
            # run are present in every complete run (the last prompt's own
            # decode), so "has trailing decode" cannot distinguish truncation
            # from normal end. The count guard below is what catches a short run.
            leading_decode += 1
    return {"file": path, "ticks_parsed": len(rows),
            "step_timing_lines_skipped": skipped,
            "leading_decode_ticks_dropped": leading_decode,
            "mixed_ticks": mixed,
            "prompts": out}


def summarise(runs):
    """p50/min/max of each per-prompt quantity over one arm's prompts."""
    import statistics

    prompts = [p for r in runs for p in r["prompts"]]
    if not prompts:
        return None
    out = {"n_prompts": len(prompts)}
    for key in ("prefill_ticks", "prefill_tick_wall_ms", "first_decode_tick_ms",
                "ttft_upper_ms"):
        vals = [p[key] for p in prompts]
        out[key] = {"p50": statistics.median(vals), "min": min(vals), "max": max(vals),
                    "values": vals}
    return out


def compare(a_runs, b_runs, a_name, b_name, tol=0.10):
    """Per-prompt prefill comparison, aligned by prompt index. Prints both
    numbers for every prompt: a direction word without the pair is not a
    reading. Returns (rows, verdict) where verdict is a list of the prompts
    whose delta exceeds tol."""
    a = [p for r in a_runs for p in r["prompts"]]
    b = [p for r in b_runs for p in r["prompts"]]
    n = min(len(a), len(b))
    rows, bad = [], []
    for i in range(n):
        x, y = a[i]["prefill_tick_wall_ms"], b[i]["prefill_tick_wall_ms"]
        d = (y - x) / x if x else None
        rows.append({"prompt": i, f"{a_name}_prefill_ms": x,
                     f"{b_name}_prefill_ms": y, "delta_frac": d})
        if d is not None and d > tol:
            bad.append(i)
    if len(a) != len(b):
        print(f"WARNING: {a_name} has {len(a)} prompts, {b_name} has {len(b)}; "
              f"compared the first {n}", file=sys.stderr)
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="engine stderr logs, one per arm")
    ap.add_argument("--json", default="", help="write the parse to this path")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="fraction above which a prompt is listed as regressed")
    ap.add_argument("--expect-prompts", type=int, default=0,
                    help="prompt count the run was supposed to serve; asserts the "
                         "parse found that many runs. Guards the real failure this "
                         "parser has: a run cut short (or a log starting "
                         "mid-stream) yields a different count and every index "
                         "after it is silently misaligned — measured on a real "
                         "earlier window, 7 runs parsed against a 6-prompt json.")
    args = ap.parse_args()

    runs = [parse(p) for p in args.logs]
    rc = 0
    for r in runs:
        if r["mixed_ticks"]:
            print(f"INSUFFICIENT: {r['file']} has {r['mixed_ticks']} tick(s) with "
                  f"dec>=1 AND pre>=1; the prefill/decode split is ambiguous there "
                  f"and the per-prompt walls below omit them. Not a reading.",
                  file=sys.stderr)
            rc = 14
        if r["leading_decode_ticks_dropped"]:
            print(f"WARNING: {r['file']} starts mid-stream "
                  f"({r['leading_decode_ticks_dropped']} leading decode ticks); "
                  f"the first prompt's prefill is NOT in this log and its run was "
                  f"dropped, so prompt indices here are shifted. Use a whole-run log.",
                  file=sys.stderr)
            rc = 14
        if args.expect_prompts:
            n_seen = len(r["prompts"])
            if n_seen != args.expect_prompts:
                print(f"WARNING: {r['file']} parsed {n_seen} prompt runs, expected "
                      f"{args.expect_prompts} — the run was cut short, the log is "
                      f"partial, or the file holds a different prompt set. Indices "
                      f"may not line up with a prompt manifest.", file=sys.stderr)
                rc = 14
        s = summarise([r])
        if s is None:
            print(f"{r['file']}: NO PROMPTS PARSED "
                  f"({r['ticks_parsed']} ticks, {r['step_timing_lines_skipped']} "
                  f"unparsed step-timing lines)")
            continue
        print(f"{r['file']}: n={s['n_prompts']} "
              f"prefill_tick_wall p50={s['prefill_tick_wall_ms']['p50']:.0f}ms "
              f"min={s['prefill_tick_wall_ms']['min']} "
              f"max={s['prefill_tick_wall_ms']['max']} "
              f"ttft_upper p50={s['ttft_upper_ms']['p50']:.0f}ms "
              f"prefill_ticks p50={s['prefill_ticks']['p50']:.0f}")

    if len(runs) == 2 and all(summarise([r]) for r in runs):
        a_name = args.logs[0].rsplit("/", 1)[-1]
        b_name = args.logs[1].rsplit("/", 1)[-1]
        rows, bad = compare([runs[0]], [runs[1]], a_name, b_name, args.tol)
        for row in rows:
            d = row["delta_frac"]
            print(f"  prompt {row['prompt']}: {a_name}={row[f'{a_name}_prefill_ms']}ms "
                  f"{b_name}={row[f'{b_name}_prefill_ms']}ms "
                  f"delta={'n/a' if d is None else f'{d * 100:+.1f}%'}")
        if bad:
            print(f"\nREGRESSED (> {args.tol * 100:.0f}%): prompts {bad}")
        else:
            print(f"\nno prompt above +{args.tol * 100:.0f}%")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"runs": runs, "summaries": [summarise([r]) for r in runs]}, f,
                      indent=2)
        print(f"wrote {args.json}")

    if not any(summarise([r]) for r in runs):
        return 14
    return rc


if __name__ == "__main__":
    sys.exit(main())
