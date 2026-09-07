"""How fast does the SSD tier get OFFERED entries? The cap's missing half.

`max_pending=32` bounds in-flight writes and the drain side is measured -- 240 MiB/s
durable, so a full queue takes 42.8 s. The arrival side is not: `ssd_offered` is a
COUNT, so nothing in the tier says offers/second, and a cap set against a drain rate
alone is a bound set without measuring what it bounds
(errors/2026-09-06-ssd-save-ms-is-page-cache-time.md).

The instrument is the counter's slope, not a new field. `ssd_offered` is monotonic and
already on `/health`, so a poller sampling it against `time.monotonic()` yields both
numbers the cap needs and leaves the runtime untouched:

* **mean** offers/second over the run -- what the steady state asks of the disk;
* **max offers in a fixed window** (1 s, 5 s, and the 42.8 s a full queue takes to
  drain) -- what actually overflows a queue, since arrival is bursty by construction
  (one 2729-token prompt publishes 6 times, `kv_cache.py:850`).

Sizing on the mean would be the same error as sizing on the drain rate: a mean is the
centre of a distribution whose TAIL fills the queue. Both are printed, separately,
with what each implies for the cap; this probe does not pick one.

**The burst must be a count in a fixed window, not a rate per poll window.** The first
version reported max(offers in one poll)/interval and it was an artifact: 7.95/s at
`--interval 0.25` and 19.50/s at 0.05, both exactly (1 or 2)/interval, on a workload
whose mean interarrival is 1.07 s -- 21x the poll. A per-poll rate climbs without bound
as the poll shrinks and reads as a discovered burst. `per_window_max_per_s` is still
printed BECAUSE it is wrong, so a reader sweeping `--interval` sees it move while
`max_offers_in_*` does not.

    scripts/pod_run.sh arrival 6 -- /work/tl013/bin/python -u \\
        scripts/probe_ssd_arrival_rate.py --sessions 12 --interval 0.25
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, "scripts")
from bench_chat_interleaved import _fillers  # noqa: E402

PORT = 8127
LOG = "/work/ssd_arrival.log"
SPILL = "/work/ssd_arrival_tier"


def _stats(port: int):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10.0) as r:
        return json.loads(r.read())["stats"]


def _dir_bytes(path: str) -> int:
    """Total bytes of the tier's files. The eviction loop deletes as it goes, so this is
    what SURVIVED, not what was written -- both are needed to tell a small write total
    apart from a large one that was mostly evicted."""
    tot = 0
    try:
        for name in os.listdir(path):
            with contextlib.suppress(OSError):
                tot += os.path.getsize(os.path.join(path, name))
    except OSError:
        return 0
    return tot


def _max_in_window(samples: list[tuple[float, int]], w: float) -> int:
    """Most offers observed inside any `w`-second wall-clock window.

    Invariant under the poll interval, which is the whole point: a per-poll-window rate
    is (offers in that tick)/interval, so it climbs as the interval shrinks even when
    arrivals are unchanged -- measured here, 7.95/s at 0.25 s and 19.50/s at 0.05 s, both
    exactly (1 or 2)/interval. A count in a fixed window cannot do that.
    """
    best, j = 0, 0
    for i in range(len(samples)):
        while samples[i][0] - samples[j][0] > w:
            j += 1
        best = max(best, samples[i][1] - samples[j][1])
    return best


class Sampler(threading.Thread):
    """Poll `ssd_offered` and keep every (t, count) sample. Slope = arrival rate.

    Also samples `/proc/diskstats` sectors-written for the backing device, because
    offers/second is only half the question: an offer is ~320 MiB, so the queue is filled
    by BYTES arriving and drained by bytes written, and the device counter is the only
    reading of the drain side taken under this workload rather than assumed from a
    single-entry microbenchmark.
    """

    def __init__(self, port: int, interval: float, dev: str = "vda2") -> None:
        super().__init__(daemon=True)
        self.port, self.interval, self.dev = port, interval, dev
        self.samples: list[tuple[float, int]] = []
        self.disk: list[tuple[float, int]] = []
        self.errors = 0
        # NOT `_stop`: Thread._stop is a real method and shadowing it makes join()
        # raise `'Event' object is not callable` AFTER the run finishes --
        # a crash in teardown that discards every sample.
        self._halt = threading.Event()

    def _sectors(self) -> int | None:
        """Sectors written to `self.dev`, field 7 of its /proc/diskstats row (512 B each)."""
        try:
            with open("/proc/diskstats", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 9 and parts[2] == self.dev:
                        return int(parts[9])
        except OSError:
            return None
        return None

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                st = _stats(self.port)
                now = time.monotonic()
                self.samples.append((now, int(st["ssd_offered"])))
                sec = self._sectors()
                if sec is not None:
                    self.disk.append((now, sec))
            except (urllib.error.URLError, OSError, KeyError, ValueError):
                # A poll that fails leaves a GAP, and a gap inflates the next interval's
                # rate denominator rather than the numerator -- so it is counted, and the
                # run is disqualified if gaps are a large share.
                self.errors += 1
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()

    def report(self) -> dict:
        s = self.samples
        if len(s) < 2:
            return {"samples": len(s), "reason": "too few samples for a slope"}
        span = s[-1][0] - s[0][0]
        total = s[-1][1] - s[0][1]
        per = [((b[1] - a[1]) / (b[0] - a[0]), b[0] - a[0]) for a, b in zip(s, s[1:])
               if b[0] > a[0]]
        rates = [r for r, _ in per]
        burst, burst_dt = max(per, key=lambda p: p[0]) if per else (0.0, 0.0)
        out = {
            "samples": len(s), "poll_errors": self.errors, "span_s": span,
            "offered_total": total,
            "mean_per_s": total / span if span > 0 else 0.0,
            # Kept only to show it is an artifact: one offer landing in a poll window
            # reads as 1/interval, so this number RISES as the interval shrinks without
            # anything about the workload changing. Never size a cap on it.
            "per_window_max_per_s": burst, "per_window_s": burst_dt,
            "nonzero_windows": sum(1 for r in rates if r > 0),
            "windows": len(rates),
        }
        # The burst measure that survives a change of poll interval: the most offers seen
        # inside a FIXED wall-clock window, which is also the quantity the queue depth is
        # compared against. Poll interval only bounds its resolution, not its value.
        for w in (1.0, 5.0, 32.0 * 1.337):
            out[f"max_offers_in_{w:.3g}s"] = _max_in_window(s, w)
        d = self.disk
        if len(d) >= 2 and d[-1][0] > d[0][0]:
            # The device's own view of the drain, measured here rather than carried over
            # from a one-entry microbenchmark on an idle disk.
            mib = (d[-1][1] - d[0][1]) * 512 / (1 << 20)
            out["disk_written_mib"] = mib
            out["disk_mib_per_s"] = mib / (d[-1][0] - d[0][0])
        return out


def _selfcheck() -> int:
    """The rates are the whole output, so the arithmetic gets a check with a hand-built
    sample list -- no GPU, no server. The load-bearing case is that the fixed-window
    burst is INVARIANT under the poll interval while the per-poll rate is not: that
    difference is what made the first two runs' burst number an artifact."""
    s = Sampler.__new__(Sampler)
    s.samples, s.errors, s.disk = [(0.0, 0), (1.0, 2), (2.0, 2), (2.5, 8), (3.5, 9)], 0, []
    r = s.report()
    assert abs(r["mean_per_s"] - 9 / 3.5) < 1e-9, r["mean_per_s"]
    # The per-poll rate: 6 offers in a 0.5 s tick reads as 12/s. Kept, and shown to be
    # the artifact by the invariance check below.
    assert abs(r["per_window_max_per_s"] - 12.0) < 1e-9, r["per_window_max_per_s"]
    assert abs(r["per_window_s"] - 0.5) < 1e-9, r["per_window_s"]
    assert (r["nonzero_windows"], r["windows"]) == (3, 4), r
    # 8 offers in [0,2.5]; the widest 1 s window holds 6 (2.0 -> 2.5 plus nothing before).
    assert r["max_offers_in_1s"] == 6, r["max_offers_in_1s"]
    assert r["max_offers_in_5s"] == 9, r["max_offers_in_5s"]
    # THE control for the artifact: same arrivals, poll twice as fine. A per-poll rate
    # doubles; a fixed-window count must not move. Sampling every 0.5 s vs every 0.25 s
    # over one arrival at t=1.0.
    coarse, fine = Sampler.__new__(Sampler), Sampler.__new__(Sampler)
    coarse.samples, coarse.errors, coarse.disk = [(0.0, 0), (0.5, 0), (1.0, 1), (1.5, 1)], 0, []
    fine.samples, fine.errors, fine.disk = (
        [(0.0, 0), (0.25, 0), (0.5, 0), (0.75, 0), (1.0, 1), (1.25, 1), (1.5, 1)], 0, [])
    c, f = coarse.report(), fine.report()
    assert abs(c["per_window_max_per_s"] - 2.0) < 1e-9, c["per_window_max_per_s"]
    assert abs(f["per_window_max_per_s"] - 4.0) < 1e-9, f["per_window_max_per_s"]
    assert c["max_offers_in_1s"] == f["max_offers_in_1s"] == 1, (c, f)
    assert "disk_mib_per_s" not in r, "no disk samples must not yield a disk rate"
    # 2048 sectors x 512 B = 1 MiB over 2 s = 0.5 MiB/s.
    s.disk = [(0.0, 1000), (2.0, 1000 + 2048)]
    d = s.report()
    assert abs(d["disk_written_mib"] - 1.0) < 1e-9, d["disk_written_mib"]
    assert abs(d["disk_mib_per_s"] - 0.5) < 1e-9, d["disk_mib_per_s"]
    one = Sampler.__new__(Sampler)
    one.samples, one.errors, one.disk = [(0.0, 0)], 0, []
    assert "reason" in one.report(), "a single sample must not yield a slope"
    # The arithmetic above ran on __new__ objects and so never started or joined a thread
    # -- which is exactly how `self._stop` shadowing `Thread._stop` reached the pod twice
    # and crashed in join() after a full 3-turn run. Drive the real lifecycle: a dead port
    # makes every poll fail, so this checks start/stop/join and the gap counter, not I/O.
    live = Sampler(1, 0.01)
    live.start()
    time.sleep(0.1)
    live.stop()
    live.join(timeout=5)
    assert not live.is_alive(), "sampler did not stop"
    assert live.errors > 0 and not live.samples, (live.errors, len(live.samples))
    assert "reason" in live.report(), "no samples must not yield a slope"
    # `_sectors` parses a file this Mac does not have, so the field index is checked
    # against a real /proc/diskstats row from the pod instead of assumed. Field 5 is
    # sectors READ and is 1.7x larger here, so an off-by-one would read plausible.
    row = ("254       2 vda2 182053544 10983589 26275406084 2485467032 98768884 "
           "90712573 15786712684 629786123 0 112564790 2996439344 0 0 0 0 0 0")
    parts = row.split()
    assert parts[2] == "vda2" and len(parts) > 9
    assert int(parts[9]) == 15786712684, "sectors-written is field 9 of the row"
    assert int(parts[5]) == 26275406084, "field 5 is sectors READ -- not this"
    print(f"selfcheck ok: mean {r['mean_per_s']:.3f}/s, max 6/1s; per-poll rate moved "
          f"{c['per_window_max_per_s']:.0f}->{f['per_window_max_per_s']:.0f} while the "
          f"1s count held at {c['max_offers_in_1s']}; disk {d['disk_mib_per_s']:.1f} MiB/s; "
          f"lifecycle ok ({live.errors} gaps counted)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true",
                    help="check the slope arithmetic and exit; no GPU, no server")
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--grow", type=int, default=40)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--max-ctx", type=int, default=49152,
                    help="block pool ceiling. Must fit the LAST turn's prompt: turn t is "
                         "roughly grow*(t+1) fillers plus every earlier turn, so 3 turns at "
                         "grow 40 reaches ~10k tokens and 8192 returns a 400 mid-run")
    ap.add_argument("--interval", type=float, default=0.25,
                    help="poll period in seconds; a floor on the burst resolution")
    ap.add_argument("--settle-s", type=float, default=120.0,
                    help="after the last request, wait up to this long for "
                         "ssd_pending to reach 0 before stopping the sampler")
    ap.add_argument("--min-tokens", type=int, default=0, help="0 = the tier's own floor")
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--spill", default=SPILL)
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()

    cmd = [sys.executable, "-u", "-m", "tilerl.cli", "serve", "--model", "qwen38-27b",
           "--host", "127.0.0.1", "--port", str(PORT), "--max-batch", "1",
           "--max-ctx", str(a.max_ctx), "--slots", str(a.slots),
           "--ssd-path", a.spill]
    if a.min_tokens:
        cmd += ["--ssd-min-tokens", str(a.min_tokens)]
    # setdefault, not override: a hardcoded /work made the child recompile on any other box.
    env = dict(os.environ)
    env.setdefault("TILELANG_CACHE_DIR", "/work/tilelang_cache")
    with open(a.log, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    sampler = None
    try:
        end = time.monotonic() + 900
        while time.monotonic() < end:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}; see {a.log}")
            try:
                st = _stats(PORT)
                break
            except (urllib.error.URLError, OSError, KeyError):
                time.sleep(1.0)
        else:
            raise TimeoutError("server not up")

        # The tier being OFF reports ssd_offered=0 forever, which reads as "no arrivals"
        # -- the same output a working tier under no load gives. Refuse instead.
        assert "ssd_offered" in st, f"SSD tier off: no ssd_* keys in /health ({sorted(st)})"
        print(f"tier on: {a.spill}  blocks_total={st['blocks_total']}  "
              f"interval={a.interval}s", flush=True)

        sampler = Sampler(PORT, a.interval)
        sampler.start()
        fillers = _fillers(a.sessions)
        convs = [[] for _ in fillers]
        aborted = ""
        for turn in range(a.turns):
            for c, filler in enumerate(fillers):
                convs[c].append({"role": "user", "content": filler * a.grow * (turn + 1)})
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    data=json.dumps({"model": "qwen38-27b", "messages": convs[c],
                                     "max_tokens": 32, "temperature": 0.0}).encode(),
                    headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=1800) as r:
                        out = json.loads(r.read())
                except urllib.error.HTTPError as exc:
                    # A 400 here is `--max-ctx` too small for the last turn, and the body
                    # says so -- urllib's own str() is only "HTTP Error 400: Bad Request".
                    # Do NOT re-raise: the samples already collected are the measurement,
                    # and a raise discards them to report a config mistake.
                    aborted = f"turn {turn} c{c} HTTP {exc.code}: {exc.read().decode()[:300]}"
                    print(aborted, flush=True)
                    break
                convs[c].append({"role": "assistant",
                                 "content": out["choices"][0]["message"]["content"]})
                st = _stats(PORT)
                print(f"turn {turn} c{c} tokens={out['usage']['prompt_tokens']:5d} "
                      f"offered={st['ssd_offered']} refusals={st['ssd_refusals']} "
                      f"saves={st['ssd_saves']} pending={st['ssd_pending']}", flush=True)
            if aborted:
                break
        # Settle before stopping the sampler: the request loop ends with entries still in
        # `_pending`, and the device is still writing them. Without this the disk rate is
        # divided by a span that excludes the tail of its own writes, and the run's
        # arrival total is compared against a partial drain.
        settle_end = time.monotonic() + a.settle_s
        while time.monotonic() < settle_end:
            st = _stats(PORT)
            if st["ssd_pending"] == 0:
                break
            time.sleep(0.5)
        drained = _stats(PORT)
        sampler.stop()
        sampler.join(timeout=10)
        rep = sampler.report()
        rep["settled_pending"] = drained["ssd_pending"]
        rep["settled_saves"] = drained["ssd_saves"]
        if aborted:
            rep["aborted"] = aborted
        final = _stats(PORT)
        rep["final"] = {k: final[k] for k in sorted(final) if k.startswith("ssd_")}
        # An offer writes a .kv AND a .st, so saves ~= 2x offers; the byte accounting has
        # to use the files that exist, not one assumed entry size.
        tot = _dir_bytes(os.path.join(a.spill, "tilerl_kvtier"))
        rep["on_disk_mib"] = tot / (1 << 20)
        # Measured numbers, then the arithmetic each implies -- kept apart because a cap
        # sized on the mean and a cap sized on the burst are different caps.
        drain_s = 42.8 / 32  # 240 MiB/s durable, 320.6 MiB entries: 1.337 s per entry
        rep["drain_per_s"] = 1.0 / drain_s
        if rep.get("mean_per_s"):
            rep["mean_vs_drain"] = rep["mean_per_s"] * drain_s
        # The queue depth a burst implies: offers that arrived within one full-drain time
        # and so are in flight together. This is the number the cap has to cover, and it
        # comes from the interval-invariant count, not the per-poll rate. Asserted, not
        # `.get()`-ed: a renamed window would otherwise report depth None as a value.
        depth_key = f"max_offers_in_{32.0 * 1.337:.3g}s"
        assert depth_key in rep, f"{depth_key} missing; windows are {sorted(rep)}"
        rep["queue_depth_needed"] = rep[depth_key]
        print(json.dumps(rep, indent=2, sort_keys=True), flush=True)
        if rep.get("poll_errors"):
            print(f"WARNING: {rep['poll_errors']} polls failed; each is a gap that "
                  f"UNDERstates the count in that window", flush=True)
        print(f"per_window_max_per_s is an ARTIFACT of the {a.interval}s poll -- it is "
              f"(offers in one tick)/interval and rises as the interval shrinks. Compare "
              f"max_offers_in_*s across two --interval values: those must not move.",
              flush=True)
    finally:
        if sampler is not None:
            sampler.stop()
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
