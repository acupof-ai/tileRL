"""One-command H20 verification of the native-fp4 w4a8 refactor (dev tool).

Five checks, one process, one summary table. Everything the 2026-08-26
native-fp4 entry claims but could not measure on a GPU-less host:

  1  memory  — the 27B loads with the checkpoint's own nibbles and NO bf16
               master. Runs AFTER the warmup and breaks resident bytes down by
               source (packed / bare bf16 / the f32 embedding cast / KV pool /
               residual), because ``model.params`` is NOT the resident total:
               the sm90 cell's f32 embedding ``Table`` forces a 4.7 GiB f32
               copy of the table into existence on the first embedding call and
               holds it for process life (audit M2). Reports the honest number
               against the refactor's COMPUTED 20.3 GiB (was 65.0, and the
               628c82d baseline measured 67.9 GiB resident); gates only on
               "no bf16 master survived".
  2  logits  — greedy-decode real prompts and print the text. A loader that
               succeeds is not evidence: 628c82d served perfect throughput
               through the WRONG lm_head. Gates on non-degenerate output and
               on different prompts producing different continuations.
  3  w4a8    — the e4m3 range invariant on the REAL kernel: max(6*scale) per
               sampled tensor, then kernel-vs-f32-dequant relative error at
               M=1/8/512. M=1 is the weight path alone (that GEMV dequants in
               f32); M=8/512 are END-TO-END — the sim's 2.3% weight requant
               plus the per-token e4m3 activation quant on top of it.
  4  perf    — decode + prefill vs the 628c82d baseline, same method
               (graph replay, 32 steady-state ticks, prefill chunked at 512).
  5  block   — THE PERF RISK. The refactor moved the scale block 32 -> the
               checkpoint's native 16, both f32, which doubles scale traffic.
               Decode is bandwidth-bound, so the memory win may be paid for
               in ms/tick. Measures decode with the native block-16 scales AND
               with load-time re-blocked block-32 scales, prints the added
               quantization error and the bandwidth-predicted delta next to
               the measured one, so the decode delta is attributed rather
               than guessed. Destructive (rewrites the params) — runs last.

Re-blocking rule: block-32 scale = ``pack_fp4``'s ``block_max/6`` over each
pair of 16-blocks. That is the max of the pair whenever both 16-blocks reach
their grid top, i.e. the tightest scale that clips NOTHING — only the nibbles
re-round onto the e2m1 grid. The mean would put values above 6*scale and clip
them, so it is not used. ``.oscale`` is untouched: the per-row max is the same
value either way, so the e4m3 renormalization survives the re-block.

GPU safety: GPUs 0-5 are the user's own training run. This script pins
CUDA_VISIBLE_DEVICES to one of GPUs 6/7 BEFORE torch can initialize the
driver, refuses to start if that GPU shows util or resident memory, and
asserts torch sees exactly one device. ``--selftest`` skips all of that, so it
forces TILERL_TARGET=cpu instead — otherwise the `auto` target would resolve
to cuda:0, which is physical GPU 0.

Usage (see scripts/POD-VERIFY.md for the full order):
    PYTHONPATH=src python3 scripts/verify_h20_fp4.py /data00/Qwen3.8-27B-NVFP4

Dev-only tooling: no bench entry (AGENTS.md scopes that gate to src/ on the
hot path). Runs on the pod only; ``--selftest`` exercises the harness itself
on a CPU tiny model with no GPU and no checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import replace

#: Physical GPU indices that are ours. 0-5 are the user's training run at
#: 100% util / ~94 GiB each and must never be touched.
OURS = (6, 7)

#: 628c82d, same box, docs/experience/wins/2026-08-26-qwen38-27b-baseline.md.
BASE = {
    "commit": "628c82d",
    "decode_b1_ms": 19.03,
    "decode_b1_tps": 52.6,
    "prefill_tps": {512: 1947.0, 2048: 1847.0, 8192: 1773.0},
    "resident_gib": 67.9,
}
#: The refactor's computed (not measured) weight footprint, and what it replaced.
CLAIM_GIB = {"after": 20.3, "before": 65.0}
#: e4m3: max finite 448, smallest normal 2**-6. The w4a8 kernel casts w*scale
#: into e4m3, so 6*scale above 448 saturates invisibly (no CPU kernel repros it).
E4M3_MAX, E4M3_MIN_NORMAL = 448.0, 2.0**-6


def _die(msg: str) -> None:
    sys.stdout.flush()  # stdout is block-buffered into a log; keep FATAL in order
    print(f"\nFATAL: {msg}\n", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _smi(fields: str, idx: int | None = None, fatal: bool = True) -> list[list[str]]:
    """``-i`` indexes PHYSICAL GPUs, so the pin must not leak into NVML."""
    cmd = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    if idx is not None:
        cmd += ["-i", str(idx)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=True, env=env
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        if fatal:  # pre-run: it cannot prove the GPU is ours and idle
            _die(f"nvidia-smi failed ({exc}) — cannot prove the GPU is ours and idle, refusing")
        print(f"  WARNING: nvidia-smi failed ({exc}); reporting no smi numbers", flush=True)
        return []
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line]


def _nn(x: float) -> float | None:
    """NaN is not JSON — `jq` rejects the bare literal python emits."""
    return None if x != x else x


def _argv_opt(name: str) -> str | None:
    """Read one --opt VALUE / --opt=VALUE out of sys.argv before argparse runs."""
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _pin() -> int | None:
    """Pick our GPU, refuse anything else, refuse a busy one, and pin the env.

    Runs before ``import torch``: CUDA_VISIBLE_DEVICES is read by the driver at
    initialization, and ``torch.cuda.is_available()`` initializes it.
    """
    if "--selftest" in sys.argv or "-h" in sys.argv or "--help" in sys.argv:
        # No pin, no busy probe on this branch, so force the CPU target: the
        # registry maps `auto` to cuda whenever a device is visible, and that
        # device would be physical GPU 0 — the user's training run.
        os.environ["TILERL_TARGET"] = "cpu"
        return None
    cli, env = _argv_opt("--gpu"), (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    want = (cli or env or str(OURS[-1])).strip()
    if cli and env and cli.strip() != env:
        print(f"NOTE: --gpu {cli} overrides the inherited CUDA_VISIBLE_DEVICES={env!r}")
    if want not in {str(g) for g in OURS}:
        _die(
            f"GPU {want!r} is not ours. GPUs 0-5 are the user's training run "
            f"(100% util, ~94 GiB each) — only {list(OURS)} may be used."
        )
    gpu = int(want)

    print("== GPU census (all devices; 0-5 are the user's, do not touch) ==", flush=True)
    for row in _smi("index,name,utilization.gpu,memory.used,memory.total"):
        mark = "  <-- OURS" if row[0] == want else ""
        print(f"  gpu{row[0]} {row[1]:<24} util {row[2]:>3}%  mem {row[3]:>6}/{row[4]} MiB{mark}", flush=True)

    max_util = int(_argv_opt("--max-util") or 10)
    max_used = int(_argv_opt("--max-used-mib") or 256)
    util, used, total = 0, 0, 0
    for probe in range(2):  # two samples: a sweep between them still trips this
        if probe:
            time.sleep(3)
        row = _smi("utilization.gpu,memory.used,memory.total", gpu)[0]
        util, used, total = max(util, int(row[0])), max(used, int(row[1])), int(row[2])
    if util > max_util or used > max_used:
        _die(
            f"gpu{gpu} is BUSY (util {util}% > {max_util}% or {used} MiB > {max_used} MiB "
            "resident). A sibling holding memory at 0% util has caused OOMs before. "
            "Wait, or pass --max-used-mib/--max-util deliberately."
        )
    print(f"  gpu{gpu} clear: util {util}%, {used}/{total} MiB used\n", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = want
    tgt = os.environ.setdefault("TILERL_TARGET", "cuda")
    if tgt != "cuda":
        _die(f"TILERL_TARGET={tgt!r}; this harness measures the CUDA target")
    cache = os.environ.get("TILELANG_CACHE_DIR")
    if not cache and os.path.isdir("/work"):
        cache = os.environ["TILELANG_CACHE_DIR"] = "/work/tilelang_cache"
        print(f"TILELANG_CACHE_DIR unset -> {cache} (persistent across container restarts)")
    elif not cache:
        print(
            "WARNING: TILELANG_CACHE_DIR unset and /work missing — every fresh shape "
            "re-pays a 30-120s NVCC build. Set it to a persistent path.",
            flush=True,
        )
    return gpu


PINNED_GPU = _pin()

import torch  # noqa: E402  — must follow the CUDA_VISIBLE_DEVICES pin

from tilerl.config import qwen38_27b, tiny  # noqa: E402
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine  # noqa: E402
from tilerl.model import build_random, load_hf  # noqa: E402
from tilerl.ops import reference  # noqa: E402
from tilerl.ops.backend import get_backend  # noqa: E402

GIB = float(2**30)
RESULTS: list[dict] = []


def _sync(backend) -> None:
    if backend.device.type == "cuda":
        torch.cuda.synchronize()


def record(n: int, name: str, result: str, evidence: str, **extra) -> None:
    RESULTS.append({"n": n, "name": name, "result": result, "evidence": evidence, **extra})
    print(f"\n  [{n}] {result}  {name}\n      {evidence}\n", flush=True)


def run_check(n: int, name: str, fn, skip: set[int]):
    if n in skip:
        record(n, name, "SKIP", "--skip")
        return None
    print(f"\n{'=' * 78}\n== check {n}: {name}\n{'=' * 78}", flush=True)
    try:
        return fn()
    except Exception:
        traceback.print_exc()
        last = traceback.format_exc().strip().splitlines()[-1]
        record(n, name, "ERROR", last)
        return None


# ------------------------------------------------------------------ check 1


def _weight_bytes(params) -> dict[str, int]:
    """Bytes per storage class. Bare keys are grouped by dtype so a resurrected
    bf16 master shows up as tonnage, not as a missing suffix."""
    out: dict[str, int] = {}
    for k, v in params.items():
        suf = k.rsplit(".", 1)[-1]
        cls = suf if suf in ("wq", "scale", "oscale", "w8", "wscale") else f"bare/{v.dtype}"
        out[cls] = out.get(cls, 0) + v.numel() * v.element_size()
    return out


def _tensor_bytes(*ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


def _sz(b: float) -> str:
    """GiB, but MiB below 1 GiB — the tiny selftest would print every source as
    0.000 GiB, which is the reporting blind spot this check exists to remove."""
    return f"{b / GIB:8.3f} GiB" if abs(b) >= GIB else f"{b / 2**20:8.1f} MiB"


def check_memory(model, backend, engine, gpu: int | None, by: dict[str, int], pre_warm: float):
    """Resident memory AFTER the warmup, attributed to named sources.

    ``model.params`` is not the resident total. ``Backend._embed_table_f32``
    holds an f32 copy of the embedding table (248320x5120 = 4.736 GiB at 27B)
    for process life, solely because the sm90 kernel cell inherits the CPU
    cell's f32 ``Table`` — audit M2. It does not exist until the first
    embedding call, so sampling before the warmup (what this check used to do)
    printed a comfortable PASS with that tonnage uncounted.
    """
    params = model.params
    total = sum(by.values())
    masters = sorted(k for k in params if f"{k}.wq" in params or f"{k}.w8" in params)
    bare = sum(v for c, v in by.items() if c.startswith("bare/"))
    quant_keys = sum(1 for k in params if k.endswith((".wq", ".w8")))
    suffixes = ("wq", "scale", "oscale", "w8", "wscale")
    big_bare = sorted(
        ((v.numel() * v.element_size(), k) for k, v in params.items() if k.rsplit(".", 1)[-1] not in suffixes),
        reverse=True,
    )[:4]

    print("  storage class            GiB")
    for cls, b in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"    {cls:<22} {b / GIB:8.3f}")
    print(f"    {'TOTAL (model.params)':<22} {total / GIB:8.3f}")
    print(f"  quantized tensors: {quant_keys}   bf16-master keys beside them: {len(masters)}")
    print("  largest bare tensors: " + ", ".join(f"{k} {b / GIB:.2f}G" for b, k in big_bare))

    packed = sum(v for c, v in by.items() if c in suffixes)
    # `t is ref()` means _dev found the table already f32 and returned it — no
    # second allocation, and counting it would double-count a params tensor.
    cast = _tensor_bytes(
        *(t for r, _, t in getattr(backend, "_embed_f32", {}).values() if t is not r())
    )
    kv, st = engine._kv, engine._states
    src = {
        "packed fp4/fp8 weights": packed,
        "non-quantized weights (bare)": total - packed,
        "f32 embedding cast (DEFECT M2)": cast,
        "KV pool": _tensor_bytes(kv.k_pool, kv.v_pool),
        "GDN state pool": _tensor_bytes(st.states, st.conv_windows),
    }
    named_gib = sum(src.values()) / GIB

    alloc = res = peak = smi_used = float("nan")
    if backend.device.type == "cuda":
        alloc = torch.cuda.memory_allocated() / GIB
        res = torch.cuda.memory_reserved() / GIB
        peak = torch.cuda.max_memory_allocated() / GIB
        rows = _smi("memory.used", gpu, fatal=False)
        smi_used = int(rows[0][0]) / 1024.0 if rows else float("nan")

    print("\n  resident source (post-warmup)")
    for name, b in src.items():
        print(f"    {name:<35} {_sz(b)}")
    print(f"    {'named subtotal':<35} {_sz(named_gib * GIB)}")
    if backend.device.type == "cuda":
        print(f"    {'everything else (residual)':<35} {_sz((alloc - named_gib) * GIB)}")
        print(f"    {'torch allocated':<35} {_sz(alloc * GIB)}")
        print(
            f"  torch allocated {pre_warm:.2f} -> {alloc:.2f} GiB across the warmup "
            f"(+{alloc - pre_warm:.2f}) | reserved {res:.2f} | "
            f"peak-allocated {peak:.2f} | nvidia-smi used {smi_used:.2f}"
        )
    print(
        f"  HONEST resident weights {_sz(total + cast)} = params {_sz(total)} "
        f"+ f32 embedding cast {_sz(cast)}   (the doc's COMPUTED "
        f"{CLAIM_GIB['after']} GiB counts params only; pre-refactor computed "
        f"{CLAIM_GIB['before']}, 628c82d measured {BASE['resident_gib']} resident)"
    )

    # Gates on "no bf16 master survived", never on a total. The 20.3 GiB is a
    # COMPUTED params-only number that assumed every linear is fp4 (the
    # per-channel FP8 ones stay at 1 B/elem) and that the M2 cast already
    # breaks — gating on it would either lie green or fail on a known defect.
    ok = not masters and bare / GIB <= 4.0 and quant_keys > 0
    why = []
    if masters:
        why.append(f"{len(masters)} bf16 masters still resident, e.g. {masters[:3]}")
    if bare / GIB > 4.0:
        why.append(f"bare tensors {bare / GIB:.2f} GiB > 4.0 (embed alone is ~2.4)")
    if not quant_keys:
        why.append("no .wq/.w8 tensors at all — nothing was served quantized")
    record(
        1,
        "native fp4, no bf16 masters",
        "PASS" if ok else "FAIL",
        f"resident weights {_sz(total + cast).strip()} = params {_sz(total).strip()} "
        f"+ f32 embedding cast {_sz(cast).strip()} (doc computes {CLAIM_GIB['after']} GiB, "
        f"params only); torch allocated {alloc:.2f}, smi {smi_used:.1f} GiB; "
        f"{quant_keys} quantized tensors, {len(masters)} masters"
        + ("" if ok else "  <-- " + "; ".join(why)),
        params_gib=total / GIB,
        embed_cast_gib=cast / GIB,
        resident_weight_gib=(total + cast) / GIB,
        computed_claim_gib=CLAIM_GIB["after"],
        torch_allocated_pre_warmup_gib=_nn(pre_warm),
        torch_allocated_gib=_nn(alloc),
        torch_reserved_gib=_nn(res),
        torch_peak_allocated_gib=_nn(peak),
        smi_used_gib=_nn(smi_used),
        residual_gib=_nn(alloc - named_gib),
        by_class_gib={c: b / GIB for c, b in by.items()},
        by_source_gib={c: b / GIB for c, b in src.items()},
        masters=masters[:8],
    )


# ------------------------------------------------------------------ check 2

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the nth Fibonacci number:\n\ndef fib(n):",
    "Q: What is 17 plus 25?\nA:",
]


def _degenerate(ids: list[int]) -> str | None:
    if len(ids) < 8:
        return f"only {len(ids)} tokens generated"
    counts = Counter(ids)
    if len(counts) <= 2:
        return f"only {len(counts)} distinct ids in {len(ids)} tokens"
    top_id, top = counts.most_common(1)[0]
    if top / len(ids) >= 0.6:
        return f"id {top_id} is {top}/{len(ids)} of the output"
    return None


def check_logits(engine, cfg, source: str, selftest: bool):
    tok, tok_err = None, None
    if not selftest:
        try:
            from tilerl.server import get_tokenizer

            tok = get_tokenizer(source)
        except Exception as exc:  # tokenizer.json absent -> ids-only evidence
            tok_err = f"{type(exc).__name__}: {exc}"
            print(f"  WARNING: no tokenizer from {source} ({tok_err}); reporting ids only")

    outs, texts = [], []
    for i, prompt in enumerate(PROMPTS):
        ids = tok.encode(prompt) if tok else [(i * 97 + j * 13) % cfg.vocab_size for j in range(16)]
        # Without the stop set, greedy decode runs 48 ticks past <|im_end|> and
        # the repeated id trips _degenerate — a false FAIL on a healthy model.
        stops = tuple(getattr(tok, "stop_token_ids", ()))
        wid = engine.submit(
            ids, SamplingParams(temperature=0.0, max_new_tokens=48, seed=0, stop_token_ids=stops)
        )
        out = _drive(engine, wid, 512)
        outs.append(out)
        texts.append(tok.decode(out) if tok else "")
        print(f"  prompt {i}: {prompt!r}" if tok else f"  prompt {i}: ids {ids[:8]}...")
        print(f"    -> {len(out)} tok, {len(set(out))} distinct, ids {out[:12]}")
        if tok:
            print(f"    -> {texts[i]!r}")

    bad = [(i, why) for i, o in enumerate(outs) if (why := _degenerate(o))]
    pairs_same = [
        (i, j) for i in range(len(outs)) for j in range(i + 1, len(outs)) if outs[i] == outs[j]
    ]
    ok = not bad and not pairs_same
    why = []
    if bad:
        why += [f"prompt {i} degenerate: {w}" for i, w in bad]
    if pairs_same:
        why.append(f"prompts {pairs_same} produced IDENTICAL output — lm_head likely wrong")
    record(
        2,
        "logits are not void (greedy decode)",
        "PASS" if ok else "FAIL",
        f"{len(outs)} prompts, all distinct, "
        f"{[len(set(o)) for o in outs]} distinct ids of 48"
        + (f"; NO TOKENIZER ({tok_err}) — ids only, read the text yourself next run" if tok_err else "")
        + ("" if ok else "  <-- " + "; ".join(why)),
        texts=texts,
        first_ids=[o[:12] for o in outs],
        tokenizer=source if tok else None,
    )
    return texts


# ------------------------------------------------------------------ check 3


def _fp4_keys(params, n: int) -> list[str]:
    """``n`` fp4 keys spread across the model, plus the largest one."""
    keys = sorted(k[:-3] for k in params if k.endswith(".wq"))
    if not keys:
        return []
    stride = max(1, len(keys) // max(1, n))
    picked = keys[::stride][:n]
    biggest = max(keys, key=lambda k: params[k + ".wq"].numel())
    return sorted(set(picked) | {biggest})


def _deq_rows(wq, scale, oscale=None, rows: int = 512):
    """f32 dequant in row chunks. The whole lm_head [248320, 5120] at once is
    ~30 GiB of int64 intermediates inside ``reference.dequant_fp4``."""
    if getattr(wq, "_tl_twiddled", False):  # slicing drops the flag; untwiddle whole
        wq = reference.untwiddle_fp4(wq)
    for i in range(0, wq.shape[0], rows):
        w = reference.dequant_fp4(wq[i : i + rows], scale[i : i + rows]).float()
        yield w if oscale is None else w * oscale[i : i + rows].float().reshape(-1, 1)


def check_w4a8(model, backend, sample: int):
    params = model.params
    keys = _fp4_keys(params, sample)
    if not keys:
        record(3, "w4a8 e4m3 range + kernel error", "FAIL", "no .wq tensors in the model")
        return None

    print(f"  {'tensor':<26} {'N':>7} {'K':>6} {'B':>3} {'6*max':>8} {'6*min':>10} {'subnorm%':>9}")
    rows, worst = {}, {}
    for key in keys:
        wq, scale = params[key + ".wq"], params[key + ".scale"]
        oscale = params.get(key + ".oscale")
        n, k = wq.shape[0], wq.shape[1] * 2
        blk = k // scale.shape[1]
        s6 = (6.0 * scale.float()).abs()
        smax, smin = s6.max().item(), s6[s6 > 0].min().item() if (s6 > 0).any() else 0.0
        sub = 100.0 * (s6 < E4M3_MIN_NORMAL).float().mean().item()
        print(f"  {key:<26} {n:>7} {k:>6} {blk:>3} {smax:>8.3f} {smin:>10.2e} {sub:>8.2f}%")
        rows[key] = {"n": n, "k": k, "block": blk, "smax6": smax, "smin6": smin, "subnormal_pct": sub}

        errs = {}
        for m in (1, 8, 512):
            x = torch.randn(m, k, device=wq.device, dtype=torch.bfloat16)
            y = backend.linear_fp4(x, wq, scale, oscale=oscale).float()
            xf = x.float()
            ref = torch.cat([xf @ w.t() for w in _deq_rows(wq, scale, oscale)], dim=1)
            fro = ((y - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            mx = ((y - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
            errs[m] = {"fro_relerr": fro, "max_relerr": mx}
            worst[m] = max(worst.get(m, 0.0), fro)
            del ref, y
        rows[key]["kernel_relerr"] = errs
        print(
            "      kernel vs f32 dequant: "
            + "  ".join(f"M={m} fro {e['fro_relerr']:.4f} max {e['max_relerr']:.4f}" for m, e in errs.items())
        )

    over = {k: v["smax6"] for k, v in rows.items() if v["smax6"] > E4M3_MAX}
    band = {k: v["smax6"] for k, v in rows.items() if not (6.0 <= v["smax6"] < 12.0)}
    ok = not over and worst.get(1, 1) <= 0.01 and worst.get(8, 1) <= 0.05 and worst.get(512, 1) <= 0.05
    why = []
    if over:
        why.append(f"6*max(scale) SATURATES e4m3 ({E4M3_MAX}) for {over}")
    if worst.get(1, 0) > 0.01:
        why.append(f"M=1 GEMV fro-relerr {worst[1]:.4f} > 1% (that arm dequants in f32)")
    for m in (8, 512):
        if worst.get(m, 0) > 0.05:
            why.append(f"M={m} w4a8 fro-relerr {worst[m]:.4f} > 5%")
    if band:
        print(f"  NOTE: 6*max(scale) outside the renormalized [6,12) band for {band}")
    record(
        3,
        "w4a8 e4m3 range invariant on the real kernel",
        "PASS" if ok else "FAIL",
        f"max 6*scale {max(v['smax6'] for v in rows.values()):.2f} (e4m3 cap {E4M3_MAX}); "
        f"worst fro-relerr vs f32 dequant M=1 {worst.get(1, float('nan')):.4f} / "
        f"M=8 {worst.get(8, float('nan')):.4f} / M=512 {worst.get(512, float('nan')):.4f} "
        f"(M=1 is the weight path alone and should be ~0; M=8/512 are END-TO-END — "
        f"the e4m3 weight requant the sim put at 0.023 PLUS the per-token e4m3 "
        f"activation quant, so ~0.03-0.04, not 0.023)"
        + ("" if ok else "  <-- " + "; ".join(why)),
        tensors=rows,
        worst_fro_relerr=worst,
        expectation={"m1_weight_path": 0.0, "m8_m512_end_to_end": 0.035},
    )
    return rows


# ------------------------------------------------------------------ check 4


def _rand_prompt(vocab: int, n: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=gen).tolist()


def _drive(engine, wid: int, max_steps: int) -> list[int]:
    for _ in range(max_steps):
        engine.step()
        done = engine.poll()
        if wid in done:
            return done[wid]
    raise RuntimeError(f"request {wid} did not finish in {max_steps} steps")


def _warmup(engine, cfg, passes: int = 2) -> None:
    """Compile the prefill shapes (chunks are always M=512) and the B=1 decode.
    Batched decode has its OWN shapes and its own lazy CUDA-graph capture, so
    each batch size warms itself in :func:`time_decode`; the second pass
    confirms the timed numbers are JIT-free."""
    for p in range(passes):
        t0 = time.perf_counter()
        wid = engine.submit(
            _rand_prompt(cfg.vocab_size, 512, seed=p),
            SamplingParams(temperature=0.0, max_new_tokens=4, seed=0),
        )
        _drive(engine, wid, 1024)
        print(f"  warmup pass {p + 1}: {time.perf_counter() - t0:.1f}s", flush=True)


def time_decode(engine, backend, cfg, b: int, ticks: int, warm: int = 8) -> float:
    """Steady-state decode at batch ``b``: settle + warm untimed, then ``ticks``
    timed graph replays.

    The engine admits ONE waiting request per tick and a 512-token prompt needs
    two prefill ticks, so a fixed settle count leaves rows still prefilling and
    puts mixed prefill+decode forwards inside the timed window. Settle on the
    phases instead. The warm ticks then pay this batch size's JIT and its lazy
    CUDA-graph capture, which happen on the FIRST pure-decode tick.
    """
    wids = [
        engine.submit(
            _rand_prompt(cfg.vocab_size, 512, seed=200 + i),
            SamplingParams(temperature=0.0, max_new_tokens=warm + ticks + 4 * b + 32, seed=i),
        )
        for i in range(b)
    ]
    for _ in range(4 * b + 32):
        engine.step()
        run = engine._running
        if len(run) == b and all(r.phase == _PHASE_DECODE for r in run):
            break
    else:
        raise RuntimeError(f"decode B={b} never reached {b} pure-decode rows")
    for _ in range(warm):
        engine.step()
    _sync(backend)
    t0 = time.perf_counter()
    for _ in range(ticks):
        engine.step()
    _sync(backend)
    ms = (time.perf_counter() - t0) / ticks * 1e3
    done: dict[int, list[int]] = {}
    for _ in range(warm + ticks + 8 * b + 64):
        done.update(engine.poll())
        if all(w in done for w in wids):
            break
        engine.step()
    if not all(w in done for w in wids):
        raise RuntimeError(f"decode B={b} did not drain — later measurements would share its blocks")
    return ms


def time_prefill(engine, backend, cfg, length: int, decode_ms: float) -> tuple[float, float]:
    """One request timed to completion (prefill chunks + the 1-token decode
    finish); prefill-only subtracts that one decode tick. Baseline method."""
    wid = engine.submit(
        _rand_prompt(cfg.vocab_size, length, seed=length),
        SamplingParams(temperature=0.0, max_new_tokens=1, seed=0),
    )
    _sync(backend)
    t0 = time.perf_counter()
    _drive(engine, wid, max(1024, length // 64))
    _sync(backend)
    prefill_ms = (time.perf_counter() - t0) * 1e3 - decode_ms
    return prefill_ms, 1000.0 * length / prefill_ms


def check_perf(engine, backend, cfg, batches, prefills, ticks, stream_gib: float):
    dec = {}
    for b in batches:
        ms = time_decode(engine, backend, cfg, b, ticks)
        # Snapshot per batch: a failed capture at B=8 sets the flag False
        # process-wide and would otherwise retroactively invalidate B=1.
        dec[b] = {
            "ms_per_tick": ms,
            "per_req_tok_s": 1000.0 / ms,
            "aggregate_tok_s": 1000.0 * b / ms,
            "decode_graph_on": bool(engine._decode_graph_on),
        }
        base_ms = BASE["decode_b1_ms"] if b == 1 else None
        note = f"  ({base_ms / ms:.3f}x of the {base_ms} ms baseline)" if base_ms else ""
        print(f"  decode B={b}: {ms:9.3f} ms/tick  {1000.0 * b / ms:8.1f} agg tok/s{note}", flush=True)
    bw = stream_gib / (dec[1]["ms_per_tick"] / 1e3)
    print(f"  implied weight-stream bandwidth at B=1: {bw:.0f} GiB/s")

    pre = {}
    print(f"  {'len':>6} {'ms/tok':>10} {'tok/s':>10} {'vs base':>9}")
    for length in prefills:
        ms, tps = time_prefill(engine, backend, cfg, length, dec[1]["ms_per_tick"])
        base = BASE["prefill_tps"].get(length)
        pre[length] = {"ms_per_tok": ms / length, "tok_s": tps, "ratio": tps / base if base else None}
        print(
            f"  {length:>6} {ms / length:>10.4f} {tps:>10.1f} "
            f"{(f'{tps / base:.3f}x' if base else '-'):>9}",
            flush=True,
        )

    base_tps = BASE["decode_b1_tps"]
    dec_ratio = dec[1]["per_req_tok_s"] / base_tps if base_tps else None
    pre_ratios = [v["ratio"] for v in pre.values() if v["ratio"] is not None]
    # A silent eager fallback makes these numbers incomparable to the baseline,
    # which was captured with the decode graph on. Only B=1 has a baseline.
    graph = dec[1]["decode_graph_on"] or backend.device.type != "cuda"
    off = [b for b, v in dec.items() if not v["decode_graph_on"]]
    ok = (dec_ratio is None or dec_ratio >= 0.95) and all(r >= 0.95 for r in pre_ratios) and graph
    why = []
    if dec_ratio is not None and dec_ratio < 0.95:
        why.append(f"decode {dec_ratio:.3f}x of baseline — a regression to attribute (see check 5)")
    if any(r < 0.95 for r in pre_ratios):
        why.append("prefill below 0.95x of baseline")
    if not graph:
        why.append("decode graph OFF at B=1 — the tick fell back to eager, not comparable")
    elif off:
        print(f"  NOTE: decode graph capture failed at B={off}; B=1 is unaffected and stands")
    record(
        4,
        f"throughput vs {BASE['commit']} baseline",
        "PASS" if ok else "FAIL",
        f"decode B=1 {dec[1]['ms_per_tick']:.2f} ms/tick = {dec[1]['per_req_tok_s']:.1f} tok/s"
        + (f" ({dec_ratio:.3f}x of {base_tps})" if dec_ratio else " (no baseline)")
        + "; prefill "
        + ", ".join(
            f"{length}:{v['tok_s']:.0f}tok/s" + (f" {v['ratio']:.3f}x" if v["ratio"] else "")
            for length, v in pre.items()
        )
        + ("" if ok else "  <-- " + "; ".join(why)),
        decode=dec,
        prefill={str(k): v for k, v in pre.items()},
        baseline=BASE,
        decode_ratio=dec_ratio,
        decode_graph_on=dec[1]["decode_graph_on"],
        implied_bw_gib_s=bw,
    )
    return dec, pre


# ------------------------------------------------------------------ check 5


def _repack32(wq, scale, rows: int = 512):
    """Re-block a native block-16 scale grid to block 32.

    ``pack_fp4``'s scale is ``block_max/6`` over the merged pair — the max of
    the two 16-block scales whenever both blocks reach their grid top, and the
    tightest scale that clips nothing otherwise. (A mean would push values past
    6*scale and clamp them.) Row-chunked because pack_fp4's round-to-nearest
    allocates 32 bytes per weight element.
    """
    wqs, scales = [], []
    if getattr(wq, "_tl_twiddled", False):
        wq = reference.untwiddle_fp4(wq)
    for i in range(0, wq.shape[0], rows):
        w = reference.dequant_fp4(wq[i : i + rows], scale[i : i + rows])
        a, b = reference.pack_fp4(w, block=32)
        wqs.append(a)
        scales.append(b)
        del w
    return torch.cat(wqs).contiguous(), torch.cat(scales).contiguous()


def check_block(model, backend, cfg, args, dec16, by16, stream16_gib: float):
    params = model.params
    keys = sorted(k[:-3] for k in params if k.endswith(".wq"))
    b16 = [k for k in keys if params[k + ".wq"].shape[1] * 2 // params[k + ".scale"].shape[1] == 16]
    if not b16:
        record(5, "scale block 16 vs 32", "SKIP", "no block-16 scale tensors to re-block")
        return None
    sampled = set(_fp4_keys(params, args.sample))

    print(f"  re-blocking {len(b16)} of {len(keys)} fp4 tensors, 16 -> 32 ...", flush=True)
    t0, errs, before, after = time.perf_counter(), {}, 0, 0
    for i, key in enumerate(b16):
        wq, sc = params[key + ".wq"], params[key + ".scale"]
        before += sc.numel() * sc.element_size()
        wq32, sc32 = _repack32(wq, sc)
        after += sc32.numel() * sc32.element_size()
        if key in sampled:  # added quantization error, on the sampled tensors only
            num = den = 0.0
            for a, b32 in zip(_deq_rows(wq, sc), _deq_rows(wq32, sc32)):
                num += float(((b32 - a) ** 2).sum())
                den += float((a**2).sum())
            errs[key] = {
                "fro_relerr": (num / max(den, 1e-30)) ** 0.5,
                "smax6_b16": (6.0 * sc.float()).max().item(),
                "smax6_b32": (6.0 * sc32.float()).max().item(),
            }
        params[key + ".wq"], params[key + ".scale"] = wq32, sc32
        del wq, sc
        if i % 64 == 0:
            gc.collect()
            if backend.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"    {i + 1}/{len(b16)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    gc.collect()
    if backend.device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  re-block took {time.perf_counter() - t0:.1f}s", flush=True)
    for k, v in errs.items():
        print(
            f"    {k:<26} added weight fro-relerr {v['fro_relerr']:.5f}  "
            f"6*max(scale) {v['smax6_b16']:.3f} -> {v['smax6_b32']:.3f}"
        )

    engine = _build(cfg, model, backend, args)
    _warmup(engine, cfg)
    dec32 = {}
    for b in args.batches:
        ms = time_decode(engine, backend, cfg, b, args.decode_ticks)
        dec32[b] = {"ms_per_tick": ms, "aggregate_tok_s": 1000.0 * b / ms}
        d = 100.0 * (dec16[b]["ms_per_tick"] / ms - 1.0)
        print(
            f"  decode B={b}: b16 {dec16[b]['ms_per_tick']:8.3f}  b32 {ms:8.3f} ms/tick  "
            f"-> block-32 is {d:+.2f}% faster",
            flush=True,
        )

    saved = (before - after) / GIB
    stream32 = stream16_gib - saved
    bw = stream16_gib / (dec16[1]["ms_per_tick"] / 1e3)  # GiB/s implied by arm b16
    predicted_ms = dec16[1]["ms_per_tick"] - stream32 / bw * 1e3  # if purely bandwidth-bound
    measured_ms = dec16[1]["ms_per_tick"] - dec32[1]["ms_per_tick"]
    max_err = max((v["fro_relerr"] for v in errs.values()), default=float("nan"))
    print(
        f"\n  scale bytes {before / GIB:.2f} -> {after / GIB:.2f} GiB (saves {saved:.2f});"
        f" weight stream {stream16_gib:.2f} -> {stream32:.2f} GiB/tick"
    )
    attrib = measured_ms / predicted_ms if predicted_ms > 1e-9 else float("nan")
    print(
        f"  B=1 ms/tick saved by block-32 (+ = b32 faster): predicted {predicted_ms:+.2f} if "
        f"purely bandwidth-bound at {bw:.0f} GiB/s, measured {measured_ms:+.2f} "
        f"({attrib:.2f}x of prediction)"
    )
    verdict = (
        "block 16 costs real decode time; the memory win is paid for in ms/tick"
        if attrib >= 0.5
        else "the block change is NOT where the decode time goes — look elsewhere"
    )
    record(
        5,
        "PERF RISK: native block-16 scales vs re-blocked block-32",
        "INFO",  # an attribution arm: it reports a delta, it asserts nothing
        f"B=1 {dec16[1]['ms_per_tick']:.2f} (b16) vs {dec32[1]['ms_per_tick']:.2f} (b32) ms/tick "
        f"= block-32 {100.0 * (dec16[1]['ms_per_tick'] / dec32[1]['ms_per_tick'] - 1):+.1f}%; "
        f"scale bytes {before / GIB:.2f} -> {after / GIB:.2f} GiB; ms/tick saved predicted "
        f"{predicted_ms:+.2f} vs measured {measured_ms:+.2f}; block-32 adds <= {max_err:.4f} "
        f"weight fro-relerr (diagnostic arm, not a shippable config at that error). {verdict}",
        decode_b16={str(k): v for k, v in dec16.items()},
        decode_b32={str(k): v for k, v in dec32.items()},
        scale_gib_b16=before / GIB,
        scale_gib_b32=after / GIB,
        stream_gib_b16=stream16_gib,
        stream_gib_b32=stream32,
        implied_bw_gib_s=bw,
        predicted_ms_saved=predicted_ms,
        measured_ms_saved=measured_ms,
        added_weight_relerr=errs,
        reblocked_tensors=len(b16),
        by_class_gib_b16={c: b / GIB for c, b in by16.items()},
    )
    return dec32


# ------------------------------------------------------------------ driver


def _build(cfg, model, backend, args):
    """Serving build with the baseline's pool sizes (256 blocks = 4096 tokens
    is too small for an 8192-token prompt; per-tick kernels are identical)."""
    return build_engine(
        cfg,
        model,
        backend,
        num_blocks=args.num_blocks,
        num_slots=max(16, max(args.batches)),
        max_batch=max(8, max(args.batches)),
        max_total_tokens=args.max_total_tokens,
    )


def _print_rope_tie(source: str) -> None:
    """`_validate_hf_config`'s rope/tie guards raise only on a key that is
    PRESENT and wrong, so an absent one is unchecked and invisible. Print the
    raw five before the 6-minute load — `None` means nothing validated it."""
    with open(os.path.join(source, "config.json")) as f:
        hf = json.load(f)
    txt = hf.get("text_config", hf)
    rope = txt.get("rope_parameters") or {}
    vals = {
        "rope_theta": rope.get("rope_theta", txt.get("rope_theta")),
        "partial_rotary_factor": rope.get(
            "partial_rotary_factor", txt.get("partial_rotary_factor")
        ),
        "rope_scaling": txt.get("rope_scaling"),
        "rope_parameters.rope_type": rope.get("rope_type"),
        "tie_word_embeddings": hf.get("tie_word_embeddings", txt.get("tie_word_embeddings")),
    }
    print("  config.json rope/tie (None = ABSENT, hence unvalidated):", flush=True)
    for k, v in vals.items():
        print(f"    {k:<26} {v!r}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", default="/data00/Qwen3.8-27B-NVFP4")
    p.add_argument("--gpu", type=int, default=OURS[-1], help=f"physical GPU; only {list(OURS)} allowed")
    p.add_argument("--max-util", type=int, default=10, help="refuse to start above this GPU util %%")
    p.add_argument("--max-used-mib", type=int, default=256, help="refuse to start above this resident MiB")
    p.add_argument("--decode-ticks", type=int, default=32)
    p.add_argument("--batches", default="1,8", help="B=1 is always measured (it owns the baseline)")
    p.add_argument("--prefill", default="512,2048,8192")
    p.add_argument("--sample", type=int, default=6, help="fp4 tensors sampled for checks 3 and 5")
    p.add_argument("--skip", default="", help="comma-separated check numbers to skip, e.g. 5")
    p.add_argument("--num-blocks", type=int, default=1024)
    p.add_argument("--max-total-tokens", type=int, default=16384)
    p.add_argument("--json", default="", help="also write the report here")
    p.add_argument("--selftest", action="store_true", help="exercise the harness on a CPU tiny model")
    args = p.parse_args()
    args.batches = sorted({1} | {int(x) for x in args.batches.split(",") if x})
    args.prefill = [int(x) for x in args.prefill.split(",") if x]
    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    t_start = time.perf_counter()

    backend = get_backend()
    if args.selftest:
        print("== SELFTEST: CPU tiny model, no GPU, no checkpoint ==", flush=True)
        cfg = replace(tiny(), fp4=True)
        model = build_random(cfg, seed=0, fuse_projections=True)
        for k in [k[:-3] for k in list(model.params) if k.endswith(".wq")]:  # force block 16
            w = reference.dequant_fp4(model.params[k + ".wq"], model.params[k + ".scale"])
            wq, sc = reference.pack_fp4(w, block=16)
            sc, osc = reference.renorm_fp4_scale(sc, model.params[k + ".oscale"])
            model.params[k + ".wq"], model.params[k + ".scale"], model.params[k + ".oscale"] = wq, sc, osc
        args.batches, args.prefill = [1, 2], [64]  # B=2 exercises the batched settle
        args.decode_ticks, args.num_blocks, args.max_total_tokens = 4, 128, 2048
        BASE["decode_b1_tps"] = BASE["decode_b1_ms"] = None  # no baseline for a tiny model
        BASE["prefill_tps"] = {}
    else:
        if backend.device.type != "cuda":
            _die(f"backend resolved to {backend.target!r}; this harness measures the CUDA target")
        if torch.cuda.device_count() != 1:
            _die(
                f"torch sees {torch.cuda.device_count()} devices — the CUDA_VISIBLE_DEVICES pin "
                "did not take (CUDA was initialized before it). Refusing: a stray allocation "
                "would land on the user's training GPUs."
            )
        if not os.path.isdir(args.source):
            _die(f"checkpoint dir {args.source!r} does not exist")
        shards = [f for f in os.listdir(args.source) if f.endswith(".safetensors")]
        if "model.safetensors" not in shards:
            _die(f"{args.source} has no model.safetensors (found {sorted(shards)[:4]})")
        print(f"checkpoint: {args.source} ({sorted(shards)})", flush=True)
        _print_rope_tie(args.source)
        cfg = qwen38_27b()
        t0 = time.perf_counter()
        model = load_hf(cfg, args.source, fuse_projections=True)
        print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    engine = _build(cfg, model, backend, args)  # materialize() moves params to the device
    pre_warm = torch.cuda.memory_allocated() / GIB if backend.device.type == "cuda" else float("nan")
    if backend.device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(
            f"gpu: {torch.cuda.get_device_name(0)} | after engine build "
            f"{(total - free) / GIB:.1f}/{total / GIB:.1f} GiB | torch allocated {pre_warm:.2f} GiB",
            flush=True,
        )

    by = _weight_bytes(model.params)
    # bytes the decode tick streams: every linear weight, every tick. The
    # embedding table is a single-row gather, not a stream.
    emb = sum(
        v.numel() * v.element_size() for k, v in model.params.items() if k.startswith("embed_tokens")
    )
    stream_gib = (sum(by.values()) - emb) / GIB
    print(f"\nweight stream per decode tick: {stream_gib:.2f} GiB (total minus the embedding table)")

    _warmup(engine, cfg)
    print(f"decode_graph_on: {engine._decode_graph_on}", flush=True)

    # Check 1 runs AFTER the warmup on purpose: _embed_table_f32's f32 copy of
    # the embedding table does not exist until the first embedding call, and
    # sampling before it reported a comfortable PASS with 4.7 GiB uncounted.
    run_check(
        1,
        "native fp4, no bf16 masters",
        lambda: check_memory(model, backend, engine, PINNED_GPU, by, pre_warm),
        skip,
    )

    run_check(2, "logits are not void", lambda: check_logits(engine, cfg, args.source, args.selftest), skip)
    run_check(3, "w4a8 e4m3 range invariant", lambda: check_w4a8(model, backend, args.sample), skip)
    perf4 = run_check(
        4,
        f"throughput vs {BASE['commit']}",
        lambda: check_perf(engine, backend, cfg, args.batches, args.prefill, args.decode_ticks, stream_gib),
        skip,
    )

    if 5 in skip or perf4 is None:
        why = "--skip" if 5 in skip else "check 4 produced no block-16 decode baseline"
        record(5, "scale block 16 vs 32", "SKIP", why)
    else:
        del engine
        gc.collect()
        if backend.device.type == "cuda":
            torch.cuda.empty_cache()
        run_check(
            5,
            "scale block 16 vs 32",
            lambda: check_block(model, backend, cfg, args, perf4[0], by, stream_gib),
            skip,
        )

    print(f"\n{'=' * 78}\n== SUMMARY\n{'=' * 78}")
    for r in RESULTS:
        print(f"  {r['n']}  {r['result']:<6} {r['name']}")
        print(f"        {r['evidence']}")
    # A SKIP is not green: --skip 5 must not exit 0 on a claim nothing tested.
    # INFO is check 5's attribution arm, which reports rather than asserts.
    bad = [r for r in RESULTS if r["result"] in ("FAIL", "ERROR", "SKIP")]
    tally = ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["result"] for r in RESULTS).items()))
    print(
        f"\n  {tally}  in {(time.perf_counter() - t_start) / 60:.1f} min"
        + (f"; NOT GREEN: {[(r['n'], r['result']) for r in bad]}" if bad else "")
    )
    # Written before the census: one transient nvidia-smi must not cost the run.
    report = {"commit": os.environ.get("BENCH_COMMIT", "?"), "source": args.source, "checks": RESULTS}
    blob = json.dumps(report, sort_keys=True, default=str)
    print("\nJSON " + blob, flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(blob)
    if backend.device.type == "cuda":
        print("\n== GPU census after the run (0-5 must be untouched by us) ==")
        for row in _smi("index,utilization.gpu,memory.used", fatal=False):
            print(f"  gpu{row[0]} util {row[1]:>3}%  mem {row[2]:>6} MiB")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
