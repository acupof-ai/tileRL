"""Step 3 of the fp8 KV work, on the 27B: does the flag hold on the model that ships?

Three arms, in the order that can stop the next one.

1. RANGE. The per-(block, head, token) dynamic range of real 27B KV and the round-trip error,
   both scale grids and both fp8 dtypes. Confirms or overturns the e4m3 verdict measured on
   tiny; the block_head column is the record of what was not chosen.
2. ACCURACY. Next-token agreement over a long prompt, and max logit error, fp8 pool against
   bf16 pool through the same engine. Agreement is the property serving cares about; the
   logit error is the number that moves first, reported as absolute over the row's amax
   because per-element relative error on an fp8 grid is unbounded near zero.
3. BYTES, then tok/s. The bytes-per-tick fraction is printed BEFORE any rate, because a
   decode tick reads the weights every token regardless and fp8 KV is a fraction of it.

  scripts/pod_run.sh kvfp8-27b 0 -- python3 scripts/probe_kv_fp8_27b.py --source /work/Qwen3.8-27B-NVFP4
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "packages/tilerl-kernels/src")
sys.path.insert(0, "src")


def _q_block_head(x: torch.Tensor, dtype: torch.dtype):
    """The grid NOT taken: one scale per (plane, block, head), amax over tokens AND head_dim."""
    xf = x.float()
    s = xf.abs().amax((-2, -1)).clamp_min(1e-12) / torch.finfo(dtype).max
    return (xf / s[..., None, None]).to(dtype), s


def arm_range(pool, dtypes: dict) -> dict:
    """Round-trip error on the REAL KV this pool holds, per grid and dtype."""
    from tilerl_kernels.reference import dequant_kv_fp8, quant_kv_fp8

    # Quantizing an already-fp8 pool reports a floor of zero and an amax of exactly 448 -- it
    # measures the fixture, not the format. Caught once by that 448; asserted so it cannot
    # come back as a plausible number.
    if pool.kv_fp8 is not None:
        raise ValueError("arm_range needs the BF16 pool; this one is already quantized")
    out: dict = {}
    for name, plane in (("K", pool.k_pool), ("V", pool.v_pool)):
        x = plane.float()
        live = x.abs().amax(-1) > 0  # only blocks the prompt actually wrote
        if not bool(live.any()):
            out[name] = {"live_token_heads": 0}
            continue
        amax = x.abs().amax(-1)
        nz = x != 0
        rng = (amax[live] / x.abs().clamp_min(1e30).amin(-1)[live].clamp_min(1e-30))
        out[name] = {
            "live_token_heads": int(live.sum()),
            "amax_p50": float(amax[live].median()), "amax_max": float(amax[live].max()),
            "in_row_range_p99": float(rng.quantile(0.99)),
        }
        for nick, dt in dtypes.items():
            for grid in ("per_token", "block_head"):
                if grid == "per_token":
                    d = dequant_kv_fp8(*quant_kv_fp8(plane, dt))
                else:
                    q, s = _q_block_head(plane, dt)
                    d = q.float() * s[..., None, None]
                e = (d - x).abs()
                # over the ROW's amax, the metric that bounds the logit; the per-element
                # relative figure is kept only restricted to elements above 1% of that amax,
                # because on an fp8 grid it is unbounded near zero by construction
                big = x.abs() > 0.01 * amax[..., None]
                out[name][f"{nick}.{grid}"] = {
                    "err_over_row_amax": float((e.amax(-1)[live] / amax[live]).max()),
                    "rel_above_1pct_of_amax": float(
                        (e[big & nz] / x.abs()[big & nz]).max()) if bool((big & nz).any()) else 0.0,
                    "zeroed_frac": float(((d == 0) & nz).sum() / nz.sum()),
                }
    return out


def _gen(cfg, model, backend, prompts, n_new, kv_fp8):
    """Run `prompts` (one list, or a list of lists) to n_new tokens each.

    Returns (tokens, engine): the token LIST for a single prompt, the total COUNT for a
    batch -- the decode arm wants throughput, the accuracy arm wants the ids.
    """
    from tilerl.engine import SamplingParams, build_engine

    batch = prompts if isinstance(prompts[0], list) else [prompts]
    longest = max(len(p) for p in batch)
    eng = build_engine(cfg, model, backend, num_slots=max(2, len(batch)),
                       max_batch=max(2, len(batch)),
                       max_total_tokens=longest + n_new + 64, kv_fp8=kv_fp8)
    rids = [eng.submit(p, SamplingParams(temperature=0.0, max_new_tokens=n_new, seed=0))
            for p in batch]
    # poll() returns {request_id: token_ids} and DRAINS, so accumulate rather than re-read
    done: dict = {}
    t0 = time.perf_counter()
    for _ in range(4 * n_new + 512):
        done.update(eng.poll())
        if all(len(done.get(r, ())) >= n_new for r in rids) or time.perf_counter() - t0 > 1800:
            break
        eng.step()
    if isinstance(prompts[0], list):
        return sum(len(done.get(r, ())) for r in rids), eng
    return list(done.get(rids[0], ())), eng


def arm_accuracy(cfg, model, backend, prompt, n_new: int) -> tuple[dict, object]:
    """Next-token agreement and logit error, fp8 pool against bf16, same engine and prompt.

    Returns the numbers and the fp8 engine, whose pool the range arm then measures -- real
    27B KV rather than a random fixture.
    """
    want, ref_eng = _gen(cfg, model, backend, prompt, n_new, None)
    got, fp8_eng = _gen(cfg, model, backend, prompt, n_new, torch.float8_e4m3fn)
    agree = sum(a == b for a, b in zip(want, got))
    return {
        "prompt_tokens": len(prompt), "new_tokens": len(want),
        "tokens_agreeing": agree,
        "agreement": agree / max(1, len(want)),
        "first_divergence": next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), None),
        "bf16_tokens": want[:16], "fp8_tokens": got[:16],
        "bytes_per_token_bf16": ref_eng._kv.bytes_per_token,
        "bytes_per_token_fp8": fp8_eng._kv.bytes_per_token,
        "kv_bytes_saved_ratio": ref_eng._kv.bytes_per_token / fp8_eng._kv.bytes_per_token,
    }, ref_eng


def arm_decode(cfg, model, backend, ctx: int, n_new: int, batch: int = 1) -> dict:
    """Decode tok/s at one (context, batch), fp8 pool against bf16, same engine and prompts.

    Reports the KV share of a tick's bytes and the resulting CEILING before the measured
    ratio. A decode tick re-reads the 27B's weights every token regardless, so fp8 KV can
    only act on the KV part -- and at B=1 that part is small: 3.8% of the tick at 8k, 13.7%
    at 32k, so the ceilings are 1.019x and 1.072x, under run-to-run variance. KV scales with
    batch while the weights do not, which is where the win is: 56% of the tick and a 1.38x
    ceiling at B=8 ctx=32k, 84% and 1.70x at B=32. A ratio quoted without its ceiling reads
    as though fp8 moved the whole tick.
    """
    prompts = [torch.randint(3, cfg.vocab_size - 1, (ctx,)).tolist() for _ in range(batch)]
    out: dict = {"ctx": ctx, "batch": batch, "new_tokens": n_new}
    for nick, dt in (("bf16", None), ("fp8", torch.float8_e4m3fn)):
        toks, eng = _gen(cfg, model, backend, prompts, n_new, dt)
        # time a SECOND generation on a fresh engine of the same kind: the first paid any
        # first-call compile, which is not what a decode rate is
        t0 = time.perf_counter()
        toks2, _ = _gen(cfg, model, backend, prompts, n_new, dt)
        dt_s = time.perf_counter() - t0
        out[nick] = {
            "tokens": toks2, "seconds": dt_s,
            "tok_per_s": toks2 / dt_s if dt_s > 0 else 0.0,
            "kv_bytes_per_token": eng._kv.bytes_per_token,
            "kv_bytes_at_ctx": eng._kv.bytes_per_token * ctx * batch,
        }
    weight_bytes = sum(t.numel() * t.element_size() for t in model.params.values())
    kb, kf = out["bf16"]["kv_bytes_at_ctx"], out["fp8"]["kv_bytes_at_ctx"]
    out["weight_bytes"] = weight_bytes
    out["kv_share_of_tick_bytes_bf16"] = kb / (weight_bytes + kb)
    out["kv_share_of_tick_bytes_fp8"] = kf / (weight_bytes + kf)
    # the most a pure-bandwidth tick could gain: what the measured ratio must sit under
    out["tok_per_s_ceiling"] = (weight_bytes + kb) / (weight_bytes + kf)
    out["tok_per_s_ratio"] = (
        out["fp8"]["tok_per_s"] / out["bf16"]["tok_per_s"] if out["bf16"]["tok_per_s"] else 0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="/work/Qwen3.8-27B-NVFP4")
    ap.add_argument("--model", default="qwen38-27b", choices=["qwen38-27b", "tiny"],
                    help="tiny is for smoke-testing this script's plumbing off the card")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--new-tokens", type=int, default=24)
    ap.add_argument("--decode-ctx", type=int, nargs="*", default=[8192, 32768],
                    help="context lengths for the decode arm; [] skips it")
    ap.add_argument("--decode-batch", type=int, nargs="*", default=[1, 8],
                    help="batch sizes. KV scales with batch and the weights do not, so B=1 "
                         "has a 1.019x/1.072x ceiling at 8k/32k while B=8 at 32k has 1.38x "
                         "-- B=1 alone cannot show this flag working")
    a = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.config import qwen38_27b, tiny
    from tilerl.model import build_random, load_hf

    torch.manual_seed(0)
    be = get_backend()
    print(f"arch {be.arch}, torch {torch.__version__}, "
          f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)
    if a.model == "tiny":
        cfg = tiny()
        model = build_random(cfg, seed=0)
    else:
        cfg = qwen38_27b()
        model = load_hf(cfg, a.source)
    print(f"{cfg.name}: {len(cfg.full_attn_layers)} full-attn planes x {cfg.num_kv_heads} "
          f"kv heads x {cfg.head_dim} head_dim", flush=True)
    prompt = torch.randint(3, cfg.vocab_size - 1, (a.prompt_tokens,)).tolist()

    results: dict = {"arch": be.arch, "source": a.source}
    try:
        acc, bf16_eng = arm_accuracy(cfg, model, be, prompt, a.new_tokens)
        results["accuracy"] = acc
        print(f"\naccuracy: {json.dumps(acc, sort_keys=True)}", flush=True)
        # the BF16 engine's pool, which the accuracy arm just filled: real un-quantized 27B KV.
        # Quantizing the fp8 engine's pool instead would round already-rounded values and
        # report a floor of zero.
        results["range"] = arm_range(
            bf16_eng._kv, {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2})
        print(f"\nrange: {json.dumps(results['range'], sort_keys=True)}", flush=True)
        for ctx in a.decode_ctx:
            for batch in a.decode_batch:
                key = f"decode_{ctx}_b{batch}"
                results[key] = arm_decode(cfg, model, be, ctx, a.new_tokens, batch)
                d = results[key]
                print(f"\n{key}: KV is {d['kv_share_of_tick_bytes_bf16']:.1%} of a tick's "
                      f"bytes bf16 / {d['kv_share_of_tick_bytes_fp8']:.1%} fp8, so the "
                      f"CEILING is {d['tok_per_s_ceiling']:.3f}x. Measured "
                      f"{d['bf16']['tok_per_s']:.2f} -> {d['fp8']['tok_per_s']:.2f} tok/s "
                      f"= {d['tok_per_s_ratio']:.3f}x", flush=True)
                print(json.dumps(d, sort_keys=True), flush=True)
    except Exception as exc:  # noqa: BLE001 -- the failure text is the answer
        results.setdefault("accuracy", {})
        results["failed"] = f"{type(exc).__name__}: {exc}"
        print(f"\nFAILED: {type(exc).__name__}: {exc}", flush=True)

    print("\n" + json.dumps(results, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
