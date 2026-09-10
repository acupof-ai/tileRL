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


def _release(*engines):
    """Drop engines and give their pools back. A fitted pool is tens of GiB, so an engine
    still bound while the next one fits itself takes 2/3 of what the first one left.
    `gc.collect` because an Engine sits in reference cycles: `del` alone does not free it."""
    import gc
    del engines
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def arm_range(pool, dtypes: dict) -> dict:
    """Round-trip error on the REAL KV this pool holds, per grid and dtype."""
    from tilerl_kernels.reference import dequant_kv_fp8, quant_kv_fp8

    # Quantizing an already-fp8 pool reports a floor of zero and an amax of exactly 448 -- it
    # measures the fixture, not the format. Caught once by that 448; asserted so it cannot
    # come back as a plausible number.
    if pool.kv_fp8 is not None:
        raise ValueError("arm_range needs the BF16 pool; this one is already quantized")
    out: dict = {}
    for name, full in (("K", pool.k_pool), ("V", pool.v_pool)):
        # Only the blocks the prompt wrote hold anything, and a fitted pool is far larger
        # than the prompt: upcasting all of it asked for 47.50 GiB and OOMed. Find the live
        # blocks by a reduction in the pool's own dtype -- `.abs()` would materialize a full
        # copy, so take amax and amin -- then upcast that slice alone.
        hi = full.amax((0, 2, 3, 4))
        lo = full.amin((0, 2, 3, 4))
        wrote = ((hi != 0) | (lo != 0)).nonzero().flatten()
        if not wrote.numel():
            out[name] = {"live_token_heads": 0}
            continue
        plane = full[:, wrote].contiguous()
        x = plane.float()
        live = x.abs().amax(-1) > 0  # token-heads inside those blocks
        if not bool(live.any()):
            out[name] = {"live_token_heads": 0}
            continue
        amax = x.abs().amax(-1)
        nz = x != 0
        rng = (amax[live] / x.abs().clamp_min(1e30).amin(-1)[live].clamp_min(1e-30))
        out[name] = {
            "live_token_heads": int(live.sum()),
            "live_blocks": int(wrote.numel()), "pool_blocks": int(full.shape[1]),
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


def _gen(cfg, model, backend, prompts, n_new, kv_fp8, step_ms: list | None = None):
    """Run `prompts` (one list, or a list of lists) to n_new tokens each.

    Returns (tokens, engine): the token LIST for a single prompt, the total COUNT for a
    batch -- the decode arm wants throughput, the accuracy arm wants the ids.

    `step_ms` collects per-tick wall clock. A whole-generation rate is NOT a decode rate:
    at B=8 ctx=8k the 65536 prefill tokens chunk into ~128 ticks against 24 decode ticks,
    so the total is 84% prefill and a decode-tick byte model does not bound it.
    """
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import BLOCK_TOKENS

    batch = prompts if isinstance(prompts[0], list) else [prompts]
    longest = max(len(p) for p in batch)
    ctx = longest + n_new + 64
    # num_blocks=0 asks `_fit_blocks` to MEASURE free memory; build_engine's default is 64
    # blocks = 1024 tokens, so leaving it out caps every arm at a pool no 2048-token prompt
    # can enter. But an UNCAPPED fit takes 2/3 of a nearly-empty card -- ~54 GiB -- and the
    # prefill transients at B=8 then have nowhere to go (OOM at 94.27 GiB in use). `max_blocks`
    # caps the fit at what this batch can actually address, the same thing cli.py:132 does.
    # The boundary arm deliberately does NOT cap: the size of the fit is its measurement.
    eng = build_engine(cfg, model, backend, num_slots=max(2, len(batch)),
                       max_batch=max(2, len(batch)), num_blocks=0,
                       max_blocks=-(-ctx // BLOCK_TOKENS) * len(batch) + 8,
                       max_total_tokens=ctx, kv_fp8=kv_fp8)
    rids = [eng.submit(p, SamplingParams(temperature=0.0, max_new_tokens=n_new, seed=0))
            for p in batch]
    # poll() returns {request_id: token_ids} and DRAINS, so accumulate rather than re-read
    done: dict = {}
    t0 = time.perf_counter()
    for _ in range(4 * n_new + 512):
        done.update(eng.poll())
        if all(len(done.get(r, ())) >= n_new for r in rids) or time.perf_counter() - t0 > 1800:
            break
        # A tick that carries ANY prefill row is a prefill tick, and it moves up to
        # max_num_batched_tokens per row where a decode tick moves one. The engine already
        # knows -- `_Req.prefilling` (engine.py:235) -- so read that rather than infer it
        # from finished ids, which only change on FINISH and would call every tick prefill.
        ts = time.perf_counter()
        pre = bool(eng._waiting) or any(r.prefilling for r in eng._running)
        eng.step()
        if step_ms is not None:
            step_ms.append((pre, (time.perf_counter() - ts) * 1e3))
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
    # Read the fp8 pool's byte rate, then let that engine go: the range arm runs next against
    # `ref_eng`'s pool, and two fitted pools alive at once is what OOMed the first card run
    # (the second fit takes 2/3 of what the first one left).
    bpt_fp8 = fp8_eng._kv.bytes_per_token
    _release(fp8_eng)
    return {
        "prompt_tokens": len(prompt), "new_tokens": len(want),
        "tokens_agreeing": agree,
        "agreement": agree / max(1, len(want)),
        "first_divergence": next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), None),
        "bf16_tokens": want[:16], "fp8_tokens": got[:16],
        "bytes_per_token_bf16": ref_eng._kv.bytes_per_token,
        "bytes_per_token_fp8": bpt_fp8,
        "kv_bytes_saved_ratio": ref_eng._kv.bytes_per_token / bpt_fp8,
    }, ref_eng


def _served_weight_bytes(cfg, model, checkpoint: str | None) -> int:
    """Served weight bytes across every device face (the ceiling's W). From the
    checkpoint headers via model.checkpoint_weight_faces (the plan weights row);
    falls back to summing the loaded params' storage when no path is given."""
    if checkpoint:
        from tilerl.model import checkpoint_weight_faces
        from tilerl.precision import nbytes
        return sum(nbytes(fmt, shape)
                   for shape, fmt in checkpoint_weight_faces(cfg, checkpoint).values())
    return sum(t.numel() * t.element_size() for t in model.params.values())


def arm_decode(cfg, model, backend, ctx: int, n_new: int, batch: int = 1,
               checkpoint: str | None = None) -> dict:
    """Decode tok/s at one (context, batch), fp8 pool against bf16, same engine and prompts.

    Reports the KV share of a tick's bytes and the resulting CEILING before the measured
    ratio. A decode tick re-reads the 27B's weights every token regardless, so fp8 KV can
    only act on the KV part -- and at B=1 that part is small: 2.1% of the tick at 8k, 8.1%
    at 32k, so the ceilings are 1.011x and 1.041x, under run-to-run variance. KV scales with
    batch while the weights do not, which is where the win is: 41% of the tick and a 1.255x
    ceiling at B=8 ctx=32k, 74% and 1.570x at B=32 ctx=32k. A ratio quoted without its
    ceiling reads as though fp8 moved the whole tick.

    Every figure here is against the MEASURED weight footprint, 24436981888 bytes = 22.76
    GiB resident. The NVFP4 checkpoint is ~12.6 GiB on disk and my first ceilings used that,
    which overstated all of them (1.019x/1.072x/1.38x/1.70x for the cells above).
    """
    prompts = [torch.randint(3, cfg.vocab_size - 1, (ctx,)).tolist() for _ in range(batch)]
    out: dict = {"ctx": ctx, "batch": batch, "new_tokens": n_new}
    for nick, dt in (("bf16", None), ("fp8", torch.float8_e4m3fn)):
        toks, eng = _gen(cfg, model, backend, prompts, n_new, dt)
        # time a SECOND generation on a fresh engine of the same kind: the first paid any
        # first-call compile, which is not what a decode rate is. Release the warm-up engine
        # first -- its fitted pool is tens of GiB, and holding it means the timed engine fits
        # itself into 2/3 of the remainder, so it would be timed at a different pool size.
        bpt = eng._kv.bytes_per_token
        _release(eng)
        ticks: list = []
        t0 = time.perf_counter()
        toks2, eng2 = _gen(cfg, model, backend, prompts, n_new, dt, step_ms=ticks)
        dt_s = time.perf_counter() - t0
        dec = [ms for pre, ms in ticks if not pre]
        pre_ms = [ms for pre, ms in ticks if pre]
        out[nick] = {
            "tokens": toks2, "seconds": dt_s,
            # whole-generation, kept only to show how little of it is decode
            "tok_per_s_whole_run": toks2 / dt_s if dt_s > 0 else 0.0,
            # the number the byte ceiling bounds: one token per row per decode tick
            "decode_ticks": len(dec), "prefill_ticks": len(pre_ms),
            "decode_ms_per_tick": sum(dec) / len(dec) if dec else 0.0,
            "decode_tok_per_s": (batch * 1e3 / (sum(dec) / len(dec))) if dec else 0.0,
            "prefill_seconds": sum(pre_ms) / 1e3,
            "kv_bytes_per_token": bpt,
            "kv_bytes_at_ctx": bpt * ctx * batch,
            "blocks_total": eng2.usable_blocks,
        }
        _release(eng2)
    weight_bytes = _served_weight_bytes(cfg, model, checkpoint)
    kb, kf = out["bf16"]["kv_bytes_at_ctx"], out["fp8"]["kv_bytes_at_ctx"]
    out["weight_bytes"] = weight_bytes
    out["kv_share_of_tick_bytes_bf16"] = kb / (weight_bytes + kb)
    out["kv_share_of_tick_bytes_fp8"] = kf / (weight_bytes + kf)
    # the most a pure-bandwidth tick could gain: what the measured ratio must sit under
    out["tok_per_s_ceiling"] = (weight_bytes + kb) / (weight_bytes + kf)
    # Against the DECODE rate, which is what the ceiling models. The whole-run ratio is
    # kept beside it because at B=8 ctx=8k the run is ~84% prefill ticks, so the two
    # differ and only one of them is bounded by a decode tick's byte traffic.
    out["decode_ratio"] = (
        out["fp8"]["decode_tok_per_s"] / out["bf16"]["decode_tok_per_s"]
        if out["bf16"]["decode_tok_per_s"] else 0.0)
    out["whole_run_ratio"] = (
        out["fp8"]["tok_per_s_whole_run"] / out["bf16"]["tok_per_s_whole_run"]
        if out["bf16"]["tok_per_s_whole_run"] else 0.0)
    return out


def arm_boundary(cfg, model, backend, ctx: int, batch: int, n_new: int) -> dict:
    """How many of a batch are RESIDENT at once, bf16 pool against fp8. The capacity claim.

    Not "bf16 raises and fp8 does not". It was written that way and that was wrong: `_admit`
    returns False when the pool is short, it does not raise, so an over-large batch is
    admitted as far as it fits and the rest WAITS. bf16 at B=32 x 32k does not OOM -- it
    serializes. `submit`'s own refusal cannot fire either, since one 32k request needs 2050
    blocks against a fitted pool's tens of thousands.

    So the number is concurrency: peak `running` over the generation. 1.969x the tokens per
    byte means fp8 should hold about 1.97x as many of the same requests resident, and the
    queue drains in correspondingly fewer passes. That is measurable, and it is the thing
    capacity actually buys.
    """
    from tilerl.engine import SamplingParams, build_engine
    prompts = [torch.randint(3, cfg.vocab_size - 1, (ctx,)).tolist() for _ in range(batch)]
    out: dict = {"ctx": ctx, "batch": batch}
    for nick, dt in (("bf16", None), ("fp8", torch.float8_e4m3fn)):
        eng, blocks = None, 0
        try:
            eng = build_engine(cfg, model, backend, num_slots=batch, max_batch=batch,
                               num_blocks=0, max_total_tokens=ctx + n_new + 64, kv_fp8=dt)
            blocks = eng.usable_blocks  # read BEFORE generating: the fit is the measurement
            rids = [eng.submit(p, SamplingParams(temperature=0.0, max_new_tokens=n_new, seed=0))
                    for p in prompts]
            done: dict = {}
            peak, t0 = 0, time.perf_counter()
            for _ in range(64 * n_new + 4096):
                done.update(eng.poll())
                peak = max(peak, eng.stats()["running"])
                if len(done) >= len(rids) or time.perf_counter() - t0 > 1800:
                    break
                eng.step()
            out[nick] = {
                "blocks_total": eng.usable_blocks, "peak_running": peak,
                "finished": len(done), "seconds": time.perf_counter() - t0,
                "kv_bytes_per_token": eng._kv.bytes_per_token,
            }
        except Exception as exc:  # noqa: BLE001 -- a raise here is a result, not a crash
            # The fitted block count is the headline, and it is known before the generation.
            # Keep it: an uncapped fit leaves little room for 32 prefills at once, so this
            # arm can OOM in the transients with the capacity answer already measured.
            out[nick] = {"raised": f"{type(exc).__name__}: {str(exc)[:200]}"}
            if blocks:
                out[nick]["blocks_total"] = blocks
        # `empty_cache` alone frees nothing while `eng` is still bound, and the bf16 pool
        # here is the largest thing either arm allocates.
        _release(eng)
        eng = None
    both = [out[k] for k in ("bf16", "fp8")]
    # blocks_ratio FIRST and on its own guard: it is the headline, it is known before either
    # generation runs, and an arm that OOMs in the prefill transients still has it. Nesting it
    # under peak_running lost the capacity answer in exactly the case that produces it.
    if all("blocks_total" in d for d in both):
        out["blocks_ratio"] = both[1]["blocks_total"] / max(1, both[0]["blocks_total"])
        # The arm separates the dtypes only if the pools were sized differently. Off CUDA
        # `_fit_blocks` returns a fixed floor for both, so equal block counts mean the fit
        # never ran -- report that rather than a ratio of 1.0 that looks like a null result.
        if both[0]["blocks_total"] == both[1]["blocks_total"]:
            out["inconclusive"] = (
                f"both pools got {both[0]['blocks_total']} blocks, so _fit_blocks did not "
                "measure free memory (it returns a floor off CUDA) -- this arm needs a card")
    if all("peak_running" in d for d in both):
        # Bounded by the batch: if fp8 holds all of it, this reads however far bf16 fell
        # short rather than how much more fp8 could have held. blocks_ratio has no ceiling.
        out["resident_ratio"] = both[1]["peak_running"] / max(1, both[0]["peak_running"])
        if both[0]["peak_running"] >= batch:
            out["inconclusive"] = (
                f"bf16 already held all {batch} requests resident, so the batch cannot show a "
                "concurrency difference -- raise --boundary-batch or --boundary-ctx")
        elif both[1]["peak_running"] >= batch:
            out["resident_ratio_is_clipped"] = (
                f"fp8 held the whole batch ({batch}), so resident_ratio understates it -- "
                f"blocks_ratio {out['blocks_ratio']:.3f}x is the capacity number")
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
    ap.add_argument("--boundary-ctx", type=int, default=32768,
                    help="context for the capacity-boundary arm")
    ap.add_argument("--boundary-batch", type=int, default=32,
                    help="batch large enough to saturate the bf16 pool, so fp8's extra "
                         "blocks show as extra resident requests; 0 skips the arm")
    ap.add_argument("--decode-batch", type=int, nargs="*", default=[1, 8],
                    help="batch sizes. KV scales with batch and the weights do not, so B=1 "
                         "has a 1.011x/1.041x ceiling at 8k/32k while B=8 at 32k has 1.255x "
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
        # The range arm is the last reader of this pool, and the decode arm below fits its
        # own against whatever is left. Held, this one was still resident at B=8 and the
        # card reached 94.27 GiB.
        _release(bf16_eng)
        bf16_eng = None
        for ctx in a.decode_ctx:
            for batch in a.decode_batch:
                key = f"decode_{ctx}_b{batch}"
                results[key] = arm_decode(cfg, model, be, ctx, a.new_tokens, batch,
                                          checkpoint=a.source)
                d = results[key]
                print(f"\n{key}: KV is {d['kv_share_of_tick_bytes_bf16']:.1%} of a DECODE "
                      f"tick's bytes bf16 / {d['kv_share_of_tick_bytes_fp8']:.1%} fp8, so the "
                      f"CEILING is {d['tok_per_s_ceiling']:.3f}x.\n"
                      f"  decode ticks: {d['bf16']['decode_ms_per_tick']:.1f} -> "
                      f"{d['fp8']['decode_ms_per_tick']:.1f} ms, "
                      f"{d['bf16']['decode_tok_per_s']:.2f} -> "
                      f"{d['fp8']['decode_tok_per_s']:.2f} tok/s = "
                      f"{d['decode_ratio']:.3f}x  <- the ceiling bounds THIS\n"
                      f"  whole run ({d['bf16']['prefill_ticks']} prefill + "
                      f"{d['bf16']['decode_ticks']} decode ticks): "
                      f"{d['whole_run_ratio']:.3f}x, not bounded by it", flush=True)
                print(json.dumps(d, sort_keys=True), flush=True)
        if a.boundary_batch:
            key = f"boundary_{a.boundary_ctx}_b{a.boundary_batch}"
            results[key] = arm_boundary(cfg, model, be, a.boundary_ctx, a.boundary_batch,
                                        a.new_tokens)
            d = results[key]
            print(f"\n{key}: blocks {d['bf16'].get('blocks_total')} -> "
                  f"{d['fp8'].get('blocks_total')}"
                  + (f" = {d['blocks_ratio']:.3f}x" if "blocks_ratio" in d else "")
                  + f", peak resident {d['bf16'].get('peak_running')} -> "
                  f"{d['fp8'].get('peak_running')} of {a.boundary_batch}"
                  + (f" = {d['resident_ratio']:.3f}x" if "resident_ratio" in d else "")
                  + (f"\n  INCONCLUSIVE: {d['inconclusive']}" if "inconclusive" in d else "")
                  + (f"\n  bf16 raised: {d['bf16']['raised']}" if "raised" in d["bf16"] else "")
                  + (f"\n  fp8 raised: {d['fp8']['raised']}" if "raised" in d["fp8"] else ""),
                  flush=True)
            print(json.dumps(d, sort_keys=True), flush=True)
    except Exception as exc:  # noqa: BLE001 -- the failure text is the answer
        results.setdefault("accuracy", {})
        results["failed"] = f"{type(exc).__name__}: {exc}"
        print(f"\nFAILED: {type(exc).__name__}: {exc}", flush=True)

    print("\n" + json.dumps(results, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
