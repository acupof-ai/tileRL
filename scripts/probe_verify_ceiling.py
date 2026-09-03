"""The ceiling on batching the drafter: what the trunk alone costs at DFlash2's
own acceptance, with no drafter in the process.

#34 measures DFlash2 1.67x slower end to end with the drafter at 68.4% of the
arm. The tranche that follows proposes batching the drafter, and its payoff is
bounded by what remains once the drafter costs nothing. That bound is
measurable and this measures it, in two phases so the drafter is absent from
the timed one.

**record** runs the real speculative arm and logs, per verify tick, the chain
width and how many drafts the trunk accepted. Acceptance decides how many
ticks a completion needs, so it is the schedule, not a scalar to average: a
fixed all-accept chain prices a configuration that does not exist.

**price** builds an engine with **no draft head at all**, replays the captured
decode graph at each width the trace used, and weights the measured per-tick
cost by the trace's own tick counts:

    ceiling = (tokens x t[W=1]) / sum over ticks of t[W(tick)]

Numerator: the unspeculated schedule for the same token count. Denominator:
the speculative schedule's trunk cost at the acceptance it actually achieved.
Nothing of the drafter is in either, so the ratio is the most a free drafter
could buy. A batched drafter is not free, so the real number is below this.

    # phase 1, needs a drafter (DFlash2 lives on #34)
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_verify_ceiling.py record \
        --source /work/Qwen3.8-27B-NVFP4 \
        --draft /work/Qwen3.8-27B-DFlash2/model.safetensors \
        --gsm8k /work/gsm8k_test.jsonl --gsm8k-n 32 --batch 8 \
        --out /work/ceiling/trace.json

    # phase 2, no drafter in the process
    CUDA_VISIBLE_DEVICES=7 ... python3 scripts/probe_verify_ceiling.py price \
        --source /work/Qwen3.8-27B-NVFP4 --trace /work/ceiling/trace.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from tilerl_kernels.backend import get_backend

from tilerl import precision
from tilerl.config import qwen38_27b
from tilerl.engine import Engine, SamplingParams, StepLimits, _DecodeGraph, build_engine
from tilerl.kv_cache import LinearStatePool, NoPrefixStore, PagedKvPool
from tilerl.model import load_hf


def host_load() -> str:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return "/".join(smi.stdout.split()) + "%"


def timed(fn, reps: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


# ----------------------------------------------------------------- record


def record(args) -> None:
    """Log (width, n_ok) per verified row. No timing here -- the tracer syncs."""
    from tilerl.eval import answer_match, generate
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.prompt import render_chat, sampling
    from tilerl.spec import load_draft
    from tilerl.tokenizer import get_tokenizer

    backend = get_backend()
    cfg = qwen38_27b()
    tok = get_tokenizer(args.source)
    model = load_hf(cfg, args.source)
    draft = load_draft(model, args.draft)
    # NoPrefixStore unconditionally: a drafter tapping the trunk's aux layers
    # builds its context only from positions this process forwarded, so an
    # adopted prefix would leave it attending over recycled blocks. Engine
    # raises on the combination; the graph is orthogonal to it.
    engine = build_engine(cfg, model, backend, num_blocks=1024,
                          num_slots=args.batch + 2, draft=draft,
                          spec_depth=max(1, args.width - 1),
                          decode_graph=args.graph,
                          prefix_store=NoPrefixStore())
    ticks: list[list[int]] = []  # one [width, n_ok, committed] per verified row
    # n_ok comes off select_step, which _verify calls once per row with exactly
    # the accepted count. Deriving it from len(r.output) instead undercounts: a
    # tick whose chain hits a stop token commits fewer tokens than it accepted
    # (_commit returns early on stop), and a tick whose FIRST token is a stop
    # commits none, which reads as -1 accepted.
    verify, select = engine._verify, engine._states.select_step
    pending: list[int] = []

    def w_select(slot, step):
        pending.append(int(step))
        return select(slot, step)

    def w_verify(rows, chains, logits, hidden):
        before = [len(r.output) for r in rows]
        pending.clear()
        out = verify(rows, chains, logits, hidden)
        assert len(pending) == len(rows), (len(pending), len(rows))
        for i, (r, n0) in enumerate(zip(rows, before)):
            ticks.append([len(chains[i]), pending[i], len(r.output) - n0])
        return out

    engine._verify = w_verify
    engine._states.select_step = w_select
    rows = [json.loads(ln) for ln in Path(args.gsm8k).read_text().splitlines()
            if ln.strip()][: args.gsm8k_n]
    params = replace(sampling(tok, False, args.max_new_tokens, temperature=0.0,
                              max_think_tokens=0, seed=0), temperature=0.0)
    prompts = [render_chat([("user", r["prompt"])], False) for r in rows]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    texts = generate(engine, tok, prompts, params, args.batch)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    s = engine.stats()
    ok = sum(answer_match(t, r["answer"]) for t, r in zip(texts, rows))

    widths = Counter(w for w, _, _ in ticks)
    acc = Counter(n for _, n, _ in ticks)
    tokens = sum(c for _, _, c in ticks)  # what the run actually committed
    short = [t for t in ticks if t[2] != t[1] + 1]
    print(f"gsm8k {ok}/{len(rows)}  {secs:.1f}s  drafted {s['spec_drafted']} "
          f"accepted {s['spec_accepted']}  decode fwd {s['decode_forwards']}")
    print(f"verified row-ticks {len(ticks)}, tokens committed through verify {tokens}")
    print(f"  width histogram   {dict(sorted(widths.items()))}")
    print(f"  accepted per tick {dict(sorted(acc.items()))}  "
          f"mean {sum(acc.elements()) / max(1, len(ticks)):.3f}")
    print(f"  n_ok+1 == committed on {len(ticks) - len(short)}/{len(ticks)}; "
          f"{len(short)} truncated by a stop token")
    assert sum(n for _, n, _ in ticks) == s["spec_accepted"], (
        f"traced accepted {sum(n for _, n, _ in ticks)} != engine "
        f"{s['spec_accepted']}: the trace is not the run")
    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps({
        "ticks": ticks, "gsm8k": [ok, len(rows)], "secs": secs, "stats": dict(s),
        "batch": args.batch, "width": args.width, "graph": args.graph,
        "gsm8k_n": args.gsm8k_n, "max_new_tokens": args.max_new_tokens,
        "source": args.source, "draft": args.draft,
    }, indent=1))
    print(f"wrote {o}")


# ------------------------------------------------------------------ price


def price(args) -> None:
    """Cost each width the trace used, on an engine holding no draft head."""
    tr = json.loads(Path(args.trace).read_text())
    ticks = [(int(w), int(n), int(c)) for w, n, c in tr["ticks"]]
    if not ticks:
        raise SystemExit(f"{args.trace}: no verified ticks recorded")
    B = args.batch or int(tr["batch"])
    # committed, not n_ok+1: a chain cut short by a stop token commits fewer, and
    # the unspeculated arm would have paid one tick per token it actually kept
    tokens = sum(c for _, _, c in ticks)
    per_width = Counter(w for w, _, _ in ticks)
    # a row-tick is one row; the engine ticks B rows at once
    tick_rows = sorted(per_width)
    print(f"trace {args.trace}: {len(ticks)} row-ticks, {tokens} tokens, "
          f"widths {dict(sorted(per_width.items()))}, batch {B}, graph {tr['graph']}")

    backend = get_backend()
    base = qwen38_27b()
    cfg = replace(base, num_layers=args.layers,
                  full_attn_layers=tuple(i for i in base.full_attn_layers if i < args.layers))
    model = load_hf(cfg, args.source)
    widths = sorted({1, *tick_rows})
    # The pools are built here rather than through build_engine because the two
    # requirements collide there: keep=W reads the step planes, and build_engine
    # sizes them from ``draft.width`` -- so no drafter means no planes and the
    # kernel takes None. Engine takes the pools directly, so the planes exist and
    # nothing drafts.
    kv_pool = PagedKvPool(1024, cfg.num_kv_heads, cfg.head_dim,
                          device=backend.device, layer_map=cfg.full_attn_layers)
    state_pool = LinearStatePool(
        B + 2, cfg.num_layers - len(cfg.full_attn_layers),
        cfg.linear_num_value_heads, cfg.linear_value_head_dim,
        device=backend.device,
        dtype=precision.dtype("recurrent_state", backend.device),
        conv_window=cfg.linear_conv_kernel_dim - 1, conv_dim=cfg.linear_qkv_dim,
        spec_steps=max(widths),  # the planes keep=W needs, with no drafter to fill them
    )
    model.params = backend.materialize(model.params)
    engine = Engine(model, backend, kv_pool, state_pool, NoPrefixStore(),
                    StepLimits(max_batch=B, max_total_tokens=8192),
                    decode_graph=True)
    assert engine._draft is None, "the timed phase must hold no draft head"
    assert state_pool.step_states is not None, "keep=W needs the step planes"

    gen = torch.Generator().manual_seed(7)
    for _ in range(B):
        engine.submit(torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist(),
                      SamplingParams(temperature=0.0, max_new_tokens=4096))
    for _ in range(8):
        engine.step()
    rows = list(engine._running)[:B]
    assert len(rows) == B, f"only {len(rows)} rows running, need {B}"

    gpool = torch.cuda.graph_pool_handle()
    ms: dict[int, float] = {}
    print(f"\n{'W':>3} {'replay ms':>10} {'vs W=1':>8}  host")
    for W in widths:
        chains = [[r.output[-1]] * W for r in rows] if W > 1 else None
        # keep=W is what a real verify tick sets; without a draft head the
        # engine never sets it, so the graph is built with it explicitly
        g = _DecodeGraph(model, backend, engine._kv, engine._states, B,
                         width=W, pool=gpool, keep=W if chains else 0)
        ms[W] = timed(lambda: g.run(rows, chains), reps=args.reps)
        print(f"{W:>3} {ms[W]:>10.3f} {ms[W] / ms[1]:>7.2f}x  {host_load()}")

    # Each engine tick advances B rows together, so B row-ticks of one width are
    # one engine tick. Rows in a tick share a width (chains are padded uniform).
    engine_ticks = {w: c / B for w, c in per_width.items()}
    spec_ms = sum(engine_ticks[w] * ms[w] for w in engine_ticks)
    base_ticks = tokens / B  # the unspeculated arm: one token per row per tick
    base_ms = base_ticks * ms[1]
    print("\n=== the ceiling: trunk-only cost of the two schedules ===")
    print(f"  tokens committed          {tokens}")
    print(f"  spec engine ticks         {sum(engine_ticks.values()):.1f} "
          f"({', '.join(f'{v:.1f}x W={k}' for k, v in sorted(engine_ticks.items()))})")
    print(f"  base engine ticks (W=1)   {base_ticks:.1f}")
    print(f"  spec trunk time           {spec_ms / 1e3:.2f} s")
    print(f"  base trunk time           {base_ms / 1e3:.2f} s")
    print(f"  forward reduction         {base_ticks / sum(engine_ticks.values()):.2f}x")
    print(f"  CEILING (free drafter)    {base_ms / spec_ms:.2f}x")
    print("\nA batched drafter is not free, so the achievable speedup is strictly")
    print("below the ceiling. Acceptance was measured with a per-row drafter and")
    print("moves if batching restructures path()'s walk, which shifts this number.")
    out = {"ms_per_width": ms, "tokens": tokens, "spec_ms": spec_ms,
           "base_ms": base_ms, "ceiling": base_ms / spec_ms,
           "engine_ticks": engine_ticks, "base_ticks": base_ticks,
           "reps": args.reps, "batch": B, "layers": args.layers}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("record", help="run the real spec arm, log (width, n_ok) per tick")
    r.add_argument("--source", required=True)
    r.add_argument("--draft", required=True)
    r.add_argument("--gsm8k", required=True)
    r.add_argument("--gsm8k-n", type=int, default=32)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--width", type=int, default=8)
    # 512, not the recipe's 256: at 256 GSM8K completions are cut mid-solution
    # (1 of 16 correct vs 170 of 200 at 512), so a 256 trace measures truncation
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--graph", action="store_true", default=True)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=record)

    p = sub.add_parser("price", help="cost the trace's widths with no drafter loaded")
    p.add_argument("--source", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--batch", type=int, default=0, help="override the trace's batch")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out")
    p.set_defaults(fn=price)

    args = ap.parse_args()
    backend = get_backend()
    assert backend.device.type == "cuda", backend.device
    args.fn(args)


if __name__ == "__main__":
    main()
