"""tilerl command-line interface: serve / train / pretrain / bench.

Heavy imports (torch, tilelang, sibling modules) happen inside the subcommand
handlers so that ``tilerl --help`` stays instant and works even before the
full runtime is wired up.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

__all__ = ["main"]

#: HF source for the 27B checkpoint (hub id or local directory). Override
#: with TILERL_QWEN38_SOURCE once the checkpoint is downloaded.
# ponytail: placeholder hub id; pin the real Qwen3-27B repo when weights land.
_QWEN38_SOURCE = os.environ.get("TILERL_QWEN38_SOURCE", "Qwen/Qwen3-27B")


def _build_model(
    model_name: str, seed: int, fuse_projections: bool = False, keep_master: bool = False
):
    """Build (cfg, model) for a named model. Lazy imports keep --help light.

    ``fuse_projections`` concats same-input fp4 projections into one GEMV at
    load (serving decode is launch-latency-bound on the small projections);
    training keeps it off and passes ``keep_master`` so the tape has masters.
    """
    from . import config as config_mod
    from . import model as model_mod

    if model_name == "qwen38-27b":
        cfg = config_mod.qwen38_27b()
        try:
            model = model_mod.load_hf(
                cfg, _QWEN38_SOURCE, fuse_projections=fuse_projections, keep_master=keep_master
            )
        except Exception as exc:
            print(
                f"error: could not load Qwen3-27B weights from {_QWEN38_SOURCE!r}: {exc}\n"
                "hint: download the checkpoint (or set TILERL_QWEN38_SOURCE to a\n"
                "      local safetensors directory), or use --model tiny.",
                file=sys.stderr,
            )
            sys.exit(1)
        return cfg, model
    cfg = config_mod.tiny()
    return cfg, model_mod.build_random(
        cfg, seed=seed, fuse_projections=fuse_projections, keep_master=keep_master
    )


def _build_engine(cfg, model, backend):
    """Wire the engine with the serving-size pools (256 blocks / 16 slots)."""
    from . import engine as engine_mod

    return engine_mod.build_engine(
        cfg,
        model,
        backend,
        num_blocks=256,
        num_slots=16,
        max_batch=8,
        max_total_tokens=8192,
    )


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .ops.backend import get_backend
    from .server import create_app, get_tokenizer

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=0, fuse_projections=True)
    engine = _build_engine(cfg, model, backend)
    tokenizer = get_tokenizer(_QWEN38_SOURCE if args.model == "qwen38-27b" else None)

    app = create_app(engine, tokenizer, model_name=cfg.name)
    engine.run()  # starts the engine's own daemon thread
    print(f"tilerl serve: model={cfg.name} target={backend.target}")
    print(f"tilerl serve: http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    import torch

    from . import train as train_mod
    from .autograd import AdamW, Tape, cosine_warmup
    from .ops.backend import get_backend

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=args.seed, keep_master=True)
    optimizer = AdamW(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    gen = torch.Generator().manual_seed(args.seed)

    print(
        f"tilerl train: model={cfg.name} layers={cfg.num_layers} "
        f"hidden={cfg.hidden_size} vocab={cfg.vocab_size} steps={args.steps}"
    )
    for step in range(args.steps):
        # ponytail: fixed batch=2 seq=64 random-token batch; a real corpus
        # plugs in here without touching train_step.
        input_ids = torch.randint(0, cfg.vocab_size, (2, 64), generator=gen)
        optimizer.lr = cosine_warmup(step, args.steps, 5, 1e-3)
        # Fresh tape per step: one backward per tape (a reused tape leaks the
        # step's intermediates and replays all history on each backward).
        loss = train_mod.train_step(model, input_ids, backend, optimizer, Tape())
        print(f"step {step + 1:4d}/{args.steps}  loss {loss:.4f}")


# ---------------------------------------------------------------------------
# pretrain
# ---------------------------------------------------------------------------


def cmd_pretrain(args: argparse.Namespace) -> None:
    from . import train as train_mod
    from .autograd import AdamW
    from .ops.backend import get_backend
    from .server import get_tokenizer

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=args.seed, keep_master=True)
    dataset = train_mod.JsonlDataset(args.data, get_tokenizer(None), args.seq_len)
    optimizer = AdamW(lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    print(
        f"tilerl pretrain: model={cfg.name} data={args.data} "
        f"seq_len={args.seq_len} steps={args.steps}"
    )
    train_mod.pretrain(
        model,
        dataset,
        backend,
        optimizer,
        args.steps,
        lr=args.lr,
        warmup=args.warmup,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=args.ckpt_every,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def cmd_bench(args: argparse.Namespace) -> None:
    if getattr(args, "suite", None):
        # Full harness: decode-vs-KV-depth / prefill-curve / kv-reuse / train,
        # with the snapshot baseline gate. Lives in scripts/ (needs sys.path
        # script-dir access); shell into it so it stays the single source.
        import subprocess
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent.parent / "scripts/bench_harness.py"
        cmd = [sys.executable, str(script), "--suite", args.suite]
        if args.source:
            cmd += ["--source", args.source]
        if args.gpu is not None:
            cmd += ["--gpu", str(args.gpu)]
        if args.batches:
            cmd += ["--batches", args.batches]
        sys.exit(subprocess.call(cmd))

    import torch

    from . import engine as engine_mod
    from .ops.backend import get_backend

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=0)
    engine = _build_engine(cfg, model, backend)
    gen = torch.Generator().manual_seed(0)

    def rand_ids(n: int) -> list[int]:
        return torch.randint(0, cfg.vocab_size, (n,), generator=gen).tolist()

    def run_to_done(req_id: int, max_ticks: int) -> None:
        for _ in range(max_ticks):
            engine.step()
            if req_id in engine.poll():
                return

    # Warmup: first tilelang calls JIT-compile kernels; never time that. Use
    # the same prompt_len as the timed run — JIT specializes per shape, so a
    # shorter warmup prompt leaves the timed prefill's NVCC compile in the
    # measurement. gen=2 compiles the decode [1,1] shapes too.
    warmup_id = engine.submit(
        rand_ids(args.prompt_len),
        engine_mod.SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=2, seed=0),
    )
    run_to_done(warmup_id, max_ticks=16)

    # Timed prefill: one tick = one forward over the whole prompt.
    req_id = engine.submit(
        rand_ids(args.prompt_len),
        engine_mod.SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=args.gen, seed=0),
    )
    t0 = time.perf_counter()
    engine.step()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    # Timed decode: one token per tick.
    decode_ms: list[float] = []
    for _ in range(args.gen):
        t0 = time.perf_counter()
        engine.step()
        decode_ms.append((time.perf_counter() - t0) * 1000.0)
    assert req_id in engine.poll(), "bench request did not complete"

    decode_avg = sum(decode_ms) / len(decode_ms)
    print(
        f"tilerl bench: model={cfg.name} target={backend.target} "
        f"prompt_len={args.prompt_len} gen={args.gen}"
    )
    print(f"{'phase':<10} {'ms/tok':>12} {'tok/s':>12}")
    print(
        f"{'prefill':<10} {prefill_ms / args.prompt_len:>12.3f} "
        f"{1000.0 * args.prompt_len / prefill_ms:>12.1f}"
    )
    print(f"{'decode':<10} {decode_avg:>12.3f} {1000.0 / decode_avg:>12.1f}")
    print(f"prefill total: {prefill_ms:.1f} ms | decode total: {sum(decode_ms):.1f} ms")


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tilerl",
        description="tileRL: TileLang inference + training (CPU/CUDA/ROCm/Metal).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="start the OpenAI-compatible HTTP server")
    p_serve.add_argument("--model", choices=["tiny", "qwen38-27b"], default="tiny")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_train = sub.add_parser("train", help="train a model on random-token batches")
    p_train.add_argument("--model", choices=["tiny"], default="tiny")
    p_train.add_argument("--steps", type=int, default=20)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.set_defaults(func=cmd_train)

    p_pretrain = sub.add_parser("pretrain", help="pretrain on a JSONL text corpus")
    p_pretrain.add_argument("--model", choices=["tiny"], default="tiny")
    p_pretrain.add_argument("--data", required=True, help="JSONL file with 'text' fields")
    p_pretrain.add_argument("--steps", type=int, default=20)
    p_pretrain.add_argument("--seq-len", type=int, default=512)
    p_pretrain.add_argument("--ckpt-dir", default=None)
    p_pretrain.add_argument("--ckpt-every", type=int, default=0)
    p_pretrain.add_argument("--lr", type=float, default=1e-3)
    p_pretrain.add_argument("--warmup", type=int, default=0)
    p_pretrain.add_argument("--seed", type=int, default=0)
    p_pretrain.set_defaults(func=cmd_pretrain)

    p_bench = sub.add_parser("bench", help="benchmark prefill/decode throughput")
    p_bench.add_argument("--model", choices=["tiny"], default="tiny")
    p_bench.add_argument("--prompt-len", type=int, default=128)
    p_bench.add_argument("--gen", type=int, default=32)
    p_bench.add_argument(
        "--suite",
        default=None,
        help="run the full harness (scripts/bench_harness.py) instead of the quick "
        "tiny timer: comma list of decode-kv,prefill,kv-reuse,train,micro",
    )
    p_bench.add_argument("--source", default=None, help="27B checkpoint dir (harness GPU suites)")
    p_bench.add_argument("--gpu", type=int, default=None, help="GPU index (harness)")
    p_bench.add_argument("--batches", default=None, help="harness decode batch sizes, e.g. 1,8")
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
