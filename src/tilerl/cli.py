"""tilerl command-line interface: serve / train / pretrain / bench / merge.

Heavy imports (torch, tilelang, sibling modules) happen inside the subcommand
handlers so that ``tilerl --help`` stays instant and works even before the
full runtime is wired up.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from . import recipes
from .recipes import RECIPES

from .eval import answer_match, last_number  # noqa: F401  (tests import last_number from here)

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
    # tiny-agent is tiny with room for one real agent turn; see config.tiny().
    cfg = config_mod.tiny(65536) if model_name == "tiny-agent" else config_mod.tiny()
    return cfg, model_mod.build_random(
        cfg, seed=seed, fuse_projections=fuse_projections, keep_master=keep_master
    )


def _build_engine(cfg, model, backend, devices=None):
    """Wire the engine with the serving-size pools (256 blocks / 16 slots).

    ``devices``: replicate across these CUDA indices instead of one. The 27B in
    NVFP4 is 23 GB against a 96 GB card, so a replica per device costs memory
    the cards already have and measures 7.54x on 8 (wins/
    2026-08-29-data-parallel-scales.md).
    """
    from . import engine as engine_mod

    # The token budget follows the model's context, and the block pool follows
    # the budget. A fixed 8192 refused a real Claude Code turn on tiny-agent:
    # ByteTokenizer is one token per BYTE, so a 21,676-byte request is 21,676
    # tokens, not the ~5,400 a BPE would make of it. The measurement that
    # matters is tokens, not characters.
    ctx = int(getattr(cfg, "max_position_embeddings", 8192))
    from .kv_cache import BLOCK_TOKENS

    kw = dict(num_blocks=max(256, (ctx * 8) // BLOCK_TOKENS), num_slots=16,
              max_batch=8, max_total_tokens=ctx)
    if not devices:
        return engine_mod.build_engine(cfg, model, backend, **kw)

    import torch

    from .model import load_hf
    from tilerl_kernels.backend import Backend, resolve_target
    from .parallel import DataParallelEngine

    def make(d, **kwargs):
        # A Backend binds torch.cuda.current_device() at construction and the
        # weights land on it, so each replica loads its own copy inside its
        # own context. Sharing one model across devices would silently serve
        # every replica from device 0's memory.
        b = Backend(resolve_target())
        m = load_hf(cfg, _QWEN38_SOURCE, fuse_projections=True) if model is None else model
        return engine_mod.build_engine(cfg, m, b, **kwargs)

    del torch  # only needed by DataParallelEngine.build
    return DataParallelEngine.build(devices, make, **kw)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from tilerl_kernels.backend import get_backend
    from .server import create_app, get_tokenizer

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=0, fuse_projections=True)
    engine = _build_engine(cfg, model, backend, devices=args.devices)
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
    from .autograd import Adafactor, AdamW, Tape, cosine_warmup
    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    # The bf16 masters are for full-parameter quantized training (the STE grad
    # lands on them). --rl and --opd train LoRA on a FROZEN base, whose backward
    # reads the quantized weight directly, so a master is ~27 GB the 27B never
    # touches — and enough to fill the card on its own.
    adapters_only = args.rl or args.opd
    cfg, model = _build_model(args.model, seed=args.seed, keep_master=not adapters_only)
    if not adapters_only:
        from .model import drop_quantized

        drop_quantized(model)
    # Full fine-tuning cannot use Adam: m+v for the 27B is 200.4 GiB against
    # 50.1 GiB of weights. Adafactor factors the second moment to 0.03 GiB and
    # clips each update, so no global grad norm is needed — which is what lets
    # train._step consume and free every gradient inside backward instead of
    # holding all 50.1 GiB of them.
    optimizer = (
        AdamW(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1) if adapters_only
        else Adafactor(lr=1e-2, weight_decay=0.1)
    )
    if args.optim == "iso" and not adapters_only:
        from .iso import ISO

        optimizer = ISO(optimizer)
    gen = torch.Generator().manual_seed(args.seed)
    log = (lambda *a: None) if getattr(args, "json", False) else print

    log(
        f"tilerl train: model={cfg.name} layers={cfg.num_layers} "
        f"hidden={cfg.hidden_size} vocab={cfg.vocab_size} steps={args.steps}"
    )
    if adapters_only:
        from .engine import SamplingParams
        from .eval import gsm8k_accuracy, mmlu_accuracy
        from .ledger import commit, file_hash, new_manifest, read_manifest, runs_root
        from .tokenizer import get_tokenizer, render_chat

        tok = get_tokenizer(_QWEN38_SOURCE if args.model == "qwen38-27b" else None)
        # --data: {"prompt", "answer"} per line, rendered as one ChatML user
        # turn; the reward is exact match on the last number of the completion
        # (GSM8K). Without it, random prompts and a demo reward.
        rows = ([json.loads(ln) for ln in Path(args.data).read_text().splitlines() if ln.strip()]
                if args.data else [])
        prompts = [tok.encode(render_chat([("user", r["prompt"])])) for r in rows] or [
            torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist() for _ in range(8)
        ]
        end_think = tuple(tok.encode("</think>\n\n")) if args.model == "qwen38-27b" else ()
        sampling = SamplingParams(
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            stop_token_ids=getattr(tok, "stop_token_ids", ()),
            thinking_budget=args.think_budget if end_think else None, end_think_ids=end_think)

        # The run id is the hash of these: same inputs = same run, a finished
        # one is returned instead of retrained. Checkpoint path + code commit is
        # the identity, not the 15 GB of weight bytes.
        inputs = {"model": args.model, "recipe": args.recipe,
                  "source": _QWEN38_SOURCE if args.model == "qwen38-27b" else "tiny",
                  "commit": commit(), "algo": "grpo" if args.rl else "opd",
                  "data": file_hash(args.data) if args.data else None, "steps": args.steps,
                  "group": args.group, "max_new_tokens": args.max_new_tokens,
                  "temperature": args.temperature, "think_budget": args.think_budget,
                  "lr": args.lr, "lora_rank": args.lora_rank, "seed": args.seed,
                  "eval_mmlu": args.eval_mmlu,
                  "eval_gsm8k": file_hash(args.eval_gsm8k) if args.eval_gsm8k else None,
                  "eval_n": args.eval_n}
        manifest = new_manifest("train", inputs)
        prev = read_manifest(runs_root(), manifest["id"])
        if prev and prev["finished"] and not args.force:
            log(f"run {prev['id']} already finished; --force reruns")
            return _finish(prev, args.json)
        manifest["metrics"] = dict.fromkeys(("reward_first", "reward_last", "ce_last",
                                             "secs_per_step_median", "tied_group_fraction", "mmlu_before", "mmlu_after",
                                             "gsm8k_before", "gsm8k_after"))
        eval_rows = ([json.loads(ln) for ln in Path(args.eval_gsm8k).read_text().splitlines()
                      if ln.strip()][: args.eval_n] if args.eval_gsm8k else [])

        def evals(engine, tag):
            if args.eval_mmlu:
                c, n = mmlu_accuracy(engine, tok, args.eval_mmlu, concurrency=8)
                manifest["metrics"][f"mmlu_{tag}"] = c / n
                log(f"mmlu 0-shot {c}/{n} = {100 * c / n:.1f}%")
            if eval_rows:
                c, n = gsm8k_accuracy(engine, tok, eval_rows, sampling, concurrency=8)
                manifest["metrics"][f"gsm8k_{tag}"] = c
                log(f"gsm8k greedy {c}/{n} = {100 * c / n:.1f}%")

    if args.rl:
        from .engine import build_engine
        from .model import add_lora

        # The engine that samples IS the model that trains — one runtime, one
        # set of weights. LoRA keeps the trainable set small enough that a
        # rollout and its update share a card.
        # Prefix cache and captured graph BOTH off, and both for correctness,
        # not speed: a cached prefix serves KV computed under the policy of an
        # earlier step, and a captured graph bakes the f32 casts of the weights,
        # which the optimizer's in-place update only invalidates on the eager
        # path. Either one makes the rollout off-policy without saying so.
        from .kv_cache import NoPrefixStore

        engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=8,
                              decode_graph=False, prefix_store=NoPrefixStore())
        trainable = add_lora(model, rank=args.lora_rank)
        if rows:
            gold = {tuple(p): r["answer"] for p, r in zip(prompts, rows)}

            def reward(prompt, completion):
                text = tok.decode([int(t) for t in completion])
                return float(answer_match(text, gold[tuple(int(t) for t in prompt)]))
        else:
            half = cfg.vocab_size // 2
            # Demo reward: dense, so an untrained policy's group has variance
            # and GRPO has a gradient at step 0.
            def reward(prompt, completion):
                return sum(1 for t in completion if t < half) / max(len(completion), 1)

        optimizer.lr = args.lr
        evals(engine, "before")  # LoRA B is zero at init, so this is the base model's score
        hist = train_mod.grpo_loop(engine, model, prompts, reward, args.steps, backend,
                                   optimizer, group=args.group, sampling=sampling,
                                   seed=args.seed, trainable=trainable)
        for i, (r, ce, secs, tied) in enumerate(hist):
            log(f"step {i + 1:4d}/{args.steps}  reward {r:.4f}  ce {ce:.4f}  "
                f"tied {tied:.2f}  {secs:.1f}s")
        evals(engine, "after")
        manifest["metrics"].update(
            reward_first=hist[0][0], reward_last=hist[-1][0], ce_last=hist[-1][1],
            secs_per_step_median=statistics.median(h[2] for h in hist),
            tied_group_fraction=statistics.mean(h[3] for h in hist))
        return _finish(manifest, args.json)
    if args.opd:
        from .engine import build_engine
        from .model import add_lora
        from .spec import load_draft

        # OPD: the teacher IS this model, generating through the engine with an
        # EMA of the adapters. Speculation is a property of the teacher engine,
        # so a draft head accelerates the rollout half without the loop knowing.
        draft = load_draft(model, args.draft) if args.draft else None
        # Engine first: build_engine materializes the params, and an adapter
        # created before that points at an object the forward no longer reads.
        # Prefix cache and captured graph off, as for GRPO: both would sample
        # from an earlier policy without raising.
        from .kv_cache import NoPrefixStore

        teacher = build_engine(cfg, model, backend, num_blocks=512, num_slots=8,
                               draft=draft, spec_depth=args.depth,
                               decode_graph=False, prefix_store=NoPrefixStore())
        trainable = add_lora(model, rank=args.lora_rank)
        evals(teacher, "before")
        losses = train_mod.opd_loop(teacher, model, prompts, args.steps, backend, optimizer,
                                    seed=args.seed, trainable=trainable, sampling=sampling)
        for i, loss in enumerate(losses):
            log(f"step {i + 1:4d}/{args.steps}  loss {loss:.4f}")
        evals(teacher, "after")
        manifest["metrics"]["ce_last"] = losses[-1]
        return _finish(manifest, args.json)
    for step in range(args.steps):
        # ponytail: fixed batch=2 seq=64 random-token batch; a real corpus
        # plugs in here without touching train_step.
        input_ids = torch.randint(0, cfg.vocab_size, (2, 64), generator=gen)
        optimizer.lr = cosine_warmup(step, args.steps, 5, 1e-3)
        # Fresh tape per step: one backward per tape (a reused tape leaks the
        # step's intermediates and replays all history on each backward).
        loss = train_mod.train_step(model, input_ids, backend, optimizer, Tape())
        print(f"step {step + 1:4d}/{args.steps}  loss {loss:.4f}")


def _finish(m: dict, as_json: bool) -> None:
    """Gate, write the manifest, print it, exit non-zero on a failed gate.
    A gate whose metric was not evaluated passes vacuously (value null)."""
    from .ledger import format_run, gates_pass, now, runs_root, write_manifest

    if not m["finished"]:
        g = m["metrics"]
        mmlu_floor = None if g["mmlu_before"] is None else g["mmlu_before"] - 0.03
        m["gates"] = [
            {"name": n, "value": v, "threshold": t,
             "passed": v is None or t is None or ok(v, t)}
            for n, v, t, ok in (
                ("reward_rises", g["reward_last"], g["reward_first"], lambda v, t: v > t),
                ("mmlu_holds", g["mmlu_after"], mmlu_floor, lambda v, t: v >= t),
                ("gsm8k_improves", g["gsm8k_after"], g["gsm8k_before"], lambda v, t: v > t),
                ("groups_untied", g["tied_group_fraction"], 0.5, lambda v, t: v < t),
            )]
        m["finished"] = now()
        write_manifest(runs_root(), m)
    print(json.dumps(m, indent=1) if as_json else format_run(m))
    if not gates_pass(m):
        sys.exit(1)


# ---------------------------------------------------------------------------
# pretrain
# ---------------------------------------------------------------------------


def cmd_pretrain(args: argparse.Namespace) -> None:
    from . import train as train_mod
    from .autograd import AdamW
    from tilerl_kernels.backend import get_backend
    from .server import get_tokenizer

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=args.seed, keep_master=True)
    from .model import drop_quantized

    drop_quantized(model)
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


def _devices(spec: str) -> list[int]:
    """``0-7`` or ``0,1,2`` or ``0-3,6``."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def cmd_generate(args: argparse.Namespace) -> None:
    """Fan a prompt corpus across devices, one process each.

    A process per device rather than one process with N contexts: the
    in-process wrapper serialises every tick's Python half on the GIL, and
    8 independent processes are what measured 7.54x on 8 H20s.
    """
    import json

    from .generate import generate

    stats = generate(
        prompts=args.prompts, out=args.out, devices=_devices(args.devices),
        source=args.source, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        max_batch=args.max_batch,
    )
    print(json.dumps(stats))


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
    from tilerl_kernels.backend import get_backend

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


def cmd_merge(args: argparse.Namespace) -> None:
    from .ledger import commit, new_manifest, now, runs_root, write_manifest
    from .merge import merge_checkpoints

    n = merge_checkpoints(args.base, args.specialists, args.out, method=args.method)
    m = new_manifest("merge", {"base": args.base, "specialists": list(args.specialists),
                               "method": args.method, "commit": commit()})
    m["metrics"], m["artifacts"], m["finished"] = {"tensors": n}, {"out": args.out}, now()
    write_manifest(runs_root(), m)
    print(f"merged {len(args.specialists)} specialists ({args.method}) -> {args.out}  run {m['id']}")


def cmd_ledger(args: argparse.Namespace) -> None:
    from .ledger import format_run, lineage, list_runs, runs_root

    runs = lineage(runs_root(), args.lineage) if args.lineage else list_runs(runs_root())
    print(json.dumps(runs, indent=1) if args.json else "\n".join(map(format_run, runs)))


def _build_parser(recipe: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tilerl",
        description="tileRL: TileLang inference + training (CPU/CUDA/Metal).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="start the OpenAI-compatible HTTP server")
    p_serve.add_argument("--model", choices=["tiny", "tiny-agent", "qwen38-27b"], default="tiny")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--devices", default="",
                         help="replicate inside ONE process across these CUDA indices, e.g. 0,1,2,3. "
                              "A CUDA fault in one replica is sticky for the whole process and takes "
                              "the others down while HTTP keeps answering; for independent endpoints "
                              "run one process per card under CUDA_VISIBLE_DEVICES instead.",
                         type=lambda v: [int(x) for x in v.split(",")] if v else [])
    p_serve.set_defaults(func=cmd_serve)

    p_train = sub.add_parser("train", help="SFT, --rl (GRPO) or --opd; --recipe for a gated flag set")
    # qwen38-27b loads from TILERL_QWEN38_SOURCE; --rl trains LoRA on the frozen
    # fp4 base, which is what fits one card.
    p_train.add_argument("--model", choices=["tiny", "tiny-agent", "qwen38-27b"], default="tiny")
    p_train.add_argument("--steps", type=int, default=20)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--opd", action="store_true",
                         help="on-policy distillation: the engine rolls out, LoRA adapters train")
    p_train.add_argument("--rl", action="store_true",
                         help="GRPO: the engine samples a group per prompt, a reward scores "
                              "them, the group mean is the baseline (no critic)")
    p_train.add_argument("--group", type=int, default=8, help="rollouts per prompt (--rl)")
    p_train.add_argument("--max-new-tokens", type=int, default=32, help="rollout length")
    p_train.add_argument("--data", help="JSONL {prompt, answer}: real prompts, exact-match "
                         "reward on the last number (scripts/gsm8k_jsonl.py)")
    p_train.add_argument("--temperature", type=float, default=1.0, help="rollout temperature")
    p_train.add_argument("--think-budget", type=int, default=0,
                         help="tokens the 27B may spend in <think> per rollout; 0 = none")
    p_train.add_argument("--eval-mmlu", type=int, default=0,
                         help="score N MMLU questions before and after (needs `datasets`)")
    p_train.add_argument("--eval-gsm8k", help="JSONL {prompt, answer}: greedy exact-match "
                         "accuracy before and after")
    p_train.add_argument("--eval-n", type=int, default=100, help="rows of --eval-gsm8k to score")
    p_train.add_argument("--force", action="store_true",
                         help="retrain even if this run's manifest is already finished")
    p_train.add_argument("--json", action="store_true",
                         help="print the run manifest as JSON instead of step lines")
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--optim", choices=["adafactor", "iso"], default="adafactor",
                         help="full-parameter SFT optimizer; --rl/--opd train LoRA and ignore it")
    p_train.add_argument("--lora-rank", type=int, default=16)
    p_train.add_argument("--draft", help="draft head safetensors: speculative rollout (--opd)")
    p_train.add_argument("--depth", type=int, default=2, help="drafts per row per tick")
    p_train.add_argument("--recipe", choices=sorted(RECIPES),
                         help="a flag set that passed a gate (recipes.py); flags override it")
    # The recipe is the subparser's defaults, so anything typed still wins.
    p_train.set_defaults(func=cmd_train, **(recipes.flags(recipe) if recipe else {}))

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

    p_gen = sub.add_parser(
        "generate", help="offline batch generation, one process per device"
    )
    p_gen.add_argument("prompts", help="JSONL, one object per line with token_ids")
    p_gen.add_argument("--out", required=True, help="JSONL to write")
    p_gen.add_argument("--devices", default="0", help="CUDA indices, e.g. 0-7 or 0,1,2")
    p_gen.add_argument("--source", default=None, help="27B checkpoint dir (omit for tiny)")
    p_gen.add_argument("--max-new-tokens", type=int, default=128)
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_gen.add_argument("--top-p", type=float, default=1.0)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--max-batch", type=int, default=32,
                       help="concurrent requests per device")
    p_gen.set_defaults(func=cmd_generate)

    p_merge = sub.add_parser("merge", help="merge specialist checkpoints that share a base")
    p_merge.add_argument("--base", required=True, help="base checkpoint dir")
    p_merge.add_argument(
        "--specialists",
        required=True,
        type=lambda v: v.split(","),
        help="comma-separated specialist checkpoint dirs",
    )
    p_merge.add_argument("--out", required=True, help="merged checkpoint dir")
    p_merge.add_argument("--method", choices=["iso", "average"], default="iso")
    p_merge.set_defaults(func=cmd_merge)

    p_ledger = sub.add_parser("ledger", help="list runs ($TILERL_RUNS, default ./runs)")
    p_ledger.add_argument("--lineage", metavar="ID", help="this run and what it descends from")
    p_ledger.add_argument("--json", action="store_true")
    p_ledger.set_defaults(func=cmd_ledger)

    return parser


def main() -> None:
    recipe = getattr(_build_parser().parse_known_args()[0], "recipe", None)
    if recipe:
        print(f"recipe {recipe}: {RECIPES[recipe]['status']}")
    args = _build_parser(recipe).parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
