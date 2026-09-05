"""tilerl CLI. Heavy imports live inside the handlers so ``--help`` stays instant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .eval import MATCHERS
from .recipes import RECIPES, flags

# ponytail: placeholder hub id; pin the real Qwen3-27B repo when weights land.
_QWEN38_SOURCE = os.environ.get("TILERL_QWEN38_SOURCE", "Qwen/Qwen3-27B")

_NO_WEIGHTS = (
    "hint: download the checkpoint (or set TILERL_QWEN38_SOURCE to a\n"
    "      local safetensors directory), or use --model tiny."
)


def _qwen38_tokenizer():
    """The 27B tokenizer, with the same hint as its weights: a bare hub id 401s."""
    from .tokenizer import get_tokenizer

    try:
        return get_tokenizer(_QWEN38_SOURCE)
    except Exception as exc:
        # HF's 401 body is a dozen lines of auth advice; the first names the cause.
        # Some exceptions (MemoryError) stringify empty, so splitlines() can be [].
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        print(f"error: could not load the Qwen3-27B tokenizer from {_QWEN38_SOURCE!r}: "
              f"{first}\n{_NO_WEIGHTS}", file=sys.stderr)
        sys.exit(1)


def _build_model(
    model_name: str, seed: int, fuse_projections: bool = False, keep_master: bool = False,
    tp: int = 1, backend=None,
):
    """(cfg, model): serving fuses projections, training keeps the bf16 masters.

    ``tp`` > 1 shards both here, because a model can only be sharded where it is
    built: the config's head counts must already be divided before any layer
    reshapes with them.
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
                f"{_NO_WEIGHTS}",
                file=sys.stderr,
            )
            sys.exit(1)
        return _shard(cfg, model, tp, backend, model_mod)
    # tiny-agent is tiny with room for one real agent turn; see config.tiny().
    cfg = config_mod.tiny(65536) if model_name == "tiny-agent" else config_mod.tiny()
    model = model_mod.build_random(
        cfg, seed=seed, fuse_projections=fuse_projections, keep_master=keep_master
    )
    return _shard(cfg, model, tp, backend, model_mod)


def _shard(cfg, model, tp: int, backend, model_mod):
    """Every rank builds the WHOLE model and keeps its slice.

    Wasteful and deliberate: sharding at load time needs a loader that reads
    per-rank slices out of the checkpoint, and that is a separate change. On the
    27B this costs each rank a transient full copy.
    # ponytail: whole-model build then slice, per-rank checkpoint reads when the
    # 27B's transient copy is the binding constraint
    """
    if tp <= 1:
        return cfg, model
    from .tensor_parallel import Mesh, shard_params, tp_config

    # dp is DERIVED from the world, never a second flag: two numbers that must
    # multiply to a third invite a launch where they do not.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world % tp:
        raise SystemExit(f"--tp {tp} does not divide WORLD_SIZE={world}")
    mesh = Mesh(dp=world // tp, tp=tp, rank=rank)  # validates rank < world and the layout
    # Every rank builds every group, tp first then dp, in the same order:
    # new_group is collective, so a rank that skips one deadlocks on first use.
    tp_groups, dp_groups = [], []
    for r in range(world):
        m = Mesh(dp=world // tp, tp=tp, rank=r)
        for g, seen in ((m.tp_group(), tp_groups), (m.dp_group(), dp_groups)):
            if g not in seen:
                seen.append(g)
    backend.init_tp(world, rank, tp_groups, dp_groups)
    return tp_config(cfg, tp), model_mod.Model(
        tp_config(cfg, tp), shard_params(model.params, cfg, mesh.tp_rank, tp))


def _build_engine(cfg, model, backend, devices=None, draft=None, depth=2, slots=16,
                  blocks=0, max_ctx=0, max_batch=8, ssd_path="", ssd_min_tokens=0):
    """Serving-size engine; ``devices`` replicates it across those CUDA indices.

    ``max_ctx`` caps the served context; it still defaults to the model's own limit,
    which for the 27B is 262144 tokens = 275 GB of f32 KV, so it is now a CAP on the
    fit rather than the pool size. ``blocks`` 0 hands the pool to build_engine, which
    fits it after materialize and the allocator reclaim — the only point where free
    memory means anything. Sizing it here instead asked for 10.21 GiB with 4.96 free
    and OOMed in PagedKvPool.

    ``slots`` sizes the GDN state pool; with a draft each slot also owns spec_steps
    of step-state, so a 32 GB card needs 4, not 16.
    """
    from . import engine as engine_mod
    from .kv_cache import BLOCK_TOKENS

    # Token budget follows the context; ByteTokenizer makes one token per byte.
    ctx = int(max_ctx or cfg.max_position_embeddings)

    kw = dict(num_blocks=blocks, num_slots=slots, max_batch=max_batch,
              max_total_tokens=ctx, max_blocks=(ctx * max_batch) // BLOCK_TOKENS)
    if draft is not None:
        kw["draft"], kw["spec_depth"] = draft, depth
    if ssd_path:
        kw["ssd_path"] = ssd_path
        if ssd_min_tokens:
            kw["ssd_min_tokens"] = ssd_min_tokens
    if not devices:
        return engine_mod.build_engine(cfg, model, backend, **kw)

    from tilerl_kernels.backend import Backend, resolve_target

    from .model import load_hf
    from .parallel import DataParallelEngine

    def make(d, **kwargs):
        # A Backend binds the current CUDA device: each replica loads its own copy there.
        b = Backend(resolve_target())
        m = load_hf(cfg, _QWEN38_SOURCE, fuse_projections=True) if model is None else model
        return engine_mod.build_engine(cfg, m, b, **kwargs)

    return DataParallelEngine.build(devices, make, **kw)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    from tilerl_kernels.backend import get_backend

    from .server import create_app, get_tokenizer

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=0, fuse_projections=True)
    draft = None
    if args.draft:
        from .spec import load_draft

        draft = load_draft(model, args.draft)
    engine = _build_engine(cfg, model, backend, devices=args.devices,
                           draft=draft, depth=args.depth, slots=args.slots,
                           blocks=args.blocks, max_ctx=args.max_ctx,
                           max_batch=args.max_batch, ssd_path=args.ssd_path,
                           ssd_min_tokens=args.ssd_min_tokens)
    tokenizer = _qwen38_tokenizer() if args.model == "qwen38-27b" else get_tokenizer(None)

    app = create_app(engine, tokenizer, model_name=cfg.name)
    # Print the pool: with --blocks 0 it is fitted to the card, so this is the served
    # context ceiling and the one number a 32 GB card gets wrong silently.
    from .kv_cache import BLOCK_TOKENS

    kv = engine.stats()["blocks_total"]
    print(f"tilerl serve: model={cfg.name} target={backend.target} "
          f"kv={kv} blocks = {kv * BLOCK_TOKENS} tokens")
    if args.warmup:
        # One capture per (bucket x chain width); generating tokens instead is a
        # lottery with no floor, since the draft's confidence sets each width.
        t0 = time.perf_counter()
        n = engine.precapture()
        print(f"tilerl serve: {n} decode graphs in {time.perf_counter() - t0:.0f}s")
    engine.run()
    print(f"tilerl serve: http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        engine.shutdown()


def cmd_train(args: argparse.Namespace) -> None:
    if args.rl or args.opd:
        if not args.data and not (args.recipe == "grpo-tiny-smoke" and args.model == "tiny"):
            sys.exit("error: --data is required for RL/OPD training")
        return _train_adapters(args)
    _train_full(args)


def _jsonl(path: str | None) -> list[dict]:
    if not path:
        return []
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    # A named file with no rows is silent otherwise: cmd_train's `or [...]` falls back
    # to random prompts, so a 100-step GRPO run trains on noise and reports a reward.
    if not rows:
        sys.exit(f"error: {path} has no rows")
    return rows


def _train_full(args: argparse.Namespace) -> None:
    """Full-parameter SFT on random tokens: Adafactor or ISO, streamed updates."""
    import torch
    from tilerl_kernels.backend import get_backend

    from . import train as train_mod
    from .autograd import Adafactor, cosine_warmup
    from .ledger import commit, new_manifest, read_manifest, runs_root
    from .model import drop_quantized

    log = (lambda *a, **k: None) if args.json else print
    # The ledger is per-RUN, not per-algorithm: sft-iso-27b exists to produce a
    # P3 verdict and had nowhere to record one.
    manifest = new_manifest("train", {
        "model": args.model, "recipe": args.recipe,
        "source": _QWEN38_SOURCE if args.model == "qwen38-27b" else "tiny",
        "commit": commit(), "algo": "sft", "optim": args.optim,
        "steps": args.steps, "lr": args.lr, "seed": args.seed})
    prev = read_manifest(runs_root(), manifest["id"])
    if prev and prev["finished"] and not args.force:
        log(f"run {prev['id']} already finished; --force reruns")
        return _finish(prev, args.json)
    if args.steps == 0:
        return _finish(manifest, args.json)
    manifest["metrics"] = dict.fromkeys(("ce_first", "ce_last", "secs_per_step_median"))

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=args.seed, keep_master=True)
    drop_quantized(model)
    # Adam's m+v on the 27B is 200.4 GiB; Adafactor is 0.03 GiB and streams its updates.
    optimizer = Adafactor(lr=args.lr, weight_decay=0.1)
    if args.optim == "iso":
        from .iso import ISO

        optimizer = ISO(optimizer)
    gen = torch.Generator().manual_seed(args.seed)
    log(f"tilerl train: model={cfg.name} layers={cfg.num_layers} "
        f"hidden={cfg.hidden_size} vocab={cfg.vocab_size} steps={args.steps}")
    losses, secs = [], []
    for step in range(args.steps):
        # ponytail: random-token batch; a real corpus plugs in here without touching train_step.
        input_ids = torch.randint(0, cfg.vocab_size, (2, 64), generator=gen)
        optimizer.lr = cosine_warmup(step, args.steps, 5, args.lr)
        t0 = time.perf_counter()
        loss = train_mod.train_step(model, input_ids, backend, optimizer)
        secs.append(time.perf_counter() - t0)
        losses.append(loss)
        log(f"step {step + 1:4d}/{args.steps}  loss {loss:.4f}  {secs[-1]:.1f}s")
    manifest["metrics"].update(
        ce_first=losses[0], ce_last=losses[-1],
        secs_per_step_median=statistics.median(secs))
    if torch.cuda.is_available():
        manifest["metrics"]["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return _finish(manifest, args.json)


def _load_adapter(trainable: dict, path: str, log) -> None:
    """Copy a saved adapter INTO the tensors add_lora just attached.

    ``copy_``, never rebind: the forward reads the objects add_lora put in
    ``model.params``, so assigning new tensors here would load an adapter the model
    never sees and re-score the base while reporting a trained number.

    Unknown or missing keys are refused rather than skipped. An adapter saved before
    the dead-adapter fix (#98) carries ``<weight>.scale.lora_*`` and ``conv1d.lora_*``
    keys that no longer exist, and silently dropping them would load a partial adapter
    under a full adapter's name.
    """
    import torch
    from safetensors.torch import load_file

    saved = load_file(path)
    extra, missing = set(saved) - set(trainable), set(trainable) - set(saved)
    if extra or missing:
        raise SystemExit(
            f"error: {path} does not match this model's adapter\n"
            + (f"  {len(extra)} unknown key(s), e.g. {sorted(extra)[:3]}\n" if extra else "")
            + (f"  {len(missing)} missing key(s), e.g. {sorted(missing)[:3]}\n" if missing else "")
            + "  hint: an adapter saved before the dead-adapter fix carries "
              ".scale/.conv1d adapters that no longer exist; retrain or strip them")
    with torch.no_grad():
        for k, v in saved.items():
            t = trainable[k]
            if tuple(v.shape) != tuple(t.shape):
                raise SystemExit(
                    f"error: {path}: {k} is {tuple(v.shape)}, expected {tuple(t.shape)}")
            t.copy_(v.to(device=t.device, dtype=t.dtype))
    log(f"loaded adapter {sum(v.numel() for v in saved.values()) / 1e6:.1f}M params <- {path}")


def _before_eval_key(args, cfg, backend, eval_params, mmlu_set) -> str | None:
    """The cache key, or None when the base model's identity is not in it.

    ``weights`` is always present and never absent-by-omission: the 27B keys on its
    checkpoint files, `tiny` is a pure function of ``--seed`` and says so, and any
    other model REFUSES to cache rather than key on a base it cannot identify --
    a key that silently omits the weights serves one model's before-arm for another.
    """
    from .ledger import file_hash

    if args.model == "qwen38-27b":
        source = Path(_QWEN38_SOURCE)
        if not source.is_dir():
            from huggingface_hub import snapshot_download

            source = Path(snapshot_download(_QWEN38_SOURCE, local_files_only=True))
        weights = [(str(p.resolve()), s.st_size, s.st_mtime_ns)
                   for p in sorted(source.iterdir()) if p.is_file() for s in [p.stat()]]
    elif args.model.startswith("tiny"):
        weights = None  # built by build_random(seed), and the seed is in `sampling`
    else:
        return None
    inputs = {
        "version": 2, "weights": weights, "config": asdict(cfg),
        # cfg is already tp_config(cfg, tp) here, so tp reaches the key through the
        # sharded dims -- but only while that call order holds. Explicit is cheaper.
        "tp": args.tp,
        "target": backend.target, "precision": backend.precision,
        "eval_file": file_hash(args.eval_gsm8k) if args.eval_gsm8k else None,
        "eval_n": args.eval_n, "matcher": args.reward, "sampling": asdict(eval_params),
        "thinking": args.max_think_tokens > 0 if args.model == "qwen38-27b" else None,
        "mmlu": mmlu_set, "concurrency": 8,
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _write_eval_rows(run_id: str, tag: str, rows: list) -> float:
    """One JSON row per problem, so two arms over the same set can be compared
    paired. Returns the mean completion length. P1 fell back to the unpaired
    interval because only totals were kept.

    Creates the run directory: `_finish` makes it, and `_finish` runs AFTER both
    eval arms, so a `not is_dir(): return` here silently wrote nothing at all --
    which is what it did on the first MATH run.
    """
    from .ledger import runs_root

    d = Path(runs_root()) / run_id
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"eval-{tag}.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return sum(r["tokens"] for r in rows) / max(1, len(rows))


#: the before-arm's mean completion must leave headroom under the rollout cap. 0.8
#: rather than 1.0 because the MEAN fitting exactly means half the rollouts do not.
_ROLLOUT_HEADROOM = 0.8


def _refuse_short_rollouts(mean_len: float | None, cap: int, allow: bool = False) -> None:
    """Stop before training when the rollouts cannot reach an answer.

    The base policy's own completion length is measured by the before-arm that just
    ran, so this compares two known numbers rather than guessing. Truncated rollouts
    never emit the answer, every sample in a group scores 0, and GRPO trains on a
    reward that is constant -- 100 steps of tied-at-the-floor groups, which looks
    like a hard task rather than a misconfiguration (measured: MATH level 5 needs
    1038 tokens against a 512 cap, 5 of the first 6 steps tied at 1.00, reward 0).

    The mirror of it is the eval cap, which scores the cap instead of the policy
    (errors/2026-09-04-the-eval-cap-measured-itself.md). Same family: a length
    parameter set without measuring the length it bounds.
    """
    if not mean_len or allow or mean_len <= _ROLLOUT_HEADROOM * cap:
        return
    sys.exit(
        f"error: the base policy averages {mean_len:.0f} completion tokens but "
        f"--max-new-tokens is {cap}. Rollouts would be truncated before they answer, "
        f"so every group ties at the floor and no gradient flows. Raise the cap above "
        f"{mean_len / _ROLLOUT_HEADROOM:.0f}, pick an easier task, or pass "
        f"--allow-short-rollouts if the truncation is deliberate."
    )


def _train_adapters(args: argparse.Namespace) -> None:
    """GRPO or OPD: LoRA on the frozen base, the engine samples, the ledger gates."""
    import torch
    from tilerl_kernels.backend import get_backend

    from . import train as train_mod
    from .autograd import AdamW
    from .engine import build_engine
    from .eval import gsm8k_accuracy, mmlu_accuracy, mmlu_questions
    from .kv_cache import NoPrefixStore
    from .ledger import commit, file_hash, new_manifest, read_manifest, runs_root
    from .model import add_lora
    from .prompt import render_chat, sampling
    from .tokenizer import get_tokenizer

    real = args.model == "qwen38-27b"
    log = (lambda *a, **k: None) if args.json else print
    tok = _qwen38_tokenizer() if real else get_tokenizer(None)
    rows, eval_rows = _jsonl(args.data), _jsonl(args.eval_gsm8k)[: args.eval_n]
    thinking = (args.max_think_tokens > 0) if real else None
    params = sampling(tok, thinking, args.max_new_tokens, temperature=args.temperature,
                      max_think_tokens=args.max_think_tokens, seed=args.seed)
    # A SEPARATE params for the eval arms. Sharing `params` scored the policy at the
    # ROLLOUT cap, so the eval measured the cap: 38.4% with mean completion 238.7
    # against a 256 cap, ~82.5% uncapped
    # (errors/2026-09-04-the-eval-cap-measured-itself.md). Same prompt template and
    # stop ids -- only the length differs, and gsm8k_accuracy forces temperature 0.
    eval_params = sampling(tok, thinking, args.eval_max_new_tokens,
                           temperature=args.temperature,
                           max_think_tokens=args.max_think_tokens, seed=args.seed)

    # Same inputs = same run: a finished one is returned instead of retrained.
    manifest = new_manifest("train", {
        "model": args.model, "recipe": args.recipe, "source": _QWEN38_SOURCE if real else "tiny",
        "commit": commit(), "algo": "grpo" if args.rl else "opd",
        "data": file_hash(args.data) if args.data else None, "steps": args.steps,
        "group": args.group, "max_new_tokens": args.max_new_tokens,
        "allow_short_rollouts": args.allow_short_rollouts,
        "temperature": params.temperature, "max_think_tokens": args.max_think_tokens,
        "lr": args.lr, "lora_rank": args.lora_rank, "seed": args.seed, "eval_mmlu": args.eval_mmlu,
        # In the id: tp=1 and tp=4 are different runs, and without this the second
        # would be handed the first's finished manifest and never train.
        "tp": args.tp,
        "reward": args.reward,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "load_adapter": file_hash(args.load_adapter) if args.load_adapter else None,
        "eval_gsm8k": file_hash(args.eval_gsm8k) if args.eval_gsm8k else None,
        "eval_n": args.eval_n})
    prev = read_manifest(runs_root(), manifest["id"])
    if prev and prev["finished"] and not args.force:
        log(f"run {prev['id']} already finished; --force reruns")
        return _finish(prev, args.json)
    manifest["metrics"] = dict.fromkeys((
        "mmlu_before", "mmlu_after", "gsm8k_before", "gsm8k_after",
        "gsm8k_before_tokens", "gsm8k_after_tokens", "peak_gib"))

    backend = get_backend()
    # LoRA on a frozen base needs no bf16 master (~27 GB on the 27B).
    cfg, model = _build_model(args.model, seed=args.seed, keep_master=False,
                              tp=args.tp, backend=backend)
    log(f"tilerl train: model={cfg.name} layers={cfg.num_layers} "
        f"hidden={cfg.hidden_size} vocab={cfg.vocab_size} steps={args.steps}")
    gen = torch.Generator().manual_seed(args.seed)
    prompts = [tok.encode(render_chat([("user", r["prompt"])], thinking)) for r in rows] or [
        torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist() for _ in range(8)]
    draft = None
    if args.opd and args.draft:
        from .spec import load_draft

        draft = load_draft(model, args.draft)
    # The pool holds every in-flight row's whole sequence, so a flat 512 blocks is
    # 8192 tokens across 8 slots -- 1024 each. Past that the rollout dies mid-step on
    # "PagedKvPool exhausted" (kv_cache.py:80), so --max-new-tokens above ~1024 was
    # unreachable however the recipe was written. Size the pool from the ask instead.
    from .kv_cache import BLOCK_TOKENS

    # 1024 floor = the old flat 512 blocks. evals submit prompts this function never
    # sees -- MMLU reaches 515 tokens against GSM8K's 183 -- so a short
    # --max-new-tokens must not shrink the pool under them. max_total_tokens only
    # guards one request and costs no memory, so it never drops below the 8192 default.
    ctx = max(max(map(len, prompts)) + args.max_new_tokens + 64, 1024)
    engine = build_engine(cfg, model, backend, num_slots=8, max_batch=8, draft=draft,
                          num_blocks=-(-ctx // BLOCK_TOKENS) * 8 + 8,
                          max_total_tokens=max(ctx, 8192),
                          spec_depth=args.depth, decode_graph=True,
                          prefix_store=NoPrefixStore())
    # After build_engine: it materializes the params an adapter must point at.
    trainable = add_lora(model, rank=args.lora_rank)
    if args.load_adapter:
        _load_adapter(trainable, args.load_adapter, log)
    optimizer = AdamW(lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    mean_len: dict[str, float | None] = {}
    mmlu_set = mmlu_questions(args.eval_mmlu) if args.eval_mmlu else None
    cache = None
    if (eval_rows or mmlu_set) and not args.load_adapter and not args.draft:
        key = _before_eval_key(args, cfg, backend, eval_params, mmlu_set)
        if key is not None:
            cache = Path(runs_root()) / "eval-cache" / f"{key}.json"
            manifest["eval_before_cache"] = {"key": key, "cache_hit": cache.is_file()}

    def evals(tag):
        if tag == "before" and cache is not None and cache.is_file():
            saved = json.loads(cache.read_text())
            manifest["metrics"].update(saved["metrics"])
            _write_eval_rows(manifest["id"], tag, saved["rows"])
            mean_len[tag] = saved["mean_len"]
            manifest["eval_before_cache"]["cache_hit"] = True
            log(f"eval before: cache hit {cache.stem}")
            return
        rows_out: list = []
        if args.eval_mmlu:
            c, n, conc = mmlu_accuracy(engine, tok, args.eval_mmlu, concurrency=8,
                                       questions=mmlu_set, per_problem=rows_out)
            manifest["metrics"][f"mmlu_{tag}"] = c / n
            manifest["metrics"][f"mmlu_{tag}_concurrency"] = conc
            manifest["metrics"][f"mmlu_{tag}_correct"] = c
            manifest["metrics"][f"mmlu_{tag}_total"] = n
            log(f"mmlu 0-shot {c}/{n} = {100 * c / n:.1f}% (seed 0, concurrency {conc})")
        if eval_rows:
            gsm_rows: list = []
            c, n, ntok = gsm8k_accuracy(engine, tok, eval_rows, eval_params, concurrency=8,
                                        thinking=thinking,
                                        match=MATCHERS[args.reward],
                                        per_problem=gsm_rows)
            mean_len[tag] = sum(r["tokens"] for r in gsm_rows) / max(1, len(gsm_rows))
            rows_out.extend(dict(r, dataset="gsm8k") for r in gsm_rows)
            manifest["metrics"][f"gsm8k_{tag}"] = c
            manifest["metrics"][f"gsm8k_{tag}_tokens"] = ntok
            manifest["metrics"][f"gsm8k_{tag}_total"] = n
            # tokens/correct, not tokens: the ratio is what a length claim compares
            # on, and it cannot be improved by getting fewer questions right.
            per = f"  {ntok} tokens ({ntok / c:.1f}/correct)" if c else f"  {ntok} tokens"
            log(f"gsm8k greedy {c}/{n} = {100 * c / n:.1f}%{per}")
        if rows_out:
            _write_eval_rows(manifest["id"], tag, rows_out)
        if tag == "before" and cache is not None:
            saved = {"metrics": {k: v for k, v in manifest["metrics"].items() if "_before" in k},
                     "rows": rows_out, "mean_len": mean_len.get(tag)}
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=cache.parent, delete=False) as f:
                json.dump(saved, f)
            os.replace(f.name, cache)

    drift = {"name": "rollouts_within_cap", "value": None,
             "threshold": _ROLLOUT_HEADROOM * args.max_new_tokens,
             "skipped": True, "passed": None}
    if args.rl:
        manifest["gates"].append(drift)
    evals("before")  # LoRA B is zero at init: the base model's score
    if args.steps == 0:
        evals("after")
        return _finish(manifest, args.json)
    _refuse_short_rollouts(mean_len.get("before"), args.max_new_tokens,
                           args.allow_short_rollouts)
    if args.rl:
        if rows:
            gold = {tuple(p): r["answer"] for p, r in zip(prompts, rows)}
            match = MATCHERS[args.reward]

            def reward(prompt, completion):
                text = tok.decode([int(t) for t in completion])
                return float(match(text, gold[tuple(int(t) for t in prompt)]))
        else:
            half = cfg.vocab_size // 2

            def reward(prompt, completion):
                return sum(1 for t in completion if t < half) / max(len(completion), 1)

        tiebreak = _judge_tiebreak(engine, tok, params) if args.judge else None

        hist = []
        for i, (r, ce, secs, tied, ntok, timings, width) in enumerate(
                train_mod.grpo_loop(engine, model, prompts, reward, args.steps, backend, optimizer,
                                    group=args.group, sampling=params, seed=args.seed,
                                    trainable=trainable, micro=args.micro,
                                    tiebreak=tiebreak, recapture_graph=True)):
            hist.append((r, ce, secs, tied, ntok))
            for phase, elapsed in timings.items():
                manifest["metrics"][phase] = manifest["metrics"].get(phase, 0.0) + elapsed
            log(f"step {i + 1:4d}/{args.steps}  reward {r:.4f}  ce {ce:.4f}  "
                f"tied {tied:.2f}  tok {ntok:.0f}  width {width}  {secs:.1f}s  "
                f"rollout {timings['rollout_secs']:.3f}s  "
                f"backward {timings['backward_secs']:.3f}s  "
                f"optimizer {timings['optimizer_secs']:.6f}s", flush=True)
            if len(hist) >= 5 and not args.allow_short_rollouts:
                mean = statistics.mean(h[4] for h in hist[-5:])
                drift.update(value=mean, step=i + 1, skipped=False,
                             passed=mean <= drift["threshold"])
                manifest["metrics"]["rollout_window_mean"] = mean
                if not drift["passed"]:
                    drift["reason"] = (
                        f"error: at step {i + 1} the last 5 steps average {mean:.1f} "
                        f"completion tokens but --max-new-tokens is {args.max_new_tokens}. "
                        f"Rollouts risk truncation before they answer. Raise the cap above "
                        f"{mean / _ROLLOUT_HEADROOM:.0f}, pick an easier task, or pass "
                        f"--allow-short-rollouts if the truncation is deliberate.")
                    log(drift["reason"], flush=True)
                    break
        # Windowed means, not hist[0] vs hist[-1]: per-step reward moves with the
        # sampled prompt, so two single steps compare two draws, not two policies
        # (tests/test_rl.py::test_grpo_loop_raises_reward uses the same windows).
        w = max(1, len(hist) // 4)
        manifest["metrics"].update(
            steps_completed=len(hist),
            reward_first=statistics.mean(h[0] for h in hist[:w]),
            reward_last=statistics.mean(h[0] for h in hist[-w:]),
            ce_last=hist[-1][1],
            secs_per_step_median=statistics.median(h[2] for h in hist),
            secs_total=sum(h[2] for h in hist),
            tied_group_fraction=statistics.mean(h[3] for h in hist),
            # --judge drives tied_group_fraction toward 0 by construction, so it
            # cannot report a bad judge. Length is the signal that can.
            tokens_first=statistics.mean(h[4] for h in hist[:w]),
            tokens_last=statistics.mean(h[4] for h in hist[-w:]))
    else:
        losses = train_mod.opd_loop(engine, model, prompts, args.steps, backend, optimizer,
                                    seed=args.seed, trainable=trainable, sampling=params,
                                    recapture_graph=True)
        for i, loss in enumerate(losses):
            log(f"step {i + 1:4d}/{args.steps}  loss {loss:.4f}")
        manifest["metrics"]["ce_last"] = losses[-1]
    if torch.cuda.is_available():  # the number the group size is really bounded by
        manifest["metrics"]["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        log(f"peak allocated {manifest['metrics']['peak_gib']:.2f} GiB")
    # Before the after-eval, not after it: a gsm8k_after that beats its own baseline is
    # the run's whole claim, and without the weights that produced it nobody can check
    # whether the metric moved or the reward was gamed. An eval that dies still leaves
    # the adapter behind.
    from safetensors.torch import save_file

    d = Path(runs_root()) / manifest["id"]
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in trainable.items()},
              str(d / "adapter.safetensors"))
    manifest["artifacts"]["adapter"] = "adapter.safetensors"
    log(f"adapter {sum(v.numel() for v in trainable.values()) / 1e6:.1f}M params -> {d}")
    if drift["passed"] is not False:
        evals("after")
    else:
        # The after-arm never ran, so `mmlu_after`/`gsm8k_after` are None -- and
        # `_finish`'s `v is None or ...` would score both gates PASS on a run that
        # measured neither. Mark them skipped so the manifest says "not measured".
        manifest["gates_skip_after"] = True
    return _finish(manifest, args.json)


def _judge_tiebreak(engine, tok, params):
    """Rank rollouts the binary reward cannot separate, using the policy as its own judge.

    `answer_match` decides first and the judge only reorders inside the all-pass or
    all-fail subgroup (judge.py enforces that split), so no judgement can lift a wrong
    answer over a right one. All C(K,2) pairs are generated in ONE batch and looked up,
    because judge_rewards asks pair by pair and 56 sequential round trips per step
    would cost more than the training step itself.
    """
    from dataclasses import replace

    from .eval import generate
    from .judge import judge_rewards
    from .prompt import render_chat

    sp = replace(params, temperature=0.0, max_new_tokens=4, max_think_tokens=0)

    def ask(q, a, b):
        return render_chat([("user",
            f"Problem:\n{q}\n\nTwo worked solutions.\n\n[A]\n{a}\n\n[B]\n{b}\n\n"
            "Which shows the better reasoning: clearer steps, no unjustified leaps, "
            "no wasted work? Reply with exactly one token: A or B or tie.")], False)

    def pick(t):
        t = (t or "").strip().upper()
        return "A" if t.startswith("A") else "B" if t.startswith("B") else "tie"

    def tiebreak(prompt, comps, passed):
        q = tok.decode([int(t) for t in prompt])
        texts = [tok.decode([int(t) for t in c]) for c in comps]
        pairs = [(i, j) for i in range(len(comps)) for j in range(i + 1, len(comps))]
        # Both orders for every pair: pair_verdict abstains unless the swapped call
        # agrees, which is the position-bias control and is not optional.
        prompts = [ask(q, texts[i], texts[j]) for i, j in pairs] + \
                  [ask(q, texts[j], texts[i]) for i, j in pairs]
        out = generate(engine, tok, prompts, sp, 8)
        n = len(pairs)
        seen = {(i, j): (pick(out[k]), pick(out[k + n])) for k, (i, j) in enumerate(pairs)}
        scores, _ = judge_rewards(list(range(len(comps))), passed,
                                  lambda a, b: seen[(a, b)] if (a, b) in seen
                                  else tuple(reversed(seen[(b, a)])))
        return scores

    return tiebreak


def _timing_snapshot(m: dict) -> None:
    """Compare this run's speed against the SOTA baseline row and record the verdict.

    steps/SECOND, not seconds/step: every row in bench-baseline.json is higher-is-better
    and the gate's three comparisons are all `>`, so raw seconds would make a SLOWER run
    read as a new record (tests/test_bench_gate.py holds that).

    Best-effort: a run's result is the manifest, and a missing bench harness must not
    fail the run that produced it.
    """
    import importlib.util

    from .ledger import runs_root

    secs = (m.get("metrics") or {}).get("secs_per_step_median")
    if not secs:
        return
    hp = Path(__file__).resolve().parents[2] / "scripts" / "bench_harness.py"
    spec = importlib.util.spec_from_file_location("bench_harness", hp)
    if spec is None or spec.loader is None:
        return
    try:
        bh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bh)
        i = m["inputs"]
        shape = f"{i['model']}-{i['algo']}-g{i.get('group')}-t{i.get('max_new_tokens')}"
        # seed_only=False AND never dirty: seeding writes the tracked json, and every
        # pytest run of grpo-tiny-smoke would seed a row -- five junk CPU rows landed in
        # it the first time this ran. A run reports against the baseline; it never
        # edits it. Adding a key stays a deliberate act.
        gate = bh.Gate(os.environ.get("TILERL_TARGET", "cpu"))
        gate.check("train-run", shape, 1.0 / secs, unit="step/s")
        gate.dirty = False
        gate.finish(Path(runs_root()) / m["id"] / "baseline-candidate.json")
    except Exception as exc:  # noqa: BLE001 - the manifest is already written
        print(f"  (timing snapshot skipped: {exc})")


#: Gates whose value comes from the after-arm. A guard stop skips that arm, so these
#: are "not measured" rather than passed -- `_finish` scores a None value as True.
_AFTER_GATES = frozenset({"mmlu_holds", "gsm8k_improves"})


def _finish(m: dict, as_json: bool) -> None:
    """Gate, write the manifest, print it, exit non-zero on a failed gate.
    A gate whose metric was not evaluated passes vacuously (value null)."""
    from .ledger import format_run, gates_pass, now, runs_root, write_manifest

    if not m["finished"]:
        g = m["metrics"]
        # .get, not [...]: "a gate whose metric was not evaluated passes
        # vacuously" already covers a metric set that never had the key, which
        # is what an SFT run's manifest is.
        mmlu_floor = None if g.get("mmlu_before") is None else g["mmlu_before"] - 0.03
        skipped = m["inputs"].get("steps") == 0
        after_skipped = bool(m.pop("gates_skip_after", False))
        m["gates"] += [
            {"name": n, "value": v, "threshold": t,
             "skipped": skipped or (after_skipped and n in _AFTER_GATES),
             "passed": None if skipped or (after_skipped and n in _AFTER_GATES)
             else v is None or t is None or ok(v, t)}
            for n, v, t, ok in (
                ("reward_rises", g.get("reward_last"), g.get("reward_first"), lambda v, t: v > t),
                ("mmlu_holds", g.get("mmlu_after"), mmlu_floor, lambda v, t: v >= t),
                ("gsm8k_improves", g.get("gsm8k_after"), g.get("gsm8k_before"),
                 lambda v, t: v > t),
                ("groups_untied", g.get("tied_group_fraction"), 0.5, lambda v, t: v < t),
                ("ce_falls", g.get("ce_last"), g.get("ce_first"), lambda v, t: v < t),
            )]
        m["finished"] = now()
        write_manifest(runs_root(), m)
        _timing_snapshot(m)
    print(json.dumps(m, indent=1) if as_json else format_run(m))
    if not gates_pass(m):
        sys.exit(1)


def cmd_pretrain(args: argparse.Namespace) -> None:
    from tilerl_kernels.backend import get_backend

    from . import train as train_mod
    from .autograd import AdamW
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
    # One process per device: an in-process wrapper serialises every tick on the GIL.
    from .generate import generate

    stats = generate(
        prompts=args.prompts, out=args.out, devices=_devices(args.devices),
        source=args.source, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        max_batch=args.max_batch,
    )
    print(json.dumps(stats))


def cmd_bench(args: argparse.Namespace) -> None:
    if args.suite:
        import subprocess

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
    from tilerl_kernels.backend import get_backend

    from . import engine as engine_mod

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

    # Warmup at the timed prompt_len: the JIT specializes per shape; gen=2 compiles decode too.
    warmup_id = engine.submit(
        rand_ids(args.prompt_len),
        engine_mod.SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=2, seed=0),
    )
    run_to_done(warmup_id, max_ticks=16)

    req_id = engine.submit(
        rand_ids(args.prompt_len),
        engine_mod.SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=args.gen, seed=0),
    )
    t0 = time.perf_counter()
    engine.step()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

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
                         help="replicate inside ONE process across these CUDA indices, e.g. 0,1,2,3 or 0-3. "
                              "A CUDA fault in one replica is sticky for the whole process and takes "
                              "the others down while HTTP keeps answering; for independent endpoints "
                              "run one process per card under CUDA_VISIBLE_DEVICES instead.",
                         type=lambda v: _devices(v) if v else [])
    p_serve.add_argument("--draft", help="MTP/NextN head safetensors: speculative decode. For "
                                        "Qwen3.8-27B-NVFP4 the mtp.* keys all live in "
                                        "model-00018-of-00018.safetensors, so pass that shard.")
    p_serve.add_argument("--depth", type=int, default=3,
                         help="drafts per row per tick; 3 fills the sm70 verify ladder's "
                              "4-row rung exactly (spec.LADDER_WIDTHS) — 4 spills to the "
                              "8-row rung and measured slower than no speculation")
    p_serve.add_argument("--slots", type=int, default=8,
                         help="GDN state slots. A slot is held from submit to finish, so "
                              "this must be >= --max-batch or that concurrency is "
                              "unreachable; above it, each slot is one more queued "
                              "request instead of a 503. They are expensive: measured on "
                              "a 32GB V100 with a draft, slots 8/16 fit 42384/12112 "
                              "tokens of context (step_states scales slots x width)")
    p_serve.add_argument("--blocks", type=int, default=0,
                         help="KV blocks (16 tokens each); 0 = fit the pool to the card, "
                              "capped by --max-ctx. Measured 2026-09-04: 3927 blocks = "
                              "62832 tokens on a 32GB V100 at --slots 3 with a draft")
    p_serve.add_argument("--max-ctx", type=int, default=0,
                         help="cap served context (0 = the model's own limit); pairs with "
                              "--blocks so a request cannot outgrow the pool")
    p_serve.add_argument("--ssd-path", default="",
                         help="directory for the SSD prefix tier (empty = off). Unlike the "
                              "host snapshot tier this one pays without concurrent sessions: "
                              "after a restart HBM is empty, so the first lookup of every "
                              "returning conversation reaches back to disk. The tier keys its "
                              "files on the model's config, so a shape change makes them "
                              "unreadable rather than serving KV from other weights")
    p_serve.add_argument("--ssd-min-tokens", type=int, default=0,
                         help="spill floor in tokens (0 = one chunk). A GDN snapshot is a "
                              "constant ~157 MB at any prefix length, so every short "
                              "publish costs as much to spill as a long one; raising this "
                              "drops the publishes a longer prefix supersedes anyway")
    p_serve.add_argument("--max-batch", type=int, default=8,
                         help="concurrent rows; drop to 2 for a single-user endpoint (a decode "
                              "graph is captured per bucket x chain width, so a lower "
                              "ceiling is fewer captures)")
    p_serve.add_argument("--no-warmup", dest="warmup", action="store_false",
                         help="skip precapturing the decode graphs; the first real messages "
                              "then pay for them (1088 ms/token falling to 26 over six "
                              "requests on sm70)")
    p_serve.set_defaults(func=cmd_serve)

    p_train = sub.add_parser("train", help="SFT, --rl (GRPO) or --opd; --recipe for a gated flag set")
    p_train.add_argument("--model", choices=["tiny", "tiny-agent", "qwen38-27b"], default="tiny")
    p_train.add_argument("--steps", type=int, default=20)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--opd", action="store_true",
                         help="on-policy distillation: the engine rolls out, LoRA adapters train")
    p_train.add_argument("--rl", action="store_true",
                         help="GRPO: the engine samples a group per prompt, a reward scores "
                              "them, the group mean is the baseline (no critic)")
    p_train.add_argument("--group", type=int, default=8, help="rollouts per prompt (--rl)")
    p_train.add_argument("--micro", type=int, default=0,
                         help="--rl: rows per backward, gradients accumulated to one "
                              "update (0 = the whole group). The normalizer stays the "
                              "batch's, so this is the same update, not a smaller one")
    p_train.add_argument("--max-new-tokens", type=int, default=32, help="rollout length")
    p_train.add_argument("--data", help="JSONL {prompt, answer}: real prompts, exact-match "
                         "reward on the last number (scripts/gsm8k_jsonl.py)")
    p_train.add_argument("--reward", choices=["number", "boxed"], default="number",
                         help="how --data's answer is matched: last number (GSM8K) or the "
                              "last \\boxed{} (MATH, scripts/math_jsonl.py)")
    p_train.add_argument("--temperature", type=float, default=None,
                         help="rollout temperature (default: the model card's, per thinking mode)")
    p_train.add_argument("--max-think-tokens", type=int, default=0,
                         help="cap on <think> per rollout, forced closed past it; 0 = thinking off")
    p_train.add_argument("--eval-mmlu", type=int, default=0,
                         help="score N MMLU questions before and after (needs `datasets`)")
    p_train.add_argument("--load-adapter",
                         help="adapter.safetensors to copy into the LoRA tensors before "
                         "training/eval; refuses a file whose keys do not match. With "
                         "--steps 0 this re-scores a finished run's adapter")
    p_train.add_argument("--allow-short-rollouts", action="store_true",
                         help="bypass the before-eval and periodic rollout-length guards; "
                              "the smoke recipes want the "
                              "truncation, a real run almost never does")
    p_train.add_argument("--eval-max-new-tokens", type=int, default=2048,
                         help="eval generation length; independent of --max-new-tokens, "
                         "which caps the ROLLOUTS. Scoring at the training cap measures "
                         "the cap, not the policy")
    p_train.add_argument("--eval-gsm8k", help="JSONL {prompt, answer}: greedy exact-match "
                         "accuracy before and after")
    p_train.add_argument("--eval-n", type=int, default=100, help="rows of --eval-gsm8k to score")
    p_train.add_argument("--judge", action="store_true",
                         help="let the policy rank rollouts the binary reward ties "
                              "(judge.py: tests decide first, order only)")
    p_train.add_argument("--force", action="store_true",
                         help="retrain even if this run's manifest is already finished")
    p_train.add_argument("--json", action="store_true",
                         help="print the run manifest as JSON instead of step lines")
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--optim", choices=["adafactor", "iso"], default="adafactor",
                         help="full-parameter SFT optimizer; --rl/--opd train LoRA and ignore it")
    p_train.add_argument("--lora-rank", type=int, default=16)
    p_train.add_argument("--tp", type=int, default=1,
                         help="tensor-parallel width; dp is WORLD_SIZE//tp, cp is 1. "
                              "Launch under torchrun --nproc_per_node=WORLD_SIZE.")
    p_train.add_argument("--draft", help="draft head safetensors: speculative rollout (--opd)")
    p_train.add_argument("--depth", type=int, help="drafts per row per tick; default is the "
                         "head's own (chain: 2; block: its checkpoint's block minus the anchor)")
    p_train.add_argument("--recipe", choices=sorted(RECIPES),
                         help="a flag set that passed a gate (recipes.py); flags override it")
    # The recipe is the subparser's defaults, so anything typed still wins.
    p_train.set_defaults(func=cmd_train, **(flags(recipe) if recipe else {}))

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
        # stderr: --json is not known until the parser is built, and stdout is the JSON stream.
        print(f"recipe {recipe}: {RECIPES[recipe]['status']}", file=sys.stderr)
    args = _build_parser(recipe).parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
