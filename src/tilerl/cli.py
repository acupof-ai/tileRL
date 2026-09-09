"""tilerl CLI. Heavy imports live inside the handlers so ``--help`` stays instant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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

#: Every name `_build_model` builds. The single source for the argparse `choices` below, so
#: a new model cannot be added to one and missed in the other.
MODEL_NAMES = ("tiny", "tiny-agent", "qwen38-27b")


def _progress(as_json: bool):
    """The run's progress printer: stdout normally, STDERR under --json.

    Not a no-op under --json, which is what it used to be. A `--json` eval-only run then
    printed nothing at all until the manifest at the end, and with the eval arm writing
    `eval-{tag}.jsonl` only on completion and the run directory holding just the
    pre-written manifest, the log was the sole progress signal -- so a healthy 67-minute
    27B eval was indistinguishable from a hung one for its whole duration, and two
    sessions began treating one as dead (errors/2026-09-08-a-silence-with-no-writer.md).

    stderr, not stdout, because `--json` exists so a caller can parse the manifest:
    `tests/test_recipes.py:47` reads `out[out.index("{"):]`, so anything containing a
    brace ahead of the manifest breaks it. stdout is already not a lone object -- the
    same test records that TileLang writes kernel-cache warnings there from C++ -- which
    is why the manifest is found by first brace rather than by parsing the whole stream,
    and why moving OUR lines off stdout is what keeps that workable.
    """
    if not as_json:
        return print
    return lambda *a, **k: print(*a, **{**k, "file": sys.stderr, "flush": True})


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

    # The fall-through below used to accept anything: a typo (`qwen38_27b`), a different
    # capitalization, or a checkpoint PATH all returned a random 64-hidden 2-layer tiny
    # without raising, and the run finished with a table that reads like the 27B. Six
    # scripts pass a user-supplied `--model` here with no argparse `choices`, so the
    # refusal belongs at this seam rather than in each of them.
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"unknown model {model_name!r}; expected one of {', '.join(MODEL_NAMES)}. "
            f"A local 27B checkpoint is selected with --model qwen38-27b plus "
            f"TILERL_QWEN38_SOURCE=<dir>, not by passing its path here"
        )
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
                  blocks=0, max_ctx=0, max_batch=8, ssd_path="", ssd_min_tokens=0,
                  dram_bytes=0, state_bytes=0, kv_fp8="", decode=None):
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
    if dram_bytes:
        kw["dram_bytes"] = dram_bytes
    if state_bytes:
        kw["state_bytes"] = state_bytes
    if kv_fp8:
        import torch

        kw["kv_fp8"] = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[kv_fp8]
    # Text stop sequences are matched on decoded ids, so the engine needs the
    # tokenizer's decode; without it `submit` refuses a request that carries one.
    if decode is not None:
        kw["decode"] = decode
    if not devices:
        return engine_mod.build_engine(cfg, model, backend, **kw)

    from tilerl_kernels.backend import Backend, resolve_target

    from .parallel import DataParallelEngine

    def make(d, **kwargs):
        # One Backend per replica: it binds the current CUDA device, so building it here is
        # what puts each replica's pools on its own card.
        b = Backend(resolve_target())
        return engine_mod.build_engine(cfg, model, b, **kwargs)

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
    # Before the engine: it takes the decode for stop sequences.
    tokenizer = _qwen38_tokenizer() if args.model == "qwen38-27b" else get_tokenizer(None)
    engine = _build_engine(cfg, model, backend, devices=args.devices,
                           draft=draft, depth=args.depth, slots=args.slots,
                           blocks=args.blocks, max_ctx=args.max_ctx,
                           max_batch=args.max_batch, ssd_path=args.ssd_path,
                           ssd_min_tokens=args.ssd_min_tokens, dram_bytes=args.dram_bytes,
                           state_bytes=args.state_bytes, kv_fp8=args.kv_fp8,
                           decode=tokenizer.decode)

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

    log = _progress(args.json)
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


def _eval_row_appender(run_id: str, tag: str):
    """Append scored rows to eval-<tag>.jsonl as they land: a killed eval arm keeps
    what finished -- the MATH before-arm died at 1h40m with zero rows on disk,
    because the write happened only after the whole arm (errors/2026-09-09-the-
    killed-eval-arm-kept-nothing.md). One open/close per row, so a kill loses at
    most the row in flight.

    Coverage is the GSM8K arm and the curve points: gsm8k_accuracy streams rows
    through on_row. The MMLU arm still lands whole -- mmlu_accuracy has no
    on_row -- so a kill mid-MMLU still loses that arm (about 12% of before/after
    wall time)."""
    from .ledger import runs_root

    path = Path(runs_root()) / run_id / f"eval-{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    def append(row: dict) -> None:
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    return append


def _read_eval_rows(run_id: str, tag: str) -> list:
    """Per-problem rows of one eval arm, or [] if the arm was not written."""
    from .ledger import runs_root

    f = Path(runs_root()) / run_id / f"eval-{tag}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.is_file() else []


def _mcnemar(before: list, after: list, dataset: str = "gsm8k") -> dict | None:
    """Paired significance on the per-question rows both arms wrote, or None.

    The comparison IS paired: `cli.py` scores one `eval_rows` list in both arms and
    `gsm8k_accuracy` forces temperature 0, so question i is the same question on both
    sides. Keeping only the two totals threw that away and left the unpaired interval,
    whose 80%-power one-sided MDE at n=500 is 7.70 pt -- ABOVE the roadmap's +5 pt
    target, so a real effect at the standard would have failed to register. The paired
    SE is `sqrt(b + c) / n` over the discordant counts, 1.00-1.41 pt at a 5-10% flip
    rate, which puts +5 pt at 3.5-5 SE instead.

    None when the arms cannot be paired -- different lengths, a missing `i`, or no
    discordant pairs at all. `b + c == 0` is not a failure: it means the two arms
    agreed on every question, so there is nothing for a paired test to resolve.
    """
    def by_i(rows):
        out = {}
        for r in rows:
            if r.get("dataset", dataset) == dataset and "i" in r:
                out[r["i"]] = bool(r["correct"])
        return out

    lo, hi = by_i(before), by_i(after)
    if not lo or lo.keys() != hi.keys():
        return None
    b = sum(1 for i in lo if lo[i] and not hi[i])   # was right, now wrong
    c = sum(1 for i in lo if not lo[i] and hi[i])   # was wrong, now right
    n = len(lo)
    if b + c == 0:
        return {"n": n, "b": b, "c": c, "delta": 0.0, "se": None, "z": None}
    se = (b + c) ** 0.5 / n
    return {"n": n, "b": b, "c": c, "delta": (c - b) / n, "se": se,
            "z": ((c - b) / n) / se}


def _paired_delta(run_dir: Path) -> dict | None:
    """`_mcnemar` over the two arms' `eval-{before,after}.jsonl`, or None if either is absent.

    Reads the record rather than in-memory state so it works on the cache-hit path too:
    a cached before-arm goes through `_write_eval_rows` like a fresh one, so the file
    exists either way.
    """
    def rows(tag):
        f = run_dir / f"eval-{tag}.jsonl"
        if not f.is_file():
            return None
        return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]

    before, after = rows("before"), rows("after")
    if before is None or after is None:
        return None
    return _mcnemar(before, after)


#: the before-arm's mean completion must leave headroom under the rollout cap. 0.8
#: rather than 1.0 because the MEAN fitting exactly means half the rollouts do not.
_ROLLOUT_HEADROOM = 0.8

#: How many eval prompts the before/after/curve arms submit at once. A constant because
#: the KV pool is sized for the WIDER of this and the rollout group -- one engine, two
#: consumers. It was three literal 8s while the rollout width was also 8, so --group 2
#: sized the pool for 2 rows and the eval arm's 8 exhausted it mid-step.
_EVAL_CONCURRENCY = 8


def _curve_rows(eval_rows: list, n: int, seed: int) -> list:
    """The curve subset: a fixed-seed shuffle's first ``n`` rows, not the file's.

    ``gsm8k_test.jsonl`` is ordered -- its first 200 rows run 5 pt low (z=3.05,
    errors/2026-09-04-the-eval-cap-measured-itself.md) -- and a 5 pt bias is the
    size of the effect the curve measures. The seed is fixed across runs so a
    curve point is paired across steps and comparable to the historical anchor.
    """
    pool = list(eval_rows)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def _write_rollout_rows(run_id: str, rows: list, written: int = 0) -> int:
    """Append the rows not yet on disk, and return the new count.

    One JSON row per completion, so length and reward stay paired. Run 2's mechanism
    claim -- short rollouts score better, so the policy lengthens -- was made from two
    group MEANS per step, which is a cross-step correlation confounded by prompt
    difficulty; each step draws a different prompt
    (wins/2026-09-06-what-a-length-term-can-recover.md). Nothing could have been
    re-derived from that run because the pairing never reached disk.

    Per step rather than once at the end, because nothing in this package handles a
    signal: run 2 took a SIGTERM at step 45 and never reached any writer. A run killed
    that way now loses at most the step in flight.
    """
    from .ledger import runs_root

    if len(rows) <= written:
        return written
    d = Path(runs_root()) / run_id
    d.mkdir(parents=True, exist_ok=True)
    with (d / "rollouts.jsonl").open("a" if written else "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows[written:])
    return len(rows)


def _length_aware(match, gold, tok, lam: float, cap: int):
    """The RL reward: correctness minus ``lam`` times the completion's fraction of the cap.

    A correctness-only reward is indifferent between two right answers of any two lengths, so
    an all-right group ties at zero advantage and produces no gradient -- run 2 collapsed that
    way at step 41 of 100 (errors/2026-09-06-the-rollouts-grew-into-the-cap.md).

    Here and NOT in `match`: `MATCHERS` also feeds `gsm8k_accuracy`, whose count becomes
    `manifest["metrics"]["gsm8k_*"]`, the number P1's exit criterion reads. A length term in
    the matcher contract would sit inside the gate.

    `lam` is a switch, not a dial. In an all-right group it cancels exactly -- the advantage
    divides by the group std, so `-(L_i - Lbar)/std(L)` has no `lam` in it -- and in a mixed
    group `lam <= cap/(cap-1)` keeps a short wrong answer from outranking a long right one
    (2048/2047 = 1.000488520, measured).

    An all-wrong group is the third case: every match is 0, so `r_i = -lam * L_i / cap` and
    the normalized advantage is again `-(L_i - Lbar) / std(L)` -- `lam` cancels for every
    lam > 0, so its magnitude does not tune this gradient and only lam = 0 turns it off (zero
    reward spread, and `group_advantages` zeroes a tied group). The difference is what the
    gradient says: length is the only signal, the shortest wrong answer gets the highest
    advantage, and on a problem the model cannot solve it learns "answer shorter" and nothing
    else. That buys seconds_per_step and cannot outrank a right answer (the bound above still
    holds). What neither covers is the direction itself: this pressure points at empty
    outputs, and `_refuse_short_rollouts` only reads the BASE policy's length before training
    -- `--allow-short-rollouts` disables it and the in-loop drift check, and it cannot see a
    policy that shortens mid-training -- while the `live` mask in `group_advantages` only
    keeps an empty row from polluting its group's normalization, not the live rows' gradient
    toward shorter. A 2026-09-09 run collapsed to all-empty outputs (GSM8K 0/500) with both
    guards in place; that run had lam=0, so it is not this gradient's doing, but it shows the
    path is reachable on this model.
    """
    def reward(prompt, completion):
        text = tok.decode([int(t) for t in completion])
        r = float(match(text, gold[tuple(int(t) for t in prompt)]))
        return r - lam * (len(completion) / cap)

    return reward


def _within_group_r(rows: list) -> float | None:
    """Pearson r of (tokens, reward) POOLED over within-group deviations.

    Centering per group is what removes prompt difficulty: a hard prompt shifts
    both its lengths and its rewards, and that shift is the confound. A tied group
    contributes zero deviation in reward and so cannot move r -- which is correct,
    it carries no signal, and it is also why r is None on a run where every group
    tied.

    Recorded, never gated: the consumer is a person reading a finished P1 run, not code.
    The sign says whether the length term is doing what run 2's diagnosis said it would,
    and that reading needs the number together with the run's context.
    """
    import collections

    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["step"]].append((r["tokens"], r["reward"]))
    dx: list[float] = []
    dy: list[float] = []
    for g in groups.values():
        if len(g) < 2:
            continue
        mx = sum(t for t, _ in g) / len(g)
        my = sum(v for _, v in g) / len(g)
        dx.extend(t - mx for t, _ in g)
        dy.extend(v - my for _, v in g)
    sxx = sum(a * a for a in dx)
    syy = sum(b * b for b in dy)
    if sxx <= 0 or syy <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sxx * syy) ** 0.5


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
    from .ledger import (
        EarlyStop,
        commit,
        curve_churn,
        file_hash,
        new_best_point,
        new_manifest,
        paired_se,
        read_manifest,
        require_paired_width,
        runs_root,
        significant_decline,
        write_manifest,
    )
    from .model import add_lora
    from .prompt import render_chat, sampling
    from .tokenizer import get_tokenizer

    real = args.model == "qwen38-27b"
    log = _progress(args.json)
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
        "group": args.group, "prompts_per_step": args.prompts_per_step,
        "max_new_tokens": args.max_new_tokens,
        "allow_short_rollouts": args.allow_short_rollouts,
        "temperature": params.temperature, "max_think_tokens": args.max_think_tokens,
        "lr": args.lr, "lora_rank": args.lora_rank, "seed": args.seed, "eval_mmlu": args.eval_mmlu,
        # In the id: tp=1 and tp=4 are different runs, and without this the second
        # would be handed the first's finished manifest and never train.
        "tp": args.tp,
        "reward": args.reward,
        # In the id: it changes what the reward MEANS, so two runs differing only here are
        # not the same run and the second must not be handed the first's manifest. Stays a
        # float although the help calls it a switch -- narrowing the type would change every
        # already-recorded id and orphan those runs' manifests.
        "length_penalty": args.length_penalty,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "load_adapter": file_hash(args.load_adapter) if args.load_adapter else None,
        "eval_gsm8k": file_hash(args.eval_gsm8k) if args.eval_gsm8k else None,
        "eval_n": args.eval_n,
        # In the id: it selects which problems the curve scores, so two runs differing
        # only here are not the same run.
        "eval_curve_seed": args.eval_curve_seed,
        # In the id: they decide which problems the curve scores, how dense it is, and
        # where the run stops. A patience on/off pair sharing an id would hand the
        # second run the first's finished manifest -- silent, and the pair is the
        # evidence for the default-flip decision.
        "eval_every": args.eval_every, "eval_curve_n": args.eval_curve_n,
        "patience": args.patience, "patience_mode": args.patience_mode})
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

    # 1024 floor = the old flat 512 blocks. max_total_tokens only guards one request and
    # costs no memory, so it never drops below the 8192 default.
    # Hand-computed, so `_fit_blocks` never runs here: passing num_blocks truthy is what
    # skips it (engine.py:1645), and it is the only path that measures free memory instead
    # of deriving a pool from context. That is deliberate for now -- training also holds
    # gradients, the tape and the optimizer state, which `_fit_blocks` does not model, so
    # its two-thirds rule is calibrated for serve. Whether training should use it is a
    # card-pending question, not an oversight.
    #
    # Two consumers, each priced on its OWN rows and its OWN length, then max(). Crossing
    # the axes instead -- widest rows x longest sequence -- costs `--group 16` twice the
    # blocks the eval needs, and over-allocation here is not slack: this path also holds the
    # gradients, the tape and the optimizer state.
    #
    # #320 took the max on the ROW axis alone and left per-row length at the rollout's cap,
    # which still exhausted at the default --eval-max-new-tokens 2048 (measured: `--group 8`,
    # 520 blocks, needs 1099). Its arms passed only because `max(2, 8)` handed the narrow
    # group 4x the rows it used, absorbing the length shortfall on the wrong axis.
    rollout_ctx = max(map(len, prompts)) + args.max_new_tokens + 64
    # 515: MMLU's longest rendered prompt, the figure the 1024 floor was chosen for. GSM8K's
    # 183 sits under any floor, so the MMLU arm is the only eval prompt that can exceed the
    # training prompts -- and this function never sees either.
    eval_ctx = max(max(map(len, prompts)), 515 if args.eval_mmlu else 0) \
        + args.eval_max_new_tokens + 64
    # Sized from --group, not a literal 8: grpo_loop submits the whole group at once
    # (train.py, one submit per g), so a group wider than the engine runs in waves and
    # every rollout in the second wave decodes at a batch the tensor core underfills.
    # The three used to be 8 while --group was a settable flag defaulting to 8, so
    # --group 16 quietly became two waves of 8.
    # A step is --prompts-per-step groups, all submitted at once, so the rollout's width is
    # their product, not --group. At the default 1 this is `max(args.group, 1)` exactly.
    rollout_batch = max(args.group, 1) * max(args.prompts_per_step, 1)
    # min(), not _EVAL_CONCURRENCY: the eval arms ask for _EVAL_CONCURRENCY rows but only
    # num_slots of them hold blocks at once, since a submit past the slots queues inside the
    # engine. Measured on the discriminating case -- `--group 4`, 520 blocks, eval cap 1500:
    # 4 rows need 384 and pass, 8 would need 768 -- and `tilerl-0a` predicted the group-16
    # exhaustion (1033 blocks) from the same model before `tilerl-48` hit it.
    eval_rows_in_flight = min(rollout_batch, _EVAL_CONCURRENCY)
    blocks = max(rollout_batch * -(-rollout_ctx // BLOCK_TOKENS),
                 eval_rows_in_flight * -(-eval_ctx // BLOCK_TOKENS)) + 8
    ctx = max(rollout_ctx, eval_ctx, 1024)
    engine = build_engine(cfg, model, backend, num_slots=rollout_batch,
                          max_batch=rollout_batch, draft=draft,
                          num_blocks=blocks,
                          max_total_tokens=max(ctx, 8192),
                          spec_depth=args.depth,
                          decode_graph=not args.deterministic,
                          prefix_store=NoPrefixStore())
    # Not in `inputs`: the id is a hash of it, so recording the pool there would make
    # every pool change a different run and hand nothing back on a rerun. It is beside
    # `metrics` because it is a property of the run, and read off the built engine
    # because the kwargs and the pool disagree (max_blocks clamps, the graph adds a row).
    manifest["engine"] = engine.config
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
        # Timed on BOTH paths, so the cache's payoff is a recorded number rather than an
        # argument: a hit writes ~0 s here and a miss writes what the arm cost, and the
        # difference is what wins/2026-09-05-before-eval-cache.md has owed since it landed
        # `pending-remote` -- 55 lines of mechanism plus 129 of test is worth it at 15 min
        # per hit and is not at 40 s.
        t_eval = time.perf_counter()
        if tag == "before" and cache is not None and cache.is_file():
            saved = json.loads(cache.read_text())
            manifest["metrics"].update(saved["metrics"])
            _write_eval_rows(manifest["id"], tag, saved["rows"])
            mean_len[tag] = saved["mean_len"]
            manifest["eval_before_cache"]["cache_hit"] = True
            manifest["metrics"][f"eval_{tag}_secs"] = time.perf_counter() - t_eval
            log(f"eval before: cache hit {cache.stem}")
            return
        rows_out: list = []
        append = _eval_row_appender(manifest["id"], tag)
        if args.eval_mmlu:
            # Per-arm, because `eval_{tag}_secs` is the SUM of both arms and no historical run
            # can be decomposed into them -- not even by subtraction, since the gsm8k arm was
            # never timed either. MMLU is prefill-dominated (1000 questions x ~515 prompt
            # tokens, 1 token generated), so its cost does not follow from any decode figure.
            t_mmlu = time.perf_counter()
            c, n, conc = mmlu_accuracy(engine, tok, args.eval_mmlu, concurrency=_EVAL_CONCURRENCY,
                                       questions=mmlu_set, per_problem=rows_out)
            for r in rows_out:
                append(r)  # mmlu first, then the gsm8k stream: same order the cache replays
            manifest["metrics"][f"mmlu_{tag}_secs"] = time.perf_counter() - t_mmlu
            manifest["metrics"][f"mmlu_{tag}"] = c / n
            manifest["metrics"][f"mmlu_{tag}_concurrency"] = conc
            manifest["metrics"][f"mmlu_{tag}_correct"] = c
            manifest["metrics"][f"mmlu_{tag}_total"] = n
            log(f"mmlu 0-shot {c}/{n} = {100 * c / n:.1f}% (seed 0, concurrency {conc}) "
                f"in {manifest['metrics'][f'mmlu_{tag}_secs']:.1f}s")
        if eval_rows:
            gsm_rows: list = []
            t_gsm = time.perf_counter()
            c, n, ntok = gsm8k_accuracy(engine, tok, eval_rows, eval_params, concurrency=_EVAL_CONCURRENCY,
                                        thinking=thinking,
                                        match=MATCHERS[args.reward],
                                        per_problem=gsm_rows,
                                        on_row=lambda r: append(dict(r, dataset="gsm8k")))
            manifest["metrics"][f"gsm8k_{tag}_secs"] = time.perf_counter() - t_gsm
            mean_len[tag] = sum(r["tokens"] for r in gsm_rows) / max(1, len(gsm_rows))
            rows_out.extend(dict(r, dataset="gsm8k") for r in gsm_rows)
            manifest["metrics"][f"gsm8k_{tag}"] = c
            manifest["metrics"][f"gsm8k_{tag}_tokens"] = ntok
            manifest["metrics"][f"gsm8k_{tag}_total"] = n
            # tokens/correct, not tokens: the ratio is what a length claim compares
            # on, and it cannot be improved by getting fewer questions right.
            per = f"  {ntok} tokens ({ntok / c:.1f}/correct)" if c else f"  {ntok} tokens"
            log(f"gsm8k greedy {c}/{n} = {100 * c / n:.1f}%{per}")
        # rows_out (mmlu + gsm8k, prompt order) feeds the before-arm cache payload;
        # the file itself was streamed above, mmlu rows in-block and gsm8k per row.
        # Read before the cache write so a hit's cost excludes the write only a miss pays,
        # but stored after it, because a duration is not a cacheable result: it belongs to
        # the run that paid it. Inside the payload it would replay a past cost onto a hit
        # -- 0.74 s where 0.0013 s was spent -- and `_secs` matches the `_before` filter.
        elapsed = time.perf_counter() - t_eval
        if tag == "before" and cache is not None:
            # `_secs` excluded, not just `eval_before_secs` by ordering: a duration belongs to
            # the run that paid it, and the per-arm timings added beside the scores DO match
            # the `_before` filter, so caching them would replay a miss's minutes onto a hit.
            saved = {"metrics": {k: v for k, v in manifest["metrics"].items()
                                 if "_before" in k and not k.endswith("_secs")},
                     "rows": rows_out, "mean_len": mean_len.get(tag)}
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=cache.parent, delete=False) as f:
                json.dump(saved, f)
            os.replace(f.name, cache)
        manifest["metrics"][f"eval_{tag}_secs"] = elapsed

    drift = {"name": "rollouts_within_cap", "value": None,
             "threshold": _ROLLOUT_HEADROOM * args.max_new_tokens,
             # Pre-seeded rather than built in `_finish`, so it carries its own `kind`:
             # a gate without one would read as `verdict` to any consumer that defaults.
             "kind": "validity",
             "skipped": True, "passed": None}
    if args.rl:
        manifest["gates"].append(drift)
    # Before the eval arms, and for BOTH algos: `write_manifest` otherwise runs only
    # inside `_finish`, so a run killed anywhere earlier left no manifest and
    # `tilerl ledger` could not see it. Measured on cpu: SIGTERM at step 6 of a grpo
    # run left 12 rollout rows and no manifest; the same kill on an opd run left no
    # run DIRECTORY at all. `_finish` overwrites this with the finished manifest.
    write_manifest(runs_root(), manifest)
    evals("before")  # LoRA B is zero at init: the base model's score
    if args.steps == 0:
        evals("after")
        manifest["engine"] = engine.config  # re-read: see the comment at the other _finish
        return _finish(manifest, args.json)
    _refuse_short_rollouts(mean_len.get("before"), args.max_new_tokens,
                           args.allow_short_rollouts)
    # The weights behind the best curve point. Every intermediate policy is otherwise
    # destroyed: `AdamW.step_one` ends in `p.copy_()` (in place, which is what lets the
    # engine keep its captured graphs), so a run that peaks mid-way can neither stop there
    # nor roll back to it. Measured 2026-09-08: score 87.4 -> 93.2 -> 93.4 -> 82.4 -> 91.2,
    # so the run shipped 91.2 and the 93.4 it had reached was gone. Out here, not in the RL
    # branch, because the save site below is shared with opd.
    best: dict = {}
    if args.rl:
        if rows:
            gold = {tuple(p): r["answer"] for p, r in zip(prompts, rows)}
            reward = _length_aware(MATCHERS[args.reward], gold, tok, args.length_penalty,
                                   max(int(args.max_new_tokens), 1))
        else:
            # No length term: this reward is a RATE, so its expectation does not grow with
            # length and the defect above is absent by construction -- a longer completion
            # earns no more, so nothing pressures the policy to lengthen. True even if this
            # path stops being the smoke-test one.
            half = cfg.vocab_size // 2

            def reward(prompt, completion):
                return sum(1 for t in completion if t < half) / max(len(completion), 1)

        tiebreak = _judge_tiebreak(engine, tok, params) if args.judge else None

        # `steps_to_score x seconds_per_step` needs the step at which a score was crossed,
        # and gsm8k_before/after cannot say which step that was. So: score a fixed held-out
        # subset every `--eval-every` steps and keep (step, score, cumulative_secs).
        # cumulative_secs is summed rather than `secs_per_step_median x step` because the
        # step length is not constant within a run -- it changes once a run hits the rollout
        # cap. The threshold stays at the READING end: the curve records the scores, and
        # which one counts as "the score" is not the ledger's business.
        #
        # `secs` is TRAINING time and excludes this scoring: grpo_loop stops its clock
        # at `train.py:496`, before the yield, so the probe's own cost is outside every
        # point. That is the quantity `time_to_score` wants -- production does not pay the
        # probe -- but it means the curve's last point is `secs_total`, not wall clock.
        # Scoring at the yield is also the only correct place: grpo_loop calls
        # `invalidate_weights()` before yielding (`:492`), so the eval sees the policy the
        # step just produced, with the decode graph already dropped.
        # Sliced from `eval_rows`, which `--eval-n` has already capped, so asking for more
        # curve rows than eval rows quietly scores fewer. `curve["n"]` records the real
        # size, but a reader looking at the run WHILE it happens sees only this line.
        curve_rows = _curve_rows(eval_rows, args.eval_curve_n, args.eval_curve_seed)
        if curve_rows and args.eval_every and len(curve_rows) < args.eval_curve_n:
            log(f"curve subset is {len(curve_rows)} rows, not the {args.eval_curve_n} asked "
                f"for: --eval-n {args.eval_n} caps it")
        # Early stopping is a verdict about the curve: with no curve points the switch can
        # never fire, so refuse at startup instead of running to --steps behind dead code.
        if args.patience and not (args.eval_every and curve_rows):
            sys.exit("--patience early-stops on the eval curve, but this run produces no "
                     "curve points: pass --eval-every and an eval set (--eval-curve-n rows).")
        curve: list[dict] = []
        # `train_secs`, not `elapsed`: the timings loop below rebinds `elapsed` on every
        # step, so an accumulator by that name silently became the last timing value --
        # measured, the curve read 0.143 s at step 4 against 0.148 at step 2, a
        # cumulative figure going DOWN.
        train_secs = 0.0
        # Patience is in curve POINTS, not steps: one unit is one `--eval-every` interval.
        # 0 never stops -- the default; flipping it on needs the seed-1 verdict.
        early = EarlyStop(args.patience)

        def score_curve(step: int) -> bool:
            nonlocal best
            # `eval_secs` per point, so the "keep the scoring under 5% of a step" criterion
            # is a fact checkable AFTER a run rather than a guess before one. Estimating it
            # from another config's eval would extrapolate across n, generation length and
            # batch shape -- and an estimated default is harder to overturn than no default,
            # because it looks calibrated. Same idiom as `eval_{tag}_secs` (#309).
            t_eval = time.perf_counter()
            # `per_problem` for the LENGTHS, not just the count. A score is not
            # interpretable without them: the same 60% can be a policy answering in 300
            # tokens or one being cut off, and 2026-09-04 shipped a 39.0% that was the
            # cap's number rather than the policy's. `at_cap` is the reading that
            # distinguishes them, so it travels with every point.
            per: list = []
            append = _eval_row_appender(manifest["id"], f"curve-{step}")
            c, n, ntok = gsm8k_accuracy(engine, tok, curve_rows, eval_params,
                                        concurrency=_EVAL_CONCURRENCY, thinking=thinking,
                                        match=MATCHERS[args.reward], per_problem=per,
                                        on_row=lambda r: append(dict(r, dataset="gsm8k")))
            eval_secs = time.perf_counter() - t_eval
            at_cap = sum(p["tokens"] >= args.eval_max_new_tokens for p in per)
            # The rows stream to disk as they land (on_row above), because the whole
            # point of the curve is comparing its points to each other and that
            # comparison is PAIRED: every point scores the same `curve_rows`. Unpaired,
            # adjacent points carry a 1.90 pt difference SE at n=500; paired at 5%
            # discordant it is 1.00 pt, and "has it stopped rising" is exactly a
            # question about a difference smaller than the arms. P1 fell back to the
            # unpaired interval for want of these rows, and `per` was being built here
            # and dropped. Streaming also means a killed run keeps the points that
            # finished.
            # Churn vs the previous point: the run's own instrument reading, recorded per
            # point so a "these two points differ by N questions" claim has N's measurement
            # beside it. Zero new evals -- these rows were just written and the previous
            # point's are in the same run dir. First point: null, not 0.
            churn = churn_dir = None
            if curve:
                prev_rows = _read_eval_rows(manifest["id"], f"curve-{curve[-1]['step']}")
                pair = curve_churn(prev_rows, per)
                if pair is None:
                    log(f"  curve step {step}: churn null -- {len(prev_rows)} rows at step "
                        f"{curve[-1]['step']} vs {len(per)} now, not comparable")
                else:
                    churn, churn_dir = pair[0] + pair[1], list(pair)
            # The first point compiles the eval's shapes and every later one hits the cache,
            # so its eval_secs is 5.6x the steady state and --eval-curve-n is calibrated off
            # point two -- recorded, because the curve is a list of equal-looking dicts.
            #
            # `tied` is the RUN's tie fraction over the steps since the previous point, not
            # anything about the eval. It is the only way to tell a plateau where the policy
            # stopped improving from one where its groups stopped disagreeing: score flat
            # with tied rising is the usable set self-consuming, both flat is another cause.
            # A whole-run mean cannot separate them -- the 2026-09-05 P1 run went 0.50 ->
            # 0.87 across its own steps while reward rose with it, so the two are confounded
            # in any single aggregate.
            since = [h[3] for h in hist[curve[-1]["step"] if curve else 0:]]
            curve.append({"step": step, "correct": c, "total": n, "score": c / max(n, 1),
                          "secs": round(train_secs, 3), "eval_secs": round(eval_secs, 3),
                          "mean_len": round(ntok / max(n, 1), 1), "at_cap": at_cap,
                          "tied": round(statistics.mean(since), 4) if since else None,
                          "churn": churn, "churn_dir": churn_dir,
                          "jit": not curve})
            # SIGNIFICANTLY greater, not merely greater. Measured 2026-09-08: re-scoring
            # one fixed set of weights across processes at temperature 0.0 moved 438/500
            # to 437/500, so this eval's own floor is 0.2 pt -- and the run's step-50 point
            # led step 25 by exactly one question. Taking the numerically higher point
            # would have bought 501.2 s of extra training for a reading inside the
            # instrument. A tie goes to the earlier point, which is not an arbitrary
            # tie-break: `time_to_score` is the objective, so when two options are the same
            # score the cheaper one wins, and "the same" is defined by the measured floor.
            # The criterion lives in `ledger.new_best_point` next to the SE formulas, so
            # the run and its post-hoc readers cannot drift apart -- its `__main__` check
            # runs this exact curve, plus the one-question case, the paired-vs-unpaired
            # case, and each one's negative control.
            #
            # The width is PAIRED: every curve point scores the same `curve_rows`, and the
            # best point's per-problem rows are on disk from when it was scored. The
            # unpaired width is 1.9x wider here (8.6% discordant, measured 2026-09-08), so
            # it would make this criterion never fire on a slow rise -- a selection that
            # always keeps the first point and does not say so. Rows missing (old runs)
            # fall back to the conservative width, marked in `se_kind` so nobody reads a
            # conservative "not greater" as "the two points are the same". With --patience
            # on, a missing width refuses instead: stopping on a width-less curve decides
            # without a sampling width, and the unpaired fallback is too wide to ever fire -- both silent.
            if best:
                rows = _read_eval_rows(manifest["id"], f"curve-{best['step']}")
                se = paired_se(rows, per)
                se_kind = "paired" if se is not None else "unpaired (conservative)"
                require_paired_width(se, args.patience, best["step"])
            else:
                se = se_kind = None  # first point: no comparison installed it
            replaced = new_best_point(curve[-1], best or None, se, mode=args.patience_mode)
            if replaced:
                # `mean_len` and `tok_per_correct` ride with the snapshot so a downstream
                # consumer can trade score against answer cost -- the 2026-09-05 run bought
                # most of its +6% as 2.74x shorter answers, and score alone cannot show that.
                best = {"step": step, "score": curve[-1]["score"],
                        "mean_len": curve[-1]["mean_len"],
                        "tok_per_correct": round(ntok / c, 1) if c else None,
                        "se_kind": se_kind,
                        "tensors": {k: v.detach().to("cpu", copy=True)
                                    for k, v in trainable.items()}}
            # A significant decline stops immediately and spends no patience: the collapse
            # is the event this feature exists for, and stopping never loses anything --
            # the best snapshot is kept either way. seed 0's recovery (412 -> 456) still
            # ended below the peak (467), so waiting for it bought less than keeping it.
            reason = early.update(replaced, significant_decline(curve[-1], best, se))
            if reason:
                manifest["early_stopped"] = {"at_step": step, "kept_step": best["step"],
                                             "patience": args.patience, "reason": reason}
                why = ("a significant decline" if reason == "decline"
                       else f"{args.patience} curve points without a significant gain")
                log(f"  early stop ({reason}) at step {step}: {why}; keeping step "
                    f"{best['step']} ({100 * best['score']:.1f}%)")
                return True
            log(f"  curve step {step}: {c}/{n} = {100 * c / max(n, 1):.1f}% "
                f"tied {curve[-1]['tied']} "
                f"at {train_secs:.1f}s cumulative, mean {ntok / max(n, 1):.0f} tok, "
                # tokens/correct, the ratio the before/after arms already log (`per`, :718).
                # It separates two things a score cannot: the 2026-09-05 run moved
                # tokens/correct 394.0 -> 143.8 (2.74x) while accuracy moved 88.0 -> 93.6
                # (+6%), so most of what that RL bought was shorter answers. A curve read on
                # score alone records that as "the rate of learning to be right".
                # Derived, not stored: it is mean_len * total / correct from fields already
                # in the point, and a second copy in the dict could disagree with them.
                f"{ntok / c if c else float('nan'):.1f} tok/correct, "
                f"{at_cap}/{n} at cap, scored in {eval_secs:.1f}s")
            return False

        hist = []
        rollouts: list = []
        written = 0
        for i, (r, ce, secs, tied, ntok, timings, width) in enumerate(
                train_mod.grpo_loop(engine, model, prompts, reward, args.steps, backend, optimizer,
                                    group=args.group, prompts_per_step=args.prompts_per_step,
                                    sampling=params, seed=args.seed,
                                    trainable=trainable, micro=args.micro,
                                    tiebreak=tiebreak, recapture_graph=True,
                                    per_rollout=rollouts, decode=tok.decode)):
            hist.append((r, ce, secs, tied, ntok))
            train_secs += secs
            written = _write_rollout_rows(manifest["id"], rollouts, written)
            if (curve_rows and args.eval_every and (i + 1) % args.eval_every == 0
                    and score_curve(i + 1)):
                break
            for phase, elapsed in timings.items():
                manifest["metrics"][phase] = manifest["metrics"].get(phase, 0.0) + elapsed
            log(f"step {i + 1:4d}/{args.steps}  reward {r:.4f}  ce {ce:.4f}  "
                f"tied {tied:.2f}  tok {ntok:.0f}  width {width}  {secs:.1f}s  "
                f"rollout {timings['rollout_secs']:.3f}s  "
                # .get: rl_step writes these, and a test or caller that substitutes it
                # still gets a log line rather than a KeyError mid-run.
                f"fwd {timings.get('forward_secs', 0.0):.3f}s  "
                f"bwd {timings.get('backward_only_secs', 0.0):.3f}s  "
                f"optimizer {timings['optimizer_secs']:.6f}s  "
                f"other {timings.get('other_secs', 0.0):.3f}s", flush=True)
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
        manifest["metrics"]["length_reward_r"] = _within_group_r(rollouts)
        # Its own top-level key, not a metric: `format_run` prints every metric inline on
        # one `tilerl ledger` row, so a curve in there would push the row past a screen.
        if curve:
            manifest["eval_curve"] = {"n": len(curve_rows), "every": args.eval_every,
                                      "points": curve}
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
    # `best_curve_point`, never `best_step`: the snapshot is taken inside `score_curve`, so
    # its resolution is `--eval-every`. At 25 a true peak at 40 is recorded as 50. This is
    # the best point we LOOKED AT, and a name promising the best step would be read as an
    # optimum.
    if args.rl and best:
        save_file({k: v.contiguous() for k, v in best.pop("tensors").items()},
                  str(d / "adapter-best.safetensors"))
        manifest["artifacts"]["adapter_best"] = "adapter-best.safetensors"
        manifest["best_curve_point"] = {**best, "every": args.eval_every}
        log(f"adapter-best step {best['step']} score {100 * best['score']:.1f}% "
            f"(best of {len(curve)} curve points, resolution {args.eval_every} steps)")
    if drift["passed"] is not False:
        evals("after")
    else:
        # The after-arm never ran, so `mmlu_after`/`gsm8k_after` are None -- and
        # `_finish`'s `v is None or ...` would score both gates PASS on a run that
        # measured neither. Mark them skipped so the manifest says "not measured".
        manifest["gates_skip_after"] = True
    # Re-read, not the build-time copy: `_graph_for` sets `_decode_graph_on = False`
    # in its `except` on a capture failure, so a snapshot taken at build time can
    # record graph-on for a run that decoded eagerly -- and the whole point of this
    # block is that a wall clock is read against it.
    manifest["engine"] = engine.config
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

#: The two classes a gate can belong to, recorded on the gate itself.
#:
#: VERDICT answers "did P1 pass": the two exit criteria `docs/roadmap.md:57-58` states.
#: VALIDITY answers "is this run interpretable at all", and the roadmap already draws
#: that line -- the tied-group criterion reads "< 50% (else the task is too easy for
#: this model and the run says nothing)". Saying nothing is not failing.
#:
#: `reward_rises` is the reason this split matters. Reward is the quantity GRPO
#: optimizes, so it rising is the definition of the optimizer working, not evidence for
#: P1's claim that RL moves a DOWNSTREAM number -- and rising reward is fully
#: compatible with a falling eval, which is what reward hacking looks like. So it must
#: never be able to make P1 read `pass`. Reward NOT rising is informative (run 2
#: collapsed that way), and that failure is coarse enough for a zero threshold to
#: catch, which is why this needs no invented number.
_VALIDITY_GATES = frozenset({"groups_untied", "reward_rises", "ce_falls",
                             "rollouts_within_cap"})


def _finish(m: dict, as_json: bool) -> None:
    """Gate, write the manifest, print it, exit non-zero on a failed gate.
    A gate whose metric was not evaluated passes vacuously (value null)."""
    from .ledger import format_run, gates_pass, now, runs_root, write_manifest

    if not m["finished"]:
        g = m["metrics"]
        # .get, not [...]: "a gate whose metric was not evaluated passes
        # vacuously" already covers a metric set that never had the key, which
        # is what an SFT run's manifest is.
        # UNITS, and they differ 13 lines apart in the writer: `mmlu_{tag}` is a
        # FRACTION (`c / n`, :575) and `gsm8k_{tag}` is a COUNT (`c`, :588), with the
        # denominator alongside it as `gsm8k_{tag}_total` (:590). So the roadmap's two
        # exit numbers encode differently, and a threshold is meaningless without the
        # units of the quantity it thresholds.
        mmlu_floor = None if g.get("mmlu_before") is None else g["mmlu_before"] - 0.02
        # roadmap P1: "GSM8K held-out (500 q) after - before >= +5 pt (SE ~ 2 pt)". The
        # +5 is a sampling margin, not a taste -- the same-batch instrument is exact, so
        # `after > before` is a real +0.2 pt, but +1 question of 500 is a gain a
        # symmetric null passes about half the time on a different set. Derived from
        # `_total`, never hardcoded to 25: `--eval-n` is a flag and the recipe's 500 is
        # not a constant.
        gsm_total = g.get("gsm8k_after_total") or g.get("gsm8k_before_total")
        gsm_floor = (None if g.get("gsm8k_before") is None or not gsm_total
                     else g["gsm8k_before"] + 0.05 * gsm_total)
        # The paired test, RECORDED beside the threshold rather than replacing it. The
        # threshold is the roadmap's exit criterion and stays the gate; McNemar says
        # whether the observed move is resolvable at all, which the threshold cannot --
        # an unpaired read of n=500 has an 80%-power MDE of 7.70 pt, above the +5 pt the
        # gate asks for. Falls back silently to threshold-only when the per-question rows
        # are absent, which is what P1 did once already for want of them.
        paired = _paired_delta(Path(runs_root()) / m["id"])
        if paired is not None:
            m["metrics"]["gsm8k_paired"] = paired
        skipped = m["inputs"].get("steps") == 0
        after_skipped = bool(m.pop("gates_skip_after", False))
        # `ce_falls` has no threshold on the RL path: `ce_first` is written only by the
        # SFT loop (:281), never by the GRPO branch (:679-690), so the vacuous-pass rule
        # below made it report `passed` over nothing on every RL run. Not measured is the
        # honest record, and the gate stays live where the SFT path does write both.
        unmeasured = frozenset() if g.get("ce_first") is not None else frozenset({"ce_falls"})
        m["gates"] += [
            {"name": n, "value": v, "threshold": t,
             "kind": "validity" if n in _VALIDITY_GATES else "verdict",
             "skipped": skipped or n in unmeasured or (after_skipped and n in _AFTER_GATES),
             "passed": None if skipped or n in unmeasured
             or (after_skipped and n in _AFTER_GATES)
             else v is None or t is None or ok(v, t)}
            for n, v, t, ok in (
                ("reward_rises", g.get("reward_last"), g.get("reward_first"), lambda v, t: v > t),
                ("mmlu_holds", g.get("mmlu_after"), mmlu_floor, lambda v, t: v >= t),
                ("gsm8k_improves", g.get("gsm8k_after"), gsm_floor, lambda v, t: v >= t),
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
    views = [f"--{v}" for v in ("table", "readme", "regress", "questions")
             if getattr(args, v, False)]
    if views or args.suite:
        import subprocess

        script = Path(__file__).resolve().parent.parent.parent / "scripts/bench_harness.py"
        cmd = [sys.executable, str(script), *views]
        if args.suite:
            cmd += ["--suite", args.suite]
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


def _se_note(r: dict) -> str:
    """The subset's sampling width, when it is wide enough to set the answer.

    5.0 pt is P1's own target effect (`roadmap.md`), so an SE at or above it means the
    crossing step is chosen by which rows are in the subset as much as by the policy.
    Silent below that: a note on every line would be read as boilerplate and skipped.

    The width comes from the point's own rate (`ledger.time_to_score`), so this fires on
    the subset's real resolution rather than on p=0.5's worst case -- which at n=50 and
    n=100 warned about subsets that do resolve the effect.
    """
    se = r.get("se_pt")
    if se is None or se < 5.0:
        return ""
    return (f"  [subset n={r['n']}, binomial SE {se:.1f} pt >= P1's +5 pt target: the "
            f"crossing step is sampling-limited, raise --eval-curve-n to narrow it]")


def cmd_ledger(args: argparse.Namespace) -> None:
    from .ledger import format_run, lineage, list_runs, runs_root, time_to_score

    runs = lineage(runs_root(), args.lineage) if args.lineage else list_runs(runs_root())
    if args.time_to_score is not None:
        # A row per run, because the answer is per run: the target's step, the interval it
        # was crossed in, and the cumulative training seconds at that point.
        rows = [(m, time_to_score(m, args.time_to_score)) for m in runs]
        if args.json:
            print(json.dumps([{"id": m["id"], **(r or {"reached": None})}
                              for m, r in rows], indent=1))
            return
        for m, r in rows:
            if r is None:
                print(f"{m['id']}  no eval_curve (run with --eval-every N to record one)")
            elif r["reached"]:
                # The interval, not an interpolated step: the target was crossed somewhere
                # in (after_step, step] and only the right end was measured. `correct/total`
                # and the SE come along because the curve scores a SUBSET -- 0.61 on 20 rows
                # is not 0.61 on 500, and the SE is the point's own rate, not p=0.5's.
                # The dip note is printed, not folded into `reached`: a transient crossing
                # and a run that held are different facts, and suppressing the step would
                # turn one noisy point into "never reached".
                dip = ("" if r.get("held", True) else
                       f"  [fell below {args.time_to_score:.3g} again at step "
                       f"{r['dipped_at']}: a transient crossing, not arrival]")
                print(f"{m['id']}  score {args.time_to_score:.3g} reached at step "
                      f"{r['step']} (in ({r['after_step']}, {r['step']}]), "
                      f"{r['secs']:.1f}s cumulative, scored {r['score']:.3g} "
                      f"({r['correct']}/{r['total']}){dip}{_se_note(r)}")
            else:
                print(f"{m['id']}  score {args.time_to_score:.3g} NOT reached in "
                      f"{r['steps_run']} steps / {r['secs']:.1f}s; best "
                      f"{r['best']:.3g} on {r['n']} rows{_se_note(r)}")
        return
    print(json.dumps(runs, indent=1) if args.json else "\n".join(map(format_run, runs)))


def _build_parser(recipe: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tilerl",
        description="tileRL: TileLang inference + training (CPU/CUDA/Metal).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="start the OpenAI-compatible HTTP server")
    p_serve.add_argument("--model", choices=MODEL_NAMES, default="tiny")
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
    p_serve.add_argument("--state-bytes", type=int, default=0,
                         help="HBM budget in bytes for resident GDN snapshots (0 = a quarter "
                              "of free memory, the default). Without it the tiers' pressure "
                              "regime is unreachable: a quarter of free on an H20 is 17.9 GiB, "
                              "116 snapshots at 157 MiB each, so DRAM pressure starts near 115 "
                              "concurrent agent sessions and no benchable load reaches it")
    p_serve.add_argument("--dram-bytes", type=int, default=0,
                         help="host budget in bytes for demoted GDN snapshots (0 = off, the "
                              "default). Turn it on only when concurrent sessions outnumber "
                              "the snapshots HBM holds (free/4, which on a 32GB V100 is 9 at "
                              "144 MiB each): measured 2 sessions -> 0 promotions, 9 -> 17, "
                              "12 -> 24. Below that threshold it is 1.51x WORSE on wall "
                              "clock -- one conversation re-reads only its newest entry, so "
                              "the LRU snapshot a demotion picks is never asked for again "
                              "(43 demotions, 0 promotions). Read /health's dram_promotions "
                              "to see whether the workload crossed it, and dram_budget to "
                              "see the tier is on at all")
    p_serve.add_argument("--kv-fp8", choices=["e4m3", "e5m2"], default="",
                         help="store the KV planes in fp8: 65536 -> 33280 bytes per token at the "
                              "27B's 16 planes x 4 heads x 256, a 1.969x saving, the 0.031 being "
                              "one f32 scale per (plane, block, kv_head, token). Off by default: "
                              "the pool and the writers are fp8 but the attention kernels still "
                              "read a dequantized plane, so this is correctness-complete and not "
                              "yet a bandwidth win, and no decode tok/s figure exists "
                              "(docs/design-fp8-kv.md). Refused on sm70, whose fused write_tokens "
                              "has no fp8 twin and would scatter into a dequantized copy. e4m3 is "
                              "the default choice on measurement, not analogy: it beats e5m2 "
                              "1.89x on the worst element with nothing underflowing on tiny")
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
    p_train.add_argument("--model", choices=MODEL_NAMES, default="tiny")
    p_train.add_argument("--steps", type=int, default=20)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--opd", action="store_true",
                         help="on-policy distillation: the engine rolls out, LoRA adapters train")
    p_train.add_argument("--rl", action="store_true",
                         help="GRPO: the engine samples a group per prompt, a reward scores "
                              "them, the group mean is the baseline (no critic)")
    p_train.add_argument("--group", type=int, default=8, help="rollouts per prompt (--rl)")
    p_train.add_argument("--prompts-per-step", type=int, default=1,
                         help="--rl: prompts in one step, each normalised within its own "
                              "--group. A step is gradient-free only when EVERY group "
                              "ties, so 2x4 beats 1x8 at the same 8 rows: measured 1.21x "
                              "more gradient-bearing steps on MATH level 5, 1.34x on "
                              "GSM8K's 0.81 tie fraction. The step is "
                              "--prompts-per-step x --group rows, which is what the "
                              "engine is sized for.")
    p_train.add_argument("--micro", type=int, default=1,
                         help="--rl: rows per backward, gradients accumulated to one "
                              "update (0 = the whole group). The normalizer stays the "
                              "batch's, so this is the same update, not a smaller one. "
                              "Default 1 because 0 does not fit the production shape: at "
                              "group 8 / cap 2048 on one H20, 0 peaks 88.21 GiB and OOMs "
                              "on step 2, against 44.55 GiB and 3/3 steps at 1")
    p_train.add_argument("--max-new-tokens", type=int, default=32, help="rollout length")
    p_train.add_argument("--data", help="JSONL {prompt, answer}: real prompts, exact-match "
                         "reward on the last number (scripts/gsm8k_jsonl.py)")
    p_train.add_argument("--reward", choices=["number", "boxed"], default="number",
                         help="how --data's answer is matched: last number (GSM8K) or the "
                              "last \\boxed{} (MATH, scripts/math_jsonl.py)")
    p_train.add_argument("--length-penalty", type=float, default=0.1,
                         help="RL reward subtracts this times completion/cap, so an all-right "
                              "group is ordered by length instead of tying at zero advantage. "
                              "A switch, not a dial: it CANCELS exactly in an all-right group "
                              "(the advantage divides by the group std), and above "
                              "cap/(cap-1) a short wrong answer can outrank a long right one. "
                              "Set 0 to disable")
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
    p_train.add_argument("--deterministic", action="store_true",
                         help="decode eager (no captured graph): bitwise reproducible "
                              "rollouts across processes, at a rollout throughput cost")
    p_train.add_argument("--eval-max-new-tokens", type=int, default=2048,
                         help="eval generation length; independent of --max-new-tokens, "
                         "which caps the ROLLOUTS. Scoring at the training cap measures "
                         "the cap, not the policy")
    p_train.add_argument("--eval-gsm8k", help="JSONL {prompt, answer}: greedy exact-match "
                         "accuracy before and after")
    p_train.add_argument("--eval-n", type=int, default=100, help="rows of --eval-gsm8k to score")
    # 0 = off, so no existing invocation changes. The subset is a fixed-seed shuffle
    # (_curve_rows): the file is ordered, so a prefix would run 5 pt low -- a bias the
    # size of the effect the curve measures. One seed for every run, so a curve point is
    # paired across steps and comparable across runs to the historical anchor.
    p_train.add_argument("--eval-every", type=int, default=0,
                         help="score the held-out curve subset every N steps (0 = off)")
    p_train.add_argument("--patience", type=int, default=0,
                         help="early-stop after N curve POINTS without a significant score "
                              "gain -- a point is --eval-every steps, not one -- or "
                              "immediately on a significant decline; 0 = never stop")
    p_train.add_argument("--patience-mode", choices=["significant", "raw"], default="significant",
                         help="ruler for --patience: significant (2xSE paired width) or raw "
                              "(strict raw-score gain, for small curve subsets where the "
                              "width exceeds the signal -- the three risks are documented "
                              "in ledger.new_best_point)")
    p_train.add_argument("--eval-curve-n", type=int, default=20,
                         help="rows of --eval-gsm8k in the curve subset; keep the scoring "
                              "under 5%% of a step")
    p_train.add_argument("--eval-curve-seed", type=int, default=0,
                         help="shuffle seed selecting the curve subset; fixed across runs "
                              "so points stay paired and comparable")
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
    for v in ("table", "readme", "regress", "questions"):
        p_bench.add_argument(f"--{v}", action="store_true",
                             help=f"bench view: {v} from the bench store, no GPU")
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
    p_ledger.add_argument("--time-to-score", type=float, metavar="SCORE",
                          help="read time_to_score off each run's eval_curve: the step that "
                               "first scored >= SCORE, the interval it was crossed in, and "
                               "the cumulative training seconds there")
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
