#!/usr/bin/env python3
"""Pre-port compile gate: compile the upstream corpus we copy kernels from
against our pinned tilelang, PASS/FAIL per file with the first error line.
Lazy-mode kernels compile on the symbolic call (no GPU tensors); autotune
wrappers go through .jit_impl.compile to skip the tuner. 15/15 PASS on
tilelang 0.1.13/sm90 (gdr_chunk_solve 60s, the rest 1-7s).
    CUDA_VISIBLE_DEVICES=7 TILELANG_DEFAULT_TARGET=cuda python3 scripts/pod_portcheck.py --corpus-dir scripts/_portcheck_corpus
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback

os.environ.setdefault("TILELANG_DEFAULT_TARGET", "cuda")

import tilelang
import tilelang.language as T

TARGET = os.environ["TILELANG_DEFAULT_TARGET"]


def _d_gemm(m):
    m.matmul.compile(M=64, N=64, K=64, block_M=64, block_N=64, block_K=32)


def _d_dequant_hopper(m):
    m.matmul.jit_impl.compile(
        64,
        64,
        64,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        num_bits=4,
        fast_dequant=True,
        block_M=64,
        block_N=64,
        block_K=64,
        num_stages=2,
        threads=256,
        split=1,
    )


def _d_dequant_gemv(m):
    m.dequantize_gemv.compile(
        1,
        1024,
        1024,
        T.float16,
        T.float16,
        T.float16,
        4,
        T.int8,
        "uint",
        4,
        32,
        True,
        False,
        True,
        -1,
        False,
    )


def _d_deepgemm(m):
    m.tl_gemm.compile(
        M=64,
        N=64,
        K=256,
        block_N=64,
        in_dtype=T.float8_e4m3fn,
        out_dtype=T.bfloat16,
        accum_dtype=T.float32,
    )


def _d_mha(m):
    m.flashattn.jit_impl.compile(
        1,
        4,
        128,
        128,
        False,
        block_M=64,
        block_N=64,
        num_stages=1,
        threads=128,
    )


def _d_cast(m):
    m.per_token_cast_to_fp8.compile(M=8, N=128, blk_m=8)


def _d_qwen36_decode(m):
    tilelang.compile(m.gdr_decode_gated_norm(B=1), target=TARGET)


def _d_qwen36_conv(m):
    tilelang.compile(m.gdr_decode_conv_gated_norm(B=1), target=TARGET)


def _gdr_driver(name):
    def _d(m):
        tilelang.compile(m.get_kernel(name), target=TARGET)

    return _d


_GDR_STAGES = [
    "gdr_chunk_prepare",
    "gdr_chunk_cumsum",
    "gdr_chunk_a",
    "gdr_chunk_solve",
    "gdr_chunk_recompute",
    "gdr_chunk_state",
    "gdr_chunk_o",
]

MANIFEST = [
    ("gemm/example_gemm.py", "example_gemm.py", _d_gemm),
    (
        "dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py",
        "example_dequant_gemm_bf16_fp4_hopper.py",
        _d_dequant_hopper,
    ),
    (
        "dequantize_gemm/example_dequant_gemv_fp16xint4.py",
        "example_dequant_gemv_fp16xint4.py",
        _d_dequant_gemv,
    ),
    (
        "deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py",
        "example_deepgemm_fp8_2xAcc.py",
        _d_deepgemm,
    ),
    ("flash_attention/example_mha_fwd_bshd.py", "example_mha_fwd_bshd.py", _d_mha),
    ("cast/example_per_token_cast_to_fp8.py", "example_per_token_cast_to_fp8.py", _d_cast),
    (
        "gdn/qwen36_gdr_decode_fused.py :: gdr_decode_gated_norm",
        "qwen36_gdr_decode_fused.py",
        _d_qwen36_decode,
    ),
    (
        "gdn/qwen36_gdr_decode_fused.py :: gdr_decode_conv_gated_norm",
        "qwen36_gdr_decode_fused.py",
        _d_qwen36_conv,
    ),
] + [
    (f"agent-infer/gated_delta_rule.py :: {name}", "gated_delta_rule.py", _gdr_driver(name))
    for name in _GDR_STAGES
]


def _load_module(corpus_dir, filename):
    path = os.path.join(corpus_dir, filename)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _first_line(text):
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-dir", required=True, help="dir holding the corpus copies (flat)")
    ap.add_argument("--out", default="portcheck_results.md", help="markdown results table path")
    ap.add_argument("--error-log", default="portcheck_errors.log", help="full tracebacks for FAILs")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.corpus_dir))

    print(f"tilelang {tilelang.__version__}  target={TARGET}  corpus={args.corpus_dir}\n")

    results = []
    loaded = {}
    with open(args.error_log, "w") as error_log:
        for label, filename, driver in MANIFEST:
            mod = loaded.get(filename)
            if mod is None and filename not in loaded:
                try:
                    mod = _load_module(args.corpus_dir, filename)
                except Exception as e:
                    loaded[filename] = None
                    results.append(
                        (label, "FAIL", 0.0, f"import: {type(e).__name__}: {_first_line(str(e))}")
                    )
                    print(f"FAIL  {label}\n      import: {e}")
                    error_log.write(f"=== {label} (import) ===\n{traceback.format_exc()}\n")
                    continue
                loaded[filename] = mod
            if mod is None:
                results.append((label, "FAIL", 0.0, "import failed (see prior entry)"))
                print(f"FAIL  {label}  (import failed)")
                continue

            t0 = time.time()
            try:
                driver(mod)
                status, err = "PASS", ""
            except Exception as e:
                status = "FAIL"
                err = f"{type(e).__name__}: {_first_line(str(e))}"
                error_log.write(f"=== {label} ===\n{traceback.format_exc()}\n")
            dt = time.time() - t0
            results.append((label, status, dt, err))
            print(f"{status}  {label}  ({dt:.1f}s){('  ' + err) if err else ''}")

    n_pass = sum(1 for r in results if r[1] == "PASS")
    print(f"\n{len(results)} entries: {n_pass} PASS, {len(results) - n_pass} FAIL")

    with open(args.out, "w") as f:
        f.write(f"portcheck — tilelang {tilelang.__version__}, target {TARGET}\n\n")
        f.write("| corpus file | result | time | first error line |\n")
        f.write("|---|---|---|---|\n")
        for label, status, dt, err in results:
            f.write(f"| {label} | {status} | {dt:.1f}s | {err} |\n")
    print(f"table -> {args.out}   tracebacks -> {args.error_log}")


if __name__ == "__main__":
    main()
