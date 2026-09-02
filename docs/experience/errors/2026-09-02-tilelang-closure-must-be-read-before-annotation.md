# A tilelang closure variable must be read before it can type an annotation — 2026-09-02

## Context

Parameterizing a kernel's IO dtype — one flag, four failed compiles across two
different kernels. Every attempt died the same way:

```
NameError: name 'io_dtype' is not defined
  Scale: T.Tensor((N, K // block), io_dtype)
```

Tried in order: `dtype` (guessed a name collision with a builder-injected
symbol), `io_dtype` (same error, so not a collision), `f32: bool` with
`"float32" if f32 else "bfloat16"` (guessed the builder could evaluate a
conditional but not a bare name), and finally a duplicated literal factory.

## Root Cause

None of those guesses was the rule. `tilelang.language.eager.builder`
re-executes the kernel body via `self.ir_gen.gen(builder)(**self.tensor_args,
**kwargs)` — only the kernel's own arguments are bound in that scope. A closure
name resolves inside an annotation ONLY if the body has already read it in
ordinary Python before the annotation runs.

That is why `make_linear_fp4_gemv_sm70_m` gets away with it:

```python
tiles = "tl_fp4_gemv_tiles_f16_m_xh" if xh else "tl_fp4_gemv_tiles_f16_m"  # xh read HERE
X: T.Tensor((M, K), "float16" if xh else "float32")                        # so this works
Scale: T.Tensor((N, K // block), "float16" if sh else "float32")           # sh: NameError
```

`xh` and `sh` are the same kind of variable on adjacent lines; the only
difference is that `xh` was read one line earlier. `write_tokens` had no such
earlier read anywhere, which is why every variant of the flag failed there.

## Fix

Bind the dtype to a local before the annotations:

```python
s_dtype = "float16" if sh else "float32"
Scale: T.Tensor((N, K // block), s_dtype)
```

`write_tokens` kept its literal f32 twin — by the time the rule was understood
the duplicate existed and worked, and 20 duplicated lines in a leaf kernel is
cheaper than a re-verification round trip on the pod. Fold it back the next
time that file is touched.

## Rule

When a framework rejects a name, find out WHERE it resolves names before
guessing at WHICH names it rejects. Four attempts here re-guessed the predicate
(the name, the type, the expression form) while the actual variable was
position. One read of `builder.py:1557` — the line already in the traceback —
would have shown the binding set.

Corollary for this codebase: a working precedent that looks identical to your
failing code differs somewhere. `xh` vs `sh` differed by one line of distance,
and that was the whole answer.
