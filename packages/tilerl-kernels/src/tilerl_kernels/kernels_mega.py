"""Single-launch dense inference program for the sm70 cell.

The instruction stream is fixed by one model. All producer/consumer edges use
one cooperative grid. The CPU interpreter is the small-model parity oracle.
"""

import tilelang
import tilelang.language as T

from .kernels_linear import _FP4_TWIDDLE_SRC_F16

# opcode, rows, columns, reduction, input, output, weight, scale, oscale, aux, aux2, mode
EMBED, NORM, LINEAR, SILU, ADD, ROPE, ATTENTION, GDN = range(8)


# SOTA copy: kernels_linear.py fp16 twiddle decoder; FP8 widens to the old BF16 face.
_FP8_SRC = r"""
__device__ __forceinline__ float tl_mega_fp8(unsigned char b, float scale) {
  int e = (b >> 3) & 15, m = b & 7;
  float x = e ? __uint_as_float(((unsigned)(e + 120) << 23) | ((unsigned)m << 20)) : m * 0.001953125f;
  if (e == 15 && m == 7) return __int_as_float(0x7fffffff);
  x = (b & 128) ? -x : x;
  x *= scale;
  unsigned u = __float_as_uint(x);
  u += 0x7fffu + ((u >> 16) & 1u);
  return __uint_as_float(u & 0xffff0000u);
}
"""


def make_megakernel(blocks: int, instructions: int, pointer_count: int, steps: int = 1,
                    min_blocks: int = 0):
    @tilelang.jit(target={"kind": "cuda", "arch": "sm_70"})
    def compile_program():
        @T.prim_func
        def megakernel(
            Program: T.Tensor((instructions, 12), "int64"),
            Pointers: T.Tensor((pointer_count,), "uint64"),
            Position: T.Tensor((1,), "int32"),
        ):
            with T.Kernel(blocks, threads=128) as bx:
                T.import_source(_FP4_TWIDDLE_SRC_F16 + _FP8_SRC)
                if min_blocks:
                    # Trades GDN's 128-float local array for registers: the grid
                    # cannot grow past what the cooperative launch keeps resident.
                    T.annotate_min_blocks_per_sm(min_blocks)
                tx = T.get_thread_binding(0)
                shared = T.alloc_shared((128,), "float32")
                qshare = T.alloc_shared((256,), "float32")
                kshare = T.alloc_shared((256,), "float32")
                vshare = T.alloc_shared((256,), "float32")
                scalars = T.alloc_shared((8,), "float32")
                acc = T.alloc_local((1,), "float32")
                sc = T.alloc_local((1,), "float32")
                state = T.alloc_local((128,), "float32")
                for token_step in T.serial(steps):
                    for ip in T.serial(instructions):
                        op = T.cast(Program[ip, 0], "int32")
                        rows = T.cast(Program[ip, 1], "int32")
                        cols = T.cast(Program[ip, 2], "int32")
                        red = T.cast(Program[ip, 3], "int32")
                        mode = T.cast(Program[ip, 11], "int32")
                        X = T.make_tensor(Pointers[Program[ip, 4]], (rows * T.max(cols, red),), "float32")
                        Y = T.make_tensor(Pointers[Program[ip, 5]], (rows * cols * T.if_then_else(op == 6, T.cast(Program[ip, 10], "int32"), 1),), "float32")
                        W = T.make_tensor(Pointers[Program[ip, 6]], (cols * T.max(red, 1),), "float32")
                        if op == 0:
                            Ids = T.make_tensor(Pointers[Program[ip, 4]], (rows * steps,), "int32")
                            Emb = T.make_tensor(Pointers[Program[ip, 6]], (red * cols,), "float16")
                            for tile in T.serial(T.ceildiv(rows * cols, blocks * 128)):
                                i = (tile * blocks + bx) * 128 + tx
                                if i < rows * cols:
                                    Y[i] = T.cast(Emb[Ids[(i // cols) * steps + token_step] * cols + i % cols], "float32")
                        if op == 1:
                            for wave in T.serial(T.ceildiv(rows, blocks)):
                                r = wave * blocks + bx
                                if r < rows:
                                    acc[0] = 0.0
                                    for j in T.serial(T.ceildiv(cols, 128)):
                                        c = j * 128 + tx
                                        if c < cols:
                                            acc[0] += X[r * cols + c] * X[r * cols + c]
                                    shared[tx] = acc[0]
                                    T.sync_threads()
                                    for level in T.unroll(7):
                                        if tx < (64 >> level):
                                            shared[tx] += shared[tx + (64 >> level)]
                                        T.sync_threads()
                                    for j in T.serial(T.ceildiv(cols, 128)):
                                        c = j * 128 + tx
                                        if c < cols:
                                            val = X[r * cols + c] * T.rsqrt(shared[0] / cols + 1e-6) * W[c]
                                            if mode == 1:
                                                Y[r * cols + c] = T.cast(T.cast(val, "float16"), "float32")
                                            else:
                                                Y[r * cols + c] = val
                                    T.sync_threads()
                        if op == 2:
                            WQ = T.make_tensor(Pointers[Program[ip, 6]], (cols * (red // 2),), "uint8")
                            Scale = T.make_tensor(Pointers[Program[ip, 7]], (cols * (red // T.max(mode, 1)),), "float16")
                            OScale = T.make_tensor(Pointers[Program[ip, 8]], (cols,), "float32")
                            XH = T.make_tensor(Pointers[Program[ip, 9]], (rows * red,), "float16")
                            if mode > 0:
                                for tile in T.serial(T.ceildiv(rows * red, blocks * 128)):
                                    i = (tile * blocks + bx) * 128 + tx
                                    if i < rows * red:
                                        XH[i] = T.cast(X[i], "float16")
                            T.sync_grid()
                            for wave in T.serial(T.ceildiv(rows * cols, blocks * 4)):
                                out_idx = (wave * blocks + bx) * 4 + tx // 32
                                rr = out_idx // cols
                                nn = out_idx % cols
                                lane = tx % 32
                                acc[0] = 0.0
                                if rr < rows:
                                    if mode > 0:
                                        for ki in T.serial(red // 512):
                                            kk = ki * 512 + lane * 16
                                            sc[0] = T.cast(Scale[nn * (red // mode) + kk // mode], "float32")
                                            T.call_extern(
                                                "tl_fp4_gemv_tiles_f16_m_xh<1,1>",
                                                T.access_ptr(WQ[nn * (red // 2) + kk // 2], "r"),
                                                T.access_ptr(XH[rr * red + kk], "r"), red, 512,
                                                T.access_ptr(sc, "r"), T.access_ptr(acc, "rw"), dtype="void")
                                    else:
                                        W8 = T.make_tensor(Pointers[Program[ip, 6]], (cols * red,), "uint8")
                                        WS = T.make_tensor(Pointers[Program[ip, 7]], (T.ceildiv(cols, 128) * T.ceildiv(red, 128),), "float32")
                                        for ki in T.serial(T.ceildiv(red, 32)):
                                            kk = ki * 32 + lane
                                            if kk < red:
                                                if mode == -1:
                                                    weight_scale = WS[(nn // 128) * T.ceildiv(red, 128) + kk // 128] * OScale[nn]
                                                    val = T.call_extern("tl_mega_fp8", W8[nn * red + kk], weight_scale, dtype="float32")
                                                    acc[0] += X[rr * red + kk] * val
                                                else:
                                                    acc[0] += X[rr * red + kk] * W[nn * red + kk]
                                    T.call_extern("tl_warp_reduce_m_f16<1>", T.access_ptr(acc, "rw"), dtype="void")
                                    if lane == 0:
                                        Y[out_idx] = acc[0] * T.if_then_else(mode > 0, OScale[nn], 1.0)
                        if op == 3 or op == 4:
                            A = T.make_tensor(Pointers[Program[ip, 9]], (rows * cols,), "float32")
                            for tile in T.serial(T.ceildiv(rows * cols, blocks * 128)):
                                i = (tile * blocks + bx) * 128 + tx
                                if i < rows * cols:
                                    if op == 3:
                                        Y[i] = X[i] * T.sigmoid(X[i]) * A[i]
                                    else:
                                        Y[i] = X[i] + A[i]
                        if op == 5:
                            # rows are batch*heads; input and output have the same head layout.
                            for wave in T.serial(T.ceildiv(rows, blocks)):
                                r = wave * blocks + bx
                                if r < rows:
                                    acc[0] = 0.0
                                    for j in T.serial(T.ceildiv(cols, 128)):
                                        c = j * 128 + tx
                                        if c < cols:
                                            acc[0] += X[r * cols + c] * X[r * cols + c]
                                    shared[tx] = acc[0]
                                    T.sync_threads()
                                    for level in T.unroll(7):
                                        if tx < (64 >> level):
                                            shared[tx] += shared[tx + (64 >> level)]
                                        T.sync_threads()
                                    for j in T.serial(T.ceildiv(cols, 128)):
                                        c = j * 128 + tx
                                        if c < cols:
                                            qshare[c] = X[r * cols + c] * T.rsqrt(shared[0] / cols + 1e-6) * W[c]
                                    T.sync_threads()
                                    for j in T.serial(T.ceildiv(cols, 128)):
                                        c = j * 128 + tx
                                        if c < cols:
                                            if c < red:
                                                half = red // 2
                                                angle = T.cast(Position[0] + token_step, "float32") * T.pow(1e7, -2.0 * (c % half) / red)
                                                other = qshare[(c + half) % red]
                                                Y[r * cols + c] = qshare[c] * T.cos(angle) + T.if_then_else(c < half, -other, other) * T.sin(angle)
                                            else:
                                                Y[r * cols + c] = qshare[c]
                                    T.sync_threads()
                        if op == 6:
                            # W stores [batch,context,kv_head,dimension,2]; aux stores head-interleaved gates.
                            KV = T.make_tensor(Pointers[Program[ip, 6]], (rows * red * mode * cols * 2,), "float32")
                            Q = T.make_tensor(Pointers[Program[ip, 4]], (rows * T.cast(Program[ip, 10], "int32") * cols,), "float32")
                            Gate = T.make_tensor(Pointers[Program[ip, 9]], (rows * T.cast(Program[ip, 10], "int32") * cols,), "float32")
                            Heads = T.cast(Program[ip, 10], "int32")
                            for wave in T.serial(T.ceildiv(rows * Heads, blocks)):
                                hidx = wave * blocks + bx
                                if hidx < rows * Heads:
                                    bb = hidx // Heads
                                    hh = hidx % Heads
                                    kh = hh * mode // Heads
                                    for j in T.serial(T.ceildiv(cols, 128)):
                                        c = j * 128 + tx
                                        if c < cols:
                                            qshare[c] = Q[(bb * Heads + hh) * cols + c]
                                            vshare[c] = 0.0
                                    if tx == 0:
                                        scalars[0] = -3.402823466e38
                                        scalars[1] = 0.0
                                    T.sync_threads()
                                    for pos in T.serial(Position[0] + token_step + 1):
                                        acc[0] = 0.0
                                        for j in T.serial(T.ceildiv(cols, 128)):
                                            c = j * 128 + tx
                                            if c < cols:
                                                acc[0] += qshare[c] * KV[(((bb * red + pos) * mode + kh) * cols + c) * 2]
                                        shared[tx] = acc[0]
                                        T.sync_threads()
                                        for level in T.unroll(7):
                                            if tx < (64 >> level):
                                                shared[tx] += shared[tx + (64 >> level)]
                                            T.sync_threads()
                                        if tx == 0:
                                            score = shared[0] * T.rsqrt(T.cast(cols, "float32"))
                                            top = T.max(scalars[0], score)
                                            scalars[2] = T.exp(scalars[0] - top)
                                            scalars[3] = T.exp(score - top)
                                            scalars[1] = scalars[1] * scalars[2] + scalars[3]
                                            scalars[0] = top
                                        T.sync_threads()
                                        for j in T.serial(T.ceildiv(cols, 128)):
                                            c = j * 128 + tx
                                            if c < cols:
                                                vshare[c] = vshare[c] * scalars[2] + KV[(((bb * red + pos) * mode + kh) * cols + c) * 2 + 1] * scalars[3]
                                        T.sync_threads()
                                    for j in T.serial(T.ceildiv(cols, 128)):
                                        c = j * 128 + tx
                                        if c < cols:
                                            Y[hidx * cols + c] = vshare[c] / scalars[1] * T.sigmoid(Gate[(bb * Heads + hh) * cols + c])
                                    T.sync_threads()
                        if op == 7:
                            # The state column belongs to one lane for the entire recurrence.
                            heads = cols // 128
                            qkv_width = 2 * red + cols
                            QKV = T.make_tensor(Pointers[Program[ip, 4]], (rows * (2 * red + cols),), "float32")
                            Conv = T.make_tensor(Pointers[Program[ip, 6]], ((2 * red + cols) * 4,), "float32")
                            Norm = T.make_tensor(Pointers[Program[ip, 7]], (128,), "float32")
                            Dt = T.make_tensor(Pointers[Program[ip, 8]], (cols // 128,), "float32")
                            Alog = T.make_tensor(Pointers[Program[ip, 9]], (cols // 128,), "float32")
                            ptr = Program[ip, 10]
                            Z = T.make_tensor(Pointers[ptr], (rows * cols,), "float32")
                            AB = T.make_tensor(Pointers[ptr + 1], (rows * 2 * (cols // 128),), "float32")
                            States = T.make_tensor(Pointers[ptr + 2], (rows * (cols // 128) * 128 * 128,), "float32")
                            Windows = T.make_tensor(Pointers[ptr + 3], (rows * 2 * 3 * (2 * red + cols),), "float32")
                            par = (Position[0] + token_step) % 2
                            for wave in T.serial(T.ceildiv(rows * heads, blocks)):
                                index = wave * blocks + bx
                                if index < rows * heads:
                                    bb = index // heads
                                    vh = index % heads
                                    kh = vh * (red // 128) // heads
                                    qc = kh * 128 + tx
                                    kc = red + qc
                                    vc = 2 * red + vh * 128 + tx
                                    for part in T.unroll(3):
                                        cc = T.if_then_else(part == 0, qc, T.if_then_else(part == 1, kc, vc))
                                        acc[0] = QKV[bb * qkv_width + cc] * Conv[cc * 4 + 3]
                                        for tap in T.unroll(3):
                                            acc[0] += Windows[((bb * 2 + par) * 3 + tap) * qkv_width + cc] * Conv[cc * 4 + tap]
                                        if part == 0:
                                            qshare[tx] = acc[0] * T.sigmoid(acc[0])
                                        elif part == 1:
                                            kshare[tx] = acc[0] * T.sigmoid(acc[0])
                                        else:
                                            vshare[tx] = acc[0] * T.sigmoid(acc[0])
                                    T.sync_threads()
                                    if tx == 0:
                                        scalars[0] = 0.0
                                        scalars[1] = 0.0
                                        for j in T.serial(128):
                                            scalars[0] += qshare[j] * qshare[j]
                                            scalars[1] += kshare[j] * kshare[j]
                                        scalars[0] = T.rsqrt(scalars[0] + 1e-12) * T.rsqrt(128.0)
                                        scalars[1] = T.rsqrt(scalars[1] + 1e-12)
                                        g = AB[bb * 2 * heads + vh] + Dt[vh]
                                        softplus = T.if_then_else(g > 20.0, g, T.log(1.0 + T.exp(g)))
                                        scalars[2] = T.exp(-T.exp(Alog[vh]) * softplus)
                                        scalars[3] = T.sigmoid(AB[bb * 2 * heads + heads + vh])
                                    T.sync_threads()
                                    qshare[tx] *= scalars[0]
                                    kshare[tx] *= scalars[1]
                                    T.sync_threads()
                                    acc[0] = 0.0
                                    for j in T.serial(128):
                                        state[j] = States[((bb * heads + vh) * 128 + j) * 128 + tx] * scalars[2]
                                        acc[0] += state[j] * kshare[j]
                                    delta = (vshare[tx] - acc[0]) * scalars[3]
                                    acc[0] = 0.0
                                    for j in T.serial(128):
                                        state[j] += delta * kshare[j]
                                        acc[0] += state[j] * qshare[j]
                                        States[((bb * heads + vh) * 128 + j) * 128 + tx] = state[j]
                                    shared[tx] = acc[0]
                                    T.sync_threads()
                                    if tx == 0:
                                        scalars[4] = 0.0
                                        for j in T.serial(128):
                                            scalars[4] += shared[j] * shared[j]
                                        scalars[4] = T.rsqrt(scalars[4] / 128.0 + 1e-6)
                                    T.sync_threads()
                                    z = Z[bb * cols + vh * 128 + tx]
                                    Y[bb * cols + vh * 128 + tx] = shared[tx] * scalars[4] * Norm[tx] * z * T.sigmoid(z)
                                    for tap in T.unroll(3):
                                        Windows[((bb * 2 + 1 - par) * 3 + tap) * qkv_width + vc] = T.if_then_else(tap < 2, Windows[((bb * 2 + par) * 3 + T.min(tap + 1, 2)) * qkv_width + vc], QKV[bb * qkv_width + vc])
                                        if vh % (heads // (red // 128)) == 0:
                                            Windows[((bb * 2 + 1 - par) * 3 + tap) * qkv_width + qc] = T.if_then_else(tap < 2, Windows[((bb * 2 + par) * 3 + T.min(tap + 1, 2)) * qkv_width + qc], QKV[bb * qkv_width + qc])
                                            Windows[((bb * 2 + 1 - par) * 3 + tap) * qkv_width + kc] = T.if_then_else(tap < 2, Windows[((bb * 2 + par) * 3 + T.min(tap + 1, 2)) * qkv_width + kc], QKV[bb * qkv_width + kc])
                                    T.sync_threads()
                        if op == 8:
                            Source = T.make_tensor(Pointers[Program[ip, 4]], (rows * red,), "float32")
                            offset = T.cast(Program[ip, 10], "int32")
                            for tile in T.serial(T.ceildiv(rows * cols, blocks * 128)):
                                i = (tile * blocks + bx) * 128 + tx
                                if i < rows * cols:
                                    c = i % cols
                                    source_c = T.if_then_else(mode > 0, (c // mode) * (2 * mode) + c % mode + offset, c + offset)
                                    Y[i] = Source[(i // cols) * red + source_c]
                        if op == 9:
                            Cache = T.make_tensor(Pointers[Program[ip, 5]], (rows * red * cols * 2,), "float32")
                            for tile in T.serial(T.ceildiv(rows * cols, blocks * 128)):
                                i = (tile * blocks + bx) * 128 + tx
                                if i < rows * cols:
                                    Cache[(((i // cols) * red + Position[0] + token_step) * cols + i % cols) * 2 + mode] = X[i]
                        T.sync_grid()
        return megakernel
    return compile_program()
