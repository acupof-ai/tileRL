"""fp4 twiddle layout: the sm70 fp16 twin must decode bit-exact to the e2m1 LUT,
and both twiddles must roundtrip. The decode is simulated in numpy (the kernel's
C extern is the same shift/mask/mul.f16x2 sequence), so a wrong _TW_POS_F16 or
slot map fails here before it can silently corrupt 27B decode."""

import numpy as np
import torch
from tilerl_kernels import reference
from tilerl_kernels.reference import (
    _TW_POS_F16,
    _TW_SLOT_ELEM,
    _fp4_codes,
    twiddle_fp4,
    twiddle_fp4_f16,
    untwiddle_fp4,
    untwiddle_fp4_f16,
)

_MASK_TARGET = np.uint32(0x8E008E00)
_MASK_FIELD = np.uint32(0x0E000E00)
_MASK_SIGN = np.uint32(0x80008000)


def _prmt_0123(w):
    b = [(w >> (8 * i)) & 0xFF for i in range(4)]
    return sum(b[3 - i] << (8 * i) for i in range(4))


def _decode_word_f16(w):
    """Kernel's tl_fp4_decode8_f16, in numpy: 1 twiddled word -> 8 e2m1 values."""
    t = np.uint32(_prmt_0123(int(w)))
    placed = np.empty(4, dtype=np.uint32)
    placed[0] = t & _MASK_TARGET
    placed[1] = (t << np.uint32(7)) & _MASK_TARGET
    placed[2] = ((t << np.uint32(14)) & _MASK_SIGN) | ((t >> np.uint32(3)) & _MASK_FIELD)
    placed[3] = ((t << np.uint32(15)) & _MASK_SIGN) | ((t << np.uint32(4)) & _MASK_FIELD)
    return (placed.view(np.float16) * np.float16(16384.0)).astype(np.float32)


def _natural_words(n_words, seed):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(n_words, 4), dtype=np.uint8)
    words = (b[:, 0].astype(np.uint32) | (b[:, 1].astype(np.uint32) << 8)
             | (b[:, 2].astype(np.uint32) << 16) | (b[:, 3].astype(np.uint32) << 24))
    return b, words


def test_fp16_twiddle_decodes_bit_exact():
    b, words = _natural_words(50_000, 0)
    lut = reference._E2M1_LUT.numpy()
    nib = np.stack([b & 0xF, b >> 4], axis=-1).reshape(len(b), 8)
    expected = lut[nib & 7] * (1.0 - 2.0 * (nib >> 3))
    tw = twiddle_fp4_f16(torch.from_numpy(b).reshape(len(b), 4).contiguous()).numpy()
    tw_words = (tw[:, 0].astype(np.uint32) | (tw[:, 1].astype(np.uint32) << 8)
                | (tw[:, 2].astype(np.uint32) << 16) | (tw[:, 3].astype(np.uint32) << 24))
    got = np.stack([_decode_word_f16(w) for w in tw_words])
    assert np.array_equal(got, expected), f"max_err={np.abs(got - expected).max()}"


def test_twiddle_roundtrips():
    b, _ = _natural_words(10_000, 1)
    wq = torch.from_numpy(b).reshape(len(b), 4).contiguous()
    for tw, untw in ((twiddle_fp4, untwiddle_fp4), (twiddle_fp4_f16, untwiddle_fp4_f16)):
        assert torch.equal(untw(tw(wq)), wq), tw.__name__


def test_fp16_layout_constants():
    # The slot/elem map is shared with the bf16 twin; the positions differ.
    assert _TW_SLOT_ELEM == (1, 3, 5, 7, 0, 2, 4, 6)
    assert _TW_POS_F16 == ((15, 11, 10, 9), (8, 4, 3, 2), (1, 14, 13, 12), (0, 7, 6, 5))
    # _fp4_codes is low-nibble-first
    wq = torch.tensor([[0x21, 0x43, 0x65, 0x87]], dtype=torch.uint8)
    assert _fp4_codes(wq).tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]
