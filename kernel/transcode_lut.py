#!/usr/bin/env python3
"""Golden 64-entry lookup table for the NVFP4 -> MXFP6 "revised" transcode.

The fused kernel never does the rounding arithmetic at runtime. The whole
per-element transcode collapses to a tiny table because:

* The NVFP4 per-block scale is E4M3, whose mantissa has only 8 values
  (m = 1 + k/8, k in 0..7).
* The NVFP4 data code is E2M1, whose 16 signed codes have only 8 distinct
  magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
* The transcode is sign-symmetric: nearest_E2M3(-c) = -nearest_E2M3(c), so
  only the magnitude matters and the sign bit is copied straight through.

So the output E2M3 magnitude depends only on (mantissa_bits, |e2m1| index) =
8 x 8 = 64 entries. Each entry is a 5-bit E2M3 magnitude code (0..31); the
E2M3 6-bit code is (sign << 5) | magnitude. The output E8M0 scale byte is
`e4m3_exp + bump + 127`, where `bump = 1 if mantissa > 1 else 0` (i.e. any
mantissa_bits != 0) is an 8-entry per-mantissa table.

This module derives the tables from the exact codebooks in
`vector_quant_error.py` and reproduces `storage_to_mxfp6(x, E4M3, "revised")`
bit-for-bit through pure table indexing.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vector_quant_error as vq  # noqa: E402


# Positive (magnitude) codebooks, ascending.
E2M1_POS = np.array(sorted({abs(float(v)) for v in vq.E2M1}), dtype=np.float64)
E2M3_POS = np.array(sorted({abs(float(v)) for v in vq.E2M3}), dtype=np.float64)

N_MANT = 8                       # E4M3 mantissa bits: m = 1 + k/8
N_MAG = int(E2M1_POS.size)       # distinct |E2M1| magnitudes (== 8)

assert N_MAG == 8, f"expected 8 E2M1 magnitudes, got {N_MAG}"
assert E2M3_POS.size == 32, f"expected 32 E2M3 magnitudes, got {E2M3_POS.size}"


# ---- E4M3 subnormal scales ---------------------------------------------------------------- #
# A subnormal E4M3 byte has exponent field 0 and significand k/8, k in 1..7, so its value is
# k * 2^-9. THAT VALUE IS ALSO A NORMALIZED PAIR: write k = m * 2^j with m in [1, 2), and the
# scale is m * 2^(j-9) where m lands on exactly {1, 1.25, 1.5, 1.75} -- four significands the
# 8-entry mantissa table already holds. So subnormal scales need no new LUT classes and no
# wider table in the kernel; they need the byte decoded into (mantissa_bits, exponent) rather
# than read as if the leading 1 were there.
#
# Getting this wrong is expensive and quiet. Treating the significand as k/8 < 1 directly --
# which is what it literally is -- and folding with bump 0 pushes the block's elements down to
# 0.06..0.75, where E2M3's step is 0.125 and a k=1 block resolves onto three codes; that is
# worth -13 to -16 points on GPQA. Normalizing first keeps the elements in the top of E2M3's
# range where its resolution is, and costs one shift.
SUB_MANT_BITS = np.array([0, 0, 0, 4, 0, 2, 4, 6], dtype=np.uint8)   # index k = byte & 7
SUB_EXP = np.array([0, -9, -8, -8, -7, -7, -7, -7], dtype=np.int16)  # k = 0 unused (byte 0 is 0.0)


def decode_e4m3_byte(byte):
    """E4M3FN scale byte -> (mantissa_bits, exponent), normalized. Subnormals included.

    ONE definition, exported, because the emulation in kernel/qlayout.py, the golden codec in
    kernel/nvfp4_codec.py and the device kernel all have to agree about it, and the last time a
    scale-format predicate was written out twice the two sides silently built different books.
    """
    b = np.asarray(byte, dtype=np.int64)
    ebits, k = (b >> 3) & 0xF, b & 0x7
    sub = ebits == 0
    return (np.where(sub, SUB_MANT_BITS[k], k).astype(np.int64),
            np.where(sub, SUB_EXP[k], ebits - 7).astype(np.int64))


def build_luts() -> tuple[np.ndarray, np.ndarray]:
    """Return (code_lut[N_MANT, N_MAG] uint8, bump_lut[N_MANT] uint8).

    code_lut[mb, gi] is the index into E2M3_POS (== the 5-bit E2M3 magnitude
    code) of the nearest E2M3 value to `m * |d| / 2**bump`.
    """
    code_lut = np.zeros((N_MANT, N_MAG), dtype=np.uint8)
    bump_lut = np.zeros(N_MANT, dtype=np.uint8)
    for mb in range(N_MANT):
        m = 1.0 + mb / 8.0
        bump = 1 if m > 1.0 else 0
        bump_lut[mb] = bump
        for gi, mag in enumerate(E2M1_POS):
            combined = m * mag / (2.0**bump)
            # Match the emulation's nearest_codes (argmin) exactly.
            code = float(vq.nearest_codes(np.array([[combined]], dtype=np.float64), vq.E2M3)[0, 0])
            code_lut[mb, gi] = int(np.argmin(np.abs(E2M3_POS - abs(code))))
    return code_lut, bump_lut


CODE_LUT, BUMP_LUT = build_luts()


def _mant_bits(mant: np.ndarray) -> np.ndarray:
    # Total, including subnormal scales, because make_scale_book stores those NORMALIZED --
    # significand in [1, 2) against a lower exponent -- for exactly this reason.
    return np.rint((np.asarray(mant, dtype=np.float64) - 1.0) * 8.0).astype(np.int64)


def _mag_index(mag: np.ndarray) -> np.ndarray:
    """Exact index of each |E2M1| magnitude into E2M1_POS."""
    return np.abs(mag[..., None] - E2M1_POS[None, :]).argmin(axis=-1)


def transcode_revised_lut(values: np.ndarray, block: int = vq.BLOCK_SIZE) -> np.ndarray:
    """Reproduce storage_to_mxfp6(values, E4M3, 'revised') via the 64-entry LUT."""
    q = vq.quantize(values, vq.E4M3, vq.E2M1, block)
    mant_bits = _mant_bits(q.mant)                 # (nblocks,)
    bump = BUMP_LUT[mant_bits].astype(np.int64)    # (nblocks,)
    exp = q.exp.astype(np.int64)                   # (nblocks,)

    data = q.data.reshape(-1, block)               # signed E2M1 code values
    sign = np.sign(data)
    mag_idx = _mag_index(np.abs(data))             # (nblocks, block)

    e2m3_mag = E2M3_POS[CODE_LUT[mant_bits[:, None], mag_idx]]
    e2m3 = sign * e2m3_mag
    mx_scale = np.exp2((exp + bump).astype(np.float64))[:, None]
    return (mx_scale * e2m3).reshape(values.shape)


def e8m0_scale_byte(exp: np.ndarray, mant: np.ndarray) -> np.ndarray:
    """E8M0 scale byte emitted per block: e4m3_exp + bump + 127."""
    return (exp.astype(np.int64)
            + BUMP_LUT[_mant_bits(mant)].astype(np.int64) + 127).astype(np.int64)


def emit_c_header(path: str) -> None:
    flat = CODE_LUT.reshape(-1)
    lines = [
        "// Auto-generated by kernel/transcode_lut.py -- NVFP4->MXFP6 revised transcode.",
        "// CODE_LUT[mantissa_bits*8 + e2m1_magnitude_index] = 5-bit E2M3 magnitude code.",
        "// E2M3 6-bit code = (sign << 5) | magnitude. bump = (mantissa_bits != 0).",
        "//",
        "// SUBNORMAL SCALE BYTES (exponent field 0) do NOT index these two tables directly.",
        "// Their value is k * 2^-9, which is also the normalized pair (SUB_MANT_BITS[k],",
        "// SUB_EXP[k]) -- four significands the mantissa table already holds -- so decode the",
        "// byte through those first and the rest of the fold is unchanged:",
        "//   sub = (byte >> 3) == 0;  k = byte & 7;",
        "//   mant_bits = sub ? SUB_MANT_BITS[k] : (byte & 7);",
        "//   exp       = sub ? SUB_EXP[k]       : ((byte >> 3) - 7);",
        "// Reading such a byte as a normal one rescales the block by up to 8x; folding it with",
        "// its literal k/8 significand instead buries the elements at the bottom of E2M3,",
        "// where the step is 0.125 and a k=1 block resolves onto three codes.",
        "#pragma once",
        "#include <cstdint>",
        "",
        f"static constexpr int NVFP4_MXFP6_LUT_N = {flat.size};",
        f"static constexpr uint8_t NVFP4_MXFP6_CODE_LUT[{flat.size}] = {{",
        "    " + ", ".join(str(int(v)) for v in flat),
        "};",
        f"static constexpr uint8_t NVFP4_MXFP6_BUMP_LUT[{BUMP_LUT.size}] = {{",
        "    " + ", ".join(str(int(v)) for v in BUMP_LUT),
        "};",
        f"static constexpr uint8_t NVFP4_MXFP6_SUB_MANT_BITS[{SUB_MANT_BITS.size}] = {{",
        "    " + ", ".join(str(int(v)) for v in SUB_MANT_BITS),
        "};",
        f"static constexpr int8_t NVFP4_MXFP6_SUB_EXP[{SUB_EXP.size}] = {{",
        "    " + ", ".join(str(int(v)) for v in SUB_EXP),
        "};",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def emit_npz(path: str) -> None:
    np.savez(path, code_lut=CODE_LUT, bump_lut=BUMP_LUT,
             e2m1_pos=E2M1_POS, e2m3_pos=E2M3_POS)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    emit_c_header(os.path.join(here, "transcode_lut.h"))
    emit_npz(os.path.join(here, "transcode_lut.npz"))
    print("64-entry CODE_LUT (rows = mantissa_bits 0..7, cols = |E2M1| idx 0..7):")
    print(CODE_LUT)
    print("BUMP_LUT (per mantissa_bits):", BUMP_LUT.tolist())
    print("E2M1_POS:", E2M1_POS.tolist())
    print("wrote transcode_lut.h and transcode_lut.npz")
