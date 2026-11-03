#!/usr/bin/env python3
"""OCP bit-level codec for the NVFP4 <-> MXFP6 transcode contract.

Encodes vector_quant_error's NVFP4 quantization into real byte/nibble layouts
and produces the golden MXFP6 (E2M3 + E8M0) output through the 64-entry LUT,
so a device kernel can be checked bit-for-bit.

Encodings (all little "sign in the top bit" OCP layouts):
* E4M3 scale byte (positive scales, sign=0): byte = ((exp+7) << 3) | mant_bits,
  so exp = ((byte>>3)&0xF) - 7 and mant_bits = byte & 0x7.
* E2M1 data nibble: code = (sign << 3) | mag_index, mag_index in 0..7 indexes
  the ascending magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
* E2M3 data (6-bit): code = (sign << 5) | mag5, mag5 in 0..31 indexes the
  ascending E2M3 magnitudes (mag5 == (exp_bits<<3)|mant_bits).
* E8M0 scale byte: exp + bump + 127.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vector_quant_error as vq  # noqa: E402
from kernel.transcode_lut import (CODE_LUT, BUMP_LUT, E2M1_POS, E2M3_POS,  # noqa: E402
                                  decode_e4m3_byte)

BLOCK = vq.BLOCK_SIZE


def encode_nvfp4(values: np.ndarray, block: int = BLOCK):
    """Return (scale_byte[nb] uint8, e2m1_code[nb, block] uint8, Quantized)."""
    q = vq.quantize(values, vq.E4M3, vq.E2M1, block)
    exp = q.exp.astype(np.int64)
    mant_bits = np.rint((q.mant - 1.0) * 8.0).astype(np.int64)
    # The book stores subnormals NORMALIZED, so a scale below the smallest normal arrives here
    # as a significand in [1, 2) against an exponent under -6. Re-encoding it means undoing
    # that: the byte's exponent field is 0 and its significand bits are the integer k with
    # value k * 2^-9.
    sub = exp < -6
    k = np.rint(q.mant * np.exp2(exp + 9)).astype(np.int64)
    exp_field = np.where(sub, 0, exp + 7).astype(np.int64)
    mant_bits = np.where(sub, k, mant_bits)
    # Bounds read from the book in force, not written down. Hardcoding [-6, 7] mirrored the
    # IEEE-style book that stops at 240; E4M3FN's top exponent field is a normal one, so
    # exp reaches 8 and every block clamped at the 448 ceiling -- which is precisely the
    # population the widening was for -- would have raised here. The packing below was
    # always right for exp == 8 (it produces 0x78 | mant); only the guard was stale.
    lo, hi = int(vq.E4M3.exp.min()), int(vq.E4M3.exp.max())
    if exp.min() < lo or exp.max() > hi:
        raise ValueError(f"E4M3 exp out of range: [{exp.min()},{exp.max()}] "
                         f"outside the book's [{lo},{hi}]")
    scale_byte = ((exp_field << 3) | mant_bits).astype(np.uint8)

    data = q.data.reshape(-1, block)
    sign = (data < 0).astype(np.uint8)
    mag = np.abs(data)
    mag_index = np.abs(mag[..., None] - E2M1_POS[None, :]).argmin(axis=-1).astype(np.uint8)
    e2m1_code = ((sign << 3) | mag_index).astype(np.uint8)
    return scale_byte, e2m1_code, q


def golden_mxfp6(scale_byte: np.ndarray, e2m1_code: np.ndarray):
    """Return (e8m0[nb] uint8, e2m3_code[nb, block] uint8) via the 64-entry LUT."""
    mant_bits, exp = decode_e4m3_byte(scale_byte)
    bump = BUMP_LUT[mant_bits].astype(np.int64)
    e8m0 = (exp + bump + 127).astype(np.uint8)

    sign = ((e2m1_code >> 3) & 1).astype(np.uint8)
    mag_index = (e2m1_code & 0x7).astype(np.int64)
    mag5 = CODE_LUT[mant_bits[:, None], mag_index]
    e2m3_code = ((sign << 5) | mag5).astype(np.uint8)
    return e8m0, e2m3_code


def dequant_mxfp6(e8m0: np.ndarray, e2m3_code: np.ndarray) -> np.ndarray:
    """Decode MXFP6 bytes back to float (for comparison with the emulation)."""
    scale = np.exp2(e8m0.astype(np.int64) - 127)
    sign = 1 - 2 * ((e2m3_code >> 5) & 1).astype(np.int64)
    mag5 = (e2m3_code & 0x1F).astype(np.int64)
    val = sign * E2M3_POS[mag5]
    return scale[:, None] * val


def selftest() -> int:
    """Codec round-trip must equal storage_to_mxfp6(..., 'revised')."""
    rng = np.random.default_rng(vq.SEED)
    fails = 0
    for name, gen in vq.TENSORS:
        x = gen(rng).astype(np.float64, copy=False)
        sb, cc, _ = encode_nvfp4(x)
        e8m0, e2m3 = golden_mxfp6(sb, cc)
        deq = dequant_mxfp6(e8m0, e2m3).reshape(x.shape)
        ref = vq.storage_to_mxfp6(x, vq.E4M3, "revised")
        ok = np.array_equal(deq, ref)
        print(f"[{'OK  ' if ok else 'FAIL'}] codec roundtrip {name:32s} max|diff|={float(np.max(np.abs(deq-ref))):.2e}")
        fails += 0 if ok else 1
    return fails


def write_testvec(out_dir: str, n_blocks: int = 4096, seed: int = 1234) -> None:
    """Write scales/codes + golden e8m0/e2m3 as raw uint8 bins for the HIP test."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Mix of distributions to exercise all mantissa/exponent/sign paths.
    x = np.concatenate([
        rng.normal(0.0, 120.0, n_blocks * BLOCK // 2),
        rng.uniform(-500.0, 500.0, n_blocks * BLOCK - n_blocks * BLOCK // 2),
    ]).astype(np.float64)
    rng.shuffle(x)
    sb, cc, _ = encode_nvfp4(x)
    e8m0, e2m3 = golden_mxfp6(sb, cc)
    sb.tofile(os.path.join(out_dir, "scales.bin"))
    cc.reshape(-1).tofile(os.path.join(out_dir, "codes.bin"))
    e8m0.tofile(os.path.join(out_dir, "golden_e8m0.bin"))
    e2m3.reshape(-1).tofile(os.path.join(out_dir, "golden_e2m3.bin"))
    with open(os.path.join(out_dir, "meta.txt"), "w") as fh:
        fh.write(f"{sb.size} {BLOCK}\n")
    print(f"wrote test vector: {sb.size} blocks x {BLOCK} to {out_dir}")


def verify(out_dir: str) -> int:
    """Compare device outputs (out_e8m0.bin/out_e2m3.bin) against golden."""
    g8 = np.fromfile(os.path.join(out_dir, "golden_e8m0.bin"), dtype=np.uint8)
    g6 = np.fromfile(os.path.join(out_dir, "golden_e2m3.bin"), dtype=np.uint8)
    d8 = np.fromfile(os.path.join(out_dir, "out_e8m0.bin"), dtype=np.uint8)
    d6 = np.fromfile(os.path.join(out_dir, "out_e2m3.bin"), dtype=np.uint8)
    ok8 = np.array_equal(g8, d8)
    ok6 = np.array_equal(g6, d6)
    n6 = int((g6 != d6).sum()) if g6.shape == d6.shape else -1
    print(f"[{'OK  ' if ok8 else 'FAIL'}] e8m0 scales  ({g8.size} blocks)")
    print(f"[{'OK  ' if ok6 else 'FAIL'}] e2m3 codes   ({g6.size} elems, mismatches={n6})")
    return 0 if (ok8 and ok6) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    here = os.path.dirname(os.path.abspath(__file__))
    tv = os.path.join(here, "tv")
    if cmd == "selftest":
        sys.exit(selftest())
    elif cmd == "gen":
        nb = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
        write_testvec(tv, nb)
    elif cmd == "verify":
        sys.exit(verify(tv))
    else:
        print(f"unknown command: {cmd}")
        sys.exit(2)
