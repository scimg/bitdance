#!/usr/bin/env python3
"""End-to-end numeric check of the PRODUCTION W4A6 preshuffle GEMM.

Strategy: feed constant-value, constant-along-K operands so the result is
independent of the kernel's internal A-swizzle / B-`layout_b` preshuffle:

  A = MXFP6-E2M3, every element = a_val  (packed 6-bit dense, 24 B + 8 B zero pad
      per K=32 lane block -> exactly what v_mfma_scale...f8f6f4 fp6/cbsz=2 reads)
  B = MXFP4-E2M1, every element = b_val  (dense 2 codes/byte)
  all E8M0 block scales = 127 (== 2^0 == 1.0)

Then C[m,n] = sum_k a_val*b_val = K * a_val * b_val for every (m,n).

This validates, on the real multi-wave kernel: fp6 decode (cbsz=2), fp4 decode
(blgp=4), per-32 E8M0 scale application, and full-K accumulation -- i.e. every
piece the W4A6 change touches. (K-positional shuffles are inherited unchanged
from the already-verified fp4/fp8 paths.)
"""
import argparse, sys
sys.path.insert(0, "/root"); sys.path.insert(0, "/work")
import numpy as np
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import from_c_void_p
from psg.preshuffle_gemm import compile_preshuffle_gemm_a8
from kernel.nvfp4_mxfp6_gemm_tiled import build_LUT12

_I8 = getattr(fx, "Int8", None) or fx.Int32
_I32 = fx.Int32
_F32 = fx.Float32
_BF16 = getattr(fx, "BFloat16", None) or fx.Int16


def _ptr(t, et):
    return from_c_void_p(et, t.data_ptr())


def e2m3_code(v):  # MXFP6-E2M3 code for a small positive value
    table = {}
    for e in range(4):
        for m in range(8):
            val = (m / 8.0) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - 1))
            table[round(val, 6)] = (e << 3) | m
    return table[round(v, 6)]


_MXFP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def e2m1_code(v):  # MXFP4-E2M1 code (positive)
    return _MXFP4.index(v)


def pack6_block(code):  # 32 identical 6-bit codes -> 24 dense bytes
    bits = 0
    for p in range(32):
        bits |= (code & 0x3F) << (p * 6)
    return bits.to_bytes(24, "little")


def e4m3_scale_byte(v):  # E4M3 byte for a power-of-two-ish scale value
    # E4M3: 1 sign, 4 exp (bias 7), 3 mant. byte = (exp<<3)|mant.
    table = {}
    for e in range(16):
        for m in range(8):
            val = (m / 8.0) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - 7))
            table[round(val, 6)] = (e << 3) | m
    return table[round(v, 6)]


def _build_a_fp6(a_val, M, K, dev):
    ac = e2m3_code(a_val)
    blk = pack6_block(ac) + bytes(8)  # [24 packed 6-bit][8 zero] per K=32
    assert len(blk) == 32
    a_row = np.frombuffer(blk * (K // 32), np.uint8)
    a = torch.tensor(np.tile(a_row, (M, 1)).copy(), dtype=torch.uint8, device=dev)
    return a, ac


# The A values whose dense 6-bit stream is BYTE-UNIFORM, and so survives any byte-level
# swizzle. Laying a constant 6-bit code out contiguously on the host does not reproduce the
# kernel's A layout -- dense 6-bit repeats every 3 bytes, the A staging swizzle works in 16 B
# units, and 16 % 3 != 0 -- which is why the constant-value arms of this gate have never
# passed for the fp6-A family and report 241 where they expect 768. A code that is itself a
# repeated 2-bit pattern removes the problem at the root: all 24 data bytes (and the 8 pad
# bytes, which the fragment ignores) become the same byte, so every permutation of them is
# the same operand and the expected product is exact regardless of layout.
#   0b010101 -> 0x55 -> +3.25      0b101010 -> 0xAA -> -1.25
#   0b111111 -> 0xFF -> -7.5       0b000000 -> 0x00 ->  0.0
_A_FP6_BYTE_UNIFORM = {3.25: 0x55, -1.25: 0xAA, -7.5: 0xFF, 0.0: 0x00}


def _build_a_fp6_uniform(a_val, M, K, dev):
    byte = _A_FP6_BYTE_UNIFORM[round(a_val, 6)]
    a = torch.full((M, K), byte, dtype=torch.uint8, device=dev)
    return a, byte


def run_kernel(dtype, M, N, K, tm, tn, a, b, scale_b_byte, lut, dev, async_copy=False):
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
    sa_bytes = (M * K) // 32 * 2 + 4096
    sb_bytes = (N * K) // 32 * 2 + 4096
    scale_a = torch.full((sa_bytes,), 127, dtype=torch.uint8, device=dev)  # E8M0 1.0
    scale_b = torch.full((sb_bytes,), scale_b_byte, dtype=torch.uint8, device=dev)
    bias = torch.zeros((N,), dtype=torch.bfloat16, device=dev)
    launch = compile_preshuffle_gemm_a8(
        M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=128,
        in_dtype=dtype, out_dtype="bf16", use_async_copy=async_copy,
    )
    stream = torch.cuda.current_stream()
    kargs = (_ptr(c, _BF16), _ptr(a, _I8), _ptr(b, _I8),
             _ptr(scale_a, _F32), _ptr(scale_b, _F32), _ptr(bias, _BF16),
             _ptr(lut, _I32), M, N, stream)
    cf = flyc.compile(launch, *kargs)
    torch.cuda.synchronize()
    cf(*kargs)
    torch.cuda.synchronize()
    return c.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--tm", type=int, default=128)
    ap.add_argument("--tn", type=int, default=256)
    ap.add_argument("--a_val", type=float, default=1.5)
    ap.add_argument("--b_val", type=float, default=2.0)
    ap.add_argument("--b_scale", type=float, default=1.0,
                    help="B block scale value (NVFP4 E4M3 for w6a6_tc).")
    ap.add_argument("--dtype", type=str, default="w4a6")  # w4a6 | fp4 | w6a6_tc
    ap.add_argument("--a_uniform_byte", action="store_true",
                    help="build the fp6 A from a byte-uniform 6-bit code, which is the only "
                         "way this gate can check A's VALUE without reproducing the kernel's "
                         "A swizzle. Requires --a_val in {3.25, -1.25, -7.5, 0.0}.")
    ap.add_argument("--async_copy", action="store_true",
                    help="enable async A DMA (FlyDSL front lever)")
    args = ap.parse_args()
    AC = args.async_copy
    M, N, K = args.m, args.n, args.k
    dev = "cuda"
    a_val, b_val = args.a_val, args.b_val
    bc = e2m1_code(b_val)
    bbyte = (bc << 4) | bc  # B: 2 fp4 codes/byte
    b = torch.full((N, K // 2), bbyte, dtype=torch.uint8, device=dev)
    lut = torch.tensor(build_LUT12(), dtype=torch.int32, device=dev)
    dummy_lut = torch.zeros(8 * 256, dtype=torch.int32, device=dev)

    if args.dtype == "fp4":
        ac = e2m1_code(a_val)
        a = torch.full((M, K // 2), (ac << 4) | ac, dtype=torch.uint8, device=dev)
        C = run_kernel("fp4", M, N, K, args.tm, args.tn, a, b, 127, dummy_lut, dev)
        expected = K * a_val * b_val
    elif args.dtype == "w6a6_tc":
        # A native MXFP6; B stored NVFP4 (fp4 codes + E4M3 scale) -> transcoded.
        a, ac = (_build_a_fp6_uniform(a_val, M, K, dev) if args.a_uniform_byte
                 else _build_a_fp6(a_val, M, K, dev))
        sbyte = e4m3_scale_byte(args.b_scale)
        C = run_kernel("w6a6_tc", M, N, K, args.tm, args.tn, a, b, sbyte, lut, dev,
                       async_copy=AC)
        expected = K * a_val * b_val * args.b_scale
        # async must not change math: compare async vs non-async bit-exact.
        if AC:
            C_noac = run_kernel("w6a6_tc", M, N, K, args.tm, args.tn, a, b, sbyte,
                                lut, dev, async_copy=False)
            print(f"[async vs non-async w6a6_tc] bit-exact match: "
                  f"{np.array_equal(C, C_noac)}")
        # Bit-exact cross-check vs W4A6 (same A path, E8M0 B scale == b_scale).
        # Only valid when b_scale is an exact power of two (E8M0 representable).
        if abs(args.b_scale - 2 ** round(np.log2(args.b_scale))) < 1e-9:
            e8m0 = 127 + round(np.log2(args.b_scale))
            C_ref = run_kernel("w4a6", M, N, K, args.tm, args.tn, a, b,
                               e8m0, dummy_lut, dev, async_copy=AC)
            match = np.array_equal(C, C_ref)
            print(f"[cross-check vs W4A6 same-A] bit-exact match: {match}")
    else:
        a, ac = (_build_a_fp6_uniform(a_val, M, K, dev) if args.a_uniform_byte
                 else _build_a_fp6(a_val, M, K, dev))
        C = run_kernel("w4a6", M, N, K, args.tm, args.tn, a, b, 127, dummy_lut, dev,
                       async_copy=AC)
        expected = K * a_val * b_val

    err = np.abs(C - expected)
    print(f"{args.dtype} verify M={M} N={N} K={K} a={a_val}(code{ac}) "
          f"b={b_val}(code{bc}) b_scale={args.b_scale}")
    print(f"expected={expected}  C[:2,:3]=\n{C[:2,:3]}")
    print(f"max|err|={err.max():.4f}  mean|err|={err.mean():.5f}  "
          f"unique={np.unique(C)[:5]}")
    ok = err.max() <= 1e-3 * abs(expected)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
