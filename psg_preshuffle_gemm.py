"""Preshuffle GEMM kernel using the @flyc.kernel API."""

import functools
import os
from typing import Optional

# Perf-probe only (numerically invalid): when set, the NVFP4->MXFP6 transcode
# skips the LUT ds_read and feeds the raw byte index into the dense packer. This
# isolates the LUT-lookup (LDS) cost from the dense-pack (VALU) cost so we can
# tell whether a cheaper lookup (v_perm/ALU) could help the compute-bound case.
_TC_PROBE_NOLUT = os.environ.get("TC_PROBE_NOLUT") == "1"

# Perf-probe only (numerically invalid): price the three pieces of the HARDWARE transcode
# separately, so an optimisation is aimed at whichever piece the shape actually pays for.
# The hardware path costs 33 VALU per 32-code B fragment (16 decodes + 16 packed multiplies
# + 1 pack) plus ~22 for the E4M3->E8M0 scale word, and w4a6 vs w6a6_tc measures only their
# sum. Disabling one piece at a time turns that sum into an itemised bill:
#   NOMUL   drop the 16 v_pk_mul_f16 that apply the E4M3 scale mantissa (1 + mb/8)
#   NOSCALE drop the E4M3->E8M0 scale-word conversion, pass the raw word to the MFMA
#   NOCVT   drop the data transcode entirely, feed raw fp4 packs as the fp6 operand
# NOCVT should land near w4a6, which is what makes it a self-check on the whole method: it
# removes exactly the work that separates the two columns, so if it does not close the gap
# then the gap is not the transcode and every conclusion here is wrong.
_TC_PROBE_NOMUL = os.environ.get("TC_PROBE_NOMUL") == "1"
_TC_PROBE_NOSCALE = os.environ.get("TC_PROBE_NOSCALE") == "1"
_TC_PROBE_NOCVT = os.environ.get("TC_PROBE_NOCVT") == "1"
if _TC_PROBE_NOMUL or _TC_PROBE_NOSCALE or _TC_PROBE_NOCVT:
    # Loud, because a silent perf probe that survives into a reported number is how a
    # "+224% decode win" got measured in s17 (the kernel had stopped computing anything).
    import sys as _sys

    print(
        "[psg] WARNING perf probe active, OUTPUT IS NUMERICALLY WRONG: "
        f"NOMUL={int(_TC_PROBE_NOMUL)} NOSCALE={int(_TC_PROBE_NOSCALE)} "
        f"NOCVT={int(_TC_PROBE_NOCVT)}",
        file=_sys.stderr,
    )

# TC_HWCVT=1 replaces the LDS-LUT transcode with the gfx950 hardware scale-converts
# (v_cvt_scalef32_pk_f16_fp4 to decode, v_cvt_scalef32_pk32_fp6_f16 to pack). This
# deletes all 16 ds_read per B fragment and frees the 8 KB LDS LUT arena. It is NOT
# bit-exact against the LUT: both round to E2M3, but at exact ties the hardware rounds
# half-to-even while the LUT table rounds the magnitude half-down (6 of 64 scale-mantissa x
# magnitude combinations; neither is closer to the true value). The LUT stays the default.
# The domain holds 12 ties, not 6 -- these are the 6 where the two rules pick differently.
# Both are symmetric about zero, so neither leaves a mean error; the LUT's leaves only a
# small gain, which is why rel-L2 cannot separate them. Do not read this as "tie rules do
# not matter": rounding the SIGNED product instead of the magnitude is a third rule, it is
# biased in the mean, and a GEMM accumulates that as K. See ACCURACY_DEBUG.md section 5.
_TC_HWCVT = os.environ.get("TC_HWCVT") == "1"

# Perf-probe only (numerically invalid): restore the pre-fix A addressing for the
# byte-per-element A dtypes (w4a6/w6a6_tc), which sized the A buffer and advanced the
# per-k-tile A offset as if A were half-byte packed. That made every k-tile overlap the
# previous one by 50% and put the upper half of A out of num_records (reads returned 0),
# i.e. it did less work than the shape asks for. Every w4a6/w6a6_tc throughput number
# measured before the fix carries that discount, so the only way to price it honestly is
# to run both addressings back-to-back in one warm session.
_A_ADDR_LEGACY = os.environ.get("A_ADDR_LEGACY") == "1"

# TC_FOLDSCALE tries to delete the 16 v_pk_mul_f16 that apply the E4M3 scale mantissa
# (1 + m/8) -- 16 of the 33 VALU per 32 codes, paid on A's staging transcode and B's alike --
# by riding that factor on a convert's scale operand instead:
#   1  fold into the 16 fp4->f16 decodes   (v_cvt_scalef32_pk_f16_fp4)
#   2  fold into the single f16->fp6 pack  (v_cvt_scalef32_pk32_fp6_f16)
# Mode 2 is the interesting one: one instruction covers all 32 codes, and the scale byte is
# uniform across a 32-code block, so it would be semantically identical for free.
#
# VERDICT (measured, kernel/s18_fold.sh): BOTH are exponent-only. Mode 1 drops the mantissa
# cleanly -- with scale byte 0x3c (m=4, factor 1.5) the folded path returns exactly 1/1.5 of
# the correct product. So the multiplies are not removable this way and the assumption in
# _tc_data_fp4_to_fp6_hw is now experimentally confirmed rather than assumed. Kept as a
# probe: it is the cheapest way to re-check the claim on a new architecture.
_TC_FOLDSCALE = int(os.environ.get("TC_FOLDSCALE") or 0)

# TC_A_LUT=1 routes w4a4_tc's A staging transcode through the 12-bit LDS LUT while leaving
# B on the hardware converts, which the LUT-vs-HWCVT sweep already settled in HWCVT's favour
# for B. The two operands are not in the same situation, which is why the switch is separate:
#
# The hardware path's peak live set is the 32 f16 (16 VGPRs) that v_cvt_scalef32_pk32_fp6_f16
# demands in consecutive registers, and A pays it inside the staging copy, on top of the
# accumulators. That is measurably what keeps w4a4_tc off the wide tiles it wants: at SQUARE
# 128x256 it needs 278 VGPRs where 256 would buy a second wave, and 278 - 254 (w4a6_tc, same
# tile, no A transcode) is almost exactly that peak. Asking the compiler for the wave instead
# (waves_per_eu) reaches 256 but spills 26 times and loses 16%. The LUT path never builds an
# f16 vector at all -- it goes byte -> two 6-bit codes via ds_read and bit-packs in i32 -- so
# it trades VALU for LDS traffic and should carry a far smaller register peak.
_TC_A_LUT = os.environ.get("TC_A_LUT") == "1"

# Convert all four E4M3 scale bytes of a word in parallel (SWAR) instead of one at a time.
# Same arithmetic, same result; set TC_SCALE_BYTEWISE=0 to get the old per-byte form back for
# an A/B. On by default because it is a strict instruction-count win with identical numerics.
_TC_SCALE_BYTEWISE = os.environ.get("TC_SCALE_BYTEWISE", "1") == "1"

# Decode subnormal E4M3 scale bytes (exponent field 0), extending the scale book's floor from
# 2^-6 down to 2^-9. Must track the host: qlayout only emits such bytes under the same
# MXFP6_E4M3_SUBNORMAL, and reading a subnormal byte with this off silently decodes it as a
# normal -- (1 + k/8) * 2^-7 instead of k * 2^-9, off by up to 8x on the whole block. That is
# why qlayout.assert_kernel_decodable() refuses to emit them unless this is set, and why the
# flag is in the kernel cache key rather than a free-floating env read.
_TC_E4M3_SUB = os.environ.get("MXFP6_E4M3_SUBNORMAL", "0") == "1"

# Synthesise the (1 + mant/8) f16 multiplier as a bit pattern instead of uitofp/mul/add/truncf.
# Same value for all 8 mantissas; set TC_MF_INT=0 for the f32 form. Forced off by TC_FOLDSCALE,
# whose probe needs the factor as a real f32 to hand to a convert's scale operand.
_TC_MF_INT = os.environ.get("TC_MF_INT", "1") == "1"

# Write the FP8 pad bytes of A's staged fp6 tile. The read side discards that dword, so skipping
# it is numerically free and does remove two zero words and 8 ds_write -- and it MEASURES WORSE:
# VGPRs 274 -> 278 and wpe=2 spills 22 -> 43, for -10% (1651 -> 1492). Splitting the pair of
# dwordx4 stores into dwordx4 + dwordx2 costs the allocator more in aligned-tuple pressure than
# the two registers it saves. Kept as a flag because the reasoning is sound and the result is not.
_TC_WRITE_PAD = os.environ.get("TC_WRITE_PAD", "1") == "1"

# Fence between A's staged transcode blocks so their 16-VGPR pk32 operand tuples do not overlap.
_TC_A_SERIAL = os.environ.get("TC_A_SERIAL", "1") == "1"

# 512-thread workgroups (8 waves instead of 4). Waves split N only, so this doubles the tile_n a
# workgroup can cover at an unchanged per-thread accumulator count, halving the A codes each
# thread stages -- w4a4_tc's last structural lever. Off by default: every shipped config is a
# 256-thread config and must stay bit-for-bit unaffected.
_WG512 = os.environ.get("WG512") == "1"
if _A_ADDR_LEGACY:
    import sys as _sys

    print(
        "[psg] WARNING A_ADDR_LEGACY active, A is mis-addressed and OUTPUT IS WRONG",
        file=_sys.stderr,
    )

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    math,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from .mfma_epilogues import mfma_epilog
from .mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    load_b_pack_k32,
    swizzle_xor16,
    tile_chunk_coord_i32,
    xcd_remap_bx_by,
)

_TILE_PRELOAD_TABLE = {
    # (tile_m, tile_n, tile_k): (dsrd_preload, dvmem_preload)
    # ── tile_m = 16 ──
    (16, 64, 256): (2, 2),
    (16, 64, 512): (8, 8),
    (16, 128, 256): (2, 2),
    (16, 128, 512): (2, 2),
    (16, 192, 256): (2, 2),
    (16, 256, 256): (2, 2),
    (16, 256, 512): (2, 2),
    (16, 512, 256): (2, 2),
    # ── tile_m = 32 ──
    (32, 64, 128): (6, 6),
    (32, 64, 256): (6, 6),
    (32, 64, 512): (2, 2),
    (32, 128, 128): (6, 6),
    (32, 128, 256): (6, 6),
    (32, 192, 128): (6, 6),
    (32, 192, 256): (6, 6),
    (32, 256, 128): (6, 6),
    (32, 256, 256): (6, 6),
    # ── tile_m = 48 ──
    (48, 64, 128): (8, 8),
    (48, 64, 256): (2, 2),
    (48, 128, 256): (6, 6),
    (48, 192, 256): (6, 6),
    (48, 256, 256): (6, 6),
    # ── tile_m = 64 ──
    (64, 64, 128): (4, 4),
    (64, 64, 256): (4, 4),
    (64, 128, 128): (8, 8),
    (64, 128, 256): (8, 8),
    (64, 192, 128): (8, 8),
    (64, 192, 256): (8, 8),
    (64, 256, 64): (8, 8),
    (64, 256, 128): (8, 8),
    (64, 256, 256): (8, 8),
    # ── tile_m = 80 ──
    (80, 64, 256): (4, 4),
    (80, 128, 256): (8, 8),
    (80, 192, 256): (8, 8),
    (80, 256, 256): (8, 8),
    # ── tile_m = 96 ──
    (96, 64, 128): (6, 6),
    (96, 64, 256): (6, 6),
    (96, 128, 128): (8, 8),
    (96, 128, 256): (6, 6),
    (96, 192, 128): (8, 8),
    (96, 192, 256): (8, 8),
    (96, 256, 128): (8, 8),
    (96, 256, 256): (8, 8),
    # ── tile_m = 112 ──
    (112, 64, 256): (8, 8),
    (112, 128, 256): (4, 4),
    (112, 192, 256): (8, 8),
    (112, 256, 256): (8, 8),
    # ── tile_m = 128 ──
    (128, 64, 128): (6, 6),
    (128, 64, 256): (8, 8),
    (128, 128, 64): (4, 4),
    (128, 128, 128): (8, 8),
    (128, 128, 256): (4, 4),
    (128, 192, 128): (8, 8),
    (128, 192, 256): (8, 8),
    (128, 256, 128): (6, 6),
    (128, 256, 256): (4, 4),
    # ── tile_m = 160 ──
    (160, 192, 128): (8, 8),
    # ── tile_m = 192 ──
    (192, 64, 128): (6, 6),
    (192, 128, 128): (6, 6),
    # ── tile_m = 224 ──
    (224, 64, 128): (4, 4),
    (224, 128, 128): (6, 6),
    (224, 192, 128): (6, 6),
    # ── tile_m = 256 ──
    (256, 64, 128): (4, 4),
    (256, 128, 128): (6, 6),
    (256, 192, 128): (6, 6),
    (256, 256, 128): (4, 4),
}

_TILE_PRELOAD_DEFAULT = (0, 0)


def _get_preload(tile_m, tile_n, tile_k):
    """Look up (dsrd_preload, dvmem_preload) from the tile table."""
    return _TILE_PRELOAD_TABLE.get(
        (int(tile_m), int(tile_n), int(tile_k)), _TILE_PRELOAD_DEFAULT
    )


@functools.lru_cache(maxsize=1024)
def compile_preshuffle_gemm_a8(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "fp16",
    lds_stage: int = 2,
    use_cshuffle_epilog: bool = False,
    waves_per_eu: Optional[int] = None,
    use_async_copy: bool = False,
    dsrd_preload: int = -1,
    dvmem_preload: int = -1,
    epilogue: str = "none",  # "none", "bias", "bias_relu", "bias_silu", "bias_gelu"
    xcd_swizzle: int = 0,
    use_hw_transcode: Optional[bool] = None,
    split_k: int = 1,
    split_reduce: str = "planes",
):
    """Compile the preshuffle GEMM kernel using the @flyc.kernel API.

    Returns a JitFunction that auto-compiles and executes when called.
    Signature:  launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias, arg_lut, M, N, stream)
                (arg_lut is the NVFP4->MXFP6 weight-transcode LUT for in_dtype='w6a6_tc';
                 pass any valid device pointer/dummy for other dtypes.)

    Compile-time constants: K, tile_m/n/k, in_dtype, out_dtype (determine loop structure).
    Runtime parameters: M, N (passed as i32 kernel args).

    Args:
        out_dtype: Output element type, "fp16" or "bf16" (default: "fp16").
        waves_per_eu: Occupancy hint (None = default, 1-4 = limit occupancy).
        use_async_copy: Use async DMA for A tile global-to-LDS transfer.
        dsrd_preload: Initial LDS-read preload count (-1 = auto from _TILE_PRELOAD_TABLE).
        dvmem_preload: Initial global-load preload count (-1 = auto from _TILE_PRELOAD_TABLE).
        use_hw_transcode: For in_dtype='w6a6_tc', transcode NVFP4->MXFP6 with the gfx950
            hardware scale-converts instead of the 12-bit LDS LUT. Measured +1..+3% on decode
            and +8..+10% on compute shapes, and it frees the 8 KB LUT arena. Rounds in
            hardware, so it is bit-identical to the LUT only when the E4M3 scale mantissa is
            0; accuracy is gated by kernel/s15_accgate.sh (rel-L2 of the mantissa fold within
            5% of the LUT's own -- measured 1.0124). None = take the TC_HWCVT env default,
            which keeps every existing script and trace valid.
        split_k: number of K slices to compute in parallel (1 = off, unchanged codegen).
            The grid gains a z dimension of this size and workgroup z computes only
            K/split_k of the reduction, so C is written as split_k separate PARTIAL planes
            and the caller must sum them: pass a C buffer of split_k*M*N elements and reduce
            over the leading axis. Decode shapes cannot fill the GPU from the M/N grid alone
            (DOWN at M=128 is 40 workgroups of 256 CUs), and re-arranging the same work as
            more workgroups with a shorter K loop measured +91% on DOWN and +174% on QKV
            (kernel/s17_wgprobe.py), which is what this exposes. Gains peak near 100% CU
            occupancy and decay past it, so pick split_k to land the grid near 256, not to
            maximise it. Requires K/split_k to be a multiple of tile_k, of 256 for the
            block-scaled 4-bit family (the scale layout addresses K in 256-element chunks),
            and >= 4*tile_k so the K pipeline still has its prologue and drain.
        split_reduce: how the split_k partial sums are combined. Ignored at split_k=1.
            "planes" (default): each workgroup owns a private C plane and the caller sums
                them. Safe -- a stray store cannot cross splits -- but it costs a full extra
                pass over split_k*M*N. On DOWN at split_k=19 that pass is 5.4 of 15.4 us,
                i.e. 35% of the throughput the split bought (640 raw -> 416 net).
            "atomic": every workgroup accumulates into ONE fp32 C plane with
                buffer_atomic_fadd, so there is no second pass and no split_k-sized
                workspace. C must be fp32 and zero-initialised by the caller.
                This is not a guess about the hardware: s14 measured signal-based reduction
                at -23..-29% against atomics on this exact shape, because CDNA4's L2 has
                native fp32 atomic-add units -- an atomicAdd is ~1 L2 transaction per
                element, and at M=128 the whole C plane is ~1.3 MB and stays L2-resident,
                so the atomics never reach HBM while "planes" writes 12.5 MB out and reads
                it back. The earlier "atomics kill split-K" result (split4 -> 60 TFLOP/s)
                was a dropped accumulate branch in the bf16 epilogue, not the atomics.
            Requires use_cshuffle_epilog=False: the cshuffle path packs bf16 pairs through
            LDS, and accumulating 19 bf16 partials in bf16 loses roughly a decimal digit.
    """
    if dsrd_preload < 0 or dvmem_preload < 0:
        if in_dtype in ("fp8", "int8") and str(get_hip_arch()) == "gfx950":
            computed_dsrd, computed_dvmem = _get_preload(tile_m, tile_n, tile_k)
        else:
            computed_dsrd, computed_dvmem = _TILE_PRELOAD_DEFAULT
        if dsrd_preload < 0:
            dsrd_preload = computed_dsrd
        if dvmem_preload < 0:
            dvmem_preload = computed_dvmem
    # Resolve the transcode path once. Selectable per call so the dispatch can pick it per
    # shape (as it already does for use_async_copy) rather than through a global default,
    # which would silently redefine the LUT arm of every existing A/B script.
    hwcvt = _TC_HWCVT if use_hw_transcode is None else bool(use_hw_transcode)
    # "w6a6_tc" was a misnomer: it named the MFMA's operand widths rather than the operands'
    # STORAGE, and if weights and activations were both really MXFP6 there would be nothing to
    # transcode. The kernel stores W as NVFP4 and A as MXFP6 and converts W, so it is w4a6_tc.
    # The old spelling stays accepted because scripts and cached kernel names use it.
    if in_dtype == "w4a6_tc":
        in_dtype = "w6a6_tc"
    if in_dtype not in (
        "fp8", "int8", "int4", "fp16", "bf16", "fp4", "fp6", "w4a6", "w6a6",
        "w6a6_tc", "w4a4_tc",
    ):
        raise ValueError(
            "in_dtype must be one of ('fp8','int8','int4','fp16','bf16','fp4','fp6',"
            f"'w4a6','w6a6','w4a6_tc','w4a4_tc'), got {in_dtype!r}"
        )
    if out_dtype not in ("fp16", "bf16"):
        raise ValueError(f"out_dtype must be 'fp16' or 'bf16', got {out_dtype!r}")
    _out_is_bf16 = out_dtype == "bf16"
    # W_MXFP4_A_MXFP6 (article's W4A6): weight B = MXFP4 (dense 16 B, blgp=4),
    # activation A = MXFP6-E2M3 (FP8-padded 32 B/lane, read 24 real bytes, cbsz=2).
    # Reuses the ENTIRE fp4 B + scale + K-iteration path; only the A operand differs.
    # Requires tile_k=128 (single K=128 MFMA step so k_unroll=1 matches the fp4 B path).
    # W6A6_TC: weights stored NVFP4 (E2M1 codes + per-32 E4M3 scale), transcoded
    # to MXFP6 in-kernel via a 12-bit LUT; activation A is native MXFP6. It shares
    # the ENTIRE W4A6 path (A=fp6 24 B/lane cbsz=2; B loaded as fp4 16 B/lane +
    # per-32 scale). Only the MFMA B operand differs: fp4->fp6 LUT transcode,
    # E4M3->E8M0 scale transcode, and blgp=2 (fp6) instead of 4 (fp4).
    # W4A4_TC: BOTH operands stored NVFP4 (E2M1 codes + per-32 E4M3 scale) and both
    # transcoded to MXFP6 in-kernel, so the MFMA still runs MXFP6 x MXFP6.
    #
    # B is transcoded exactly as in W4A6_TC (per B fragment, hoisted above the A-row loop).
    # A is transcoded at the global->LDS staging copy instead of per MFMA fragment: staged
    # once per workgroup, the tile is reused across all tile_n columns and by all four waves,
    # where per-fragment conversion would have each wave convert the same A rows. Staging also
    # leaves the MFMA inner loop, its register pressure and the whole B path byte-identical to
    # W4A6_TC: A lands in LDS in the very same FP8-padded MXFP6 layout W4A6 already reads.
    #
    # What staging does NOT do is make A's transcode cheap, which an earlier version of this
    # comment claimed. Per MFMA, A's conversion costs ~1/tile_n and B's ~1/tile_m, because each
    # is paid once per tile and there are tile_m*tile_n*tile_k MFMA-lanes to spread it over. B's
    # is amortised the same way A's is, so at tile_m == tile_n the two tiles hold equally many
    # codes and W4A4_TC simply does TWICE the transcode of W4A6_TC -- measured at SQUARE
    # 128x128x128: 256 decodes / 256 packed multiplies / 16 packs against 128/128/8, and 18.6
    # VALU per MFMA against 11.1 (kernel/s18_isa.py). That factor, not the loss of async, is
    # what W4A4_TC has to pay for storing A at 4 bits, and it is why the dtype prefers a much
    # wider tile than W4A6_TC does.
    #
    # The price is use_async_copy: that lever replaces this staging copy with a hardware
    # global->LDS DMA, and a DMA cannot transform its payload. Measured cost of giving it up
    # (s18_async.sh): 6-11% on the four memory-bound rows, 28-33% on the eight compute rows.
    is_w4a4_tc = in_dtype == "w4a4_tc"
    is_w6a6_tc = (in_dtype == "w6a6_tc") or is_w4a4_tc
    # A's staging transcode follows the LUT whenever the kernel is on the LUT at all, and can
    # also be put there on its own while B keeps the hardware converts (see _TC_A_LUT).
    a_lut = is_w4a4_tc and (_TC_A_LUT or not hwcvt)
    # Anything that reads the LUT needs it resident, so the arena and its cooperative fill
    # follow "someone uses it", not "B uses it".
    want_lut = is_w6a6_tc and ((not hwcvt) or a_lut)
    # W6A6: both operands native MXFP6 (E2M3 codes + one E8M0 per 32 elements), no
    # transcode. This is the block-scaled 6-bit arm, and it exists because the arm named
    # `fp6` is not one: fp6 predates the scaled MFMA path here and inherits the fp8 path's
    # scaling, i.e. it feeds the MFMA a scale operand pinned to E8M0=1.0 and applies one
    # f32 per token and per output channel in the epilogue -- 32x coarser than every other
    # quantized arm, which is a scale-granularity difference reported as an element-width
    # one. Same instruction (mfma_scale_f32_16x16x128_f8f6f4) and same 6-bit element codes
    # in both; the difference is entirely in the scale operand.
    #
    # It reuses proven pieces rather than adding a path: A is exactly W4A6's A (FP8-padded
    # fp6, 24 real bytes/lane, cbsz=2), B is exactly the fp6 arm's B (FP8-padded fp6, two
    # 16 B groups per K=128), and the per-32 scale load/opsel routing is the fp4 one
    # untouched -- scale bytes are one per 32 elements whatever the element width, so that
    # path never had to care.
    is_w6a6 = in_dtype == "w6a6"
    is_w4a6 = (in_dtype == "w4a6") or is_w6a6_tc or is_w6a6
    if is_w4a6 and int(tile_k) != 128:
        raise NotImplementedError(
            "w4a6/w6a6/w6a6_tc/w4a4_tc currently requires tile_k=128")
    if is_w4a4_tc and use_async_copy:
        # Fail loudly: the async DMA would copy raw fp4 codes into an LDS tile the MFMA
        # reads as MXFP6, i.e. silently wrong numerics at full speed.
        raise ValueError(
            "w4a4_tc is incompatible with use_async_copy: A is transcoded fp4->fp6 in the "
            "global->LDS staging copy, which the async DMA path replaces with a hardware "
            "copy that cannot transform its payload"
        )
    is_fp4 = (in_dtype == "fp4") or is_w4a6
    # MXFP6-E2M3: FP8-padded A/B tiles (24 real + 8 zero-pad bytes per K=32),
    # i.e. identical 32-byte/lane layout to fp8, but the f8f6f4 MFMA selects the
    # 6-bit element format via cbsz/blgp=2. Reuses the entire fp8 load/LDS/frag path.
    is_fp6 = in_dtype == "fp6"
    is_int4 = in_dtype == "int4"
    is_int8 = (in_dtype == "int8") or is_int4
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    elem_bytes = (
        1
        if (
            in_dtype
            in ("fp8", "int8", "int4", "fp4", "fp6", "w4a6", "w6a6", "w6a6_tc",
                "w4a4_tc")
        )
        else 2
    )
    # W6A6 is the one block-scaled arm with no 4-bit operand: both sides are FP8-padded
    # fp6 (1 byte per element), so both take the fp6 arm's pack=1 addressing. k_unroll
    # follows from this (128 bytes / 1 / 64 = 2), which is what puts B's second 16 B group
    # in b_tile_in[1] for the MFMA to pick up as its third dword pair.
    a_elem_vec_pack = 1 if is_w6a6 else (2 if is_fp4 else 1)
    b_elem_vec_pack = 1 if is_w6a6 else (2 if is_fp4 else 1)
    # A-operand footprint packing: fp4 packs 2 elems/byte (16 B/lane); W4A6's fp6
    # activation is FP8-padded (1 "elem"/byte, 32 B/lane) so the A tile/LDS/load use
    # pack=1 exactly like the proven fp6 path, while k_unroll + the B path keep
    # a_elem_vec_pack=2 (tile_k=128 -> k_unroll=1).
    a_pack_eff = 1 if is_w4a6 else a_elem_vec_pack
    # A's HBM footprint and its LDS footprint stop agreeing under w4a4_tc: A is stored NVFP4
    # (2 codes/byte) and staged as MXFP6 (1 code/byte, FP8-padded). a_pack_eff keeps meaning
    # "LDS side" -- so the LDS tile, its row stride and the swizzle block count are untouched
    # and the MFMA reads exactly the W4A6 layout -- while a_pack_glob drives every global-side
    # quantity: bytes per tile, per-thread load count, row stride and the tile->thread map.
    a_pack_glob = 2 if is_w4a4_tc else a_pack_eff

    # Divisor for A's *addressing* (buffer size and k-tile advance) as opposed to A's tile
    # shape. These are the same quantity; the probe flag exists only to reproduce the old
    # wrong value so the fix can be priced. It differs from a_pack_glob for w4a6/w6a6_tc
    # alone (2 vs 1) -- fp4 and w4a4_tc are genuinely half-byte packed, so they are
    # unaffected and serve as the control arm.
    a_pack_addr = a_elem_vec_pack if _A_ADDR_LEGACY else a_pack_glob

    KERNEL_NAME = (
        f"preshuffle_gemm_{in_dtype}_{out_dtype}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_lds{lds_stage}"
        f"_pl{dsrd_preload}x{dvmem_preload}"
    )
    if use_cshuffle_epilog:
        KERNEL_NAME += "_csh"
    if use_async_copy:
        KERNEL_NAME += "_async"
    if waves_per_eu is not None:
        KERNEL_NAME += f"_wpe{waves_per_eu}"
    if epilogue != "none":
        KERNEL_NAME += f"_ep_{epilogue}"
    if xcd_swizzle > 0:
        KERNEL_NAME += f"_xcd{xcd_swizzle}"
    # Probe arms must not share a cache key with the real kernel or with each other. The flags
    # are traced from module globals, so a probe script can flip them between compiles -- and
    # would then silently measure whichever binary was cached first, which is how s16's
    # TC_HWSCALE A/B "proved" a change that had never been compiled in.
    if _TC_PROBE_NOMUL:
        KERNEL_NAME += "_pnomul"
    if _TC_PROBE_NOSCALE:
        KERNEL_NAME += "_pnoscale"
    if _TC_PROBE_NOCVT:
        KERNEL_NAME += "_pnocvt"
    if _A_ADDR_LEGACY:
        KERNEL_NAME += "_palegacy"
    if _TC_FOLDSCALE:
        KERNEL_NAME += f"_fold{_TC_FOLDSCALE}"
    if not _TC_SCALE_BYTEWISE:
        KERNEL_NAME += "_scbyte"
    if _TC_E4M3_SUB:
        # Different scale decode, so a different binary: reusing a cached non-subnormal one
        # would misread every subnormal byte as a normal without failing anything.
        KERNEL_NAME += "_e4sub"
    if not _TC_MF_INT:
        KERNEL_NAME += "_mff32"
    if not _TC_WRITE_PAD:
        KERNEL_NAME += "_nopad"
    if not _TC_A_SERIAL:
        KERNEL_NAME += "_apar"
    if _WG512:
        KERNEL_NAME += "_wg512"
    if a_lut and hwcvt:
        # A on the LUT while B keeps the hardware converts: a distinct binary from both the
        # all-HWCVT and all-LUT arms, so it needs its own cache key.
        KERNEL_NAME += "_alut"

    tile_k_bytes = int(tile_k) * int(elem_bytes)

    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )

    _min_k_unroll = tile_k_bytes // a_elem_vec_pack // 64
    if is_fp4 and _min_k_unroll < 2 and int(tile_k) != 128:
        raise ValueError(
            f"FP4 requires tile_k=128 or tile_k >= {64 * 2 * a_elem_vec_pack} "
            f"(mfma_scale_f32_16x16x128 needs k_unroll >= 1), "
            f"got tile_k={tile_k} (k_unroll={_min_k_unroll})"
        )
    if is_fp4 and int(tile_k) == 128 and lds_stage != 2:
        raise NotImplementedError("FP4 tile_k=128 currently only supports lds_stage=2")

    # The 4-bit B path packs two N accumulator blocks into each mfma_scale operand, so the MFMA
    # loop is driven by num_acc_n//2 = tile_n//128. At tile_n=64 that floors to zero: the loop body
    # never runs, C is never written, and the kernel "wins" every decode sweep by computing nothing
    # (s17 measured up to +224% from this). It compiled and ran silently because the harness fed
    # all-zero scales, which makes an all-zero C the correct answer. Reject the tile here so the
    # shape is impossible to select, rather than relying on the harness output check to notice.
    # Waves split N and nothing else, so doubling the workgroup doubles tile_n at an unchanged
    # per-thread accumulator count -- and halves the A codes each thread stages, which is the
    # whole point for w4a4_tc. 128x512 on 8 waves gives each wave the same 64-wide slice and the
    # same 128 accumulators that 128x256 on 4 waves does.
    num_waves_const = 8 if _WG512 else 4
    # The packed-N MFMA consumes accumulator blocks two at a time, so num_acc_n must be a positive
    # EVEN number. Both failure modes are silent and both look like speedups:
    #   num_acc_n == 0  -> no MFMA issues at all, C is never written (s17's tile_n=64, +224% fake).
    #   num_acc_n odd   -> the loop runs num_acc_n//2 pairs and DROPS the leftover block, so the
    #                      kernel computes (num_acc_n-1)/num_acc_n of N. s19 measured tile_n=384 at
    #                      8 waves (num_acc_n=3) reporting 2956 TFLOP/s against a true best of 2177
    #                      -- it was 36% "faster" because it computed two thirds of the output.
    # The old form of this check hardcoded tile_n//64, which is num_acc_n only at 4 waves, and it
    # tested only for zero. Derive the count from num_waves_const and reject odd as well.
    _num_acc_n = int(tile_n) // num_waves_const // 16
    if is_fp4 and (_num_acc_n < 2 or _num_acc_n % 2):
        raise NotImplementedError(
            f"4-bit B (fp4/w4a6/w6a6_tc/w4a4_tc) needs an even number of N accumulator blocks per "
            f"wave, got num_acc_n={_num_acc_n} for tile_n={tile_n} on {num_waves_const} waves "
            f"({int(tile_n) // num_waves_const}-wide slices). The packed-N MFMA takes 2 blocks per "
            f"step, so an odd count silently drops the last block and computes only "
            f"{max(_num_acc_n - 1, 0)}/{max(_num_acc_n, 1)} of N, while a zero count writes no C at "
            f"all. Either way the kernel measures FASTER, so this must be rejected at compile time "
            f"rather than caught by a benchmark."
        )

    # Split-K: each workgroup z reduces only K/split_k, so K keeps its role as the operand
    # ADDRESSING stride everywhere while the loop TRIP COUNTS switch to the slice. Rejecting the
    # bad divisors here means the K-loop driver below can treat _k_loop as an ordinary compile-time
    # constant, exactly as it treated K.
    split_k = int(split_k)
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if split_k > 1:
        if K % split_k:
            raise ValueError(f"split_k={split_k} must divide K={K}")
        _k_slice_chk = K // split_k
        if _k_slice_chk % int(tile_k):
            raise ValueError(
                f"split_k={split_k} gives a K slice of {_k_slice_chk}, which is not a multiple "
                f"of tile_k={tile_k}"
            )
        if is_fp4 and _k_slice_chk % 256:
            raise ValueError(
                f"split_k={split_k} gives a K slice of {_k_slice_chk}; the block-scaled 4-bit "
                "path addresses scales in 256-element K chunks (_K1 = K//256), so the slice must "
                "be a multiple of 256"
            )
        if _k_slice_chk < 4 * int(tile_k):
            raise ValueError(
                f"split_k={split_k} gives a K slice of {_k_slice_chk} = "
                f"{_k_slice_chk // int(tile_k)} k-tiles; the pipelined K loop peels a prologue and "
                f"drains up to 3 tiles, so it needs at least 4 (tile_k={tile_k})"
            )

    if split_reduce not in ("planes", "atomic"):
        raise ValueError(f"split_reduce must be 'planes' or 'atomic', got {split_reduce!r}")
    # Deliberately NOT gated on split_k > 1. atomic at split_k=1 is pointless for speed but it is
    # the control that separates "each split computes the wrong K range" from "the epilogue visits
    # each output element more than once": a store is idempotent so a double visit is invisible in
    # the default path, while an atomic multiplies by the visit count. The first cut of this path
    # came out at exactly 2.00x the reference for every split_k, which is the signature of the
    # latter, and there is no way to tell them apart without being able to run split_k=1.
    _atomic_split = split_reduce == "atomic"
    if _atomic_split:
        # Every workgroup now writes the SAME C plane, so the guard that made "planes" safe --
        # each workgroup bounded to its own buffer resource -- is gone. Fail loudly on the
        # combinations where that would silently corrupt rather than merely be slow.
        if use_cshuffle_epilog:
            raise ValueError(
                "split_reduce='atomic' requires use_cshuffle_epilog=False: the cshuffle path "
                "stages bf16 pairs through LDS and would accumulate the partials in bf16, "
                "which at split_k=19 costs about a decimal digit of accuracy"
            )
        if epilogue != "none":
            raise ValueError(
                f"split_reduce='atomic' cannot be combined with epilogue={epilogue!r}: a bias "
                "or activation is a function of the FULL sum, but each workgroup holds only "
                "K/split_k of it, so applying it per split is wrong split_k times over"
            )

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    gpu_arch = get_hip_arch()

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    total_threads = 512 if _WG512 else 256
    if _WG512 and int(tile_n) % (num_waves_const * 16):
        raise ValueError(
            f"WG512 splits tile_n={tile_n} across {num_waves_const} waves in 16-wide MFMA "
            f"blocks, so tile_n must be a multiple of {num_waves_const * 16}"
        )
    bytes_a_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes) // a_pack_glob

    a_load_bytes = 16
    # How many threads take part in staging A: normally all of them, fewer when the A tile is
    # too small to give every thread a whole dwordx4.
    #
    # This is the constraint that set w4a4_tc's decode performance, and it set it invisibly.
    # A 32-row tile at tile_k=128 is 32*128/2 = 2048 B of NVFP4, which is 8 B per thread over
    # 256 -- half a dwordx4 -- so the builder raised, the decode sweep caught the exception,
    # and tile_m=64 was recorded as w4a4_tc's best tile when it was its ONLY tile. At M=32
    # that computes 64 rows to keep 32, on the one arm that also transcodes its A operand.
    #
    # No other arm reaches it. The arms with a byte-per-element A (w4a6, w6a6, w6a6_tc, fp6)
    # have twice the A bytes and land on 16 at tile_m=32; fp4 is half-packed like w4a4_tc but
    # is free to take tile_k=256, which puts it back at 16. w4a4_tc is the only arm that is
    # 4-bit on A *and* pinned to tile_k=128 by the w4a6-family fragment layout, so it is the
    # only one for which the two constraints multiply.
    #
    # The fix is not a smaller load. 16 B is the dwordx4 that swizzle_xor16 is defined on, and
    # an 8 B store would write half a swizzle block. It is to let the surplus threads REPEAT a
    # staging thread's chunk: tx and tx + a_stage_threads derive the same coordinate, load the
    # same global bytes, transcode them identically and write the same LDS address with the
    # same value. The duplicate global read hits L1 (A is 2 KB at decode) and the duplicate LDS
    # write is idempotent, so this needs no predicate, no branch and no divergence -- which is
    # what makes it safe to apply to every arm rather than special-cased to this one.
    if bytes_a_per_tile % a_load_bytes:
        raise ValueError(
            f"A tile of {bytes_a_per_tile} B is not a whole number of {a_load_bytes} B "
            f"loads: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}, "
            f"pack={a_pack_glob}"
        )
    a_stage_threads = min(total_threads, bytes_a_per_tile // a_load_bytes)
    if total_threads % a_stage_threads:
        raise ValueError(
            f"{total_threads} threads do not divide evenly into {a_stage_threads} A-staging "
            f"threads, so the surplus threads cannot mirror them 1:1"
        )
    bytes_per_thread_a = bytes_a_per_tile // a_stage_threads
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load_bytes}"
        )
    if a_stage_threads != total_threads and use_async_copy:
        # The async DMA path indexes A by tx over all threads and has no idempotent-write
        # story; it is also already incompatible with the arm this exists for.
        raise ValueError(
            f"a small A tile ({bytes_a_per_tile} B, {a_stage_threads} staging threads) is "
            "incompatible with use_async_copy"
        )
    a_async_load_bytes = 4 if gpu_arch == "gfx942" else 16
    a_async_load_dword = a_async_load_bytes // 4

    bytes_b_per_tile = int(tile_n) * int(tile_k) * int(elem_bytes) // b_elem_vec_pack
    bytes_per_thread_b = bytes_b_per_tile // total_threads
    b_load_bytes = 16
    num_b_loads = bytes_per_thread_b // b_load_bytes

    wave_size = 64
    # The A tile as it sits in LDS, which is what the MFMA loop reads back and therefore what
    # the ds_read scheduler hint must count. Under w4a4_tc this is twice bytes_a_per_tile.
    lds_bytes_a_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes) // a_pack_eff
    num_a_lds_load = lds_bytes_a_per_tile // wave_size // a_load_bytes

    _is_gfx950 = str(gpu_arch).startswith("gfx950")
    _is_gfx942 = str(gpu_arch).startswith("gfx942")
    use_mfma_k32 = _is_gfx950 and is_f16_or_bf16

    lds_stride_bytes = tile_k_bytes

    Vec = fx.Vector

    def _fp8_dtype():
        return (
            fx.Float8E4M3FN
            if (_is_gfx950 or str(gpu_arch).startswith("gfx12"))
            else fx.Float8E4M3FNUZ
        )

    def _elem_dtype():
        if is_f16:
            return fx.Float16
        if is_bf16:
            return fx.BFloat16
        if is_fp4:
            return fx.Int8
        return fx.Int8 if is_int8 else _fp8_dtype()

    def _elem_type():
        return _elem_dtype().ir_type

    def _vec16_type():
        if is_f16:
            return Vec.make_type(8, fx.Float16)
        if is_bf16:
            return Vec.make_type(8, fx.BFloat16)
        if is_fp4:
            return Vec.make_type(16, fx.Int8)
        return Vec.make_type(16, fx.Int8 if is_int8 else _fp8_dtype())

    def _mfma_pack_ty():
        if is_f16:
            return Vec.make_type(4, fx.Float16)
        if is_bf16:
            return Vec.make_type(4, fx.Int16)
        return fx.Int64.ir_type

    def _out_dtype():
        return fx.BFloat16 if _out_is_bf16 else fx.Float16

    def _out_elem():
        return _out_dtype().ir_type

    # ── LDS sizing (pure Python, no MLIR ops) ────────────────────────────────
    lds_tile_bytes = int(tile_m) * int(lds_stride_bytes) // a_pack_eff
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0

    lds_pong_offset = 0
    lds_ping_offset = 0
    lds_alloc_offset = 0
    # W6A6_TC: reserve an LDS-resident copy of the 12-bit NVFP4->MXFP6 transcode
    # LUT (8 mantissa rows * 256 bytes * 4 = 8192 B) at the start of the pong
    # arena. Reading the LUT from LDS (vs global) removes DRAM/L2 traffic that
    # otherwise competes with the fp4 weight loads (fatal in bandwidth-bound
    # decode) and cuts lookup latency. gfx950 has 160 KB LDS/CU, so 8 KB is cheap.
    # HWCVT computes the transcode arithmetically, so the LUT arena is dead weight.
    lut_lds_bytes = (8 * 256 * 4) if want_lut else 0
    lut_lds_offset = 0
    if lut_lds_bytes:
        lut_lds_offset = allocator_pong._align(allocator_pong.ptr, 128)
        allocator_pong.ptr = lut_lds_offset + lut_lds_bytes
    if int(lds_stage) == 2:
        assert lds_out_bytes % 2 == 0, "lds_out_bytes should be multiple of 2"
        buffer_size_bytes = max(lds_tile_bytes, lds_out_bytes // lds_stage)
        buffer_size_elems = (
            buffer_size_bytes if elem_bytes == 1 else (buffer_size_bytes // 2)
        )

        lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = lds_pong_offset + buffer_size_elems * elem_bytes

        lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
        allocator_ping.ptr = lds_ping_offset + buffer_size_elems * elem_bytes
    else:
        lds_total_bytes = max(lds_tile_bytes, lds_out_bytes)
        lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

        lds_alloc_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = lds_alloc_offset + lds_total_elems * elem_bytes

    # ── Kernel function ────────────────────────────────────────────────────
    _has_epilogue = epilogue != "none"
    _has_bias = epilogue in ("bias", "bias_relu", "bias_silu", "bias_gelu")
    _has_relu = epilogue == "bias_relu"
    _has_silu = epilogue == "bias_silu"
    _has_gelu = epilogue == "bias_gelu"

    # Fused epilogue is implemented inside body_row (the direct store path).
    # When use_cshuffle_epilog=True, the epilogue path goes through
    # write_row_to_lds -> store_pair and returns before body_row, which would
    # silently drop the bias + activation. Reject the unsupported combination.
    if _has_epilogue and use_cshuffle_epilog:
        raise ValueError(
            "Fused epilogue (epilogue != 'none') is not supported with "
            "use_cshuffle_epilog=True; the cshuffle path bypasses body_row "
            "where the bias/activation fusion lives."
        )

    # AMDGPU defaults max_flat_workgroup_size to 256; an 8-wave workgroup has to declare itself
    # or the launch is rejected. Only stated for WG512 so the 256-thread kernels keep the exact
    # attributes -- and the exact register budget -- they compile with today.
    _kernel_deco = (
        flyc.kernel(known_block_size=[total_threads, 1, 1]) if _WG512 else flyc.kernel
    )

    @_kernel_deco
    def kernel_gemm(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_lut: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        c_m = fx.Index(i32_m)
        c_n = fx.Index(i32_n)

        # ---- Types ----
        acc_init = (
            Vec.filled(4, 0, fx.Int32) if is_int8 else Vec.filled(4, 0.0, fx.Float32)
        )

        # ---- Layouts ----

        _k_div4_factor = (K * elem_bytes) // 4 // a_pack_glob

        # Split-K slice length. Every K-derived value below is one of two kinds, and only the
        # second kind moves: addressing strides stay at full K (the operands are still full-K
        # tensors), while the K-loop trip counts use _k_loop. At split_k=1 they are the same
        # number, so nothing about the generated kernel changes.
        _k_loop = K // split_k

        kpack_bytes = 8 if is_int4 else 16
        kpack_elems = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
        k_bytes_b = K * elem_bytes // b_elem_vec_pack
        n0_val = N // 16
        k0_val = k_bytes_b // 64
        _stride_nlane = kpack_elems
        _stride_klane = 16 * _stride_nlane
        _stride_k0 = 4 * _stride_klane
        _stride_n0 = k0_val * _stride_k0
        layout_b = fx.make_layout(
            (n0_val, k0_val, 4, 16, kpack_elems),
            (_stride_n0, _stride_k0, _stride_klane, _stride_nlane, 1),
        )

        lds_k_dim = tile_k // a_pack_eff
        k_blocks16 = fx.Index(tile_k_bytes // a_pack_eff // 16)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        bx, by = xcd_remap_bx_by(
            bx,
            by,
            c_m,
            tile_m=tile_m,
            tile_n=tile_n,
            N=N,
            xcd_swizzle=xcd_swizzle,
        )

        # ---- LDS (separate ping/pong buffers for no-alias guarantee) ----
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()

        lds_a_pong_ptr = SmemPtr(
            base_ptr_pong, lds_alloc_offset, _elem_type(), shape=(1,)
        )
        lds_a_ping_ptr = lds_a_pong_ptr
        lds_out_ptr = SmemPtr(base_ptr_pong, lds_alloc_offset, _out_elem(), shape=(1,))

        if const_expr(lds_stage == 2):
            lds_a_pong_ptr = SmemPtr(
                base_ptr_pong, lds_pong_offset, _elem_type(), shape=(tile_m * tile_k,)
            )
            lds_a_ping_ptr = SmemPtr(
                base_ptr_ping, lds_ping_offset, _elem_type(), shape=(tile_m * tile_k,)
            )

            if const_expr(use_cshuffle_epilog):
                lds_out_ptr = SmemPtr(
                    base_ptr_pong,
                    lds_pong_offset,
                    _out_elem(),
                    shape=(tile_m * tile_n,),
                )
            else:
                lds_out_ptr = SmemPtr(
                    base_ptr_pong, lds_pong_offset, _out_elem(), shape=(1,)
                )
        else:
            lds_a_pong_ptr = SmemPtr(
                base_ptr_pong, lds_alloc_offset, _elem_type(), shape=(lds_total_elems,)
            )
            lds_a_ping_ptr = lds_a_pong_ptr
            if const_expr(use_cshuffle_epilog):
                lds_out_ptr = SmemPtr(
                    base_ptr_pong,
                    lds_alloc_offset,
                    _out_elem(),
                    shape=(tile_m * tile_n,),
                )
            else:
                lds_out_ptr = SmemPtr(
                    base_ptr_pong, lds_alloc_offset, _out_elem(), shape=(1,)
                )

        lds_a_pong = lds_a_pong_ptr.get()
        lds_a_ping = lds_a_ping_ptr.get()
        lds_out = lds_out_ptr.get()

        # ---- Buffer resources (runtime byte sizes for OOB protection) ----
        # A's row length must come from A's own STORAGE packing (a_pack_glob), not from
        # a_elem_vec_pack. They differ for exactly the w4a6 family, whose A is FP8-padded
        # MXFP6 at 1 byte/element while a_elem_vec_pack is 2 for the whole fp4 family. Using
        # the latter here declared the A buffer as half the tensor, so every row from M/2 up
        # read out of range and returned zero: w4a6 and w6a6_tc produced an output whose
        # upper half was entirely zeros (s18_kadv.py, 64/128 rows). No gate saw it because
        # every A/B correctness gate in this project uses uniform operands on purpose.
        _a_nrec = fx.Int64(c_m * (K * elem_bytes // a_pack_addr))
        # 4 bytes/element under atomic split-K: C is an fp32 accumulator there, not bf16.
        _c_elem_bytes = 4 if _atomic_split else 2
        _c_nrec = fx.Int64(c_m * c_n * _c_elem_bytes)

        def _ptr_buffer_resource(ptr, num_records_bytes=None, offset_bytes=None):
            addr = fx.ptrtoint(ptr)
            addr_i64 = fx.arith.index_cast(T.i64, addr)
            if const_expr(offset_bytes is not None):
                addr_i64 = addr_i64 + fx.arith.index_cast(T.i64, offset_bytes)
            if num_records_bytes is None:
                return buffer_ops.create_buffer_resource_from_addr(addr_i64)
            return buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=num_records_bytes
            )

        # Split-K operand offsets. Sliding each operand's BASE address to the start of this
        # workgroup's K slice leaves every k index inside the loop unchanged -- they are all
        # relative to the base -- so the slicing costs one add per operand instead of threading a
        # k_base through the whole load path. The offsets are the per-slice byte counts of each
        # layout, all compile-time except the block index:
        #   A   rows are K long                        -> bytes(K slice) per row
        #   B   is preshuffled (n0, k0, 4, 16, kpack), where one k0 unit is 64 B of K for each of
        #       16 N lanes                             -> 16 * bytes(K slice)
        #   scales are (row group, K/256, 64) i32, i.e. one byte per element of K per row
        #                                               -> the K slice itself
        # Each n0 block and row group has full-K extent, so one uniform base shift lands every one
        # of them on the same slice. num_records shrinks by the same amount it advanced, keeping the
        # last valid byte -- and therefore the M-tail masking -- exactly where it was.
        _split_off_a = _split_off_b = _split_off_sc = _split_off_c = None
        if const_expr(split_k > 1):
            bz = gpu.block_id("z")
            _split_off_a = bz * ((_k_loop * elem_bytes) // a_pack_addr)
            _split_off_b = bz * (16 * ((_k_loop * elem_bytes) // b_elem_vec_pack))
            _split_off_sc = bz * _k_loop
            if const_expr(not _atomic_split):
                # C is written as split_k partial planes for the caller to sum; bound each
                # workgroup to its own plane so a stray store cannot land in another split's
                # output. Under "atomic" there is exactly one plane and every workgroup adds
                # into it, so the offset stays None on purpose.
                _split_off_c = bz * (c_m * c_n * 2)

        def _shrink_nrec(nrec, offset_bytes):
            if const_expr(offset_bytes is None):
                return nrec
            return nrec - fx.arith.index_cast(T.i64, offset_bytes)

        a_rsrc = _ptr_buffer_resource(
            arg_a, _shrink_nrec(_a_nrec, _split_off_a), _split_off_a
        )
        c_rsrc = _ptr_buffer_resource(arg_c, _c_nrec, _split_off_c)
        _needs_per_token_scale = not is_f16_or_bf16 and not is_fp4
        scale_a_rsrc = None
        if const_expr(not is_f16_or_bf16):
            if const_expr(is_fp4):
                _scale_a_rows = (c_m + fx.Index(31)) // fx.Index(32)
                _scale_a_stride_elems = fx.Index((K // (32 * 4 * 2)) * 64)
                _scale_a_nrec = fx.Int64(
                    _scale_a_rows * _scale_a_stride_elems * fx.Index(4)
                )
            else:
                _scale_a_nrec = fx.Int64(c_m * fx.Index(4))
            # only the block-scaled family indexes scales along K; fp6/fp8 scales are one value
            # per token or channel, shared by every K slice
            _sc_off = _split_off_sc if is_fp4 else None
            scale_a_rsrc = _ptr_buffer_resource(
                arg_scale_a, _shrink_nrec(_scale_a_nrec, _sc_off), _sc_off
            )

        # ---- Bias buffer resource (for fused epilogue) ----
        # Use max_size=True so the buffer descriptor's size is taken from the
        # actual arg_bias tensor; this avoids hardcoding the output element
        # size (was c_n * 2, which broke if out_dtype became fp32 etc.).
        bias_rsrc = None
        if const_expr(_has_bias):
            bias_rsrc = _ptr_buffer_resource(arg_bias)
        b_rsrc = _ptr_buffer_resource(arg_b, None, _split_off_b)
        scale_b_rsrc = (
            None
            if (is_f16_or_bf16)
            else _ptr_buffer_resource(
                arg_scale_b, None, _split_off_sc if is_fp4 else None
            )
        )
        # NVFP4->MXFP6 weight transcode LUT (12-bit: byte -> two 6-bit fp6 codes),
        # 8 mantissa rows * 256 bytes = 2048 i32 = 8192 B. L1-cached; read-only.
        lut_rsrc = (
            _ptr_buffer_resource(arg_lut, fx.Int64(8 * 256 * 4))
            if want_lut
            else None
        )
        # LDS-resident copy of the transcode LUT (see sizing note above). The
        # 2048 i32 entries are copied cooperatively from global once at kernel
        # start; the K-loop transcode then reads from LDS (ds_read) instead of
        # global, freeing the vmem/L2 path for the fp4 weight loads.
        lut_lds_ptr = None
        _lut_row_view = None
        memref_lut = None
        _lut_idx = None
        if const_expr(want_lut):
            from flydsl._mlir.dialects import memref as memref_lut
            from flydsl._mlir.extras import types as _mlirT
            from flydsl.expr.gpu import lds_space as _lds_space_fn
            from flydsl.utils.smem_allocator import (
                get_index_value as _gidx,
                get_op_result_or_value as _gval,
            )

            lut_lds_ptr = SmemPtr(
                base_ptr_pong, lut_lds_offset, T.i32, shape=(8 * 256,)
            )
            for _c in range_constexpr((8 * 256) // 256):
                _si = tx + fx.Index(_c * 256)
                _v = buffer_ops.buffer_load(
                    lut_rsrc, fx.Int32(_si), vec_width=1, dtype=fx.Int32
                )
                lut_lds_ptr.store(_v, [_si])
            gpu.barrier()

            _lut_row_ty = _mlirT.memref(
                256, _mlirT.i32(), memory_space=_lds_space_fn()
            )

            def _lut_idx(i32v):
                return _gidx(i32v)

            def _lut_row_view(mant_bits):
                # Per-block LDS view of the mantissa row (256 i32). byte offset =
                # lut_lds_offset + mant_bits * 256 * 4. Folds the row index out of
                # the 16 per-code lookups (removes 16 `| base` ORs / block).
                off_i32 = ArithValue(fx.Int32(lut_lds_offset)) + (
                    ArithValue(mant_bits) * ArithValue(fx.Int32(256 * 4))
                )
                return _gval(
                    memref_lut.view(
                        _lut_row_ty, base_ptr_pong, _gidx(off_i32), sizes=[]
                    )
                )

        # ── NVFP4 -> MXFP6 transcode primitives ──────────────────────────────
        # Position-preserving: the fp4 code at logical element p is widened to the fp6 code
        # at the same p (the MFMA maps element p to the same K for fp4/fp6; only the bit
        # width changes), so the transcode is layout-agnostic.
        #
        # These live here, at kernel-body scope, rather than inside compute_tile because
        # w4a4_tc calls the same two functions from the A global->LDS staging copy, which
        # runs outside compute_tile. One definition, two callers: a second copy for the A
        # side could drift from the B side and the numerics gate compares them against each
        # other, so it would be comparing a bug to itself.
        def _ci(v):
            return arith.constant(v, type=T.i32)

        def _val(x):
            if hasattr(x, "ir_value"):
                return x.ir_value()
            if hasattr(x, "value"):
                return x.value
            return x

        def _tc_e4m3_parts(b):
            """(mantissa bits, E8M0 byte at bump=0) for one E4M3 scale byte.

            A subnormal byte -- exponent field 0 -- encodes k * 2^-9 with no implicit leading
            one, and an implicit leading one is the single thing every consumer downstream
            assumes it has. Renormalizing here instead of branching there keeps that assumption
            true: k = 2^p * (1 + f) makes the value (1 + f) * 2^(p-9), a significand in [1, 2)
            like every other byte. The 8 LUT rows, the (1 + mb/8) multiply in the hardware path
            and the bump all then work on subnormals untouched, and the whole extension costs
            two compares and a select on a value that is already in a register.

            This matters because it is the bottom three octaves of the book. Weight blocks ask
            for scales down to ~0.0019 (kernel/p128_sweep_vs_e2e.py) against a normals-only
            floor of 2^-6 = 0.0156, so 88-99% of them clamp; subnormals reach 2^-9 = 0.00195
            and essentially none of them do.
            """
            mant = b & _ci(7)
            ebits = (b >> _ci(3)) & _ci(0xF)
            if const_expr(not _TC_E4M3_SUB):
                return mant, ebits + _ci(120)
            sub = arith.cmpi(CmpIPredicate.eq, _val(ebits), _ci(0))
            ge4 = arith.cmpi(CmpIPredicate.uge, _val(mant), _ci(4))
            ge2 = arith.cmpi(CmpIPredicate.uge, _val(mant), _ci(2))
            # p = floor(log2 k), so the renormalizing shift is 3 - p. Spelled as a select over
            # the three constant shifts rather than a variable one: same two compares either
            # way, and the constant form does not need p as a value. k = 0 is a zero scale,
            # whose elements are all zero, so the p = 0 arm it lands in cannot be observed.
            smant = arith.select(
                ge4,
                _val((ArithValue(_val(mant)) << _ci(1)) & _ci(7)),
                arith.select(
                    ge2,
                    _val((ArithValue(_val(mant)) << _ci(2)) & _ci(7)),
                    _val((ArithValue(_val(mant)) << _ci(3)) & _ci(7)),
                ),
            )
            # normals (ebits - 7) + 127 = ebits + 120; subnormals (p - 9) + 127 = p + 118
            sexp = arith.select(ge4, _ci(120), arith.select(ge2, _ci(119), _ci(118)))
            return (
                ArithValue(arith.select(sub, smant, _val(mant))),
                ArithValue(arith.select(sub, sexp, _val(ebits + _ci(120)))),
            )

        def _tc_mant_bits(sc, opsel_byte, opsel_dyn):
            # Mantissa nibble of the selected E4M3 scale byte. The B side knows its byte at
            # trace time (the fragment loop is unrolled); the A staging side does not -- its
            # selector depends on the staged row and k-offset, both runtime. The two are
            # separate parameters rather than one sniffed argument because the loop indices
            # here are constexpr objects, not python ints, so isinstance() picks the wrong
            # arm and the static case dies inside ArithValue.
            s = ArithValue(sc)
            if const_expr(opsel_dyn is None):
                b = (s >> _ci(int(opsel_byte) * 8)) & _ci(0xFF)
            else:
                b = (s >> (ArithValue(_val(opsel_dyn)) * _ci(8))) & _ci(0xFF)
            if const_expr(not _TC_E4M3_SUB):
                return b & _ci(7)
            return _tc_e4m3_parts(b)[0]

        def _tc_bscale_e4m3_to_e8m0(sc):
            # sc packs up to 4 NVFP4 E4M3 scale bytes (opsel-selected).
            # Convert each to MXFP6 E8M0: e8m0 = e4m3_exp + 120 + bump,
            # bump = (mantissa_bits != 0). (== exp + bump + 127 - 7.)
            if const_expr(_TC_PROBE_NOSCALE):
                # PERF PROBE (wrong numerics): hand the MFMA the raw E4M3 word.
                return _val(sc)
            s = ArithValue(sc)
            if const_expr(_TC_SCALE_BYTEWISE and not _TC_E4M3_SUB):
                # All four bytes at once. The per-byte form below extracts, converts and
                # reassembles each byte separately -- a compare and a select among them --
                # for ~4x the instructions, and it is called once per A row-repeat AND once
                # per B fragment per k-tile, so w4a4_tc pays it on both operands. Nothing
                # here can carry across a byte lane: exp <= 15 and bump <= 1 put the sum at
                # most 136, and mant + 7 <= 14 stays inside the low nibble, which is what
                # makes bit 3 of that sum exactly (mant != 0).
                #
                # Subnormals cannot ride this: renormalizing needs a per-byte compare on the
                # exponent field, so _TC_E4M3_SUB takes the per-byte loop below instead.
                mant = s & _ci(0x07070707)
                bump = ((mant + _ci(0x07070707)) >> _ci(3)) & _ci(0x01010101)
                exp = (s >> _ci(3)) & _ci(0x0F0F0F0F)
                return _val(exp + bump + _ci(0x78787878))
            out = None
            for _bi in range_constexpr(4):
                b = (s >> _ci(_bi * 8)) & _ci(0xFF)
                mant, e8_base = _tc_e4m3_parts(b)
                bump = arith.select(
                    arith.cmpi(CmpIPredicate.ne, _val(mant), _ci(0)),
                    _ci(1),
                    _ci(0),
                )
                e8 = ((e8_base + bump) & _ci(0xFF)) << _ci(_bi * 8)
                out = e8 if out is None else (out | e8)
            return _val(out)

        def _tc_data_fp4_to_fp6_hw(b0, b1, sc, opsel_byte=0, *, opsel_dyn=None):
            # HWCVT: same result as the LUT path, computed instead of looked up.
            #
            # The LUT stores round(mant * |E2M1| / 2^bump) per (mant, code). Here
            # the product is reassociated so the 2^-bump factor rides free on the
            # convert's scale operand (which honours only the operand's exponent),
            # leaving the mantissa (1 + mb/8) as one f16 multiply:
            #   fp4 --cvt(scale=2^-bump)--> f16 --* (1+mb/8)--> f16 --cvt--> fp6
            # Costs 16 decodes + 16 packed multiplies + 1 pack per 32 codes, and
            # zero LDS reads (the LUT path spends 16 ds_read + ~30 shift/OR).
            #
            # Returns 8 i32: the 32 codes densely packed as fp6 in words 0..5, then two
            # zero words. Words 0..5 are exactly the 24 bytes the MXFP6 MFMA fragment
            # wants, which is why the A staging path can write them straight to LDS.
            from flydsl._mlir.dialects import rocdl as _rocdl_raw

            mant_bits = _tc_mant_bits(sc, opsel_byte, opsel_dyn)
            nz = arith.cmpi(CmpIPredicate.ne, _val(mant_bits), _ci(0))
            # scale operand = 2^-bump (bump = mant != 0); exponent-only, exact
            sc_f32 = arith.select(
                nz,
                arith.constant(0.5, type=T.f32),
                arith.constant(1.0, type=T.f32),
            )
            # mantissa factor (1 + mb/8), exact in f16 for all 8 values of mb
            mf_f32 = None
            if const_expr(_TC_MF_INT and not _TC_FOLDSCALE):
                # Build the f16 bit pattern rather than computing it: exponent field 15 is
                # 2^0, and mantissa field mb*128 is mb/8 exactly (mb <= 7, so it cannot
                # reach the implicit bit). Both halves of the pk_mul operand are filled in
                # place, which also removes the broadcast copy the f32 form needed. Four
                # arithmetic converts per block become two shifts and an or.
                _mx = ArithValue(_val(mant_bits)) << _ci(7)
                _mw = (_mx | (_mx << _ci(16))) | _ci(0x3C003C00)
                mf2 = vector.bitcast(
                    T.vec(2, T.f16),
                    vector.from_elements(T.vec(1, T.i32), [_val(_mw)]),
                )
            else:
                mf_f32 = ArithValue(
                    arith.uitofp(T.f32, _val(mant_bits))
                ) * ArithValue(arith.constant(0.125, type=T.f32)) + ArithValue(
                    arith.constant(1.0, type=T.f32)
                )
                mf_f16 = arith.truncf(T.f16, _val(mf_f32))
                mf2 = vector.from_elements(T.vec(2, T.f16), [mf_f16, mf_f16])
            # Ride the mantissa on a convert's scale operand instead of 16 multiplies.
            pack_sc = arith.constant(1.0, type=T.f32)
            if const_expr(_TC_FOLDSCALE == 1):
                sc_f32 = _val(ArithValue(_val(mf_f32)) * ArithValue(sc_f32))
            elif const_expr(_TC_FOLDSCALE == 2):
                pack_sc = _val(ArithValue(_val(mf_f32)) * ArithValue(sc_f32))
                sc_f32 = arith.constant(1.0, type=T.f32)

            v2 = vector.from_elements(T.vec(2, T.i64), [b0, b1])
            w4 = vector.bitcast(T.vec(4, T.i32), v2)
            f16x2_ty = T.vec(2, T.f16)
            halves = []
            for j in range_constexpr(4):
                wj = vector.extract(w4, static_position=[j], dynamic_position=[])
                for sel in range_constexpr(4):
                    # one instruction per 2 nibbles: this is the widest fp4->f16
                    # decode gfx950 has (pk8/pk16 are gfx1250-only)
                    # srcSelIndex is an I32Attr, so it takes a python int
                    p = _rocdl_raw.cvt_scalef32_pk_f16_fp4(f16x2_ty, wj, sc_f32, sel)
                    if const_expr(not (_TC_PROBE_NOMUL or _TC_FOLDSCALE)):
                        p = arith.mulf(p, mf2)
                    halves.append(p)
            # Concatenate 16 x f16x2 -> f16x32 for the single packing convert.
            #
            # At tile_m=256 this region carries 128 v_mov_b32 per unrolled body (141
            # against w4a6's 13) because the pack wants its 32 halves in 16 consecutive
            # registers and at 250 VGPRs the allocator copies instead. Re-describing the
            # assembly at dword granularity (bitcast a vec<16xi32>) does NOT remove them:
            # both forms canonicalise to identical ISA, same 974 VALU, 4 more VGPRs. The
            # copies are register pressure, not an artifact of this expression, so they
            # are not addressable from here. tile_m<=128 does not pay them at all.
            elems = []
            for p in halves:
                for e in range_constexpr(2):
                    elems.append(
                        vector.extract(p, static_position=[e], dynamic_position=[])
                    )
            h32 = vector.from_elements(T.vec(32, T.f16), elems)
            r6 = _rocdl_raw.cvt_scalef32_pk32_fp6_f16(T.vec(6, T.i32), h32, pack_sc)
            # MFMA wants 8 i32 lanes; the fp6 fragment fills 6, rest zero
            z = _ci(0)
            wv = [
                vector.extract(r6, static_position=[i], dynamic_position=[])
                for i in range_constexpr(6)
            ] + [z, z]
            return vector.from_elements(T.vec(8, T.i32), wv)

        def _tc_bdata_fp4_to_fp6(b0, b1, sc, opsel_byte, *, opsel_dyn=None):
            # b0,b1: i64 ir (16 fp4 bytes = 32 codes). sc: packed E4M3.
            # opsel_byte: python int selecting this block's scale byte.
            # opsel_dyn: same selector as an IR value, for the A staging caller, whose byte
            # depends on the staged row and k-offset (both runtime).
            # VALU-min: fold the mantissa row into a per-block LDS view (removes
            # the 16x `| base_lut` index ORs); top byte of each i32 word skips
            # the &0xFF mask (>>24 already <256).
            #
            # Kernel-body scope, like the hardware path above and for the same reason: the A
            # staging copy calls it from outside compute_tile, and a second copy for the A
            # side could drift from the B side while the numerics gate compares the two.
            mant_bits = _tc_mant_bits(sc, opsel_byte, opsel_dyn)
            row = _lut_row_view(mant_bits)
            v2 = vector.from_elements(T.vec(2, T.i64), [b0, b1])
            w4 = vector.bitcast(T.vec(4, T.i32), v2)
            pairs = []
            for j in range_constexpr(4):
                wj = ArithValue(
                    vector.extract(w4, static_position=[j], dynamic_position=[])
                )
                idxs = [
                    wj & _ci(0xFF),
                    (wj >> _ci(8)) & _ci(0xFF),
                    (wj >> _ci(16)) & _ci(0xFF),
                    wj >> _ci(24),
                ]
                for idx in idxs:
                    if const_expr(_TC_PROBE_NOLUT):
                        # PERF PROBE (wrong numerics): skip the LDS lookup,
                        # feed the raw index into the packer. Isolates the
                        # LUT/LDS cost from the dense-pack VALU cost.
                        pairs.append(ArithValue(idx) & _ci(0xFFF))
                    else:
                        pairs.append(
                            ArithValue(memref_lut.load(row, [_lut_idx(idx)]))
                        )
            words = [None] * 6
            for t in range_constexpr(16):
                pair = pairs[t]
                bitpos = t * 12
                w = bitpos // 32
                off = bitpos % 32
                lo = pair << _ci(off) if off else pair
                words[w] = lo if words[w] is None else (words[w] | lo)
                if off + 12 > 32:
                    hi = pair >> _ci(32 - off)
                    words[w + 1] = (
                        hi if words[w + 1] is None else (words[w + 1] | hi)
                    )
            z = _ci(0)
            wv = [(w if w is not None else z) for w in words] + [z, z]
            return vector.from_elements(T.vec(8, T.i32), wv)

        bx_m = bx * tile_m
        by_n = by * tile_n

        # ---- Wave / lane decomposition ----
        wave_size = 64
        layout_wave_lane = fx.make_layout((num_waves_const, wave_size), (64, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        row_a_lds = lane_mod_16
        kpack_elems = 16 if elem_bytes == 1 else 8
        col_offset_base = lane_div_16 * kpack_elems
        col_offset_base_bytes = (
            col_offset_base if elem_bytes == 1 else col_offset_base * elem_bytes
        )

        m_repeat = tile_m // 16
        k_unroll = tile_k_bytes // a_elem_vec_pack // 64

        num_waves = num_waves_const
        n_per_wave = tile_n // num_waves
        num_acc_n = n_per_wave // 16

        n_tile_base = wave_id * n_per_wave

        n_intra_list = []
        n_blk_list = []
        for i in range_constexpr(num_acc_n):
            global_n = by_n + n_tile_base + (i * 16) + lane_mod_16
            n_blk_list.append(global_n // 16)
            n_intra_list.append(global_n % 16)

        # ── B load helpers ────────────────────────────────────────────────
        def load_b_pack(base_k, ki_step, ni):
            return load_b_pack_k32(
                buffer_ops,
                fx.arith,
                fx.vector,
                arg_b=arg_b,
                b_rsrc=b_rsrc,
                layout_b=layout_b,
                base_k=base_k,
                ki_step=ki_step,
                n_blk=n_blk_list[ni],
                n_intra=n_intra_list[ni],
                lane_div_16=lane_div_16,
                elem_type=_elem_type(),
                kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes,
                unpack_int4=is_int4,
            )

        c64_b = 64

        _b_stride_n0_c = fx.Index(_stride_n0)
        _b_stride_k0_c = fx.Index(_stride_k0)
        _b_stride_klane_c = fx.Index(_stride_klane)
        _b_stride_nlane_c = fx.Index(_stride_nlane)

        _b_dword_stride_n0 = _stride_n0 // 4
        _b_dword_stride_k0 = _stride_k0 // 4
        _b_dword_stride_klane = _stride_klane // 4
        _b_dword_stride_nlane = _stride_nlane // 4

        _b_n_full_dword_list = []
        for _ni in range_constexpr(num_acc_n):
            _n_dword = (
                n_blk_list[_ni] * fx.Index(_b_dword_stride_n0)
                + n_intra_list[_ni] * fx.Index(_b_dword_stride_nlane)
                + lane_div_16 * fx.Index(_b_dword_stride_klane)
            )
            _b_n_full_dword_list.append(_n_dword)

        _b_dword_stride_k0_c = fx.Index(_b_dword_stride_k0)
        _c64_elem = fx.Index(64 // elem_bytes * b_elem_vec_pack)

        def _extract_b_packs(b16):
            b_i64x2 = Vec(b16).bitcast(fx.Int64)
            b0_i64 = b_i64x2[0]
            b1_i64 = b_i64x2[1]
            if const_expr(not is_f16_or_bf16 or use_mfma_k32):
                return b0_i64.ir_value(), b1_i64.ir_value()
            b0_v1 = Vec.from_elements([b0_i64], fx.Int64)
            b1_v1 = Vec.from_elements([b1_i64], fx.Int64)
            if const_expr(is_f16):
                return b0_v1.bitcast(fx.Float16), b1_v1.bitcast(fx.Float16)
            return b0_v1.bitcast(fx.Int16), b1_v1.bitcast(fx.Int16)

        def _load_b_single(k_dword_offset, ni):
            """Load one 16B B vector using pre-computed k dword offset."""
            dword_idx = _b_n_full_dword_list[ni] + k_dword_offset
            dword_idx_i32 = fx.Int32(dword_idx)
            b_vec4 = buffer_ops.buffer_load(
                b_rsrc, dword_idx_i32, vec_width=4, dtype=fx.Int32
            )
            b16 = Vec(b_vec4).bitcast(_elem_dtype())
            return _extract_b_packs(b16)

        def load_b_packs_k64(base_k, ku: int, ni: int):
            if const_expr(is_int4):
                ki0 = (ku * 2) + 0
                ki1 = (ku * 2) + 1
                return load_b_pack(base_k, ki0, ni), load_b_pack(base_k, ki1, ni)

            base_k_bytes = base_k * elem_bytes
            k0 = base_k_bytes // c64_b + ku
            idx_pack = (
                n_blk_list[ni] * _b_stride_n0_c
                + k0 * _b_stride_k0_c
                + lane_div_16 * _b_stride_klane_c
                + n_intra_list[ni] * _b_stride_nlane_c
            )
            vec_elems = 16 if elem_bytes == 1 else 8
            b16 = _buffer_load_vec(
                buffer_ops,
                fx.vector,
                b_rsrc,
                idx_pack,
                elem_type=_elem_type(),
                vec_elems=vec_elems,
                elem_bytes=elem_bytes,
                offset_in_bytes=(elem_bytes == 1),
            )
            return _extract_b_packs(b16)

        def load_b_tile(base_k):
            if const_expr((not is_int4) and (not is_f16_or_bf16)):
                base_k_bytes = base_k * elem_bytes
                k0_base = base_k_bytes // c64_b
                k_dwords = []
                for ku in range_constexpr(k_unroll):
                    k_dwords.append((k0_base + ku) * _b_dword_stride_k0_c)
                packs0_per_ku = [[] for _ in range(k_unroll)]
                packs1_per_ku = [[] for _ in range(k_unroll)]
                for ni in range_constexpr(num_acc_n):
                    for ku in range_constexpr(k_unroll):
                        b0, b1 = _load_b_single(k_dwords[ku], ni)
                        packs0_per_ku[ku].append(b0)
                        packs1_per_ku[ku].append(b1)
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    b_tile.append((packs0_per_ku[ku], packs1_per_ku[ku]))
                return b_tile

            packs0_per_ku = [[] for _ in range(k_unroll)]
            packs1_per_ku = [[] for _ in range(k_unroll)]
            for ni in range_constexpr(num_acc_n):
                for ku in range_constexpr(k_unroll):
                    b0, b1 = load_b_packs_k64(base_k, ku, ni)
                    packs0_per_ku[ku].append(b0)
                    packs1_per_ku[ku].append(b1)
            b_tile = []
            for ku in range_constexpr(k_unroll):
                b_tile.append((packs0_per_ku[ku], packs1_per_ku[ku]))
            return b_tile

        # ── A LDS load/store helpers (now take lds_buffer memref directly) ──
        lds_base_zero = fx.Index(0)

        _lds_k_dim_c = fx.Index(lds_k_dim)

        def lds_load_16b(curr_row_a_lds, col_base, lds_buffer):
            col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
            col_base_swz = (
                col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes // 2)
            )
            idx_a16 = curr_row_a_lds * _lds_k_dim_c + col_base_swz
            return Vec.load(_vec16_type(), lds_buffer, [idx_a16])

        def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
            loaded_a16 = lds_load_16b(curr_row_a_lds, col_base, lds_buffer)
            a_i64x2 = Vec(loaded_a16).bitcast(fx.Int64)
            a0_i64 = a_i64x2[0]
            a1_i64 = a_i64x2[1]

            if const_expr(not is_f16_or_bf16 or use_mfma_k32):
                return a0_i64.ir_value(), a1_i64.ir_value()

            a0_v1 = Vec.from_elements([a0_i64], fx.Int64)
            a1_v1 = Vec.from_elements([a1_i64], fx.Int64)
            if const_expr(is_f16):
                return a0_v1.bitcast(fx.Float16), a1_v1.bitcast(fx.Float16)
            return a0_v1.bitcast(fx.Int16), a1_v1.bitcast(fx.Int16)

        # ── A global→reg load ─────────────────────────────────────────────
        num_a_loads = bytes_per_thread_a // a_load_bytes
        tile_k_dwords = (
            (tile_k * 2) // 4 if elem_bytes == 2 else tile_k // 4 // a_pack_glob
        )
        layout_a_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))
        c4 = fx.Index(4)
        # Surplus threads mirror a staging thread instead of indexing past the tile (see
        # a_stage_threads). Left as a bare `tx` when every thread stages, so the IR of every
        # already-measured configuration is unchanged rather than merely equivalent.
        tx_stage = tx if a_stage_threads == total_threads else tx % fx.Index(a_stage_threads)
        tx_i32_base = tx_stage * c4

        def load_a_16(idx_elem):
            return buffer_copy_gmem16_dwordx4(
                buffer_ops,
                fx.vector,
                elem_type=_elem_type(),
                idx_i32=idx_elem,
                rsrc=a_rsrc,
                vec_elems=(16 if elem_bytes == 1 else 8),
                elem_bytes=elem_bytes,
            )

        def a_tile_chunk_coord_i32(i: int):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_base,
                i=i,
                # The chunk stride is the number of threads that actually advance through the
                # tile, not the workgroup size, or chunk i>0 would skip the mirrored half.
                total_threads=a_stage_threads,
                layout_tile_div4=layout_a_tile_div4,
            )

        # ── w4a4_tc: the A scale byte that belongs to a staged 32-code group ──────
        # A's scales live in the same preshuffled layout the MFMA reads (built above as
        # _scale_a_base_elems): a group of 32 rows x 256 K owns 64 consecutive i32 words,
        # word = kb4*16 + r16 inside it, and the byte inside the word = half*2 + imxdl.
        #
        # The staging copy must read the SAME byte the MFMA will later hand the hardware as
        # that fragment's E8M0 exponent, because the two halves of one scale are applied in
        # different places: the E4M3 mantissa is folded into the data here, the exponent is
        # applied by the MFMA there. Reading different bytes would silently pair one block's
        # mantissa with another's exponent.
        _a_sc_row_stride = ((K // 256) * 64) if is_w4a4_tc else 0
        if const_expr(is_w4a4_tc and (int(tile_m) % 32 != 0)):
            raise NotImplementedError(
                f"w4a4_tc needs tile_m divisible by 32 (scale row groups), got {tile_m}"
            )

        def _a_stage_scale(row_a_local, col_a_local_i32, base_k):
            rg = bx_m + row_a_local
            kb4 = col_a_local_i32 // c4  # which 32-code block of this k-tile
            word = (
                (rg // fx.Index(32)) * fx.Index(_a_sc_row_stride)
                + (base_k // fx.Index(256)) * fx.Index(64)
                + kb4 * fx.Index(16)
                + (rg % fx.Index(16))
            )
            byte_sel = ((base_k // fx.Index(128)) % fx.Index(2)) * fx.Index(2) + (
                (rg % fx.Index(32)) // fx.Index(16)
            )
            sc = buffer_ops.buffer_load(
                scale_a_rsrc, word, vec_width=1, dtype=fx.Int32
            )
            return sc, fx.Int32(byte_sel)

        def load_a_tile(base_k_div4, base_k):
            parts = []
            for i in range_constexpr(num_a_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                row_a_global = bx_m + row_a_local
                idx_i32 = row_a_global * _k_div4_factor + (
                    base_k_div4 + col_a_local_i32
                )
                idx_elem = idx_i32 if elem_bytes == 1 else idx_i32 * 2
                a_16B = load_a_16(idx_elem)
                if const_expr(is_w4a4_tc):
                    # Issue the scale load next to its data load so one waitcnt covers both.
                    sc, byte_sel = _a_stage_scale(row_a_local, col_a_local_i32, base_k)
                    parts.append((Vec(a_16B).bitcast(fx.Int32), sc, byte_sel))
                else:
                    parts.append(Vec(a_16B).bitcast(fx.Int32))
            return parts

        def store_a_tile_to_lds(vec_a_parts, lds_buffer):
            for i in range_constexpr(num_a_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                col_local_bytes = col_a_local_i32 * c4
                col_swz_bytes = swizzle_xor16(row_a_local, col_local_bytes, k_blocks16)
                col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
                idx0 = row_a_local * _lds_k_dim_c + col_swz + lds_base_zero
                if const_expr(not is_w4a4_tc):
                    v16 = Vec(vec_a_parts[i]).bitcast(_elem_dtype())
                    v16.store(lds_buffer, [idx0])
                    continue
                # w4a4_tc: 16 B of NVFP4 (32 codes) -> 24 B of dense MXFP6, written in the
                # exact FP8-padded layout the W4A6 MFMA path already reads back. A lane's
                # 32-code fragment lives as 16 B at col and 8 B at col+64.
                #
                # The second slot's trailing 8 B is the FP8 pad. The read side loads it and
                # drops it on the floor (a2, _a3 = ...; _a3 is never used), so its contents
                # are unobservable and the ring leaving stale bytes there is fine. Writing
                # only the 8 real bytes keeps the transcode's two zero words from ever being
                # materialised, which is two fewer live VGPRs at the point where this kernel
                # is register-bound, and halves the second store.
                data_i32x4, sc_word, byte_sel = vec_a_parts[i]
                a_i64x2 = Vec(data_i32x4).bitcast(fx.Int64)
                _tc_a = _tc_bdata_fp4_to_fp6 if const_expr(a_lut) else _tc_data_fp4_to_fp6_hw
                f6 = _tc_a(
                    a_i64x2[0].ir_value(),
                    a_i64x2[1].ir_value(),
                    sc_word,
                    0,
                    opsel_dyn=byte_sel,
                )
                _nw = 8 if const_expr(_TC_WRITE_PAD) else 6
                _w = [
                    vector.extract(f6, static_position=[j], dynamic_position=[])
                    for j in range_constexpr(_nw)
                ]
                col_swz_hi = swizzle_xor16(
                    row_a_local, col_local_bytes + fx.Index(64), k_blocks16
                )
                idx1 = row_a_local * _lds_k_dim_c + col_swz_hi + lds_base_zero
                Vec.from_elements(_w[0:4], fx.Int32).bitcast(_elem_dtype()).store(
                    lds_buffer, [idx0]
                )
                Vec.from_elements(_w[4:_nw], fx.Int32).bitcast(_elem_dtype()).store(
                    lds_buffer, [idx1]
                )
                if const_expr(_TC_A_SERIAL):
                    # Each staged block needs 16 consecutive VGPRs for the pk32 pack's f16
                    # source. Interleaving the blocks makes those aligned tuples co-exist,
                    # and this kernel is register-bound, not latency-bound -- the staging
                    # copy is amortised over the whole workgroup. Keep the peaks apart.
                    rocdl.sched_barrier(0)

        # ── A DMA async: direct global→LDS transfer ─────────────────────
        num_a_async_loads = bytes_per_thread_a // a_async_load_bytes
        tx_i32_async_base = tx * a_async_load_dword
        k_bytes_factor = K * elem_bytes // a_pack_addr

        def a_tile_chunk_coord_i32_async(i: int):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_async_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_a_tile_div4,
                chunk_i32=a_async_load_dword,
            )

        def dma_a_tile_to_lds(
            base_k_div4,
            lds_buffer,
            *,
            wave_id_v,
            wave_size_v,
            dma_bytes_v,
            num_a_async_loads_v,
            a_tile_chunk_coord_i32_async_fn,
            c4_v,
            k_blocks16_v,
            bx_m_v,
            k_bytes_factor_v,
            total_threads_v,
            a_rsrc_v,
        ):
            from flydsl._mlir.dialects import memref as memref_dialect

            wave_offset = rocdl.readfirstlane(
                fx.Int64.ir_type,
                fx.Int64(wave_id_v * fx.Index(wave_size_v * dma_bytes_v)),
            )
            lds_base = memref_dialect.extract_aligned_pointer_as_index(lds_buffer)
            lds_ptr_base = buffer_ops.create_llvm_ptr(
                fx.Int64(lds_base), address_space=3
            )
            lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, wave_offset)

            for i in range_constexpr(num_a_async_loads_v):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32_async_fn(i)
                col_a_local_sw = swizzle_xor16(
                    row_a_local, col_a_local_i32 * c4_v, k_blocks16_v
                )
                row_a_global = bx_m_v + row_a_local
                global_byte_idx = row_a_global * k_bytes_factor_v + (
                    base_k_div4 * c4_v + col_a_local_sw
                )
                global_offset = fx.Int32(global_byte_idx)

                if const_expr(i > 0):
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr,
                        static_byte_offset=total_threads_v * dma_bytes_v,
                    )

                size_i32 = fx.Int32(dma_bytes_v)
                soffset = fx.Int32(0)
                offset_imm = fx.Int32(0)
                aux = fx.Int32(1)

                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc_v,
                    lds_ptr,
                    size_i32,
                    global_offset,
                    soffset,
                    offset_imm,
                    aux,
                )

        def prefetch_a_to_lds(
            base_k, lds_buffer, *, a_elem_vec_pack_v, dma_a_tile_to_lds_fn
        ):
            # Same correction as prefetch_a_tile: A's k-tile stride follows A's storage
            # packing. The parameter is kept (many call sites pass it) but ignored in favour
            # of a_pack_glob, which is the only correct divisor for every dtype.
            base_k_div4 = base_k // 4 // a_pack_addr
            dma_a_tile_to_lds_fn(
                base_k_div4,
                lds_buffer,
                wave_id_v=wave_id,
                wave_size_v=wave_size,
                dma_bytes_v=a_async_load_bytes,
                num_a_async_loads_v=num_a_async_loads,
                a_tile_chunk_coord_i32_async_fn=a_tile_chunk_coord_i32_async,
                c4_v=c4,
                k_blocks16_v=k_blocks16,
                bx_m_v=bx_m,
                k_bytes_factor_v=k_bytes_factor,
                total_threads_v=total_threads,
                a_rsrc_v=a_rsrc,
            )

        def prefetch_a_tile(base_k):
            # a_pack_glob, not a_elem_vec_pack: for the w4a6 family A is 1 byte/element, so
            # the k-tile base advances 128 B per tile_k=128, not 64. Advancing by half meant
            # consecutive k-tiles overlapped by 64 B -- the kernel re-read the first half of
            # every A row, never read the second half, and mis-parsed the dense 6-bit codes
            # because the fragment no longer started on a K=32 block boundary.
            base_k_bytes = base_k * elem_bytes // a_pack_addr
            base_k_div4 = base_k_bytes // 4
            return load_a_tile(base_k_div4, base_k)

        def prefetch_b_tile(base_k):
            base_k_packed = base_k // b_elem_vec_pack if b_elem_vec_pack > 1 else base_k
            return load_b_tile(base_k_packed)

        def prefetch_ab_tile(base_k):
            a_regs = prefetch_a_tile(base_k)
            b_regs = prefetch_b_tile(base_k)
            return a_regs, b_regs

        # ── FP4 scale pre-fetch (outside compute_tile for latency hiding) ──
        _fp4_tilek128 = False

        def load_fp4_scale_chunk(_base_k):
            raise RuntimeError("load_fp4_scale_chunk called when is_fp4=False")

        if const_expr(is_fp4):
            _fp4_pack_M_outer = 2
            _fp4_pack_N_outer = 2
            _fp4_pack_K_outer = 2
            _fp4_tilek128 = int(tile_k) == 128
            _fp4_scale_chunk_k = 32 * 4 * _fp4_pack_K_outer
            _K1_outer = K // (32 * 4 * _fp4_pack_K_outer)
            _k_unroll_packed_outer = (
                1 if _fp4_tilek128 else (k_unroll // _fp4_pack_K_outer)
            )
            _m_repeat_packed_outer = m_repeat // _fp4_pack_M_outer
            _num_acc_n_packed_outer = num_acc_n // _fp4_pack_N_outer
            _fp4_scale_k_stride = tile_k // (32 * 4 * _fp4_pack_K_outer)
            _fp4_use_scheduler = tile_m >= 64

            _scale_lane_elem_off = lane_div_16 * fx.Index(16) + lane_mod_16
            _scale_row_stride_elems = _K1_outer * 64

            _scale_a_base_elems = []
            for mi in range_constexpr(_m_repeat_packed_outer):
                mni_a = fx.Index(mi) + bx_m // fx.Index(_fp4_pack_M_outer * 16)
                _scale_a_base_elems.append(
                    mni_a * fx.Index(_scale_row_stride_elems) + _scale_lane_elem_off
                )

            _scale_b_base_elems = []
            for ni in range_constexpr(_num_acc_n_packed_outer):
                mni_b = fx.Index(ni) + (by_n + n_tile_base) // fx.Index(
                    _fp4_pack_N_outer * 16
                )
                _scale_b_base_elems.append(
                    mni_b * fx.Index(_scale_row_stride_elems) + _scale_lane_elem_off
                )

            _stride_k0_elems = 64

            def load_fp4_scales(base_k_scale_idx):
                a_scales, b_scales = [], []
                base_k_elem_off = base_k_scale_idx * fx.Index(_stride_k0_elems)
                for ku in range_constexpr(_k_unroll_packed_outer):
                    ku_elem_off = base_k_elem_off + fx.Index(ku * _stride_k0_elems)
                    for ni in range_constexpr(_num_acc_n_packed_outer):
                        b_scales.append(
                            buffer_ops.buffer_load(
                                scale_b_rsrc,
                                _scale_b_base_elems[ni] + ku_elem_off,
                                vec_width=1,
                                dtype=fx.Int32,
                            )
                        )
                    for mi in range_constexpr(_m_repeat_packed_outer):
                        a_scales.append(
                            buffer_ops.buffer_load(
                                scale_a_rsrc,
                                _scale_a_base_elems[mi] + ku_elem_off,
                                vec_width=1,
                                dtype=fx.Int32,
                            )
                        )
                return a_scales, b_scales

            def load_fp4_scale_chunk(base_k):
                return load_fp4_scales(base_k // fx.Index(_fp4_scale_chunk_k))

        # ── Compute tile (MFMA) ───────────────────────────────────────────
        def compute_tile(
            accs_in,
            b_tile_in,
            lds_buffer,
            *,
            is_last_tile=False,
            a0_prefetch=None,
            fp4_scales=None,
            fp4_scale_half=0,
        ):
            scales_pf = {}
            if const_expr(is_last_tile and (not is_f16_or_bf16)):
                s_b_vals = []
                for ni in range_constexpr(num_acc_n):
                    col_g = by_n + n_tile_base + (ni * 16) + lane_mod_16
                    s_b_vals.append(
                        buffer_ops.buffer_load(
                            scale_b_rsrc, col_g, vec_width=1, dtype=fx.Float32
                        )
                    )
                scales_pf["s_b_vals"] = s_b_vals
                scales_pf["s_a_vecs"] = []
                row_off_base = lane_div_16 * 4
                for mi in range_constexpr(m_repeat):
                    row_base_m = bx_m + (mi * 16)
                    row_g_base = row_base_m + row_off_base
                    s_a_vec = buffer_ops.buffer_load(
                        scale_a_rsrc, row_g_base, vec_width=4, dtype=fx.Float32
                    )
                    scales_pf["s_a_vecs"].append(Vec(s_a_vec))

            current_accs_list = list(accs_in)

            use_mfma_scale_128 = (
                str(gpu_arch).startswith("gfx95")
                and (not is_int8)
                and (not is_int4)
                and (not is_f16_or_bf16)
            )
            if const_expr(use_mfma_scale_128):
                if const_expr((int(tile_k) % 128) != 0):
                    raise ValueError(
                        f"tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_k}"
                    )
                mfma_res_ty = Vec.make_type(4, fx.Float32)
                c0_i64 = fx.Int64(0)

                # W4A6: A=MXFP6 -> cbsz=2 (E2M3), B=MXFP4 -> blgp=4 (E2M1).
                # W6A6_TC: A=MXFP6 (cbsz=2) and B transcoded fp4->MXFP6 -> blgp=2.
                _fp4_cbsz = 2 if is_w4a6 else (4 if is_fp4 else 0)
                # W6A6: B is stored fp6, so it selects the same 6-bit format the transcode
                # arms produce in registers -- blgp=2 either way, different provenance.
                _fp4_blgp = 2 if (is_w6a6_tc or is_w6a6) else (4 if is_fp4 else 0)
                _fp4_pack_M = 2 if is_fp4 else 1
                _fp4_pack_N = 2 if is_fp4 else 1
                _fp4_pack_K = 2 if is_fp4 else 1
                _quant_block_size = 32
                _K1 = K // (_quant_block_size * 4 * _fp4_pack_K) if is_fp4 else 1
                _k_unroll_packed = k_unroll // _fp4_pack_K
                _m_repeat_packed = m_repeat // _fp4_pack_M
                _num_acc_n_packed = num_acc_n // _fp4_pack_N

                def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                    return Vec.from_elements([x0, x1, x2, x3], fx.Int64).bitcast(
                        fx.Int32
                    )

                # The fp4->fp6 data transcode and the E4M3->E8M0 scale transcode are
                # defined at kernel-body scope (search `_tc_data_fp4_to_fp6_hw`): w4a4_tc
                # calls them from the A staging copy too. Only the LUT arm stays here,
                # because only B ever uses it.
                if const_expr(is_fp4):
                    _fp4_a_sc, _fp4_b_sc = fp4_scales if fp4_scales else ([], [])
                    ku128_iters = 1 if _fp4_tilek128 else _k_unroll_packed
                    ikxdl_iters = 1 if _fp4_tilek128 else _fp4_pack_K
                    for ku128 in range_constexpr(ku128_iters):
                        a_scale_base = 0 if _fp4_tilek128 else ku128 * _m_repeat_packed
                        b_scale_base = 0 if _fp4_tilek128 else ku128 * _num_acc_n_packed
                        # W6A6_TC: transcode every B fragment (fp4->fp6) and its
                        # scale (E4M3->E8M0) ONCE per tile here, independent of the
                        # A-row (mi) loop, then reuse across all mi. Doing it inside
                        # the mi loop would re-transcode each B block m_repeat times.
                        _tc_b = {}
                        _tc_bs = {}
                        if const_expr(is_w6a6_tc):
                            for ni_p in range_constexpr(_num_acc_n_packed):
                                _bsv = _fp4_b_sc[b_scale_base + ni_p]
                                _tc_bs[ni_p] = _tc_bscale_e4m3_to_e8m0(_bsv)
                                for ikxdl in range_constexpr(ikxdl_iters):
                                    _kidx = (
                                        0
                                        if _fp4_tilek128
                                        else ku128 * _fp4_pack_K + ikxdl
                                    )
                                    _bp0, _bp1 = b_tile_in[_kidx]
                                    _sks = fp4_scale_half if _fp4_tilek128 else ikxdl
                                    for inxdl in range_constexpr(_fp4_pack_N):
                                        _nidx = ni_p * _fp4_pack_N + inxdl
                                        _ops = _sks * _fp4_pack_N + inxdl
                                        if const_expr(_TC_PROBE_NOCVT):
                                            # PERF PROBE (wrong numerics): no transcode at
                                            # all, feed the raw fp4 packs to the fp6 MFMA.
                                            # Leaves every other difference from w4a6 in
                                            # place (blgp=2, scale routing), so it isolates
                                            # the transcode arithmetic itself.
                                            _tc_b[(ni_p, ikxdl, inxdl)] = (
                                                pack_i64x4_to_i32x8(
                                                    _bp0[_nidx], _bp1[_nidx],
                                                    c0_i64, c0_i64,
                                                )
                                            )
                                        else:
                                            _tc_fn = (
                                                _tc_data_fp4_to_fp6_hw
                                                if hwcvt
                                                else _tc_bdata_fp4_to_fp6
                                            )
                                            _tc_b[(ni_p, ikxdl, inxdl)] = _tc_fn(
                                                _bp0[_nidx], _bp1[_nidx], _bsv, _ops
                                            )
                        for mi_p in range_constexpr(_m_repeat_packed):
                            a_scale_val = _fp4_a_sc[a_scale_base + mi_p]
                            if const_expr(is_w4a4_tc):
                                # A is stored NVFP4, so its scale word is E4M3 and the MFMA
                                # wants MXFP6's E8M0. Only the exponent (+bump) is applied
                                # here: the E4M3 mantissa was already folded into the A data
                                # by the staging transcode. Hoisted above the ni loop, so it
                                # costs m_repeat_packed conversions per k-tile, not one per
                                # MFMA.
                                a_scale_val = _tc_bscale_e4m3_to_e8m0(a_scale_val)
                            for ni_p in range_constexpr(_num_acc_n_packed):
                                b_scale_val = _fp4_b_sc[b_scale_base + ni_p]
                                # W6A6_TC: reuse the pre-transcoded E8M0 scale.
                                b_scale_mfma = (
                                    _tc_bs[ni_p]
                                    if const_expr(is_w6a6_tc)
                                    else b_scale_val
                                )
                                for ikxdl in range_constexpr(ikxdl_iters):
                                    k_idx = (
                                        0
                                        if _fp4_tilek128
                                        else ku128 * _fp4_pack_K + ikxdl
                                    )
                                    b_packs0, b_packs1 = b_tile_in[k_idx]
                                    # W6A6's B needs 24 real bytes per lane, and the B
                                    # loader hands out 16 per k64 group, so the third dword
                                    # pair is the first half of the next group -- the same
                                    # 16/8 split across a 64 B stride that A reads out of
                                    # LDS, and for the same reason: the tile is FP8-padded
                                    # fp6, 24 real + 8 zero per 32 elements.
                                    b_packs2 = (
                                        b_tile_in[k_idx + 1][0]
                                        if const_expr(is_w6a6)
                                        else None
                                    )
                                    col_base = (
                                        col_offset_base_bytes
                                        if _fp4_tilek128
                                        else (
                                            col_offset_base_bytes
                                            + fx.Index((k_idx * 128) // a_elem_vec_pack)
                                        )
                                    )
                                    scale_k_sel = (
                                        fp4_scale_half if _fp4_tilek128 else ikxdl
                                    )
                                    for imxdl in range_constexpr(_fp4_pack_M):
                                        mi_idx = mi_p * _fp4_pack_M + imxdl
                                        curr_row_a_lds = row_a_lds + (mi_idx * 16)
                                        a0 = fx.Int64(0).ir_value()
                                        a1 = fx.Int64(0).ir_value()
                                        if const_expr(
                                            (a0_prefetch is not None)
                                            and (k_idx == 0)
                                            and (mi_idx == 0)
                                            and (not is_w4a6)
                                        ):
                                            a0, a1 = a0_prefetch
                                        else:
                                            a0, a1 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base, lds_buffer
                                            )
                                        if const_expr(is_w4a6):
                                            # A is MXFP6 (FP8-padded 32 B/lane): the two
                                            # k64 halves live 64 B apart in the fp6 LDS.
                                            # Read 24 real bytes (a0,a1,a2); drop pad a3.
                                            a2, _a3 = lds_load_packs_k64(
                                                curr_row_a_lds,
                                                col_base + 64,
                                                lds_buffer,
                                            )
                                            a128 = pack_i64x4_to_i32x8(
                                                a0, a1, a2, c0_i64
                                            )
                                        else:
                                            a128 = pack_i64x4_to_i32x8(
                                                a0, a1, c0_i64, c0_i64
                                            )
                                        for inxdl in range_constexpr(_fp4_pack_N):
                                            ni_idx = ni_p * _fp4_pack_N + inxdl
                                            b0 = b_packs0[ni_idx]
                                            b1 = b_packs1[ni_idx]
                                            _opsel_b = (
                                                scale_k_sel * _fp4_pack_N + inxdl
                                            )
                                            if const_expr(is_w6a6_tc):
                                                # Reuse the per-tile pre-transcode.
                                                b128 = _tc_b[(ni_p, ikxdl, inxdl)]
                                            elif const_expr(is_w6a6):
                                                b128 = pack_i64x4_to_i32x8(
                                                    b0, b1, b_packs2[ni_idx], c0_i64
                                                )
                                            else:
                                                b128 = pack_i64x4_to_i32x8(
                                                    b0, b1, c0_i64, c0_i64
                                                )
                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            if const_expr(not _fp4_use_scheduler):
                                                rocdl.sched_barrier(0)
                                            current_accs_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        b128,
                                                        current_accs_list[acc_idx],
                                                        _fp4_cbsz,
                                                        _fp4_blgp,
                                                        scale_k_sel * _fp4_pack_M
                                                        + imxdl,
                                                        a_scale_val,
                                                        _opsel_b,
                                                        b_scale_mfma,
                                                    ],
                                                )
                                            )
                else:
                    for ku128 in range_constexpr(k_unroll // 2):
                        ku0 = ku128 * 2
                        ku1 = ku0 + 1
                        b0_packs0, b0_packs1 = b_tile_in[ku0]
                        b1_packs0, b1_packs1 = b_tile_in[ku1]
                        col_base0 = col_offset_base_bytes + (ku0 * 64)
                        col_base1 = col_offset_base_bytes + (ku1 * 64)

                        for mi in range_constexpr(m_repeat):
                            curr_row_a_lds = row_a_lds + (mi * 16)
                            a0 = fx.Int64(0).ir_value()
                            a1 = fx.Int64(0).ir_value()
                            if const_expr(
                                (a0_prefetch is not None) and (ku0 == 0) and (mi == 0)
                            ):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(
                                    curr_row_a_lds, col_base0, lds_buffer
                                )
                            a2, a3 = lds_load_packs_k64(
                                curr_row_a_lds, col_base1, lds_buffer
                            )
                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                            for ni in range_constexpr(num_acc_n):
                                b128 = pack_i64x4_to_i32x8(
                                    b0_packs0[ni],
                                    b0_packs1[ni],
                                    b1_packs0[ni],
                                    b1_packs1[ni],
                                )
                                acc_idx = mi * num_acc_n + ni
                                # fp6 (MXFP6-E2M3) selects the 6-bit format via
                                # cbsz/blgp=2; fp8 uses 0. Scale operand stays
                                # E8M0=1.0 (0x7F): block scales applied per-token
                                # in the epilogue (same path as fp8).
                                _f6cb = 2 if is_fp6 else 0
                                current_accs_list[acc_idx] = (
                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        mfma_res_ty,
                                        [
                                            a128,
                                            b128,
                                            current_accs_list[acc_idx],
                                            _f6cb,
                                            _f6cb,
                                            0,
                                            0x7F7F7F7F,
                                            0,
                                            0x7F7F7F7F,
                                        ],
                                    )
                                )
                return current_accs_list, scales_pf

            mfma_res_ty = Vec.make_type(4, fx.Int32 if is_int8 else fx.Float32)
            if const_expr(use_mfma_k32):
                mfma_fn_k32 = (
                    rocdl.mfma_f32_16x16x32_f16
                    if is_f16
                    else rocdl.mfma_f32_16x16x32_bf16
                )

                def i64x2_to_v8(lo, hi):
                    return Vec.from_elements([lo, hi], fx.Int64).bitcast(
                        fx.Float16 if is_f16 else fx.BFloat16
                    )

                def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                    av = i64x2_to_v8(a0, a1)
                    bv = i64x2_to_v8(b0, b1)
                    return mfma_fn_k32(mfma_res_ty, [av, bv, acc_in, 0, 0, 0])

            else:
                if const_expr(is_int8):
                    mfma_fn = mfma_i32_k32
                elif const_expr(is_f16):
                    mfma_fn = rocdl.mfma_f32_16x16x16f16
                elif const_expr(is_bf16):
                    mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
                else:
                    mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

                def mfma_step(acc_in, a, b):
                    return mfma_fn(mfma_res_ty, [a, b, acc_in, 0, 0, 0])

                def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                    acc_mid = mfma_step(acc_in, a0, b0)
                    return mfma_step(acc_mid, a1, b1)

            for ku in range_constexpr(k_unroll):
                b_packs0, b_packs1 = b_tile_in[ku]
                ki64 = ku * 64
                col_base = col_offset_base_bytes + ki64
                for mi in range_constexpr(m_repeat):
                    curr_row_a_lds = row_a_lds + (mi * 16)
                    a0 = fx.Int64(0).ir_value()
                    a1 = fx.Int64(0).ir_value()
                    if const_expr(
                        (a0_prefetch is not None) and (ku == 0) and (mi == 0)
                    ):
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_packs_k64(
                            curr_row_a_lds, col_base, lds_buffer
                        )
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        current_accs_list[acc_idx] = mfma_k64_bytes(
                            current_accs_list[acc_idx],
                            a0,
                            a1,
                            b_packs0[ni],
                            b_packs1[ni],
                        )
            return current_accs_list, scales_pf

        # ── Epilogue (store output) ───────────────────────────────────────
        def store_output(final_accs, scales):
            s_b_vals = []
            s_a_vecs = []
            if const_expr(not (is_f16_or_bf16 or is_fp4)):
                s_b_vals = scales["s_b_vals"]
                s_a_vecs = scales["s_a_vecs"]

            if const_expr(use_cshuffle_epilog):
                if const_expr(lds_out is None):
                    raise RuntimeError(
                        "use_cshuffle_epilog=True but lds_out is not allocated."
                    )
                gpu.barrier()

                def write_row_to_lds(
                    *,
                    mi,
                    ii,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n,
                    lds_out,
                ):
                    s_a = fx.Float32(1.0)
                    if const_expr(_needs_per_token_scale):
                        s_a_vec4 = s_a_vecs[mi]
                        s_a = Vec(s_a_vec4)[ii]
                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = Vec(acc)[ii]
                        if const_expr(is_int8):
                            val = fx.Float32(val)
                        if const_expr(is_f16_or_bf16 or is_fp4):
                            val_s = val
                        elif const_expr(_needs_per_token_scale):
                            val_s = (val * s_a) * s_b_vals[ni]
                        else:
                            val_s = val
                        v16 = _out_dtype()(val_s)

                        lds_idx = row_base_lds + col_local
                        v1 = Vec.from_elements([v16], _out_dtype())
                        v1.store(lds_out, [lds_idx], alignment=2)

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    idx_out = row * c_n + col_g0
                    byte_off = idx_out * 2
                    e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                    if const_expr(e_vec == 4):
                        frag_i32x2 = Vec(frag).bitcast(fx.Int32)
                        buffer_ops.buffer_store(
                            frag_i32x2, c_rsrc, byte_off, offset_is_bytes=True
                        )
                    else:
                        frag_i32x1 = Vec(frag).bitcast(fx.Int32)
                        frag_i32 = frag_i32x1[0]
                        buffer_ops.buffer_store(
                            frag_i32, c_rsrc, byte_off, offset_is_bytes=True
                        )

                e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                mfma_epilog(
                    use_cshuffle=True,
                    arith=fx.arith,
                    vector=fx.vector,
                    gpu=gpu,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=e_vec,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    write_row_to_lds=write_row_to_lds,
                    store_pair=store_pair,
                    frag_elem_type=_out_elem(),
                )
                return

            def body_row(*, mi, ii, row_in_tile, row):
                s_a = fx.Float32(1.0)
                if const_expr(_needs_per_token_scale):
                    s_a_vec4 = s_a_vecs[mi]
                    s_a = Vec(s_a_vec4)[ii]
                col_base = by_n + n_tile_base + lane_mod_16
                idx_base = row * c_n + col_base
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    acc = final_accs[acc_idx]
                    val = Vec(acc)[ii]
                    if const_expr(is_int8):
                        val = fx.Float32(val)
                    if const_expr(is_f16_or_bf16 or is_fp4):
                        val_s = val
                    elif const_expr(_needs_per_token_scale):
                        val_s = (val * s_a) * s_b_vals[ni]
                    else:
                        val_s = val

                    # ── Fused epilogue: bias + activation ──
                    if const_expr(_has_bias and bias_rsrc is not None):
                        col_idx = col_base + (ni * 16)
                        bias_val_f16 = buffer_ops.buffer_load(
                            bias_rsrc, col_idx, vec_width=1, dtype=_out_dtype()
                        )
                        bias_val_f32 = fx.Float32(bias_val_f16)
                        val_s = val_s + bias_val_f32

                    if const_expr(_has_relu):
                        # ReLU(x) = max(x, 0). Use maximumf rather than
                        # cmpf+select: the lower-level cmpf wrapper requires
                        # an integer CmpFPredicate enum value, not the string
                        # "ogt", so the previous form failed at compile time
                        # the moment the bias_relu epilogue was actually
                        # exercised (test coverage gap).
                        zero_f32 = fx.Float32(0.0)
                        val_s = fx.Float32(val_s).maximumf(zero_f32)
                    elif const_expr(_has_silu):
                        # SiLU(x) = x * sigmoid(x). Compute as
                        #   sigmoid_x = 1 / (1 + exp(-x))    # one rcp instead of fdiv
                        #   val_s    = val_s * sigmoid_x
                        # to lower to v_rcp_f32 + v_mul_f32 instead of v_div_*
                        # (~4x faster than fdiv on AMD GPUs).
                        neg_one = fx.Float32(-1.0)
                        neg_val = val_s * neg_one
                        exp_neg = math.exp(neg_val)
                        one_f32 = fx.Float32(1.0)
                        denom = one_f32 + exp_neg
                        sigmoid_x = one_f32 / denom
                        val_s = val_s * sigmoid_x
                    elif const_expr(_has_gelu):
                        # GeLU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        # math.tanh has no AMD libcall, so expand it via exp.
                        # Numerically stable form using only non-positive
                        # exponent (avoids fp32 overflow for large |x|):
                        #   a = -2 * |y|              (a <= 0, exp(a) in [0,1])
                        #   tanh(y) = sign(y) * (1 - exp(a)) / (1 + exp(a))
                        #   1 + tanh(y) = 1 + sign(y) * (1 - exp(a))/(1+exp(a))
                        # We compute (1 + tanh(y)) directly from y because we
                        # need the GeLU output, which is half * x * (1 + tanh).
                        half_f32 = fx.Float32(0.5)
                        coeff_f32 = fx.Float32(0.044715)
                        sqrt2pi_f32 = fx.Float32(0.7978845608)
                        neg_two_f32 = fx.Float32(-2.0)
                        one_f32 = fx.Float32(1.0)
                        zero_f32 = fx.Float32(0.0)
                        x3 = val_s * val_s * val_s
                        y = sqrt2pi_f32 * (val_s + coeff_f32 * x3)
                        # |y| via max(y, -y) — avoids math.absf dependency
                        neg_y = zero_f32 - y
                        abs_y = fx.Float32(y).maximumf(neg_y)
                        # exp(-2|y|) is in [0, 1], no overflow.
                        e_neg2abs = math.exp(neg_two_f32 * abs_y)
                        denom = one_f32 + e_neg2abs
                        # tanh(|y|) = (1 - e_neg2abs) / denom
                        # tanh(y)   = sign(y) * tanh(|y|)
                        # 1 + tanh(y):
                        #   y >= 0: 1 + tanh(|y|) = (denom + (1 - e)) / denom
                        #                         = (2)             / denom
                        #                          (because denom = 1 + e and
                        #                           denom + 1 - e = 2)
                        #   y <  0: 1 - tanh(|y|) = (denom - (1 - e)) / denom
                        #                         = (2 * e)          / denom
                        two_f32 = fx.Float32(2.0)
                        # numerator = 2          when y >= 0
                        #           = 2 * e_neg2abs  when y <  0
                        sign_pred = y > zero_f32
                        num_pos = two_f32
                        num_neg = two_f32 * e_neg2abs
                        numerator = sign_pred.select(num_pos, num_neg)
                        recip = one_f32 / denom
                        one_plus_tanh = numerator * recip
                        val_s = half_f32 * val_s * one_plus_tanh

                    idx_out = idx_base + (ni * 16)
                    if const_expr(_atomic_split):
                        # One fp32 atomic per element into the single shared C plane. Scalar
                        # rather than packed: buffer_atomic_pk_add_bf16 exists on gfx950 and
                        # would halve the transaction count, but it would also make bf16 the
                        # accumulation type across all split_k partials, and the point of
                        # this path is to keep the combine in fp32.
                        # The offset must be a plain i32 byte count. idx_base is built from
                        # Index-typed values (row * c_n + col_base), and handing that straight
                        # to the intrinsic fails MLIR verification with an empty message, so
                        # the cast is explicit.
                        _boff = fx.arith.index_cast(T.i32, idx_out * 4)
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_s, c_rsrc, _boff, fx.Int32(0), fx.Int32(0),
                        )
                    else:
                        val_f16 = _out_dtype()(val_s)
                        buffer_ops.buffer_store(val_f16, c_rsrc, idx_out)

            mfma_epilog(
                use_cshuffle=False,
                arith=fx.arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=body_row,
            )

        # ── Scheduling hints ──────────────────────────────────────────────
        rocdl.sched_barrier(0)

        def hot_loop_scheduler():
            def _build_scheduler(numer: int, denom: int):
                if const_expr(denom <= 0):
                    return []
                if const_expr(numer <= 0):
                    return [0] * denom
                out = []
                prev = 0
                for i in range_constexpr(denom):
                    cur = ((i + 1) * numer + (denom - 1)) // denom
                    out.append(cur - prev)
                    prev = cur
                return out

            if const_expr(_is_gfx942):
                mfma_group = num_acc_n
                mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)
                if const_expr(num_acc_n < 4):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)
                dswr_tail = num_a_loads
                dstr_advance = 2
                if const_expr(dswr_tail > sche_iters):
                    dswr_tail = sche_iters
                dswr_start = max(sche_iters - dswr_tail - dstr_advance, 0)
                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if const_expr(sche_i >= dswr_start - 1):
                        rocdl.sched_dswr(1)
            else:
                mfma_group = num_acc_n
                if const_expr(use_mfma_k32):
                    element_k_per_mfma = 32
                elif const_expr(_is_gfx950):
                    element_k_per_mfma = 128
                else:
                    element_k_per_mfma = 32
                num_mfma_per_tile_k = tile_k // element_k_per_mfma
                mfma_total = num_mfma_per_tile_k * m_repeat * mfma_group
                num_ds_load = num_a_lds_load
                # w4a4_tc splits each staged 16 B global chunk into two LDS stores (the
                # MXFP6 fragment's 16 B + 8 B halves live 64 B apart), so the ds_write tail
                # is twice the global load count, not equal to it.
                dswr_tail = num_a_loads * (2 if is_w4a4_tc else 1)
                dstr_advance = 2
                if const_expr(dswr_tail > mfma_total):
                    dswr_tail = mfma_total
                num_gmem_loads = num_b_loads + num_a_async_loads
                if const_expr(is_fp4 and tile_k != 128):
                    num_fp4_scale_k_groups = (
                        1 if int(tile_k) == 128 else (k_unroll // 2)
                    )
                    num_a_scale_loads = num_fp4_scale_k_groups * (m_repeat // 2)
                    num_b_scale_loads = num_fp4_scale_k_groups * (num_acc_n // 2)
                    num_gmem_loads += num_a_scale_loads + num_b_scale_loads
                dsrd_preload_eff = min(int(dsrd_preload), num_ds_load)
                dvmem_preload_eff = min(int(dvmem_preload), num_gmem_loads)
                vmem_remaining = num_gmem_loads - dvmem_preload_eff
                dsrd_remaining = num_ds_load - dsrd_preload_eff
                vmem_schedule = []
                if const_expr(vmem_remaining > 0 and vmem_remaining < mfma_total):
                    vmem_schedule = _build_scheduler(vmem_remaining, vmem_remaining) + [
                        0
                    ] * (mfma_total - vmem_remaining)
                else:
                    vmem_schedule = _build_scheduler(vmem_remaining, mfma_total)
                dsrd_schedule = _build_scheduler(dsrd_remaining, mfma_total)
                dswr_start = max(mfma_total - dswr_tail - dstr_advance, 0)
                last_dsrd_mfma_idx = -1
                for sched_idx in range_constexpr(mfma_total):
                    if const_expr(dsrd_schedule[sched_idx]):
                        last_dsrd_mfma_idx = sched_idx
                dswr_start = max(dswr_start, last_dsrd_mfma_idx + 1)
                idx_ds_read = dsrd_preload_eff
                idx_gmem_load = dvmem_preload_eff
                idx_ds_write = 0
                if const_expr(dvmem_preload_eff):
                    rocdl.sched_vmem(dvmem_preload_eff)
                if const_expr(dsrd_preload_eff):
                    rocdl.sched_dsrd(dsrd_preload_eff)
                for mfma_idx in range_constexpr(mfma_total):
                    rocdl.sched_mfma(1)
                    n_dsrd = dsrd_schedule[mfma_idx]
                    if const_expr(n_dsrd and (idx_ds_read < num_ds_load)):
                        if const_expr(idx_ds_read + n_dsrd > num_ds_load):
                            n_dsrd = num_ds_load - idx_ds_read
                        if const_expr(n_dsrd):
                            rocdl.sched_dsrd(n_dsrd)
                            idx_ds_read += n_dsrd

                    n_vmem = vmem_schedule[mfma_idx]
                    if const_expr(n_vmem and (idx_gmem_load < num_gmem_loads)):
                        if const_expr(idx_gmem_load + n_vmem > num_gmem_loads):
                            n_vmem = num_gmem_loads - idx_gmem_load
                        if const_expr(n_vmem):
                            rocdl.sched_vmem(n_vmem)
                            idx_gmem_load += n_vmem
                    if const_expr(
                        (not use_async_copy)
                        and (idx_ds_write < dswr_tail)
                        and (mfma_idx >= dswr_start)
                    ):
                        rocdl.sched_dswr(1)
                        idx_ds_write += 1
                # if any other ds_write is not issued, issue here.
                _n_dswr_total = num_a_loads * (2 if is_w4a4_tc else 1)
                if const_expr((not use_async_copy) and (idx_ds_write < _n_dswr_total)):
                    rocdl.sched_dswr(_n_dswr_total - idx_ds_write)
                # for ds_write_idx in range_constexpr(num_a_loads):
                #     rocdl.sched_dswr(1)

            rocdl.sched_barrier(0)

        # ── Main pipeline ─────────────────────────────────────────────────
        def _flatten_b_tile(bt):
            flat = []
            for packs0, packs1 in bt:
                flat.extend(packs0)
                flat.extend(packs1)
            return flat

        def _unflatten_b_tile(flat):
            bt = []
            idx = 0
            for _ in range_constexpr(k_unroll):
                p0 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                p1 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                bt.append((p0, p1))
            return bt

        n_accs = num_acc_n * m_repeat
        n_btile = k_unroll * 2 * num_acc_n
        n_a0pf = 2
        n_fp4_asc = 0
        n_fp4_bsc = 0

        if const_expr(is_fp4):
            n_fp4_asc = _k_unroll_packed_outer * _m_repeat_packed_outer
            n_fp4_bsc = _k_unroll_packed_outer * _num_acc_n_packed_outer

        def _pack_state(accs_l, bt_flat, a0pf, fp4_scales=None, *, is_fp4_v):
            state = list(accs_l) + list(bt_flat) + [a0pf[0], a0pf[1]]
            if const_expr(is_fp4_v):
                a_scales, b_scales = fp4_scales
                state.extend(a_scales)
                state.extend(b_scales)
            return state

        def _unpack_state(
            vals, *, n_accs_v, n_btile_v, n_a0pf_v, is_fp4_v, n_fp4_asc_v, n_fp4_bsc_v
        ):
            accs_l = list(vals[:n_accs_v])
            bt_flat = list(vals[n_accs_v : n_accs_v + n_btile_v])
            a0pf = (vals[n_accs_v + n_btile_v], vals[n_accs_v + n_btile_v + 1])
            if const_expr(not is_fp4_v):
                return accs_l, bt_flat, a0pf, None
            sc_base = n_accs_v + n_btile_v + n_a0pf_v
            a_scales = list(vals[sc_base : sc_base + n_fp4_asc_v])
            b_scales = list(
                vals[sc_base + n_fp4_asc_v : sc_base + n_fp4_asc_v + n_fp4_bsc_v]
            )
            return accs_l, bt_flat, a0pf, (a_scales, b_scales)

        def _build_pingpong_body(
            k_iv,
            inner_state,
            *,
            _unpack_state,
            _unflatten_b_tile,
            _fp4_tilek128,
            tile_k,
            use_async_copy,
            prefetch_a_to_lds,
            a_elem_vec_pack,
            dma_a_tile_to_lds,
            prefetch_a_tile,
            prefetch_b_tile,
            compute_tile,
            lds_a_pong,
            lds_a_ping,
            store_a_tile_to_lds,
            hot_loop_scheduler,
            num_b_loads,
            gpu,
            prefetch_a0_pack,
            load_fp4_scale_chunk,
            is_fp4,
            rocdl,
            _pack_state,
            _flatten_b_tile,
            lds_load_packs_k64,
            row_a_lds,
            col_offset_base_bytes,
            n_accs,
            n_btile,
            n_a0pf,
            n_fp4_asc,
            n_fp4_bsc,
        ):
            accs_in, bt_flat_in, a0pf_in, fp4_scales_pong_in = _unpack_state(
                inner_state,
                n_accs_v=n_accs,
                n_btile_v=n_btile,
                n_a0pf_v=n_a0pf,
                is_fp4_v=is_fp4,
                n_fp4_asc_v=n_fp4_asc,
                n_fp4_bsc_v=n_fp4_bsc,
            )
            b_tile_pong_in = _unflatten_b_tile(bt_flat_in)

            if const_expr(_fp4_tilek128):
                next_k1 = k_iv + tile_k
                if const_expr(use_async_copy):
                    prefetch_a_to_lds(
                        next_k1,
                        lds_a_ping,
                        a_elem_vec_pack_v=a_elem_vec_pack,
                        dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                    )
                else:
                    a_tile_ping = prefetch_a_tile(next_k1)
                b_tile_ping = prefetch_b_tile(next_k1)
                accs_in, _ = compute_tile(
                    accs_in,
                    b_tile_pong_in,
                    lds_a_pong,
                    a0_prefetch=a0pf_in,
                    fp4_scales=fp4_scales_pong_in,
                    fp4_scale_half=0,
                )
                if const_expr(not use_async_copy):
                    store_a_tile_to_lds(a_tile_ping, lds_a_ping)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                a0_prefetch_ping = prefetch_a0_pack(
                    lds_a_ping,
                    lds_load_packs_k64_fn=lds_load_packs_k64,
                    row_a_lds_v=row_a_lds,
                    col_offset_base_bytes_v=col_offset_base_bytes,
                )

                next_k2 = k_iv + (tile_k * 2)
                _sc_ping = load_fp4_scale_chunk(next_k2) if is_fp4 else None
                rocdl.sched_barrier(0)
                if const_expr(use_async_copy):
                    prefetch_a_to_lds(
                        next_k2,
                        lds_a_pong,
                        a_elem_vec_pack_v=a_elem_vec_pack,
                        dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                    )
                else:
                    a_tile_pong = prefetch_a_tile(next_k2)
                b_tile_pong_new = prefetch_b_tile(next_k2)
                accs_in, _ = compute_tile(
                    accs_in,
                    b_tile_ping,
                    lds_a_ping,
                    a0_prefetch=a0_prefetch_ping,
                    fp4_scales=fp4_scales_pong_in,
                    fp4_scale_half=1,
                )
                if const_expr(not use_async_copy):
                    store_a_tile_to_lds(a_tile_pong, lds_a_pong)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                a0_prefetch_pong_new = prefetch_a0_pack(
                    lds_a_pong,
                    lds_load_packs_k64_fn=lds_load_packs_k64,
                    row_a_lds_v=row_a_lds,
                    col_offset_base_bytes_v=col_offset_base_bytes,
                )

                return _pack_state(
                    accs_in,
                    _flatten_b_tile(b_tile_pong_new),
                    a0_prefetch_pong_new,
                    _sc_ping,
                    is_fp4_v=is_fp4,
                )

            next_k1 = k_iv + tile_k
            if const_expr(use_async_copy):
                prefetch_a_to_lds(
                    next_k1,
                    lds_a_ping,
                    a_elem_vec_pack_v=a_elem_vec_pack,
                    dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                )
            else:
                a_tile = prefetch_a_tile(next_k1)
            _sc_ping = load_fp4_scale_chunk(k_iv + fx.Index(tile_k)) if is_fp4 else None
            b_tile_ping = prefetch_b_tile(next_k1)
            accs_in, _ = compute_tile(
                accs_in,
                b_tile_pong_in,
                lds_a_pong,
                a0_prefetch=a0pf_in,
                fp4_scales=fp4_scales_pong_in,
            )
            if const_expr(not use_async_copy):
                store_a_tile_to_lds(a_tile, lds_a_ping)
            hot_loop_scheduler()
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
            a0_prefetch_ping = prefetch_a0_pack(
                lds_a_ping,
                lds_load_packs_k64_fn=lds_load_packs_k64,
                row_a_lds_v=row_a_lds,
                col_offset_base_bytes_v=col_offset_base_bytes,
            )

            next_k2 = k_iv + (tile_k * 2)
            if const_expr(use_async_copy):
                prefetch_a_to_lds(
                    next_k2,
                    lds_a_pong,
                    a_elem_vec_pack_v=a_elem_vec_pack,
                    dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                )
            else:
                a_tile = prefetch_a_tile(next_k2)
            _sc_pong = load_fp4_scale_chunk(k_iv + (tile_k * 2)) if is_fp4 else None
            b_tile_pong_new = prefetch_b_tile(next_k2)
            accs_in, _ = compute_tile(
                accs_in,
                b_tile_ping,
                lds_a_ping,
                a0_prefetch=a0_prefetch_ping,
                fp4_scales=_sc_ping,
            )
            if const_expr(not use_async_copy):
                store_a_tile_to_lds(a_tile, lds_a_pong)
            hot_loop_scheduler()
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
            a0_prefetch_pong_new = prefetch_a0_pack(
                lds_a_pong,
                lds_load_packs_k64_fn=lds_load_packs_k64,
                row_a_lds_v=row_a_lds,
                col_offset_base_bytes_v=col_offset_base_bytes,
            )

            return _pack_state(
                accs_in,
                _flatten_b_tile(b_tile_pong_new),
                a0_prefetch_pong_new,
                _sc_pong,
                is_fp4_v=is_fp4,
            )

        if const_expr(lds_stage == 2):

            def prefetch_a0_pack(
                lds_buffer,
                *,
                lds_load_packs_k64_fn,
                row_a_lds_v,
                col_offset_base_bytes_v,
            ):
                return lds_load_packs_k64_fn(
                    row_a_lds_v, col_offset_base_bytes_v, lds_buffer
                )

            k0 = fx.Index(0)
            b_tile0 = prefetch_b_tile(k0)
            if const_expr(use_async_copy):
                prefetch_a_to_lds(
                    k0,
                    lds_a_pong,
                    a_elem_vec_pack_v=a_elem_vec_pack,
                    dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                )
            else:
                store_a_tile_to_lds(prefetch_a_tile(k0), lds_a_pong)
            gpu.barrier()
            accs = [acc_init] * n_accs
            a0_prefetch_pong = prefetch_a0_pack(
                lds_a_pong,
                lds_load_packs_k64_fn=lds_load_packs_k64,
                row_a_lds_v=row_a_lds,
                col_offset_base_bytes_v=col_offset_base_bytes,
            )
            fp4_scales0 = load_fp4_scale_chunk(fx.Index(0)) if is_fp4 else None

            final_accs = 1
            scales = 1
            num_tiles = _k_loop // tile_k
            if const_expr(_fp4_tilek128):
                if const_expr((num_tiles % 2) == 1):
                    c_k_main = _k_loop - tile_k
                    init_state = _pack_state(
                        accs,
                        _flatten_b_tile(b_tile0),
                        a0_prefetch_pong,
                        fp4_scales0,
                        is_fp4_v=is_fp4,
                    )
                    results = init_state
                    for iv, inner in range(0, c_k_main, tile_k * 2, init=init_state):
                        results = yield _build_pingpong_body(
                            iv,
                            inner,
                            _unpack_state=_unpack_state,
                            _unflatten_b_tile=_unflatten_b_tile,
                            _fp4_tilek128=_fp4_tilek128,
                            tile_k=tile_k,
                            use_async_copy=use_async_copy,
                            prefetch_a_to_lds=prefetch_a_to_lds,
                            a_elem_vec_pack=a_elem_vec_pack,
                            dma_a_tile_to_lds=dma_a_tile_to_lds,
                            prefetch_a_tile=prefetch_a_tile,
                            prefetch_b_tile=prefetch_b_tile,
                            compute_tile=compute_tile,
                            lds_a_pong=lds_a_pong,
                            lds_a_ping=lds_a_ping,
                            store_a_tile_to_lds=store_a_tile_to_lds,
                            hot_loop_scheduler=hot_loop_scheduler,
                            num_b_loads=num_b_loads,
                            gpu=gpu,
                            prefetch_a0_pack=prefetch_a0_pack,
                            load_fp4_scale_chunk=load_fp4_scale_chunk,
                            is_fp4=is_fp4,
                            rocdl=rocdl,
                            _pack_state=_pack_state,
                            _flatten_b_tile=_flatten_b_tile,
                            lds_load_packs_k64=lds_load_packs_k64,
                            row_a_lds=row_a_lds,
                            col_offset_base_bytes=col_offset_base_bytes,
                            n_accs=n_accs,
                            n_btile=n_btile,
                            n_a0pf=n_a0pf,
                            n_fp4_asc=n_fp4_asc,
                            n_fp4_bsc=n_fp4_bsc,
                        )
                    accs, bt_flat, a0pf, fp4_scales_final = _unpack_state(
                        results,
                        n_accs_v=n_accs,
                        n_btile_v=n_btile,
                        n_a0pf_v=n_a0pf,
                        is_fp4_v=is_fp4,
                        n_fp4_asc_v=n_fp4_asc,
                        n_fp4_bsc_v=n_fp4_bsc,
                    )
                    b_tile_pong_final = _unflatten_b_tile(bt_flat)
                    final_accs, scales = compute_tile(
                        accs,
                        b_tile_pong_final,
                        lds_a_pong,
                        is_last_tile=not is_fp4,
                        a0_prefetch=a0pf,
                        fp4_scales=fp4_scales_final,
                        fp4_scale_half=0,
                    )
                else:
                    c_k_stop = _k_loop - (tile_k * 3)
                    init_state = _pack_state(
                        accs,
                        _flatten_b_tile(b_tile0),
                        a0_prefetch_pong,
                        fp4_scales0,
                        is_fp4_v=is_fp4,
                    )
                    results = init_state
                    for iv, inner in range(0, c_k_stop, tile_k * 2, init=init_state):
                        results = yield _build_pingpong_body(
                            iv,
                            inner,
                            _unpack_state=_unpack_state,
                            _unflatten_b_tile=_unflatten_b_tile,
                            _fp4_tilek128=_fp4_tilek128,
                            tile_k=tile_k,
                            use_async_copy=use_async_copy,
                            prefetch_a_to_lds=prefetch_a_to_lds,
                            a_elem_vec_pack=a_elem_vec_pack,
                            dma_a_tile_to_lds=dma_a_tile_to_lds,
                            prefetch_a_tile=prefetch_a_tile,
                            prefetch_b_tile=prefetch_b_tile,
                            compute_tile=compute_tile,
                            lds_a_pong=lds_a_pong,
                            lds_a_ping=lds_a_ping,
                            store_a_tile_to_lds=store_a_tile_to_lds,
                            hot_loop_scheduler=hot_loop_scheduler,
                            num_b_loads=num_b_loads,
                            gpu=gpu,
                            prefetch_a0_pack=prefetch_a0_pack,
                            load_fp4_scale_chunk=load_fp4_scale_chunk,
                            is_fp4=is_fp4,
                            rocdl=rocdl,
                            _pack_state=_pack_state,
                            _flatten_b_tile=_flatten_b_tile,
                            lds_load_packs_k64=lds_load_packs_k64,
                            row_a_lds=row_a_lds,
                            col_offset_base_bytes=col_offset_base_bytes,
                            n_accs=n_accs,
                            n_btile=n_btile,
                            n_a0pf=n_a0pf,
                            n_fp4_asc=n_fp4_asc,
                            n_fp4_bsc=n_fp4_bsc,
                        )
                    accs, bt_flat, a0pf, fp4_scales_ep = _unpack_state(
                        results,
                        n_accs_v=n_accs,
                        n_btile_v=n_btile,
                        n_a0pf_v=n_a0pf,
                        is_fp4_v=is_fp4,
                        n_fp4_asc_v=n_fp4_asc,
                        n_fp4_bsc_v=n_fp4_bsc,
                    )
                    b_tile_pong_ep = _unflatten_b_tile(bt_flat)

                    last_k = fx.Index(_k_loop - tile_k)
                    b_tile_ping = prefetch_b_tile(last_k)
                    if const_expr(use_async_copy):
                        prefetch_a_to_lds(
                            last_k,
                            lds_a_ping,
                            a_elem_vec_pack_v=a_elem_vec_pack,
                            dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                        )
                    else:
                        a_regs_ping = prefetch_a_tile(last_k)
                    accs, _ = compute_tile(
                        accs,
                        b_tile_pong_ep,
                        lds_a_pong,
                        a0_prefetch=a0pf,
                        fp4_scales=fp4_scales_ep,
                        fp4_scale_half=0,
                    )
                    if const_expr(not use_async_copy):
                        store_a_tile_to_lds(a_regs_ping, lds_a_ping)
                    rocdl.s_waitcnt(num_b_loads)
                    gpu.barrier()
                    a0_prefetch_ping = prefetch_a0_pack(
                        lds_a_ping,
                        lds_load_packs_k64_fn=lds_load_packs_k64,
                        row_a_lds_v=row_a_lds,
                        col_offset_base_bytes_v=col_offset_base_bytes,
                    )
                    final_accs, scales = compute_tile(
                        accs,
                        b_tile_ping,
                        lds_a_ping,
                        is_last_tile=not is_fp4,
                        a0_prefetch=a0_prefetch_ping,
                        fp4_scales=fp4_scales_ep,
                        fp4_scale_half=1,
                    )
            elif const_expr((num_tiles % 2) == 1):
                c_k_main = _k_loop - tile_k
                init_state = _pack_state(
                    accs,
                    _flatten_b_tile(b_tile0),
                    a0_prefetch_pong,
                    fp4_scales0,
                    is_fp4_v=is_fp4,
                )
                results = init_state
                for iv, inner in range(0, c_k_main, tile_k * 2, init=init_state):
                    results = yield _build_pingpong_body(
                        iv,
                        inner,
                        _unpack_state=_unpack_state,
                        _unflatten_b_tile=_unflatten_b_tile,
                        _fp4_tilek128=_fp4_tilek128,
                        tile_k=tile_k,
                        use_async_copy=use_async_copy,
                        prefetch_a_to_lds=prefetch_a_to_lds,
                        a_elem_vec_pack=a_elem_vec_pack,
                        dma_a_tile_to_lds=dma_a_tile_to_lds,
                        prefetch_a_tile=prefetch_a_tile,
                        prefetch_b_tile=prefetch_b_tile,
                        compute_tile=compute_tile,
                        lds_a_pong=lds_a_pong,
                        lds_a_ping=lds_a_ping,
                        store_a_tile_to_lds=store_a_tile_to_lds,
                        hot_loop_scheduler=hot_loop_scheduler,
                        num_b_loads=num_b_loads,
                        gpu=gpu,
                        prefetch_a0_pack=prefetch_a0_pack,
                        load_fp4_scale_chunk=load_fp4_scale_chunk,
                        is_fp4=is_fp4,
                        rocdl=rocdl,
                        _pack_state=_pack_state,
                        _flatten_b_tile=_flatten_b_tile,
                        lds_load_packs_k64=lds_load_packs_k64,
                        row_a_lds=row_a_lds,
                        col_offset_base_bytes=col_offset_base_bytes,
                        n_accs=n_accs,
                        n_btile=n_btile,
                        n_a0pf=n_a0pf,
                        n_fp4_asc=n_fp4_asc,
                        n_fp4_bsc=n_fp4_bsc,
                    )
                accs, bt_flat, a0pf, fp4_scales_final = _unpack_state(
                    results,
                    n_accs_v=n_accs,
                    n_btile_v=n_btile,
                    n_a0pf_v=n_a0pf,
                    is_fp4_v=is_fp4,
                    n_fp4_asc_v=n_fp4_asc,
                    n_fp4_bsc_v=n_fp4_bsc,
                )
                b_tile_pong_final = _unflatten_b_tile(bt_flat)
                final_accs, scales = compute_tile(
                    accs,
                    b_tile_pong_final,
                    lds_a_pong,
                    is_last_tile=not is_fp4,
                    a0_prefetch=a0pf,
                    fp4_scales=fp4_scales_final,
                )
            else:
                c_k_stop = _k_loop - (tile_k * 3)
                init_state = _pack_state(
                    accs,
                    _flatten_b_tile(b_tile0),
                    a0_prefetch_pong,
                    fp4_scales0,
                    is_fp4_v=is_fp4,
                )
                results = init_state
                for iv, inner in range(0, c_k_stop, tile_k * 2, init=init_state):
                    results = yield _build_pingpong_body(
                        iv,
                        inner,
                        _unpack_state=_unpack_state,
                        _unflatten_b_tile=_unflatten_b_tile,
                        _fp4_tilek128=_fp4_tilek128,
                        tile_k=tile_k,
                        use_async_copy=use_async_copy,
                        prefetch_a_to_lds=prefetch_a_to_lds,
                        a_elem_vec_pack=a_elem_vec_pack,
                        dma_a_tile_to_lds=dma_a_tile_to_lds,
                        prefetch_a_tile=prefetch_a_tile,
                        prefetch_b_tile=prefetch_b_tile,
                        compute_tile=compute_tile,
                        lds_a_pong=lds_a_pong,
                        lds_a_ping=lds_a_ping,
                        store_a_tile_to_lds=store_a_tile_to_lds,
                        hot_loop_scheduler=hot_loop_scheduler,
                        num_b_loads=num_b_loads,
                        gpu=gpu,
                        prefetch_a0_pack=prefetch_a0_pack,
                        load_fp4_scale_chunk=load_fp4_scale_chunk,
                        is_fp4=is_fp4,
                        rocdl=rocdl,
                        _pack_state=_pack_state,
                        _flatten_b_tile=_flatten_b_tile,
                        lds_load_packs_k64=lds_load_packs_k64,
                        row_a_lds=row_a_lds,
                        col_offset_base_bytes=col_offset_base_bytes,
                        n_accs=n_accs,
                        n_btile=n_btile,
                        n_a0pf=n_a0pf,
                        n_fp4_asc=n_fp4_asc,
                        n_fp4_bsc=n_fp4_bsc,
                    )
                accs, bt_flat, a0pf, fp4_scales_ep = _unpack_state(
                    results,
                    n_accs_v=n_accs,
                    n_btile_v=n_btile,
                    n_a0pf_v=n_a0pf,
                    is_fp4_v=is_fp4,
                    n_fp4_asc_v=n_fp4_asc,
                    n_fp4_bsc_v=n_fp4_bsc,
                )
                b_tile_pong_ep = _unflatten_b_tile(bt_flat)

                last_k = fx.Index(_k_loop - tile_k)
                b_tile_ping = prefetch_b_tile(last_k)
                if const_expr(use_async_copy):
                    prefetch_a_to_lds(
                        last_k,
                        lds_a_ping,
                        a_elem_vec_pack_v=a_elem_vec_pack,
                        dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                    )
                else:
                    a_regs_ping = prefetch_a_tile(last_k)
                _sc_last = load_fp4_scale_chunk(last_k) if is_fp4 else None
                accs, _ = compute_tile(
                    accs,
                    b_tile_pong_ep,
                    lds_a_pong,
                    a0_prefetch=a0pf,
                    fp4_scales=fp4_scales_ep,
                )
                if const_expr(not use_async_copy):
                    store_a_tile_to_lds(a_regs_ping, lds_a_ping)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                a0_prefetch_ping = prefetch_a0_pack(
                    lds_a_ping,
                    lds_load_packs_k64_fn=lds_load_packs_k64,
                    row_a_lds_v=row_a_lds,
                    col_offset_base_bytes_v=col_offset_base_bytes,
                )
                final_accs, scales = compute_tile(
                    accs,
                    b_tile_ping,
                    lds_a_ping,
                    is_last_tile=not is_fp4,
                    a0_prefetch=a0_prefetch_ping,
                    fp4_scales=_sc_last,
                )
            store_output(final_accs, scales)
        else:
            a_regs0, b_tile0 = prefetch_ab_tile(fx.Index(0))
            store_a_tile_to_lds(a_regs0, lds_a_pong)
            gpu.barrier()
            accs = [acc_init] * n_accs
            bt_flat0 = _flatten_b_tile(b_tile0)

            init_state = list(accs) + list(bt_flat0)
            for iv, state in range(0, _k_loop - tile_k, tile_k, init=init_state):
                accs_in = list(state[:n_accs])
                bt_flat_in = list(state[n_accs:])
                b_tile_in = _unflatten_b_tile(bt_flat_in)

                next_k = iv + tile_k
                a_next, b_next = prefetch_ab_tile(next_k)
                _fp4_sc = (
                    load_fp4_scales(
                        iv // fx.Index(tile_k) * fx.Index(_fp4_scale_k_stride)
                    )
                    if is_fp4
                    else None
                )
                accs_in, _ = compute_tile(
                    accs_in, b_tile_in, lds_a_pong, fp4_scales=_fp4_sc
                )
                gpu.barrier()
                store_a_tile_to_lds(a_next, lds_a_pong)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                results = yield list(accs_in) + _flatten_b_tile(b_next)

            accs_final = list(results[:n_accs])
            bt_final = _unflatten_b_tile(list(results[n_accs:]))
            _last_fp4_sc = (
                load_fp4_scales(
                    fx.Index((_k_loop - tile_k) // tile_k * _fp4_scale_k_stride)
                )
                if is_fp4
                else None
            )
            final_accs, scales = compute_tile(
                accs_final,
                bt_final,
                lds_a_pong,
                is_last_tile=not is_fp4,
                fp4_scales=_last_fp4_sc,
            )
            store_output(final_accs, scales)

    # ── Host launcher ──────────────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_lut: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        from flydsl._mlir import ir

        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        kernel_gemm._func.__name__ = KERNEL_NAME
        launcher = kernel_gemm(
            arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias, arg_lut,
            i32_m, i32_n,
        )
        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(
                        hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func"
                    ):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            fx.Int32.ir_type, _wpe
                        )
        launcher.launch(
            grid=(gx, gy, split_k),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm


def compile_preshuffle_gemm_w4(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp4",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    lds_stage: int = 2,
    use_cshuffle_epilog: bool = False,
    waves_per_eu: int = None,
    use_async_copy: bool = False,
    dsrd_preload: int = 2,
    dvmem_preload: int = 2,
    xcd_swizzle: int = 0,
):
    """MXFP4 preshuffle GEMM — delegates to compile_preshuffle_gemm_a8 with fp4 config."""
    if a_dtype == "fp8":
        raise NotImplementedError(
            "fp8-A not yet supported with MXFP4 kernel (op_sel_a overflow)"
        )
    if str(get_hip_arch()) != "gfx950":
        raise RuntimeError(f"FP4 GEMM requires gfx950, got {get_hip_arch()}")
    inner = compile_preshuffle_gemm_a8(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype="fp4",
        lds_stage=lds_stage,
        out_dtype=out_dtype,
        use_cshuffle_epilog=use_cshuffle_epilog,
        waves_per_eu=waves_per_eu,
        use_async_copy=use_async_copy,
        dsrd_preload=dsrd_preload,
        dvmem_preload=dvmem_preload,
        xcd_swizzle=xcd_swizzle,
    )
    return inner


__all__ = ["compile_preshuffle_gemm_a8", "compile_preshuffle_gemm_w4"]
