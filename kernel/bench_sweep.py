#!/usr/bin/env python3
"""Multi-config throughput sweep for the FlyDSL preshuffle GEMM (psg copy).

Benchmarks a list of (label, in_dtype, tile) configs in a single process.
Values are irrelevant for throughput; only shapes + tile config matter.
"""
import argparse
import sys
import time

sys.path.insert(0, "/root")
sys.path.insert(0, "/work")

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import from_c_void_p
from psg.preshuffle_gemm import compile_preshuffle_gemm_a8

_I8 = getattr(fx, "Int8", None) or fx.Int32
_I32 = fx.Int32
_F32 = fx.Float32
_BF16 = getattr(fx, "BFloat16", None) or (fx.Int16 if hasattr(fx, "Int16") else fx.Float32)


def _ptr(t, et):
    return from_c_void_p(et, t.data_ptr())


def bench_one(M, N, K, tm, tn, tk, in_dtype, warmup, iters,
              wpe=None, dsrd=-1, dvmem=-1,
              lds_stage=2, async_copy=False, xcd=0, cshuffle=False, hw_transcode=None,
              reps=1, split_k=1, split_reduce="planes"):
    dev = "cuda"
    atomic_split = split_reduce == "atomic"
    is_fp4 = in_dtype == "fp4"
    is_fp6 = in_dtype == "fp6"
    is_w4a6 = in_dtype == "w4a6"
    is_w4a4_tc = in_dtype == "w4a4_tc"
    is_w6a6_tc = in_dtype == "w6a6_tc"
    # w6a6 is 6-bit on BOTH sides like fp6, and block-scaled like the 4-bit family. Reading
    # only the first half of that put it in the wrong branch of every test below: it was given
    # fp6's per-row f32 scales -- M floats and N floats -- while the kernel indexed one E8M0
    # per 32 elements along K, i.e. (M*K)/32 bytes. That over-read faults outright at
    # split_k=1, and at split_k>1 it lands inside the larger C allocation and returns a
    # number, which is worse. Nine of the 21 shapes died and the other twelve recorded
    # timings taken against garbage scales. Every w6a6 row from this harness before this fix
    # is void; the end-to-end numbers are not affected, because kernel/qlayout.py sizes the
    # operands from the arm's spec rather than from a hand-maintained list.
    is_w6a6 = in_dtype == "w6a6"
    # The one predicate that decides scale layout, spelled once.
    block_scaled = is_fp4 or is_w4a6 or is_w6a6 or is_w6a6_tc or is_w4a4_tc
    # A/B storage bytes per row.
    # W4A4_TC stores A as NVFP4 too (2 codes/byte) and transcodes it to MXFP6 in the
    # global->LDS staging copy, so its A footprint is fp4's, not fp6's.
    a_pack = 2 if (is_fp4 or is_w4a4_tc) else 1   # fp6/fp8 A: 1 byte/elem (32B per K32)
    # W4A6 and W6A6_TC store B as fp4-dense (16 B/lane); W6A6_TC transcodes to fp6.
    b_pack = 2 if (is_fp4 or is_w4a6 or is_w6a6_tc or is_w4a4_tc) else 1
    a = torch.randint(0, 255, (M, K // a_pack), dtype=torch.uint8, device=dev)
    b = torch.randint(0, 255, (N, K // b_pack), dtype=torch.uint8, device=dev)
    # split_k workgroups along K each write a PARTIAL plane of C, so the buffer holds split_k of
    # them and the caller sums. The timed loop below includes that sum: whoever calls this kernel
    # cannot use the partials, so a dispatch number that excluded the reduction would be fiction.
    # Under "atomic" every workgroup accumulates into ONE fp32 plane, so there are no partials
    # to hold and no second pass to time -- but C must be re-zeroed before each launch, and that
    # zeroing is part of the cost the caller pays, so it goes inside the timed loop exactly as
    # the reduction does.
    if atomic_split:
        c = torch.zeros((M, N), dtype=torch.float32, device=dev)
    else:
        c = torch.zeros((split_k, M, N), dtype=torch.bfloat16, device=dev)
    # NVFP4->MXFP6 weight-transcode LUT (12-bit): 8*256 i32. Values are irrelevant
    # for throughput; a real LUT is only needed for numeric verification.
    lut = torch.zeros(8 * 256, dtype=torch.int32, device=dev)
    if block_scaled:
        # per-32 E8M0 block scales (1 byte each) for the fp4/mixed MFMA path
        sa_bytes = (M * K) // 32 * 2 + 4096
        sb_bytes = (N * K) // 32 * 2 + 4096
        scale_a = torch.zeros(sa_bytes, dtype=torch.uint8, device=dev)
        scale_b = torch.zeros(sb_bytes, dtype=torch.uint8, device=dev)
    else:
        # fp6/fp8: per-token / per-channel f32 scales (applied in epilogue)
        scale_a = torch.ones((M,), dtype=torch.float32, device=dev)
        scale_b = torch.ones((N,), dtype=torch.float32, device=dev)
    bias = torch.zeros((N,), dtype=torch.bfloat16, device=dev)

    launch = compile_preshuffle_gemm_a8(
        M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
        in_dtype=in_dtype, out_dtype="bf16",
        waves_per_eu=wpe, dsrd_preload=dsrd, dvmem_preload=dvmem,
        lds_stage=lds_stage, use_async_copy=async_copy,
        xcd_swizzle=xcd, use_cshuffle_epilog=cshuffle,
        use_hw_transcode=hw_transcode, split_k=split_k, split_reduce=split_reduce,
    )
    stream = torch.cuda.current_stream()
    kargs = (_ptr(c, _F32 if atomic_split else _BF16), _ptr(a, _I8), _ptr(b, _I8),
             _ptr(scale_a, _F32), _ptr(scale_b, _F32), _ptr(bias, _BF16),
             _ptr(lut, _I32), M, N, stream)
    cf = flyc.compile(launch, *kargs)
    torch.cuda.synchronize()
    # s17: did the kernel write ANYTHING? tile_n=64 on the fp4-family B path (fp4/w4a6/w6a6_tc)
    # leaves C entirely untouched, and timing a kernel that produces no output made it the
    # fastest candidate in every decode row -- worth a fabricated +20% to +224%, and it won those
    # rows for FlyDSL. Guard the whole class of no-op configurations, not just that tile.
    #
    # The check needs non-degenerate operands, which this harness does NOT otherwise use: the
    # fp4-family scale bytes are filled with 0 (E8M0 0 = 2^-127, so every product underflows to
    # zero) and the transcode LUT with 0 (so every B element decodes to zero). Under those inputs
    # a perfectly healthy kernel writes an all-zero C, which is exactly why nothing here ever
    # noticed. Borrow the tensors for one launch, then restore them, so the timed loop below sees
    # byte-identical inputs to every measurement this project has already recorded.
    _degenerate = block_scaled
    if _degenerate:
        scale_a.fill_(127)  # E8M0 2^0
        scale_b.fill_(127)
        lut.fill_(0x25252525)  # any nonzero code; the check needs non-degeneracy, not accuracy
    c.zero_()
    cf(*kargs)
    torch.cuda.synchronize()
    wrote = bool(c.any())
    if _degenerate:
        scale_a.zero_()
        scale_b.zero_()
        lut.zero_()
    if not wrote:
        raise RuntimeError(
            f"kernel wrote NO output at tile=({tm},{tn},{tk}) dtype={in_dtype}: C is entirely "
            "zero after a launch with unit scales and a nonzero LUT, so any timing here is "
            "meaningless")
    red_out = None
    if split_k > 1 and not atomic_split:
        red_out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)

    def one_call():
        if atomic_split:
            # An accumulating epilogue is only correct into a zeroed buffer, so the zeroing is
            # this path's equivalent of the other path's reduction and is timed the same way.
            # It touches M*N fp32 rather than split_k*M*N bf16, which is the whole point.
            c.zero_()
        cf(*kargs)
        if red_out is not None:
            torch.sum(c, 0, out=red_out)

    for _ in range(warmup):
        one_call()
    torch.cuda.synchronize()
    # s15: the node is shared, so a single timed loop reports whatever contention it happened
    # to meet. The HIP harness has always taken the max over repeats (best_of), and measuring
    # only one side that way biased the whole dispatch table towards HIP -- an audit run during
    # heavy external load read FlyDSL 5.8x low while HIP reproduced its record. Repeat here,
    # inside the process, so the extra samples cost no startup or JIT recompile.
    dt = None
    for _ in range(max(1, reps)):
        t0 = time.time()
        for _ in range(iters):
            one_call()
        torch.cuda.synchronize()
        t = (time.time() - t0) / iters
        if dt is None or t < dt:
            dt = t
    return 2.0 * M * N * K / dt / 1e12, dt * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--in_dtype", type=str, default="fp8")
    ap.add_argument("--tiles", type=str, default="128,256,128;256,256,128;128,128,256;256,128,256;128,256,256",
                    help="semicolon-separated tm,tn,tk triples")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--wpe", type=int, default=None, help="waves_per_eu override")
    ap.add_argument("--dsrd", type=int, default=-1, help="dsrd_preload override")
    ap.add_argument("--dvmem", type=int, default=-1, help="dvmem_preload override")
    ap.add_argument("--lds_stage", type=int, default=2, help="LDS ring depth (STAGES)")
    ap.add_argument("--async_copy", action="store_true", help="async global->LDS DMA")
    ap.add_argument("--xcd", type=int, default=0, help="xcd_swizzle rasterization")
    ap.add_argument("--cshuffle", action="store_true", help="cshuffle epilogue")
    ap.add_argument("--reps", type=int, default=1,
                    help="repeat each timed loop N times and keep the best, matching the HIP "
                         "harness's best_of. Contention-robust on a shared node.")
    ap.add_argument("--splits", type=str, default="1",
                    help="comma-separated split_k values to sweep against every tile. K/split_k "
                         "must be a multiple of tile_k, of 256 for the 4-bit family, and at least "
                         "4 k-tiles; illegal combinations are reported and skipped. The timed loop "
                         "includes the reduction over partials.")
    ap.add_argument("--split_reduce", choices=["planes", "atomic"], default="planes",
                    help="how split_k partials are combined. 'planes' writes split_k private C "
                         "planes and the caller sums them (an extra pass over split_k*M*N, "
                         "measured at 35%% of the win on DOWN at split_k=19). 'atomic' has every "
                         "workgroup accumulate into one fp32 plane with buffer_atomic_fadd, so "
                         "there is no second pass; the buffer zeroing it needs instead is timed.")
    ap.add_argument("--hw_transcode", choices=["on", "off"], default=None,
                    help="w6a6_tc transcode path: hardware scale-converts (on) or the "
                         "12-bit LDS LUT (off). Default: follow the TC_HWCVT env var.")
    args = ap.parse_args()
    torch.manual_seed(0)
    M, N, K = args.m, args.n, args.k
    print(f"=== in_dtype={args.in_dtype}  MNK=({M},{N},{K})  "
          f"lds_stage={args.lds_stage} async={args.async_copy} "
          f"xcd={args.xcd} cshuffle={args.cshuffle} ===")
    best = (0.0, None, 1)
    # tile and split_k are swept jointly, not one after the other: the best tile already proved to
    # be path-dependent on SQUARE (LUT wants 128x256, HWCVT wants 256x128), and split_k changes how
    # much K work each workgroup has, which is exactly what tile_m trades against occupancy.
    splits = [int(s) for s in str(args.splits).split(",") if s.strip()]
    for tri in args.tiles.split(";"):
        tri = tri.strip()
        if not tri:
            continue
        tm, tn, tk = (int(x) for x in tri.split(","))
        if N % tn != 0 or K % 64 != 0:
            print(f"  tile=({tm},{tn},{tk}) SKIP (divisibility)")
            continue
        for sk in splits:
            tag = f"  tile=({tm},{tn},{tk}) split_k={sk}"
            try:
                tflops, ms = bench_one(M, N, K, tm, tn, tk, args.in_dtype, args.warmup, args.iters,
                                        wpe=args.wpe, dsrd=args.dsrd, dvmem=args.dvmem,
                                        lds_stage=args.lds_stage, async_copy=args.async_copy,
                                        hw_transcode=(None if args.hw_transcode is None
                                                      else args.hw_transcode == "on"),
                                        xcd=args.xcd, cshuffle=args.cshuffle, reps=args.reps,
                                        split_k=sk, split_reduce=args.split_reduce)
                print(f"{tag}  {ms:.4f} ms  {tflops:.1f} TFLOPS")
                if tflops > best[0]:
                    best = (tflops, (tm, tn, tk), sk)
            except Exception as e:
                msg = str(e).splitlines()[0] if str(e) else repr(e)
                print(f"{tag}  ERROR: {msg}")
    if best[1]:
        print(f"BEST {args.in_dtype}: {best[0]:.1f} TFLOPS @ tile={best[1]} split_k={best[2]}")


if __name__ == "__main__":
    raise SystemExit(main())
