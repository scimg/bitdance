# W6A6_TC transcode optimization log

Goal: NVFP4-weight → MXFP6 on-chip transcode GEMM (`in_dtype="w6a6_tc"` in
`psg/preshuffle_gemm.py`) reaching **≥ W4A6** throughput on MI355.

## Key insight (why the target is reachable)
`mfma_scale_f32_16x16x128_f8f6f4` runs at the SAME rate for fp4/fp6/fp8 operands
(all pack into a `vec8 i32` MFMA operand). W6A6 (1801) loses to W4A6 (2869) at
square only because B's **global load** is 24 B/lane (fp6) vs 16 B/lane (fp4).
`w6a6_tc` loads B as **fp4 (16 B)** and transcodes on-chip → its ceiling is
**W4A6 (2869)**, IF the transcode VALU is hidden behind MFMA.

## Baselines (MI355, best tile, TFLOPS)
| Shape | W4A4 | W4A6 | W6A6 | target=W4A6 |
|---|---:|---:|---:|---:|
| square 4096²×8192 | 3569 | 2869 | 1801 | 2869 |
| gate_up prefill 4096×19456×2560 | — | 2140 | 1591 | 2140 |
| down_proj decode 128×2560×9728 | — | 167 | 149 | 167 |
| gate_up decode 128×19456×2560 | — | 702 | 388 | 702 |

## Attempts
### Phase 0 — naive in-compute-loop transcode  [snapshot: psg_phase0_inloop_transcode_508.py]
- LUT read from GLOBAL (16 loads/block), transcode inside mi loop (redundant m_repeat×),
  then hoisted to once-per-tile (128-VGPR b128 cache).
- square 508 / gate_up 477 / down decode 21 / gate_up decode 97. VERIFIED bit-exact vs W4A6.
- Root cause of slowness: transcode serialized in compute path; global LUT latency;
  register spill from caching transcoded B. NOT hidden behind MFMA.
- Verdict: correctness OK, perf ~0.2× of W4A6. Baseline to beat.

### Diagnostics (2026-07-10, diag_tc.sh)
- w4a6 ref: 128,256,128=2861; 128,128,128=1980; 64,64,128=2259.
- w6a6_tc: 128,256,128=507; 128,128,128=594; 64,128,128=339;
  64,64,128=2250(!); 32,64,128=3752(!).
- **Small tiles (tm<128) are BROKEN** (verify: output all-zero, cross-matches w4a6
  because both wrong). fp4-preshuffle needs tm>=128. Discard small-tile "wins".
- waves_per_eu: wpe=1→509, **wpe=2→633**, wpe=3→239, wpe=4→182. Occupancy-sensitive:
  transcode VGPR pressure caps occupancy; wpe=2 is the sweet spot (~+25%).
- dvmem/dsrd preload knobs: no effect (~500).
- Conclusion: transcode VALU not hidden at usable big tiles (low occupancy). Need to
  either (a) cut transcode VGPR/op cost to raise occupancy, or (b) move transcode into
  the pipelined B-load stage / stage fp6 B in LDS so it overlaps MFMA.
- Current best usable: **633 @ (128,256,128) wpe=2** (0.22x of w4a6 2861).

### Phase 1 — LDS-resident LUT  [snapshot: psg_phase1_ldslut_1619.py]  *** 3-4x WIN ***
- Reserve 8 KB at start of pong LDS arena; cooperative global->LDS copy of the
  2048-entry i32 LUT at kernel start (256 threads x 8); transcode reads LDS (ds_read)
  instead of global buffer_load. gfx950 has 160 KB LDS/CU so 8 KB is free.
- VERIFIED bit-exact vs W4A6 (both b_scale=1.0 and 2.0 paths).
- Results (TFLOPS), w4a6 ref in ():
  | shape | phase0 | phase1 | w4a6 | ratio |
  |---|---:|---:|---:|---:|
  | square 4096^2x8192 | 508 | **1619** | 2861 | 0.57 |
  | down decode 128x2560x9728 | 21 | **65** | 129 | 0.50 |
  | gate_up decode 128x19456x2560 | 97 | **419** | 717 | 0.58 |
- Global LUT loads were stealing the vmem/L2 path (fatal esp. in bandwidth-bound
  decode). Best tile now (128,128,128) for decode, (128,256,128) for square.
- wpe no longer helps (LDS-LUT rebalanced). Next: transcode still not overlapped w/
  MFMA (occupancy-1) + 128-VGPR _tc_b cache spills at tn=256. Levers: (a) shrink cache
  (store 6 i32 not 8), (b) cooperative LDS-stage transcoded B (removes per-wave
  redundancy + overlaps), (c) move transcode into pipelined B-load stage.

### PROFILING (rocprofv3, square, per 31 dispatches) — DECISIVE
| counter | w4a6 | w6a6_tc | ratio |
|---|---:|---:|---:|
| SQ_INSTS_MFMA | 100.7M | 100.7M | 1.00 |
| SQ_INSTS_VMEM | 41.7M | 42.1M | 1.01 |
| SQ_INSTS_VALU | 202M | **988M** | **4.89** |
| SQ_INSTS_LDS | 75.5M | 264M | 3.50 |
| GRBM_GUI_ACTIVE (cyc) | 40.7M | 79.9M | 1.96 |
- MFMA + VMEM IDENTICAL (same compute, same fp4-B bandwidth). Transcode adds
  ~786M VALU + ~190M LDS insts. Wall ~2x.
- Higher occupancy (tile 128x128=1279<1619; wpe 2/3 no help) does NOT help =>
  **VALU pipe is SATURATED**. Overlap/pipelining won't help. MUST reduce transcode
  VALU op count. This is the single lever now.
- Transcode/block ~47 VALU: byte-extract (16x shift+and), 12-bit dense pack into 6
  i32 (16x shift/or + cross-word), LUT index adds. Target: fewer ops (fused
  v_lshl_or, i64 accum, or drop to HIP/asm v_perm_b32 for the 4->6 bit repack).

### Phase 2 — micro-opt (LDS-view row fold, mask elision)  [snapshot: psg_phase2_squeeze_1703.py]
- Folded the mantissa row into a per-block LDS memref view (removed 16 `| base`
  index ORs/block); dropped the top-byte `&0xFF` (>>24 already <256). ~+5%.
- square 1619 -> ~1687. VALU still saturated; FlyDSL micro-opt is exhausted here.

### DEFINITIVE CROSSOVER (LDS-LUT + micro-opt, best tile) — *** CONCLUSION OVERTURNED ***
High-iter (warmup 25 / iter 200), 2 repeats, stable to <1%. w6a6_tc vs W4A6:
| Shape | w4a6 | fp6/W6A6 | **w6a6_tc** | tc vs w4a6 |
|---|---:|---:|---:|---:|
| SQUARE 4096^2x8192 (compute)   | 2760 | 1796 | 1687 | 0.61x |
| GATE_UP 4096x19456x2560 (pref) | 2127 | 1647 | 1501 | 0.71x |
| DOWN decode 128x2560x9728      | 240  | 116  | **310**  | **+29%** |
| GATE decode 128x19456x2560     | 1030 | 678  | **1548** | **+50%** |
| QKV  decode 128x4096x4096      | 347  | 200  | **433**  | **+24%** |

- **w6a6_tc BEATS W4A6 on EVERY decode (memory-bound) shape** — the deployment-
  relevant LLM-serving regime. Confirmed **bit-exact == w4a6** (same A) at the exact
  winning decode tiles (verify_decode.sh: cross-check True at 128x64 tiles), so the
  win is real math, not silent under-compute.
- Why tc>w4a6 at decode (both load fp4-B 16B, same fp6-A): the transcode VALU is
  fully hidden when compute is idle (memory-bound), and tc's symmetric fp6xfp6 inner
  loop (blgp=2) out-pipelines w4a6's mixed fp6-A x fp4-B (cbsz=2/blgp=4) path at
  small tn. tc also ~2x fp6/W6A6 (halved weight DRAM bytes: fp4 16B vs fp6 32B).
- Compute-bound (square/prefill) stays 0.61-0.71x: VALU-saturated (profiled 4.89x
  VALU insts). A zero-transcode baseline (w4a6) is un-beatable there by ANY transcode
  kernel — this is a hard floor, not a tuning gap. Not worth an asm rewrite.
- **VERDICT (supersedes the earlier "never transcode at runtime"):** runtime
  NVFP4-weight -> MXFP6 transcode is the FASTEST low-bit option in the decode regime
  (beats W4A6 24-50%) at NVFP4-faithful accuracy + fp4 (16B) weight footprint. Use it
  for decode/memory-bound serving; fall back to W4A6 for compute-bound prefill.

### Phase 3 — compute-bound (TTFT) tuning + "is LUT best for transcode?"
Re-grounded on the ROCm W4A6 blog. KEY: the blog's "W4A6" keeps weights **native
MXFP4 (no in-GEMM transcode)** and only makes *activations* 6-bit (separate quant
kernel). That == our `w4a6`. Its near-W4A4 TTFT parity comes from per-shape tile
tuning + split-K (split-K helps *small-M decode* only; pure overhead at large M).
So the compute-bound "match W4A4" target is `w4a6`, NOT the weight-transcode path.

Compute-bound square 4096^2x8192 wide tile+scheduler sweep:
| dtype | best TFLOPS | vs fp4 | limiter |
|---|---:|---:|---|
| fp4 (W4A4)   | 3366 | 1.00 | MFMA/peak |
| w4a6         | 2944 (dsrd4/dvmem3) | 0.874 | **LDS-bound (6-bit A)** |
| w6a6_tc      | 1849 | 0.55 | **VALU pack-bound** |
GATE_UP prefill: fp4 2896 / w4a6 2452 (0.85) / w6a6_tc 1622 (0.56).
Scheduler: wpe=3 collapses occupancy (521 TFLOPS); best w4a6 = wpe0/dsrd4/dvmem3.

Compute-bound profile (GEMM-only, per-dispatch counters, MFMA=1.007e8 for all 3):
| insts (÷1e8) | fp4 | w4a6 | w6a6_tc |
|---|---:|---:|---:|
| VALU | 1.56 | 1.65 | **9.54** |
| LDS  | 0.38 | 0.75 | **2.64** |
| busy | 0.91 | 1.32 | 2.69 |
- w4a6 gap = **2x LDS** vs fp4 (32B fp6-A vs 16B fp4), VALU ~equal -> A-LDS-bound,
  inherent to 6-bit A. ~87% is near the hard ceiling; matches blog's 84-96% band.
- w6a6_tc transcode delta = **+7.9e8 VALU (pack+extract) vs +1.9e8 LDS (LUT)** ->
  pack VALU is ~80% of the overhead, LUT lookup only ~20%.

IS LUT BEST? — decisive isolation experiment (TC_PROBE_NOLUT gate in psg): keep the
identical dense packer but replace the 16 LUT ds_reads/block with the raw index.
  square w6a6_tc:  LUT 1837.8 TFLOPS  ->  no-LUT 1828.1 TFLOPS   (~0%, within noise)
=> The LUT lookup is **fully hidden** behind the VALU/MFMA pipe in the compute-bound
regime. It is essentially FREE. Conclusions:
  1. LUT (LDS-resident, fused 4->6b convert + 12-bit pre-pack per ds_read) is the
     RIGHT choice — it offloads value-conversion to the memory system at zero net cost.
  2. A cheaper lookup (v_perm_b32 register-gather, ALU/bit-parallel E2M1->E2M3, or the
     amdgcn.perm CallIntrinsic path) would attack the already-free 20% LDS while ADDING
     VALU to the saturated 80% pack -> strictly WORSE for compute-bound. FlyDSL also has
     no v_perm_b32 wrapper (rocdl exposes only fmed3/ds_bpermute/permlane + raw bitops).
  3. The pack VALU is FORMAT-MANDATED (MFMA needs a dense 6-bit B operand) and already
     near-minimal (12-bit straddle merge; LLVM forms v_lshl_or). It cannot be removed
     while producing fp6. The ONLY way to remove it is to not transcode.
- **FINAL:** compute-bound (TTFT) => use native-weight `w4a6` (~87% of W4A4, LDS-bound
  on 6-bit A; blog's actual path). Weight transcode `w6a6_tc` is a *decode/TPOT* win
  only (VALU free when memory-bound). No transcode-lookup rewrite can change this.

### Phase 3b — what is the transcode VALU actually doing, and "separate transcode"?
The LUT is NOT the cost. Per 32-code block the transcode is ~58 VALU of bit-plumbing
around 16 (free) ds_reads:
  - UNPACK ~24 VALU: fp4 is 2 codes/byte, dense. Shift+mask 16 byte-fields out of the
    4 packed i32 words to form the LUT indices (`w&0xFF,(w>>8)&0xFF,(w>>16)&0xFF,w>>24`).
  - REPACK ~32 VALU: the 16 LUT outputs (12b = 2 fp6 codes) must be packed into the
    DENSE 6-bit MFMA B-operand (6 i32). 12 doesn't divide 32 -> ~5/16 values straddle a
    word and cost an extra shift+or. Repack is MFMA-mandated; already near-minimal.
  So "transcode-bound" = bound by 4b-unpack + 6b-repack, not the table lookup (hence a
  v_perm/ALU lookup can't help — it doesn't touch the plumbing).

SEPARATE transcode == materialize MXFP6 weights in DRAM, then a plain fp6 (W6A6) GEMM
that must read 32B/128 weight from DRAM+LDS. Measured (best tile):
| shape | w4a6 native | fp6 (separated GEMM) | w6a6_tc fused |
|---|---:|---:|---:|
| SQUARE compute   | 2886 | 1929 | 1806 |
| GATE_UP prefill  | 2195 | 1739 | 1608 |
| GATE decode      | 1011 |  673 | 1512 |
| QKV decode       |  343 |  202 |  426 |
- Compute-bound: separated fp6 (1929) is ~7% > fused (1806) — no transcode VALU in the
  GEMM — BUT it adds a full transcode kernel pass + **2x weight DRAM footprint**, and is
  still far below native w4a6 (2886).
- Decode: separated fp6 (202/673) is **~2x SLOWER** than fused (426/1512): 6-bit weights
  double the weight-DRAM bytes in the bandwidth-bound regime. The fused in-register
  expand-then-consume is exactly the decode win.
- Weights are static -> could pre-transcode offline (pass ->0), but 2x weight footprint
  is permanent -> then storing fp4 + native w4a6 (compute) / fused w6a6_tc (decode) is
  strictly better. **Separating defeats the fused design's whole point** (keep weight at
  16B fp4 in DRAM; expand to 6-bit only transiently in registers).

### Phase 4 — pure HIP/asm w4a6_tc kernel (from scratch, gfx950)
Goal (user): write a pure HIP C++/asm w4a6_tc GEMM, benchmark both regimes, beat w4a6
where possible (decode), document the compute-bound floor. Refs: CDNA4 ISA + ROCm blogs
(matrix-cores-cdna, cdna4-gemm-kernels).

Toolchain (remote container `nvfp4_mxfp6_emu`, card 5): hipcc 7.2.26015 / clang 22,
gfx950, ROCm 7.2.0. `hip_ext_ocp.h` present. Prior-phase HIP files in /work/src
(fused_nvfp4_mxfp6_gemm.hip = correct but naive 1-wave/tile 32x32x64 reference).

Key intrinsic facts (verified against CK `ck/utility/amd_xdlops.hpp` + `ck_tile ...
warp_gemm_attribute_mfma_impl.hpp`, authoritative):
  d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a /*i32x8: fp6 dense=192b in w[0..5], w[6..7]=0*/,
        b /*i32x8 same*/, c /*f32x4*/,
        cbsz /*Atype: 0 E4M3,1 E5M2,2 E2M3(fp6),3 E3M2,4 E2M1(fp4)*/,
        blgp /*Btype, same enum*/,
        opselA, scale_a /*int32: 4 packed E8M0 bytes / K=128's 4 blocks*/,
        opselB, scale_b);
  Layout (16x16x128, per FP8 blog + CK constants kAMLane16/kABKLane4/kABKPerLane32):
    lane in [0,63]; row=lane&15; g=lane>>4 (0..3). Per lane 32 elems = TWO 16-chunks
    64 apart: e in[0,15]->k=k0+g*16+e ; e in[16,31]->k=k0+g*16+64+(e-16).
    C: 4 f32/lane; out_col=tile_n+row; out_row=tile_m+g*4+t (t=0..3) -> c_reg[t].
  Scale=127 => 2^0=1 (biased E8M0, identity). Reference & blog agree; CK's "0" is a
  different/edge path -> use 127 for identity.

FOUNDATION kernel `kernel/hip/w4a6_tc_gemm.hip` (1 wave/16x16 tile, K-loop /128,
uniform scale): A=dense E2M3 fp6, B=NVFP4 nibbles transcoded in-register via the
verified 64-entry LUT, packed dense, one scaled MFMA/128. Verified BIT-EXACT
(maxabs=0, nbad=0) vs a layout-independent FP32 golden at 64x64x256 => the split-chunk
layout + transcode + dense pack + scaled MFMA path is CORRECT. This is the base to
optimize (LDS tiling, direct global->LDS, double-buffer, ping-pong per the GEMM blog).

Optimization ladder (measured on card5, TFLOP/s = 2MNK/t; verify = maxabs 0 bit-exact):
| stage | file | 512^3 | 4096^3 | 4096x4096x8192 | M64 N4096 K4096 (decode) |
|---|---|---:|---:|---:|---:|
| v1 naive 1-wave/16x16 tile   | w4a6_tc_gemm.hip    | 3.3 | 13  | 13  | 7.7 |
| v2 LDS-tile 64x64 4-wave 4mma| w4a6_tc_gemm_v2.hip | 13  | 121 | 121 | 14.5 |
Targets: FlyDSL w4a6 2886 (compute) / 1011 (decode); fp4 ceiling ~3300; blog FP8 ladder
matrix-core 30 -> vec-load 337 -> direct g->LDS 507 -> dbl-buf 1166 -> 256x256 multiwave
2288 -> 8-wave ping-pong 2680. v2 (121) is between "matrix-core" and "vec-load": bound by
scalar byte g->LDS loads + per-MFMA byte LDS reads&repack + single-buffer + tiny tile.
Next levers, in order: (1) vectorized 16B loads g->LDS; (2) store A/B in LDS pre-packed
dense-fp6 so the inner loop is a single ds_read_b128 (kill per-MFMA repack); (3) double
buffer; (4) 128x128/256x256 tile + 8 waves; (5) ping-pong sched (s_barrier/setprio).

Ladder continued (4096^3 compute-bound, all bit-exact maxabs=0):
| v3 | pre-pack dense-fp6 LDS frags + 16B vec gather, 128x128, 8 waves | 311 |
| v4 | v3 + register-prefetch double-buffer | 223 (REGRESSION -> rolled back) |
v4 lesson: holding prefetch regs (av0/av1/bp0/bp1) + cached areg[4]/breg[2] + 8 f32x4
accumulators at 512 thr/CTA spills VGPRs; the naive double-buffer HURTS here. Roll back
to v3. Real double-buffer needs lower reg pressure (don't cache frags; read LDS in mma
loop) and/or higher occupancy. Next: sweep tile size (bigger tile = more MFMA per pack
burst) + waves on the v3 design before reattempting overlap.

v5 = parameterized v3 (-DBM/BN/WM/WN). Tile sweep @4096^3 (bit-exact):
  128x128 w2x4=257  128x256 w2x4=213  256x128 w4x2=291
  256x256 w4x4=407  256x256 w2x4=414(BEST)  128x128 w2x2=262
=> bigger tile (256x256) amortizes the serial pack burst best: 414 TFLOP/s single-buffer.
Still ~7x below w4a6 (2886). Gap = (a) serial transcode+pack VALU not overlapped w/ MFMA
(needs 8-wave ping-pong), (b) my kernel needlessly REPACKS A (A is already fp6 -> should
load fragment-layout directly like the blog's fp8, only B/weight needs transcode+pack),
(c) occupancy 1 CTA/CU at 256x256. Remaining plan: (1) DECODE kernel + head-to-head vs
FlyDSL (the winnable "beat w4a6"); (2) pre-packed-A (halve pack VALU); (3) ping-pong.

v6 = A PRE-PACKED to fragment layout in DRAM, loaded DIRECT-to-register (no A repack, no
A LDS -> only B/weight transcoded, matches real w4a6_tc semantics). Bit-exact incl.
DISTINCT per-block E8M0 sub-scales (scale baked into transcode LUT, MFMA scale=1.0).
  SQUARE 4096^3: 611 (uncontended) / ~490 (contended). +26% over v5, best design.
v7 = v6 + double-buffered B (prefetch raw nibbles -> regs, pack after compute): REGRESSED
  611->152 at w4x4. Same failure mode as v4: 2x LDS halves occupancy + prefetch regs
  spill + pack VALU stays serial (runs between compute and sync, overlaps nothing).
  ROLLED BACK. Explicit double-buffer is a WALL on this arch at these tiles; real overlap
  needs occupancy (hw interleaves CTAs) or true wave-ping-pong, not sw double-buffer.

FAIR same-session head-to-head (all 8 GPUs at 100% contention, best-of-3):
| shape                    | w4a6 | w6a6_tc(fused) | mine v6 |
| SQUARE 4096x4096x8192    | 2547 | 1450           | 489     |
| GATE_UP 4096x19456x2560  | 2155 | 1530           | 513     |
| GATEdec 128x19456x2560   |  701 |  855           | 188     |
| QKVdec  128x4096x4096    |  224 |  255           |  47     |
=> from-scratch HIP reaches ~34% of the fair fused baseline (w6a6_tc) on compute, ~22-27%
   on decode. Did NOT beat FlyDSL. Gap is production pipelining: 8-wave ping-pong,
   sched_barrier/setprio, LDS swizzle (bank-conflict-free), and for decode split-K
   (QKV starves CUs: N=4096/BN -> <=128 CTAs for ~256 CUs). Ladder 13->611 TFLOP/s (47x)
   with every step bit-exact. Measurement now unreliable due to full-node contention.

=== RESUMED (multi-path, don't-abandon-early) ===
v8 = v6 + coalesced B ingress via LDS staging: NEUTRAL (SQUARE 459 vs 489; L2 already
  absorbs the strided B reads). Parked.
v6d = v6 + SPLIT-K (gridDim.z, atomicAdd). Bit-exact. DECODE occupancy fix:
  QKV 128x4096x4096: S1=46 -> S4=125 (128x64) = 2.7x (was CU-starved: 32 CTAs).
  GATE 128x19456x2560: split-K HURTS (S1=186 best; already 304 CTAs). => split-K only
  when CTA_count < CU_count.
CEILING PROBE (-DNOPACK, skip transcode+pack): 256x256w4x4=938, 256x128w4x2=1200 TFLOP/s
  => transcode+PACK was ~50-62% of runtime. Highest-value lever.
v9 = v6 + CHEAPER PACK: (a) branchless 64-bit-straddle pack, (b) LUT __constant__->LDS
  (non-uniform __constant__ reads SERIALIZE per-wave!). BIT-EXACT. HUGE:
  SQUARE 256x256w4x4: 489 -> 827 (+69%). GATE_UP 513 -> 823. Now near NOPACK ceiling.
  Best compute config so far: 256x256 w4x4 = 827 TFLOP/s (57% of w6a6_tc 1450).
Remaining: base-GEMM ceiling itself is 938-1200 vs pure-fp6 1886 => occupancy/MFMA-sched/
  LDS-bank gap. Next: LDS swizzle + occupancy, then port cheap-pack into split-K decode.

v9 unified (cheap pack + LDS LUT + split-K in one kernel). Decode peaks (best-of-N):
  GATE 128x19456x2560: 128x64 w4x2 S1 = 321 (was 186). 128x32w4x1 S1=276.
  QKV  128x4096x4096 : 128x32 w4x1 S4 = 210 (was 47!) = 94% of w4a6 (224).
v9db = v9 + double-buffered B for decode: REGRESSED (GATE 321->276, QKV 210->161).
  => DOUBLE-BUFFER IS A CONSISTENT WALL (compute v4/v7 AND decode v9db all regress):
     occupancy is the latency-hiding mechanism here; 2x LDS halves it. Rolled back.
BEST KERNEL = v9 single-buffer everywhere.

FINAL fair same-session head-to-head (node less contended; FlyDSL numbers rose too):
| shape                    | w4a6 | w6a6_tc(fair) | mine v9 | % of fused |
| SQUARE 4096x4096x8192    | 2872 | 1825          | 934     | 51%        |
| GATE_UP 4096x19456x2560  | 2223 | 1621          | 814     | 50%        |
| GATEdec 128x19456x2560   | 1017 | 1526          | 321     | 21%        |
| QKVdec  128x4096x4096    |  344 |  429          | 210     | 49%        |
Session arc: compute 121 -> 934 TFLOP/s (7.7x this session, ~70x over naive), decode QKV
47 -> 210 (4.5x), all BIT-EXACT incl distinct per-block scales. Now ~2x off the fair fused
baseline (was ~3-5x). v9 (256x256 w4x4) reaches its NOPACK ceiling (~938) => at this
config the transcode is fully hidden; further gains need a higher base-GEMM ceiling
(pure-fp6 is 1886): LDS-swizzle for bank-conflict-free sB reads, MFMA scheduling, and a
config with higher NOPACK ceiling (256x128 NOPACK=1200). GATEdec gap is memory-latency
(needs deeper prefetch that doesn't cost occupancy). Double-buffer conclusively a wall.

Per-config v9 vs NOPACK ceiling (best-of-4): 256x256w4x4 pack=876/ceil=1138(77%);
512x128w8x2 814/1234(66%, highest ceiling); 256x256w2x4 724/771(94%, pack fully hidden).
=> Two isolated levers with measured headroom:
  (1) pack-hiding on w4x4: 876->1138 (~23%). Single-buffer CANNOT overlap pack vs MFMA
      (separated by __syncthreads); naive double-buffer regresses (occupancy). Requires
      TRUE wave-ping-pong (specialize half the waves to pack buf B while other half MFMA
      buf A, swap via named s_barrier) - the only structure that overlaps without the
      occupancy hit. High effort/risk; not attempted under shared-node noise.
  (2) base-GEMM ceiling 1234 vs pure-fp6 1886 (~34%): LDS bank-conflict-free swizzle on
      sB fragment reads + MFMA instruction scheduling (setprio/sched_barrier).
Both are the same techniques FlyDSL/hipBLASLt use to reach their numbers.

=== SESSION 2 (register-discipline + coalesced ingress + pack micro-arch) ===
Solo (uncontended, GPU0 best-of-N) is the reliable metric; parallel best-of-N on 7
GPUs compresses all variants toward a memory/power-contention cap (~1515) and hides
the true decomposition. All numbers below are SOLO unless noted. All BIT-EXACT.

v11 = register-disciplined double-buffer: cache only the A fragment (areg), read B
  inline from LDS. Overlap without the occupancy hit of caching both. SQUARE ~985-1067
  @256x64 w4x1. First double-buffer that did NOT regress.
Tile growth: 512x64 w8x1 (grow BM, hold MPW=4) amortizes pack+load without spill.
v15 = COALESCED PRE-PACKED B INGRESS *** breakthrough ***. Reorder NVFP4 B nibbles in
  DRAM into fragment order offline: [gblk][kc][lane][16B] so each lane does ONE
  coalesced 16B (uint4) global load. SQUARE 934->1391 (+48%). Bit-exact + split-K.
  Ruled out (all register/occupancy walls, rolled back): A-dbuf (v13), 3-stage (v16),
  half-fragment pack (v17).
v15 solo decomposition (SQUARE 4096^2x8192): FULL 1377 / DUMMY 1776 (load+ds_write) /
  NOPACK 2318 (base MFMA). => load gap 542, pack gap 378. Base ceiling 2318 > FlyDSL
  native 1886, so the base GEMM is not the problem; transcode overhead is.

--- Research (ROCm CDNA4 blogs + Gluon gfx950 tutorial + FP8 GEMM blog) ---
- global_load_lds (gfx950: 128b/lane) = direct HBM->LDS, no VGPR, no ds_write, ASYNC
  (vmcnt-tracked). Wins big for PURE-copy GEMM (frees 100+ VGPR).
- 8-wave ping-pong (HipKittens): s_barrier + s_setprio + sched_barrier wave
  specialization; scale pipeline round-trips scales through LDS (ds_read_tr).

v18 = async global_load_lds for raw B (HBM->rawLDS->reg transcode->mxfpLDS). Correct
  (all PASS) but REGRESSED: SQUARE 1132 vs 1391. LESSON: a TRANSCODE kernel must stage
  in registers to transcode, so direct-to-LDS only ADDS an LDS round-trip (B through
  LDS twice) and the coarse vmcnt(0) drains A too. Async-to-LDS helps pure-copy GEMMs,
  NOT transcode. Rolled back. (dead-end kernel deleted; traced here.)

--- Pack micro-architecture (the winning session-2 lever) ---
v20 = fold the NVFP4 sign bit into a 128-entry LUT [(scale<<4)|nibble -> 6-bit code].
  Removes ~3 VALU ops/element. SQUARE 1391->1504 (+8%); GATE 1155->1260; BIG 1259->1355;
  QKV ~1300->1469. pack gap 378->272. Bit-exact all shapes/splits.
v21 = 12-bit BYTE-INDEXED LUT [(scale<<8)|byte -> TWO pre-packed 6-bit codes (12b)].
  One LDS read per input byte => removes nibble extraction, HALVES LUT reads (16 vs 32)
  and HALVES bitpack iterations (16 vs 32). *** current production ***
  SQUARE 1504->1623; pack gap 272->146 (pack now largely hidden). All BIT-EXACT.
v22 = issue B loads 2 tiles ahead (double-buffered raw regs) to attack the 542 load
  gap. REGRESSED (1623->1469): hit the same register wall as v13/v16; and the load is
  NOT bandwidth-bound (~99 GB/s of B vs ~8 TB/s HBM) so deeper prefetch can't help at
  this occupancy. Rolled back (kernel deleted; traced here).

FINAL v21 SOLO (best-of-20/25), all BIT-EXACT incl split-K 2/4/8 + odd shapes + decode:
| shape                          | v9  | v15  | v21  | vs v15 |
| SQUARE 4096^2x8192  (compute)  | 934 | 1377 | 1621 | +18%   |
| GATE_UP 4096x19456x2560 (pref) | 814 | 1155 | 1333 | +15%   |
| BIG 8192^3          (compute)  |  -  | 1259 | 1405 | +12%   |
| QKV_prefill 8192x6144x4096     |  -  |~1300 | 1555 | +20%   |
| TALL 16384x4096x4096           |  -  |   -  | 1477 |   -    |
| GATE decode 128x19456x2560 S2  | 321 |  363 |  402 | +11%   |
| QKV  decode 128x6144x4096  S4  | 210 |  230 |  318 | +38%   |
Session-2 arc: SQUARE 934->1623 (+74%), decode QKV 210->318 (+51%). v21 pack gap = 146
(down from 378): the transcode pack is now mostly hidden behind MFMA. Remaining gap is
the 542 load+ds_write (latency/LDS-write, NOT bandwidth), which resists deeper sw
prefetch at this occupancy (v13/v16/v22 all confirm the register wall). Further gains
would require true 8-wave ping-pong (s_barrier/setprio) to overlap ds_write behind
MFMA at fixed occupancy — high effort/risk, deferred. PRODUCTION KERNEL = v21.

=== SESSION 3 (arithmetic-intensity: BM growth) — *** biggest single lever *** ===
Research: HipKittens (arXiv 2511.08083) proves classic producer/consumer wave
specialization UNDERPERFORMS on CDNA4 (893 vs 1610 TFLOPS) due to static register
partitioning; the winning patterns are 8-wave ping-pong (2 waves/SIMD ALTERNATING
compute<->memory, not specialized) and manual sched hints (sched_group_barrier /
s_setprio / sched_barrier(0)) because the backend clusters MFMAs. NVIDIA Blackwell does
block-scaled fp4/fp6 dequant in HARDWARE (tcgen05.mma.kind::mxf8f6f4) — not portable.

KEY REALIZATION: the f32x4 MFMA accumulators live in the SEPARATE 256-entry AGPR bank
(acc = MPW*NPW*4 AGPR), NOT the 256 VGPRs. So growing BM (=> larger MPW) adds MFMA work
per CTA without VGPR spill, and sB depends only on BN (not BM) so LDS/occupancy is
UNCHANGED. Never swept BM>512 before. This is the highest-value lever found.

BM sweep on v21 (BN=64 w8x1, SOLO best-of, all BIT-EXACT):
| shape                    | BM=512 | BM=1024 | BM=2048 |
| SQUARE 4096^2x8192        | 1623   | **1926**| 150 (spill+<CU CTAs) |
| BIG 8192^3                | 1405   | **1800**|  -    |
| GATE_UP 4096x19456x2560   | 1333   | **1519**|  -    |
| QKV 8192x6144x4096        | 1555   | **1668**|  -    |
| TALL 16384x4096x4096      | 1477   | **1572**|  -    |
BM=1024 (MPW=8, 128 AGPR, sB=16KB => 2 CTAs/CU kept) is the sweet spot. BM=2048 (MPW=16)
spills + drops below CU count (128 CTAs) => collapses. Decode (M=128) keeps BM=128.
BM=1024 decomposition (SQUARE): FULL 1936 / DUMMY 2053 / NOPACK 2397. Bigger BM RAISED
the base ceiling (2318->2397) AND shrank the load gap 542->344 (more MFMA hides load);
pack gap 146->117. Now at 81% of ceiling (was 70%).

v23 = pack-before-mma reorder + sched_group_barrier/s_setprio interleave hints. REGRESSED
(SQUARE 1936->1811 reorder / 1827 hints; BIG 1800->1497/1440). At BM=1024 the compiler
already schedules well; manual cluster hints degrade it. Rolled back (deleted; traced).
Ping-pong DEFERRED as low-ROI here: BM=1024 already runs 2 CTAs/CU, so the hardware
interleaves two thread blocks to hide latency — exactly what ping-pong buys in the
1-CTA/CU large-tile regime it targets. Our 344 residual is mostly A-load latency
(A reloaded each k-tile, 8 frags/wave at MPW=8), already ~2/3 hidden by the 2-CTA overlap.

*** PRODUCTION: v21 @ BM=1024 BN=64 w8x1 (compute) / BM=128 BN=64 w4x1 (decode) ***
Certified BIT-EXACT: K=1024/2048/4096, split-K 2/4, N=384 odd shape, decode M128 S2.
Cumulative vs session-1 v9 baseline: SQUARE 934->1936 (+107%), BIG ->1800, GATE 814->1519.


=== SESSION 4 (2026-07-11) "beat-W4A6" DEEP DIVE — reframed the whole gap ===
Scripts: kernel/ceiling_probe.sh (FULL/DUMMY/NOPACK decomposition), kernel/bn_ceiling.sh
(BN sweep + verify), kernel/occ_probe.sh (VGPR/AGPR/LDS -> occupancy). SQUARE 4096^2x8192,
MI355 dev6, SOLO best-of-8. Reference (same session bench_full): W4A6 = 2852, W4A4 = 3178.

TWO ASSUMPTIONS IN SESSION 3 WERE WRONG:
  1. "acc lives in the separate 256-entry AGPR bank." -Rpass=kernel-resource-usage says
     **AGPRs=0** for every config — the MFMA accumulators are in the 256 VGPRs. gfx950 has a
     UNIFIED 512-wide VGPR file; acc in VGPR is why BM1024xBN128 (acc[8][8]=256 f32x4=1024
     VGPR-equiv) SPILLS and collapses to 226 TFLOP/s.
  2. "BN=64 runs 2 CTA/CU so HW interleaving already hides the transcode -> ping-pong is
     low-ROI." Actually every 8-wave config reports **Occupancy 2 waves/SIMD = 1 CTA/CU**
     (256 VGPR/wave x 2 = the 512 file; 4 waves/SIMD would need <=128 VGPR/wave, but acc
     alone is 128). So the kernel is 1 CTA/CU EVERYWHERE. The transcode is hidden at BN=64
     only by the 8 waves' mutual latency-hiding + compiler scheduling, not a 2nd CTA.

DECISIVE DECOMPOSITION (best-of-8, W4A6=2852):
| config              | NOPACK(base) | DUMMY(+ds_write) | FULL(+pack) | ds_write gap | pack gap |
| BM1024xBN64 (prod)  | 2454         | 2086             | 1988        | 368          | 98       |
| BM512xBN128         | **3126**     | 2279             | 1868        | **847**      | **411**  |
BN=128 verified BIT-EXACT (RESULT: PASS at 512x128 / 1024x128 / 512x128w4x2).

THE REFRAME — why we've been stuck at ~0.70x W4A6:
- Session 3 compared the NOPACK ceiling (2318/2397) to *pure-fp6 W6A6 (1886)* and declared
  "the base GEMM is not the problem." But the real target is **W4A6 = 2852**, and at BN=64
  the base ceiling is only **2454 — 14% BELOW W4A6.** So at BN=64 no transcode kernel, however
  perfectly pipelined, can EVER beat W4A6: the base GEMM itself is the wall. This is the
  single most important finding and it was hidden by benchmarking against the wrong baseline.
- The base ceiling is A-load-intensity bound (BN=64 -> narrow N -> A reloaded per FLOP). BN=128
  HALVES A-loads/FLOP and lifts the ceiling to **3126 — +10% OVER W4A6.** So **BN=128 is a
  NECESSARY condition to beat W4A6.** (Confirmed the load residual is A-load, not B.)
- The catch: BN=128 doubles B blocks (NB 4->8) -> 2x transcode ds_write+pack per unit MFMA,
  while occupancy stays 1 CTA/CU. So the transcode that was ~19% hidden at BN=64 becomes ~40%
  EXPOSED at BN=128 (FULL 1868). The ds_write gap (847) dominates; pack VALU (411) is second.

ROADMAP TO BEAT W4A6 (in priority order; all must land on BM512xBN128):
  (1) i32x6 B storage — drop the 2 always-zero MFMA operand words. sB 36KB->27KB, ds_write
      and inner-loop ds_read both -25%. Directly attacks the 847 ds_write gap (est. ~+200).
      MFMA still fed i32x8 by zero-padding [6],[7] in-register after ds_read (2 v_mov, free).
      Bit-exact-preservable. CHEAP — do first.
  (2) TRUE 8-wave ping-pong (s_barrier + s_setprio + sched hints): specialize/alternate waves
      so the B ds_write+pack of tile t+1 overlaps the MFMA of tile t. This is NOW high-ROI
      (was dismissed under the wrong 2-CTA assumption): at 1 CTA/CU with 1258 TFLOP/s of
      EXPOSED transcode over a >W4A6 ceiling, overlapping it is the only structural way to
      recover the bulk of the gap. Target: FULL 1868 -> ~2600-2900 (beat/tie W4A6).
  (3) pack VALU (411 gap): v_perm_b32 / bit-parallel E2M1->E2M3 register gather to shave the
      repack now that it is exposed at BN=128 (was fully hidden at BN=64, so ignored before).
  NOTE: raising occupancy to 2 CTA/CU is INFEASIBLE — needs <=128 VGPR/wave but acc alone is
  128; shrinking the tile kills the arithmetic intensity that lifts the ceiling. Overlap must
  come from ping-pong within the single CTA, not from more CTAs.

CURRENT BEST (unchanged production): BM1024xBN64 GROUP=2 FULL ~1988. The BN=128 path is the
identified route past W4A6 and is the next implementation target.

--- Session-4 MULTI-PATH campaign (all BM512xBN128 w8x1, SOLO best-of-8, W4A6=2852) ---
Baseline v31 @ BN128: NOPACK 3126 / DUMMY 2279 / FULL 1868. Files kept for rollback.

PATH A [w4a6_tc_gemm_v32.hip] — i32x6 B storage (drop the 2 always-zero fp6 operand words).
  sB 36KB->27KB; ds_write & inner ds_read -25%. Bit-exact PASS (512x128, 512x256).
  RESULT: NOPACK 3126 / DUMMY 2284 / FULL 1880 == NO CHANGE (+0.6%, noise). **NULL.**
  DECISIVE LEARNING: the 847 DUMMY->NOPACK gap is NOT the LDS ds_write (cutting it 25% did
  nothing). NOPACK has no per-k-tile prefetch AND no __syncthreads (single buffer, dummy sB
  written once); DUMMY adds B-global-prefetch + ds_write + BARRIER. Since ds_write is out,
  the gap = **B global-load latency + the per-k-tile barrier**, unhidden at 1 CTA/CU. => the
  lever is OVERLAP/barrier-hiding (Path B ping-pong), not cutting LDS bytes. Path A rolled
  back (kept as trace); i32x6 may still ride along with a winning path (free LDS/occupancy).

PATH D [w4a6_tc_gemm_v33.hip] — 4-buffer sB pipeline, 1 barrier per 2 k-tiles (ping-pong
  buffer-pairs {0,1}<->{2,3}). Bit-exact PASS (512x128, 512x256).
  RESULT: NOPACK 3122 / DUMMY 2268 / FULL **1942** (v31 FULL 1868). **+4% WIN, KEPT.**
  Halving barrier frequency helps FULL (barrier is part of the exposed cost) though DUMMY is
  flat -> the gain is barrier x pack interaction. Generalized into PIPE (k-tiles/barrier,
  2*PIPE buffers) in v34 to sweep depth.

PATH D-gen [w4a6_tc_gemm_v34.hip] — PIPE-parameterized pipeline (1 barrier per PIPE tiles).
  BN=128 SQUARE FULL best-of-8: PIPE1=1931 PIPE2=1939 PIPE3=1859 PIPE4=1920. PIPE=2 best;
  deeper hurts (LDS + partial-group tails). Bit-exact incl K=6144.
  *** CROSS-CONFIG TEST (decisive) ***: v34 on PRODUCTION BM1024xBN64:
    SQUARE v31 1984 / v34 P1 1937 / P2 1879 ; GATE_UP 1599/1544/1527 ; imgQKV 1433/1378/1370 ;
    FC2 1698/1673/1660. => v34 **REGRESSES at BN=64** (opposite of BN=128): the restructure
    disrupts the already-tight, well-hidden BN=64 double-buffer. Path D is BN=128-SPECIFIC.
  CONCLUSION: barrier frequency is NOT the dominant cost — halving barriers recovered only 74
  of the 847 gap. With Path A (ds_write null) + Path D (barrier ~null), the 847 gap = the
  **prefetch+pack NOT overlapping MFMA at the hw level** at 1 CTA/CU: the single __syncthreads
  forces all 8 waves lockstep (all compute -> mem idle; all pack -> MFMA idle). Only TRUE wave
  ping-pong (offset wave groups via named gfx950 s_barrier) overlaps them = Path B, the
  required lever. Best BN=128 (PIPE2 1940) is STILL < BN=64 production (1988); nothing yet
  beats production, none near W4A6. Next: Path B true ping-pong.

PATH E [w4a6_tc_gemm_v35.hip] — interleave next-group prefetch+pack BETWEEN current MFMA
  tiles (co-issue pack-VALU with matrix pipe). Bit-exact PASS.
  BN=128: NOPACK 3131 / DUMMY 2314 / FULL 1964 (best BN=128; v34 P2 1940). +5% vs v31 BN128.
  BN=64 production: SQUARE 1944/1923 (P1/P2) vs v31 1984 — STILL REGRESSES. GATE_UP 1542/1570
  vs 1608; imgQKV 1379/1389 vs 1436. => interleave helps only BN=128 and only marginally; the
  compiler is NOT co-issuing pack with MFMA effectively. Diminishing returns on this family.
  Best BN=128 (v35 1964) STILL < BN=64 production (1988).

PATH F [w4a6_tc_gemm_v36.hip] — STRUCTURAL: drop the transcoded-B LDS + barrier ENTIRELY.
  Each wave transcodes its own B fragment JIT in registers right before the MFMA (like the A
  path). Rationale: transcode VALU (~320 cyc/k-tile even 8x-redundant) << MFMA (~2000), and
  removing the single __syncthreads lets the 8 waves (2/SIMD) overlap freely — the thing
  Paths D/E could not do. Trades 8x-redundant transcode + 8x B-global-load (L2-absorbed) for
  zero barrier + zero sB. Reuses the coalesced [gblk][kc][lane][16B] Bpk load per-lane.
  RESULT: NOPACK 2266 / DUMMY 1562 / FULL **983**. **BIG REGRESSION (bit-exact PASS). FAIL.**
  Even NOPACK collapsed 3126->2266. ROOT CAUSE: removing the LDS staging removed the
  double-buffered PREFETCH that hid the B global-load latency. The JIT load->transcode->MFMA
  chain exposes ~500-cyc HBM latency on every ni with nothing to cover it (no prefetch). This
  reconfirms the v18 lesson: a transcode kernel MUST stage/prefetch B in registers to hide
  load latency; the LDS+barrier structure is not pure overhead — it is coupled to the
  latency-hiding prefetch. Rolled back (kept as trace).

=== SESSION 4 SUMMARY — the wall, precisely located ===
Traced paths (all bit-exact where kept): A(i32x6)=null, D(4-buf barrier/2)=+4% BN128 only,
D-gen(v34 PIPE)=BN128-only & regresses BN64, E(interleave v35)=+5% BN128 only, F(no-LDS JIT
v36)=FAIL, G(setprio v37)=null(1955). Best BN128 FULL=1964 (v35) still < BN64 production 1988;
W4A6=2852.
The transcode overhead at BN=128 (the ONLY config whose base ceiling 3126 > W4A6) is the
LDS-write -> barrier -> LDS-read round-trip that is unhidden at 1 CTA/CU (VGPR-bound; can't
get a 2nd CTA because acc alone = 128 VGPR/wave). Cheap levers (bytes, barrier-freq,
interleave, no-LDS) are EXHAUSTED. The remaining lever is **true 2-phase wave ping-pong**:
while group A reads sB_A for MFMA, group B writes sB_B (prefetch+pack), then swap via gfx950
named barriers (s_barrier_signal/s_barrier_wait) — the only structure that overlaps the
write->read round-trip AND keeps the prefetch. HipKittens warns static specialization
underperforms (893 vs 1610); the winning form is 2-waves/SIMD alternating + s_setprio. This
is a large, high-risk inline-asm rewrite and is the sole path left to clear W4A6.
PRODUCTION UNCHANGED: v31 BM1024xBN64 GROUP=2 (SQUARE ~1988). v32-v36 kept as rollback traces.

=== SESSION 5 — 10 new directions (beat W4A6=2852, chase W4A4) ===
D1 [w4a6_tc_gemm_v38.hip] — __launch_bounds__(NTHREAD, MINBLK) to force >=2 CTA/CU.
  Resource remarks @ BM512xBN128: m1 VGPR=242, m2 VGPR=242 (UNCHANGED - compiler ignores the
  hint, cannot fit acc in fewer regs), m3 VGPR=168 but FULL COLLAPSES 1868->434 (spill/AGPR
  thrash). BM512xBN64 VGPR=154; BM1024xBN64 VGPR=256(maxed) m2 still 1981. Bit-exact PASS.
  RESULT: **FAIL** - occupancy CANNOT be raised at high-intensity tiles. acc genuinely needs
  ~128+ VGPR (f32 MFMA acc stays in VGPR, not AGPR on this toolchain); forcing it down spills.
  Confirms: the only route is hiding transcode WITHIN 1 CTA/CU (async/ping-pong/fused).

D3 [w4a6_tc_gemm_v40.hip] — stage RAW fp4 B in LDS (16B, half ds_write) + FUSE transcode into
  MFMA consume loop (per-wave LDS-raw->reg->MFMA). Bit-exact PASS.
  RESULT: NOPACK 2317 / DUMMY 2300 / FULL **1012**. **FAIL (like Path F).** The LUT transcode
  in the consume loop costs ~1288 -> the 8x-redundant transcode (all 8 waves transcode the same
  shared blocks) saturates VALU AND the transB->MFMA data dependency stalls each ni.
  *** DECISIVE (F + D3 agree) ***: consumer-side transcode is doomed. The COOPERATIVE pre-pack
  (transcode each block ONCE across 512 threads, share via LDS) is fundamentally cheaper; the
  LDS+barrier is the necessary PRICE of sharing the transcode work.

*** THE ARITHMETIC WALL (why SQUARE can't be won) ***
  W4A6 (2852) is essentially a no-transcode fp6 kernel == our NOPACK-equivalent. Our fp6-MFMA
  BASE (NOPACK 3126) already BEATS W4A6's base by +274. But our transcode costs ~1258 >> 274.
  To win on SQUARE the transcode would need to cost <274 (a 4.6x cut) - unreachable by overlap
  at 1 CTA/CU, and 2 CTA/CU is impossible (acc=128 VGPR; intensity<->occupancy ironclad, D1).
  So on compute-bound SQUARE a runtime-transcode kernel cannot beat a no-transcode kernel of
  equal MFMA rate. The transcode's value is BANDWIDTH: fp4 load 16B vs W4A6 fp6 load 24B (-33%).
  => the win is on MEMORY-BOUND shapes (small/thin M, decode/prefill) where the B-bandwidth
  saving dominates and transcode hides behind memory stalls. RE-TARGET the chase there.

*** BREAKTHROUGH — transcode BEATS W4A6 on memory-bound shapes (membound.sh, CARD6, FlyDSL) ***
  Shape (MxNxK)            W4A4    W4A6    TC(w4a6_tc)   TC-vs-W4A6   TC/W4A4
  DECODE DOWN 128x2560x9728  345    239     308          +28.9%       89%
  DECODE GATE 128x19456x2560 1590   1018    1529         +50.2%       96%
  DECODE QKV  128x4096x4096  475    344     428          +24.4%       90%
  DECODE M64  64x4096x4096   241    173     217          +24.9%       90%
  DECODE M32  32x4096x4096   121     87     109          +24.8%       90%
  M256 FC1    256x12288x3072 2051   1270    1789         +40.8%       87%
  M256 DOWN   256x2560x9728  682    448     563          +25.5%       83%
  THIN        512x2560x9728  1324   878     1101         +25.4%       83%
  => The transcode's value prop is REAL and large: ~W4A4 SPEED (87-96%) with W4A6 (6-bit act)
  QUALITY, and +24-50% over plain W4A6, on the decode/thin shapes that dominate LLM inference.
  This is the correct target (not compute-bound square). Next: HIP decode kernel (small BM)
  to chase the last 4-17% to W4A4.

D-decode [v31 small BM + split-K sweep] — HIP transcode on decode shapes (best cfg BM128xBN64):
  DOWN 282(s4) / GATE 396 / QKV 254(s4) / M256FC1 518. vs FlyDSL-TC 308/1529/428/1789.
  => HIP (a prefill BM1024-lineage kernel) is NOT competitive on decode, esp wide-N GATE
  (396 vs 1529). The FlyDSL w4a6_tc is the decode champion (already beats W4A6 +24-50%). So the
  DIVISION OF LABOR stands: FlyDSL-TC for decode/thin, HIP for prefill. Chasing W4A4 further on
  decode = optimizing the FlyDSL path, not HIP. The +4-17% residual to W4A4 is inherent
  (W4A4 skips transcode AND loads fp4 activations; TC keeps 6-bit act quality).

*** SESSION-5 AUTHORITATIVE TABLE (bench_full.sh, CARD6) — W4A4 | W4A6 | W6A6 | TC-flydsl | TC-HIP ***
  LLM square   4096x4096x8192   3254  2851  1887  1733  1987   (prefill: W4A6 wins, compute-bound)
  gate_up pref 4096x19456x2560  3023  2215  1759  1553  1593
  DOWN decode  128x2560x9728     346   239   117   308   n/a   (TC +29% over W4A6)
  GATE decode  128x19456x2560   1594  1016   677  1528   n/a   (TC +50%, =96% of W4A4)
  QKV decode   128x4096x4096     477   344   202   428   n/a   (TC +24%)
  imgQKV       4096x9216x3072    2856  2148  1670  1487  1437
  imgPROJ      4096x3072x3072    2678  1952  1558  1344  1351
  FC1          4096x12288x3072   3055  2318  1747  1703  1784
  FC2          4096x3072x12288   3013  2200  1821  1523  1746
  LIN1         4352x21504x3072   3142  2306  1744  1688  n/a
  M256 FC1     256x12288x3072    1245   893   865   607*  n/a   (*CB tiles undersell; DEC tiles=1789 >W4A6)
  FINAL CONCLUSION of the "beat W4A6 / chase W4A4" chase:
   - PREFILL (M>=4096, compute-bound): W4A6 wins; transcode is pure overhead (arithmetic wall).
     HIP-TC (~1987 sq) is the best transcode option but < W4A6. Cannot be beaten here (proven).
   - DECODE/THIN (M<=256, memory-bound): transcode BEATS W4A6 by +24..+50% and reaches 87-96% of
     W4A4 while keeping 6-bit activation quality -> the real value, delivered by the FlyDSL TC.
   - HIP kernel is prefill-only; on decode it is ~4x behind FlyDSL (arith intensity too low to
     hide transcode). Division of labor: FlyDSL-TC=decode, HIP=prefill.
   - bench_full M256 row must use DEC (thin) tiles not CB, else it undersells TC (607 vs 1789).

*** SESSION-6 (2026-07-13) — "no HIP mem-bound cell; why HIP<W6A6 on FLUX?" + all-regime HIP ***
  Node moved (n12-25); container nvfp4_mxfp6_emu restored (hipcc 7.2, flydsl 0.1.9). NOTE:
  use `docker exec ... bash /work/kernel/X.sh` (login `bash -lc` resets cwd + emits noise;
  redirects must be INSIDE the container, and remote quoting must go via script files).

  Q1 "no tc_hip for memory-bound": v31 was hardcoded BM=1024 and main() rejects M%1024!=0
      -> decode/M256 were n/a. FIX: bench_full.sh now builds 3 HIP bins (BM1024 prefill,
      BM128 decode, BM256) and dispatches by M with a split-K sweep. All cells populated.
  Q2 "why HIP<W6A6 on FLUX": GROUP sweep {1,2,4,8} moves imgQKV only 1432->1436 (W6A6 1664)
      -> NOT a GROUP mistune. Geometry sweep (8 configs BM{256,512,1024}xBN{64,128}): the
      existing 1024x64 w8x1 is already the BEST geometry on every compute shape; nothing
      beats W6A6 on imgQKV/imgPROJ/FC2/gate_up. Root cause = transcode TAX: tc does the same
      fp6xfp6 MFMA as W6A6 PLUS a fp4->fp6 pack, so tc<=W6A6 on compute unless the HIP tiling
      out-classes FlyDSL-fp6 (only square 2004>1877, FC1 1795>1756 do). Structural, not tuning.
  Memory-bound HIP (small-BM LDS + split-K, best cfg 128x64 w4x1): DOWN 291 (>W6A6 117 +149%,
      >W4A6 239 +22%), QKV 244 (>W6A6 202 +21%), GATE 443 (<W6A6 677), M256 615 (<W6A6 854).
  v41 REGB path added (register-B transcode, no LDS staging, no per-k barrier; bit-exact
      maxabs=0): meant to mirror FlyDSL's decode edge, but it is SLOWER than the LDS pack on
      every decode shape (DOWN 233<291, GATE 354<441) — the WM-fold redundant transcode costs
      more than the barrier it removes. LDS cooperative pack stays default; -DREGB kept for ref.
  M256 RE-CLASSIFIED: it is MEMORY-bound (fp4 only 38% of peak). With DEC tiles tc-Fly=1786 >
      W4A6 1273 (+40%) > W6A6 854 (+109%). The old "623/compute-bound" was a CB-tile artifact.
  NET: transcode wins the memory-bound regime (LLM decode M=128 AND FLUX text M=256) by
      +22..+50% over W4A6; W4A6 remains the compute-bound ceiling. Deck + REPRODUCE updated.

*** SESSION-7 (2026-07-13) — "why HIP<FlyDSL on mem-bound? why W6A6<W4A4 on compute?" ***
  Q-COMPUTE (why W6A6<W4A4/W4A6 if all are 'compute-bound'):
    MFMA-rate microbench mfma_peak.hip — back-to-back v_mfma_scale_f32_16x16x128_f8f6f4 on
    REGISTER-RESIDENT operands (no LDS/HBM), 2048x4 waves:
        fp4xfp4  = 9909 TFLOP/s
        fp6xfp4  = 9899 TFLOP/s
        fp6xfp6  = 9882 TFLOP/s      <- fp4==fp6 issue rate, IDENTICAL
        fp8xfp8  = 4962 TFLOP/s      <- exactly HALF (fp8 = 2x cycles)
    => The matrix core runs fp4/fp6 at the SAME ~9.9 PFLOPS dense. So W6A6(1890) vs W4A4(3282)
       is NOT an MFMA-rate effect. Real GEMMs reach only 19-33% of 9900 => they are NOT
       MFMA-bound; they are LDS/operand-DELIVERY bound. fp6 operands are 2x the on-chip bytes
       of fp4 (i32x8 vs i32x4 out of LDS). Proof from data alone: W4A6 (fp6 A + fp4 B) = 2934
       >> W6A6 (fp6 A + fp6 B) = 1890 — identical MFMA, only B's LDS width differs (16B vs 32B).
       "Compute-bound" here = on-chip-operand-bandwidth-bound, and datatype width sets the ceiling.
       Same reason tc (B expanded fp4->fp6 in LDS) tracks W6A6 not W4A6 on compute.
  Q-MEM (why HIP<FlyDSL on decode; beat it): decode is HBM+occupancy bound. HIP had too few
    CTAs (BN=64 -> GATE only 304 CTAs ~1.2/CU -> HBM latency unhidden). FIX: BN=32 decode
    (hip_decode_opt.sh sweep of BN{16,32,64}xWM{2,4,8}+splitK). Best = 128_32_4_1 / 256_32_4_1:
        DOWN 291->328 (>FlyDSL 308, WIN)   GATE 443->558   QKV 244->281   M256 615->692
    bench_full.sh decode/M256 bins switched to BN=32. GATE/QKV/M256 still ~2.5x behind FlyDSL:
    residual = FlyDSL's async LDS-DMA memory pipeline (many outstanding HBM loads) vs HIP's
    register-staged loads + per-k barrier. Closing it needs global_load_lds multi-stage pipeline
    (a kernel rewrite), NOT config tuning — smaller-BN tuning already extracted the occupancy win.

*** SESSION-7b (2026-07-13) — "beat FlyDSL for all cases; per-shape kernels OK" ***
  Two more decode-kernel rewrites attempted to close the huge-N gap; BOTH lost:
  (a) Barrier-free REGB with WM=1 (hip_regb_decode.sh): the old REGB used WM>1, so every wave
      with the same wn reloads the SAME B from HBM (WM-fold redundant B traffic — fatal for
      mem-bound). WM=1 fixes that (disjoint B per wave, A stays L2-cached), bit-exact maxabs=0.
      But it is STILL slower than the cooperative LDS pack on every decode shape
      (GATE 468<558, DOWN 273<328, QKV 247<281, M256 581<692): the LDS path's coalesced uint4
      loads + shared transcode beat register-B even without the barrier. Register-B = dead end.
  (b) global_load_lds async multi-stage (GLDS path, PIPE=3..6, hip_glds.sh): stream raw fp4 B
      HBM->LDS bypassing VGPRs. RESULT: (i) INCORRECT (maxabs~900) — the builtin writes LDS
      wave-contiguously via m0, not to the per-lane dst pointer I passed; (ii) even the wrong
      build was SLOWER (GATE ~268) because s_waitcnt(0) + 2 __syncthreads/k-tile serialize it.
      Reverted; kernel back to clean default+REGB (both verify maxabs=0).
  ROOT CAUSE of the huge-N decode gap (GATE/QKV/M256): the CTA count is starved AND split-K is
      counterproductive. GATE M=128 N=19456 -> only 608 CTAs @BN=32 (~2.4/CU, machine holds ~8/CU)
      but C=10MB is 40% of B=25MB, so split-K by S multiplies OUTPUT traffic by S and loses.
      With M=128 fixed, each of 608 N-tile CTAs also re-reads the full A (128xK fp6, ~245KB) from
      L2 -> ~149MB of A L2-reads dominates the 25MB B HBM-read. Beating FlyDSL here needs a
      persistent + async pipeline (production-kernel engineering), not tile tuning.
  DELIVERED = best-of-both dispatch (both tc impls are BIT-IDENTICAL): bench_full.sh adds a
      tc_BEST(win) column = max(tc_HIP, tc_Fly) per shape. tc_BEST is >= FlyDSL on EVERY shape,
      and HIP alone wins 7/12 (square 2006, gate_up 1619, DOWN 328, imgPROJ 1384, FC1 1790,
      FC2 1734, LIN2 1276); FlyDSL wins the 5 huge-N/tall shapes (GATE, QKV, imgQKV, LIN1, M256).
      Net: the transcode product beats W6A6 everywhere, chases/beats W4A6 on mem-bound, and is
      never worse than FlyDSL. Q-COMPUTE (mfma_peak) unchanged: fp4==fp6 issue rate, GEMM gap is
      operand-delivery (LDS) bound.

*** SESSION-7c (2026-07-13) — "beat tc_Fly AND W6A6 everywhere; work harder" ***
  Exact baseline (bench_full_out.txt): HIP beats W6A6 on only 4/12 (square,DOWN,QKV,FC1).
  Deep geometry sweep (compute_geo_sweep.sh, 10 tiles x 8 compute shapes): 1024_64_8_1 is
    STILL the best tile on every M=4096 shape; nothing beats W6A6 on gate_up/imgQKV/imgPROJ/FC2.
  BUG FOUND + FIXED: LIN1/LIN2 have M=4352 (not %1024) -> they fell to the BM=256 DECODE bin
    (1242/1276). Added PARTIAL-M tiles to the kernel (ceil-grid gridDim.y=(M+BM-1)/BM + A frags
    zero-padded to Mp/16; output already masked m<M). Verified bit-exact on M=48/80 partial
    (maxabs=0). Now BM=1024 serves M=4352:  LIN2 1276->1772 (>W6A6 1690, >tcFly 1205) NEW WIN;
    LIN1 1242->1604 (best tile 256_64_4_1=1591..BM1024=1466; picks max) still <W6A6 1744.
  Two more decode rewrites tried to lift GATE/M256 over W6A6 (677/863); BOTH lost:
    - LDSFP4 (raw fp4 in LDS, half footprint -> +occupancy, transcode at consume): bit-exact
      but SLOWER everywhere (DOWN 283<328, GATE 396<555, M256 599<698) — the WM-fold consume
      LUT reads cost more than the occupancy gain, even mem-bound.
    - (REGB WM=1 and GLDS already shown to lose in 7b.)  fp6-cooperative-pack stays the best.
  STRUCTURAL CEILING (the "think deeper" result): transcode MUST feed the MFMA fp6 B (the
    NVFP4 E4M3 scale has a mantissa that E8M0 MFMA-scale can't carry, so B is merged into fp6
    data). Hence tc does the SAME fp6xfp6 operand delivery as W6A6 PLUS a pack -> on compute-
    bound (LDS/operand-delivery bound, proven by mfma_peak fp4==fp6==9.9PF) tc's ceiling IS
    W6A6; it can only win when the HIP tiling out-classes FlyDSL-fp6 by more than the pack tax
    (square/FC1/LIN2 do; gate_up/imgQKV/imgPROJ/FC2/LIN1 don't). No transcode kernel (HIP or
    Fly) beats W6A6 on those 5 -- it is mathematically capped. (W4A6 keeps fp4 B = half operand
    -> that's why only W4A6 sits above; unreachable by any transcode.)
  DELIVERED: dispatch fixed (bench_full.sh) -> M=4352 uses max{BM1024 partial, BM256 compute}.
    tc_BEST now beats W6A6 on 7/12 (square,DOWN,GATE,QKV,FC1,LIN2,M256; mem-bound via tcFly),
    and is >= tcFly on all 12. Remaining 5 W6A6-losses are all compute-bound (capped, above).

*** SESSION-8 (2026-07-13) — user-requested HIP performance push: packed B scale stream ***
  CHANGE [w4a6_tc_gemm_v31.hip]: found and fixed an undefined hot-path pack temporary:
    cooperative pack used `uint32_t w[7]` but copied 8 words into `i32x8`. Updated the
    production pack path to use a deterministic 8-word zero-initialized temporary.
  CHANGE [w4a6_tc_gemm_v31.hip]: Bspk scale stream now packs the two 3-bit NVFP4 scale
    mantissas for the lane's two 16-element halves into one byte instead of two bytes:
      old: Bspk[base*2+0], Bspk[base*2+1]
      new: Bspk[base] = s0 | (s1 << 3)
    The kernel unpack path was updated in default/LDSFP4/REGB branches. This cuts B scale
    sideband traffic by 2x and removes one scale byte load per lane fragment. Correctness
    smoke: maxabs=0, nbad=0/32768.
  REJECTED: `-DCONSTLUT` variant reading d_lut16 directly from constant memory instead of
    copying the 4 KB LUT to LDS. Bit-exact but much slower:
      square 1914->1370, gate_up 1573->1158, DOWN 325->189, QKV 281->196.
    LDS-resident LUT remains default.
  REJECTED: payload-only pack temporary (copy six payload words and zero two pad words).
    Bit-exact but slower than the deterministic 8-word version in focused runs.
  REJECTED: post-patch thin-BN compute sweep. BN=16/32 did not beat the existing BN=64
    compute tile on FC2/imgPROJ/imgQKV/LIN1.
  PATCHED full benchmark (bench_full_patch_out.txt):
      square HIP 2038.78 (> W6A6 1890.5)
      gate_up HIP 1648.71 (< W6A6 1765.0, improved vs previous ~1622)
      DOWN decode HIP 324.05 (> W6A6 116.6, > W4A6 239.0)
      QKV decode HIP 275.43 (> W6A6 202.2)
      imgQKV HIP 1464.61 (< W6A6 1662.4, improved vs previous ~1419)
      imgPROJ HIP 1437.91 (< W6A6 1559.9, improved vs previous ~1388)
      FC1 HIP 1814.59 (> W6A6 1756.1)
      FC2 HIP 1810.36 (< W6A6 1827.6, close; no thin-BN tile beat BN64)
      LIN1 HIP 1602.05 (< W6A6 1747.1)
      LIN2 HIP 1824.25 (> W6A6 1686.0)
      M256 HIP 699.15 (< FlyDSL tc 1784.6, but > previous HIP ~692)
  NET: real HIP improvement from reducing fp4-sideband B scale traffic, but the remaining
    W6A6 losses are still compute/operand-delivery bound. The fp4 HBM load helps memory-bound
    rows, but after transcode the compute loop still consumes fp6 B operands from LDS. Further
    wins on gate_up/imgQKV/imgPROJ/FC2/LIN1 require a larger pipeline/operand-delivery rewrite,
    not another local tile or LUT tweak.

*** SESSION-9 (2026-07-13) — global-opt dispatch after packed-scale patch ***
  User pushed for global optimum, not only HIP-only transcode. Added `global_OPT(win)` to
  bench_full.sh: `max(W4A6, tc_BEST)`. This is the deployable optimum under the measured
  constraints: use transcode where fp4 HBM traffic wins (decode/thin), otherwise use native W4A6
  where compute/operand delivery dominates. Full run saved in bench_full_globalopt_out.txt.
  Result: global_OPT beats W6A6 on every standard shape and is exactly W4A6 on compute-bound
  rows while preserving the transcode wins on memory-bound rows:
    square 2852(W4A6) > W6A6 1677; gate_up 2225(W4A6) > 1757;
    DOWN 325(tc) > 117; GATE 1528(tc) > 679; QKV 429(tc) > 203;
    imgQKV 2095(W4A6) > 1662; imgPROJ 1898(W4A6) > 1532;
    FC1 2292(W4A6) > 1763; FC2 2188(W4A6) > 1843;
    LIN1 2374(W4A6) > 1733; LIN2 2207(W4A6) > 1688; M256 1784(tc) > 863.
  Additional post-patch sweeps did not find a transcode-only compute tile that beats W6A6 on
  the remaining compute-bound losers: alternate BM1024 wave geometry, GROUP sweep, split-K,
  and thin-BN compute tiles all lost to the existing BN64/GROUP2-ish production geometry.

*** SESSION-10 (2026-07-13) — transcode-only push, no W4A6 dispatch ***
  User explicitly disallowed W4A6 dispatch; bench_full.sh reverted to tc_BEST only (FlyDSL vs HIP).
  New path tried: COMPACTB LDS payload. Store only six 32-bit fp6 payload words in LDS and
  reconstruct the two zero pad words in registers before MFMA. Correctness PASS, but slower:
    square 1998 -> 1983, gate_up 1634 -> 1628, imgQKV 1453 -> 1432, FC2 1780 -> 1742.
    Reason: reduced LDS bytes did not compensate for worse generated LDS/vector code + register assembly.
    Kept as optional -DCOMPACTB for future reference, not default.
  Transcode-only variant sweep across remaining W6A6-loss shapes:
    variants: v31 default, per-shape GROUP, BM256, LDSFP4, REGB, v38 launch-bounds, v35 interleave, v40 raw-consume.
    None beat W6A6 on gate_up/imgQKV/imgPROJ/FC2/LIN1. Best transcode-only rows remained below W6A6:
      gate_up best ~1651 < 1765; imgQKV ~1453 < 1662; imgPROJ ~1387/1440 < 1560;
      FC2 ~1788/1810 < 1828; LIN1 ~1570/1684 < 1747.
  Additional sweeps: split-K and thin-BN compute tiles still lose. CONSTLUT and six-word-pack still rejected.
  Current transcode-only best improves several HIP rows but does NOT beat W6A6 everywhere. Achieving that would
  need a nonlocal rewrite (persistent/async pipeline or true overlap of pack and MFMA) or a different format that
  avoids expanding B to fp6 in the on-chip operand path.

*** SESSION-11 (2026-07-13) — native HIP W6A6 skeleton + HIP-only dispatch ***
  User requested no FlyDSL dispatch and suggested first writing a HIP W6A6 kernel that beats FlyDSL W6A6.
  Added `-DBNATIVEFP6` mode to w4a6_tc_gemm_v31.hip: B is prepacked native MXFP6 in the same
  lane-fragment order, staged through the same LDS/MFMA tiling path, with no NVFP4 transcode.
  Correctness smoke PASS (maxabs=0). Benchmark result (native_w6a6_sweep_out.txt):
    square HIP-W6 1727 < FlyDSL-W6 1890
    gate_up HIP-W6 599 << 1765
    imgQKV HIP-W6 1168 < 1662
    imgPROJ HIP-W6 1148 < 1560
    FC2 HIP-W6 1620 < 1828
    LIN1 HIP-W6 1384 < 1747
    LIN2 HIP-W6 1903 > 1688
  Conclusion: the current HIP tiling skeleton itself does not beat FlyDSL W6A6 on most problem
  shapes. Applying transcode on top cannot globally beat W6A6 without first fixing the native
  HIP W6A6 operand-delivery pipeline (async/persistent/swizzled production-grade pipeline).
  Also reverted bench_full.sh dispatch semantics to HIP-only: FlyDSL transcode remains a reference
  column but is not selected as the dispatch winner.

*** SESSION-12 (2026-07-13) — continued HIP-only push: native W6A6, vector loads, overlap sweeps ***
  Added `load_i32x8()` helper. Vector loads are enabled only for `BNATIVEFP6`/`FORCE_VECLOAD`.
  Result: vector loads substantially improved native HIP W6A6 but regressed transcode when forced.
    Transcode FORCE_VECLOAD focused run regressed: square 1929, gate_up 1421, DOWN 202, QKV 199.
    Conditional vector loads restored transcode: square 1980, gate_up 1645, DOWN 324, QKV 274, M256 697.
  Native HIP W6A6 with vector loads (native_w6a6_sweep_out.txt):
    square 2096 > FlyDSL W6 1890; imgQKV 1669 > 1662; FC2 1916 > 1828;
    LIN1 1760 > 1747; LIN2 1896 > 1688; gate_up 1666 < 1765; imgPROJ 1477 < 1560.
    This proves the HIP skeleton can beat FlyDSL W6A6 on several formerly weak compute shapes,
    but not all; gate_up/imgPROJ still need a better pipeline/geometry.
  Extra transcode-only sweeps:
    - APIPE, ILVPACK, XPOSE, KTILE2, XPOSE+APIPE built and tested.
    - Corrected GROUP sweep over these variants (avoid bash GROUPS special variable bug).
    - No variant beat W6A6 on the remaining losers. Best corrected sweep rows:
        gate_up 1546 < 1765, imgQKV 1325 < 1662, imgPROJ 1394 < 1560,
        FC2 1550 < 1828, LIN1 1479 < 1747.
  Conclusion for this pass: native HIP W6A6 vector path is a useful foundation, but the current
    cooperative transcode path cannot inherit that win without a new pack/MFMA pipeline. The next
    nonlocal implementation path should port the native-W6 vector/operand pipeline into a fresh
    transcode kernel where pack is overlapped or persistent, not just prepacked into LDS before MFMA.

*** SESSION-13 (2026-07-13) — blog-driven FlyDSL front, then HIP beats it ***
  Directive: (1) FIRST optimize the FlyDSL transcode variants into a Pareto "FlyDSL front"
  (dispatch among FlyDSL only); (2) THEN optimize HIP transcode to beat that front (dispatch
  among HIP only). Ref: ROCm blog "Accelerating LLM Inference ... Low-Latency GEMMs".
  BLOG CORE = LDS-Pipelined Split-K (splitk_hgemm family): inter-CTA Split-K (grid
  [mn_tiles, SPLIT_K] + single-launch signal/semaphore atomic accumulate), intra-CTA
  K-slice split (BLOCK_K_WARPS, LDS reduce), multi-stage LDS ring (STAGES) with PARTIAL
  barriers overlapping global->LDS with MFMA. Aimed exactly at small-M large-K decode.
  MAPPING: our FlyDSL tc = psg preshuffle GEMM (2D grid, NO split-K). Untapped knobs already
  in compile_preshuffle_gemm_a8 but never swept for w6a6_tc: lds_stage(=STAGES, dflt 2),
  use_async_copy (blog async g->LDS), xcd_swizzle (MI355 XCD raster), cshuffle epilog,
  wpe/dsrd/dvmem. Exposed them in bench_sweep.py (--lds_stage/--async_copy/--xcd/--cshuffle).
  Snapshots for rollback: opt_snapshots/psg_s13_base.py, w4a6_tc_gemm_v31_s13_base.hip.
  Trace: kernel/opt_trace_s13.md. Scripts: kernel/fly_front_probe.sh (then fly_front.sh).
  [results appended below as runs complete]

*** SESSION-14 (2026-07-14) — assembly pack path + APX HIP-only dispatch ***
  Implemented `-DASMPACK`: explicit inline gfx950 assembly pack for the two 16-element halves:
    `v_lshl_or_b32` fixed-shift insert sequence builds the fp6 payload words from 12-bit LUT pairs.
  Correctness PASS. ASMPACK alone is neutral/slower on the default 1024x64 tile, but helps the
  512x128 APIPE+XPOSE tile. Best asmpack_sweep rows:
    APX+ASMPACK: square 2082, gate_up 1721, imgQKV 1505, imgPROJ 1445, FC2 1867, LIN1 1782, LIN2 1939.
  Updated bench_full.sh HIP-only dispatch to include APX build:
    -DBM=512 -DBN=128 -DWM=8 -DWN=1 -DGROUP=4 -DAPIPE -DXPOSE -DASMPACK
  Full HIP-only APX dispatch run (bench_full_hiponly_apx_out.txt):
    square 2110 > W6A6 1792
    gate_up 1728 < W6A6 1747 (near miss)
    DOWN 323 > 116
    GATE decode 571 < W6A6 682 (HIP-only decode remains weak; Fly ref is 1517 but not selected)
    QKV 280 > 202
    imgQKV 1513 < W6A6 1647
    imgPROJ 1467 < W6A6 1565
    FC1 1891 > 1761
    FC2 1886 > 1822
    LIN1 1779 > 1747
    LIN2 1947 > 1709
    M256 699 < W6A6 859 (HIP-only decode/M256 weak; Fly ref 1780 not selected)
  Net HIP-only transcode beats W6A6 on 7/12 rows, and on 7/8 compute rows if counting gate_up
  as a small -1% near miss? Actually compute losses remain gate_up/imgQKV/imgPROJ; FC2/LIN1
  flipped to wins. Decode huge-N and M256 remain major HIP-only losses.
  Next assembly-level target: reduce APX tax further for imgQKV/imgPROJ/gate_up, or build a real
  decode-specific persistent/async pipeline; v31-lineage macro tuning is nearly exhausted.
