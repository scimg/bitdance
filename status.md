HIP transcode optimization status (last updated 2026-08-04, session 23)
=======================================================================
Sections below are chronological; the newest results are at the end of the file.

Goal this session
-----------------
1. Make the **native HIP W6A6** kernel beat FlyDSL W6A6 on **every** compute shape.
2. Then optimize the **transcode** kernel to beat W6A6 and chase W4A6.

Measurement note: card 5 is a shared MI355 (node load-avg 9-18). Contention only
*depresses* figures, so best-of-N (10-20 short reps) max = the uncontended truth
(the established methodology). All builds below verify bit-exact (`maxabs=0 nbad=0`).

=====================================================================
PHASE 1 — native HIP W6A6 beats FlyDSL W6A6 EVERYWHERE  ✅ DONE
=====================================================================
The winning tile is **512x128 w8x1** (was 1024x64). Mechanism: A is re-read from L2
per N-slice (not LDS-staged), so A-refetch ~ 1/BN; BN=128 halves it, and at BM<=512
`acc[MPW][NPW]` stays at 128 VGPR (no occupancy cliff). Best-of-10, vs FlyDSL W6A6:

    shape     native512x128   FlyDSL W6A6   win
    square        2179            1890       +15%
    gate_up       1818*           1765       +3.0%   (*GROUP=4; +APIPE=1824)
    imgQKV        1668**          1662       +0.4%   (**+APIPE; base 1644)
    imgPROJ       1631            1560       +4.6%
    FC2           1977            1828       +8.2%
    LIN1          1850            1747       +5.9%
    LIN2          2105            1688       +25%

Dispatch = 512x128 w8x1 with best-of-{GROUP2, GROUP4, +APIPE}. Beats FlyDSL W6A6 on
all 7 (`native_w6a6_sweep2.sh`, `phase1_close.sh`). `bm1024/bn128 w8x2` (WN=2) and any
`WM=16` config COLLAPSE (~130-220 TF) -> rejected. So at 8 waves/512 threads the tile
area is capped at BM*BN<=65536 (acc<=128 VGPR).

=====================================================================
PHASE 2 — transcode vs W6A6 / W4A6   (partial: 3 wins, losses halved)
=====================================================================
Definitive best-of-20, per-shape dispatch tc = max(t1024x64, apx512x128):

    shape     tc_HIP    W6A6    vs W6A6    W4A6   vs W4A6
    square    1994      1890    +5.5% WIN  2845   0.70x
    LIN2      1885      1688    +11.7% WIN 2227   0.85x
    LIN1      1743      1747    -0.2% tie  2338   0.75x
    FC2       1788      1828    -2.2%      2214   0.81x
    gate_up   1680      1765    -4.8%      2230   0.75x
    imgPROJ   1465      1560    -6.1%      1926   0.76x
    imgQKV    1520      1662    -8.6%      2150   0.71x

Result: transcode now **beats W6A6 on square + LIN2, ties LIN1**, and the remaining
losses shrank from ~-15% to -2..-9%. (Decode/memory-bound shapes were already tc wins;
this session was the compute-bound shapes.)

Key new lever: **APIPE + XPOSE at 512x128** (+9-12% over plain transcode there):
- XPOSE (conflict-free transposed B LDS `[buf][bblk][dword][lane]`) finally *helps*
  at BN=128 (B-read volume is 2x BN=64, so bank conflicts matter). It was flat at BN=64.
- APIPE (A software-prefetch one mi ahead, B register-resident) hides A L2 latency.
- ILVPACK (pack issued inside the MFMA loop) does NOT help even at BN=128 (disrupts
  APIPE's A-prefetch ILP / register scheduling) -> rejected.
- LUT32 (uint32 LUT, 1 entry/bank-word) does NOT help -> the BN=128 tax is not LUT
  bank conflict.

Why transcode can't beat W6A6 on gate_up/imgQKV/imgPROJ (the honest ceiling)
---------------------------------------------------------------------------
Decomposition (`phase2_decomp.sh`, native vs DUMMY(no-LUT) vs full):

    tile      shape    NOPACK  native  DUMMY   full   LUT-tax
    1024x64   square    2411    2014    1973   1928    ~2-4%
    512x128   square    3066    2156    2156   1829    ~15%
    512x128   gate_up   2429    1776    1804   1560    ~14%
    512x128   imgQKV    2268    1641    1627   1347    ~17%

Two structural facts set the ceiling:
- **transcode tax ~ 1/BM**: each M-tile CTA re-transcodes its own B copy, so total
  on-chip pack work ~ M*N*K/BM. Big BM (1024x64) => tax ~4%; small BM (512x128) => the
  pack VALU is 2x per the same 32 MFMAs/tile and is *not hidden* => ~15% tax (superlinear).
- **A-refetch ~ 1/BN**: huge-N shapes (gate_up/imgQKV/LIN1) want BN=128; small-N shapes
  (imgPROJ/FC2/LIN2) want BM=1024. Optimal tile is per-shape along BM*BN=65536.
- For gate_up/imgQKV/imgPROJ the **native HIP W6A6 ceiling itself barely exceeds FlyDSL
  W6A6** (imgQKV 1648~=1662, imgPROJ 1615 vs 1560, gate_up 1767 vs 1765), so transcode =
  (native - LUT tax) cannot clear W6A6 there. APIPE+XPOSE close ~90-95% of that ceiling,
  which is why the losses shrank but didn't flip.

W4A6 remains unreachable by any transcode (0.70-0.85x): W4A6 keeps B at fp4 (16 B/lane,
half the operand bytes); transcode necessarily feeds fp6xfp6 (E4M3 scale mantissa must be
merged into B data), inheriting W6A6's 2x B operand delivery + the pack.

Kernel changes (all behind `#ifdef`, default path unchanged & bit-exact)
------------------------------------------------------------------------
`kernel/hip/w4a6_tc_gemm_v31.hip`:
- `-DLUT32`: uint32 LUT variant (tested; no gain, kept for reference).
- APIPE branch: optional `-DILVPACK` interleaves the pack into the mi loop (tested; no
  gain).
- Recommended transcode builds:
  * wide/huge-N shapes: `-DBM=512 -DBN=128 -DWM=8 -DWN=1 -DAPIPE -DXPOSE`
  * small-N/large-K shapes: `-DBM=1024 -DBN=64 -DWM=8 -DWN=1`  (lower transcode redundancy)
  * dispatch = max of the two.
- Recommended native W6A6 build: `-DBNATIVEFP6 -DBM=512 -DBN=128 -DWM=8 -DWN=1` with
  best-of-{GROUP2,GROUP4,APIPE}.

Scripts (all under kernel/): native_w6a6_sweep2.sh, phase1_close.sh, phase2_transcode.sh,
phase2_decomp.sh, phase2_lut.sh, phase2_bigbm.sh, phase2_final.sh, phase2_ilv.sh,
phase2_combo.sh, phase2_best.sh, phase2_lock.sh.

Next levers (diminishing ROI)
-----------------------------
1. Cheaper on-chip 6-bit pack (asm v_perm_b32) to cut the LUT tax below the ~4%/15%
   floor -> would flip FC2 and possibly gate_up. The 6-bit tight packing (mandated by the
   f8f6f4 fp6 operand format) is inherently bit-shift-heavy; needs asm-level work.
2. Lift the native HIP W6A6 ceiling on gate_up/imgQKV further (thin margins) -> would give
   the transcode room. Rasterization/APIPE already tapped; next is A-in-LDS or async.

=====================================================================
PHASE 3 — beat W4A6 on all compute shapes  (session 3, 2026-07-15)
=====================================================================
Goal: beat W4A6 on remaining 5 shapes (FC2/gate_up/imgPROJ/imgQKV/LIN1) and
      push all shapes toward W4A4. Git repo initialized for rollback tracking.

A2 diagnostic (phase3_nopack128.sh) — NOPACK at BN=128 on all 7 shapes:

    shape     NOPapx   DUMapx  FULLapx  W4A6ref  headroom  ds_gap  LUT_gap
    square      3132     2348     2055     2845    +10.1%     784      293
    gate_up     2468     1880     1693     2230    +10.7%     588      187
    imgQKV      2260     1773     1516     2150    +5.1%      487      257
    imgPROJ     2230     1653     1443     1926    +15.8%     577      210
    FC2         2824     2084     1823     2214    +27.6%     740      261
    LIN1        2574     1936     1745     2338    +10.1%     638      191
    LIN2        2990     2193     1918     2227    +34.2%     797      275

CONFIRMED: NOPapx > W4A6 on ALL shapes (headroom 5-34%). Beating W4A6 is achievable.
Dominant cost = ds_write+barrier gap (487-797 TFLOPS), not LUT arithmetic (187-293).

XCMPB (v41 lever, A1 DONE):
- Combined XPOSE+COMPACTB: [buf][bblk][6][64] int layout, conflict-free+compact.
- Saves 8KB LDS at BN=128 (-25% ds_write/read bytes). LDS: 24KB vs 32KB.
- FULLapx column in A2 table IS XCMPB+APIPE (v41). Confirmed +10% vs old XPOSE+APIPE
  (Phase2 had 1868, A2 FULLapx=2055 on square = XCMPB improvement).
- verify_xcmpb.sh (load~4): square +1.5%, imgQKV +0.3%, FC2 +1.7% vs XPOSE+APIPE.
- FULLapx vs W6A6: beats W6A6 on square(2055>1890), LIN2(1918>1688), ties FC2(1823≈1828).
  Loses on gate_up(1693<1765), imgQKV(1516<1662), imgPROJ(1443<1560), LIN1(1745≈1747).

WAVEPIPE (A3) — SHELVED (CDNA4 deadlock):
- CDNA4 does not guarantee interleaved scheduling of waves within a CTA.
- Consumer waves spin-polling sFlag starve producers -> hung indefinitely on gfx950.
- Confirmed by measurement: wavepipe binary ran 40+ min on square with no output.
- Kept behind #ifdef WAVEPIPE in v41 for documentation only.

VPERM (B1) — SHELVED (37000x slower):
- Measured 0.04 TFLOPS vs 2055 baseline on loaded node (not just load: ratio holds).
- Root cause: d_vperm_lut indexed by per-lane VGPR scale value forces per-lane
  constant-memory reads on CDNA. Constant memory is fast ONLY for uniform SGPR indices.
  With per-lane scales, each lane needs a different word -> scalar cache thrash / vector
  memory emulation -> catastrophic slowdown.
- LDS LUT (sLUT[2048]) is the correct approach for this non-uniform per-lane pattern.
  No further work on VPERM; the LUT LDS path is already near-optimal.

KTILE2+XCMPB (new lever): 2 K-tiles per barrier, compact+conflict-free layout.
- LDS: 53KB at BN=128 (2*2*NB*6*64*4 = 49KB + LUT 4KB). Halves barrier count.
- KTILE2 was flat at BN=64 (barrier is small there), but at BN=128 the barrier+ds_write
  gap is 784 TFLOPS. Halving it with KTILE2 could add ~200-400 TFLOPS.
- Built and compiles clean. Not yet measured (node too loaded).

=====================================================================
PHASE 3 FINAL RESULTS (session 3, 2026-07-15)
=====================================================================
bench_full_phase3.txt: fresh container, 80 iters, 10 reps.
v41 dispatch: APX=XCMPB G4 (512x128), APX2=XCMPB G2, max(pre/pre4/apx/apx2).

    shape       tc_HIP     W6A6    vs_W6A6   W4A6    vs_W4A6
    square       2149       1901    +13.1%   2863    -24.9%  ✅ beats W6A6
    gate_up      1721       1769    -2.7%    2220    -22.5%
    imgQKV       1523       1659    -8.2%    2134    -28.7%
    imgPROJ      1494       1562    -4.4%    1927    -22.4%
    FC1          1909       1753    +9.0%    2282    -16.3%  ✅ beats W6A6
    FC2          1889       1833    +3.1%    2243    -15.7%  ✅ beats W6A6
    LIN1         1778       1742    +2.1%    2311    -23.1%  ✅ beats W6A6
    LIN2         1956       1689    +15.8%   2222    -12.0%  ✅ beats W6A6

Beats W6A6 on 5/7 compute shapes (up from 2/7 in phase 2).
vs Phase 2 tc_HIP: +5-12% on all shapes.

Levers tried and outcome:
- XCMPB: ✅ DELIVERED (+7-12% vs old XPOSE, in dispatch)
- WAVEPIPE: ❌ SHELVED (CDNA4 intra-CTA spin-lock deadlock)
- VPERM: ❌ SHELVED (non-uniform VGPR constant-mem index = 37000x slower)
- KTILE2+XCMPB: ❌ SHELVED (2x pack per barrier = 60-65% of XCMPB+APIPE, slower)

Next lever candidates for phase 4:
- gate_up/imgQKV/imgPROJ remain below W6A6 by 3-9%. These shapes have thin margins
  in the native W6A6 ceiling itself. Further gains require improving the GEMM loop
  efficiency for huge-N shapes (A-refetch dominates at large N).
- Possible: A-in-LDS staging for huge-N (cooperative A load once, reuse across
  N-tiles). High complexity; needs fresh investigation.
- Decode shapes (M=128) already win W4A6 by 24-50% — no further work needed there.

New scripts: phase3_nopack128.sh, phase3_xcmpb.sh, verify_xcmpb.sh, phase3_run.sh
New kernel:  kernel/w4a6_tc_gemm_v41.hip (branched from v31, all old paths intact)

=====================================================================
PHASE 4 (session 3 cont, 2026-07-15/16) — deep pipeline + SCHED + W4A6 re-examination
=====================================================================
Goal: beat W6A6 AND W4A6 everywhere. Studied 3 refs (CDNA4 ISA, HIP perf guide,
low-latency GEMM blog). Branched v42 from v41.

DECISIVE probe (phase4_probe.sh, quiet node load 1.5): NOPACK > W4A6 on ALL shapes
  square NOPACK=3183 (W4A6=2863), gate_up 2489 (2220), imgQKV 2266 (2134),
  imgPROJ 2268 (1927), LIN2 3083 (2222). The GEMM skeleton beats W4A6 everywhere;
  the entire gap is pack tax = ds_write(~26%) + LUT VALU(~12%).

Levers tried in v42 (all bit-exact, kept behind #ifdef for trace):
- SCHED (sched_group_barrier VALU/MFMA co-exec): ✅ +1-1.5%, INTEGRATED into dispatch.
    gate_up 1705->1725, imgQKV 1511->1535, FC2 1839->1865, LIN1 1785->1804.
    This is the blog's key lever (instruction-level pipe overlap). SCHED_VALU=4 default.
- STAGE3 (NSTAGE-deep register-ring pipeline): ❌ -16% (register ring bloats VGPR past
    the 128-VGPR-acc occupancy floor).
- GLDS (global_load_lds async HBM->LDS): ❌ -9% (double LDS traffic: raw->LDS then
    LDS-raw->transcode->LDS-fp6 costs more than HBM->reg->transcode->LDS). Intrinsic
    verified working per-lane 16B on gfx950.
- WM=16 (BM=1024 to halve pack redundancy): ❌ collapses ~180 TF (16 waves exceed SIMD
    register capacity regardless of BM).
- VPERM2 (LDS 64-entry table + v_perm_b32): ❌ -13% (16 v_perm+sign-fold VALU > LUT
    shift-pack; LDS-LUT reads were never the bottleneck).
- APXILV (interleave packto into mi loop): ❌ disrupts A-prefetch ILP.
- Tile sweep: 512x128 confirmed optimal (768x128 collapses, 256x128 too small,
    1024x64 XCMPB slightly below). GROUP flat (rasterization exhausted).
- Split-K on imgPROJ (thought CTA-starved at 192 CTAs): ❌ worse (atomic C cost).

ARCHITECTURAL CONCLUSION — beating W4A6 is impossible for the fused single-pass design:
  - occupancy pinned at 1 CTA/CU: acc[MPW][NPW]=128 VGPR floor (output_tile*4/threads);
    measured VGPR=252/occ=2waves/spill=0. Can't get 2 CTA/CU without a smaller tile,
    which raises pack redundancy (tax~1/BM) more than the occupancy gain helps.
  - the pack tax can only be hidden by a 2nd CTA (blocked) — SCHED gets the intra-CTA
    overlap but the barrier stall of the shared-B cooperative pack remains.
  - fused constraint forces per-CTA B re-transcode (B stored fp4 in DRAM); pre-transcode
    to fp6 DRAM = just W6A6. So huge-N re-transcode redundancy is irreducible.
  - imgQKV: MY native fp6 (1646) already ~ties FlyDSL W6A6 (1659), so even zero-tax
    transcode can't beat W6A6 there — the GEMM itself is at the fp6 delivery cap.

FINAL v42 result (bench_full_v42.txt): beats W6A6 on 5/7 compute shapes.
  square +14.6%, FC1 +8.7%, FC2 +5.5%, LIN1 +2.3%, LIN2 +16%.
  gate_up -1.1%, imgQKV -7%, imgPROJ -4.1% (structural fp6-delivery cap).
  Decode (M=128): tc-Fly beats W4A6 by +24-50% (unchanged).

Git commits: fc70cf2 (baseline) ... a558c65(STAGE3) 048c4a8(GLDS) 9800dc3(SCHED)
  0b0aa0c(VPERM2) 002ec0f(v42 bench). v42 = production (XCMPB+SCHED 512x128).

=====================================================================
PHASES 5-9 — post-v42 progression (2026-07-15 .. 2026-07-22)
=====================================================================
All levers below live behind #ifdef in kernel/w4a6_tc_gemm_v42.hip (no separate
file); all bit-exact (maxabs=0 nbad=0). Full analysis: kernel/phase4_breakdown.md.

PHASE 5 — PIPE (deep LDS ring) ✅ KEPT
  CDNA4 LDS is 160 KB/block (NOT 64 KB as assumed) -> an NSTG=3 fp6-B ring (36 KB)
  is affordable. PIPE + s_setprio (prioritize MFMA) + sched_barrier overlap: +2..23%.

PHASE 7c — bf16 output (OUTBF16) ✅ KEPT — the cheapest big win
  bf16 C halves the C-write HBM traffic (C ~27% of decode traffic; large fraction on
  huge-N). Flips ALL 8 compute shapes over BOTH W6A6 and FlyDSL — the last fp32
  holdouts (imgQKV/imgPROJ, -4%/-2% in fp32) now win. fp32 accumulate stays bit-exact.

PHASE 8 — CDNA4 ISA MFMA scan (read the 608pp ISA) — NO kernel change
  - non-scaled v_mfma_f32_16x16x128_f8f6f4: no gfx950 clang builtin, identical cycle
    count, identity-scale already used -> inline asm = zero-cycle gain. Not pursued.
  - v_mfma_scale_f32_32x32x64_f8f6f4: 6943 vs 8309 TFLOP/s (16x16x128 faster under
    dep); same MAC/cycle + VGPR density -> big rewrite, zero gain. Not pursued.
  Verdict: v_mfma_scale_f32_16x16x128_f8f6f4 is the optimal MFMA for this kernel.

PHASE 8b — HW scale-convert (v_cvt_scalef32_pk32_fp6_f16 ...) — BLOCKED by contract
  Emits the 6-dword fp6 fragment directly (would replace LUT+bitpack) BUT does real
  float arithmetic -> mismatches the reference "revised" LUT for any non-zero E4M3
  scale mantissa (28/32 codes differ). Arithmetically valid, different numbers ->
  not a drop-in under bit-exact. Documented as future option if contract relaxes.

PHASE 9 — Primus-Turbo PR410/424 techniques (2026-07-22)
  Studied AMD-AGI/Primus-Turbo PR#410 (grouped MXFP8) / PR#424 (MXFP4) FlyDSL backends.
  Implemented XCD-aware CTA->tile swizzle (-DXCDSW): MI355 dispatches CTAs round-robin
  over 8 XCDs (verified via HW_REG_XCC_ID); remap a contiguous run onto one XCD's L2.
  Bit-exact; measured +0.3-0.8% (G2) / flat-to-2% (G4) = node-load noise. Added as a
  safe HAPX4 max() dispatch candidate. Marginal because the FUSED per-CTA B-transcode
  reads B fresh from HBM (no cross-CTA L2 reuse to exploit). Other PR techniques
  (separate scale preshuffle, dual-quant, int64 SRD) don't map to the fused design.

FINAL STANDING (best-of, bf16 compute): HIP wins 9/12 vs FlyDSL (8 compute +22-77%,
DOWN decode when uncontended). 3 memory-bound M<=256 decode shapes (GATE/QKV/M256
text) remain FlyDSL's (occupancy starvation; needs signal-based split-K). Ship
dispatch = max(tc_HIP, tc_Fly) per shape. v42.hip carries all levers behind #ifdef;
v44 MFMANS + XCDSW are documented experimental paths, production build unchanged.

=====================================================================
SESSION 14 (2026-07-28) — defect fixes + the three uncracked surfaces
=====================================================================
Full trace with hypotheses, ISA evidence and rollback tags: kernel/opt_trace_s14.md.
Guardrails this session: git tag per lever, every lever behind its own #ifdef, the
default build path untouched, and — new — a **correctness gate that runs before any
timing is accepted**, so no unverified number can win a sweep.

TWO DEFECTS FIXED (these were wrong results, not tuning gaps)
- OUTBF16 silently disabled split-K: the bf16 epilogue dropped the accumulate branch,
  so splits overwrote each other, and the S>1 zero-fill was sized in float against a
  bf16 buffer (an M*N*2 byte OOB write). Fix: split-K accumulates into an fp32
  workspace and a cvt_acc_out pass converts once at the end. For fp32 output that
  workspace IS C, so the pre-s14 path is byte-identical and costs nothing.
  DOWN decode 94.98 -> 185.21 (+95%) on the affected config.
- `bench` mode never verified — it returned before the gold compare while `hbest`
  swept S=1..16 and took the max, so every bf16 decode number with S>1 was untrusted.
  Fix: bench now runs a split-K self-consistency pass and prints REFUSED instead of
  a time when it fails.

FINAL 12-SHAPE RESULT (bench_full.sh, card 5 @2% use, best-of-3, all 17 binaries gated)

    shape              baseline   final    delta
    LLM square          2245.78  2472.93   +10.1%
    LLM gate_up         1858.85  2042.55    +9.9%
    LLM DOWN decode      340.77   342.56    +0.5%
    LLM GATE decode      609.55   715.11   +17.3%
    LLM QKV decode       277.64   278.87    +0.4%
    FLUX imgQKV         1584.00  1781.11   +12.4%
    FLUX imgPROJ        1553.40  1745.23   +12.4%
    FLUX FC1            1965.40  2154.15    +9.6%
    FLUX FC2            2002.31  2353.69   +17.5%
    FLUX LIN1           1894.11  1948.45    +2.9%
    FLUX LIN2           2034.35  2352.13   +15.6%
    FLUX txt FC1 M256        --   668.77   (gate was failing at baseline)

Geomean +9.7% over the 11 comparable shapes; nothing regressed.

THE TWO WINNING LEVERS (both now dispatch candidates)
- HWCVT ✅ (phase 5, the largest single win) — `__builtin_amdgcn_cvt_scalef32_pk32_fp6_f16`
  emits the 6-dword fp6 fragment directly and deletes the entire LUT + bit-pack stage.
  +5-9% compute, +10-16% decode. **This reverses the PHASE 8b verdict above.** That
  verdict was correct on its own terms — HWCVT is not bit-exact against the revised LUT —
  but bit-exactness against the LUT was never the
  actual requirement; accuracy against fp32 is. Measured rel-L2 vs an fp32 reference:
  ratio 0.998-1.004 vs the LUT on three shapes, i.e. indistinguishable, and better on
  two of three. Essentially all the error in both paths is fp6 quantization itself.
  Enumerating all 64 (scale-mantissa x |E2M1|) pairs on hardware (s14_hwcvt_trace.sh)
  shows the disagreement is only **6/64, and every one an exact tie** — the hardware
  rounds half-to-even, the table's argmin rounds half-down, so neither is more accurate.
  This corrects the "28/32 codes differ" figure in PHASE 8b above: that probe placed the
  E4M3 mantissa on the convert's scale operand, where only the exponent is honoured, so
  it measured a broken variant rather than the shipped flow.
  Gated on rel-L2 ratio <= 1.05 rather than on a bit-compare it fails by construction.
  The LUT path remains the default and fully working.
- BARAMORT+PRIOPK ✅ (phase 3) — with the NSTG-deep ring, a stage written at tile t is
  not consumed until t+NSTG-1, so one barrier per ring rotation suffices instead of one
  per k-tile; PRIOPK raises s_setprio around the pack. +2-3% on top of HWCVT.

NEGATIVE RESULTS WITH ROOT CAUSE (do not re-run these)
- MFMANS (phase 2, non-scaled MFMA inline asm) ❌ -1..-2% across all 6 combos. The scale
  operand is not on the critical path; saving 2 VGPR/operand buys nothing at 1 CTA/CU
  where the 128-VGPR accumulator already pins occupancy, and the asm blocks scheduling.
  This closes the "free lever never benchmarked" item — it was never free.
- APFK (phase 4, A-prefetch a whole tile ahead) ❌ -36%. Pushed VGPRs past 256 and the
  kernel spilled to scratch; confirmed in the ISA dump, not inferred.
- PIPEGLDS (phase 4, async raw-B global_load_lds combined with the PIPE ring) ❌ -8..-11%.
  gfx950 counts A loads and raw-B global_load_lds in a single vmcnt, so a partial wait
  cannot separate them; the required s_waitcnt(0) drains the A prefetch too and
  serialises precisely what the ring exists to overlap. Architectural, not tunable.
- SIGSPLIT (phase 6, signal-based split-K) ❌ -23..-29% vs atomic on DOWN, the only shape
  that wants split-K at all. Correct (maxrel=0) but slower. Three cuts were needed to
  get there: __threadfence() lowered to buffer_wbl2+buffer_inv (6x slowdown — every CTA
  invalidating the caches destroyed the B stream in L2); agent-scope atomics were no
  better and additionally tagged stores sc1, bypassing L2 to HBM; the final version uses
  plain stores + s_waitcnt(0) + a relaxed counter + `buffer_inv sc0` (L1 only). Even
  then it loses, because **CDNA4's L2 has native fp32 atomic-add units** — an atomicAdd
  is ~1 L2 transaction per element while SIGSPLIT costs S stores + S loads + a store
  plus an exposed serial reduction tail. On this hardware the atomic IS the cheap path.
  This refutes the phase's premise: "atomics killed split-K (split4 -> 60 TF)" was the
  OUTBF16 defect above, not the atomics. With it fixed, atomic split-K gives DOWN 298-336.

THE 619 vs 1524 GATE GAP — correctly diagnosed at last
Split-K cannot close it: GATE's optimum is S=1, so no reduction ever runs. GATE is
128x19456x2560, K=2560 = only 20 k-tiles; splitting an already-short K leaves each CTA
10 or 5 tiles while the fixed prologue/epilogue stay constant. The gap is **wave
quantization plus short K** — 608 output tiles over 256 CUs at 1 CTA/CU is 2.4 waves, so
the last wave runs 38% populated. Closing it needs a persistent or stream-K
decomposition (different kernel structure). What did move it: HWCVT + BARAMORT+PRIOPK,
610 -> 715 (+17%).

New flags: -DBARAMORT -DPKAOV=n -DBARROV=n -DPRIOPK -DSCHEDGRP -DPIPEGLDS -DAPFK
           -DHWCVT -DSIGSPLIT   (all #ifdef; default build path unchanged)
New scripts: s14_p34_bench.sh, s14_p5_acc.sh, s14_p5b_bench.sh, s14_p6_bench.sh,
             s14_p6b_bench.sh, s14_p46_bench.sh, s14_probe_builtins.sh, s14_diag2.sh,
             s14_hwprobe.sh, s14_sig_isa.sh, s14_supervise.sh, s14_host_launch.sh

=====================================================================
SESSION 15 (2026-07-29) - porting HWCVT to the FlyDSL transcode kernel
=====================================================================
Full trace: kernel/opt_trace_s15.md.

Premise: session 14's HWCVT (hardware scale-converts instead of the 12-bit LUT) beat
the LUT in the native HIP kernel, so try the same substitution in the FlyDSL kernel
(psg_preshuffle_gemm.py, in_dtype="w6a6_tc") to lift the three decode shapes FlyDSL owns.

RESULT: ADOPT. On the production front (use_async_copy=True) HWCVT wins on all three
dispatch-owned decode shapes and ~+9% on compute (warm=25 iter=200, 3 interleaved reps):

  shape (tile)                          LUT         HWCVT       delta
  GATE decode 128x19456x2560 (128x64)   1730-1732   1778-1785   +2.7 / +3.0 / +2.9%
  QKV decode  128x2560x9728  (128x64)   380         383-384     +1.0 / +0.9 / +0.9%
  M256 text   256x2560x9728  (128x64)   685         690-692     +1.1 / +0.9 / +0.8%
  SQUARE   4096x4096x8192    (256x256)  1858-1867   2027-2039   +9.1 / +8.6 / +9.2%
  GATE prefill 4096x19456x2560(256x256) 1654        1812        +9.6% (1 rep)

*** CORRECTION: this entry first recorded the OPPOSITE verdict (-22..-32%, "dead end").
Cause: the first benchmarks used bench_sweep.py's DEFAULT flags, which leave
use_async_copy OFF, while session 13 (opt_trace_s13 E1) had already established
use_async_copy=True as the production w6a6_tc front (bit-exact, +4..52% memory-bound).
The superseded verdict is in commits b1cc3ea and a4a6c84. Without async HWCVT really
does lose 22-34%; async does not scale this lever, it INVERTS it. ***

Correctness gate: at b_scale=2.0 the E4M3 scale mantissa is 0, so bump=0 and HWCVT must
match the LUT BIT-EXACTLY. All four combinations of {LUT,HWCVT} x {sync,async} pass with
identical outputs on both a compute tile and the GATE decode tile. Note bench_sweep.py
verifies NOTHING (it fills the LUT with arbitrary values), and at low warmup/iters it
measures launch overhead - the same LUT config read 286 TFLOPS at iters=1 vs 1493 at
iters=50.

WHY IT WINS HERE, AND WHY ONLY WITH ASYNC
The LDS reads are irrelevant in BOTH configs, which is the surprise. TC_PROBE_NOLUT=1
deletes all 16 ds_read/fragment and is worth -0.4..+0.1% (sync AND async, decode AND
square). So FlyDSL's win is NOT the HIP win: in HIP the gain was LDS-port relief
(ds_read 104->24, -77%, on a kernel limited by operand delivery, +5-9%/+10-16% while
ADDING 160 VALU). Here the reads cost nothing, so the gain is the other thing HWCVT
deletes - the ~30 shift/OR that bit-pack 16 non-byte-aligned 12-bit fields into a dense
6-dword fragment, which pk32_fp6_f16 emits directly (VALU only: 33 ops vs ~46).
The sign flip is scheduling slack: without async, A is staged global->register->LDS and
those loads/stores occupy vmem and VALU issue slots, so the transcode runs in their
shadow. The LUT path's ~46 ops are wide and INDEPENDENT and fill that shadow well;
HWCVT's 33 are a DEPENDENCY CHAIN (16 decodes, then 16 multiplies, all retiring before
the single 32-input pk32 can issue), so it cannot use idle slots. async hands A's copy
to the DMA engine, the shadow disappears, the transcode becomes the critical path, and
the shorter chain wins - hence compute (least slack) gains most, +9%.
Not directly measured: pk32 latency, issue-slot occupancy, realised VGPR counts
(rocprofv3 killed the container twice; FlyDSL's JIT hides its code object from
GPU_DUMP_CODE_OBJECT).

HEADROOM FOR ANY FURTHER TRANSCODE LEVER (s15_headroom.sh, async front)
Bracketed against the no-transcode baselines; fp6 runs the same fp6xfp6 MFMA with B
already MXFP6, so (fp6 - w6a6_tc) bounds what any transcode change can ever return:
  GATE decode: tc_LUT 1726 | tc_HWCVT 1775 | fp6 728 (-58%) | w4a6 1777
  SQUARE:      tc_LUT 1874 | tc_HWCVT 2031 | fp6 2331 (+24%) | w4a6 2774
- DECODE IS DONE: fp6 is 2.4x SLOWER than transcoding (fp4 B is half the bytes on a
  memory-bound shape). No transcode headroom remains on decode; +2.9% is near the end.
- COMPUTE has ~24% headroom, of which HWCVT now takes 9%. The remaining ~15% is the one
  place a further transcode lever could pay - but FlyDSL compute is still under HIP's
  HWCVT square (~2365), so it changes no dispatch decision today.

ALSO SETTLED: no wider convert exists on gfx950. pk8_fp4_f16 / pk16_fp6_f16 would cut
decode instructions 4x but are gfx1250-only ("needs target feature gfx1250-insts").
gfx950 has only the 2-wide pk_f16_fp4 decode and the 32-wide pk32_fp6_f16 packer -
exactly what both kernels already use. So this lever is closed for the HIP kernel too.

LESSONS: (1) benchmark the SHIPPED configuration, not the tool's defaults; (2) a lever's
sign can depend on another lever, so interacting levers must be evaluated in the
combination that ships; (3) the same change can win for different reasons in different
kernels - attributing a win to the wrong resource is how you mispredict the next port.

New flag: TC_HWCVT=1 (FlyDSL)
New scripts: s15_async_ab.sh, s15_gate_async.sh, s15_headroom.sh, s15_why.sh,
             s15_probe_pk8.sh, s15_build.sh, s15_bench2.sh, s15_wpe.sh, s15_bench.sh

=====================================================================
SESSION 15b (2026-07-29) - the dispatch harness was measuring the two
                           kernels unequally; DOWN decode mis-assigned
=====================================================================
Full trace: kernel/opt_trace_s15.md ("The bigger find" section).

Chasing why the async front was missing from the s15 benchmarks led to bench_full.sh, the
harness whose table decides per shape whether HIP or FlyDSL ships (policy: max(tc_HIP,
tc_Fly)). It measures the two sides on very different terms:
  HIP side    hbest/best_of: split-K in {1,2,4,8,16} x HREPS=3 repeats, MAX taken, then
              maxof across up to 9 binaries, 300 internal iterations.
  FlyDSL side ONE bench_sweep.py call at WARM=15 ITER=80, with use_async_copy NEVER passed.

FlyDSL was therefore understated on every memory-bound row (s15_harness_bias.sh):
  row            harness  +async  +steady  shipped   understated
  DOWN decode      307.7   376.7    310.3    380.0      -19%
  QKV decode       427.9   509.1    435.6    519.3      -18%
  GATE decode     1514.5  1709.3   1557.6   1745.6      -13%
  M256 text       1787.5  1961.0   1824.0   1995.6      -10%

Since the policy is a max over the two columns, this MIS-ASSIGNED A SHAPE. Head-to-head in
one session on one card, HIP given its full best-of advantage (s15_head2head.sh):
  row            HIP best (recorded)   FlyDSL   winner   margin
  DOWN decode      340.1  (342.6)       384.1   FlyDSL   +12.9%   <- was shipping HIP
  QKV decode       279.6  (278.0)       530.8   FlyDSL   +89.8%
  GATE decode      710.3  (720.2)      1791.8   FlyDSL   +152%
HIP reproduces its recorded numbers to within 1.5%, so the harness's HIP side is sound and
only the FlyDSL side was wrong. GATE/QKV were already FlyDSL's by too wide a margin to flip.
DOWN decode was assigned to HIP on 247.9 vs 342.6 when FlyDSL actually does 384.1 - a 12.9%
loss on that row, purely from measurement.

FIX APPLIED to bench_full.sh
- fly() now offers BOTH async and non-async and takes the max, mirroring the HIP side's max
  over binaries. Both are needed: the lever is regime-specific (+4..52% memory-bound, -9% on
  some compute shapes), so neither setting can be hardcoded.
- new fly_iters() scales iterations with shape so every sample times >=25ms of steady state
  (2*M*N*K / 300 TFLOP/s, clamped to [ITER,2000]) instead of a flat 80.
- row() now APPLIES the shipped max(tc_HIP,tc_Fly) policy and names the winner. It previously
  hardcoded best="$h"; who="HIP" and printed FlyDSL as an unselected reference column - which
  is how this stayed invisible: a row could show a faster FlyDSL number and still print (HIP).
Verified by executing the patched functions themselves (s15_harness_verify.sh extracts them
from the file with sed rather than re-implementing them):
  LLM_DOWN_decode  iters=1177   380.9  (was 247.9)  HIP 340.1 -> FlyDSL wins
  LLM_QKV_decode   iters=1747   521.1  (was 427.9)  HIP 279.6 -> FlyDSL wins
  LLM_GATE_decode  iters= 589  1751.6  (was 1514.5) HIP 710.3 -> FlyDSL wins

CANONICAL-SHAPE CORRECTION: s15 also mislabelled two shapes. 128x2560x9728 is DOWN decode,
not QKV; the M256 row is 256x12288x3072, not 256x2560x9728. So the real QKV decode
(128x4096x4096) and the real M256 row were never measured. Re-measured at steady-state
iteration counts, with a LUT sample before AND after each HWCVT sample so drift cancels
(s15_canon4.sh, all reps kept, drift <=0.4%):
  QKV decode  128x4096x4096    LUT  519.3 -> HWCVT  530.5   +2.1%
  M256 text   256x12288x3072   LUT 1997.2 -> HWCVT 2018.2   +1.0%
  DOWN decode 128x2560x9728    LUT  380.1 -> HWCVT  383.9   +1.0%
  GATE decode 128x19456x2560   LUT 1742.6 -> HWCVT 1791.0   +2.8%
HWCVT wins on every canonical shape measured.

STILL OPEN: making TC_HWCVT=1 the FlyDSL default. HWCVT rounds differently from the LUT once
the scale mantissa is non-zero, so a bit-compare rejects it by construction and says nothing
about quality. HIP handles this with gate_acc (rel-L2 vs the exact fp32 dequantized product
within 5% of the LUT's own; measured 1.002). No equivalent accuracy harness exists on the
FlyDSL path - verify_w4a6.py only does bit-exact cross-checks. Building it is the prerequisite
to shipping the +1..9%.

LESSONS: (4) a harness that picks a winner must measure both candidates on equal terms - one
side got best-of-3 x best-of-split-K x best-of-9-binaries and the other a single
under-iterated run with the production flag missing; that is a handicap, not a benchmark.
(5) never let a comparison table hardcode its own conclusion; if a harness implements a
policy it must EVALUATE the policy. (6) iteration counts must scale with kernel size and the
failure is NOT monotonic - the same QKV config read 427.9 at 80 iters, 285..526 at 200, and
519.6 +-0.1% at 800, so a tight-looking spread at low iters proves nothing.

New scripts: s15_harness_bias.sh, s15_head2head.sh, s15_harness_verify.sh, s15_canon4.sh,
             s15_stab.sh, s15_canon.sh, s15_canon2.sh, s15_canon3.sh (last three superseded)

=====================================================================
SESSION 15c (2026-07-29) - accuracy gate for the FlyDSL transcode, and
                           HWCVT wired in as a per-shape parameter
=====================================================================
Full trace: kernel/opt_trace_s15.md ("Shipping it" section).

HWCVT rounds in hardware while the LUT rounds through a table built offline, so once the E4M3
scale mantissa is non-zero a bit-compare rejects HWCVT by construction and says nothing about
quality. The HIP side settles this with gate_acc (rel-L2 vs the exact fp32 dequantised product
within 5% of the LUT's own; measured 1.002). The FlyDSL path had no equivalent - verify_w4a6.py
only does bit-exact cross-checks - so one was built: kernel/fly_acc_gate.py + s15_accgate.sh.

METHOD: verify_w4a6.py keeps its data uniform precisely so it never has to replicate the
kernel's A-swizzle or B-preshuffle. The transcode is an element-wise function of (fp4 code,
E4M3 scale byte), so sweep every such pair with uniform tensors and read one output element.
192 samples over 8 codes x 8 mantissas x 3 exponents.

THE FIRST VERSION PASSED FOR THE WRONG REASON, which is why it is recorded. Against the
analytic target it reported rel-L2 = 0.46 (worst element 56% off) and warned that uniform
input was producing NON-UNIFORM output. That is this harness inheriting the known host-side
fp6-A layout gap (REPRODUCE.md): some k-terms drop out, so every measurement carries an
unknown common factor. Being common-mode across both paths it left the ratio looking healthy
(0.9945) while inflating the denominator enough to mask any real difference.
FIX: at scale-mantissa 0 the transcode is provably EXACT (bump=0, and every MXFP4 value
0.5..6 is exactly representable in E2M3), so that measurement pins the unknown factor:
    exact(c, exp, m) = measured(c, exp, m=0) * (1 + m/8)
independent of A's layout, of accumulation order, and of how many k-terms survive. m=0 carries
no information and is excluded from the metric.

RESULT (147 informative samples):
  rel-L2 of the mantissa fold   LUT   0.025581
  rel-L2 of the mantissa fold   HWCVT 0.025898
  ratio HWCVT/LUT               1.0124   (gate <= 1.05)  -> PASS
  the 24 mantissa-0 samples agree EXACTLY between the paths, as they must
~2.6% is the expected cost of folding a 3-bit scale mantissa into 3-bit E2M3 data; the worst
case (code 1 = 0.5, mantissa 2, 20% off) is identical on both paths - E2M3 resolution, not a
transcode defect.

WIRING: selection is now a compile parameter, not a flipped global default -
compile_preshuffle_gemm_a8(..., use_hw_transcode=None|True|False), exposed as
bench_sweep.py --hw_transcode on|off. None keeps the TC_HWCVT env default, which matters
because ~15 scripts in this trace use `unset TC_HWCVT` as their LUT arm; flipping the global
default would silently have turned those A/B comparisons into HWCVT-vs-HWCVT. bench_full.sh's
fly() now offers the transcode path as a w6a6_tc candidate and takes the max - the same
treatment the HIP side gives its own HWCVT binary.
Verified (s15_param_verify.sh, GATE decode, async on):
  no flag, TC_HWCVT unset (= LUT)      1746.6
  --hw_transcode off                   1744.9
  --hw_transcode on                    1796.4
  TC_HWCVT=1, no flag (= HWCVT)        1794.5
  TC_HWCVT=1 but --hw_transcode off    1744.2   <- flag overrides env, both directions

STILL OPEN: the library's DEFAULT transcode is still the LUT. Turning it over is now only a
decision, not a missing measurement.

LESSON: a gate that passes for the wrong reason is worse than one that fails. This one
reported a healthy ratio while a 46% common-mode error sat in the denominator masking the
signal. Sanity-check a gate's ABSOLUTE numbers against what the physics should be, not just
the pass/fail it prints - 46% was never plausible for a 3-bit mantissa fold.

New files: kernel/fly_acc_gate.py, kernel/s15_accgate.sh, kernel/s15_param_verify.sh

================================================================================
SESSION 16 - the last transcode arithmetic lever is closed; two more measurement
             defects found and fixed (one in the harness, one in the file sync)
================================================================================
FULL TRACE: kernel/opt_trace_s16.md

LEVER CLOSED (measured, not assumed): can the E4M3 scale mantissa (1 + mb/8) ride the
convert's fp32 scale operand instead of costing 16 v_pk_mul_f16 per 32-code fragment?
That would take the HWCVT transcode from 33 VALU ops to 17, and in FlyDSL all 32 codes of a
fragment share ONE scale byte, so a single operand on the pk32 pack could absorb all 16.
kernel/s16_scaleop.hip runs both converts with scales 1.0 / 1.5 / 1.875 - same exponent,
different mantissa:
  cvt_scalef32_pk_f16_fp4     byte-identical output for all three  -> EXPONENT-ONLY
  cvt_scalef32_pk32_fp6_f16   byte-identical output for all three  -> EXPONENT-ONLY
"scalef32" names the operand's TYPE, not how much of it the hardware uses. Both kernels had
assumed this correctly (v42 multiplies separately, FlyDSL's comment says so) but neither had
measured it. The 16 multiplies are irreducible: 33 ops is the floor for the hardware path.
The experimental TC_HWSCALE flag was reverted rather than left as dead code.

DEFECT 1 - a stale workspace duplicate was pushed over the working kernel, and every
downstream check still passed. cff6749 made the repo-root copies canonical, but the workspace
still had kernel/psg_preshuffle_gemm.py. Edits went to the duplicate while the sync pushed the
root copy: 18 days old, 2489 lines vs 2580, zero occurrences of _TC_HWCVT. Consequences:
  - fly_acc_gate.py asserts TC_HWCVT is set in the ENVIRONMENT, not that the kernel honours
    it, so --mode hwcvt silently measured the LUT;
  - both arms then agreed to the last digit (0.025581), which reads as a perfect pass;
  - bench_sweep.py --hw_transcode on raised TypeError on the old signature, was swallowed
    per-tile, returned empty, and maxof read it as 0 - the candidate vanished from dispatch.
Restored from git HEAD, the gate reproduced s15c exactly: LUT 0.025581, HWCVT 0.025898,
ratio 1.0124, 18/192 samples differing. s15's conclusions are unaffected.
FIX: kernel/sy.sh now REFUSES to push psg_preshuffle_gemm.py if it lacks _TC_HWCVT or
use_hw_transcode, or if it shrank >50 lines vs the live file. Workspace duplicate deleted.

DEFECT 2 - the harness was STILL biased against FlyDSL, but only under contention. s15b
equalized flags, iteration counts, candidates and the dispatch policy; it did not equalize
repeat count. HIP's hbest takes max over 3 reps x 5 split-K x up to 9 binaries (45 samples);
fly() took ONE timed loop per variant. Mid-audit an external job took all 8 cards to 100%:
  GATE decode FlyDSL w6a6_tc   recorded 1745-1796   this run 508.8   (5.8x low)
  GATE decode tc_HIP           recorded 720.18      this run 719.52  (within 0.1%)
Best-of-N finds a quiet moment; a single sample takes what it is given. A shared node therefore
biases the table toward whichever side is sampled more - regardless of the kernels. Those rows
were discarded.
FIX: bench_sweep.py --reps N repeats the timed loop IN-PROCESS and keeps the best, so extra
samples cost no startup and no JIT recompile; bench_full.sh drives it with FLYREPS=3 = HREPS.

KILL-RESILIENCE: the sweep now runs for hours and this container was SIGKILLed 4 times this
session. bench_full.sh appends each completed row to $RESULTS and replays cached rows on
rerun; HIP binaries are skipped when newer than both kernel sources (FORCE_BUILD=1 overrides);
kernel/s16_audit.sh re-invokes until the table is full. Two kills absorbed, no rows lost.

LESSONS: an instruction's operand TYPE is not its semantics - measure it. A flag whose failure
mode is silent cannot be A/B'd: TC_HWSCALE returned identical numbers because it was never
compiled in, which is indistinguishable from "the change is exact" (and there was even a
correct-sounding reason why it would be exact). Prove the variant is IN EFFECT before comparing.

New files: kernel/opt_trace_s16.md, kernel/s16_scaleop.hip, kernel/s16_scaleop.sh,
kernel/s16_cachecheck.sh, kernel/s16_pathcheck.sh, kernel/s16_recover.sh,
kernel/s16_restore.sh, kernel/s16_hwarm.sh, kernel/s16_audit.sh, kernel/s16_hwscale.sh,
kernel/sy.sh
DEFECT 3 (found while collecting the audit table) - the retry loop poisoned its own data.
With the kernel restored and the harness equalized, the first four rows STILL read wrong, but
in a specific pattern: both kernels roughly halved on the latency-bound rows while the
compute-bound row was within 2% of its record.
  GATE decode tc_fly  1745-1796 recorded -> 845    (/2.08)
  GATE decode tc_HIP  720.2     recorded -> 417.2  (/1.73)
  DOWN decode tc_fly  384       recorded -> 177.8  (/2.16)
  LLM square  tc_fly  1692      recorded -> 1666.5 (/1.02)
A depression that hits BOTH implementations is not a kernel property, so the card was the first
suspect. kernel/s16_cardhealth.sh ran one known config on all eight: 1760.5 / 1764.7 / 1749.6 /
1749.7 / 1764.0 / 1762.0 / 1768.4 / 1751.6 - every card healthy, including the one in use.
ROOT CAUSE: killing a docker exec client does NOT kill the process inside the container. Every
time the host driver died or was killed, its in-container bench_full kept running, and the next
attempt started a SECOND sweep on the same card. pgrep found SIX orphans. Two sweeps sharing a
card halve each other - indistinguishable from external contention, which a genuine 8-card
external load hours earlier had already primed me to expect.
FIX: s16_audit.sh reaps bench_full.sh / bench_sweep.py inside the container before each attempt.
Those four rows were discarded. Tell to remember: contention that scales with a shape's
sensitivity is load; a uniform factor across every row is a throttle or a config error.

STATUS at end of session: the 12-shape dispatch audit is running on the fixed harness (equal
flags, equal iteration counts, equal repeat counts, orphan-reaped, resumable). The question it
answers - are other memory-bound rows mis-assigned the way DOWN decode was - is still open.
AUDIT RESULT (12/12 rows, equalized harness) - max(tc_HIP, tc_Fly) now picks FlyDSL on 7 of 12:
  FlyDSL: gate_up pref, GATE decode, QKV decode, imgQKV, FC1(+4%), LIN1, txt FC1 M256
  HIP:    LLM square, DOWN decode(+2.6%), imgPROJ(+9%), FC2(+6%), LIN2
ABSOLUTE VALUES ARE SUPPRESSED in that table: GATE decode's FlyDSL column reads 959.8 there but
1760 as a standalone probe - and 1749-1768 on ALL EIGHT cards, so it is not a bad card. These
decode kernels are 12-19 us, so the measurement is launch-rate bound and sensitive to NODE-level
activity (host CPU, fabric), not just to sharing a card: a probe on card 6 read 1068 while the
sweep worked on card 4. Both columns sag together, so large margins survive and small ones do not.

QUIET-NODE VERIFICATION (s15_head2head.sh, idle node, HIP given its full 4-binary x split-K x
best-of-3 advantage):
  QKV decode    HIP  233.6   FlyDSL  531.2   -> FlyDSL by 127%   (ships HIP today: 2.3x on the table)
  GATE decode   HIP  709.6   FlyDSL 1797.7   -> FlyDSL by 153%   (ships HIP today: 2.5x on the table)
  DOWN decode   HIP  297.6   FlyDSL  223.7   -> HIP by 33%

CORRECTION TO s15b: s15b's headline was that DOWN decode was mis-assigned (FlyDSL 384 vs HIP 340,
"12.9% left on the table"). The SAME script now reads FlyDSL 223.7 / HIP 297.6 - HIP ahead by 33%
- and the audit agrees (196 vs 201). HIP drifted too (297.6 vs 342.6 recorded), so the whole row
moves 15-40% between sessions. DOWN decode is therefore NOT established as mis-assigned and the
12.9% claim does not reproduce. This is the very error s15b was written to fix: comparing numbers
taken under conditions that were not held constant. What survives re-measurement are the rows
whose margin dwarfs the drift - QKV 127%, GATE 153%, M256 FC1 131%.

LESSON: a margin is only a result if it exceeds the drift. Record the spread next to the margin
and treat any decision inside the spread as UNMEASURED rather than as a close call.

STILL OPEN: imgPROJ (9%), FC2 (6%) and FC1 (4%) sit inside the drift band and need an interleaved
A/B/A/B on an idle node before either assignment is trusted.

================================================================================
s17  (2026-07-30)  same-session confirmation, breakdown, and the tile defect
================================================================================
Task: re-test within one session, confirm the numbers, break the performance down, optimize
further, update the slides.

HEADLINE: the decode tile this project has always used, 128,64,128, LEAVES C ENTIRELY UNWRITTEN
on the fp4-family B path (fp4 / w4a6 / w6a6_tc; fp6 is unaffected). It timed as the fastest tile
because it did not compute anything, and bench_full.sh led every memory-bound tile list with it.
Everything above about decode is therefore wrong, INCLUDING the s16 quiet-node verification block
directly above this one.

  evidence (kernel/s17_tn64.py, s17_tile_equiv.py): C is 100% zero at tile_n=64 for fp4, w4a6 and
  w6a6_tc, at M=128, 256 and 512, on both transcode paths; fp6 writes normally. fp4 goes from 0%
  zero at tile_n=128 to 100% at tile_n=64, so it is not an artifact of the A-layout gap.

  what it was worth: GATE decode w6a6_tc 1813.6 -> 688.0 (+163.6% fabricated), DOWN decode
  383.7 -> 118.6 (+223.5%), M256 FC1 2046.7 -> 789.8 (+159.1%), QKV decode 211.0 -> 176.3 (+19.7%).
  fp4 was inflated too (+47% to +143%), so even ratios between dtypes are not safe.

  RETRACTED: "QKV decode FlyDSL by 127%", "GATE decode FlyDSL by 153%", "M256 FC1 131%". At
  gate-verified tiles the clean audit reads GATE decode FlyDSL 687 vs HIP 711.6 and QKV decode
  FlyDSL 176.4 vs HIP 278.7 - HIP wins both. s16's conclusion that HIP wins DOWN decode was right,
  but for the wrong reason.

  ALSO RETRACTED: "transcoding from NVFP4 beats storing MXFP6 by 2-2.8x on decode". At verified
  tiles GATE decode reads fp6 754.3 vs tc 687 - storing MXFP6 is ~10% FASTER there. The 2-2.8x
  came from comparing a real fp6 number against a tc number produced by a kernel writing nothing.

  FIXED: tile_n=64 removed from MEM/DEC/MEM256 in bench_full.sh. bench_sweep.py now refuses to
  report a timing for any tile that leaves C all-zero - which required non-degenerate operands,
  because the harness fills the fp4-family scale bytes AND the transcode LUT with zeros, so an
  all-zero C is the CORRECT result under its own inputs. That is why nothing ever noticed. The
  guard borrows the tensors for one launch with unit scales and a nonzero LUT, then restores them
  so the timed loop sees byte-identical inputs to every previously recorded measurement. Verified
  to fire on tile_n=64 for fp4/w6a6_tc and not to fire for fp6 or any valid tile.

HWCVT vs LUT, interleaved A/B/A/B in ONE session, 3 samples per arm, spread reported:
  GATE decode  +2.69% (spread 0.14/0.16)   QKV decode +2.17%   DOWN decode +0.95%
  M256 FC1     +1.01%                      SQUARE     +4.54% (spread 0.35/1.24)
  GATE prefill +0.16% -> INSIDE NOISE
  s15's decode claims reproduce to the digit (+2.7/+3.0/+2.9 -> +2.69). s15's COMPUTE claim does
  NOT: it said +8.6-9.6%, today SQUARE gives +4.54% and GATE prefill nothing. s15's square was
  measured at tile 256,256 and today's at 128,256 (the lever's size depends on the tile), and its
  GATE prefill was a single rep taken just before the container was SIGKILLed.

BREAKDOWN (compute regime, same session, SQUARE): fp4 3628 -> w4a6 3587 -> fp6 2151 -> tc 1917.
  A being 6-bit costs 1%. B being 6-bit costs 40%. The transcode arithmetic costs 11%. The
  transcode is the SMALL term; the operand format is the big one. fp4 at 3628 is only ~44% of the
  ~8300 fp4 MFMA peak, so compute shapes are operand-delivery bound, not MFMA bound. Since B is
  held in registers and A streams through LDS, a 6-bit B costs register pressure -> occupancy.

OPTIMIZATION: every compute row ever measured here used tile_m=128 (CB="128,256,128;128,128,128"),
while HIP's compute win came from BM=512. The transposed ratio 256,128,128 was never offered:
  SQUARE 2002.9 -> 2359.5 (+17.8%)   LIN2 1688.0 -> 2191.3 (+29.8%)
  FC1    2172.4 -> 2403.8 (+10.7%)   FC2  2184.5 -> 2281.5 (+4.4%)
  and it flips SQUARE, LIN2 and imgPROJ from HIP to FlyDSL. --xcd 1 adds ~1.5% more. Gated by
  equivalence: 256,128 is bit-identical to the incumbent 128,256 across 36 configs on both
  transcode paths (also 128,128, 128,512, 256,256). Added to CB.

LEVER CLOSED: tile_k=256 for w6a6_tc (rejected today because the transcode assumes k_unroll=1) is
  NOT worth building - tile_k=256 hurts the dtypes that already allow it: fp6 -24.9% on SQUARE and
  -37.7% on LIN2. fp6 shares the 6-bit operand path, so the transcode path would inherit the loss.

METHOD: I contaminated my own audit again - probes on card 6 during the card-4 audit pulled its FC1
  FlyDSL cell to 1626 against 2403 standalone. Same class as s16's orphan stacking, but deliberate
  rather than leftover. One measurement at a time on this node, even across different cards.
  And: an implausible win is a bug report. tile_n=64 beating the no-transcode kernel was the tell;
  the ceiling columns (fp4, w4a6) exist precisely to make that check possible.
s17b (2026-07-30 pm)  the open compute A/B: node still busy, and the two levers interact
----------------------------------------------------------------------------------------
NODE NOT QUIET. Measured it rather than eyeballing: a 5-sample repeat probe on card 4 spread 69.1%
(1958.9/1907.8/2141.1/2155.5/666.0) and s17_cardscan.sh found the quietest of all 8 cards at 4.18%
with the rest 6-20%. Nothing on this node can resolve a 4-10% effect right now. s17_quiet_ab.sh
aborts on its own spread instead of producing numbers that have to be discarded.

BEST-OF UNDER LOAD, AND THE LIMIT OF A HALF-SPLIT CHECK: contention only depresses, so max-of-many
should converge. It did not settle SQUARE: the run reported +94.8% with BOTH HALVES AGREEING,
because the LUT arm was depressed in all four passes. A half-split rules out one unlucky sample,
not a sustained busy stretch that lands on one arm. Added a floor check (an arm below what it
already achieved at a WORSE tile is depressed, not slow).

THE FLOOR CHECK THEN MIS-FLAGGED A REAL EFFECT, which is the actual finding. SQUARE's LUT read
1316.6 and 1371.2 half an hour apart while every other arm sat at its quiet value. Measuring both
paths at both tiles interleaved in one run (s17_tilepath.sh):
  SQUARE  128,256,128:  LUT 1617.8  HWCVT 1519.2   (-6.1%)
  SQUARE  256,128,128:  LUT 1181.6  HWCVT 2189.0  (+85.3%)
  -> on SQUARE the best tile is PATH-DEPENDENT: LUT prefers 128,256 (-27.0% at the new tile),
     HWCVT prefers 256,128 (+44.1%). LIN2 and FC1 both prefer the new tile on both paths.

CONSEQUENCE: the tile lever and the transcode lever are NOT separable. "+17.8% from the tile" and
"+4.5% from HWCVT" describe a decomposition that does not exist -- on SQUARE the new tile is a 27%
loss for the LUT and a 44% gain for HWCVT. Same shape of error as s15's async x HWCVT inversion:
a lever measured against one fixed setting of an interacting lever is a statement about that
setting, not about the lever. The tile sweep that chose 256,128,128 ran --hw_transcode on only.

PRODUCTION IS FINE: bench_full.sh's fly() maximises over the joint (tile x path) space rather than
optimising them separately, and the audit's tc column matches the HWCVT arm to 1-2%.

STILL OPEN: clean absolutes for the compute-side A/B. The signs and ordering above are
within-run-valid; the sizes need a quiet node.
================================================================================
s17c  THE "NOISY NODE" WAS THE CARD WARMING UP; ALL 12 ROWS RE-MEASURED WARM
================================================================================

RESOLVED THE OPEN ITEM. The compute-side A/B now has clean absolutes, and the reason the two
previous attempts failed was not contention.

THE QUIETNESS GATE WAS ABORTING ON ITS OWN WARM-UP. On card 7 it read 1936.7 then 2553.9 / 2558.9 /
2562.8 -- three samples agreeing to 0.35% and one low first sample -- and aborted at 24.4%
"spread". Printing every tile of every invocation instead of one cell (s17_cold.sh) settled it:

  inv  128,128,128  128,256,128  256,128,128  256,256,128
   1        935.6       1721.6       1540.9       2052.2
   2       2031.2       2052.4       2539.8       2155.5
   4       1949.5       1989.9       2550.8       2056.2
   7       2053.2       2059.9       2564.1       2178.6
   8       2054.8       2060.8       2564.7       2179.1

Invocation 1 is low in ALL FOUR columns at once and every column climbs monotonically to a plateau
where the last two invocations agree to 0.08%. Contention hits one cell; a clock ramp lifts the
whole row. The ramp takes ~6 invocations, not one.

CONSEQUENCES:
  1. Card 7 was quiet all along -- warmed, the same gate reads 0.16% and passes.
  2. Any script measuring arm A before arm B while ramping is BIASED AGAINST A. Best-of-N over
     passes was load-bearing, not merely defensive: the max silently discarded the ramp.
  3. Numbers measured on the ramp are simply low. The previous audit is under-reported.
  4. The earlier card scan ("best card 4.18%, rest 6-20%") is partly this artifact. Genuinely busy
     cards stay identifiable by WHERE the low sample falls: mid-sequence, not first.
PROTOCOL NOW: warm until two consecutive samples agree within 0.5%, THEN gate, THEN measure.
Added to bench_full.sh as warm(), and to every s17c script.

JOINT (TILE x PATH) SWEEP, warmed card 7, best of 3, spreads 0.04-0.5% (s17_joint.sh). Measuring
the two interacting levers jointly instead of pricing one at a fixed setting of the other:
  shape      LUT@best  HWCVT@best   HWCVT gain    new tile vs old (HWCVT / LUT)
  SQUARE       1949.1      2564.5       +31.6%      +24.4% / -24.6%
  GATEpref     2168.0      2361.4        +8.9%      +29.9% / +18.7%
  imgQKV       1890.0      2148.7       +13.7%      +23.5% / +14.0%
  imgPROJ      1659.3      1956.0       +17.9%      +36.5% / +21.4%
  FC1          2280.3      2449.2        +7.4%      +27.3% / +19.5%
  LIN2         1938.9      2183.2       +12.6%      +28.6% / +24.1%
So HWCVT is worth +7..32% with each path at its own best tile (the old "+4.5%" priced it at one
tile), and the 256x128 tile is worth +24..37% on the HWCVT path for all six shapes. The s17b
path-dependence on SQUARE is CONFIRMED warm, not an artifact: the tile is -24.6% for the LUT and
+24.4% for HWCVT there.

ALL 12 ROWS RE-MEASURED WARM (bench_full_s17c.tsv). Dispatch now 7 FlyDSL / 5 HIP:
  shape                     fp4    w4a6     fp6  tc_fly  tc_HIP   ships
  LLM square              3409.9  3352.9  2269.6  2564.2  2458.6   FlyDSL +4.3%
  LLM gate_up pref        3096.9  3016.4  1982.4  2350.5  2042.2   FlyDSL +15.1%
  LLM DOWN decode          232.3   142.8   133.1   118.6   342.0   HIP    +188%
  LLM GATE decode         1149.1   856.5   753.4   717.8   713.9   FlyDSL +0.5%  <- FLIPPED
  LLM QKV decode           360.1   231.1   251.3   176.5   279.5   HIP    +58%
  FLUX imgQKV             2932.8  2738.0  1882.7  2136.9  1771.9   FlyDSL +20.6%
  FLUX imgPROJ            2786.3  2643.1  1771.9  1951.7  1758.8   FlyDSL +11.0%
  FLUX FC1                3113.9  3097.6  2034.4  2454.9  2147.8   FlyDSL +14.3%
  FLUX FC2                3204.2  3088.6  2034.7  2268.8  2333.4   HIP    +2.8%
  FLUX LIN1               3148.0  3156.9  2010.8  2343.2  2090.9   FlyDSL +12.1%
  FLUX LIN2               3225.3  3151.0  2051.4  2187.4  2384.3   HIP    +9.0%
  FLUX txt FC1 M256       1268.4  1037.0   938.0   774.3   851.2   HIP    +9.9%

TWO ROWS WERE CAUGHT DEPRESSED AND RE-MEASURED, by cross-checking the audit against the joint
sweep: FC1 recorded 2134.0 where the joint sweep said 2449.2 (12.9% apart). A warm re-measure
(s17_recheck.sh) returned 2443.8, i.e. the joint sweep was right and the audit cell was depressed;
imgPROJ likewise (1894.0 -> 1948.4). Both shapes showed 6-16% spread across three passes, so this
is intermittent contention a single pass can land in. Dropped both rows and re-ran with FLYREPS=7:
FC1 2454.9, imgPROJ 1951.7. The whole FC1 row rose (fp4 2973.9 -> 3113.9), confirming the row was
uniformly depressed rather than one bad cell. SQUARE agreed across all three measurements to 0.11%.

WARM DECODE A/B (s17_decab.sh), replacing unwarmed deck figures:
  shape        LUT     sp     HWCVT     sp     gain
  DOWNdec     87.8   0.34%     118.5  0.17%   +35.0%
  GATEdec    535.5   0.39%     732.8  5.58%   +36.8%
  QKVdec     132.0   0.08%     176.3  0.11%   +33.6%
  M256FC1    638.6   0.19%     788.6  0.27%   +23.5%
Cross-checks the audit cells: DOWN 118.5 vs 118.6, QKV 176.3 vs 176.5.

WHY tile_n=64 WROTE NOTHING -- root cause found by reading, fixed at compile time:
  num_acc_n         = tile_n // num_waves // 16  =  tile_n // 64    (num_waves = 4)
  _num_acc_n_packed = num_acc_n // _fp4_pack_N   =  tile_n // 128   (_fp4_pack_N = 2 for 4-bit B)
The 4-bit B path packs two N accumulator blocks per mfma_scale operand, so the MFMA loop is driven
by the second line. At tile_n=64 it floors to ZERO: no MFMA is issued and C is never written. fp6
carries pack_N=1 and computes correctly at the same tile -- exactly the dtype split observed.
compile_preshuffle_gemm_a8 now REFUSES to compile the combination, quoting the arithmetic, so the
shape cannot be selected at all rather than being caught after being built and timed
(s17_guard64.py: the three 4-bit dtypes raise, fp6 at tile_n=64 still compiles, all four shipped
tiles unaffected).

WHY HIP OWNS DECODE, QUANTIFIED. MI355X = 256 CUs. FlyDSL's grid is (M/tile_m) x (N/tile_n) and the
kernel has NO SPLIT-K. At decode M=128 gives one M block and tile_n floors at 128, so the grid is
N/128:
  DOWN  N=2560  ->  20 WGs =  7.8% of CUs  -> 0.35x HIP
  QKV   N=4096  ->  32 WGs = 12.5%         -> 0.63x HIP
  M256  N=12288 -> 192 WGs = 75.0%         -> 0.91x HIP
  GATE  N=19456 -> 152 WGs = 59.4%         -> 1.01x HIP  (ships FlyDSL)
The deficit tracks occupancy, not the dtype. NEXT LEVER: HIP's decode win uses split-K and BN=32.
BN=32 is the cheaper port and is blocked by one choice -- the kernel splits its 4 waves along N only
(n_per_wave = tile_n/4), so tile_n=32 gives num_acc_n=0. Split the waves along M for decode tiles:
each wave takes 2 of the 8 M blocks, keeps full tile_n=32 so num_acc_n=2 and the packed-N MFMA gets
its pair, and DOWN's grid becomes 80 WGs -- 4x the occupancy. Bounded change to the wave mapping
plus the epilogue row/col derivation.

NODE: the container took FOUR external SIGKILLs (exit 137, OOMKilled=false, no memory limit, 2.5 TB
free) during this session, one inside 80s of starting. This is the condition bench_full.sh's row
resume was built for in s15. Added s17_supervise.sh (host-side restart-and-resume loop) and made
the 17-binary verify gate skippable when its recorded pass is newer than every binary -- attempts
were dying inside the gate, so no row could ever complete.

DECK: 13 slides, 0 geometry problems. New slide 12 "Why decode goes to HIP: the kernel cannot fill
the GPU" (occupancy table + the blocked BN=32 lever). Slide 9 matrix, slide 10 both ladders, the
routes table and the lever table all carry the warm numbers.

--------------------------------------------------------------------------------
s17c (cont.)  THE OCCUPANCY ACCOUNT PAID OFF: tile_m WAS NEVER SWEPT ON DECODE
--------------------------------------------------------------------------------

Acting on the §5e occupancy arithmetic instead of just recording it. Every decode tile this project
ever measured was 128,64 / 128,128 / 128,256 -- all tile_m=128, which at M=128 is exactly ONE M
block. tile_m=64 splits M in two and doubles the grid with no kernel change. Legal: m_repeat=4,
_m_repeat_packed = 4//2 = 2, A-loads 16 B/thread. tile_m=32 is NOT legal (8 B/thread fails the
16-byte A load), so 64 is the floor.

  decode row   WGs @128  of 256   WGs @64  of 256   tile_m=64
  DOWN               20    7.8%        40   15.6%     +14.1%
  QKV                32   12.5%        64   25.0%     +29.5%
  GATE              152   59.4%       304  118.8%      -5.7%
  M256 FC1          192   75.0%       384  150.0%      +9.9%

THE GAIN IS ORDERED BY HOW STARVED THE ROW WAS, and the single row that LOSES is the one where
doubling overshoots 100% of the CUs (304 WGs = 118.8%, so the second pass runs 19% full). The
occupancy model predicted the sign, not just the story.

Bit-identical over 36 configurations (s17_equiv64.sh) -> shipped in DEC, MEM, MEM256 with tile_m=128
retained so dispatch takes the max. Four memory rows re-measured warm:
  DOWN decode      118.6 -> 137.6   (+16.0%)   HIP 341.6 still wins
  QKV decode       176.5 -> 228.4   (+29.4%)   HIP 279.0 still wins, gap -36.9% -> -18.1%
  M256 FC1         774.3 -> 859.5   (+11.0%)   FLIPS TO FLYDSL (HIP 848.9)
  GATE decode      717.8 -> 702.0   (tie)      new tile not selected here
Reference columns moved with them (QKV fp4 360.1 -> 549.7, w4a6 231.1 -> 371.0), so this is a
whole-family gain rather than a transcode-specific one.

GATE decode is a TIE and should be reported as one: 717.8 vs 713.9 in one audit, 702.0 vs 708.8 in
the next -- inside 1% both ways.

WARM DECODE A/B RE-TAKEN AT THE SHIPPED TILE LIST (pricing the lever at a configuration nobody runs
is how the +4.5% compute error happened, so the decode A/B now uses the same DEC list as production):
  DOWNdec   LUT 109.4  HWCVT 137.3  +25.5%
  GATEdec   LUT 542.6  HWCVT 696.0  +28.3%
  QKVdec    LUT 178.9  HWCVT 228.3  +27.6%
  M256FC1   LUT 641.1  HWCVT 857.9  +33.8%

FINAL DISPATCH 7 FlyDSL / 5 HIP: FlyDSL takes 6 of 8 compute rows plus M256 FC1; HIP takes FC2,
LIN2 and the three M=128 decode rows.

NEXT LEVER (unchanged, now with a measured floor under it): split the 4 waves along M instead of N so
tile_n=32 becomes reachable -- another 4x on DOWN's grid on top of tile_m=64. Remaining gap after
this session: DOWN 342 vs 138, QKV 279 vs 228.

DECK: slide 12 retitled "Decode is an occupancy problem, not a transcode problem", now carrying the
tile_m table; 13 slides, 0 geometry problems.

--------------------------------------------------------------------------------
s17c (cont.)  THE WAVE-GRID SURGERY CANNOT PAY -- MEASURED BEFORE WRITING IT
--------------------------------------------------------------------------------

The recommendation to port HIP's BN=32 by splitting the 4 waves along M was WRONG on the economics.
Killed for the cost of one measurement.

TWO STRUCTURAL FACTS. (1) B is loaded buffer_load-direct from global into each wave's REGISTERS (no
LDS staging), indexed via n_blk_list/n_intra_list which derive from n_tile_base = wave_id *
n_per_wave. Waves split along M therefore all need the SAME B columns and each issues its own loads:
4x1 M-split quadruples B load work per workgroup, a 2x2 grid doubles it. HIP pays none of this
because it LDS-packs B once per block. (2) tile_n=32 is unreachable anyway: 8 bytes/thread fails the
16-byte B load, so tile_n=64 is the floor.

fp6 PRICED THE SURGERY FOR FREE. fp6 packs ONE N accumulator block instead of two, so tile_n=64 is
already legal for it today, in the current 4x1 wave grid, four waves splitting 64 columns, hence NO B
amplification -- a strict upper bound on what the surgery could buy (s17_narrown.sh, warm, best of 3):

  shape       best tn=64   best tn=128   narrow-N
  DOWNdec          107.4         108.3      -0.8%
  GATEdec          376.7         396.8      -5.1%
  M256FC1          465.4         508.1      -8.4%
  QKVdec           154.7         251.4     -38.5%   (spreads 11-74%, magnitude unreliable)

Narrow-N does not pay even when it is FREE, so the 4-bit path paying 2x B traffic for the same grid
certainly cannot. MECHANISM, visible in the same runs: at tile_n=64 each wave gets n_per_wave=16 ->
num_acc_n=1, one accumulator block, so per-wave work collapses faster than the workgroup count rises.
Meanwhile tile_m=64 gains +30% on fp6 DOWN (83.4 -> 108.3) because splitting M adds workgroups
WITHOUT shrinking the MFMA-to-load ratio.

=> OCCUPANCY IS NECESSARY, NOT SUFFICIENT. Both tile_m and tile_n add workgroups; only tile_m keeps
each wave's work intact. That is why one was worth +10..30% and the other is worth nothing.

REMAINING LEVER, CORRECTED: split-K. It multiplies the grid by S while each workgroup keeps its full
tile over K/S, so per-wave MFMA-to-load is unchanged and total B traffic stays N*K. Exactly what HIP's
decode win rests on (its own atomicAdd split-K was worth DOWN 183 -> 352). Cost is real: a reduction
across partials (fp32 atomics into C, or workspace + second reduce pass), which means changing the
epilogue output path and zero-initialising C -- larger than anything in this session.

================================================================================
s17d (2026-07-30, cont.): SPLIT-K IMPLEMENTED. The decode gap was a missing lever.
================================================================================

The s17c trace closed with "remaining lever, corrected: split-K", plus an estimate that it was larger
than anything in that session. It was priced first, then built, gated and shipped.

PRICED FIRST -- AND THE FIRST PROBE WAS THE WRONG INSTRUMENT.
s17_splitk_probe.py ran S independent K/S GEMMs concurrently on S streams (same total FLOPs, same
total operand bytes -> aggregate throughput should bound what K-parallelism buys). It reported a loss
everywhere: DOWN -16.4%, QKV -40.9%, GATE -53.0%, M256 -45.9%. Its own ms column refuted it: 1 launch
0.057 ms, 2 launches 0.061 ms, where two concurrent half-K launches should each take ~0.025 ms. The
launches were SERIALISING and S launch overheads were being billed to split-K.
=> A probe that cannot separate its own overhead from the effect is not evidence, either direction.

THE RIGHT INSTRUMENT: trade N against K at constant FLOPs, in ONE launch. (M, N*S, K/S) does the same
multiply-accumulates, reads the same operand bytes (B is N*K either way), and gives S x the workgroups
with a K loop S x shorter. No launch overhead, no reduction, nothing to correct for (s17_wgprobe.py,
warmed to plateau; FLOPs fixed so TFLOP/s is directly comparable):

  row                     S=1          S=2          S=4          S=8         S=16      best
  DOWN 128x2560x9728   131.4/40    251.5/80        skip         skip         skip    +91.4%
  QKV  128x4096x4096   228.4/64    341.3/128    626.2/256    595.2/512   528.4/1024  +174.1%
  GATE 128x19456x2560  687.5/152   891.6/304       skip         skip         skip    +29.7%
  M256 256x12288x3072  858.2/384  1030.7/768    961.0/1536      skip         skip    +20.1%
  (TFLOP/s / workgroups; skip = K/S not a multiple of 256, which the fp4 scale layout requires)

QKV is the occupancy model on one line: peak at S=4 = exactly 256 WGs on 256 CUs, DECAYING at S=8 and
S=16. So S should land the grid near the CU count, not be maximised -- and reproducing the predicted
shape is what made this confirmation rather than just a number.

IMPLEMENTATION (psg_preshuffle_gemm.py, ~40 lines). K was doing two unrelated jobs; only one moves:
  - addressing strides stay FULL K (operands are still full-K tensors): _k_div4_factor, k_bytes_b,
    _K1_outer, _scale_row_stride_elems, k_bytes_factor -- 7 sites
  - loop trip counts become _k_loop = K // split_k: num_tiles, both c_k_main, both c_k_stop, both
    last_k, range(0, K-tile_k, tile_k), final load_fp4_scales index -- 9 sites, all in the K driver
Every k index in the load path is relative to an operand base, so sliding four base pointers to the
slice makes the whole existing load path correct with NO further edits:
  A                 rows are K long                                   -> bytes(K/S) per row
  B                 preshuffled (n0,k0,4,16,kpack); one k0 unit is
                    64 B of K for each of 16 N lanes                  -> 16 * bytes(K/S)
  scale_a/scale_b   (row group, K/256, 64) i32, 1 byte per K element  -> K/S
  C                 one partial plane per split                       -> bz * M*N*2
num_records shrinks by exactly what the base advanced, preserving the M-tail masking. Grid -> (gx,gy,S).
Compile-time rejection of bad divisors: K/S must be a multiple of tile_k, of 256 for the 4-bit family,
and >= 4*tile_k (the pipelined loop peels a prologue and drains 3 tiles).

TWO DSL GOTCHAS, one compile cycle each:
  - const_expr is the IDENTITY at runtime; it is a marker for the AST transformer to keep a branch at
    Python level. A plain `if x is None:` in a traced helper became a device predicate, so BOTH arms
    were evaluated and the None arm reached index_cast(i64, None).
  - fx.ptrtoint returns i64, not index: the offset must be cast and added in i64 space.

GATE: sum(partials) == full-K output. Random operands with PER-K-CHUNK-VARYING block scales, so no two
K slices contribute the same thing and a duplicated/wrong-slice bug cannot pass (s17_splitk_check.py).
All 6 cases OK at rel 2.33-2.35e-3 across w6a6_tc / fp6 / fp4 and both tile_m -- exactly the bf16
re-rounding level, since partials round to bf16 where the unsplit kernel held one fp32 accumulator.

WARM MEASUREMENT (iters 300, reps 3), kernel-only then net of a reduction over the partials:
  row                  S=1     best S   kernel-only        net of reduce
  DOWN decode         133.3      19    688.2 (+416%)    448.0 (+236%)
  QKV decode          227.8       8    617.1 (+171%)    371.4  (+63%)
  GATE decode         695.7       5    993.8  (+43%)    622.3 (-10.6%)
  M256 FC1            853.5       2   1005.5  (+18%)    717.9 (-15.9%)
  DOWN decode / fp6   140.6      19    510.2 (+263%)    360.1 (+156%)
  QKV decode / fp4    289.8       8    761.8 (+163%)    412.8  (+42.5%)
The reduction is real, not an artifact: GATE at S=5 moves ~30 MB in ~7.7 us = ~4 TB/s, near the limit
for a separate pass. Starved rows pay for it many times over; rows already near full occupancy hand
back the entire win. Next lever is therefore a CHEAPER REDUCTION (fuse it / fp32 atomics in the
epilogue), not more splitting.

DISPATCH, through the audit harness (split_k wired into bench_sweep.py, swept JOINTLY with the tile;
FlyDSL number INCLUDES the reduction; HIP's decode column has always been the best over its own
split-K sweep, so this is finally like-for-like). 8 compute rows replayed from cache:

  row                  FlyDSL before   with split-K      HIP     ships
  LLM DOWN decode          137.6      381.2 (+177%)    341.94   FlyDSL (was HIP)
  LLM QKV decode           228.4      303.4  (+33%)    279.52   FlyDSL (was HIP)
  LLM GATE decode          702.0      699.1            707.11   HIP, by 1.1%
  FLUX txt FC1 M256        859.5      857.1            833.29   FlyDSL

=> DISPATCH IS NOW 9 ROWS FLYDSL, 3 ROWS HIP (FC2, LIN2, GATE decode). The decode assignment three
sessions read as "HIP's decode kernel is better designed" was one missing lever: HIP split K across
workgroups and FlyDSL could not. Second time in this table that a regime which looked like a design
verdict was an absent knob (the first was tile_m=256 on compute).

Deck rebuilt: 13 slides, 0 geometry problems (MXFP6_Emu_s17d.pptx). Slide 9 matrix + both charts
recomputed from bench_full_s17d.tsv (the decode bar chart and the tc/W4A6 ratios were still on an
older dataset than the matrix above them); slide 12 rewritten around split-K; slide 13 routes and
optimizations updated. Skills doc gains rules 3n (price a lever, and check the probe cannot charge its
own overhead), 3o (separate the two jobs a compile-time constant is doing), 3p (gate a reduction-order
change on the invariant, not bit-equality), 3q (a per-workgroup gain is not a shipped gain until the
reduction is paid; check whether the other side's number already includes its equivalent cost).

--------------------------------------------------------------------------------------------------
s17e (chase W4A6 / beat W6A6 everywhere): THE GAP IS ITEMISED, 88% OF IT IS AT THE HARDWARE FLOOR,
AND FOUR CANDIDATE LEVERS ARE CLOSED BY EVIDENCE. One shippable win (+4.2%, flips the last decode
row); no code change kept. Timing work cut short by another tenant taking the node.

WHERE THE GAP IS. Measured warm, quiet card, reference columns timed in the SAME round-robin window
so ratios are not stitched across scripts (gauges 0.09-0.31%):

  row                          tc      fp6     w4a6   tc vs fp6   tc vs w4a6
  GATE decode 128x19456x2560  702.3   775.9    916.3     -9.5%      -23.4%
  M256 FC1    256x12288x3072  877.4   980.8   1178.0    -10.5%      -25.5%
  SQUARE      4096x4096x8192 2511.0  2228.5   3606.7    +12.7%      -30.4%
  LIN2       4352x3072x15360 2157.3  1836.3   3513.6    +17.5%      -38.6%

w4a6 and w6a6_tc are the SAME kernel over the SAME fp4 B bytes; only the B operand differs. So
tc/w4a6 is the transcode's price at constant traffic, and w4a6/fp6 is what halving B's bytes buys:
+19.6% (GATE), +20.6% (M256), +52.8% (SQUARE), +87.9% (LIN2), against a transcode costing 23-39%
throughout. That single pair of columns explains the sign of every row. It is structural: the
transcode is paid once per M-block ((M/tile_m)*N*K) and at M=128 there is ONE M-block, so the kernel
converts the entire weight matrix inline for a single use -- and that row is latency-bound anyway
(18% of peak FLOPs, B traffic 3.1us of an 18.2us kernel, 152 of 256 CUs busy), so the saved bytes buy
almost nothing. Not two rows needing a fix; a crossover in M to locate and turn into a dispatch rule.

THE BILL (probe arms, each its own cache key, VALU per unrolled body, GATE config):
  tc full 706 | -mantissa muls 541 (=165, 55%) | -scale word 672 (=34, 11%)
  -all transcode 407 (=299 total; decodes+pack = 100, 33%) | w4a6 352
s16 already proved the 55% piece irreducible (both cvt_scalef32 ops honour only the scale operand's
EXPONENT, so the E4M3 mantissa must be a separate multiply) and the 33% piece is 16 decodes/fragment
because pk8/pk16 fp4->f16 are gfx1250-only. So 88% is at the floor, and the bytewise trick I had
ready was for the 11%. Itemising first is what stopped that.

FOUR LEVERS CLOSED:
 1. waves_per_eu (never set on any row; preloads came from an fp8-tuned table). +4.2% on GATE decode
    (702.3 -> 732.1) and <0.5% on the other three. GATE decode is the row HIP holds by 1.1%, so this
    knob alone flips it -> dispatch would be 10 of 12 FlyDSL. Preload depths move nothing: the table
    already specifies 8 for the decode tiles, so the sweep's 1-4 were shallower by construction.
 2. The occupancy cliff was an artifact of my own dumps. tc at 174 VGPR/2 waves vs w4a6 162/3 looked
    decisive -- but those dumps lacked --async_copy, which SHIPS ON. In the shipped config tc is
    162/3 waves, and on SQUARE tc uses FEWER registers than w4a6 (250 vs 260) with 2 waves against
    w4a6's 1. No cliff anywhere.
 3. LUT/hardware hybrid (ds_read absorbs a decode AND a multiply = the 88% the ISA forbids touching,
    hardware pk32 removes the LUT's ~30 bit-shuffle ops). Priced from compiled ISA before writing it:
    LUT 1169 VALU/body, minus packing ~937, vs hardware 974. A tie, with 2x the ds_read and the 8KB
    arena. Not built.
 4. 128 register copies at tile_m=256 (141 v_mov vs w4a6's 13, 26% of tc's VALU, absent at tile_m<=128).
    Restating the pack operand as 16 whole dwords instead of 32 extracted f16 halves -> IDENTICAL ISA,
    4 more VGPRs. Allocator pressure, not an IR artifact. Reverted; no dead lever left behind.

TWO INSTRUMENT FAILURES, both caught, both now encoded as rules (3r-3u):
 - The first wpe sweep read SQUARE's baseline at 1707 where the same call gave 2562 and a card scan
   gave 2400-2478 on all eight cards, then "found" +17%/+40.6%. It compiled a fresh kernel before
   each sample, and one 80-iteration SQUARE sample is <10ms of work -- it timed post-compile cold
   clocks. s17_rr.py now compiles everything first, uses >=100ms samples, interleaves round-robin,
   and prints the baseline's own spread as a self-disqualifying gauge.
 - The bill's first run had every arm equal at a third of true speed, and its own self-check caught
   it (NOCVT deletes exactly what separates tc from w4a6, so +0.8% is impossible). My fault: the ISA
   verification phase compiled under FLYDSL_DUMP_IR into the same JIT cache the timing phase loads
   from. Dump compiles now get their own cache via HOME.

S18 -- w4a4_tc (BOTH operands NVFP4, A transcoded to MXFP6 in the staging copy)
=============================================================================
SHIPPED: SWAR E4M3->E8M0 SCALE CONVERT (on by default; TC_SCALE_BYTEWISE=0 reverts).
The old form peeled one scale byte at a time -- shift, mask, mask, shift, compare, select,
add, mask, shift back, or -- for each of 4 bytes. All four go at once and nothing carries
across a byte lane, because exp<=15 plus bump<=1 lands at most 136, and mant+7<=14 stays in
the low nibble, which is what makes bit 3 of that sum exactly (mant != 0):
    mant = s & 0x07070707 ; bump = ((mant+0x07070707)>>3) & 0x01010101
    exp  = (s>>3) & 0x0F0F0F0F ; out = exp + bump + 0x78787878
Bit-identical output (the w4a4_tc value sweep reports 3.600e-01 / 9.484e-02 both ways; the
byte-uniform A gate stays exact at K=512 and 8192). VALU per unrolled body:
    w4a4_tc 128x128 1188 -> 1064   128x256 1678 -> 1534   (-9%)
    w4a6_tc 128x128  694 ->  674   128x256 1186 -> 1146   (-3%)
w4a4_tc gained ~7% in time, the baseline ~1-3%. It is called once per A row-repeat AND once
per B fragment per k-tile, which is why the doubly-transcoding kernel gains three times more.

SHIPPED: MANTISSA FACTOR AS A BIT PATTERN (TC_MF_INT, on; =0 for the f32 form, forced off under
TC_FOLDSCALE which needs a real f32). (1+mb/8) was uitofp -> *0.125 -> +1.0 -> truncf -> broadcast.
In f16 bits it is exponent field 15 with mantissa mb*128, exact for every mb (mb<=7 cannot reach
the implicit bit), and filling both halves of the dword in place also deletes the broadcast copy:
    mx = mant << 7 ; mf2 = bitcast<2xf16>((mx | mx<<16) | 0x3C003C00)
Bit-identical on scale bytes 0x38..0x3f (all eight mantissas), gates pass. All four f32 opcodes
go to zero. VALU 1534 -> 1464 (w4a4_tc) and 1146 -> 1100 (w4a6_tc), VGPRs unchanged.

TIMED, one card, reference re-timed after every arm and stable to 0.3% (2103-2109):
    w4a4_tc 128x128  1259.8 -> 1288.8 (swar) -> 1319.9 (mfint)   +4.8%
    w4a4_tc 128x256  1493.0 -> 1533.0        -> 1578.8           +5.7%
    w4a6_tc 128x256  2078.0 ->               -> 2106.5           +1.4%

MARGINAL VALU ELASTICITY DEPENDS ON OCCUPANCY -- the cross-arm law hides this. w4a4_tc converts
its 12.8% VALU cut at 0.45; the baseline converts its 7.3% cut at only 0.19. One wave has nothing
to co-issue against, two waves do. So VALU work is worth ~2.4x more to the 1-wave kernel, and
"how many waves" is worth more than either.

THE OCCUPANCY CLIFF MOVED, AND THE OLD CONCLUSION IS STALE. wpe=2 at 128x256 previously cost
16-18%; on the lighter kernel it WINS:
    wpe=none/1  vgpr 274  1 wave   0 spill insts   1580.3
    wpe=2       vgpr 256  2 waves  22 spill insts  1752.6   (+10.9%)
    wpe=3       vgpr 168  3 waves  189 spill insts  418.7   (-73%)
The two VALU cuts bought the allocator the room to reach 256 on 22 spill instructions instead of
189. w4a4_tc on SQUARE: 1493 -> 1752.6, +17.4% this session, and 0.718 -> 0.833 of the baseline.
LESSON: a register-pressure lever priced against a fatter kernel must be RE-PRICED after any cut
that touches liveness. NEXT LEVER, unbuilt: 18 genuine VGPRs would take those 22 spills to zero.
Candidates -- carry mant_bits (3 bits) instead of the scale word plus byte selector across the A
prefetch, drop the two zero pad words the staging transcode materialises and writes (the fp6
fragment read discards that dword, so its LDS content is never observed), or drop A's prefetch
depth entirely for w4a4_tc, which cannot use async_copy anyway and is VALU-bound rather than
latency-bound.

THE FAMILY IS VALU-ISSUE-LIMITED ACROSS ARMS, and the constant is measurable. TFLOPs x (VALU/MFMA), one
node, one quiet reference, tile 128x256:
    fp4      5.11 x 3927 = 20067      w4a6_tc  8.95 x 2235 = 20003
    w4a4_tc 11.98 x 1542 = 18473  (8% under the line: 1 wave vs the baseline's 2, and 2 ds_write per load)
fp4 and w4a6_tc sit within 0.3% of the same product across a 1.76x throughput range. So an
instruction deleted from the transcode is throughput gained roughly 1:1 -- which is both why
SWAR paid and why the remaining gap is not a tuning problem.

SHIPPED: FENCE A'S STAGED TRANSCODE BLOCKS APART (TC_A_SERIAL, on; =0 to interleave).
The 18-register gap to two waves was never scalar liveness -- it was two aligned tuples. Each
staged block needs 16 CONSECUTIVE VGPRs for the pk32 pack's f16 source, and the scheduler was
interleaving the two blocks a thread stages per k-tile, so both tuples were live at once. One
rocdl.sched_barrier(0) at the end of each block's stores:
    interleaved, no hint   vgpr 274  1 wave   0 spills   1315.0
    interleaved, wpe=2     vgpr 256  2 waves  22 spills  1660.3
    FENCED, no hint        vgpr 256  2 waves   0 spills  1660.2
    fenced, wpe=2          vgpr 256  2 waves   0 spills  1674.9
Two waves with zero spills and NO occupancy hint, VALU unchanged at 1464, bit-identical output
at every tile. The kernel is register-bound, not latency-bound: the staging copy amortises over
the workgroup, so serialising it costs nothing measurable.

FAILED, and worth keeping as a flag (TC_WRITE_PAD, default ON = keep writing it). A's staged fp6
tile ends in an FP8 pad dword that the read side loads and drops (a2, _a3 = ...; _a3 unused), so
not writing it is numerically free and removes two zero words plus 8 ds_write. It measures WORSE:
vgpr 274 -> 278, wpe=2 spills 22 -> 43, -10% (1651 -> 1492). Splitting the pair of dwordx4 stores
into dwordx4 + dwordx2 costs more in aligned-tuple pressure than the two registers it saves.
Together with the fence result: in this kernel VGPR count is set by ALIGNED TUPLE pressure, and
reasoning about scalar liveness predicts the sign of the change wrong.

GATES after all three changes (SWAR, integer mantissa, fence): A-value gate exact for w4a6 and
w4a6_tc at K=256/512/2048/8192 x tiles 128x256 and 128x128; K-advance CORRECT for w4a6, w4a6_tc
and w4a4_tc at K=512/1024/4096 with 0/128 zero rows; w4a4_tc value sweep unchanged across all
eight scale mantissas; fenced == interleaved bit for bit at 128x128, 128x256, 256x128.
NOTE two gate sections first came back VACUOUS -- they passed --tiles and --dtype to probes that
take --tm/--tn and --dtypes, so one printed no verdict and the other compared "" to "". A gate
that cannot fail is worse than no gate; both now require non-empty probe output.

SHIPPED (gated): 512-THREAD WORKGROUPS (WG512=1, off by default). The lever the cost law has been
pointing at all along. Waves split N and nothing else -- n_per_wave = tile_n // num_waves -- so 8
waves at tile_n=512 hand each wave the SAME 64-wide slice and the SAME 128 accumulators that 4
waves at tile_n=256 do. Per-thread everything is unchanged except the A codes each thread stages,
which halve. Four edits: num_waves_const 4->8, total_threads 256->512, the hardcoded 4 in
layout_wave_lane, and block=(256,1,1) at the launcher; plus known_block_size on the kernel, since
AMDGPU defaults max_flat_workgroup_size to 256 and rejects the launch otherwise.
    w4a4_tc 128x256 wg256  vgpr 256  2 waves  0 spill  valu 1464  packs 24  ds_write 32
    w4a4_tc 128x512 wg512  vgpr 234  2 waves  0 spill  valu 1283  packs 20  ds_write 16
packs 24->20 is exactly the intended halving: A's 8 blocks per thread became 4, B's 16 unchanged.
GATED: everything is behind _WG512 and the 256-thread paths are bit-for-bit unaffected (verified).
VERIFIED: A-value gate exact at 128x512 for w4a6 and w6a6_tc at K=512 and 4096; K-advance CORRECT
for fp4, w4a6, w6a6_tc AND w4a4_tc at K=512/2048 with 0/128 zero rows; the w4a4_tc value sweep at
512 threads matches the 256-thread answer exactly on two scale-byte sets.

WG512 NARROWS THE GAP, and reading it off SQUARE ALONE says the opposite. Square is the shape
where WG512 helps least and the two arms move together (+4.3% base, +5.1% tc), so the first run
-- square only -- concluded the baseline gained more and the gap widened. Across shapes, on a
quiet card (reference 2181-2191, 0.5%):
    shape          base 256->512   w4a4_tc 256->512   ratio before   ratio after
    LLM_square         +4.3%            +5.1%            0.825         0.832
    LLM_gate_up       +18.2%           +38.7%            0.749         0.878
    FLUX_imgQKV        +2.9%           +22.5%            0.718         0.855
    FLUX_FC2          +31.5%           +55.4%            0.696         0.821
    FLUX_LIN1         +18.2%           +41.7%            0.721         0.864
    LLM_GATE_dec      -30.2%           -20.6%            0.723         keep 256
w4a4_tc gains 22-55% where the baseline gains 3-32%, because halving the A codes per thread is
worth double to the arm that transcodes A. Compute rows land at 0.82-0.88. The decode row LOSES
for both: at M=128, tile_n=512 leaves N/512 = 38 workgroups for the machine, so WG512 is a
compute-shape lever and the decode rows keep 256 threads. DISPATCH: thread count now joins tile
as a per-shape choice.

WHAT async_copy IS ACTUALLY WORTH, AND WHY IT IS NOT A LEVER FOR w4a4_tc. LDS staging in this
kernel is A-ONLY -- B goes global->registers per fragment and never touches LDS, so there is no
B-side async to enable. async is worth 28% to the baseline (2113.6 vs 1647.3 at 128x256), and
the reason is occupancy, not the DMA:
    w6a6_tc async=on   vgpr 256  2 waves  1100 valu   0 ds_write   2113.6
    w6a6_tc async=off  vgpr 289  1 wave   1127 valu  32 ds_write   1647.3
    w4a4_tc (no async) vgpr 256  2 waves  1464 valu  32 ds_write   1782.6
Holding A in registers across compute_tile costs the baseline 33 VGPRs and its second wave. The
fence already bought w4a4_tc that wave, so it is not missing anything async would give it -- and
on equal staging footing w4a4_tc is 8.2% FASTER than the baseline (1782.6 vs 1647.3) despite 337
more VALU, because it reads half of A's global bytes. Routing A through async into an LDS scratch
and transcoding LDS->LDS would only add a ds_read to buy a wave already in hand.

A'S STAGING HAS NO FAT LEFT, priced against the right control. Measured against w6a6_tc NON-async
(same register staging, same 32 ds_write, transcodes B only), A costs (1464-1127)/8 = 42 VALU per
staged block against B's (1127-493)/16 = 39.6. Six percent. The earlier "12 ops per block of
addressable overhead" was an artifact of differencing against the ASYNC baseline, which folds in
async's 33 registers. Nothing to reclaim in byte_sel, opsel or scale addressing.

GEOMETRY EXHAUSTED AT 256 THREADS. tile_k=256 does not compile for ANY dtype, baseline included;
tile_n=192 does not compile; 64x256 runs and loses (1352.6), exactly as the cost law says, since
halving tile_m doubles the B term. 128x256 is optimal and wpe is now a no-op on it (1784.8 vs
1786.9). The single remaining structural lever is 512-thread workgroups: at tile 128x512 the
accumulators stay at 128/thread while A's staged blocks per thread halve, taking VALU to ~1296
for a predicted +5-9%. total_threads=256 is not a parameter -- it feeds the per-thread byte
budgets, the coord functions, the wave-to-output mapping, the accumulator layout and the
epilogue. Not attempted.

FINAL 12 ROWS AFTER THE FENCE. Reference 2112-2116 on eleven rows; LLM_square's row read 1311
and was retaken on its own (reference 2108-2112 across all fifteen of its arms):
    row              fp4   w4a6  w4a6_tc  w4a4_tc  tc/base   (start)  w4a4_tc gain
    LLM_square      3674   3086     2112     1789    0.847     0.718     +19.8%
    LLM_gate_up     3095   2670     1890     1681    0.889     0.681     +17.0%
    LLM_DOWN_dec     247    190      130      111    0.856     0.875      +4.7%
    LLM_GATE_dec    1155    957      707      544    0.770     0.759      +4.6%
    LLM_QKV_dec      546    426      250      192    0.768     0.707     +15.7%
    FLUX_imgQKV     3235   2730     1831     1568    0.857     0.693     +17.9%
    FLUX_imgPROJ    2857   2495     1771     1348    0.761     0.667     +13.7%
    FLUX_FC1        3449   2924     2022     1741    0.861     0.688     +16.3%
    FLUX_FC2        3413   2859     2001     1523    0.761     0.634     +18.0%
    FLUX_LIN1       3339   2819     1926     1678    0.871     0.741     +15.1%
    FLUX_LIN2       3432   2905     1871     1550    0.828     0.691     +19.8%
    FLUX_FC1_M256   1284   1174      879      682    0.776     0.771      +9.7%
w4a4_tc is now 0.76-0.89 of w4a6_tc (was 0.63-0.88) and 0.35-0.54 of native fp4. wpe=2 is no
longer needed anywhere -- the fence delivers two waves on its own (1789 without it, 1780 with).
Still short of the baseline, and still for the structural reason: a strictly positive A term.

EARLIER 12 ROWS (SWAR + integer mantissa, before the fence), kept for the delta:
    row              fp4   w4a6  w4a6_tc  w4a4_tc  tc/base  (tc/base at session start)
    LLM_square      3640   3116     2116     1754    0.829   0.690
    LLM_gate_up     3328   2844     1999     1609    0.805   0.681
    LLM_DOWN_dec     239    191      130      108    0.833   0.875
    LLM_GATE_dec    1117    957      700      520    0.742   0.759
    LLM_QKV_dec      546    427      251      171    0.683   0.707
    FLUX_imgQKV     3224   2729     1820     1551    0.852   0.693
    FLUX_imgPROJ    2840   2490     1769     1283    0.725   0.667
    FLUX_FC1        3434   2917     2032     1705    0.839   0.688
    FLUX_FC2        3369   2901     1996     1464    0.734   0.634
    FLUX_LIN1       3333   2831     1925     1663    0.864   0.741
    FLUX_LIN2       3472   2918     1867     1558    0.834   0.691
    FLUX_FC1_M256   1271   1169      876      653    0.745   0.771
Compute rows moved 0.69 -> 0.83-0.86, which is where w4a4_tc was WORST. The four memory-bound
rows read slightly lower than at session start, and that is the earlier measurement being
corrected, not a regression: those rows carried ref 88-93% then and 100% now, and steady
interference compresses the spread between arms. Trust this run.

CEILING, now a prediction rather than a measurement. VALU/MFMA = base + tcB/tile_m + tcA/tile_n.
w4a4_tc's is the baseline's plus a strictly positive A term, so under the law above it cannot
win at any tile on any VALU-bound shape. 12 rows with per-arm tile selection, reference bracketed
at row boundaries: 0.63-0.88x w4a6_tc (worst on compute, best on the most memory-bound decode),
0.30-0.53x fp4. No shape escapes into A-bandwidth-bound, where the halved A bytes would pay:
fp4 still reaches 88-100% of its peak on the A-heavy rows (FC2, LIN2).

WHY A MUST BE WIDENED AT ALL -- and it is not the reason it looks like. fp4->fp6 is EXACT
(fp6_code = fp4_code << 2; E2M3 has E2M1's range and two spare mantissa bits), so the widening
buys zero accuracy. Its only job is to hold the E4M3 scale mantissa that E8M0 cannot express.
That reframes the whole kernel: the transcode exists solely to convert scales, and A's 388
VALU/body is rent on 3 mantissa bits. Two ways out, both priced:
 (a) current -- round scale up, fold (1+m/8)/2 into fp6 data. RMS 9.5%, max 36%. Costs the 388.
 (b) round the E4M3 scale to NEAREST E8M0 and feed A as raw fp4 (mixed MFMA, A=fp4 x B=fp6).
     Costs ZERO A VALU and halves A's HBM bytes -- under the law it should land at or above
     w4a6_tc. Accuracy: the residual is uniform per block, 1/(1+m/8) or 2/(1+m/8), max 33%,
     RMS ~19.7% over m -- about 2x worse than (a). NOT BUILT: it is a numerics-contract change,
     not an optimization, and the contract is the project.
Data-transcode cost is already ~1 VALU/code (33 per 32), and the per-code work provably cannot
go below that: fp4 magnitudes are 2^k or 3*2^k, so (1+m/8)*value has only two mantissa outcomes
per m -- a shift plus a per-block constant, still one op per code. Nothing left to shave there.

CLOSED THIS SESSION: lds_stage 3 and 4 fail to compile for the ENTIRE fp4 family, baseline
included -- so the deeper ring that would hide A's added global->transcode->LDS latency does not
exist to be tried. Untried and predictable: 512-thread workgroups would halve A's 1/tile_n term
(1534 -> ~1340 VALU, ratio ~0.79) at the cost of the whole tile-to-thread mapping. Still a loss.

NODE LOST MID-SESSION. Another tenant: all 8 GPUs at 100%, 8 processes holding 267GB each, per-card
spread 0.16% -> 12-17%, the fixed reference config reading 1121 where quiet it reads 2400-2478.
NOTE the reference read 1121 at a 1.94% SPREAD -- steady interference, so a spread-only gate passes
it cleanly. s17_queue.py therefore gates on the reference's absolute VALUE and refuses to measure,
with four queued measurements ready to run in one command when the card frees: (1) confirm GATE
wpe=3 against HIP 707 for the dispatch flip, (2) tile_m 32/48 on the decode rows (the tile lists have
only ever held 64 and 128), (3) locate the M crossover where tc overtakes fp6, (4) re-take the bill
in time rather than instruction counts.

================================================================================
FINAL DISPATCH TABLE -- thread count selected per shape alongside tile
s18_dispatch.sh; full output in kernel/s18_results/s18_dispatch_table.txt
Each arm picks its own best over {256thr x 3 tiles} + {512thr x 128x512 where N%512==0}.
Reference re-timed after EVERY row; quiet card reads 2100-2190.

row                 fp4    w4a6 w4a6_tc w4a4_tc tc/bas tc/fp4    ref  w4a4_tc picks
LLM_square         3774    3263    2283    1902  0.833  0.504   2183  128,512,128/512t
LLM_gate_up (r)    3439    2941    2056    1789  0.870  0.520   2180  128,512,128/512t
FLUX_imgQKV        3015    2630    1776    1420  0.800  0.471   2102  128,512,128/512t
FLUX_imgPROJ       2577    2300    1496    1334  0.892  0.518   2080  128,512,128/512t
FLUX_FC1           3175    2647    1953    1561  0.800  0.492   2102  128,512,128/512t
FLUX_FC2           3561    3006    1975    1621  0.821  0.455   2092  128,512,128/512t
FLUX_LIN1          3264    2732    1982    1781  0.899  0.546   2192  128,512,128/512t
FLUX_LIN2          3875    3248    2116    1767  0.835  0.456   2184  128,512,128/512t
LLM_DOWN_dec        173     149     109     100  0.910  0.576   2187  64,128,128/256t
LLM_GATE_dec       1105     756     561     452  0.807  0.409   2105  64,128,128/256t
LLM_QKV_dec (r)     547     426     250     191  0.765  0.350   2179  64,128,128/256t
FLUX_FC1_M256      1285    1171     893     692  0.775  0.538   2185  64,128,128/256t

w4a4_tc lands at 0.765-0.910 of w4a6_tc, median 0.83, AHEAD ON NO ROW. That is exactly what the
cost law says must happen: VALU/MFMA = base + tcB/tile_m + tcA/tile_n, so the doubly-transcoding
arm carries a strictly positive extra term at every tile. Against the no-transcode ceiling it is
0.35-0.58 of fp4.

THREAD COUNT IS NOW A DISPATCH AXIS. w4a4_tc picks 512 threads on 8 of 8 compute rows; the
baseline picks them on only 4 of 8 (square, imgPROJ, LIN1, LIN2). The arm that transcodes A gains
roughly double, because 8 waves at tile_n=512 halve the A codes each thread stages while leaving
per-thread accumulators unchanged. All four decode rows keep 256 threads in both arms.

The earlier prediction in this log -- "512-thread workgroups would halve A's 1/tile_n term
(1534 -> ~1340 VALU, ratio ~0.79). Still a loss." -- was right that it stays a loss and WRONG about
the size: it lands at 0.83 median and 0.91 best, because the estimate priced only the VALU cut and
not the register relief (VGPR 256 -> 234 at 128x512).

TWO ROWS WERE DISCARDED AND RETAKEN (marked r). gate_up and QKV_dec first landed while the card was
shared -- references 1385 and 1653 against a 2180 quiet baseline. At face value gate_up read 0.556
AND NAMED A DIFFERENT WINNING TILE (128,256/256t instead of 128,512/512t); retaken quiet it reads
0.870. The lesson is sharper than "noisy rows read low": a contended row can select the wrong
configuration, so a dispatch table built on one ships the wrong kernel. Re-timing the reference
after every row is the only thing that caught it.

================================================================================
WHY 256 -> 512 THREADS PAYS: the lever is tile_n, the threads are the enabler
s18_wgwhy.sh; full table in kernel/s18_results/s18_wgwhy.txt. Both arms non-async (the
equal-staging control). Reference 1682.3 before, 1678.5 after -- quiet.

The shipped A/B (128x256/256t vs 128x512/512t) moves tile_n AND thread count together and so
cannot say which one pays. Four points separate them (square / gate_up / FC2):

  w4a4_tc  128x256/256t  1802 1289 1091   vgpr 256 w2 sp0   valu 1464  packs 24  acc/thr 128
  w4a4_tc  128x256/512t  1551 1468 1296   vgpr 144 w3 sp0   valu  837  packs 12  acc/thr  64
  w4a4_tc  128x512/512t  1906 1787 1686   vgpr 234 w2 sp0   valu 1283  packs 20  acc/thr 128
  w4a4_tc  128x512/256t  1138 1048  862   vgpr 512 w1 sp60  valu 3077  packs 40  acc/thr 256
  w6a6_tc  128x256/256t  1678 1587 1376   vgpr 289 w1 sp0   valu 1127  packs 16  acc/thr 128
  w6a6_tc  128x256/512t  1685 1516 1326   vgpr 154 w3 sp0   valu  627  packs  8  acc/thr  64
  w6a6_tc  128x512/512t  2111 1929 1923   vgpr 256 w2 sp10  valu 1077  packs 16  acc/thr 128
  w6a6_tc  128x512/256t  1745 1543 1446   vgpr 512 w1 sp0   valu 2638  packs 32  acc/thr 256

NEITHER KNOB WORKS ALONE. Threads alone (128x256 at 512t) halves the accumulators too (mfma
128->64), trading register-blocking reuse for occupancy: -13.9% on square even though VGPR falls
to 144 and occupancy rises to 3 waves. tile_n alone (128x512 at 256t) doubles accumulators to 256
per thread, hits the 512-VGPR ceiling, drops to 1 wave, spills 60 times, -40%. Only the pair holds
n_per_wave = tile_n/num_waves at 64 while halving the A tile's per-thread share.

MOST OF THE GAIN IS NOT THE TRANSCODE. Per-thread packs go 24 -> 20 for w4a4_tc (A 8 -> 4, B held
at 16) but 16 -> 16 for the baseline, which has no A packs at all -- and the baseline still gains
25.8% on square. The shared prize is HBM traffic: A is loaded once per workgroup and the grid
covers N in N/tile_n workgroups, so doubling tile_n halves A's global re-reads. A's halved
transcode is w4a4_tc's private extra on top, and it is what closes the gap, but it is second order.
(The baseline here also collects 289 VGPR/1 wave -> 256/2 waves, an occupancy step the SHIPPED
async baseline already has -- which is why its shipped WG512 gain is the smaller +3..32%.)

This also fixes the write-up. Crediting "512-thread workgroups" as the lever was the natural
reading of the shipped A/B and it is wrong: the lever is tile_n, which the cost law
(base + tcB/tile_m + tcA/tile_n) had named three sessions earlier. Stated that way it further
predicts the decode loss, which the "more threads" story does not: at M=128 the grid is 1 x N/512,
so doubling tile_n halves an already-too-small workgroup count.

DOCS UPDATED: kernel/opt_trace_s18.md (full session trace, incl. 6.1 on the above),
kernel_optim_skills.md (levers + three new mental models), PROGRESS.md (dated s18 section),
REPRODUCE.md (w4a4_tc dtype row + the tuning-flag table incl. the w4a6_tc alias), and the deck
MXFP6_Emu.pptx, which gains slides 14 ("W4A4: transcoding both operands, and the wall it hits")
and 15 ("W4A4 final: thread count joins tile as a dispatch choice"). Deck is built from
build_pptx.py; this box has no Windows Python, so it now builds under WSL with
MXFP6_ROOT/MXFP6_TEMPLATE pointed at /mnt/c paths (see _build.sh).


================================================================================
2026-07-31 s19 -- quiet-node re-measurement of every arm, and the fp6 control
================================================================================

ASKED: re-measure on a quiet node including all references, verify they are measured correctly,
then optimise w4a6_tc as close to w4a6 as possible, then update the docs.

NODE. All 8 GPUs at 0% utilisation but 93-94% of VRAM allocated -- an idle tenant holding memory,
the state that can wake mid-sweep. So the reference (w4a6_tc square 128x256/256t async) is re-timed
after EVERY row and printed. It held 2177-2189 all session, a 0.55% band. s18's references sat at
2083-2110 with two rows visibly contended, which is why some s19 rows look 10-23% better for
reasons that have nothing to do with the kernel.

CRITICAL -- w4a6_tc was being tuned on the wrong axis. Its cost law has no A term (A arrives
already MXFP6): VALU/MFMA = base + tcB/tile_m. Every WG512 config shipped in s18 raises tile_n,
which is the A-side knob. The B-side knob is tile_m, and 512 threads make 256x256 reachable at
constant accumulators: acc_VGPR = (tile_m/16)(n_per_wave/16)(4) = 16*2*4 = 128, the same 128 the
shipped configs use. Measured: VALU 1077 -> 881 (lowest of any config), per-thread B packs 16 -> 8
exactly as the law predicts, vgpr 248, 2 waves, no spills. Shipped; wins 5 of 12 rows.
  256x512/512t was predicted to fail before running (256 acc VGPRs) and failed the predicted way:
  VGPR pinned at the cap, 348 spills, a tenth of throughput.

CRITICAL -- a config that measured +36% because it computed two thirds of the output.
128x384/512t reported 2957 on FC2 against a true best of 2177. tile_n=384 on 8 waves gives
n_per_wave=48 so num_acc_n=3; the packed-N MFMA takes TWO blocks per step, runs 3//2=1 pair, and
silently drops the third. mfma=64 where every legal config issues 128. Gate behaviour is the part
to keep: the K-advance probe said CORRECT (it counts zero ROWS; this defect drops COLUMNS -- a gate
is blind along any axis it does not enumerate); the A-value gate caught it only through
"# WARN nonuniform spread=" lines, its max_rel/rms_rel being identical to the good configs; the ISA
probe caught it outright. A defect that removes work makes the kernel faster, so the timing harness
ranks it first -- only a counter that must move if the story is true can refuse it.

FIXED. The old guard was `if is_fp4 and (tile_n // 64) < 2`: tile_n//64 is num_acc_n only at 4
waves, and `< 2` catches only the zero case. Now derived from the real wave count and rejecting odd
as well: _num_acc_n = tile_n // num_waves_const // 16; reject if < 2 or odd. Verified by calling the
compiler directly and reading the exception text, since "failed to compile" and "rejected by my
guard" are different observations. Legality depends on tile_n AND wave count: tile_n=384 is fine at
4 waves (num_acc_n=6), degenerate at 8 (3). The old form could not express that, which is why 384
was reachable. All previously shipped configs still accepted at the wave count they ship at.

FALSE ALARM, recorded because the reasoning generalises. The same gate run reported max_rel 0.749 /
rms 0.537 for w6a6_tc at the new 256x256/512t, which looked like a hard failure. Those are the
probe's numbers for w6a6_tc at EVERY tile including the one shipped for six sessions; w4a4_tc's
0.360/0.0898 in the same run matches its documented fold error, confirming calibration. The alarm
came from reading an absolute error figure without its control, and through `tail -5` which cut the
verdict line off.

CRITICAL -- three quarters of the w4a6_tc -> w4a6 gap is not the transcode. The two kernels do not
compute the same product: w4a6 feeds B as MXFP4 (16 B/lane operand), w4a6_tc must feed it as MXFP6
(32 B/lane) because that is the only way to carry NVFP4's E4M3 scale mantissa. w4a6 is the LESS
accurate kernel. Adding `fp6` as a fourth reference (B fed as fp6, no transcode) decomposes it:
    gap = (fp6 operand format) x (transcode arithmetic)
    square  0.706 = 0.792 x 0.891      gate_up 0.705 = 0.758 x 0.930
    FC2     0.681 = 0.765 x 0.890      imgQKV  0.671 = 0.761 x 0.881
The transcode costs 5-12% on compute shapes and buys HALF of B's HBM bytes. Against the format it
actually needs, w4a6_tc runs at 0.88-0.99 of a kernel handed that format for free. Closing the rest
needs a 16 B/lane 6-bit MFMA operand, which is not a kernel's choice. Decode is the exception:
DOWN_dec is 0.691 of fp6 because at M=128/tile_m=64 the tcB/tile_m term is least amortised.
QKV_dec's 0.993 is not a triumph -- fp6 itself collapses to 251 there.

FINAL TABLE (TFLOP/s, quiet, reference per row in 2177-2189):
row             fp4   w4a6    fp6  w4a6_tc  w4a4_tc  tc/w4a6  tc/fp6  best w4a6_tc
LLM_square     3797   3235   2563    2284     1938    0.706   0.891  128,512/512t
LLM_gate_up    3447   2942   2229    2073     1786    0.705   0.930  256,256/512t *
LLM_DOWN_dec    236    191    188     130      110    0.679   0.691  64,128/256t
LLM_GATE_dec   1105    963    757     703      544    0.730   0.929  64,128/256t
LLM_QKV_dec     548    426    251     249      191    0.586   0.993  64,128/256t
FLUX_imgQKV    3306   2799   2131    1878     1580    0.671   0.881  256,128/256t
FLUX_imgPROJ   2961   2595   2020    1839     1498    0.709   0.910  256,256/512t *
FLUX_FC1       3546   3015   2310    2148     1877    0.712   0.930  128,512/512t
FLUX_FC2       3820   3208   2454    2184     1712    0.681   0.890  256,256/512t *
FLUX_LIN1      3443   2911   2177    2068     1779    0.710   0.950  256,256/512t *
FLUX_LIN2      3894   3261   2540    2223     1773    0.682   0.875  256,256/512t *
FLUX_FC1_M256  1273   1181    956     886      696    0.751   0.927  64,128/256t

ATTRIBUTION, measured back to back on the same card rather than across sessions:
  gate_up +0.6%, imgPROJ +2.9%, FC2 +5.2%, LIN1 +0.2%, LIN2 +5.0%  -> median +2.9%.
The raw s18->s19 deltas on those rows read +0.8/+23/+10.6/+4.3/+5.1%; imgPROJ would have been
published as a 23% optimisation when 2.9% is real. Small gain despite an 18% VALU cut because these
shapes are not VALU-bound at 2 waves -- s18's exchange rate (~0.19 per marginal VALU at 2 waves)
predicts ~3.4%, which is what happened. The cost law names the knob; the exchange rate prices it.

128x384/256t is newly legal at 4 waves (num_acc_n=6), was swept on all six rows whose N it divides,
and won none: legal but never best, not shipped as a default.

DOCS UPDATED: kernel/opt_trace_s19.md, kernel/s19_results/{s19_bside,s19_verify,
s19_dispatch_table,s19_attrib}.txt, PROGRESS.md, kernel_optim_skills.md, REPRODUCE.md, and the deck.

OPERATIONAL: /tmp/<name>.log on the shared host can be owned by another tenant -- the redirect
fails with Permission denied and the job never starts; write run logs under ~. And `pgrep -f s19_gap`
matched the ssh command carrying the pattern, reporting a job as running when nothing was: check for
the process AND for output the job must have produced.

================================================================================
s20 (2026-08-01) -- READ THE AMD LITERATURE, FOUND A FLAG PINNED OFF IN EVERY SWEEP
================================================================================

INPUT: five AMD documents, of which the W4A6 quantization blog is about this exact kernel family
(FlyDSL preshuffle GEMM, W_MXFP4_A_MXFP6). The useful outcome was not a new technique.

THE FINDING IS A REGRESSION REPORT, NOT A DISCOVERY. Split-K was designed, implemented,
correctness-gated, MEASURED AT +177% ON DOWN_dec, and presented on deck slide 12 ("Decode is an
occupancy problem, not a transcode problem") back in s17 -- that slide even names the follow-up
lever, "the next lever is a cheaper reduction, not more splitting". Then s18 and s19 rewrote the
sweep drivers and every one of them passes `--splits 1`. The kernel kept the feature, the repo kept
the gate, the deck kept the slide; the dispatch tables quietly stopped using it.

THE REAL DAMAGE: s19 then published DOWN_dec at 130 TFLOP/s -- against 381 already on record -- and
EXPLAINED the number: "at M=128/tile_m=64 the tcB/tile_m term is least amortised." Plausible,
mechanism-shaped, wholly wrong. A regression got absorbed into the mental model as a property of the
kernel. Restoring the flag reproduced s17 exactly (130 -> 393 now, 138 -> 381 then).

What the AMD blog contributed was the prompt to look. It names split-K as THE decode lever and
describes our exact shape:
  "the number of CTAs scales with the OUTPUT size M x N, and not at all with K. At decode there
   are not enough of them to fill the GPU."

WORTH (kernel/s20_splitk.sh, reference held 2185-2190 all run):
  LLM_DOWN_dec  w4a6_tc  130 -> 393  (+202%)   also fp4 +45%, w4a6 +124%, fp6 +118%, w4a4_tc +228%
  LLM_QKV_dec   w4a6_tc  250 -> 340  (+36%)
  LLM_GATE_dec  and FLUX_FC1_M256: nothing (-1.0% .. +1.1%)
  LLM_square control: -0.2% for both w4a6 and w4a6_tc -- pure overhead when the grid already fills

THE PREDICTOR IS WORKGROUP COUNT, NOT M. GATE_dec and DOWN_dec are both M=128. DOWN_dec launches
2x20 = 40 workgroups on 256 CUs and gains 202%; GATE_dec (N=19456) launches 304 and gains nothing.
"Decode shape" is the wrong abstraction; compute ceil(M/tm)*ceil(N/tn) and compare it to the CUs.

GATED, because a +202% claim on this path deserves disbelief -- s19's lesson was a config that
measured +36% by computing 2/3 of N, and split-K here once hid a silent epilogue bug worth "+95%".
kernel/s17_splitk_check.py already had the right invariant (summed partials must reproduce full-K,
with block scales varying per K chunk so a skipped slice cannot hide). It had one hole: w4a4_tc was
missing from the family list and was being handed per-token f32 scales and an unpacked A. Fixed.
Every arm, every split: rel = 2.3e-03 against a 1e-2 threshold. Split-K is correct.

THE NEXT BOTTLENECK IS THE REDUCTION -- WHICH s17 ALSO ALREADY SAID. On DOWNdec at split_k=19 the kernel
runs at 640 TFLOP/s raw and 416 net of the torch.sum over 19 bf16 partial planes -- the combine is
~5.4us against a 10.0us kernel, moving 13.1 MB at ~2.4 TB/s. A THIRD of the achievable throughput
on the split-K rows is spent in a separate elementwise kernel. Every number we report is the net
one (bench_sweep times kernel+reduce as one call), so this headroom is real and unclaimed -- and it
has been the top lever for three sessions. Folding it in-kernel (blog 5 uses semaphore/signal args)
is the fix, but note s17 already tried a signal-based variant (SIGSPLIT) and measured it -23..-29%
against the atomic on DOWN: the blog's design is not automatically right here, and whatever gets
built must beat both. Caveat: the current epilogue deliberately binds each workgroup to its own C
plane so a stray store cannot cross splits -- that is the safety an in-kernel reduction gives up,
and where the old silent bug lived.

CORRECTION TO s19 (repeated because it is the point): s19 attributed DOWN_dec's 0.691 of fp6 to
tcB/tile_m amortisation at tile_m=64. Wrong. The row was launching 40 workgroups onto 256 CUs with
split-K switched off in the harness. The tile was never the problem, and the project's own slide 12
had said so three sessions earlier.

Also: K/split_k must be a multiple of 256 (scales are addressed in 256-element K chunks) as well as
of tile_k, with >=4 k-tiles left. For K=9728 that admits only {1,2,19}. There is no way to tune the
grid to ~256 workgroups even though the docstring rightly advises it -- sk=19 gives 760. A second
reason the fix is the in-kernel reduction rather than better split_k selection.

WHAT THE DOCUMENTS CONFIRMED (so these axes are closed, not unexplored):
  - 256x256 at 512 threads is the best config: blog 3 sweeps 128x128/512t=1829, 256x256/512t=2288,
    256x256/1024t=2228 and picks the middle. Same config s19 derived from the B-side cost law,
    including that going wider on threads is a regression.
  - MFMA is 16 cycles for FP4 and FP6 ALIKE (32 only if an operand is FP8). FP6 and FP4 issue
    identically, so the 0.792 "operand format" factor in our gap decomposition is entirely a
    DELIVERY cost -- register footprint and LDS traffic -- with zero MFMA contribution. Nothing on
    the compute side can touch it. Upgrades s19's inference to a documented fact.
  - lds_stage looked like another pinned axis (harness defaults to 2, no script overrides) but the
    compiler rejects anything else for the 4-bit family at tile_k=128. Closed by constraint.

TOP UNTRIED LEVER: 8-wave ping-pong scheduling. Blog 3's last step takes 256x256/512t from 2288 to
2680 (+17%) with the HipKittens pattern -- 2 waves per SIMD alternating memory and MFMA, steered by
s_barrier on a wave-half predicate, s_setprio(1) around the MFMA, sched_barrier(0) fences. w4a6_tc
now ships EXACTLY that configuration. We emit sched_barrier and gpu.barrier; we lack the predicated
wave-half barrier (gpu.barrier is whole-workgroup) and s_setprio on the FlyDSL path. Not attempted:
it restructures the hot loop, and split-K was the higher-value use of the measurement time.

FINAL TABLE (TFLOP/s, quiet, reference per row 2184-2192, a 0.37% band):
row                 fp4    w4a6     fp6 w4a6_tc w4a4_tc tc/w4a6  tc/fp6  best_w4a6_tc
LLM_square         3798    3244    2565    2285    1938   0.704   0.891  128,512/sk1/512t
LLM_gate_up        3449    2943    2235    2073    1786   0.704   0.928  256,256/sk1/512t
LLM_DOWN_dec        416     421     409     392     367   0.931   0.958  64,128/sk19/256t  <<<
LLM_GATE_dec       1107     962     768     708     547   0.737   0.923  64,128/sk1/256t
LLM_QKV_dec         546     425     335     335     311   0.788   1.001  64,128/sk4/256t   <<<
FLUX_imgQKV        3299    2800    2134    1878    1566   0.671   0.880  256,128/sk1/256t
FLUX_imgPROJ       2962    2591    2020    1843    1493   0.711   0.913  256,256/sk1/512t
FLUX_FC2           3830    3220    2456    2178    1718   0.676   0.887  256,256/sk1/512t
FLUX_LIN1          3442    2910    2180    2069    1781   0.711   0.949  256,256/sk1/512t
FLUX_LIN2          3889    3253    2541    2221    1771   0.683   0.874  256,256/sk1/512t
FLUX_FC1_M256      1287    1182     955     896     664   0.758   0.938  64,128/sk1/256t
(FLUX_FC1 not re-taken: the node acquired another tenant on all 8 cards and the reference never
held quiet across the row. s19's clean value stands -- 3546/3015/2310/2148/1877, tc/fp6 0.930 --
and split-K does not apply to it, by the LLM_square control.)

Ten of twelve rows reproduce s19 within ~1%; the two that moved are the two the mechanism predicts.
w4a6_tc is now >= 0.87 of fp6 everywhere and above 0.95 on three rows.

PROCESS FIX OWED: the sweep drivers have no shared defaults block, which is how two rewrites lost a
gated, measured, presented result. Any script that pins an axis should have to say why, in the
script. And when a row regresses against a number already published, treat the prior result as the
reference and go find the delta before reaching for a mechanism that explains the new one.

DOCS UPDATED: kernel/opt_trace_s20.md, kernel/s20_results/s20_dispatch_table.txt, kernel_optim_
skills.md (levers, measure-discipline 6b/6c, mental models 1/16/17), REPRODUCE.md (new 3.2),
PROGRESS.md, and the deck.

OPERATIONAL: bench_sweep's --tiles is SEMICOLON-separated and it prints tile=(128, 256, 128) with
spaces INSIDE the tuple. Two successive versions of the split-K driver printed a complete, cleanly
formatted table of ZEROS -- first from passing spaces to --tiles (every invocation crashed unpacking
the triple, no BEST line), then from a whitespace field split on the tuple. Neither failed loudly.
The reference cell at the top of each run is what caught both, because it was the one number with a
known expected value. Put a known-answer cell in every table your harness prints, and read it first.

================================================================================
s21 (2026-07-31) - WHY A SHIPPED RESULT DISAPPEARED, AND THE FIX THAT MAKES IT
                   IMPOSSIBLE; + Gemma 4 31B shapes; + atomic split-K (rejected)
================================================================================

ROOT CAUSE OF THE SPLIT-K LOSS -- IT WAS NOT CONTEXT LENGTH.
Three separable structural causes, none of which more memory would have fixed:
 1. The finding lived in a DIFFERENT PROGRAM than the shipping numbers.
    bench_sweep.py defaults --splits to "1"; s17_splitk_probe.py defaults it to
    "1,2,4,8,16". Split-K was only ever exercised by its own bespoke probe and was
    never wired into the tool that produces published tables. No s17 .sh driver
    passes --splits at all. It was never in the main path to begin with.
 2. The driver helper was COPY-PASTED forward, freezing the flag.
    s18_dispatch.sh:19 and s19_dispatch.sh:19 are byte-identical, --splits 1
    included -- one go() function copied across sessions, not two independent
    decisions. Nobody re-reads a helper they copied; it becomes infrastructure the
    moment it is pasted. The value sat on line 19 in every session's context.
 3. Knowledge was stored in PROSE, and prose cannot fail a run.
    Slides, status.md and the opt traces were all correct and none are executable.
 What was needed was never "what did the last session say" but "what is this row
 supposed to score" -- a COMPARISON, which no amount of context performs for you.

THE MECHANISM: kernel/bestcfg.py + kernel/best_known.json
 - bestcfg.py space   -> the ONLY source of tiles and splits; drivers ask for a
   config space instead of carrying one, so there is no flag left to freeze.
   s21_bench.sh hardcodes no shape, no tile and no split_k.
 - bestcfg.py check   -> exits nonzero when a cell is >3% below its record. Replayed
   against the actual s19 number it reports
     REGRESSION LLM_DOWN_dec/w6a6_tc: 130 vs record 392.0 (-66.8%)
   and tells the reader to reproduce the recorded config BEFORE theorising, because
   the s19 failure was not missing the regression, it was explaining it.
 - bestcfg.py purge   -> a record taken during contention is worse than no record,
   because it becomes the bar every later run hides under.
 - split_k candidates are DERIVED (legal divisors landing the grid in 0.5-6x the
   256 CUs) rather than hand-listed. Pins live in the registry with a reason string.

GEMMA 4 31B DENSE ADDED (9 shapes, from the published Gemma4_31B config:
 embed_dim 5376, MLP intermediate 21504, 32 heads x 256, 16 KV heads, 60 layers).
 Only the 50 local-sliding layers are modelled; the 10 global layers use a unified
 K=V whose fused width could not be pinned from public config, and an invented shape
 looks exactly like a measured one once it is in the table.

 HEADLINE: on G4_DOWN_dec the TRANSCODE kernel BEATS native fp6 -- w6a6_tc 1060 vs
 fp6 953, tc/fp6 = 1.113. G4_O_dec too, at 1.019. Mechanism: w6a6_tc stores B
 fp4-dense, so on a memory-bound row it reads HALF the weight bytes fp6 does, and
 the transcode ALU cost amortises over K. Gemma down_proj has K=21504 (longest in
 the suite) against a narrow N=5376 -- exactly where "half the bytes" beats "no
 conversion". Qwen DOWN_dec hinted at it (0.958 at K=9728); doubling K passes parity.
 The Gemma decode rows also hold the three best tc/w4a6 ratios in the suite
 (0.887, 0.862, 0.849 against a 0.711 median).

 Split-K on Gemma confirms the predictor without exception: G4_DOWN_dec sk=12 (504
 WGs), G4_O_dec sk=8 (336), G4_DOWN_M256 sk=6 (504), G4_QKV_dec sk=3 (384), and
 G4_GATEUP_dec -- which already launches 336 workgroups alone -- sk=1. K=21504 admits
 every divisor of 84, so sk=12 is expressible; K=9728 allows only {1,2,19}, which is
 why the old suite could not show that split_k has an OPTIMUM rather than a direction.

ATOMIC SPLIT-K EPILOGUE -- built, correct, and REJECTED on speed.
 split_reduce="atomic": every workgroup buffer_atomic_fadds into one fp32 C plane,
 deleting the caller's reduction pass (which s20 measured at 35% of the split's win).
 Correctness 2.0-3.5e-3 vs split_k=1, BETTER than the default path's 3.0-4.7e-3
 (it accumulates in fp32 instead of rounding each partial to bf16). Still loses:
   DOWN_dec  sk=2  planes 207 / atomic 218   atomic +5%
   DOWN_dec  sk=19 planes 392 / atomic 251   planes +56%
   G4DOWN    sk=7  planes 834 / atomic 714   planes +17%
   G4DOWN    sk=21 planes 822 / atomic 512   planes +61%
 Pure arithmetic: atomic cost is LINEAR in split_k (split_k*M*N atomics) while the
 planes reduction is bandwidth-bound. The rows that want split-K want a LARGE split,
 so the lever and this fix pull in opposite directions. The project's own record said
 "on this hardware the atomic IS the cheap path" -- true against SIGSPLIT, whose cost
 included an exposed serial tail, and false against planes + a parallel torch.sum.
 BEATING A BAD DESIGN IS NOT EVIDENCE THAT YOU BEAT THE SHIPPED ONE.

TWO HARNESS TRAPS, both of which produced confident wrong answers:
 - flyc.compile LAUNCHES the kernel once. Every check in this project relied on
   stores being IDEMPOTENT, so that free extra launch never mattered. An accumulating
   epilogue is not idempotent: the gate read exactly 2.00x the reference at every
   split_k, which looks precisely like a kernel bug. What located it in one shot was
   a control the kernel had to be MODIFIED to allow -- atomic at split_k=1, where any
   ratio but 1.0 cannot be about splitting. It read 1.9927. A guard that rejects the
   useless configuration also rejects your best diagnostic.
 - The first crossover sweep said atomic won by up to +60%. It was contended, and
   contention hit the arms UNEQUALLY: planes is bandwidth-bound and fell to 82 where
   it later measured 207, while the atomic path is L2-atomic-rate-bound and barely
   moved. A shared node does not merely add noise to an A/B -- it can REVERSE it.

=====================================================================
2026-08-03 -- s22: the arms inside Gemma 4, and the accuracy answer
=====================================================================
Everything before this session measured a GEMM. This one measured what the GEMM
does to a model's answers. Cloze scoring (one forward pass, argmax over the
answer-letter logits), deltas measured against OUR OWN bf16 run on identical
questions so every harness difference cancels.

MMLU-Pro, Gemma 4 12B, n=2000 stratified, paired McNemar:

    arm        wbits  abits  scale            acc      delta      p
    bf16          16     16  --              48.75%      --      --
    w4a6_tc        4      6  E4M3 /32        49.70%    +0.95    0.372  ns
    w4a6           4      6  E8M0 /32        47.80%    -0.95    0.320  ns
    w4a4_tc        4      4  E4M3 /32        45.55%    -3.20    0.005  *
    w6a6           6      6  f32 per row     39.10%    -9.65   <1e-4   *
    w4a4           4      4  E8M0 /32        37.75%   -11.00   <1e-4   *

On 12B this reads as the thesis holding: w4a6_tc indistinguishable from bf16 at 4
bits per weight. IT DOES NOT SURVIVE 31B. Same protocol, n=1000, bf16 = 68.20%:

    arm        12B delta   31B delta   31B p
    w4a6         -0.95 ns     -2.70    0.004  *   <- best arm at 31B
    w6a6_tc      +0.95 ns     -7.70   <1e-4   *
    fp4         -11.00        -8.00   <1e-4   *
    w6a6         -9.65        -7.80   <1e-4   *
    w4a4_tc      -3.20       -12.20   <1e-4   *   <- best 4-bit at 12B, worst at 31B

The two E4M3 arms swap places with their E8M0 counterparts. Also: the 12B row that
started this (w4a6 -0.95 vs w6a6_tc +0.95) has p=0.32 and p=0.37 -- both within
noise of bf16 and therefore of each other. The +1.9 "advantage" was never real, and
reading it as one is what made 31B look like a surprise instead of a correction.

ROOT CAUSE: THE E4M3 SCALE BOOK IS TOO SMALL FOR WEIGHTS, AND THERE IS NO GLOBAL
SCALE. E4M3 spans 2^-6..240; E8M0 spans 2^-127..2^127, and _choose_scale silently
clamps. An E2M1 block wants scale amax/6, so it needs amax >= 0.094; Gemma's weights
sit near 0.02. Measured on real weights: 96% of 12B blocks and 99.8% of 31B blocks
clamp LOW. On weights the per-block scale had stopped being per-block at all, and
each block was squeezed into the bottom of {0,.5,1,1.5,2,3,4,6} -- a nominally 4-bit
block carrying 2-3 bits. Activations are O(1), sit inside the book, and E4M3 there
delivers the promised +2.2 to +2.9 dB. That resolves the ordering completely:
w4a6->w6a6_tc changes only weights (pure loss), fp4->w4a4_tc changes both (activation
gain vs weight loss; gain wins at 12B, loss wins at 31B). down_proj activations fail
the other way -- amax 3504 puts blocks ABOVE the book, and a 0.03% clamp rate costs
6.85 dB because it clips the largest values. This is NOT a property of NVFP4:
production NVFP4 carries a per-tensor FP32 scale for exactly this reason.

THE FIX, AND ITS LIMIT. A per-tensor global scale costs no kernel change (a
per-tensor constant on an operand is one on the output). It makes E4M3 weights beat
E8M0 by ~1 dB as intended and recovers +2.40 (w6a6_tc) and +1.30 (w4a4_tc) on 31B.
It does NOT close the gap: w4a6 stays best at -2.70 while w6a6_tc_g is 2.6 points
behind it carrying weights that measure ~1 dB BETTER. Left stated, not rationalised.
Off by default; the fixed arms are recorded as separate arms (_g) so the earlier
numbers keep meaning what they meant.

THE TRANSCODE IS FREE AND WAS NOT THE CULPRIT. Measured with the transcode included,
tc tracks e4m3 to within 0.05 dB. Cheap hypothesis, tested, false.

THE LEVER IS SCALE GRANULARITY, NOT ELEMENT WIDTH, and the table contains its own
control: w6a6 carries the MOST bits per weight and finishes second to last, because
it is the only arm not block-scaled (one f32 scale per row, not per 32 elements).
32x coarser scales cost more than 50% more element width buys.

ACTIVATION PRECISION DOMINATES WEIGHT PRECISION: w4a4 and w4a6 store weights
identically and differ only in activation format. That is worth ten points.

A GATE THAT COULD NOT FAIL. Three arms (w4a6, w6a6, w4a6_tc) had never run on
non-constant activations. verify_w4a6.py builds A constant along K, which makes it
permutation-invariant -- a wrong host byte order feeds the kernel a permutation of
identical values and the answer is right anyway. It passed for EVERY layout, so it
constrained none. Same blind spot covered the A-scale layout for all five arms.
Fixed (16 bytes at col, 8 at col+64 per K=32 block) and replaced by
e2e_layout_gate.py: random data, all arms, NumPy reference, corrupted-operand
negative controls.

PER-LAYER SNR PREDICTED THE WRONG ORDERING. It put w4a6 1-5 dB ahead of w4a6_tc;
end-to-end says the reverse. SNR weights every layer and output element equally and
the model does not. When a model-free proxy disagrees with a paired end-to-end
measurement, discard the proxy.

SAME ACCURACY IS NOT SAME BEHAVIOUR. Every arm changes bf16's answer on 33-50% of
questions while accuracy moves ~1 point. Two readings fit that with opposite
verdicts, so it was checked rather than assumed: the flips land where bf16's
top-two logit gap is ~1.8 nats vs ~6.3 elsewhere, and where bf16 itself scores
23-37%. Quantization is reshuffling near-ties the model was already failing. Also:
w4a6 (0.673 top-1 agreement) tracks bf16 more faithfully than w4a6_tc (0.586),
which scores higher. Highest accuracy and closest fidelity are different arms.

THE KERNEL IS NO LONGER THE COST. End to end the arms are ~8x SLOWER than bf16 and
~80% of that is the torch activation quantizer. It is launch-bound, not
bandwidth-bound: scaling M over 8192x makes it only 3.6-4.4x slower, and
torch.compile recovers 1.2-3.3x and no more. Quantize-then-GEMM as two passes
cannot win regardless of GEMM quality; a fused prologue is the only route. Also,
all 40 shape/arm combinations missed best_known.json, which is keyed on FUSED QKV
 and GATE_UP shapes while stock transformers keeps them unfused. Accuracy is
 unaffected (fusion is along N, quantization blocks run along K) but no perf claim
 should be read off that configuration.

=====================================================================
2026-08-04 -- s23: the W6A6 anomaly, the benchmark that measured option
                   position, and the two kernels around the GEMM
=====================================================================
Full trace: kernel/opt_trace_s23.md. Three s22 loose ends, all of which looked
like properties of the hardware and were not.

W6A6 CAME LAST ON SCALE GRANULARITY, NOT ELEMENT WIDTH. The plot labels the fp6
 arm "W6A6" and it finished below every 4-bit arm, which cannot be an
 element-width effect: E2M3 holds 32 magnitudes per sign to E2M1's 8. fp6 is the
 only arm in the suite NOT block-scaled -- one f32 per token and per output
 channel, 32x coarser along K. Quantizing the SAME 6-bit elements at both
 granularities on real captured activations (kernel/p15_fp6_scale_study.py),
 rowwise is 3.1x the error of block-scaled on average and 4.6x on down_proj,
 whose K is longest and whose outlier channels are worst. The axis label named
 the element format because that is what the kernel work was about; the variable
 that decided the ordering was carried only in prose.

THE FIX IS PER-INPUT-CHANNEL EQUALIZATION (kernel/qsmooth.py). Divide input
 channel j of A by s_j, multiply column j of W by s_j: exact identity in real
 arithmetic, so no kernel change and no runtime cost (weights folded at load, the
 division rides in the quantizer already reading every element).
 s_j = amax_i|A_ij|^a / amax_o|W_oj|^(1-a), normalised, clamped to [1/16, 16].
 Alpha sweep on fp6 layer error: plain 0.0990; a=0.5 -> 0.0365, a=0.6 -> 0.0363
 (2.7x), both endpoints WORSE than doing nothing (a=0 dumps everything on the
 weights, a=1 the reverse). 0.0363 vs block-scaled 6-bit's 0.0316 -- within 15%
 of an arm carrying 32x more scale metadata.

...AND IT DESTROYED THE ARM END TO END. fp6_eq scored 28.3% on GPQA (chance 25%)
 and 33.6% on MMLU-Pro against plain fp6's 39.4% and 46.4%. 2.7x better layer
 error, 13 points worse model. kernel/p16_fp6_eq_debug.py measures the same 328
 layers twice, changing only where the activations come from:
   captured from the calibration sentences   21.42 -> 28.43 dB   1/328 hurt
   captured from the real benchmark prompts  19.41 -> 25.60 dB  33/328 hurt
 On the prompts the model is scored on, L0.mlp.down_proj loses 18.9 dB (25.0 ->
 6.1) and 32 other down_proj layers lose too. MECHANISM: s is frozen at
 calibration, so a channel hotter at inference is divided by too small an s_j and
 comes out LARGER; with one f32 scale per token that single amplified channel
 raises the row amax and squeezes every other channel in the token. THE COARSE
 GRANULARITY THAT MOTIVATED THE EQUALIZER IS WHAT MAKES ITS MISTAKES NON-LOCAL.
 Block-scaled arms confine the same mistake to one K=32 block, which is why their
 _eq variants only wobbled a point or two.
 MEAN SNR ROSE ~6 dB IN BOTH CASES. The failure was local and severe and the mean
 is neither; the worst layer is the number to read off a proxy metric.

BAD DATA OR BAD FORMULA, PRICED INSTEAD OF ARGUED. kernel/p16_smooth_cap.py
 crosses 2 calibration sets x 2 centrings x 7 caps, all scored out of sample on
 activations from the real prompts:
   calibration   cap    mean     worst    dmean   hurt
   none           --    19.40 dB  2.72 dB    --     --
   prose          16    25.68    10.37    +6.28     33
   prose         inf    25.64     7.51    +6.24     33
   IN-DOMAIN      16    28.74    16.82    +9.34      1
   in-domain     inf    28.80    16.82    +9.40      1
 It was the data. In-domain = the MMLU-Pro VALIDATION split, disjoint from every
 scored test item. Layers hurt 33 -> 1, worst surviving regression -9.1 -> -0.3
 dB, worst layer 10.4 -> 16.8 dB. The cap is insurance, not the fix: it costs
 0.06 dB when calibration is right and buys +2.9 dB on the worst layer when it is
 wrong, so it stays on at T=16. Sweeping only the cap would have found +0.14 dB
 and concluded the method was hopeless.

W6A6 IS NOW THE MOST ACCURATE ARM (debiased harness, MMLU-Pro n=1000 / GPQA 198):
   arm         MMLU  plain   _eq   _eq2      GPQA  plain   _eq   _eq2
   bf16              56.7%    --     --            44.4%    --     --
   fp6 (W6A6)        46.4%  33.6%  54.4%           39.4%  28.3%  40.4%
   w6a6_tc           52.8%  51.7%  49.4%           40.4%  37.9%  40.4%
   w4a6              50.7%  50.1%  50.8%           38.9%  38.9%  38.9%
   w4a4_tc           48.7%  48.2%  47.9%           35.9%  37.4%  40.4%
   fp4               47.6%  48.6%  48.0%           31.3%  38.9%  29.8%
 fp6_eq2 is -2.3 points from bf16, the smallest gap of any arm (next w6a6_tc
 -3.9), and on GPQA is inside noise of bf16 (McNemar p=0.29). TWO LIMITS STATED:
 equalization only pays for the ROWWISE arm -- on block-scaled arms it is a wash
 or slightly negative (w6a6_tc 52.8 -> 49.4), consistent with the mechanism, so
 it is ON for fp6 and OFF elsewhere rather than on because it helped somewhere.
 And GPQA cannot separate these arms at all: one arm's s.e. is 3.5 points, so an
 unpaired gap needs ~9.9 points; the 40.4% shared by three arms is one number
 three times.

GPQA DISAGREED WITH MMLU-PRO BECAUSE IT WAS PARTLY MEASURING OPTION POSITION.
 Symptom: fp4 the BEST quantized arm on GPQA (-2.5 ns) and the WORST on MMLU-Pro
 (-11.0, p~1e-24). Two harness defects.
 (1) The GPQA prompt was not one task: load_bench took the question field on
     trust assuming options were inline, and item 0 carries two stacked option
     lists where item 1 carries one. Both benchmarks now render options through a
     single prompt builder.
 (2) Position bias dominated what was left. Share of answers on the most-chosen
     option slot, against a 25% uniform base rate, old -> new harness:
       bf16 67% -> 29%   entropy 1.44 -> 1.99 of 2.00
       fp6  67% -> 27%           1.42 -> 2.00
       w4a6 53% -> 34%           1.70 -> 1.96
       fp4  31% -> 29%           1.93 -> 1.99
     bf16 answered "A" on two thirds of items. Quantization perturbs that prior
     in a direction unrelated to accuracy cost -- and fp4, the arm that looked
     best on GPQA, was the LEAST position-biased arm to start with, which is most
     of what its lead was.
 FIX: score each item under all 4 cyclic rotations of its options and average the
 per-option logprobs (eval_bench --perms 4). 4x the forward passes; removes the
 bias by construction instead of by hoping. Cross-benchmark agreement, Spearman
 over the 5 arms: accuracy +0.30 (old) -> +0.40 (debiased) -> +0.70 on mean
 log P(gold), which uses the whole score vector and is the ranking metric to
 read. The residual disagreement is POWER, not bias, and the tooling now says so:
 eval_analyze reports se_one_arm and min_resolvable_gap per benchmark and
 p13_tradeoff.py draws that band. Both benchmarks got harder and more
 informative: bf16 37.9 -> 44.4% on GPQA, 48.75 -> 56.70% on MMLU-Pro, MMLU-Pro
 top-slot 31 -> 18%. The check that catches this is free -- answer histogram vs
 base rate -- and this project already owned it (choice_dist.py, s22) without
 having run it on the harness itself.

THE TWO KERNELS AROUND THE GEMM. s22's profile: decode step 5078 us with GEMM
 1272, split-K reduction 1113, activation quantizer 800.
 - QUANTIZER TILE WAS PICKED ON PREFILL SHAPES. NB=64 blocks per program leaves 2
   programs at M=1 on a 256-CU part. Retuned per M (kernel/p16_qtile.py,
   qfused.tile_for): M=1 729.8 -> 395.0 us (1.85x), M=8 1.74x, M=32 1.57x,
   M=128 1.20x, every config bit-exact.
 - SPLIT-K REDUCTION WAS AT TORCH'S FLOOR, NOT THE HARDWARE'S. c.sum(0) costs
   2.70-2.98 us where an empty Triton launch costs 1.77, so ~1 us was work and
   the rest dispatch. Hand-written _skred lands at 1.85-1.91 us -- 0.08 us above
   an empty launch -- and is MORE accurate (f32 plane accumulation where torch
   rounds each partial to bf16), so it is deliberately not bit-exact against
   torch and is gated against f64 instead (kernel/p16_reduce_gate.py).
 - FP6 ROW SCALE FOLDED INTO THE KERNEL: 12.9 -> 6.4 us per layer, then 2.07-3.50
   us after tuning (kernel/p16_rowscale_tile.py) vs 2.40-4.73 unfused. Cost a day
   to one ULP: tl.max(acc)/DMAX mismatched torch on ~2/3 of rows because the AMD
   backend lowers / to multiply-by-reciprocal and 1/7.5 is inexact, which flips
   E2M3 codes for elements on a rounding boundary. Precomputing RDMAX=1.0/7.5 in
   Python restores bit-exactness. A FAST-MATH-LEGAL TRANSFORM IS NOT A NO-OP WHEN
   A DOWNSTREAM COMPARISON IS EXACT.
 - ONE FUSION MEASURED AND REJECTED (kernel/p16_fuserow.py). Fusing the fp6
   arm's two quantizer kernels is byte-exact and 0.76x at best: per decode step
   two kernels 1529 us, fused 2006 (512 blocks) / 2081 (128) / 3287 (32). The
   premise was wrong -- two launches in a CUDA graph OVERLAP, so the pair already
   ran at 3.60-4.34 us, at or under two empty serial launches: there was no
   launch to save. And the reduction must see all of K, so the program that
   quantizes must be the one that reduced, pinning the grid to one program per
   row and discarding the K-parallelism the emit pass wants. Flag off, table in
   the code, so the next person to measure 4.53 us does not rebuild it.

NOT CONFIRMED: the vLLM re-measurement never ran. The node's own nightly vLLM
 regression suite SIGKILLs any container holding a GPU and killed the sweep
 during weight load 60+ times; 0 of 6 arms measured. Two real bugs were found
 getting there (Hub 429s during engine init -> HF_HUB_OFFLINE=1; a fresh --rm
 container per attempt handing the reaper a new target -> persistent container),
 but the kill is external. Standing end-to-end numbers remain s22-era: 1.21-1.22x
 bf16 at batch 32, 1.24-1.27x at batch 8. The kernel measurements predict that
 improves; the prediction is not a measurement, and the tradeoff plot's
 throughput axis is still the pre-optimization sweep.

REGISTRY: a cell is now keyed by SCORING PROTOCOL as well as benchmark, read from
 the per-question records (perms) rather than a directory name, so mmlu_pro and
 mmlu_pro@p4 are different questions with separate records and both gate at once.
 Changing the protocol moved every number without any arm changing behaviour,
 which is the fastest way to make a registry lie. s23 logs go to
 kernel/e2e_results_v2 so neither generation overwrites the other's evidence. 64
 cells, gate passes. Also fixed: `record` used to overwrite a cell's session stamp
 even when accuracy had not moved, so any session that merely re-ran it relabelled
 the whole registry as its own work.
