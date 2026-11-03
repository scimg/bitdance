# Fused NVFP4 → MXFP6 GEMM on AMD MI350/MI355 — Progress

## Motivation

Low-bit floating-point formats trade accuracy for GEMM throughput on modern
accelerators. Two relevant OCP microscaling formats:

- **NVFP4** — 4-bit E2M1 data + a per-16/32 **E4M3** block scale. Smallest
  footprint, but lowest precision.
- **MXFP6** — 6-bit E2M3 data + a per-32 **E8M0** block scale. Noticeably better
  accuracy than FP4 at a modest size increase.

The idea explored here: **store activations (and weights) compactly as NVFP4, but
compute in the more accurate MXFP6 by transcoding NVFP4 → MXFP6 *at runtime inside
the GEMM prologue*** — fused, so no MXFP6 tensor is ever materialized in DRAM and
no separate transcode pass/kernel is needed. On CDNA4 (gfx950, MI350/MI355) the
scaled matrix instruction `v_mfma_scale_f32_16x16x128_f8f6f4` executes fp4/fp6/fp8
at the *same* rate, so MXFP6 math is "free" relative to FP4 at the MFMA level — the
only extra cost is the transcode itself.

Goal: a correct, fused NVFP4→MXFP6 GEMM on FlyDSL, and push its performance toward
the best available same-GPU GEMM baseline.

## Baselines (MI355, 4096×4096×8192, measured)

| Kernel | TFLOPS |
|---|---:|
| FlyDSL **MXFP4** preshuffle GEMM (most performant, no transcode) | **3369** |
| FlyDSL FP8 preshuffle GEMM | 1794 |
| Composable Kernel native MXFP6 GEMM | 1133 |

The MXFP4 preshuffle kernel (3369) is the target ceiling: it does no transcode and
is a mature, fully-pipelined production kernel.

## What has been done

### Phase 0 — Transcode contract (bit-exact, host)
- Derived a **64-entry** magnitude LUT (`kernel/transcode_lut.py`) that reproduces
  the reference `storage_to_mxfp6(..., "revised")` transcode bit-for-bit. Because
  NVFP4's E4M3 scale mantissa has 8 values and E2M1 has 8 magnitudes, the whole
  per-element requant collapses to a table.
- Codec `kernel/nvfp4_codec.py`: NVFP4 encode, golden MXFP6, dequant. Verified.
- Runtime contract used by the kernel:
  - `e2m3_code = LUT6[mant_bits*16 + e2m1_code]` (sign folded in)
  - `e8m0 = ((scale>>3)&0xF) + 120 + (mant_bits != 0)` (= exp + bump + 127)

### Phase 1 — Environment
- Container `nvfp4_mxfp6_emu` on MI355 (gfx950), ROCm 7.2, torch 2.9.1,
  FlyDSL 0.1.9. LUT verified bit-exact in-container.

### Phase 2 — HIP reference (bit-exact)
- Device transcode, FP6 E2M3 MFMA layout, block-scaled tile, and a fused
  multi-tile NVFP4→MXFP6 GEMM — all match the dequant-MXFP6 golden exactly.
- Key hardware fact: the scaled MFMA applies **one E8M0 per operand per lane** over
  the instruction's full K; per-32 scaling ⇒ one lane's K-slice == one MX block.

### Phase 3 — FlyDSL MFMA recipe (bit-exact probe)
- Isolated probe validated the `f8f6f4` fragment layout for (a) fp8, (b) fp6-A
  (`cbsz=2`) / fp4-B (`blgp=4`) dense packing, (c) per-lane E8M0 via `opsel` byte0.
  Recipe in `kernel/W4A6_MFMA_RECIPE.md`.

### Phase 4 — Kernels (bit-exact)
- `kernel/w4a6_gemm.py` — W_MXFP4 × A_MXFP6 (fp6 A / fp4 B), fully preshuffled.
- `kernel/w4a6_fused.py` — activation-only NVFP4→MXFP6 transcode in prologue.
- **`kernel/nvfp4_mxfp6_gemm.py`** — canonical kernel: **both operands** stored as
  NVFP4, both transcoded to MXFP6 in the prologue, scaled MFMA `cbsz=2/blgp=2`
  (fp6×fp6). Bit-exact (rel-L2 = 0) vs the dequant-MXFP6 golden; `rel-L2` vs FP32
  equals the MXFP6×MXFP6 emulation error (≈0.148 for N(0,1)) ⇒ the fused kernel
  adds **zero** error beyond quantization.
- `kernel/nvfp4_mxfp6_gemm_tiled.py` — register-blocked throughput kernel
  (`BM×BN` macro-tile/wave); also has a `fused=False` control that loads pre-packed
  fp6 directly (no transcode) for isolating transcode cost.

### Phase 5 — Integration
- `kernel/nvfp4_mxfp6_op.py` — Torch entry point
  `matmul(A_codes, A_scale, B_codes, B_scale, tile)` with a per-shape compile cache.
- `kernel/README.md` — usage, transcode contract, correctness, perf, roadmap.

## Performance (4096×4096, 128×128 tile, MI355)

Fused (both operands transcoded), by K — **K-invariant** ⇒ transcode-bound:

| K | fused TFLOPS |
|---|---:|
| 512 | 221 |
| 4096 | 227 |

No-transcode fp6 skeleton (same kernel, load pre-packed fp6), by K — **scales with K**:

| K | no-transcode TFLOPS |
|---|---:|
| 512 | 850 |
| 2048 | 1540 |
| 4096 | 1766 |
| 8192 | **1975** |

Register-block progression at K=512 (fused): 1-MFMA/wave 23 → 64×64 99 →
128×128 per-nibble-LUT 146 → 128×128 **byte-pair-LUT 221**.

**Corrected analysis (the key finding):**
- The fp6 **compute path is already fast** — 1975 TFLOPS at K=8192 (59% of the 3369
  fp4 baseline) with only a simple single-wave register-blocked kernel. The earlier
  "850" was measured at K=512 where the epilogue dominates; it is *not* a ceiling.
- The **fused kernel is transcode-VALU-throughput-bound**: its TFLOPS is K-invariant
  (≈225), i.e. the per-element NVFP4→MXFP6 transcode VALU alone sets the runtime.
  At K=8192 that's an ≈8.7× wall (1975 → 227).
- **Consequence:** speeding up the fp6 skeleton (LDS/pipelining, toward 3369) will
  *not* lift the fused number as long as transcode VALU > MFMA+mem time. The only
  levers that help the fused kernel are (a) fewer transcode VALU ops, or
  (b) fewer transcodes (pre-transcode static weights / share across waves).
- The no-transcode kernel at large K is likely **L2-bandwidth-bound** on redundant
  operand loads (each A/B strip is re-read by every block in its row/col); LDS
  staging is the fix to push it toward 3369.

## Multi-wave LDS kernels (implemented)

Two new bit-exact kernels stage operands in LDS so the NVFP4→MXFP6 transcode is
**shared across waves** (transcode once, reuse by every wave):

- `kernel/nvfp4_mxfp6_gemm_lds.py` — 4 waves share one LDS-staged A tile; each wave
  computes its own N-slice with register-loaded B.
- `kernel/nvfp4_mxfp6_gemm_lds2.py` — **both** A and B staged in LDS, cooperatively
  transcoded once per block, 2-D wave grid (auto: one 128×128 tile per wave),
  optional double-buffering.

Result (4096×4096×8192, MI355), fused both-operand:

| Kernel | fused TFLOPS | no-transcode TFLOPS |
|---|---:|---:|
| single-wave register-block (`_tiled`) | 227 | 1975 |
| **LDS stage-both 256×256, 4 waves (`_lds2`)** | **380** | 1207 |

**Fused: 227 → 380 TFLOPS = 1.67×**, bit-exact (rel-L2 = 0). Sharing each transcode
across 4 waves cut the transcode's relative cost from **8.7×** (single-wave) to
**3.2×** (1207/380).

Findings that shaped the design:
- **Total transcode VALU work = `num_blocks·(MR+NR) ∝ MN/16·(1/BM+1/BN)`** →
  minimized by bigger square blocks. But block size is bounded below by **occupancy**:
  4096² needs ≥ ~256 blocks to fill the CUs, so 512×512 (only 64 blocks) collapses
  to 54 TFLOPS. **256×256 (256 blocks) is the equilibrium** → 380.
- **Double-buffering barely helped fused** (371 vs 365) — fused is transcode-VALU-
  bound, so barrier stalls are negligible vs transcode time; overlap needs explicit
  scheduler hints, not just ping/pong.
- The no-transcode LDS path (1207) is *below* the single-wave direct kernel (1975):
  LDS round-trip + barriers cost more than the redundant-load savings when there is
  no transcode to amortize.

## Strategy going forward

1. **Scheduler-hint overlap** (`rocdl.sched_mfma/sched_dsrd/sched_vmem/sched_barrier`,
   as in the reference `blockscale_preshuffle_gemm`) to hide transcode VALU under
   MFMA. Quantified ceiling ≈ **554 TFLOPS** (= perfect overlap: `min(compute 1207,
   transcode-only 554)`). Fragile but the next real lever for the fused kernel.
2. **Algorithmic transcode reduction** (fewer VALU ops/element) — the L1-resident
   8 KB byte-pair LUT is the cache sweet spot; further cuts need a different scheme
   (e.g. register-table `v_perm` gather).
3. **Pre-transcode static weights** to halve per-GEMM transcodes (if the "both
   operands transcoded at runtime" constraint can be relaxed for weights).
4. Reaching the **3369 fp4 ceiling** requires the full production pipeline (XOR
   swizzle, async global→LDS copy, cshuffle epilogue, scheduler hints); the fp6
   compute path itself already sustains ~1975 single-wave.

## File index

| File | Role |
|---|---|
| `kernel/nvfp4_mxfp6_gemm.py` | Canonical correctness kernel (both operands fused) + golden |
| `kernel/nvfp4_mxfp6_gemm_tiled.py` | Single-wave register-blocked kernel (fused 227) |
| `kernel/nvfp4_mxfp6_gemm_lds2.py` | **Fastest fused kernel**: multi-wave LDS stage-both, transcode shared across waves (fused 380) |
| `kernel/nvfp4_mxfp6_gemm_lds.py` | Multi-wave LDS share-A variant (fused 370) |
| `kernel/nvfp4_mxfp6_op.py` | Torch `matmul(...)` entry point |
| `kernel/bench_nvfp4_mxfp6.py` | TFLOPS harness |
| `kernel/transcode_lut.py`, `kernel/nvfp4_codec.py` | Transcode LUT + bit-exact codec |
| `kernel/W4A6_MFMA_RECIPE.md` | Verified MFMA fragment/scale semantics |
| `kernel/w4a6_gemm.py`, `kernel/w4a6_fused.py` | Earlier W4A6 variants (reference) |
| `kernel/README.md` | Usage + design notes |
| `kernel/bench_sweep.py` | Multi-config tile sweep over the FlyDSL preshuffle GEMM (fp4/fp6/fp8) |
| `psg_preshuffle_gemm.py` | Local editable copy of the production preshuffle GEMM (`/root/psg`), fp6 added |

---

# W4A4 / W4A6 / W6A6 / NVFP4→MXFP6 comparison (article method)

Following the ROCm blog *"MXFP6 and MXFP4 Mixed Precision …"* (w4a6-quant-mm),
which builds W_MXFP4_A_MXFP6 by **adapting FlyDSL's MXFP4 preshuffle GEMM to a
6-bit A operand** — "a different `cbsz` and an FP8-padded A-tile (24 real bytes +
8 zero-pad per K=32) instead of dense FP4 packing".

## What was implemented

- Reused the production `compile_preshuffle_gemm_a8` (the kernel behind the 3378
  TFLOPS W4A4 baseline). Both fp4 and fp8 paths already share the same
  `mfma_scale_f32_16x16x128_f8f6f4`; they differ only in operand byte-width
  (fp4 = 16 B/lane dense, fp8 = 32 B/lane) and `cbsz/blgp` (4 vs 0).
- **Added an `in_dtype="fp6"` mode = W6A6**: identical 32 B/lane FP8-padded layout
  to fp8 (so the entire load/LDS/fragment/scale path is reused verbatim), with the
  f8f6f4 MFMA selecting the 6-bit format via `cbsz=blgp=2`. This is exactly the
  article's "FP8-padded tile + different cbsz" recipe, applied to *both* operands.

## Measured throughput (MI355X / gfx950, bf16 out)

| Config | Operand bytes (A+B) | 4096²×8192 | 8192³ | vs W4A4 |
|---|---|---|---|---|
| **W4A4** (fp4×fp4, dense 16 B) | 1.0× | **3378** | **3764** | 100% |
| **W4A6** (fp6 A 32 B + fp4 B 16 B) | 1.5× | *~2700 (est)* | *~3000 (est)* | ~80% |
| **W6A6** (fp6×fp6, FP8-padded 32 B) | 2.0× | **2009** | **2240** | ~59% |
| **NVFP4→MXFP6 fused** (fp4-stored, transcoded→fp6 at runtime) | 1.0× + transcode VALU | **380** | — | ~11% |

(fp8 reference on the same path: 1789 @ 4096²×8192 — W6A6 ≈ fp8, as expected since
both move 32 B/lane operands.)

## Key finding — why fp6 does *not* match fp4 here

MXFP6 on the f8f6f4 MFMA is **FP8-padded**: 32 elements occupy 32 bytes (24 real +
8 pad), exactly like fp8 and **2× the bytes of fp4's dense 16-byte packing**. At these
large square shapes the GEMM is **operand-memory-bound**, so:

- **W6A6 (both 32 B) ≈ 59% of W4A4** — the 2× operand traffic, not the ALU, is the
  ceiling (the f8f6f4 unit runs fp6/fp4 at the *same* MFMA rate).
- **W4A6 (A 32 B + B 16 B = 1.5×) lands between** W4A4 and W6A6 (~80% of W4A4 by the
  operand-bandwidth model, interpolating the two measured endpoints). The blog's
  "within 3.6–15.7% of MXFP4" is on **per-shape-tuned LLM matmuls** (qkv/o_proj/
  down_proj + split-K), several of which are more compute-bound than a plain square;
  the same-rate ALU means W4A6→W4A4 parity is reachable only when *not* operand-BW-bound.
- **NVFP4→MXFP6 fused (380)** stores both operands as dense fp4 (W4A4-like 16 B memory)
  and transcodes to fp6 in the prologue, so it is **transcode-VALU-bound**, not
  memory-bound — a different regime entirely (see earlier sections).

## Status of the real W4A6 kernel

W4A6 is the one config that needs a **mixed-operand** kernel: A on the fp6/fp8
32 B path *and* B on the fp4 16 B dense path within one MFMA loop. In
`compile_preshuffle_gemm_a8` the A/B byte layouts are already decoupled
(`a_elem_vec_pack` vs `b_elem_vec_pack`), but the MFMA K-iteration is written as two
monolithic branches (fp4 packed-K vs fp8 2×k64) whose schemes must be merged for the
A-32B / B-16B hybrid. That merge (plus a bit-exact host-side mixed preshuffle) is the
blog's actual kernel contribution and is the remaining work item; the W4A6 numbers
above are the operand-bandwidth interpolation of the two measured endpoints.

---

# Real Qwen3-4B layer shapes (Y = X[M,K] @ W[N,K]ᵀ)

Config: hidden=2560, intermediate=9728, 32 heads × 128, 8 KV heads.
- qkv_proj: K=2560, N=6144 · o_proj: K=4096, N=2560
- gate_up_proj: K=2560, **N=19456** · down_proj: **K=9728**, N=2560

## Real W4A6 mixed kernel (reproduced from the blog) — MEASURED

Implemented the article's **W_MXFP4 · A_MXFP6** kernel directly inside the
production preshuffle GEMM (`psg/preshuffle_gemm.py`, `in_dtype="w4a6"`), *not* a
bandwidth estimate. The mixed-precision MFMA runs **A = MXFP6 (E2M3, `cbsz=2`,
FP8-padded 32 B/lane, 24 real bytes read) × B = MXFP4 (E2M1, `blgp=4`, dense
16 B/lane)** with per-32 E8M0 block scales, via
`v_mfma_scale_f32_16x16x128_f8f6f4`.

Design that made it tractable without a full K-iteration refactor:
- Fix **`tile_k=128`** → one K=128 MFMA step, so `k_unroll=1` and the **entire fp4
  B path (load / preshuffle `layout_b` / per-32 scale / MFMA) is reused unchanged**.
- Give the A operand the **proven fp6 footprint** (`a_pack_eff=1`: 32 B/lane LDS,
  identical layout/swizzle to the working W6A6/fp8 A path), while `a_elem_vec_pack=2`
  keeps `k_unroll`/B on the fp4 schedule.
- In the fp4 MFMA branch, read **24 real A bytes** (two k64 halves 64 B apart in the
  fp6 LDS, drop the pad) and set `cbsz=2`. Only ~4 footprint sites + the A read change.

| Shape | M | W4A4 (fp4) | W6A6 (fp6) | **W4A6 (measured)** | best tile | W4A6/W4A4 |
|---|---|---:|---:|---:|---|---:|
| square 4096²×8192 | 4096 | 3378 | 2009 | **2996** | 128×256×128 | **89%** |
| gate_up (N=19456,K=2560) prefill | 4096 | 3035 | 1760 | **2087** | 128×256×128 | 69% |
| down_proj (N=2560,K=9728) prefill | 2048 | — | — | **1665** | 128×256×128 | — |
| down_proj (N=2560,K=9728) decode | 128 | 199 | 130 | **166** | 32×256×128 | 83% |

**W4A6 beats W6A6 on every shape** (the weight-bandwidth win is real, as predicted),
and reaches 83–89% of W4A4 on compute-dense (square) and weight-dominated (decode)
shapes — matching the blog's "close to MXFP4" claim. The remaining gap vs the square
case on wide-N prefill (gate_up) comes from the **doubled A-LDS** (fp6 activation)
forcing `tile_m=128` and limiting occupancy at large `tile_n`; small `tile_m`
(32–64) recovers most of it at decode.

Note `tile_k` is pinned to 128 for W4A6 (one MFMA K-step); `tile_m=16` is invalid
(doubled A footprint → `bytes_per_thread_a` not divisible by 16).

### Correctness status (`kernel/verify_w4a6.py`)

Uses constant-value, constant-along-K operands with all E8M0 scales = 1, so
`C[m,n] = K·a·b` regardless of the internal preshuffle.

- **Single-wave MFMA recipe** (`kernel/w4a6_gemm.py`, fp6-A × fp4-B, `cbsz=2/blgp=4`,
  per-32 E8M0): **PASS** (bit-exact vs numpy) — validates the instruction-level math.
- **Production kernel, shared fp4 path** (`--dtype fp4`, a=1.5·b=2.0): **PASS**,
  `max|err|=0` — validates the harness + fp4-B load/`layout_b`/scale/full-K accum
  that W4A6 reuses unchanged.
- **Production kernel, fp6-A** (`--dtype w4a6`): **not yet bit-exact from host.**
  fp6 is dense-6-bit (1-byte/elem gives garbage: −128), so 32 codes → 24 packed
  bytes. That 3-byte packing period does **not** align to the kernel's 16-byte
  A-store/swizzle granularity, so a host-contiguous `[24 packed][8 pad]` block is
  scrambled by the swizzle (result 1928 + zeros instead of 6144).

**Remaining work for end-to-end fp6-A verification:** emit the fp6 A global bytes
in the exact order the kernel's `store_a_tile_to_lds` + `swizzle_xor16` +
`lds_load_packs_k64` (col, col+64) chain consumes them — i.e. mirror the kernel's
A layout math on the host (or add a tiny device "layout probe"). This same fp6-A
global layout is used by the pre-existing **W6A6/fp6** path, which was likewise
never numerically verified, so this closes both. The MFMA math and the entire
fp4-B half are already proven, so risk is confined to this one byte-ordering.

## The controlling principle (answers "why W4A6 ≈ W4A4")

In an LLM linear layer the **weight `W[N,K]` dominates operand memory**; the
**activation `X[M,K]` is small** (tiny at decode, ≤ weight at prefill).

- **W6A6** makes the *weight* fp6 (32 B, 2× fp4) → doubles the dominant term →
  **stuck at ~58–65% of W4A4 on every real shape.** (So the earlier hypothesis
  "W6A6 ≈ W4A6" is **false** — W6A6 pays the 2× weight tax.)
- **W4A6** keeps the weight fp4 (16 B) and only inflates the *activation* to fp6 →
  penalty scales as `M/(M+N)`: **~5% at decode (M≪N), ~15% at prefill M=4096.**
  This is exactly the blog's "within 3.6–15.7% of MXFP4", and it improves further
  with split-K on small-M decode shapes (more CTAs → fills the GPU).

So the right way to get **W4A6 ≈ W4A4 is to pick weight-dominated shapes**
(decode, or high-intensity prefill), which is precisely how the blog operates.

## NVFP4→MXFP6 fused transcode on real Qwen shapes (current `lds2` kernel)

| Layer | M | fused transcode | W4A4 ref | vs W4A4 |
|---|---|---|---|---|
| gate_up (N=19456,K=2560) prefill | 4096 | **320** | 3035 | 11% |
| down_proj (N=2560,K=9728) prefill | 2048 | **114** | — | — |
| down_proj (N=2560,K=9728) decode | 128 | **5.8** | 199 | 3% |

The fused kernel is **transcode-VALU-bound** and, at decode (M=128), also
**CTA-starved** (only ⌈M/bm⌉·⌈N/bn⌉ ≈ 10 workgroups) — the same problem the blog
fixes with **split-K**. These are the numbers to beat in the transcode-optimization
phase.

## Weight-only transcode (NVFP4 weight → MXFP6, native MXFP6 activation)

**Idea (user):** since the *weight* dominates operand bandwidth, transcode **only the
weight** NVFP4→MXFP6 on-chip and keep the **activation native MXFP6** (no transcode).
This gives **MXFP6-compute accuracy at fp4-weight DRAM footprint**, and — because
NVFP4 uses an E4M3 block scale — the transcode folds the E4M3 mantissa into the
E2M3 data, so the weight is **NVFP4-faithful** (strictly better than the MXFP4/E8M0
weight of the W4A6 kernel at the same 4-bit footprint). Only worthwhile if the weight
is genuinely NVFP4 (E4M3); an MXFP4/E8M0 weight would gain nothing from the transcode.

Implemented as `weight_tc` in `kernel/nvfp4_mxfp6_gemm_lds2.py`: A staged via
`load_fp6` (native MXFP6), B via `transcode` (NVFP4→MXFP6, 64-entry LUT). **Bit-exact
(`rel_L2 = 0`, PASS)** vs the dequant reference.

lds2 (same kernel → isolates transcode cost):

| Shape | native fp6 (no tc) | **weight-only tc** | both-operand tc |
|---|---:|---:|---:|
| square 4096²×8192 | 1201 | **588** | 380 |
| gate_up 4096×19456×2560 | 912 | **475** | 321 |

**Weight-only transcode is +48–55% over transcoding both** — confirms the instinct:
skip the activation transcode. It still costs ~2× vs native fp6 on this *un-pipelined*
kernel (transcode VALU not hidden behind MFMA).

Caveat: lds2's absolute throughput is low (its native-fp6 ceiling 912–1201 is well
below the **production W6A6 = 2009 / W4A6 = 2996**), so these are *relative* numbers.
A production headline vs W4A6/W6A6 needs the weight-transcode ported onto the
production preshuffle path (per-32 E8M0, i.e. the W4A6 base — *not* the per-channel
W6A6 path, whose scale scheme is incompatible with the transcode's E8M0 output).
That port adds a LUT arg + LDS-staged LUT + B fp4→fp6 transcode in the load prologue.

## Production weight-transcode (`in_dtype="w6a6_tc"` in `psg/preshuffle_gemm.py`)

Ported onto the W4A6 base: A = native MXFP6 (cbsz=2); B stored NVFP4 (fp4 16 B/lane
+ per-32 E4M3 scale), transcoded **in-kernel** to MXFP6 via a 12-bit LUT
(`build_LUT12`, read from a new `arg_lut` global, L1-cached), then fp6×fp6 MFMA with
`blgp=2`. E4M3→E8M0 scale + fp4→fp6 data transcode are **position-preserving**, done
**once per tile** (hoisted above the `mi` loop). **Verified**: `w6a6_tc` output is
**bit-exact** to `w4a6` with the same A and equivalent B (fp4 code→fp6 code via LUT,
E4M3 scale=1.0→E8M0=1.0) — i.e. the transcode plumbing is correct. (The constant-value
*absolute* check still FAILs identically for both, from the pre-existing fp6-A host
swizzle-mirror gap; that's orthogonal.)

### Optimization arc (each step verified bit-exact vs W4A6; snapshots in `opt_snapshots/`)

| Stage | square TFLOPS | vs W4A6 | key change |
|---|---:|---:|---|
| Phase 0 — naive in-compute-loop, **global** LUT | 508 | 0.18× | global LUT loads steal L2/vmem; transcode serialized |
| Phase 1 — **LDS-resident LUT** (coop global→LDS copy) | 1619 | 0.57× | ds_read LUT; frees the vmem path (esp. decode) — **3.2×** |
| Phase 2 — micro-opt (row-fold LDS view, mask elision) | 1687 | 0.61× | −16 index ORs/block, −top-byte mask — **+5%** |

**Profiling (rocprofv3, square, decisive):** MFMA & VMEM insts are *identical* to W4A6
(same compute, same fp4-B bandwidth); the transcode adds **4.89× VALU** (202M→988M) +
3.5× LDS. Raising occupancy (smaller tiles, `waves_per_eu`) does **not** help ⇒ the VALU
pipe is **saturated**. FlyDSL VALU micro-opt is exhausted; further cuts need asm
`v_perm_b32`, which still cannot beat a **zero-transcode** baseline in compute-bound.

### Definitive crossover (LDS-LUT + micro-opt, best tile; warmup 25 / 200 iters, 2 repeats, <1% spread)

| Shape (M×N×K) | W4A4 | W4A6 | W6A6/fp6 | **W6A6_TC** | TC vs W4A6 |
|---|---:|---:|---:|---:|---:|
| square 4096²×8192 (compute)      | 3359 | 2760 | 1796 | 1687 | 0.61× |
| gate_up 4096×19456×2560 (prefill)| —    | 2127 | 1647 | 1501 | 0.71× |
| **down decode 128×2560×9728**    | —    | 240  | 116  | **310**  | **+29%** |
| **gate decode 128×19456×2560**   | —    | 1030 | 678  | **1548** | **+50%** |
| **qkv  decode 128×4096×4096**    | —    | 347  | 200  | **433**  | **+24%** |

**Finding — the bandwidth hypothesis is CONFIRMED (earlier "refuted" verdict was an
artifact of the un-optimized global-LUT kernel).** With the LUT in LDS, `w6a6_tc`
**beats W4A6 on every decode (memory-bound) shape by 24–50%**, the deployment-relevant
LLM-serving regime, at **NVFP4-faithful accuracy and a 4-bit (16 B/lane) weight
footprint**. Confirmed **bit-exact == W4A6** (same A) at the exact winning decode tiles
(`kernel/verify_decode.sh`, cross-check `True`) ⇒ the win is real math, not under-compute.

Why it works / where it doesn't (rocprofv3, decode tiles 128×64, `kernel/prof_decode.sh`):
- **Decode: tc wins, and the mechanism is measured — NOT bandwidth.** vs W4A6 at decode:
  `FETCH_SIZE` (DRAM bytes) **identical** (±2%), `SQ_INSTS_VALU` **identical**, `SQ_WAVES`
  identical — but `SQ_INSTS_LDS` is **0.51×** (tc issues HALF the LDS instructions), and
  wall cycles track that (0.82–0.87×). FETCH is only ~30 KB/dispatch ⇒ decode is **not
  HBM-bound**; at M=128 the launch is tiny so it's **instruction-issue/latency/occupancy
  bound**. There the transcode VALU is free (compute idle on latency; VALU count unchanged),
  and tc's advantage is purely **fewer LDS ops**: it transcodes each B tile *once into
  registers* and its symmetric fp6×fp6 MFMA consumes B from registers, avoiding W4A6's
  heavier LDS-operand traffic (mixed fp6-A×fp4-B: 32 B fp6-A read as two k64 + dense fp4-B
  from LDS each step). Issue-bound ⇒ halving LDS directly cuts cycles → the 24–50% win.
  (The earlier "2× fp4-weight bandwidth over W6A6" reasoning was a coincidence for the
  tc-vs-w4a6 case; both move identical DRAM, so BW is not the differentiator.)
- **Compute-bound (square/prefill): 0.61–0.71×.** Opposite regime — VALU IS saturated by
  the transcode (profiled 4.89× VALU insts), and a **zero-transcode** kernel (W4A6) is a
  hard floor no transcode kernel can cross here.

**Recommendation (revised):** DO use runtime NVFP4-weight→MXFP6 transcode for
**memory-bound / decode serving** — it is the fastest low-bit option there (beats W4A6),
keeps the 4-bit weight DRAM footprint, and preserves the NVFP4 E4M3 scale mantissa
(strictly better accuracy than an E8M0/MXFP4 weight at the same size). For **compute-bound
prefill**, prefer W4A6 (no transcode). A serving stack can dispatch on M: `w6a6_tc` for
small-M decode, `w4a6` for large-M prefill.

---

# W4A4 baseline added + FLUX.1-schnell shape study (`kernel/bench_all.sh`)

Added **W4A4 (`in_dtype="fp4"`, fp4×fp4 dense)** to every benchmark row as the throughput
ceiling. How close fp4 sits to its own square peak (~3290 TFLOPS this run) is used as an
**empirical compute-bound classifier** (near peak ⇒ compute-bound; far below ⇒
memory/latency/occupancy-bound), which is more honest for small-M than pure roofline.

## FLUX.1-schnell architecture → GEMM shapes
`dim=3072` (24 heads × 128), 19 dual-stream + 38 single-stream blocks, `mlp=12288`,
text seq=256, guidance off. At **1024×1024** the image latent patchifies to **4096 tokens**;
single-stream runs the joint **4352** (=4096 img + 256 text). Diffusion has **no decode
regime** — every denoising step processes the full sequence ⇒ M is always large except the
text stream. Representative linear layers (weight `[N,K]`, activation `[M,K]`):

| Layer | M | N | K |
|---|---:|---:|---:|
| dual img QKV (fused) | 4096 | 9216 | 3072 |
| dual img attn-out | 4096 | 3072 | 3072 |
| dual img MLP fc1 | 4096 | 12288 | 3072 |
| dual img MLP fc2 | 4096 | 3072 | 12288 |
| single linear1 (qkv+mlp_in) | 4352 | 21504 | 3072 |
| single linear2 (attn+mlp_out) | 4352 | 3072 | 15360 |
| text-stream fc1 (small-M) | 256 | 12288 | 3072 |

Roofline (fp4, 0.5 B/weight): AI = 2MNK/(NK·0.5) = **4M** FLOP/B; MI355 ridge ≈ peak/HBM ≈
3290 TF / 8 TB·s⁻¹ ≈ **411 FLOP/B**. M=4096 ⇒ AI≈16k (≈40× ridge, deeply compute-bound);
M=256 ⇒ AI≈1k (2.4× ridge by arithmetic, but only **~12% of fp4 peak measured** ⇒ in
practice **occupancy/latency-bound** — too few workgroups to fill the GPU).

## Measured (MI355, best tile, TFLOPS)

| FLUX shape (M×N×K) | fp4 (W4A4) | W4A6 | fp6 (W6A6) | **w6a6_tc** | fp4 %peak | bound | best accurate |
|---|---:|---:|---:|---:|---:|---|---|
| dual QKV 4096×9216×3072 | 2936 | 2163 | 1658 | 1494 | 89% | compute | **W4A6** |
| dual PROJ 4096×3072×3072 | 2715 | 1950 | 1549 | 879 | 83% | compute | **W4A6** |
| dual FC1 4096×12288×3072 | 2627 | 1931 | 1631 | 1413 | 80% | compute | **W4A6** |
| dual FC2 4096×3072×12288 | 2386 | 1801 | 1524 | 1245 | 73% | compute | **W4A6** |
| single L1 4352×21504×3072 | 2871 | 2100 | 1566 | 1545 | 87% | compute | **W4A6** |
| single L2 4352×3072×15360 | 2413 | 2038 | 1456 | 1078 | 73% | compute | **W4A6** |
| **text fc1 256×12288×3072** | 388 | 266 | 259 | **329** | 12% | **latency** | **w6a6_tc (+24%)** |

## Conclusion for FLUX
- **FLUX.1-schnell at 1024px is overwhelmingly compute-bound** (M=4096/4352, fp4 at
  73–89% of peak). In that regime runtime transcode pays an unhideable VALU tax, so
  **W4A6 is the right accurate low-bit kernel for FLUX**; `w6a6_tc` trails (0.45–0.74×),
  worst on the low-N (N=3072) proj/down layers where there's little N to amortize the
  transcode.
- **`w6a6_tc` only helps the small-M text-stream projections** (M=256, latency-bound,
  +24%), a small fraction of total FLUX FLOPs.
- This mirrors the LLM result exactly and confirms the single controlling variable is
  **M (the token/compute dimension), not the model**: large-M ⇒ compute-bound ⇒ W4A6;
  small-M ⇒ latency-bound ⇒ w6a6_tc. Diffusion (always full-sequence) sits on the large-M
  side; autoregressive LLM decode (M≈batch) sits on the small-M side.

## 2026-07-11 — Unified 5-kernel benchmark + HIP v31 rasterization + M=256 correction

Re-ran **all five kernels on one pass** (`kernel/bench_full.sh` → `kernel/bench_full_out.txt`)
at low node load (load avg ~1.7). Kernels: FlyDSL `fp4/w4a6/fp6/w6a6_tc` **plus the
hand-written HIP kernel** `kernel/hip/w4a6_tc_gemm_v31.hip` (BM=1024 prefill, **CTA M-major
rasterization**, GROUP=8). Both harnesses report `2·M·N·K/t`, so numbers are directly
comparable. HIP is `n/a` where M%1024≠0 (decode, M=4352, M=256).

**w4a6_tc HIP (v31) vs w4a6_tc FlyDSL**, on the large-M shapes HIP can serve:

| shape | tc-FlyDSL | tc-HIP | Δ |
|---|---:|---:|---:|
| LLM square 4096×4096×8192 | 1739 | **2001** | **+15.1%** |
| LLM gate_up 4096×19456×2560 | 1597 | 1614 | +1.1% |
| FLUX dual FC2 4096×3072×12288 | 1550 | **1743** | **+12.5%** |
| FLUX dual FC1 4096×12288×3072 | 1682 | **1793** | **+6.6%** |
| FLUX dual imgPROJ 4096×3072×3072 | 1355 | 1385 | +2.2% |
| FLUX dual imgQKV 4096×9216×3072 | 1504 | 1436 | −4.5% |

The HIP kernel wins 5/6 large-M shapes; grouped-M rasterization improves L2 reuse of the
shared B tiles (each M-group replays the same N-columns), directly attacking the B-ingress
latency. It reaches **0.70–0.79× of W4A6** on compute-bound shapes vs the FlyDSL transcode's
0.53–0.73× — still under the zero-transcode W4A6 floor (VALU tax is unhideable when the MFMA
pipe is full), but a meaningful narrowing.

**Correction — FLUX text M=256 is compute-bound, not latency-bound.** The earlier table
(text fc1: w6a6_tc 329 "+24%") was taken under node contention. At full clocks (3 reps, 300
iters, <1% spread): **fp4 1269 · W4A6 908 · fp6 877 · w6a6_tc 623** — W4A6 wins, tc loses.
So FLUX @1024px is compute-bound *everywhere* and W4A6 is the accurate choice for all of it;
the genuine tc win is the **LLM decode regime (M=128)**, which reproduced to <1% (GATE
1522 vs W4A6 1016 = +50%, QKV 428 vs 344 = +24%, DOWN 308 vs 239 = +29%). The controlling
variable is still M, but the crossover sits at M≈128 (true decode), not M≈256.

## 2026-07-11 — HIP v31 GROUP rasterization sweep → new default GROUP=2

Swept the CTA-rasterization `GROUP` (grouped-M tile ordering) over `{2,4,8,16,32}` on the
six large-M shapes (`kernel/seq_group_sweep.sh`, MI355 dev6, best-of-3, 200 iters). TFLOP/s:

| shape | G=2 | G=4 | G=8 | G=16 | G=32 |
|---|---:|---:|---:|---:|---:|
| LLM square 4096×4096×8192 | 1983 | **1987** | 1986 | 1986 | 1986 |
| LLM gate_up 4096×19456×2560 | **1599** | 1594 | 1596 | 1595 | 1598 |
| FLUX imgQKV 4096×9216×3072 | **1432** | 1432 | 1429 | 1431 | 1429 |
| FLUX imgPROJ 4096×3072×3072 | **1381** | 1360 | 1355 | 1358 | 1356 |
| FLUX FC1 4096×12288×3072 | **1784** | 1779 | 1780 | 1779 | 1781 |
| FLUX FC2 4096×3072×12288 | 1720 | 1739 | **1752** | 1743 | 1731 |

Two findings:
1. **GROUP=2 is the best single default** — it wins square(≈tie)/gate_up/imgQKV/imgPROJ/FC1
   and loses only ~1.8% on FC2. Changed the kernel default `#define GROUP 8 → 2`
   (`kernel/hip/w4a6_tc_gemm_v31.hip`); still **bit-exact** (verify maxabs=0, nbad=0). This
   recovers the earlier imgQKV number to ~1432 and lifts imgPROJ/FC1.
2. **The imgQKV "regression" is not a rasterization problem** — throughput is flat (~1430)
   across *all* GROUP values. Grouped-M ordering is already saturating L2 reuse of B; the
   remaining gap is **B-ingress / mainloop-barrier bound**, not locality-bound.

**Conclusion / next lever.** Rasterization tuning is exhausted (≤2% spread). The mainloop
issues one `__syncthreads()` per k-tile between the MFMA and the NVFP4→MXFP6 transcode
(`packto`) that fills the next shared buffer; that barrier + transcode tail is the critical
path. The next optimization is **wave-specialization / deeper ping-pong** (dedicated
producer waves running `prefetch`+`packto` while consumer waves stay in MFMA) to hide the
barrier and the VALU transcode behind the matrix pipe — the only structural way left to
close on the zero-transcode W4A6 floor.

## 2026-07-11 — "beat-W4A6" deep dive: the base-ceiling reframe (full detail in `kernel/opt_log.md` Session 4)

Decomposed FULL / DUMMY(ds_write only) / NOPACK(base ceiling) for the HIP kernel on SQUARE
4096²×8192 (MI355 dev6, SOLO best-of-8; W4A6 ref = 2852). Two long-standing assumptions
were **wrong**: (a) `-Rpass=kernel-resource-usage` shows **AGPRs=0** — accumulators are in
the 256 VGPRs of the unified 512-wide file, not a separate AGPR bank; (b) every 8-wave
config runs **1 CTA/CU** (2 waves/SIMD), so the transcode was never hidden by a 2nd CTA.

| config | NOPACK (base) | DUMMY (+ds_write) | FULL | ds_write gap | pack gap |
|---|---:|---:|---:|---:|---:|
| BM1024×BN64 (production) | 2454 | 2086 | 1988 | 368 | 98 |
| **BM512×BN128** | **3126** | 2279 | 1868 | **847** | **411** |

**The reframe:** prior work compared the base ceiling to *pure-fp6 W6A6 (1886)* and concluded
"the base GEMM is fine." Against the real target **W4A6 (2852)**, the BN=64 base ceiling
(2454) is itself **14% below W4A6** — so *no* transcode kernel can beat W4A6 at BN=64. Only
**BN=128 lifts the base ceiling above W4A6 (3126, +10%)** by halving A-loads/FLOP, making
BN=128 a **necessary condition**. The cost: BN=128 doubles the B-transcode per unit MFMA at
fixed 1-CTA/CU occupancy, so the transcode goes from ~19% hidden (BN=64) to ~40% exposed
(gap 1258, ds_write-dominated). Route to beat W4A6: BM512×BN128 + (1) i32×6 B storage
(−25% ds_write/read), (2) **true 8-wave ping-pong** to overlap ds_write+pack behind MFMA
(now high-ROI, not low as previously thought), (3) v_perm pack. Occupancy can't be raised
(acc alone needs 128 VGPR/wave), so overlap must be intra-CTA.

## 2026-07-13 — Counter-driven diagnosis (rocprofv3, gfx950): the real bottleneck is the synchronous pipeline

Guided by the ROCm HIP performance guide ("measure the bottleneck with counters, don't
infer"), I pulled hardware counters instead of wall-clock probe deltas. **This corrects the
earlier "barrier tax 22–28% / occupancy artifact" conclusion** (it was inferred from
`PROBE_BAR` timing; the counters show the barrier *instruction* is 4.7% SALU). Square prefill
(`kernel/prof_run.sh`, `prof2.sh`):

| metric | value | reading |
|---|---:|---|
| VGPRs / Occupancy | 255 / 2 waves/SIMD | 1 CTA/CU (VGPR-limited) |
| `MfmaUtil` | ~20% | MFMA pipe idle ~80% |
| `OccupancyPercent` | ~20% | **MfmaUtil tracks occupancy exactly** |
| `SALUBusy` | 4.7% | the `s_barrier` is **not** the cost |
| `SQ_INSTS_VALU : SQ_INSTS_MFMA` | 3.6 : 1 | transcode is VALU-heavy |
| `SQ_VALU_MFMA_COEXEC / BUSY` | ~6% | VALU and MFMA run **serially** |
| `SQ_LDS_BANK_CONFLICT / LDS_LOAD` | ~7.9 | real 8-way B conflict, but off critical path |

Five new levers added behind `#ifdef` in `w4a6_tc_gemm_v31.hip` (production build unchanged),
each **bit-exact** (`maxabs=0`) where run, each **measured** — all flat or worse:

| lever (probe) | Δ perf | rules out |
|---|---:|---|
| **XPOSE** — conflict-free transposed B LDS `[buf][bblk][dword][lane]` (`xpose_probe.sh`) | −1..+1.3% | LDS bank conflict (killed the counter, no speedup) |
| **APIPE** — software-prefetch A one `mi` ahead, B register-resident (`apipe_probe.sh`) | −0.3..+0.9% | A global-load latency |
| **ILVPACK** — issue the pack inside the MFMA loop (`ilvpack_probe.sh`) | −0.6..+0.9% | VALU/MFMA co-execution (in-order issue) |
| **occupancy sweep** — BM/WM for ≥2 CTA/CU (`occ_sweep.sh`) | worse; MfmaUtil ↓ | occupancy (256×4: occ 23.7% but util 15.1%, 1513 vs 1996) |
| **DUMMYPACK** — drop the entire transcode arithmetic (`tax_decide.sh`) | **+2..+4% only** | transcode VALU (DUMMY imgQKV 1514 still < W6A6 1673) |

**Conclusion (evidence-based).** MFMA utilization is pinned at ~20% by the **synchronous
pipeline structure** — `prefetch B → LDS store → __syncthreads → LDS read → MFMA`, one
full-CTA barrier per K-tile at 1 CTA/CU. It is **not** transcode VALU (~4%), **not** bank
conflicts, **not** A latency, **not** occupancy (more waves lowers MfmaUtil — the 1024×64 tile
is already at the GEMM low-occupancy sweet spot the HIP guide describes), **not** the barrier
instruction. Removing transcode entirely still loses to W6A6 on huge-N, so the gap is FlyDSL's
better GEMM pipeline, not our transcode.

**The candidate lever was a barrier-free async-LDS pipeline** (`buffer`/`global_load_lds`
HBM→LDS + `s_waitcnt` multi-buffer, no `__syncthreads`). Before committing to that multi-day,
high-risk rewrite (a prior ad-hoc `global_load_lds` attempt was incorrect + slower), I ran a
**cheap de-risk experiment** to measure its ceiling.

### NOBAR de-risk (2026-07-13, `kernel/nobar_probe.sh`): the barrier-free ceiling is small

`#ifdef NOBAR` drops the per-K-tile `__syncthreads()` from the production loop — output is
**incorrect** (races the double buffer), run purely to read the barrier-free perf / MfmaUtil
ceiling (BM1024×BN64, best-of-3):

| shape | baseline | NOBAR (wrong) | Δ |
|---|---:|---:|---:|
| square 4096²×8192 | 1992 | 2116 | **+6%** |
| imgQKV 4096×9216×3072 | 1457 | 1619 | **+11%** |
| FC2 4096×3072×12288 | 1766 | 2018 | **+14%** |

NOBAR `MfmaUtil` = **23.3%** (vs baseline ~20%). So **removing the barrier entirely lifts
perf only +6–14% and MFMA util only ~3 points** — the barrier idle is *not* the dominant cap;
the low MFMA duty is mostly the synchronous LDS-read→MFMA dependency chain + operand delivery,
which persists barrier-free.

**Decision — async rewrite shelved (low, mostly-unreachable ROI).** Two reasons: (1) the
absolute ceiling is only +6–14%; (2) a *correct* async pipeline **cannot even reach it** — B is
shared across all 8 waves of the CTA, and cross-wave LDS visibility fundamentally needs a CTA
`__syncthreads()` (`s_waitcnt` is per-wave only). Async `global_load_lds` can overlap only the
**HBM→LDS load** portion, which `taxbreakdown.sh` already measured at ~4–5% (`+PF`). So the
realistic async gain is low single digits, at the cost of a multi-day rewrite that risks the
bit-exact production kernel. **The HIP kernel is at its architectural optimum** for the shared-B
cooperative-pack design; the remaining gap to FlyDSL/W6A6 on compute-bound huge-N is the
mathematical fp6-operand-delivery cap (see "Why no transcode kernel can beat W6A6"), not a
tuning miss. `NOBAR` is kept behind `#ifdef` as a documented diagnostic (production unchanged).

## 2026-07-15/16 — Phase 4: XCMPB + SCHED + exhaustive pipeline exploration (v41/v42)

Session goal (user): beat W6A6 AND W4A6 everywhere. Studied CDNA4 ISA, HIP perf guide,
and the ROCm low-latency-GEMM blog. Git-tracked every step for rollback.

### Delivered (all bit-exact, in production dispatch)
- **XCMPB** (`w4a6_tc_gemm_v41.hip`): combined XPOSE+COMPACTB B-LDS layout
  `[buf][bblk][6][64]` — conflict-free (dword-major) AND compact (24 B/lane, no pad).
  +10% over the old XPOSE at BN=128 (square 1868→2055). Saves 8 KB LDS.
- **SCHED** (`v42`): `__builtin_amdgcn_sched_group_barrier` to co-issue transcode VALU
  with MFMA. +1–1.5% (gate_up 1705→1725, FC2 1839→1865).
- **ADDR32** (`v42`): 32-bit incremental A-load addressing (ISA showed ~659 of ~800 VALU
  insts were 64-bit address math). +0.5–2% on square/gate_up/LIN1/LIN2.

### Definitive best-of-30 result (`kernel/bench_final_bestof30.txt`)
| shape | tc_BEST | W6A6 | vs W6A6 |
|---|---:|---:|---:|
| square | 2118 | 1901 | **+11.4%** |
| FC2 | 1893 | 1833 | **+3.3%** |
| LIN1 | 1816 | 1742 | **+4.2%** |
| LIN2 | 1959 | 1689 | **+16.0%** |
| gate_up | 1735 | 1769 | −1.9% |
| imgQKV | 1537 | 1659 | −7.4% |
| imgPROJ | 1483 | 1562 | −5.1% |

**Beats W6A6 on 5/7 compute shapes.** Decode (M=128) still beats W4A6 by +24–50% (tc-FlyDSL).

### Levers tried and REJECTED (all bit-exact where correct, kept behind `#ifdef` for trace)
| lever | result | why |
|---|---|---|
| WAVEPIPE (producer/consumer waves) | ❌ deadlock | CDNA4 no intra-CTA wave-interleave guarantee; consumer spin starves producer |
| VPERM / VPERM2 (`v_perm_b32` table) | ❌ 13–37000× slower | v_perm+sign-fold VALU > LUT shift-pack; LDS-LUT reads never the bottleneck |
| KTILE2 (2 K-tiles/barrier) | ❌ 60–65% | 2× pack per barrier on the critical path |
| STAGE3 (register-ring deep pipeline) | ❌ −16% | raw-fp4 register ring bloats VGPR past the 128-VGPR-acc occupancy floor |
| GLDS (`global_load_lds` async) | ❌ −9% | intrinsic works per-lane 16 B, but raw→LDS→transcode→LDS doubles LDS traffic |
| WM=16 (BM=1024, halve pack redundancy) | ❌ ~180 TF | 16 waves exceed SIMD register capacity |
| APXILV / PREPACK (reorder packto vs MFMA) | ❌ flat/worse | MfmaUtil unchanged; compiler already schedules the double-buffer overlap |
| split-K on compute shapes | ❌ worse | atomic-C accumulation cost; grids already fill the GPU |

### Definitive profiling (rocprofv3, v42 gate_up)
`MfmaUtil 18.5%`, `VALUBusy 27%`, `MemUnitStalled 0.3%`, `VGPR 252 / occ 2 waves/SIMD / spill 0`.
**Both compute pipes idle → latency/barrier-bound at 1 CTA/CU, not throughput-bound.**

### Architectural ceiling (exhaustively proven)
1. Occupancy is pinned at 1 CTA/CU: `acc[MPW][NPW]` = 128 VGPR = output_tile·4 B / 512 threads.
   Reaching 2 CTA/CU needs a smaller tile, which raises the pack tax (~1/BM) more than the
   occupancy gain helps (256×128 = 1437 vs 512×128 = 2118 on square).
2. The per-K-tile `__syncthreads` (required for shared-B cooperative pack) serializes
   pack→MFMA with no 2nd CTA to fill the gap. NOBAR ceiling is only +6–14% and incorrect.
3. **W4A6 is unreachable by any transcode on compute-bound shapes**: NVFP4's E4M3 scale has a
   mantissa that must fold into the B *data* → B becomes fp6 (32 codes in 6×i32), i.e. 2× the
   LDS bytes of W4A6's fp4 B. The fused constraint forces per-CTA re-transcode (pre-transcode
   to DRAM would just be W6A6). imgQKV: our native fp6 (1646) already ties FlyDSL W6A6 (1659),
   so even a zero-tax transcode cannot beat W6A6 there.

**Conclusion:** v42 (XCMPB+SCHED+ADDR32, 512×128) is the architectural optimum of the fused
single-pass NVFP4→MXFP6 GEMM. It beats W6A6 on 5/7 compute shapes and wins decode vs W4A6 by
24–50%. Beating W4A6 on compute-bound shapes is mathematically precluded by the fp6 operand
delivery requirement, not a tuning miss. All negative results retained behind `#ifdef` for
rollback and future reference.

## 2026-07-16/17 — Phase 5–7: deep-LDS PIPE + bf16 output milestone

All levers behind `#ifdef` in `kernel/w4a6_tc_gemm_v42.hip` (no new file). Full trace in
`kernel/phase4_breakdown.md`; results in `kernel/bench_full_v43*.txt`.

- **PIPE (deep LDS ring, `-DPIPE -DNSTG=3`) — KEPT.** Discovery: CDNA4 LDS is **160 KB/block**
  (not the 64 KB previously assumed), so a 3-stage transcoded-fp6-B ring (~36 KB) fits. Combined
  with `s_setprio(1)` around the MFMA and `sched_barrier` overlap shaping: **+2..23%**. The packed
  stage is `NSTG-1` k-tiles old ⇒ fully ready ⇒ no stall on in-flight pack.
- **bf16 output (`-DOUTBF16`) — KEPT, cheapest big win.** Emits bf16 C (fp32 accumulate stays
  bit-exact). Halves C-write HBM traffic — decisive on huge-N where C is a large traffic fraction.
  **Flips all 8 compute shapes over both W6A6 and FlyDSL**; the last fp32 holdouts imgQKV/imgPROJ
  (−4%/−2% in fp32) now win.
- **Symmetric ceiling (both directions proven):** compute-bound HIP 2267 > FlyDSL-fp6 1949;
  memory-bound decode FlyDSL 1524 > best HIP decode 619. 11 from-scratch HIP decode kernels all
  plateaued ≤619 — decode needs FlyDSL's signal-based split-K, not atomics (split4 → 60 TF).

## 2026-07-17 — Phase 8/8b: CDNA4 ISA instruction scan (read the 608-pp ISA)

- **Alternative MFMA ops — no kernel change.** Non-scaled `v_mfma_f32_16x16x128_f8f6f4` has no
  gfx950 clang builtin and identical cycle count (identity scale already used ⇒ zero-cycle gain
  from inline asm). `v_mfma_scale_f32_32x32x64_f8f6f4` microbenched **6943 vs 8309 TFLOP/s** for
  16x16x128 (longer per-instr latency under dependency), same MAC/cycle & VGPR density ⇒ large
  rewrite for zero gain. **`v_mfma_scale_f32_16x16x128_f8f6f4` is the optimal MFMA here.**
- **HW scale-convert (`v_cvt_scalef32_pk32_fp6_f16` …) — blocked by the bit-exact contract.**
  Emits the 6-dword fp6 fragment directly (would replace the whole LUT + bit-pack) but does real
  float `fp4→f16→×scale→fp6 RNE` arithmetic, which mismatches the reference "revised" MXFP6 LUT for
  any non-zero E4M3 scale mantissa (28/32 codes differ). Arithmetically valid but a *different
  numerical contract* — documented as a future option only if bit-exactness is relaxed.

## 2026-07-22 — Phase 9: Primus-Turbo PR410/424 techniques

Studied AMD-AGI/Primus-Turbo **PR#410** (grouped MXFP8 GEMM) and **PR#424** (MXFP4 GEMM), both
FlyDSL gfx950 backends. Only one technique maps to the fused in-kernel-transcode design:

- **XCD-aware CTA→tile swizzle (`-DXCDSW`).** Verified via `s_getreg hwreg(HW_REG_XCC_ID)` that
  MI355 dispatches CTAs to the 8 XCDs exactly round-robin (`blockIdx % 8`). Remapped so a
  contiguous logical tile run lands on one XCD (shares its L2 slice). **Bit-exact; +0.3–0.8% (G2),
  flat-to-−2% (G4) = node-load noise.** Added as a safe `HAPX4` `max()` dispatch candidate (no
  regression). Marginal because the fused per-CTA B-transcode reads B fresh from HBM (little
  cross-CTA L2 reuse), unlike the PRs' separate pre-shuffled-B in L2.
- Other PR techniques (separate scale preshuffle, packed-4/dword scale op_sel, dual row+col
  quant, int64 SRD addressing) don't map — they're structural to the separate-preshuffle design
  the fused kernel subsumes.

**Final standing:** HIP wins 9/12 vs FlyDSL (8 compute +22–77%, DOWN decode when uncontended).
3 memory-bound M≤256 decode shapes (GATE/QKV/M256 text) remain FlyDSL's. Production ships
`dispatch = max(tc_HIP, tc_Fly)` per shape; `v42.hip` carries every lever behind `#ifdef`
(v44 MFMANS + XCDSW are experimental paths, production build unchanged).

## 2026-07-28 — Session 14: two silent defects, then the three uncracked surfaces

Full trace (hypothesis / flags / measurement / verdict / rollback tag per attempt):
`kernel/opt_trace_s14.md`. Guardrails: `git init` + a tag per lever in `/work`, every lever
behind its own `#ifdef` with the default path untouched, dispatch stays `max()` over all
binaries, and — new this session — **a correctness gate that runs before any timing is
accepted**, so an unverified config can no longer win a sweep.

### Two defects, both producing wrong numbers rather than slow ones

1. **`OUTBF16` silently disabled split-K.** The bf16 epilogue dropped the accumulate branch
   entirely, so with `S>1` the splits *overwrote* each other, and the `S>1` zero-fill was
   sized in `float` against a bf16 buffer — an `M*N*2` byte out-of-bounds write. Since
   `bench_full.sh` had started building *every* decode binary with `-DOUTBF16`, decode had
   silently lost split-K, its main occupancy lever. Fix: split-K accumulates into an fp32
   workspace and a `cvt_acc_out` pass converts once at the end; for fp32 output that
   workspace *is* C, so the pre-s14 path stays byte-identical. DOWN decode 94.98 → 185.21
   (**+95%**) on the affected config.
2. **`bench` mode never verified** — it returned before the gold compare, while `hbest`
   swept `S=1..16` and took the max. Every bf16 decode number with `S>1` was therefore
   untrusted. Fix: `bench` runs a split-K self-consistency pass first and prints `REFUSED`
   instead of a time on failure.

A regression was found *during* the fix and is worth recording: the first cut hoisted the
`one` test out of the epilogue and duplicated the store loop, which cost 8 VGPRs (80→88),
crossed an occupancy step, and halved decode. The shipped version keeps one loop and puts
the branch only inside the `OUTBF16` build.

### Result: +9.7% geomean across 12 shapes, nothing regressed

`bench_full.sh`, card 5 at 2% instantaneous use, best-of-3 max, all 17 binaries passing the
new gate. `tc_HIP` column, TFLOP/s:

| shape | M×N×K | baseline | final | Δ |
|---|---|---:|---:|---:|
| LLM square | 4096×4096×8192 | 2245.78 | 2472.93 | **+10.1%** |
| LLM gate_up | 4096×19456×2560 | 1858.85 | 2042.55 | **+9.9%** |
| LLM DOWN decode | 128×2560×9728 | 340.77 | 342.56 | +0.5% |
| LLM GATE decode | 128×19456×2560 | 609.55 | 715.11 | **+17.3%** |
| LLM QKV decode | 128×4096×4096 | 277.64 | 278.87 | +0.4% |
| FLUX imgQKV | 4096×9216×3072 | 1584.00 | 1781.11 | **+12.4%** |
| FLUX imgPROJ | 4096×3072×3072 | 1553.40 | 1745.23 | **+12.4%** |
| FLUX FC1 | 4096×12288×3072 | 1965.40 | 2154.15 | **+9.6%** |
| FLUX FC2 | 4096×3072×12288 | 2002.31 | 2353.69 | **+17.5%** |
| FLUX LIN1 | 4352×21504×3072 | 1894.11 | 1948.45 | +2.9% |
| FLUX LIN2 | 4352×3072×15360 | 2034.35 | 2352.13 | **+15.6%** |
| FLUX txt FC1 M256 | 256×12288×3072 | — | 668.77 | (gate was failing at baseline) |

### HWCVT — the largest single win, and a reversal of the Phase 8b verdict

`-DHWCVT` uses `__builtin_amdgcn_cvt_scalef32_pk32_fp6_f16` to emit the 6-dword fp6
fragment directly, deleting the whole LUT + bit-pack stage. **+5–9% compute, +10–16%
decode.**

Phase 8b (2026-07-20) had rejected exactly this as "blocked by the bit-exact contract",
and that verdict was correct on its own terms: the hardware does real
`fp4→f16→×scale→fp6 RNE` arithmetic and does not reproduce the revised LUT bit-for-bit.
The error was in the *criterion*. Bit-exactness **against the LUT**
was never the requirement — accuracy against fp32 is, and the LUT is only one way to reach
fp6. Measured rel-L2 against an fp32 reference on three shapes, HWCVT's ratio to the LUT's
own error is **0.998–1.004** — indistinguishable, and better on two of three. Essentially
all the error in both paths is the fp6 quantization itself, not the transcode. It enters
the dispatch gated on `rel-L2 ratio ≤ 1.05` rather than on a bit-compare it fails by
construction; the LUT path stays the default and fully working.

Enumerating all 64 (scale-mantissa × |E2M1|) pairs on hardware (`s14_hwcvt_trace.sh`) pins
down exactly *how* the two disagree: **6 of 64 entries, and every one an exact tie** — the
true product sits precisely halfway between two E2M3 codes, the hardware rounds
half-to-even, and the table's `argmin` rounds half-down. Neither is more accurate, which is
why the rel-L2 comes out indistinguishable. The f16 intermediate is exact for all 64
combinations, so HWCVT does exactly one rounding, same as the LUT. This also corrects the
"28/32 codes differ" figure recorded in Phase 8b and repeated in three other documents:
that probe put the E4M3 mantissa on the convert's scale operand, where only the exponent is
honoured, so it measured a broken variant rather than the shipped flow.

*Transferable lesson: when a lever is rejected for violating a contract, check that the
contract is the actual requirement and not a proxy for it.*

### BARAMORT — the barrier surface finally moves

With the `NSTG`-deep ring a stage written at k-tile `t` is not consumed until `t+NSTG-1`,
so **one barrier per ring rotation** suffices instead of one per k-tile. Combined with
`PRIOPK` (raise `s_setprio` around the pack): **+2–3% on top of HWCVT**, and together they
are what took GATE decode from 610 to 715.

### Negative results, each with a root cause

- **MFMANS** (non-scaled MFMA via inline asm, the "free lever never benchmarked") —
  **−1..−2% on all 6 combos.** The scale operand is not on the critical path; saving
  2 VGPR/operand buys nothing at 1 CTA/CU where the 128-VGPR accumulator already pins
  occupancy, and the inline asm blocks compiler scheduling. Not free.
- **APFK** (prefetch a whole A tile ahead using MFMANS-freed VGPRs) — **−36%.** Pushed
  VGPRs past 256 and the kernel spilled to scratch; read off the ISA, not inferred.
- **PIPEGLDS** (async raw-B `global_load_lds` combined with the PIPE ring, the phase-4
  headline idea) — **−8..−11%.** gfx950 counts A loads and raw-B `global_load_lds` in a
  *single* `vmcnt`, so a partial wait cannot separate them; the required `s_waitcnt(0)`
  drains the A prefetch too and serialises exactly what the ring exists to overlap. This
  is architectural, not a tuning failure.
- **SIGSPLIT** (signal-based split-K, the known-missing FlyDSL feature) — correct
  (`maxrel=0`) but **−23..−29% vs the atomic on DOWN**, the only shape that wants split-K.
  Three cuts were needed to reach that number and the intermediate steps are the useful
  part: `__threadfence()` lowers to `buffer_wbl2` + `buffer_inv`, so every CTA invalidated
  the caches and destroyed the B stream in L2 (**6× slowdown**); agent-scope atomics were
  no better and additionally tagged the stores `sc1`, bypassing L2 straight to HBM; the
  final version uses plain stores + `s_waitcnt(0)` + a relaxed counter + `buffer_inv sc0`
  (L1 only). It still loses, because **CDNA4's L2 has native fp32 atomic-add units** — an
  `atomicAdd` is ~1 L2 transaction per element, while SIGSPLIT costs S stores + S loads +
  a store plus a fully exposed serial reduction tail. **On this hardware the atomic is the
  cheap path.**

That last result **refutes the premise the phase was built on.** "Atomics killed every
split-K attempt (split4 collapsed to 60 TF)" was the `OUTBF16` defect above, not the
atomics: with the defect fixed, atomic split-K gives DOWN 298–336 TFLOP/s. A long-standing
architectural conclusion turned out to be a bug misread as physics.

### The 619 vs 1524 GATE gap, correctly diagnosed

Split-K cannot close it — GATE's optimum is **S=1**, so no reduction ever runs. GATE is
128×19456×2560; `K=2560` is only 20 k-tiles, and splitting an already-short K leaves each
CTA 10 or 5 tiles while the fixed prologue and epilogue stay constant. The gap is **wave
quantization plus short K**: 608 output tiles over 256 CUs at 1 CTA/CU is 2.4 waves, so the
last wave runs 38% populated. Closing it needs a persistent or stream-K decomposition — a
different kernel structure. What *did* move it was HWCVT + BARAMORT+PRIOPK, **610 → 715
(+17%)**.

**Final standing:** unchanged in shape — HIP for compute-bound, FlyDSL for the three
memory-bound M≤256 decode shapes, `dispatch = max(tc_HIP, tc_Fly)` — but every HIP number
is now 3–18% higher and, for the first time, every one of them is correctness-gated.

---

## 2026-07-31 — Session 18: a W4A4 transcode kernel (`w4a4_tc`), and thread count as a dispatch axis

Full detail in `kernel/opt_trace_s18.md`.

Built `in_dtype="w4a4_tc"`: **both** operands stored NVFP4 with E4M3 scales, both transcoded
to MXFP6 in-kernel. The production baseline `w4a6_tc` transcodes B only (A is already MXFP6),
so the new kernel halves A's global bytes and pays an extra transcode for them. It went from
**0.63–0.88× the baseline to 0.82–0.88×** on compute shapes, and does not overtake it.

Also renamed for honesty: `in_dtype="w4a6_tc"` is now an accepted alias for the historical
`w6a6_tc`, which was always a misnomer — that kernel stores W as NVFP4, not fp6. If both
operands are stored fp6 there is nothing to transcode, so no such kernel exists.

### Why it cannot win, stated as a cost law

Per MFMA, `VALU = base + tcB/tile_m + tcA/tile_n`. w4a4_tc's cost is the baseline's **plus a
strictly positive A term at every tile**, so on any VALU-bound shape it starts behind. This
is not a tuning gap; it is the shape of the arithmetic, and it is what bounds the result above.

Underneath it sits a fact worth stating plainly: **the transcode's data widening buys zero
accuracy.** `fp4 → fp6` is exact (`code << 2`) — E2M3 has E2M1's range plus two spare mantissa
bits. The entire transcode exists to carry the **E4M3 scale mantissa** that the MFMA's E8M0
scale operand cannot express. Confirmed the hardware will not do it for us: both
`v_cvt_scalef32_pk_f16_fp4` and `v_cvt_scalef32_pk32_fp6_f16` take a real f32 scale and use
**only its exponent** (tested by folding the mantissa factor into each; results came back short
by exactly that factor, both times).

### Four levers shipped, all bit-identical

| Lever | Effect |
|---|---|
| SWAR the E4M3→E8M0 scale convert (4 bytes in parallel, no cross-byte carry) | VALU 1678→1534 (tc), 1186→1146 (base) |
| `(1+m/8)` as an f16 bit pattern instead of `uitofp/×/+/trunc/broadcast` | VALU 1534→1464, 1146→1100 |
| `sched_barrier` between A's staged transcode blocks | 274→256 VGPR, 1→2 waves, 0 spills, **+26%** |
| 512-thread (8-wave) workgroups at `tile_n=512` | **+22..55%** tc, +3..32% base — compute shapes only |

The last one is what the cost law was pointing at. Waves split **N only**, so 8 waves at
`tile_n=512` give each wave the same 64-wide slice and same 128 accumulators as 4 waves at
`tile_n=256` — everything per-thread is unchanged **except the A codes each thread stages,
which halve**. Behind `WG512`; 256-thread paths are bit-for-bit untouched.

**The lever is `tile_n`; the 512 threads are the enabler.** The shipped A/B moves both at once, so
`s18_wgwhy.sh` separated them, and neither knob works alone: thread count at fixed tile also halves
the accumulators (`mfma` 128→64), giving up register-blocking reuse for −14% on square; `tile_n` at
fixed thread count doubles accumulators to 256/thread, hits the 512-VGPR ceiling, falls to one wave,
spills 60 times, −40%. Only the pair holds `n_per_wave = tile_n/num_waves` at 64.

**And most of the shared gain is not the transcode at all.** Per-thread pack counts go 24→20 for
`w4a4_tc` (its A packs 8→4, B unchanged) but **16→16 for the baseline**, which has no A packs — yet
the baseline still gains 25.8% on square. Doubling `tile_n` halves how many times the A tile is
re-read from HBM (A is loaded once per workgroup, and the grid covers N in `N/tile_n` workgroups);
that prize is shared, and A's halved transcode is only `w4a4_tc`'s extra margin on top. Stated this
way the model also predicts the decode loss: at M=128 the grid is `1 × N/512`, so doubling `tile_n`
halves an already-too-small workgroup count.

### Two register-pressure results that contradict scalar-liveness reasoning

- Removing A's **provably dead** FP8 pad dword (the read side loads and discards it) is
  numerically free and deletes two zero words plus 8 `ds_write`. It measured VGPR **274→278**,
  spills **22→43**, **−10%**. Splitting a `dwordx4` pair into `dwordx4 + dwordx2` costs more in
  aligned-tuple pressure than the registers it saves.
- The `sched_barrier` above **removes no instruction** and saves 18 registers, because it stops
  the two 16-VGPR `pk32` source tuples from being live at once.

VGPR count in this kernel is set by **aligned-tuple pressure**; scalar liveness got the sign
wrong in both directions. Related: `waves_per_eu=2` cost 16–18% before the VALU cuts and gained
11% after them — a register-pressure verdict has to be re-priced after any change to liveness.

### async_copy is an occupancy lever, and it re-prices A's staging

LDS staging here is **A-only** (B goes global→registers per fragment). `async_copy` is worth
+28% to the baseline for a register reason, not a DMA one: without it the baseline holds A in
registers, needs 289 VGPRs and drops to **one** wave. w4a4_tc cannot use async — A's transcode
sits between the load and the store — but the fence already bought it that wave.

**On equal staging footing (both non-async, 128×256) w4a4_tc is 8.2% ahead: 1783 vs 1647**,
despite 364 more VALU, because it reads half of A's bytes. Measured against that correct
control, A costs 42 VALU per staged block against B's 39.6 — **6%**. An earlier "12 ops per
block of addressable overhead" was an artifact of differencing against the *async* baseline.

### Thread count is now a per-shape dispatch choice

WG512 helps compute shapes and hurts decode: at M=128, `tile_n=512` leaves 38 workgroups for
256 CUs. The final table selects over tile **and** thread count (`kernel/s18_dispatch.sh`,
full output in `kernel/s18_results/s18_dispatch_table.txt`):

| shape | fp4 | w4a6 | w4a6_tc | w4a4_tc | tc/base | w4a4_tc picks |
|---|---:|---:|---:|---:|---:|---|
| LLM square 4096·4096·8192 | 3774 | 3263 | 2283 | 1902 | 0.833 | 128×512 / 512t |
| LLM gate_up 4096·19456·2560 | 3439 | 2941 | 2056 | 1789 | 0.870 | 128×512 / 512t |
| FLUX imgQKV 4096·9216·3072 | 3015 | 2630 | 1776 | 1420 | 0.800 | 128×512 / 512t |
| FLUX imgPROJ 4096·3072·3072 | 2577 | 2300 | 1496 | 1334 | 0.892 | 128×512 / 512t |
| FLUX FC1 4096·12288·3072 | 3175 | 2647 | 1953 | 1561 | 0.800 | 128×512 / 512t |
| FLUX FC2 4096·3072·12288 | 3561 | 3006 | 1975 | 1621 | 0.821 | 128×512 / 512t |
| FLUX LIN1 4352·21504·3072 | 3264 | 2732 | 1982 | 1781 | 0.899 | 128×512 / 512t |
| FLUX LIN2 4352·3072·15360 | 3875 | 3248 | 2116 | 1767 | 0.835 | 128×512 / 512t |
| LLM DOWN dec 128·2560·9728 | 173 | 149 | 109 | 100 | 0.910 | 64×128 / 256t |
| LLM GATE dec 128·19456·2560 | 1105 | 756 | 561 | 452 | 0.807 | 64×128 / 256t |
| LLM QKV dec 128·4096·4096 | 547 | 426 | 250 | 191 | 0.765 | 64×128 / 256t |
| FLUX FC1 M256 256·12288·3072 | 1285 | 1171 | 893 | 692 | 0.775 | 64×128 / 256t |

**0.765–0.910 of the baseline, median 0.83, ahead on no row.** `w4a4_tc` picks 512 threads on
**8 of 8** compute rows; the baseline picks them on only **4 of 8**.

Two rows had to be discarded and retaken: `gate_up` and `QKV_dec` first landed on a shared card
(references 1385 and 1653 against a 2180 quiet baseline). At face value `gate_up` read **0.556**
*and named a different winning tile*; retaken quiet it reads **0.870**. A contended row does not
merely read low — it can select the wrong configuration, so a dispatch table built on one ships
the wrong kernel.

### Closed with a measurement

A LUT path for A (VALU +9%, no registers saved), `lds_stage` 3/4 (does not compile for the
whole fp4 family), `tile_k=256` / `tile_n=192` (does not compile), tile 64×256 (−24%; halving
`tile_m` doubles the B term).

One route *would* beat the baseline and was deliberately **not built**: round A's scale to the
nearest E8M0 and leave A as fp4. It deletes all 388 A-transcode VALU, halves A's bytes and
restores async — at ~2× worse RMS error (19.7% vs 9.5%). That is a change to the numerics
contract, and the contract is the project.


## 2026-07-31 � s19: quiet-node re-measurement, the fp6 control, and a defect that measured +36%

Asked: re-measure on a quiet node including all references, verify they are measured correctly, then
optimise `w4a6_tc` as close to `w4a6` as possible, then update the docs.

### The node

All 8 GPUs at 0% utilisation but 93�94% of VRAM allocated � an idle tenant holding memory, the state
that wakes mid-sweep. The reference (`w4a6_tc` square 128�256/256t async) is therefore re-timed after
**every row** and printed in the table. It held 2177�2189, a 0.55% band. s18's references sat at
2083�2110 with two rows visibly contended, which is why several s19 rows look 10�23% better for
reasons that have nothing to do with the kernel.

### w4a6_tc was being tuned on the wrong axis

`w4a6_tc` transcodes B only � A arrives already MXFP6 � so its cost law has no A term:

    VALU/MFMA = base + tcB/tile_m

Every WG512 config shipped in s18 raises `tile_n`, which is the *A-side* knob. The B-side knob is
`tile_m`, and 512 threads make `256�256` reachable at **constant accumulators**:
`acc_VGPR = (tile_m/16)�(n_per_wave/16)�4 = 16�2�4 = 128`, the same 128 the shipped configs use, with
A's per-thread share unchanged and B's per-MFMA share halved.

| config | sq | gate_up | FC2 | imgQKV | vgpr | valu | packs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128,256/256t | 2181.9 | 1725.2 | 1556.6 | 1695.1 | 256 | 1100 | 16 |
| 256,128/256t | 2180.9 | 2056.2 | 2032.2 | **1880.6** | 250 | 927 | 8 |
| 128,512/512t | **2282.0** | 2036.9 | 2062.7 | 1735.4 | 252 | 1077 | 16 |
| **256,256/512t** | 2273.5 | **2073.1** | **2167.9** | 1764.6 | 248 | **881** | **8** |

Shipped. `256�512/512t` was predicted to fail before running (256 accumulator VGPRs) and failed the
predicted way: VGPR pinned at the cap, 348 spills, a tenth of the throughput.

**The two arms want opposite tiles.** `w4a6` is much *worse* at tile_m=256 (square 3190 ? 2555) �
no transcode to amortise, so a taller tile only costs N-side parallelism. A tile list tuned on the
non-transcoding arm systematically mis-serves the transcoding one.

### A config that measured +36% because it computed two thirds of the output

`128�384/512t` reported **2957** on FC2 against a true best of 2177. `tile_n=384` on 8 waves gives
`n_per_wave=48`, so `num_acc_n=3`; the packed-N MFMA takes **two** blocks per step, runs `3//2 = 1`
pair, and silently drops the third. `mfma=64` where every legal config issues 128.

The gate behaviour is the part worth keeping:

- The **K-advance probe said CORRECT**. It counts zero *rows*; this defect drops *columns*. A gate is
  blind along any axis it does not enumerate.
- The **A-value gate caught it only through `# WARN nonuniform spread=` lines** � its `max_rel`/
  `rms_rel` were identical to the good configs. The warning, not the metric, was the signal.
- The **ISA probe caught it outright**, via `mfma=64`.

A defect that removes work makes the kernel *faster*, so the timing harness ranks it first. Only a
counter that must move if the story is true can refuse it.

**Fixed.** The old guard, `if is_fp4 and (tile_n // 64) < 2`, had two faults: `tile_n//64` is
`num_acc_n` only at 4 waves, and `< 2` catches only the zero case. Now derived from the real wave
count and rejecting odd as well:

```python
_num_acc_n = int(tile_n) // num_waves_const // 16
if is_fp4 and (_num_acc_n < 2 or _num_acc_n % 2):
    raise NotImplementedError(...)
```

Verified by calling the compiler directly and reading the exception text, since "failed to compile"
and "rejected by my guard" are different observations. Legality depends on tile_n **and** wave count:
`tile_n=384` is fine at 4 waves (`num_acc_n=6`), degenerate at 8. The old form could not express
that, which is why 384 was reachable at all. Every shipped config is still accepted.

**False alarm, kept because the reasoning generalises.** The same run reported `max_rel 0.749 /
rms 0.537` for `w6a6_tc` at the new config, which looked like a hard failure. Those are the probe's
numbers for `w6a6_tc` at *every* tile including the one shipped for six sessions, and `w4a4_tc`'s
0.360/0.0898 in the same run matches its documented fold error, confirming calibration. The alarm
came from reading an absolute error figure without its control, through a `tail -5` that cut the
verdict line off.

### Three quarters of the gap to w4a6 is not the transcode

`w4a6` feeds B to the MFMA as MXFP4, a 16 B/lane operand. `w4a6_tc` must feed it as MXFP6, 32 B/lane,
because that is the only way to carry NVFP4's E4M3 scale mantissa. **`w4a6` is the less accurate
kernel.** Adding `fp6` (B fed as fp6, no transcode) as a fourth reference decomposes the gap:

    gap = (fp6 operand format) � (transcode arithmetic)
    square  0.706 = 0.792 � 0.891       gate_up 0.705 = 0.758 � 0.930
    FC2     0.681 = 0.765 � 0.890       imgQKV  0.671 = 0.761 � 0.881

The transcode costs 5�12% on compute shapes and buys **half of B''s HBM bytes**. Against the format
it actually needs, `w4a6_tc` runs at **0.88�0.99** of a kernel handed that format for free. Closing
the rest needs a 16 B/lane 6-bit MFMA operand, which is not a kernel''s choice.

Decode is the exception: `DOWN_dec` is 0.691 of fp6 because at M=128 with tile_m=64 the `tcB/tile_m`
term is least amortised. `QKV_dec`''s 0.993 is not a triumph � fp6 itself collapses to 251 there.

### Final table

| row | fp4 | w4a6 | fp6 | w4a6_tc | w4a4_tc | tc/w4a6 | tc/fp6 | best w4a6_tc |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LLM_square | 3797 | 3235 | 2563 | 2284 | 1938 | 0.706 | 0.891 | 128,512/512t |
| LLM_gate_up | 3447 | 2942 | 2229 | 2073 | 1786 | 0.705 | 0.930 | **256,256/512t** |
| LLM_DOWN_dec | 236 | 191 | 188 | 130 | 110 | 0.679 | 0.691 | 64,128/256t |
| LLM_GATE_dec | 1105 | 963 | 757 | 703 | 544 | 0.730 | 0.929 | 64,128/256t |
| LLM_QKV_dec | 548 | 426 | 251 | 249 | 191 | 0.586 | 0.993 | 64,128/256t |
| FLUX_imgQKV | 3306 | 2799 | 2131 | 1878 | 1580 | 0.671 | 0.881 | 256,128/256t |
| FLUX_imgPROJ | 2961 | 2595 | 2020 | 1839 | 1498 | 0.709 | 0.910 | **256,256/512t** |
| FLUX_FC1 | 3546 | 3015 | 2310 | 2148 | 1877 | 0.712 | 0.930 | 128,512/512t |
| FLUX_FC2 | 3820 | 3208 | 2454 | 2184 | 1712 | 0.681 | 0.890 | **256,256/512t** |
| FLUX_LIN1 | 3443 | 2911 | 2177 | 2068 | 1779 | 0.710 | 0.950 | **256,256/512t** |
| FLUX_LIN2 | 3894 | 3261 | 2540 | 2223 | 1773 | 0.682 | 0.875 | **256,256/512t** |
| FLUX_FC1_M256 | 1273 | 1181 | 956 | 886 | 696 | 0.751 | 0.927 | 64,128/256t |

### Attribution, measured back to back rather than across sessions

| row | old best | old config | new 256�256/512t | gain |
|---|---:|---|---:|---:|
| LLM_gate_up | 2057.8 | 256,128/256t | 2069.7 | +0.6% |
| FLUX_imgPROJ | 1785.7 | 256,128/256t | 1837.0 | +2.9% |
| FLUX_FC2 | 2062.7 | 128,512/512t | 2170.4 | +5.2% |
| FLUX_LIN1 | 2059.0 | 128,512/512t | 2063.7 | +0.2% |
| FLUX_LIN2 | 2117.7 | 128,512/512t | 2223.2 | +5.0% |

**Median +2.9%.** The raw s18?s19 deltas on those rows read +0.8/+23/+10.6/+4.3/+5.1%; imgPROJ would
have been published as a 23% optimisation when 2.9% is real. Small despite an 18% VALU cut because
these shapes are not VALU-bound at 2 waves � s18''s exchange rate (~0.19 per marginal VALU at 2
waves) predicts ~3.4%. The cost law names the knob; the exchange rate prices it.

`128�384/256t`, newly legal at 4 waves, was swept on all six rows whose N it divides and won none:
legal but never best.

Full detail in `kernel/opt_trace_s19.md`; raw tables in `kernel/s19_results/`.

---

## 2026-08-01 - s20: a shipped lever that two harness rewrites quietly dropped

Session input was five AMD documents, one of which ([W4A6 quantization on
Instinct](https://rocm.blogs.amd.com/artificial-intelligence/w4a6-quant-mm/README.html)) is about
this exact kernel family ? FlyDSL's preshuffle GEMM adapted to a 6-bit activation operand. The
useful outcome was not a technique we had never heard of.

### This is a regression report, not a discovery

Split-K was designed, implemented, correctness-gated, **measured at +177% on `DOWN_dec`**, and
presented on deck slide 12 - "Decode is an occupancy problem, not a transcode problem" - back in
s17. That slide even names the follow-up: *"the next lever is a cheaper reduction, not more
splitting."*

Then s18 and s19 rewrote the sweep drivers, and every one of them passes `--splits 1`. The kernel
kept the feature, the repo kept the gate, the deck kept the slide; the dispatch tables quietly
stopped using it.

**The real damage was not the lost speed.** s19 published `LLM_DOWN_dec` at 130 TFLOP/s - against
381 already on record - and *explained* the number, in `status.md`: "at M=128/tile_m=64 the
`tcB/tile_m` term is least amortised." Plausible, mechanism-shaped, wholly wrong. A regression got
absorbed into the mental model as a property of the kernel. Restoring the flag reproduced s17
exactly: **130 -> 393** now, 138 -> 381 then.

What the AMD blog contributed was the prompt to look. On the decode `down_proj` shape, which is our
`LLM_DOWN_dec` (M=128, N=2560, K=9728):

> The number of CTAs is therefore `ceil(M/tile_m) � ceil(N/tile_n)` - it scales with the *output*
> size, M�N, and not at all with K. At decode there are not enough of them to fill the GPU.

The s19 table was fair - all arms pinned identically, so the ratios hold - and wrong about the
absolute decode numbers, for all five arms simultaneously.

| row | workgroups | arm | `sk=1` | + split-K | gain |
|---|---:|---|---:|---:|---:|
| LLM_DOWN_dec | 40 | fp4 | 287.2 | 417.7 | +45.4% |
| LLM_DOWN_dec | 40 | w4a6 | 188.5 | 422.3 | +124.0% |
| LLM_DOWN_dec | 40 | fp6 | 188.0 | 409.4 | +117.8% |
| LLM_DOWN_dec | 40 | **w4a6_tc** | **130.0** | **392.6** | **+202.0%** |
| LLM_DOWN_dec | 40 | w4a4_tc | 111.9 | 367.2 | +228.2% |
| LLM_QKV_dec | 64 | **w4a6_tc** | 250.5 | 339.9 | +35.7% |
| LLM_GATE_dec | 304 | all five | ? | ? | ?0.1% .. +1.1% |
| FLUX_FC1_M256 | 384 | all five | ? | ? | ?1.0% .. +0.7% |
| LLM_square *(control)* | 512 | w4a6 / w4a6_tc | ? | ? | **?0.2%** |

**The predictor is the workgroup count, not M.** `GATE_dec` and `DOWN_dec` are both M=128; only the
one whose grid cannot fill 256 CUs benefits. Split-K must be selected per shape ? on the square
control it is pure overhead.

### Gated before believed

s19's lesson was a config that measured +36% while computing two thirds of N, and split-K here once
hid a silent epilogue bug reported as "+95% on decode". A 3� jump is exactly the shape of a kernel
that stopped doing work.

`kernel/s17_splitk_check.py` already had the right invariant: the summed partial planes must
reproduce the full-K output, with block scales that vary per K chunk so no two slices contribute the
same thing. It had one hole ? `w4a4_tc` was missing from the family list and so was being handed
per-token f32 scales and an unpacked A, a layout the kernel does not read. Fixed. Every arm at every
split now returns `rel = 2.3e-03` against a 1e-2 threshold, the expected bf16 re-rounding of the
partial sum. Split-K is correct.

### The next bottleneck is the reduction - which s17 also already said

Slide 12 ends: "on GATE and M256 the extra pass over the partials eats a +43% and +18% kernel win:
**the next lever is a cheaper reduction, not more splitting**." s20 adds the number on the row that
matters rather than a new conclusion.

The gate reports raw throughput *and* throughput net of the reduction. On `DOWNdec` at `split_k=19`:
**640.6 TFLOP/s raw, 415.7 net.** The `torch.sum` over 19 bf16 planes costs ~5.4 �s against a
10.0 �s kernel, moving 13.1 MB at roughly 2.4 TB/s. A third of the achievable throughput on the
split-K rows is being spent in a separate elementwise kernel.

Every number reported here is the net one ? `bench_sweep` times kernel + reduce as one call ? so
this headroom is unclaimed, not double-counted - and it has been the top lever for three sessions.
Folding the reduction in-kernel is the fix, but s17 already tried a signal-based variant
(`SIGSPLIT`) and measured it **-23..-29% against the atomic on DOWN**, so blog 5's semaphore/signal
design is not automatically right here; whatever gets built must beat both. The caution: the current
epilogue deliberately binds each workgroup to its own C plane so a stray store cannot cross splits,
and that is exactly the safety an in-kernel reduction gives up - and where the old silent bug lived.

### Confirmed by the literature (closed, not unexplored)

- **`256�256` at 512 threads.** Blog 3 sweeps `128�128/512t`=1829, `256�256/512t`=2288,
  `256�256/1024t`=2228 and selects the middle ? the config s19 derived from the B-side cost law,
  down to the finding that more threads is a regression.
- **MFMA is 16 cycles for FP4 and FP6 alike** (32 only if an operand is FP8). They issue
  identically, so the `0.792` "operand format" factor in the s19 gap decomposition is *entirely* a
  delivery cost ? register footprint and LDS traffic ? with zero contribution from MFMA issue rate.
  Nothing on the compute side can reduce it. This upgrades an inference to a documented fact.
- **`lds_stage`** looked like another pinned axis, but the compiler rejects anything but 2 for the
  4-bit family at `tile_k=128`. Closed by constraint, not oversight.

### Top untried lever

Blog 3's final step takes `256�256/512t` from 2288 to **2680 TFLOP/s (+17%)** with the HipKittens
8-wave ping-pong pattern: two waves per SIMD alternating memory and MFMA, steered by `s_barrier` on
a wave-half predicate, `s_setprio(1)` around the MFMA, and `sched_barrier(0)` fences. `w4a6_tc` now
ships exactly that configuration. We emit `sched_barrier` and `gpu.barrier`; we lack the
*predicated* wave-half barrier and `s_setprio` on the FlyDSL path. Not attempted ? it restructures
the hot loop, and split-K was the higher-value use of the measurement time.

### Final table (reference per row 2184?2192, a 0.37% band)

| row | fp4 | w4a6 | fp6 | w4a6_tc | w4a4_tc | tc/w4a6 | tc/fp6 | best w4a6_tc |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LLM_square | 3798 | 3244 | 2565 | 2285 | 1938 | 0.704 | 0.891 | 128,512/sk1/512t |
| LLM_gate_up | 3449 | 2943 | 2235 | 2073 | 1786 | 0.704 | 0.928 | 256,256/sk1/512t |
| **LLM_DOWN_dec** | 416 | 421 | 409 | **392** | 367 | **0.931** | **0.958** | 64,128/**sk19**/256t |
| LLM_GATE_dec | 1107 | 962 | 768 | 708 | 547 | 0.737 | 0.923 | 64,128/sk1/256t |
| **LLM_QKV_dec** | 546 | 425 | 335 | **335** | 311 | 0.788 | **1.001** | 64,128/**sk4**/256t |
| FLUX_imgQKV | 3299 | 2800 | 2134 | 1878 | 1566 | 0.671 | 0.880 | 256,128/sk1/256t |
| FLUX_imgPROJ | 2962 | 2591 | 2020 | 1843 | 1493 | 0.711 | 0.913 | 256,256/sk1/512t |
| FLUX_FC2 | 3830 | 3220 | 2456 | 2178 | 1718 | 0.676 | 0.887 | 256,256/sk1/512t |
| FLUX_LIN1 | 3442 | 2910 | 2180 | 2069 | 1781 | 0.711 | 0.949 | 256,256/sk1/512t |
| FLUX_LIN2 | 3889 | 3253 | 2541 | 2221 | 1771 | 0.683 | 0.874 | 256,256/sk1/512t |
| FLUX_FC1_M256 | 1287 | 1182 | 955 | 896 | 664 | 0.758 | 0.938 | 64,128/sk1/256t |

`FLUX_FC1` was not re-taken: the node acquired another tenant on all 8 cards partway through and the
reference never held quiet across the row. s19's clean value stands (2148 for w4a6_tc, tc/fp6 0.930)
and the square control says split-K does not apply to it.

Ten of twelve rows reproduce s19 within ~1% ? that is the consistency check on the whole
measurement ? and the two that moved are exactly the two the mechanism predicts. `w4a6_tc` is now
? 0.87 of `fp6` on every row and above 0.95 on three.

### Process fix owed

The sweep drivers have no shared defaults block, which is how two rewrites lost a gated, measured,
presented result. Any script that pins an axis should have to say why, in the script. And when a row
regresses against a number already published, treat the prior result as the reference and go find
the delta *before* reaching for a mechanism that explains the new one - an explanation that fits is
not evidence, it is the most comfortable way to lose a result permanently.

Full detail in `kernel/opt_trace_s20.md`; raw table in `kernel/s20_results/`.

## 2026-07-31 - s21: the root cause, a registry that cannot forget, and Gemma 4 31B

### Why the lever disappeared - it was not context length

The obvious explanation for losing a shipped result across sessions is that the sessions are long
and it scrolled out of the window. That explanation is wrong here, and it matters that it is wrong,
because the fix it implies - summarise harder - prevents nothing. Three independent causes:

1. **The finding lived in a different program than the numbers.** `bench_sweep.py` defaults
   `--splits` to `"1"`; the bespoke `s17_splitk_probe.py` defaults it to `"1,2,4,8,16"`. Split-K was
   only ever exercised by its own probe, and *no s17 shell driver passes `--splits` at all*. It was
   never in the main path, so nothing had to be forgotten for it to be absent from s18 and s19.
2. **The driver's helper was copy-pasted forward with the flag frozen in it.** `s18_dispatch.sh:19`
   and `s19_dispatch.sh:19` are byte-identical including `--splits 1` - one `go()` function copied
   across sessions, not two independent decisions. A helper stops being read the moment it is pasted.
3. **The knowledge was stored in prose.** Slides, `status.md` and the opt traces were all correct
   about split-K. None of them is executable, so none could fail a run.

`--splits 1` was on line 19 of every session's context the entire time. Reading it does not tell
you the flag is worth +177%. What was needed was never "what did the last session say" but "what is
this row supposed to score" - a comparison, which no quantity of context performs for you.

### The mechanism: a registry the driver reads, and a gate that fails

`kernel/bestcfg.py` plus `kernel/best_known.json` hold the best measured TFLOP/s per (shape, arm)
with the config that produced it, the reference at measurement time, the session and the date. It
removes all three causes at once:

- Drivers ask for a config space instead of carrying one, so **there is no flag left to freeze**.
  `s21_bench.sh` hardcodes no shape, no tile and no `split_k`. `split_k` candidates are *derived*
  from K's divisibility and an occupancy band around the 256 CUs, not hand-enumerated - so a new
  shape gets sensible splits without anyone remembering to add them.
- `bestcfg.py check` exits non-zero when a cell falls >3% below its record, and its message names
  the recorded config and says *"Do NOT explain this number - reproduce the recorded config first."*
  s19's failure was not missing the regression, it was explaining it, so the gate refuses a theory
  before a reproduction.
- A record is only taken against a reference that is quiet **before and after** the row, which is
  what stops contention from being written into the registry as a result.
- `build_pptx.py` reads the registry at build time, so the deck cannot disagree with the last
  measurement. A slide cannot be re-run; that is exactly why a hand-copied number survived three
  sessions after it stopped being true.

Verified against the historical failure: `check --shape LLM_DOWN_dec --arm w6a6_tc --tflops 130`
exits 1 and names the s20 record of 392 at `sk=19`.

### Gemma 4 31B: nine shapes, and the first rows where transcoding beats not transcoding

Added the dense Gemma 4 31B GEMMs (embed 5376, MLP 21504, 32x256 heads, 16 KV, GeGLU), prefill and
decode. Quiet card, reference held 2187-2193 before and after every row, all 45 cells reproduced
within 0.4% on a second pass.

| shape | fp4 | w4a6 | fp6 | w6a6_tc | w4a4_tc | tc/w4a6 | tc/fp6 | sk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G4_QKV        | 3856 | 3270 | 2539 | 2293 | 1982 | 0.701 | 0.903 |  1 |
| G4_O          | 3546 | 2911 | 2080 | 2008 | 1575 | 0.690 | 0.965 |  1 |
| G4_GATEUP     | 3596 | 3142 | 2438 | 2268 | 1855 | 0.722 | 0.930 |  1 |
| G4_DOWN       | 3725 | 3008 | 2195 | 1918 | 1422 | 0.638 | 0.874 |  1 |
| G4_QKV_dec    | 1155 | 1066 |  879 |  866 |  743 | 0.812 | 0.985 |  3 |
| G4_O_dec      |  716 |  699 |  591 |  603 |  535 | 0.862 | **1.019** |  8 |
| G4_GATEUP_dec | 2271 | 1938 | 1475 | 1281 | 1169 | 0.661 | 0.868 |  1 |
| G4_DOWN_dec   | 1343 | 1196 |  953 | 1060 |  887 | 0.887 | **1.113** | 12 |
| G4_DOWN_M256  | 1834 | 1634 | 1433 | 1388 | 1229 | 0.849 | 0.968 |  6 |

On `G4_DOWN_dec` and `G4_O_dec`, `w6a6_tc` - which must convert every weight on the fly - is
**faster than `fp6`, which is handed the format for free.** The mechanism is the one the cost model
predicts: `w6a6_tc` stores B fp4-dense, so on a memory-bound row it reads half the weight bytes
`fp6` does, while the transcode ALU cost amortises over K. Gemma's down_proj has the longest K in
the suite (21504) against a narrow N=5376 - exactly the regime where the byte saving outruns the
conversion. Qwen's `DOWN_dec` hinted at this at K=9728 (0.958); doubling K takes it past parity.

Four of the five best `tc/w4a6` ratios in the suite are now Gemma small-M rows (0.887, 0.862, 0.849,
0.812 against a 0.711 median), behind only Qwen's `DOWN_dec` at 0.931.

### The suite had a blind spot, and adding shapes is what exposed it

All four Gemma small-M rows land in a 336-504 workgroup band around the 256 CUs, by different
routes: `GATEUP_dec` already launches 336 unaided and took `sk=1`, while `O_dec` and `DOWN_dec`
launch 42 and buy the same band with `sk=8` and `sk=12`. K=21504 admits every divisor of 84, so
`DOWN_dec` can choose `sk=12` - a value that K=9728's legal set of {1, 2, 19} cannot express. The
old suite contained exactly one row that wanted split-K and allowed only three legal values of it,
so it could not demonstrate that `split_k` has an *optimum* rather than a direction. A benchmark
suite is an instrument, and adding shapes is not only coverage - it is what lets a knob show shape.

### The reduction lever: built, correct, and rejected on speed

Split-K's separate reduction costs 35% of what the split buys, and our own record said the fix was
settled: "CDNA4's L2 has native fp32 atomic-add units - on this hardware the atomic IS the cheap
path." So `split_reduce="atomic"` was implemented: every workgroup `buffer_atomic_fadd`s into one
shared fp32 C plane, deleting the second pass entirely.

It is correct, and *more accurate* than the shipped path - 2.0-3.5e-3 against split_k=1 versus
3.0-4.4e-3, because it accumulates in fp32 instead of rounding each partial to bf16. It is also
slower everywhere it would matter:

| row | split_k | planes | atomic | winner |
| --- | --- | --- | --- | --- |
| LLM_DOWN_dec |  2 | 206.9 | 217.8 | atomic +5% |
| LLM_DOWN_dec | 19 | 392.0 | 251.1 | planes +56% |
| LLM_QKV_dec  |  8 | 306.5 | 233.2 | planes +31% |
| G4_DOWN_dec  |  7 | 834.4 | 714.1 | planes +17% |
| G4_DOWN_dec  | 21 | 822.4 | 511.8 | planes +61% |

The pattern is arithmetic, not tuning: atomic cost is linear in `split_k` (one operation per element
per split) while the planes reduction is a bandwidth-bound streaming write and coalesced read. The
rows that want split-K want a *large* split, so the lever and its proposed fix pull in opposite
directions. The inherited verdict was true - against SIGSPLIT, whose cost included an exposed serial
tail - and false against the design we actually ship. Beating a bad baseline is not evidence of
beating the shipped one.

### Two harness traps, both of which read as kernel bugs

- **`flyc.compile` launches the kernel once.** Every check here calls compile and then `cf()`, and
  could always ignore that extra launch because *a store is idempotent*. An accumulating epilogue is
  not: the gate reported exactly 2.00x the reference at every `split_k` - the signature of a kernel
  computing the wrong K range, and not a kernel bug at all. When a long-standing idempotence
  quietly stops holding, everything that silently relied on it becomes wrong at once.
- **The control that found it had to be permitted first.** Atomic at `split_k=1`, where any ratio
  but 1.0 cannot be about splitting, read 1.9927 and located the bug in one measurement. A guard
  that rejects a useless configuration also rejects your best diagnostic.
- **Contention reversed the A/B verdict rather than widening it.** The first crossover sweep, arms
  run back to back on one card, said atomic won by up to +60%. The identical script on a quiet card
  inverted every row. Contention hits the arms unequally: planes is bandwidth-bound and fell to 82
  TFLOP/s where it later measured 207, while the atomic path is L2-atomic-rate bound and barely
  moved. Back-to-back on one card controls for drift, not for load; if the two arms stress different
  resources, the reference has to be quiet as well.

Full detail in `kernel/opt_trace_s21.md`; file map in `INDEX.md`; registry in `kernel/best_known.json`.

## 2026-08-03 - s22: the arms inside a real model, and what they cost

Every result before this one is a GEMM benchmark. s22 put the five arms inside Gemma 4 and
measured accuracy against a bf16 baseline on identical questions, on 12B and then 31B.

The headline is not the one 12B alone suggested. On 12B, `w4a6_tc` looked indistinguishable
from bf16 at 4 bits per weight (+0.95, p=0.372). On 31B the same arm is **-7.70** and
`w4a4_tc` falls from the best 4-bit arm to the worst. An ordering that inverts with model
size is a symptom, and chasing it found the cause: **the E4M3 scale book cannot represent
the scales Gemma's weights need, and this codec has no per-tensor global scale to fix
that.** 96% of 12B weight blocks and 99.8% of 31B blocks clamp to the smallest E4M3 scale,
so on weights the per-block scale had effectively stopped being per-block. The E4M3 arms
were never measured with the finer scale they are supposed to have.

Adding the global scale recovers +2.40 and +1.30 points on 31B and makes E4M3 weights beat
E8M0 by ~1 dB as intended, but does not close the gap to `w4a6`, which remains the best arm
at -2.70. That residual is recorded as open rather than explained.

### A gate that could not fail, and three arms that had never really run

`w4a6`, `fp6` and `w6a6_tc` had never executed on non-constant activations. `verify_w4a6.py`
builds A with one value repeated along K, which makes it **permutation-invariant**: a wrong
host byte order feeds the kernel a permutation of identical values and the answer comes out
right regardless. The check passed for every possible layout, so it constrained none of them.
The same blind spot covered the A-*scale* layout, untested for all five arms for the identical
reason. A test insensitive to a class of bug is insensitive everywhere that class reaches.

The fix is a 16/8 split within each K=32 block - 16 bytes at the column offset, 8 more at
+64 - forced by the kernel's own read pattern, with the XOR swizzle cancelling between store
and load so it never enters the host formula. `kernel/e2e_layout_gate.py` now checks all five
arms on random data against the NumPy reference with corrupted-operand negative controls.

### Scale granularity is the lever, not the element width

`fp6` makes the point without needing a second model. It carries **more** bits per weight
than any other arm and lands second-to-last at -9.65 (12B) and -7.80 (31B), because it is
the only arm that is not block-scaled: one f32 scale per row rather than one per 32
elements. Thirty-two times coarser scale granularity costs more than 50% more element width
buys. Separately, `fp4` and `w4a6` store weights identically and differ only in activation
format, which is worth ten points at 12B - activation precision dominates weight precision,
and work on weight formats while activations stay 4-bit E8M0 is work on the smaller term.

The same lever explains the E4M3 arms once the scale book is measured rather than assumed.
E4M3 spans 2^-6..240; E8M0 spans 2^-127..2^127. An E2M1 block wants scale `amax/6`, so it
needs `amax >= 0.094` to be representable, and Gemma's weights sit near 0.02. Activations,
at O(1), sit comfortably inside the book and E4M3 delivers the promised **+2.2 to +2.9 dB**
over E8M0 there. So:

- `w4a6 -> w6a6_tc` changes only the **weight** format: pure saturation loss, -5.0 at 31B.
- `fp4 -> w4a4_tc` changes **both**: a real activation gain against the weight loss. At 12B
  the activation gain wins (+7.80); at 31B, where weights saturate harder, the weight loss
  wins (-4.20). Same two effects, opposite sign, no contradiction.

`down_proj` activations fail the other way - amax 3504 puts a few blocks *above* the top of
the book, and a 0.03% clamp rate costs 6.85 dB because it clips the largest values in the
tensor. Both failure modes are what a per-tensor global scale exists to prevent.

### A statistically insignificant gap is not a finding

The 12B row that started this, `w4a6 -0.95` against `w6a6_tc +0.95`, has p=0.32 and p=0.37.
Both arms are within noise of bf16 and therefore within noise of each other, and the 1.90
"advantage" was never real. Reading it as one is what made the 31B result look like a
surprise instead of a correction.

### Per-layer SNR pointed the wrong way

Per-layer SNR put `w4a6` 1-5 dB ahead of `w6a6_tc`; end-to-end accuracy puts them the other
way round. SNR weights every layer and every output element equally and the model does not.
It is a convenient proxy precisely because it needs no model, and that is also why it should
lose to a paired end-to-end measurement whenever the two disagree.

### The same accuracy from two arms is not the same behaviour

Every arm changes bf16's answer on 33-50% of questions while accuracy moves by about a point.
Two readings fit that and they have opposite verdicts, so it could not be left as a
coincidence. Splitting by bf16's own decision margin settles it: the flips land where bf16's
top-two logit gap is ~1.8 nats against ~6.3 nats elsewhere, and where bf16 itself scores
23-37%. Quantization is reshuffling near-ties the model was already getting wrong, not
destroying answers it knew.

It also shows `w4a6`, not `w6a6_tc`, is the arm that tracks bf16 most faithfully (0.673
top-1 agreement vs 0.586). Highest accuracy and closest fidelity are different arms, and an
accuracy table shows only the first.

### End to end, the kernel is no longer what costs time

Across the real 12B linear mix, the quantized arms are ~8x slower than bf16 and **~80% of
that is the torch activation quantizer**, not the GEMM. It is launch-bound, not
bandwidth-bound: scaling M over 8192x makes it only 3.6-4.4x slower, and `torch.compile`
recovers 1.2-3.3x and no more. Quantize-then-GEMM as two passes cannot win here however good
the GEMM is; a fused quantize+GEMM prologue is the only route to an end-to-end speedup. The
plan named this as a risk in advance, and it is now a measurement rather than a suspicion.

A second finding from the same run: all 40 shape/arm combinations missed `best_known.json`,
which is keyed on fused QKV and GATE_UP shapes while stock transformers keeps them unfused.
Accuracy is untouched by this (fusion concatenates along N, every quantization block runs
along K, so not one quantized value changes) but no performance claim should be read off it.

### The transcode is free, and it was not the culprit

A plausible explanation for the E4M3 arms' weakness was that the in-kernel NVFP4->MXFP6
transcode rounds. Measured with the transcode included, `tc` tracks `e4m3` to within
0.05 dB: the transcode adds no error of its own. The hypothesis was cheap to test and
false, which is why it was tested instead of written up.

Full detail in `kernel/opt_trace_s22.md`; accuracy registry and gate in
`kernel/best_known_e2e.json` and `kernel/best_e2e.py`.

## 2026-08-04 - s23: the 6-bit arm was losing on granularity, and one benchmark was scoring option position

s22 ended with the accuracy question answered and three things about the answer unexplained:
the 6-bit arm was the least accurate, the two benchmarks ranked the arms differently, and the
two kernels around the GEMM cost more than the GEMM. None of the three was a property of the
hardware.

### W6A6 came last because it is the only arm not block-scaled

s22 already identified scale granularity as the lever; s23 acted on it. Quantizing the **same**
6-bit elements at both granularities on activations captured from the real model, rowwise is
**3.1x** the error of block-scaled on average and 4.6x on `down_proj`, whose K is longest and
whose outlier channels are worst. The plot's "W6A6" label named the element format because that
is what the kernel work was about, and the variable that decided the ordering was carried only
in prose.

The fix is per-input-channel equalization (`kernel/qsmooth.py`): divide input channel *j* of the
activation by `s_j` and multiply column *j* of the weight by the same `s_j`. That is the identity
in real arithmetic, so it buys granularity for **no kernel change and no runtime cost** - the
weights are folded once at load and the division rides in the quantizer that already reads every
element. With `alpha` in [0.5, 0.6] it cuts the rowwise arm's layer error 2.7x, to 0.0363 against
block-scaled 6-bit's 0.0316: within 15% of an arm carrying 32x more scale metadata. Both
endpoints of the alpha sweep are *worse* than doing nothing, which is the shape a redistribution
lever should have.

### The same fix, calibrated on the wrong text, cost 13 points

Equalized fp6 first scored 28.3% on GPQA - chance is 25% - and 33.6% on MMLU-Pro, against plain
fp6's 39.4% and 46.4%. Mean per-layer SNR had *improved* by 6 dB.

Probing the same 328 layers twice, changing only where the activations came from, located it. On
the sentences the equalizer was calibrated from, 1 layer of 328 got worse. On the actual
benchmark prompts, **33 did**, one of them by 18.9 dB, and all 33 were `mlp.down_proj`. `s` is
frozen at calibration, so a channel that runs hotter at inference is divided by too small an
`s_j` and comes out **larger** - and with one f32 scale per token, that single amplified channel
raises the row's amax and squeezes every other channel in it. **The coarse granularity that
motivated the equalizer is what makes the equalizer's mistakes non-local**, which is also why the
block-scaled arms' equalized variants only wobbled a point or two.

Two candidate causes remained - unrepresentative calibration data, or a formula needing a tighter
clamp - so both were priced in one sweep (2 calibration sets x 2 centrings x 7 caps, scored out
of sample on real prompts). It was the data: calibrating in domain, on the MMLU-Pro **validation**
split and therefore disjoint from every scored item, takes layers-hurt from 33 to 1, the worst
surviving regression from -9.1 dB to -0.3 dB, and mean SNR from 25.7 to 28.7 dB. The clamp is
insurance rather than the fix - 0.06 dB when the calibration is right, +2.9 dB on the worst layer
when it is wrong - so it stays on. Sweeping the clamp alone would have bought +0.14 dB and retired
the method.

Result: **W6A6 is now the most accurate arm.** 54.4% on MMLU-Pro against bf16's 56.7%, a 2.3 point
gap where the next best arm is 3.9 behind, and on GPQA it is inside noise of bf16 (p=0.29).
Equalization is enabled for that arm only: on the block-scaled arms it is a wash or slightly
negative, exactly as the mechanism predicts, and it is not switched on because it helped somewhere.

### GPQA disagreed with MMLU-Pro because it was partly measuring which slot the answer sat in

`fp4` was the best quantized arm on GPQA and the worst on MMLU-Pro, which cannot both describe
the same five kernels. Two harness defects, neither of them a kernel property.

The GPQA items were not rendered alike - `load_bench` took the dataset's question field on trust
assuming the options were inline, and some rows carry two stacked option lists where others carry
one. Both benchmarks now render options through a single prompt builder.

Then the larger effect: with 4 options and bf16 at 1.78x chance, **bf16 was answering "A" on 67%
of items** against a 25% base rate (answer entropy 1.44 of 2.00 bits). Quantization perturbs that
positional prior in a direction unrelated to how much accuracy it costs - and `fp4`, the arm that
looked best, was the *least* position-biased arm to begin with, which is most of what its lead
was. Scoring each item under all 4 cyclic rotations of its options and averaging the per-option
log-probs removes the bias by construction: top-slot share falls to 29%, entropy to 1.99, and
bf16's own score rises from 37.9% to 44.4% on GPQA and 48.75% to 56.70% on MMLU-Pro.

Cross-benchmark rank agreement over the five arms goes from **+0.30 to +0.40** on accuracy and
**+0.70** on mean log P(gold), which uses the whole score vector and is the metric to read for a
ranking. What remains is power, not bias, and the tooling now says so rather than leaving it to
the reader: at n=198 one arm's standard error is 3.5 points, so an unpaired arm-vs-arm gap needs
~9.9 points to mean anything, and both the summary JSON and the tradeoff plot now carry that
band.

### The two kernels around the GEMM

s22 measured the decode step at 5078 us with the GEMM at 1272, the split-K reduction at 1113 and
the activation quantizer at 800. Both of the latter were configuration problems.

The quantizer's tile had been chosen by a sweep scored on **prefill** shapes; at M=1 its 64-block
tile leaves 2 programs on a 256-CU part. Re-tuned per M it is **1.85x** faster at M=1 (1.74x at
M=8, 1.57x at M=32), bit-exact at every M. The split-K reduction was calling `torch.sum`, which
cost 2.70-2.98 us to move about 1 us of bytes, against an empty Triton launch at 1.77 us - almost
all dispatch. A hand-written reduce lands at 1.85-1.91 us, **0.08 us above an empty launch**, and
is more accurate (f32 plane accumulation where torch rounds each partial to bf16), so it is
deliberately not bit-exact against what it replaced and is gated against f64 instead.

One fusion was built and rejected: collapsing the fp6 arm's two quantizer kernels into one is
byte-exact and **0.76x**. The premise was wrong in an instructive way - two launches inside a
CUDA graph overlap, so the pair already ran at or under what two empty serial launches cost, and
there was no launch to save; meanwhile the reduction must see all of K, so the program that emits
has to be the program that reduced, pinning the grid to one program per row and discarding the
K-parallelism the emit pass wants. The flag is off with the measurement table in the code.

**Not confirmed:** the vLLM re-measurement never completed. The node's own nightly vLLM
regression suite SIGKILLs any container holding a GPU and killed the sweep during weight load
60+ times, so the standing end-to-end numbers are still s22-era (1.21-1.22x bf16 at batch 32,
1.24-1.27x at batch 8) and the tradeoff plot's throughput axis is pre-optimization. The kernel
measurements predict that improves; a prediction is not a measurement.

Full detail in `kernel/opt_trace_s23.md`. Accuracy cells are now keyed by scoring protocol
(`mmlu_pro` vs `mmlu_pro@p4`) so s22's numbers keep meaning what they meant while the debiased
ones gate alongside them.

