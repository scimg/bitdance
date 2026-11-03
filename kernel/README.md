# Fused NVFP4 → MXFP6 GEMM (gfx950 / MI350–MI355)

> **This file documents the original from-scratch correctness kernel** (`nvfp4_mxfp6_gemm.py`,
> peaking at 221 TFLOPS). It is kept for the transcode contract and the MFMA recipe, both of
> which are still current. **The production path is elsewhere:** `psg_preshuffle_gemm.py`, an
> editable copy of the FlyDSL preshuffle GEMM, selected by `in_dtype`:
>
> | `in_dtype` | transcodes | notes |
> |---|---|---|
> | `w4a6_tc` (alias of the historical `w6a6_tc`) | B only | production; A is already MXFP6 |
> | `w4a4_tc` | A **and** B | both operands stored NVFP4; 0.77–0.91× `w4a6_tc` |
> | `w6a6` | nothing | both operands stored MXFP6 with per-32 E8M0; the block-scaled 6-bit ceiling (s24). 8 bits/weight in HBM — the fp6 tile is byte-padded — which is the trade `w4a6_tc` exists to avoid |
> | `fp4`, `w4a6`, `fp6` | nothing | reference arms. **`fp6` is per-row f32, not MX**: it hands the MFMA a scale operand of 1.0. Use `w6a6` for a 6-bit arm comparable to the others |
>
> See `REPRODUCE.md` §3 for the entry point and the tuning flags (`TC_HWCVT`, `WG512`, …),
> `PROGRESS.md` for the dated history, and `kernel/opt_trace_s1*.md` for the per-session traces.

Runtime-fused GEMM that stores both operands in **NVFP4** (4-bit E2M1 + per-32
E4M3 block scale) and **transcodes them to MXFP6** (6-bit E2M3 + per-32 E8M0)
*inside the kernel prologue*, feeding the CDNA4 scaled MFMA. No separate
transcode pass and no MXFP6 tensors ever touch DRAM.

Built on FlyDSL, using the native scaled instruction
`v_mfma_scale_f32_16x16x128_f8f6f4` with `cbsz=2 / blgp=2` (fp6 × fp6).

## Files

| File | Purpose |
|------|---------|
| `nvfp4_mxfp6_gemm.py` | Canonical correctness kernel (both operands fused, 1 wave/16×16 tile) + numpy golden. |
| `nvfp4_mxfp6_gemm_tiled.py` | Register-blocked (`BM×BN` macro-tile/wave) throughput kernel; `python … --bm 128 --bn 128` self-checks. |
| `nvfp4_mxfp6_op.py` | Torch entry point: `matmul(A_codes, A_scale, B_codes, B_scale)`. |
| `bench_nvfp4_mxfp6.py` | TFLOPS harness (`--module …`, `--bm/--bn`). |
| `transcode_lut.py`, `nvfp4_codec.py` | 64/128-entry transcode LUT + bit-exact golden codec. |
| `W4A6_MFMA_RECIPE.md` | Verified MFMA fragment layout + scale semantics. |
| `w4a6_gemm.py`, `w4a6_fused.py` | Earlier W4A6 (fp6 A / fp4 B) reference variants. |

## Usage

```python
import torch
from kernel.nvfp4_mxfp6_op import matmul

# A: activation NVFP4 [M,K] nibbles + [M,K//32] E4M3 scale bytes
# B: weight     NVFP4 [N,K] nibbles + [N,K//32] E4M3 scale bytes   (B stored as B^T)
C = matmul(A_codes, A_scale, B_codes, B_scale, tile=128)   # -> float32 [M,N]
```

Constraints: `K % 128 == 0`; `M`, `N` divisible by the chosen tile (128/64/32/16).

## Transcode contract (bit-exact vs `nvfp4_codec.golden_mxfp6`)

Per element (sign folded into the table):

```
e2m3_code = LUT6[mant_bits*16 + e2m1_code]          # 128-entry byte table
e8m0      = ((scale >> 3) & 0xF) + 120 + (mant_bits != 0)   # == exp + bump + 127
```

`mant_bits = scale & 7`, `exp_field = (scale>>3)&0xF`. The kernel then densely
packs the 32 six-bit codes of a lane into 24 bytes (6 i32 words) and issues the
scaled MFMA. Each lane's 32 elements map to exactly one contiguous MX-32 block
(preshuffle: operand pos `p` of lane `(g,row)` ← source `K = k128*128 + g*32 + p`),
so one E8M0 per lane/operand is exact.

## Correctness

Bit-exact (rel-L2 = 0) vs the dequantized-MXFP6 matmul golden at multiple sizes
and tiles; `rel-L2` vs the FP32 matmul equals the MXFP6×MXFP6 emulation error
(~0.15 for N(0,1) inputs) — i.e. the fused kernel adds **zero** error beyond
quantization itself.

## Performance (4096×4096, MI355, this implementation)

| Config | TFLOPS |
|--------|-------:|
| 1 MFMA / wave | 23 |
| register block 32×32 | 51 |
| 64×64 (per-nibble LUT) | 99 |
| 128×128 (per-nibble LUT) | 146 |
| 128×128 + **byte-pair LUT** | **221** |

Reference baselines on the same GPU (no transcode): FlyDSL MXFP4 preshuffle
GEMM = **3369**, FP8 preshuffle = 1794, CK native MXFP6 = 1133 TFLOPS.

### What limits it (measured decomposition)

The same register-blocked kernel with the transcode **removed** (load pre-packed
fp6 directly) runs at **850 TFLOPS** at 128×128. So the fused-vs-baseline gap is
two independent factors:

* **Transcode overhead** (850 → 221): the runtime NVFP4→MXFP6 transcode is
  VALU-heavy. Throughput is K-invariant and grows monotonically with tile size
  (better transcode:MFMA reuse ratio `1/MR + 1/NR`), i.e. **VALU-throughput-bound**,
  not occupancy/bandwidth. The byte-pair LUT (2 nibbles → one 12-bit result) cut
  op-count ~2× (146 → 221). Register pressure caps the single-wave tile at
  128×128 (256×256 spills → 59 TFLOPS).
* **Implementation maturity** (3369 → 850): even transcode-free, this from-scratch
  kernel is ~4× under the production baseline (no software pipelining / LDS
  double-buffering / tuned epilogue).

## Optimization roadmap (future work)

1. **Multi-wave blocks with LDS operand sharing** — transcode each A/B strip once
   and share across waves, pushing the effective tile past the per-wave register
   limit (the main remaining transcode-reuse lever).
2. **Software-pipeline / LDS double-buffer** the K loop so transcode VALU overlaps
   MFMA issue (overlap ceiling ≈ 300 TFLOPS at the current op-count).
3. **Pre-transcode static weights**: in inference B is constant, so its transcode
   can be amortized once (halves per-GEMM transcode) if the use case allows.
4. bf16 epilogue + `cshuffle`-style coalesced stores.
