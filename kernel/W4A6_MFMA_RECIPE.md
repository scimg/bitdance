# W_MXFP4 × A_MXFP6 scaled MFMA recipe (gfx950 / MI355)

Verified bit-exact via `kernel/w4a6_probe.py` (single 16×16×128 tile, 1 wave).
Intrinsic: `rocdl.mfma_scale_f32_16x16x128_f8f6f4`
Call operands: `[a128, b128, acc_f32x4, cbsz, blgp, opselA, scaleA_i32, opselB, scaleB_i32]`

## Format select
- `cbsz` / `blgp` encode operand fmt: 0=FP8/E4M3, 1=FP8/E5M2, **2=FP6/E2M3**, 3=FP6/E3M2, **4=FP4/E2M1**.
- W4A6: **A = FP6/E2M3 → cbsz=2**, **B = FP4/E2M1 → blgp=4**.

## Fragment layout (64-lane wave), lane `l`: `g=l//16`, `row=l%16`
- Operand element index `p in [0,32)` → contraction k-index:
  - `k = g*16 + p`            if `p < 16`
  - `k = 64 + g*16 + (p-16)`  if `p >= 16`
  (i.e. two 16-wide K strips at offsets `g*16` and `64+g*16`; 32 elems/lane → full K=128.)
- A operand register `a128` (`vector<8xi32>` = 32 bytes):
  - FP6: 32 elems densely packed 6-bit → **24 real bytes + 8 zero pad**. (`a0,a1,a2` i64 + 0)
  - FP4: 32 elems densely packed 4-bit → **16 real bytes + 16 zero pad**.
  - FP8: 32 bytes (full).
  - Dense bit packing: element `p` occupies bits `[p*w, p*w+w)` little-endian (`w`=6 for fp6, 4 for fp4).
- Output: `C_out[l*4 + ii] = C[g*4 + ii, l%16]`, `ii in 0..3` (result is `f32x4` per lane).

## Scaling (KEY finding, matches Phase-2 HIP)
- The scaled MFMA applies **ONE E8M0 scale per operand per lane over that lane's 32 elements
  (= the whole instruction K=128)**. NOT 4 independent per-32 sub-block scales inside one call.
- `scaleA`/`scaleB` are per-lane i32 (up to 4 E8M0 bytes); **`opselA`/`opselB` (0..3) select which
  byte** is used. Verified: with scale bytes `[2^0,2^1,2^2,2^3]`, `C[0,0] = 128 * 2^opselA`.
- Therefore for MXFP6/MXFP4 per-32 microscaling, the DATA must be arranged so each lane's 32
  operand elements = one logical MX-32 block, and its E8M0 is supplied in the opsel-selected byte.
  (This is exactly what the fp4 preshuffle kernel's `shuffle_scale_w4` / lane-indexed scale load do.)

## Consequences for the full kernel (fork of preshuffle_gemm.py fp4 path)
- Keep B = fp4 preshuffle load + scale path unchanged (`blgp=4`).
- Change A: load 24 bytes/lane (a0,a1,a2) instead of 16 (fp4) → `a128 = pack_i64x4_to_i32x8(a0,a1,a2,0)`; `cbsz=2`.
- Store A activations in "fp8-padded fp6" tile layout: per lane block = 24 dense fp6 bytes + 8 zero
  (32-byte stride) so fp8-style tile/LDS load machinery is reused; MFMA reads the 24 real bytes.
- For the fused path, the NVFP4→MXFP6 prologue transcode produces this padded fp6 A-tile directly.
