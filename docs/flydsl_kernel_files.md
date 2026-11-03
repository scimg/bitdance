# FlyDSL kernel files included

This repository intentionally does not vendor the full FlyDSL/AITER source tree.
It includes only the final usable kernel entry point and the minimal reference
kernel files needed to understand and reproduce the MXFP6/NVFP4 preshuffle GEMM
work.

## Project entry point

- `psg_preshuffle_gemm.py`

This is the editable local integration point used by the benchmark scripts.  It
contains the `fp4`, `w4a6`, `fp6`, and `w6a6_tc` paths used in this work.

## Reference FlyDSL kernel snapshots

- `ref_flydsl_src/kernels/preshuffle_gemm.py`
- `ref_flydsl_src/kernels/blockscale_preshuffle_gemm.py`
- `ref_flydsl_src/kernels/mfma_preshuffle_pipeline.py`
- `ref_flydsl_src/kernels/mfma_epilogues.py`
- `ref_flydsl_src/kernels/kernels_common.py`
- `ref_flydsl_src/kernels/__init__.py`

These files provide the upstream-style FlyDSL preshuffle GEMM context used for
comparison and porting notes.

## AITER FlyDSL reference snapshots

- `ref_aiter_flydsl/preshuffle_gemm.py`
- `ref_aiter_flydsl/mfma_preshuffle_pipeline.py`
- `ref_aiter_flydsl/mfma_epilogues.py`
- `ref_aiter_flydsl/kernels_common.py`
- `ref_aiter_flydsl/__init__.py`

These are compact reference files for the AITER-style FlyDSL kernel structure.

For full environment setup, use `REPRODUCE.md`.  The full upstream dependency is
expected to come from the ROCm/FlyDSL/AITER environment rather than this repo.
