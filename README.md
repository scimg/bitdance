# bitdance

This repository contains the reproducible notes, kernels, and benchmark scripts for the NVFP4-storage to MXFP6-compute transcode experiment on AMD MI355 / CDNA4.

## Main entry points

- `REPRODUCE.md` - how to reproduce the kernel builds, correctness checks, and benchmark runs.
- `PROGRESS.md` - chronological project notes and performance analysis.
- `status.md` - latest optimization status and conclusions.
- `kernel/opt_log.md` - detailed kernel optimization log and rollback trace.
- `kernel/hip/w4a6_tc_gemm_v31.hip` - current HIP transcode kernel with the latest optimization paths.
- `kernel/bench_full.sh` - unified benchmark driver.

## Current focus

The project evaluates storing B weights in compact NVFP4-like FP4 format, transcoding on chip to MXFP6/E2M3 fragments, and feeding CDNA4 scaled MFMA instructions.

The latest HIP work includes:

- packed B scale sideband (`s0 | (s1 << 3)`),
- assembly-assisted pack path (`ASMPACK`),
- native HIP W6A6 mode for isolating the tiling ceiling,
- HIP-only dispatch experiments, and
- comparison against FlyDSL/W4A6/W6A6 baselines.

See `REPRODUCE.md` and `kernel/opt_log.md` for details.
