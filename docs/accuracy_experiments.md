# Accuracy experiment files

This document lists the accuracy/quality experiment code included in this repository.
It intentionally includes the runnable entry points and compact summary outputs, not all
large generated result directories.

## Scalar and block quantization sweeps

- `quant_error_sweep.py`
  - Sweeps scalar values and compares direct NVFP4 / MXFP4 / MXFP6 / FP4SE formats and transcode recipes.
  - Produces `quant_error.png` when run in the full local environment.

- `vector_quant_error.py`
  - NumPy block/vector quantization experiment with block size 32.
  - Compares direct storage formats and transcode recipes on bounded random distributions.

- `vector_quant_error.txt`
  - Saved output from the block/vector experiment.

- `vector_quant_error_agg.csv`
  - Compact aggregate table for vector quantization error.

- `vector_block_dist.csv`
  - Compact distribution summary used by the vector/block analysis.

## Real-weight quantization error analysis

- `remote/quant_real_weights.py`
  - Quantizes real model weights and evaluates error behavior.

- `remote/error_stats.py`
  - Computes compact error statistics for real-weight experiments.

- `remote/error_distribution.py`
  - Analyzes error distributions.

- `remote/block_error_dist.py`
  - Computes block-level error/SQNR distribution statistics.

- `plots/error_stats.csv`
- `plots/error_stats.txt`
  - Small summary outputs from the real-weight error study.

Large generated plots and full output folders are intentionally excluded from the first GitHub import.

## End-to-end / logit / fidelity experiments

- `kernel/eval_analyze.py`
  - Analyzes generated/evaluation outputs.

- `kernel/eval_bench.py`
  - Evaluation benchmark harness.

- `kernel/eval_gen.py`
  - Generation/evaluation runner utilities.

- `kernel/logit_agreement.py`
  - Logit agreement checks between quantization variants.

- `kernel/p137_error_recheck.py`
  - Recheck script for published error numbers.

- `kernel/p72_fidelity.py`
  - Fidelity analysis helper.

- `kernel/p76_agree.py`
  - Agreement metric helper.

- `kernel/quant_headroom.py`
  - Quantization headroom analysis.

- `kernel/run_eval_all.sh`
  - Convenience script for running the evaluation suite.

## Notes

The full local tree contains many large intermediate JSONL outputs, plots, model-specific
runs, and scratch scripts. Those are not part of this minimal upload. Add them later only if
a reviewer needs a specific raw result bundle.
