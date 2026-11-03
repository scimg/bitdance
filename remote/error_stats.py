#!/usr/bin/env python3
"""Whole-model quantization-error statistics for every method.

Parallel over layers. Each worker accumulates *exact* sufficient statistics
over all 2-D weights of a layer for every method:
  - sum(err), sum(err^2), sum(|err|), max|err|              (moments)
  - log-spaced histogram of |err|/amax                      (percentiles)
  - sum(err^2) and count per |w|/amax decile                (error vs magnitude)
plus the method-independent sum(w^2) and global amax.

The main process aggregates across layers and prints:
  - a per-method table: SQNR(dB), rel-L2, bias, RMS, MAE, |err| percentiles
    (bias/RMS/MAE/percentiles are expressed as % of the per-layer amax)
  - an error-RMS-vs-weight-magnitude table (% of amax) per decile.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

import quant_real_weights as Q

# |err|/amax percentile bins (log-spaced)
PB = np.logspace(-6.0, 0.0, 601)
NMAG = 10  # |w|/amax deciles
_STATE: dict = {}


build_index = Q.build_layer_index      # layout resolution lives in one place; see its docstring


def method_names():
    return [name for name, _, _ in Q.METHODS]


def method_map():
    return {name: (fn, gbook) for name, fn, gbook in Q.METHODS}


def _init(shard_paths, layers, min_elems):
    _STATE["shard_paths"] = shard_paths
    _STATE["layers"] = layers
    _STATE["min_elems"] = min_elems
    _STATE["files"] = {}
    _STATE["mmap"] = method_map()


def _open(shard):
    files = _STATE["files"]
    if shard not in files:
        files[shard] = Q.SafeTensorsFile(_STATE["shard_paths"][shard])
    return files[shard]


def _load_layer_flat(tmap, min_elems):
    chunks = []
    for name in sorted(tmap):
        stf = _open(tmap[name])
        if len(tuple(stf.meta(name)["shape"])) != 2:
            continue
        w = stf.get_float32(name)
        cols = w.shape[1]
        keep = (cols // Q.BLOCK_SIZE) * Q.BLOCK_SIZE
        if keep == 0:
            continue
        if keep != cols:
            w = w[:, :keep]
        flat = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
        if flat.size < min_elems:
            continue
        chunks.append(flat)
    return np.concatenate(chunks)


def _work(lid: int):
    tmap = _STATE["layers"][lid]
    mmap = _STATE["mmap"]
    flat = _load_layer_flat(tmap, _STATE["min_elems"])
    aw = np.abs(flat)
    amax = float(aw.max()) or 1.0
    n = flat.size
    sw2 = float(np.dot(flat, flat))
    # per-element magnitude decile index in [0, NMAG-1]
    mag = np.minimum((aw / np.float32(amax) * NMAG).astype(np.int64), NMAG - 1)
    out = {}
    for name, (fn, gbook) in mmap.items():
        pred = Q.run_method(fn, gbook, flat)
        err = pred - flat
        ae = np.abs(err)
        se = float(err.sum())
        se2 = float(np.dot(err, err))
        sae = float(ae.sum())
        mxe = float(ae.max())
        ph, _ = np.histogram(ae / np.float32(amax), bins=PB)
        # error^2 and count per magnitude decile
        mse2 = np.bincount(mag, weights=err.astype(np.float64) ** 2, minlength=NMAG)
        mcnt = np.bincount(mag, minlength=NMAG)
        out[name] = dict(se=se, se2=se2, sae=sae, mxe=mxe,
                         ph=ph.astype(np.int64), mse2=mse2, mcnt=mcnt)
    return lid, n, sw2, amax, out


def _pct_from_hist(counts: np.ndarray, q: float) -> float:
    tot = counts.sum()
    if tot == 0:
        return 0.0
    target = q * tot
    cum = np.cumsum(counts)
    idx = int(np.searchsorted(cum, target))
    idx = min(idx, len(counts) - 1)
    return float((PB[idx] * PB[idx + 1]) ** 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="out/error_stats.txt")
    ap.add_argument("--csv", default="out/error_stats.csv")
    ap.add_argument("--min-elems", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    layers = build_index(args.repo)
    ids = sorted(layers)
    names = method_names()
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    print(f"repo={args.repo} layers={len(ids)} methods={len(names)}", flush=True)

    # Q.build_layer_index already yields ABSOLUTE LOCAL paths, so this map is the identity. It used
    # to re-download by filename through the hub, which cannot work for a cache holding the weights
    # but not the index -- the Gemma 4 12B case, and the reason this sweep had no 12B column.
    shard_paths = {s: s for lid in ids for s in layers[lid].values()}

    agg = {m: dict(se=0.0, se2=0.0, sae=0.0, mxe=0.0,
                   ph=np.zeros(len(PB) - 1, np.int64),
                   mse2=np.zeros(NMAG), mcnt=np.zeros(NMAG)) for m in names}
    tot_n = 0
    tot_sw2 = 0.0
    amax_g = 0.0

    nworkers = max(1, min(args.workers, len(ids), os.cpu_count() or 1))
    with mp.Pool(nworkers, initializer=_init,
                 initargs=(shard_paths, layers, args.min_elems)) as pool:
        for lid, n, sw2, amax, out in pool.imap_unordered(_work, ids):
            tot_n += n
            tot_sw2 += sw2
            amax_g = max(amax_g, amax)
            for m in names:
                a, o = agg[m], out[m]
                a["se"] += o["se"]; a["se2"] += o["se2"]; a["sae"] += o["sae"]
                a["mxe"] = max(a["mxe"], o["mxe"])
                a["ph"] += o["ph"]; a["mse2"] += o["mse2"]; a["mcnt"] += o["mcnt"]
            print(f"layer {lid:3d} done", flush=True)

    # ---- build report ---------------------------------------------------- #
    L = []
    L.append(f"Whole-model quantization-error statistics: {args.repo}")
    L.append("=" * 100)
    L.append(f"layers={len(ids)}  elements/method={tot_n:,}  block={Q.BLOCK_SIZE}  "
             f"global amax={amax_g:.4f}")
    L.append(f"bias/RMS/MAE/max are % of the GLOBAL amax ({amax_g:.4f}); the |err| percentiles "
             f"and the decile table are % of each layer's OWN amax -- two different "
             f"denominators, do not compare across that boundary")
    L.append("rel-L2 = ||err||/||w||; SQNR and rel-L2 are ratios and need no amax at all")
    L.append("bias is the MEAN error, and it is the column to watch: a coherent error grows as "
             "K down a GEMM's reduction while a random one grows as sqrt(K), so a bias far "
             "below the RMS can still dominate the output (kernel/p140_output_snr.py)")
    L.append("")
    hdr = (f"{'method':<28}{'SQNR(dB)':>9}{'relL2%':>9}{'bias%':>9}"
           f"{'RMS%':>8}{'MAE%':>8}{'p50%':>8}{'p99%':>8}{'p99.9%':>9}{'max%':>9}")
    L.append(hdr)
    L.append("-" * len(hdr))
    csv = ["method,sqnr_db,rel_l2_pct,bias_pct,rms_pct,mae_pct,p50_pct,p99_pct,p999_pct,max_pct"]
    rmsn = tot_sw2 ** 0.5

    def row(m):
        a = agg[m]
        rel_l2 = (a["se2"] / tot_sw2) ** 0.5 if tot_sw2 else 0.0
        sqnr = -20.0 * np.log10(rel_l2) if rel_l2 > 0 else float("inf")
        # NORMALIZATION. These four divide by the GLOBAL amax, while the percentile and decile
        # columns below come from histograms `_work` built against each layer's OWN amax. Both
        # are defensible, but they are not the same denominator, and the header used to claim
        # per-layer for all of them -- so RMS% and p50% sat in one row while measuring against
        # different scales, and a reader comparing them across columns was comparing nothing.
        # Left as global (a single stated scale for the whole model) and now labelled as such.
        bias = a["se"] / tot_n / amax_g
        rms = (a["se2"] / tot_n) ** 0.5 / amax_g
        mae = a["sae"] / tot_n / amax_g
        p50 = _pct_from_hist(a["ph"], 0.50)
        p99 = _pct_from_hist(a["ph"], 0.99)
        p999 = _pct_from_hist(a["ph"], 0.999)
        mx = a["mxe"] / amax_g
        return sqnr, rel_l2, bias, rms, mae, p50, p99, p999, mx

    for m in names:
        s, rl, b, r, mae, p50, p99, p999, mx = row(m)
        L.append(f"{m:<28}{s:>9.2f}{100*rl:>9.3f}{100*b:>+9.4f}"
                 f"{100*r:>8.3f}{100*mae:>8.3f}{100*p50:>8.3f}"
                 f"{100*p99:>8.3f}{100*p999:>9.3f}{100*mx:>9.3f}")
        csv.append(f"{m},{s:.3f},{100*rl:.4f},{100*b:.5f},{100*r:.4f},"
                   f"{100*mae:.4f},{100*p50:.4f},{100*p99:.4f},{100*p999:.4f},{100*mx:.4f}")

    # ---- error RMS vs |w| decile ---------------------------------------- #
    subset = [m for m in [
        "Direct MXFP6", "Direct MXFP4", "Direct NVFP4 (2lvl)",
        "Direct NVFP4 b16 (2lvl)", "Direct FP4SE4 (2lvl)",
        "NVFP4->MXFP6 revised (2lvl)", "NVFP4->MXFP6 naive (2lvl)"] if m in agg]
    L.append("")
    L.append("Error RMS (% of amax) by |w|/amax decile  [d0=smallest .. d9=largest]")
    L.append("-" * 100)
    dh = f"{'method':<28}" + "".join(f"{'d'+str(i):>7}" for i in range(NMAG))
    L.append(dh)
    for m in subset:
        a = agg[m]
        cnt = np.maximum(a["mcnt"], 1)
        rms_d = np.sqrt(a["mse2"] / cnt) / amax_g * 100.0
        L.append(f"{m:<28}" + "".join(f"{v:>7.3f}" for v in rms_d))

    report = "\n".join(L)
    print("\n" + report, flush=True)
    with open(args.out, "w") as fh:
        fh.write(report + "\n")
    with open(args.csv, "w") as fh:
        fh.write("\n".join(csv) + "\n")
    print(f"\nwrote {args.out} and {args.csv}", flush=True)


if __name__ == "__main__":
    main()
