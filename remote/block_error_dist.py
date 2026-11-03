#!/usr/bin/env python3
"""Distribution-aware weight-quantization error over the whole model.

A single averaged rel-L2 hides everything: per-layer rel-L2 is essentially
constant across depth, yet the worst-case tail differs 3x between methods.
This script reports the *distribution* of the reconstruction error instead of
one number, using exact (sampling-free) accumulators over every 2-D weight.

Parallel over layers (one layer per worker, like error_stats.py). Per layer,
for each plotted method it accumulates:
  * per-block SQNR(dB) histogram          -> spread across the 32-elem blocks
  * per-block rel-L2 histogram (log bins)  -> box/whisker percentiles
  * per-element |err|/amax histogram (log) -> exact complementary CDF (tail)
plus block counts and totals. The block (32 elements) is the quantization and
compute unit, so its error distribution is the object that actually matters.

The main process sums the histograms across layers and emits:
  * out/block_error_dist.txt / .csv   (per-block percentiles + tail fractions)
  * block_sqnr_hist.png    overlaid per-block SQNR distributions
  * err_ccdf.png           P(|err|/amax > t), log-log tail comparison
  * block_box.png          per-method box/whisker of per-block rel-L2 (%)
  * blocks_below_sqnr.png  % of blocks under an SQNR threshold (tail scalar)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

import quant_real_weights as Q

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PLOT_METHODS = [
    "Direct MXFP6",
    "Direct MXFP4",
    "Direct NVFP4 (2lvl)",
    "Direct NVFP4 b16 (2lvl)",
    "NVFP4->MXFP6 revised (2lvl)",
    "Direct FP4SE4 (2lvl)",
    "FP4SE4->MXFP6 (2lvl)",
    "Direct FP4SE5 (2lvl)",
    "FP4SE5->MXFP6 (2lvl)",
    "Direct FP4SE6 (2lvl)",
    "FP4SE6->MXFP6 (2lvl)",
]

# Fixed bin edges (shared by all workers so counts are summable).
SB = np.linspace(-20.0, 80.0, 501)          # per-block SQNR, dB
RB = np.logspace(-3.0, 0.5, 701)            # per-block rel-L2 (0.001 .. ~3.16)
PB = np.logspace(-6.0, 0.0, 601)            # |err|/amax (tail CDF)
SQNR_THRESHOLDS = (10.0, 15.0, 20.0)        # dB, for "fraction of bad blocks"

_STATE: dict = {}

build_index = Q.build_layer_index      # layout resolution lives in one place; see its docstring


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
    amax = float(np.max(np.abs(flat))) or 1.0

    xb = flat.reshape(-1, Q.BLOCK_SIZE).astype(np.float64)
    x2 = np.einsum("ij,ij->i", xb, xb)          # per-block signal energy
    nonzero = x2 > 0.0
    n_blocks = int(nonzero.sum())

    ltype = "attn" if any(".self_attn." in n for n in tmap) else (
        "linear" if any(".linear_attn." in n for n in tmap) else "other")

    out = {}
    for m in PLOT_METHODS:
        fn, gbook = mmap[m]
        pred = Q.run_method(fn, gbook, flat)
        err = (pred - flat).astype(np.float64)

        eb = err.reshape(-1, Q.BLOCK_SIZE)
        e2 = np.einsum("ij,ij->i", eb, eb)      # per-block error energy
        e2n = e2[nonzero]
        x2n = x2[nonzero]
        rel = np.sqrt(e2n / x2n)                # per-block rel-L2
        # SQNR(dB); perfect blocks (e2==0) -> +inf, clipped into the top bin.
        with np.errstate(divide="ignore"):
            sqnr = 10.0 * np.log10(np.where(e2n > 0.0, x2n / e2n, np.inf))
        sqnr = np.clip(sqnr, SB[0] + 1e-9, SB[-1] - 1e-9)

        sqnr_h, _ = np.histogram(sqnr, bins=SB)
        rel_h, _ = np.histogram(np.clip(rel, RB[0], RB[-1] - 1e-12), bins=RB)
        tail_h, _ = np.histogram(np.abs(err) / np.float64(amax), bins=PB)

        out[m] = dict(
            sqnr_h=sqnr_h.astype(np.int64),
            rel_h=rel_h.astype(np.int64),
            tail_h=tail_h.astype(np.int64),
            max_rel=float(rel.max()) if rel.size else 0.0,
            max_abs_norm=float(np.max(np.abs(err)) / amax),
        )
    return lid, ltype, flat.size, n_blocks, out


# --------------------------------------------------------------------------- #
# percentile helpers on fixed histograms
# --------------------------------------------------------------------------- #
def _quantile(counts: np.ndarray, edges: np.ndarray, q: float, log: bool) -> float:
    tot = counts.sum()
    if tot == 0:
        return float("nan")
    cum = np.cumsum(counts)
    idx = int(np.searchsorted(cum, q * tot))
    idx = min(idx, len(counts) - 1)
    lo, hi = edges[idx], edges[idx + 1]
    return float((lo * hi) ** 0.5) if log else float(0.5 * (lo + hi))


def _frac_below(sqnr_counts: np.ndarray, thr: float) -> float:
    tot = sqnr_counts.sum()
    if tot == 0:
        return 0.0
    below = sqnr_counts[SB[1:] <= thr].sum()
    return float(below) / float(tot)


def _ccdf(tail_counts: np.ndarray):
    tot = tail_counts.sum()
    if tot == 0:
        return PB[1:], np.zeros(len(PB) - 1)
    surv = 1.0 - np.cumsum(tail_counts) / float(tot)
    return PB[1:], np.clip(surv, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="out/block_error_dist.txt")
    ap.add_argument("--csv", default="out/block_error_dist.csv")
    ap.add_argument("--plotdir", default="out")
    ap.add_argument("--min-elems", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="SQNR(dB) threshold for the blocks_below_sqnr bar")
    args = ap.parse_args()

    layers = build_index(args.repo)
    ids = sorted(layers)
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.plotdir).mkdir(parents=True, exist_ok=True)
    print(f"repo={args.repo} layers={len(ids)} methods={len(PLOT_METHODS)}", flush=True)

    # build_index already resolved every name to a LOCAL path, so the shard map is the identity.
    # It used to re-download by filename through the hub, which cannot work for a cache that has the
    # weights but not the index -- the case that matters here.
    shard_paths = {s: s for lid in ids for s in layers[lid].values()}

    agg = {m: dict(sqnr_h=np.zeros(len(SB) - 1, np.int64),
                   rel_h=np.zeros(len(RB) - 1, np.int64),
                   tail_h=np.zeros(len(PB) - 1, np.int64),
                   max_rel=0.0, max_abs_norm=0.0) for m in PLOT_METHODS}
    tot_elems = 0
    tot_blocks = 0

    nworkers = max(1, min(args.workers, len(ids), os.cpu_count() or 1))
    with mp.Pool(nworkers, initializer=_init,
                 initargs=(shard_paths, layers, args.min_elems)) as pool:
        for lid, ltype, nel, nbl, out in pool.imap_unordered(_work, ids):
            tot_elems += nel
            tot_blocks += nbl
            for m in PLOT_METHODS:
                a, o = agg[m], out[m]
                a["sqnr_h"] += o["sqnr_h"]
                a["rel_h"] += o["rel_h"]
                a["tail_h"] += o["tail_h"]
                a["max_rel"] = max(a["max_rel"], o["max_rel"])
                a["max_abs_norm"] = max(a["max_abs_norm"], o["max_abs_norm"])
            print(f"layer {lid:3d} [{ltype:6s}] done", flush=True)

    # ---- text / csv summary --------------------------------------------- #
    L = []
    L.append(f"Per-block weight-error distribution: {args.repo}")
    L.append("=" * 96)
    L.append(f"layers={len(ids)}  elements/method={tot_elems:,}  "
             f"blocks/method={tot_blocks:,}  block={Q.BLOCK_SIZE}")
    L.append("per-block rel-L2 percentiles (%); SQNR percentiles (dB); "
             "tail = |err|/amax percentiles (%)")
    L.append("")
    hdr = (f"{'method':<30}{'relL2 p10':>10}{'p50':>8}{'p90':>8}{'p99':>8}"
           f"{'max':>8}{'SQNR p50':>10}{'SQNRp10':>9}"
           + "".join(f"{'<'+str(int(t))+'dB%':>9}" for t in SQNR_THRESHOLDS))
    L.append(hdr)
    L.append("-" * len(hdr))
    csv = ["method,rell2_p10,rell2_p50,rell2_p90,rell2_p99,rell2_max,"
           "sqnr_p50,sqnr_p10,"
           + ",".join(f"blocks_below_{int(t)}db_pct" for t in SQNR_THRESHOLDS)
           + ",tail_p99,tail_p999,tail_max"]

    for m in PLOT_METHODS:
        a = agg[m]
        r10 = 100 * _quantile(a["rel_h"], RB, 0.10, log=True)
        r50 = 100 * _quantile(a["rel_h"], RB, 0.50, log=True)
        r90 = 100 * _quantile(a["rel_h"], RB, 0.90, log=True)
        r99 = 100 * _quantile(a["rel_h"], RB, 0.99, log=True)
        rmax = 100 * a["max_rel"]
        s50 = _quantile(a["sqnr_h"], SB, 0.50, log=False)
        s10 = _quantile(a["sqnr_h"], SB, 0.10, log=False)
        fr = [100 * _frac_below(a["sqnr_h"], t) for t in SQNR_THRESHOLDS]
        t99 = 100 * _quantile(a["tail_h"], PB, 0.99, log=True)
        t999 = 100 * _quantile(a["tail_h"], PB, 0.999, log=True)
        tmax = 100 * a["max_abs_norm"]
        L.append(f"{m:<30}{r10:>10.3f}{r50:>8.3f}{r90:>8.3f}{r99:>8.3f}"
                 f"{rmax:>8.2f}{s50:>10.2f}{s10:>9.2f}"
                 + "".join(f"{v:>9.3f}" for v in fr))
        csv.append(f"{m},{r10:.4f},{r50:.4f},{r90:.4f},{r99:.4f},{rmax:.4f},"
                   f"{s50:.3f},{s10:.3f},"
                   + ",".join(f"{v:.4f}" for v in fr)
                   + f",{t99:.4f},{t999:.4f},{tmax:.4f}")

    report = "\n".join(L)
    Path(args.out).write_text(report + "\n")
    Path(args.csv).write_text("\n".join(csv) + "\n")
    print("\n" + report, flush=True)
    print(f"\nwrote {args.out} and {args.csv}", flush=True)

    # ---- plots ----------------------------------------------------------- #
    pdir = Path(args.plotdir)
    colors = plt.cm.tab20(np.linspace(0, 1, len(PLOT_METHODS)))

    # 1) per-block SQNR distribution (density, step)
    plt.figure(figsize=(9, 5.2))
    for m, c in zip(PLOT_METHODS, colors):
        cnt = agg[m]["sqnr_h"].astype(np.float64)
        tot = cnt.sum()
        dens = cnt / (tot * (SB[1] - SB[0])) if tot > 0 else cnt
        plt.stairs(dens, SB, label=m, color=c, linewidth=1.5)
    plt.xlabel("per-block SQNR (dB)  [right = better]")
    plt.ylabel("density")
    plt.title(f"Per-block SQNR distribution (block=32)  {args.repo}")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pdir / "block_sqnr_hist.png", dpi=130)
    plt.close()

    # 2) complementary CDF of |err|/amax (log-log): the tail
    plt.figure(figsize=(9, 5.2))
    for m, c in zip(PLOT_METHODS, colors):
        x, surv = _ccdf(agg[m]["tail_h"])
        plt.plot(x, np.maximum(surv, 1e-9), label=m, color=c, linewidth=1.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("|error| / amax")
    plt.ylabel("P(|error|/amax > t)  [lower = better]")
    plt.title(f"Error tail (complementary CDF)  {args.repo}")
    plt.legend(fontsize=7, loc="lower left")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(pdir / "err_ccdf.png", dpi=130)
    plt.close()

    # 3) box/whisker of per-block rel-L2 (%) per method
    bxp_stats = []
    for m in PLOT_METHODS:
        rh = agg[m]["rel_h"]
        q = {p: 100 * _quantile(rh, RB, p, log=True)
             for p in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)}
        bxp_stats.append(dict(
            label=m.replace(" (2lvl)", "").replace("Direct ", ""),
            med=q[0.50], q1=q[0.25], q3=q[0.75],
            whislo=q[0.10], whishi=q[0.90],
            fliers=[q[0.99], 100 * agg[m]["max_rel"]],
        ))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bxp(bxp_stats, showfliers=True, flierprops=dict(marker="x", markersize=5,
           markeredgecolor="#B00020"))
    ax.set_ylabel("per-block rel-L2 (%)  [lower = better]")
    ax.set_title(f"Per-block error spread (box=IQR, whiskers=p10/p90, x=p99/max)  {args.repo}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(pdir / "block_box.png", dpi=130)
    plt.close()

    # 4) % of blocks below the SQNR threshold (tail scalar)
    thr = args.threshold
    vals = [100 * _frac_below(agg[m]["sqnr_h"], thr) for m in PLOT_METHODS]
    labels = [m.replace(" (2lvl)", "").replace("Direct ", "") for m in PLOT_METHODS]
    plt.figure(figsize=(9, 5.2))
    bars = plt.bar(range(len(vals)), vals, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=8)
    plt.ylabel(f"% of blocks with SQNR < {thr:.0f} dB")
    plt.title(f"Fraction of poorly-quantized blocks (block=32)  {args.repo}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(pdir / "blocks_below_sqnr.png", dpi=130)
    plt.close()

    print(f"wrote plots in {pdir}/ (block_sqnr_hist.png, err_ccdf.png, "
          f"block_box.png, blocks_below_sqnr.png)", flush=True)


if __name__ == "__main__":
    main()
