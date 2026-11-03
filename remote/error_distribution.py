#!/usr/bin/env python3
"""Compare the *distribution* of quantization error across methods.

Reuses the exact quantizers from quant_real_weights.py so numbers stay
consistent with the rel-L2 tables. For each method it reports, over the
concatenated layer-0 weight matrices:

  * signed bias   = mean(q - x)                 (systematic offset)
  * rms           = sqrt(mean((q - x)^2))
  * |error| tail percentiles p50/p90/p99/p99.9/max   (as % of tensor amax)
  * rms(error) bucketed by |x| decile           (where the error concentrates)

It also saves three plots (copied back locally):
  err_hist.png  - overlaid signed-error histograms (error / amax)
  err_cdf.png   - CDF of |error| / amax (log-x)  -> tail comparison
  err_by_mag.png- rms(error) vs |x| decile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import quant_real_weights as Q

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Methods to draw in plots (kept small for readability); text table shows all.
PLOT_METHODS = [
    "Direct MXFP6",
    "Direct MXFP4",
    "Direct NVFP4 (2lvl)",
    "Direct NVFP4 b16 (2lvl)",
    "Direct FP4SE4 (2lvl)",
    "Direct FP4SE6 (2lvl)",
    "NVFP4->MXFP6 revised (2lvl)",
    "NVFP4->MXFP6 naive (2lvl)",
]


def load_concat(repo: str, prefix: str, min_elems: int):
    targets, shards = Q.resolve_shards(repo, prefix)
    opened = {s: Q.SafeTensorsFile(Q.hf_hub_download(repo, s)) for s in shards}
    chunks = []
    names = []
    for name in sorted(targets):
        stf = opened[targets[name]]
        info = stf.meta(name)
        shape = tuple(info["shape"])
        if len(shape) != 2:
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
        names.append(name)
    x = np.concatenate(chunks)
    return x, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prefix", default="model.language_model.layers.0.")
    ap.add_argument("--out", default="out/error_distribution.txt")
    ap.add_argument("--plotdir", default="out")
    ap.add_argument("--min-elems", type=int, default=4096)
    ap.add_argument("--sample", type=int, default=6_000_000,
                    help="random elements kept for percentiles/plots")
    ap.add_argument("--seed", type=int, default=20260707)
    args = ap.parse_args()

    x, names = load_concat(args.repo, args.prefix, args.min_elems)
    amax = float(np.max(np.abs(x)))
    x2 = float(np.dot(x, x))
    rms_x = (x2 / x.size) ** 0.5
    print(f"loaded {len(names)} tensors, {x.size:,} elements, amax={amax:.5g}, rms={rms_x:.5g}")

    rng = np.random.default_rng(args.seed)
    if x.size > args.sample:
        sidx = rng.choice(x.size, size=args.sample, replace=False)
        sidx = np.sort(sidx)
    else:
        sidx = np.arange(x.size)
    x_s = x[sidx]
    absx_s = np.abs(x_s)

    # Decile edges of |x| on the sample (for the magnitude-bucket table).
    dec_edges = np.quantile(absx_s, np.linspace(0, 1, 11))
    dec_bin = np.clip(np.searchsorted(dec_edges, absx_s, side="right") - 1, 0, 9)

    pcts = [50, 90, 99, 99.9, 100.0]
    rows = []
    err_samples: dict[str, np.ndarray] = {}
    bias_map, rms_map = {}, {}
    bucket_rms: dict[str, np.ndarray] = {}

    for mname, fn, gbook in Q.METHODS:
        pred = Q.run_method(fn, gbook, x)
        err = pred - x
        del pred
        rms = (float(np.dot(err, err)) / err.size) ** 0.5
        bias = float(np.mean(err))
        e_s = err[sidx]
        del err
        err_samples[mname] = e_s
        bias_map[mname] = bias
        rms_map[mname] = rms
        abse = np.abs(e_s)
        ptail = np.percentile(abse, pcts)
        # rms(error) per |x| decile.
        br = np.zeros(10)
        for b in range(10):
            m = dec_bin == b
            br[b] = (np.mean(e_s[m] ** 2)) ** 0.5 if m.any() else 0.0
        bucket_rms[mname] = br
        rows.append((mname, bias, rms, ptail))
        print(f"done {mname}")

    def p(v):  # percent of amax
        return f"{100.0 * v / amax:8.3f}%"

    L = []
    L.append(f"Error distribution: {args.repo}  prefix={args.prefix}")
    L.append("=" * 72)
    L.append(f"elements={x.size:,}  amax={amax:.6g}  rms(x)={rms_x:.6g}  sample={x_s.size:,}")
    L.append("all error stats below are expressed as % of tensor amax")
    L.append("")
    col = max(len(m) for m, *_ in rows) + 1
    hdr = f"{'method':{col}s} {'bias':>9s} {'rms':>9s} {'p50':>9s} {'p90':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s}"
    L.append(hdr)
    L.append("-" * len(hdr))
    for mname, bias, rms, ptail in rows:
        L.append(
            f"{mname:{col}s} {100*bias/amax:8.3f}% {p(rms)} "
            + " ".join(p(v) for v in ptail)
        )
    L.append("")
    L.append("rms(error) as % of amax, bucketed by |x| decile (D0=smallest |w| .. D9=largest)")
    L.append("-" * 72)
    L.append(f"{'method':{col}s} " + " ".join(f"D{i:<7d}" for i in range(10)))
    for mname in PLOT_METHODS:
        br = bucket_rms[mname]
        L.append(f"{mname:{col}s} " + " ".join(f"{100*v/amax:7.3f}" for v in br))
    L.append("")
    L.append("decile |x| edges (abs weight): " + ", ".join(f"{e:.4g}" for e in dec_edges))

    text = "\n".join(L)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print("\n" + text)

    # ---------------- plots ----------------
    pdir = Path(args.plotdir)
    pdir.mkdir(parents=True, exist_ok=True)

    # 1) signed-error histogram (error / amax)
    plt.figure(figsize=(8, 5))
    rng_lim = np.percentile(np.abs(err_samples["NVFP4->MXFP6 naive (2lvl)"]), 99.9) / amax
    bins = np.linspace(-rng_lim, rng_lim, 121)
    for mname in PLOT_METHODS:
        plt.hist(err_samples[mname] / amax, bins=bins, histtype="step",
                 density=True, label=mname, linewidth=1.3)
    plt.yscale("log")
    plt.xlabel("signed error / amax")
    plt.ylabel("density (log)")
    plt.title(f"Signed error distribution  {args.prefix}")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(pdir / "err_hist.png", dpi=130)
    plt.close()

    # 2) CDF of |error| / amax (log-x) -> tail behaviour
    plt.figure(figsize=(8, 5))
    for mname in PLOT_METHODS:
        a = np.sort(np.abs(err_samples[mname]) / amax)
        cdf = np.linspace(0, 1, a.size)
        plt.plot(np.maximum(a, 1e-6), cdf, label=mname, linewidth=1.4)
    plt.xscale("log")
    plt.xlabel("|error| / amax (log)")
    plt.ylabel("cumulative fraction")
    plt.title(f"|error| CDF  {args.prefix}")
    plt.legend(fontsize=7, loc="lower right")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(pdir / "err_cdf.png", dpi=130)
    plt.close()

    # 3) rms(error) vs |x| decile
    plt.figure(figsize=(8, 5))
    xs = np.arange(10)
    for mname in PLOT_METHODS:
        plt.plot(xs, 100 * bucket_rms[mname] / amax, marker="o", label=mname, linewidth=1.4)
    plt.xlabel("|x| decile (0=smallest .. 9=largest weights)")
    plt.ylabel("rms(error) as % of amax")
    plt.title(f"Where error concentrates  {args.prefix}")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pdir / "err_by_mag.png", dpi=130)
    plt.close()

    print(f"\nwrote {args.out} and plots in {pdir}/ (err_hist.png, err_cdf.png, err_by_mag.png)")


if __name__ == "__main__":
    main()
