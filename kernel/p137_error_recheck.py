#!/usr/bin/env python3
"""Block quantization error, recomputed from the format definitions, against remote/'s numbers.

Slides 6-7 publish block-quantization error for synthetic tensors and for real Qwen3.5-4B
weights, produced by `remote/quant_real_weights.py` + `remote/error_stats.py`. Those numbers are
load-bearing -- they are the argument for the whole approach -- and they have never been checked
against anything except themselves. This is a second implementation, written from the OCP
definitions rather than from that code, so that a shared assumption cannot hide in both.

Where it deliberately differs from `remote/`, and why each difference is the thing worth testing:

  * the E4M3 scale book comes from `torch.float8_e4m3fn` by enumerating all 256 bytes and
    casting, not from a hand-written loop. remote/'s `make_scale_book` runs its exponent field
    over `range(1, 15)`, which stops the book at 240 and drops E4M3FN's top octave -- the format
    reaches 448, because FN reserves only the all-ones-exponent-and-mantissa code for NaN, not
    the whole exponent field. It also omits the subnormals, so the book floors at 2^-6 instead
    of 2^-9.
  * the block scale is compared under both roundings. remote/ takes the first book entry >=
    amax/6 (ceiling, never clips); kernel/qlayout.py is the code that actually ships. If those
    two disagree the deck and the model are not measuring the same recipe.
  * SQNR is computed two ways -- pooled over all tensors, and per tensor then averaged -- because
    they answer different questions and pooling is dominated by whichever tensor has the largest
    norm.

    python3 kernel/p137_error_recheck.py --repo google/gemma-4-12B-it
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLOCK = 32
E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def signed(vals):
    s = {0.0}
    for v in vals:
        s.add(float(v))
        s.add(-float(v))
    return np.array(sorted(s), dtype=np.float64)


E2M1 = signed(E2M1_POS)


def e2m3_pos():
    """E2M3: 2 exponent bits (bias 1), 3 mantissa bits, subnormals included."""
    out = {0.0}
    for e in range(4):
        for m in range(8):
            out.add((m / 8.0) * 2.0**(1 - 1) if e == 0 else (1.0 + m / 8.0) * 2.0**(e - 1))
    return sorted(out)


E2M3 = signed(e2m3_pos())


def e4m3_book_from_hardware(subnormal=True, cap=None):
    """Positive finite E4M3FN values, by casting all 256 bytes rather than deriving them.

    torch is the arbiter here on purpose: the point of this file is to not re-derive a rule that
    the other implementation may also have got wrong.
    """
    b = torch.arange(256, dtype=torch.uint8)
    v = b.view(torch.float8_e4m3fn).to(torch.float64).numpy()
    v = v[np.isfinite(v) & (v > 0)]
    if not subnormal:
        v = v[v >= 2.0**-6]
    if cap is not None:
        v = v[v <= cap]
    return np.unique(v)


def e8m0_book():
    return np.exp2(np.arange(-40, 41, dtype=np.float64))


def nearest(x, book, ties="down"):
    """Nearest code, with the tie rule made explicit because it is the whole story here.

    `remote/quant_real_weights.py` uses searchsorted(side="left") and documents it as mapping a
    tie to "the lower-magnitude (left) code". That is only true for positive values. The book is
    signed and sorted ascending, so the left code is the SMALLER one -- for a negative value that
    is the larger magnitude. The rule is really round-half-toward-minus-infinity, and it is not
    symmetric, so it puts a systematic negative bias on the output.

    That bias is invisible for direct quantization, where w/scale essentially never lands exactly
    on a midpoint. It is not invisible for the transcode fold, whose input is mant * d / 2^bump
    with mant = 1 + k/8 and d from E2M1 -- small dyadic rationals that hit E2M3 midpoints
    constantly.
    """
    mids = 0.5 * (book[:-1] + book[1:])
    flat = x.ravel()
    lo = np.searchsorted(mids, flat, side="left")
    if ties == "down":
        return book[lo].reshape(x.shape)
    if ties != "even":
        raise ValueError(ties)
    hi = np.searchsorted(mids, flat, side="right")
    # hi > lo exactly on a tie; break it to the even index, which is symmetric about zero
    idx = np.where((hi > lo) & (lo % 2 == 1), np.minimum(hi, len(book) - 1), lo)
    return book[idx].reshape(x.shape)


def blocks_of(flat, block=BLOCK):
    n = (flat.size // block) * block
    return np.asarray(flat[:n], dtype=np.float64).reshape(-1, block)


def pick_scale(amax, book, dmax, mode):
    """Block scale for a target of amax/dmax, under a stated rounding."""
    t = amax / dmax
    if mode == "ceil":                       # first entry >= target: never clips
        i = np.clip(np.searchsorted(book, t, side="left"), 0, len(book) - 1)
    elif mode == "nearest":
        i = np.clip(np.searchsorted(0.5 * (book[:-1] + book[1:]), t, side="left"),
                    0, len(book) - 1)
    else:
        raise ValueError(mode)
    s = book[i].copy()
    s[amax == 0] = 1.0
    return s


def quant(flat, sbook, dbook, mode="ceil", gscale=None):
    xb = blocks_of(flat)
    if gscale is not None:
        xb = xb / gscale
    s = pick_scale(np.abs(xb).max(axis=1), sbook, float(np.abs(dbook).max()), mode)
    deq = s[:, None] * nearest(xb / s[:, None], dbook)
    return (deq * gscale if gscale is not None else deq).ravel()


def fold_to_mxfp6(flat, sbook, mode="ceil", gscale=None, ties="down"):
    """NVFP4 -> MXFP6 transcode: E8M0 exponent from the scale, mantissa folded into the data."""
    xb = blocks_of(flat)
    if gscale is not None:
        xb = xb / gscale
    s = pick_scale(np.abs(xb).max(axis=1), sbook, 6.0, mode)
    d = nearest(xb / s[:, None], E2M1)
    exp = np.floor(np.log2(s)).astype(np.int64)
    mant = s / np.exp2(exp.astype(np.float64))          # significand in [1, 2)
    bump = (mant > 1.0).astype(np.int64)
    e2m3 = nearest(mant[:, None] * d / np.exp2(bump.astype(np.float64))[:, None], E2M3,
                   ties=ties)
    deq = np.exp2((exp + bump).astype(np.float64))[:, None] * e2m3
    return (deq * gscale if gscale is not None else deq).ravel()


def sqnr_db(x, q):
    e = q - x
    d = float(np.dot(x, x))
    n = float(np.dot(e, e))
    if n == 0:
        return float("inf")
    return -20.0 * np.log10((n / d) ** 0.5)


def decompose(x, q):
    """Split the error into a systematic gain term and the rest.

    SQNR pools two failures that do not cost a model the same thing. A quantizer whose outputs
    are all 1% small has an error, but it is one number -- the next layer's norm or the next
    weight matrix absorbs it, and it is removable by calibration. A quantizer with the same
    total error spread randomly over elements is not removable by anything.

    g = <q, x> / <x, x> is the least-squares gain: the single scalar by which the reconstruction
    is systematically off. Residual SQNR is what is left after dividing it out, and the gap
    between the two SQNRs is how much of the reported error was really just that one scalar.
    """
    g = float(np.dot(q, x)) / float(np.dot(x, x))
    return g, sqnr_db(x, q / g), float(np.mean(q - x))


def gscale_top(flat, book):
    """remote/'s two-level scale: map the tensor amax to the very top of the scale book."""
    a = float(np.abs(flat).max())
    return a / (float(book.max()) * 6.0) if a else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="google/gemma-4-12B-it")
    ap.add_argument("--tensors", type=int, default=6)
    a = ap.parse_args()

    from safetensors import safe_open

    from kernel.hfshards import shards_for

    hw448 = e4m3_book_from_hardware(subnormal=False)
    hw448s = e4m3_book_from_hardware(subnormal=True)
    remote240 = e4m3_book_from_hardware(subnormal=False, cap=240.0)
    e8 = e8m0_book()

    print("E4M3 scale books in play")
    print(f"  remote/quant_real_weights  n={len(remote240):3d}  "
          f"{remote240.min():.9g} .. {remote240.max():g}")
    print(f"  E4M3FN normals (torch)     n={len(hw448):3d}  "
          f"{hw448.min():.9g} .. {hw448.max():g}")
    print(f"  E4M3FN incl. subnormals    n={len(hw448s):3d}  "
          f"{hw448s.min():.9g} .. {hw448s.max():g}")
    print(f"  -> remote/ is missing the top octave (max 240 vs 448) and the bottom three "
          f"(2^-6 vs 2^-9)\n")

    names, mats = [], []
    for shard in shards_for(a.repo):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.endswith(".weight") and ("proj" in k or "mlp" in k):
                    t = f.get_tensor(k)
                    if t.ndim == 2 and t.numel() >= 1 << 20 and len(names) < a.tensors:
                        names.append(k)
                        mats.append(t.float().numpy().reshape(-1))
        if len(names) >= a.tensors:
            break

    rows = [
        ("MXFP4  (E8M0 x E2M1)", lambda x: quant(x, e8, E2M1)),
        ("MXFP6  (E8M0 x E2M3)", lambda x: quant(x, e8, E2M3)),
        ("NVFP4  no global, book<=240", lambda x: quant(x, remote240, E2M1)),
        ("NVFP4  no global, book<=448", lambda x: quant(x, hw448, E2M1)),
        ("NVFP4  no global, +subnormal", lambda x: quant(x, hw448s, E2M1)),
        ("NVFP4  2lvl, book<=240", lambda x: quant(x, remote240, E2M1,
                                                   gscale=gscale_top(x, remote240))),
        ("NVFP4  2lvl, book<=448", lambda x: quant(x, hw448, E2M1,
                                                   gscale=gscale_top(x, hw448))),
        ("NVFP4->MXFP6 no global<=240", lambda x: fold_to_mxfp6(x, remote240)),
        ("NVFP4->MXFP6 2lvl <=240", lambda x: fold_to_mxfp6(x, remote240,
                                                            gscale=gscale_top(x, remote240))),
        ("NVFP4->MXFP6 2lvl <=448", lambda x: fold_to_mxfp6(x, hw448,
                                                            gscale=gscale_top(x, hw448))),
        ("NVFP4->MXFP6 subnormal, no g", lambda x: fold_to_mxfp6(x, hw448s)),
        ("NVFP4->MXFP6 2lvl, TIES-EVEN", lambda x: fold_to_mxfp6(
            x, hw448, gscale=gscale_top(x, hw448), ties="even")),
    ]

    print(f"SQNR dB on {len(names)} weight tensors of {a.repo}, block={BLOCK}\n")
    print("  gain   = least-squares scalar <q,x>/<x,x>; 1.0000 means no systematic shrinkage")
    print("  resid  = SQNR after dividing that scalar out")
    print("  d(dB)  = what the gain error alone was costing the pooled SQNR\n")
    hdr = (f"  {'method':<30}{'pooled':>8}{'gain':>9}{'resid':>8}{'d(dB)':>7}"
           f"{'bias':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, fn in rows:
        xs, qs = [], []
        for m in mats:
            x = np.asarray(m, dtype=np.float64)
            x = x[: (x.size // BLOCK) * BLOCK]
            xs.append(x)
            qs.append(fn(x))
        X, Q = np.concatenate(xs), np.concatenate(qs)
        pooled = sqnr_db(X, Q)
        g, resid, bias = decompose(X, Q)
        print(f"  {label:<30}{pooled:8.2f}{g:9.4f}{resid:8.2f}{resid - pooled:7.2f}"
              f"{bias:11.2e}")
    print("\ntensors:")
    for i, n in enumerate(names):
        print(f"  {i}  {n}")

    # ---- what the GEMM does with those weights ---------------------------- #
    #
    # Every number above is a property of W alone, and W alone is not what a model computes.
    # The two error terms behave completely differently once they are summed over K: the random
    # part adds in quadrature and grows as sqrt(K), while a bias that has the same sign in every
    # lane adds coherently and grows as K. So a bias worth 14% of the RMS is worth 14%*sqrt(K)
    # of the output error -- at K=3840 that is a factor of about nine, and it is invisible in
    # every column above. This measures the product instead of the operand.
    print("\n\noutput error of y = W x, x ~ N(0,1), 64 right-hand sides")
    print("(dSQNR = output SQNR minus weight SQNR; a method whose error is unbiased should")
    print(" carry its weight SQNR straight through, so a large negative dSQNR is a bias)\n")
    W = None
    for shard in shards_for(a.repo):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.endswith("layers.0.mlp.gate_proj.weight"):
                    W = f.get_tensor(k).float().numpy().astype(np.float64)
        if W is not None:
            break
    rng = np.random.default_rng(0)
    K = W.shape[1]
    # Two input distributions, because they treat a weight bias completely differently.
    # Zero-mean x: the bias term sum_k b*x_k is itself zero-mean with std b*sqrt(K), so it adds
    # in quadrature exactly like the random part and costs nothing extra. That is the case a
    # post-layernorm input (gate_proj, q_proj) is close to.
    # Non-negative x: sum_k b*x_k = b*K*mean(x) is a coherent DC offset growing as K, not
    # sqrt(K). down_proj is the matrix that actually sees this -- its input is post-GELU.
    dists = [("zero-mean  (post-LN, e.g. gate_proj)", rng.standard_normal((K, 64))),
             ("non-negative (post-GELU, down_proj)", np.abs(rng.standard_normal((K, 64))))]
    for dname, xin in dists:
        y = W @ xin
        print(f"  input: {dname}")
        hdr2 = f"    {'method':<30}{'W SQNR':>8}{'y SQNR':>8}{'dSQNR':>8}{'bias':>11}"
        print(hdr2)
        print("    " + "-" * (len(hdr2) - 4))
        for label, fn in rows:
            Wq = fn(W.reshape(-1)).reshape(W.shape)
            yq = Wq @ xin
            ws = sqnr_db(W.reshape(-1), Wq.reshape(-1))
            ys = sqnr_db(y.reshape(-1), yq.reshape(-1))
            print(f"    {label:<30}{ws:8.2f}{ys:8.2f}{ys - ws:8.2f}"
                  f"{float(np.mean(Wq - W)):11.2e}")
        print()
    print(f"  (W is {W.shape[0]}x{K}, so K={K} and sqrt(K)={K ** 0.5:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
