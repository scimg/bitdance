#!/usr/bin/env python3
"""NumPy block/vector quantization experiment.

All formats use block size 32. Each block chooses one scale from the datatype's
scale codebook using an amax coverage rule:

    scale >= max(abs(block)) / max_abs_data_code

Then each element is rounded to the nearest signed data code. Transcode paths
first quantize into the storage format, then emulate the MXFP6 data/scale
conversion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


BLOCK_SIZE = 32
N_ELEMS = 131_072
SEED = 20260707
OUT = Path("vector_quant_error.txt")


def signed(values: list[float]) -> np.ndarray:
    out = {0.0}
    for v in values:
        out.add(float(v))
        out.add(float(-v))
    return np.array(sorted(out), dtype=np.float64)


E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = signed(E2M1_POS)


def fp6_e2m3_positive_values() -> list[float]:
    vals = [0.0]
    bias = 1
    for exp_bits in range(0, 4):
        for mant_bits in range(0, 8):
            if exp_bits == 0:
                vals.append((mant_bits / 8.0) * (2.0 ** (1 - bias)))
            else:
                vals.append((1.0 + mant_bits / 8.0) * (2.0 ** (exp_bits - bias)))
    return sorted(set(vals))


E2M3 = signed(fp6_e2m3_positive_values())


@dataclass(frozen=True)
class ScaleBook:
    value: np.ndarray
    exp: np.ndarray
    mant: np.ndarray


def make_scale_book(ebits: int, mbits: int, bias: int, fn: bool = False,
                    subnormal: bool = True) -> ScaleBook:
    """Positive finite values of an (ebits, mbits, bias) float, ascending.

    `fn` selects the OCP "finite" encoding, which is the one NVFP4 stores its block
    scale in. IEEE reserves the all-ones exponent field for Inf/NaN and that is what
    the default does; E4M3FN has no infinities and spends only the all-ones
    significand on NaN, so the top exponent field carries normals and the book reaches
    one octave higher -- 448 rather than 240 for E4M3. That octave is not decorative:
    it is the difference between a book that covers Gemma's down_proj activations and
    one that clips them.

    `subnormal` adds the exp_bits==0 encodings, and it is the bottom of the book the way
    `fn` is the top. THIS FUNCTION USED TO RETURN NORMALS ONLY, and for E4M3 that put the
    floor at 2^-6 when the format's smallest positive value is 2^-9 -- three octaves of the
    encoding thrown away, at the end where the weights are. The cost was not subtle and it
    was misread for four sessions as a property of the FORMAT: `_choose_scale` clamps a
    target below the book to the book's first entry, so 78-99% of Gemma 12B weight blocks
    and 99.6-99.9% of 31B's were handed a scale 1.4-4.1x larger than they asked for and
    spent their four element bits on range they never used. E8M0's book has no such floor
    (`make_e8m0_book` runs to 2^-40), so "E4M3 vs E8M0" was never a comparison of three
    mantissa bits against none -- it was that, minus a truncated book on one side. Restoring
    the subnormals is what makes the two books comparable, and it is what the format says.

    Subnormals are exact in the encoding and decode for free anywhere the byte is cast by
    hardware or by `torch.float8_e4m3fn`; the value is (mant_bits / 2^mbits) * 2^(1-bias)
    rather than (1 + mant_bits / 2^mbits) * 2^(exp_bits-bias). The all-zero byte is the
    number zero and is excluded -- a zero scale is not a scale.

    THEY ARE STORED NORMALIZED: `value` is the subnormal's exact value, but the (exp, mant)
    decomposition is the one with mant in [1, 2), not the literal (1-bias, k/2^mbits). Every
    consumer of this book -- storage_to_mxfp6, kernel/transcode_lut.py, kernel/nvfp4_codec.py --
    reads mant as a significand with the leading 1 present, and for E4M3 the subnormals land on
    exactly {1, 1.25, 1.5, 1.75}, which are four of the eight significands those already handle.
    So normalizing here is what lets the subnormal range cost no new LUT classes and no wider
    table in the kernel. Handing them the literal k/8 instead is not merely inelegant: the fold
    would push the block's elements to 0.06..0.75, where E2M3's step is 0.125, and a k=1 block
    would resolve onto three codes.
    """
    values: list[float] = []
    exps: list[int] = []
    mants: list[float] = []
    top_exp_bits = (1 << ebits) if fn else (1 << ebits) - 1
    nan_exp_bits, nan_mant_bits = (1 << ebits) - 1, (1 << mbits) - 1
    for exp_bits in range(0 if subnormal else 1, top_exp_bits):
        sub = exp_bits == 0
        for mant_bits in range(0, 1 << mbits):
            if fn and exp_bits == nan_exp_bits and mant_bits == nan_mant_bits:
                continue                      # the one NaN encoding
            if sub:
                if mant_bits == 0:
                    continue                  # the all-zero byte is 0.0, not a scale
                # k * 2^(1-bias-mbits) renormalized to mant in [1, 2): k = mant * 2^j.
                j = int(mant_bits).bit_length() - 1
                mant = mant_bits / float(1 << j)
                exp = j + 1 - bias - mbits
            else:
                mant = 1.0 + mant_bits / float(1 << mbits)
                exp = exp_bits - bias
            values.append(mant * (2.0**exp))
            exps.append(exp)
            mants.append(mant)
    order = np.argsort(np.array(values, dtype=np.float64), kind="stable")
    return ScaleBook(
        value=np.array(values, dtype=np.float64)[order],
        exp=np.array(exps, dtype=np.int16)[order],
        mant=np.array(mants, dtype=np.float64)[order],
    )


def make_e8m0_book() -> ScaleBook:
    exps = np.arange(-40, 41, dtype=np.int16)
    return ScaleBook(
        value=np.exp2(exps.astype(np.float64)),
        exp=exps,
        mant=np.ones_like(exps, dtype=np.float64),
    )


# E4M3 is E4M3FN, the format NVFP4 actually stores a block scale in, so its book runs to
# 448 and not to the 240 the IEEE-style layout stops at. `MXFP6_E4M3_LEGACY_BOOK=1`
# restores the 240 book, which is what every result recorded up to s26 was measured
# against; see the note in `kernel/qlayout.py`.
#
# ONE parser, exported, because two spellings of the same predicate is how the host codec,
# the transcode LUT and the "independent" reference in kernel/qlayout_gate.py end up
# disagreeing about which format they are testing. This module is the lower of the two --
# kernel/qlayout.py, kernel/transcode_lut.py and kernel/nvfp4_codec.py all derive from it --
# so the definition lives here and qlayout imports it. Previously this accepted only
# ("0", "", "no") while qlayout also accepted "off" and "false", so MXFP6_E4M3_LEGACY_BOOK=off
# silently built the 448 book on one side and the 240 book on the other, with nothing
# raising: `_choose_scale` clamps rather than complains.
def e4m3_legacy_book() -> bool:
    """Is the IEEE-style 240-max scale book selected? (Default: no, use E4M3FN's 448.)"""
    return os.environ.get("MXFP6_E4M3_LEGACY_BOOK", "0").strip().lower() \
        not in ("0", "", "no", "off", "false")


def e4m3_subnormal() -> bool:
    """Does the E4M3 scale book include the subnormals? (Default: no.)

    Exported for the same reason `e4m3_legacy_book` is: kernel/qlayout.py builds its own copy of
    this book, and the last time the two sides parsed one variable separately they silently
    built a 448 book here and a 240 book there with nothing raising. The default is OFF and the
    reason is in kernel/qlayout.py's E4M3_SUBNORMAL -- the values are right, the device fold
    cannot decode them, and accuracy runs the real kernel.
    """
    return os.environ.get("MXFP6_E4M3_SUBNORMAL", "0").strip().lower() \
        not in ("0", "", "no", "off", "false")


E4M3_FN = not e4m3_legacy_book()
E4M3_SUBNORMAL = e4m3_subnormal()
E4M3 = make_scale_book(ebits=4, mbits=3, bias=7, fn=E4M3_FN, subnormal=E4M3_SUBNORMAL)
# The FP4SE books take `subnormal`'s True default while E4M3 above defaults to OFF, so the two
# families differ at BOTH ends of the book -- E4M3FN reaches one octave higher and gives up three
# at the bottom -- and read as a comparison tilted toward FP4SE. It is not one HERE, and the
# reason is the data rather than the books: these experiments sweep [0, 500], so the scale a block
# asks for is amax/6 in [15.8, 83.3] and the scalar sweep's smallest nonzero x is 0.1, while the
# nearest floor is 2^-6. kernel/p229_bookmatch.py runs the whole method set under all-normals-only,
# all-full-format and the published convention: identical to every digit, 0 of 65,536 blocks
# clamped at either endpoint, 0 of 1,200 swept scalars changed. Matching the books here would be a
# no-op, so it is not done silently -- the asymmetry is recorded instead.
#
# On REAL weights the floor is the whole story: amax/6 there is around 2^-7 and 78-99% of Gemma
# 12B blocks clamp, so a comparison across these two families on real tensors must match the
# convention first. That is what makes this a documentation issue on slides 5-6 and a measurement
# issue on 7-8.
E5M3 = make_scale_book(ebits=5, mbits=3, bias=15)
E6M2 = make_scale_book(ebits=6, mbits=2, bias=31)
E4M4 = make_scale_book(ebits=4, mbits=4, bias=7)
E8M0 = make_e8m0_book()


@dataclass
class Quantized:
    deq: np.ndarray
    scale: np.ndarray
    exp: np.ndarray
    mant: np.ndarray
    data: np.ndarray


@dataclass
class Metrics:
    rel_l2: float
    rel_rmse: float
    max_abs: float
    rel_linf: float


def require_finite(values: np.ndarray, what: str = "input") -> None:
    """Refuse non-finite input instead of quantizing it into a plausible wrong number.

    Nothing in this pipeline checked, and what it did instead is measured rather than guessed. Put
    one NaN in a 64-element tensor of weight-scale values and the block holding it comes back like
    this: `choose_scales` reduces with `np.max`, NaN propagates, `np.searchsorted` places the NaN
    target past the end of the book, `np.clip` pins it to the last index, and the block is scaled
    by 448.0 -- the TOP of the E4M3FN book -- when it asked for 0.015625. Divided by 448 every real
    value in the block rounds to zero, so 31 of the 32 elements are returned as -0.0 and the NaN
    survives as a NaN. The block is destroyed, not perturbed. +-Inf takes the same path.

    Whether anyone would notice depends on who consumes it, which is the argument for stopping
    here: `metrics` uses `np.dot`, so one NaN turns that tensor's rel-L2 into NaN and the run is
    obviously broken. The histogram-accumulating sweeps do not -- `np.histogram` drops non-finite
    samples without counting them, so a poisoned block silently contributes 31 zeros and vanishes
    from the denominators, which is a quiet bias toward "no error" in exactly the tail the deck
    argues from.

    Fail-fast is the chosen policy because the alternatives both publish a number: sanitizing to
    zero changes the tensor being measured and then reports an error against the sanitized version,
    and a special-value bypass would have to define what a NaN weight means in a format whose
    elements have no NaN encoding. A checkpoint with a NaN in it is a broken checkpoint, and the
    sweep that reads it should stop rather than average it away.

    Signed zeros are finite and pass. -0.0 then rounds to the +0.0 code because `signed()` folds
    the two into one entry, which is a property of the codebook and not of this check.
    """
    v = np.asarray(values)
    if np.isfinite(v).all():
        return
    n_nan = int(np.isnan(v).sum())
    n_pos = int(np.isposinf(v).sum())
    n_neg = int(np.isneginf(v).sum())
    raise ValueError(
        f"{what} contains {n_nan + n_pos + n_neg} non-finite value(s) of {v.size} "
        f"(NaN={n_nan}, +Inf={n_pos}, -Inf={n_neg}); refusing to quantize. A NaN block returns "
        f"the book's largest scale times a rounded NaN, which is finite and wrong."
    )


def nearest_codes(x: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Nearest code, ties resolved on the MAGNITUDE so the rule is symmetric about zero.

    This was `argmin(|x - codebook|)` over the signed book. argmin returns the first minimum, so
    an exact tie took the lower index -- the more negative code. That is round-half-toward-minus-
    infinity: it shrinks positive values and grows negative ones, which leaves a systematic MEAN
    error rather than a symmetric magnitude error.

    Direct quantization barely noticed, because w/scale almost never lands exactly on a midpoint.
    `lut_to_e2m3` did: its input is mant * d / 2^bump with mant = 1 + k/8 and d from E2M1, small
    dyadic rationals that hit E2M3 midpoints constantly. Under a two-level global scale that
    pushed the fold's mean error to ~50x the direct NVFP4 it comes from. Weight SQNR hides it
    (0.03 dB) because a mean error is a vanishing share of the RMS; a GEMM does not, because a
    coherent error accumulates as K down the reduction while a random one accumulates as
    sqrt(K). Measured against a non-negative input -- which is what down_proj gets, its operand
    being post-activation -- that was worth 14 dB (kernel/p137_error_recheck.py).

    The device never did this. kernel/transcode_lut.build_luts rounds the positive magnitude and
    puts the sign back afterwards, so the shipped fold is symmetric and its tie error is a gain
    term, which costs almost nothing. This function was modelling a fold the kernel does not
    execute, and modelling it as worse than it is; rounding on the magnitude is what makes
    `transcode_revised_lut`'s claim to reproduce `storage_to_mxfp6` actually true.
    """
    pos = codebook[codebook >= 0]
    idx = np.abs(np.abs(x)[..., None] - pos[(None,) * x.ndim + (slice(None),)]).argmin(axis=-1)
    return np.sign(x) * pos[idx]


def choose_scales(blocks: np.ndarray, scales: ScaleBook, data_codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amax = np.max(np.abs(blocks), axis=1)
    data_max = float(np.max(np.abs(data_codebook)))
    target = np.divide(amax, data_max, out=np.zeros_like(amax), where=data_max != 0)
    idx = np.searchsorted(scales.value, target, side="left")
    idx = np.clip(idx, 0, len(scales.value) - 1)
    zero = amax == 0
    scale = scales.value[idx].copy()
    exp = scales.exp[idx].copy()
    mant = scales.mant[idx].copy()
    scale[zero] = 1.0
    exp[zero] = 0
    mant[zero] = 1.0
    return scale, exp, mant


def as_blocks(values: np.ndarray, block: int = BLOCK_SIZE) -> np.ndarray:
    if values.size % block != 0:
        raise ValueError(f"N_ELEMS must be divisible by {block}")
    return values.reshape(-1, block)


def quantize(values: np.ndarray, scales: ScaleBook, data_codebook: np.ndarray,
             block: int = BLOCK_SIZE) -> Quantized:
    require_finite(values, "quantize() input")
    blocks = as_blocks(values, block)
    scale, exp, mant = choose_scales(blocks, scales, data_codebook)
    normalized = blocks / scale[:, None]
    data = nearest_codes(normalized, data_codebook)
    deq = scale[:, None] * data
    return Quantized(
        deq=deq.reshape(values.shape),
        scale=scale,
        exp=exp,
        mant=mant,
        data=data,
    )


def lut_to_e2m3(combined: np.ndarray) -> np.ndarray:
    return nearest_codes(combined[:, None, :], E2M3)[:, 0, :]


def storage_to_mxfp6(values: np.ndarray, scales: ScaleBook, recipe: str,
                     block: int = BLOCK_SIZE) -> np.ndarray:
    q = quantize(values, scales, E2M1, block)
    if recipe == "naive":
        bump = np.zeros_like(q.exp, dtype=np.int16)
    elif recipe in {"revised", "fp4se"}:
        bump = (q.mant > 1.0).astype(np.int16)
    else:
        raise ValueError(f"unknown recipe: {recipe}")

    combined = q.mant[:, None] * q.data / np.exp2(bump.astype(np.float64))[:, None]
    e2m3 = lut_to_e2m3(combined)
    mx_scale = np.exp2((q.exp + bump).astype(np.float64))
    return (mx_scale[:, None] * e2m3).reshape(values.shape)


def metrics(true: np.ndarray, pred: np.ndarray) -> Metrics:
    err = pred - true
    err2 = float(np.dot(err, err))
    true2 = float(np.dot(true, true))
    max_err = float(np.max(np.abs(err)))
    max_true = float(np.max(np.abs(true))) or 1.0
    rel_l2 = (err2 / true2) ** 0.5 if true2 else 0.0
    rmse = (err2 / true.size) ** 0.5
    rms_true = (true2 / true.size) ** 0.5 if true2 else 1.0
    return Metrics(
        rel_l2=rel_l2,
        rel_rmse=rmse / rms_true,
        max_abs=max_err,
        rel_linf=max_err / max_true,
    )


def gen_normal(rng: np.random.Generator) -> np.ndarray:
    return np.clip(rng.normal(250.0, 80.0, N_ELEMS), 0.0, 500.0)


def gen_uniform(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 500.0, N_ELEMS)


def gen_exponential(rng: np.random.Generator) -> np.ndarray:
    return np.clip(rng.exponential(80.0, N_ELEMS), 0.0, 500.0)


def gen_outlier(rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(N_ELEMS) < 0.03
    values = rng.normal(80.0, 25.0, N_ELEMS)
    values[mask] = rng.normal(420.0, 55.0, int(mask.sum()))
    return np.clip(values, 0.0, 500.0)


TENSORS: list[tuple[str, Callable[[np.random.Generator], np.ndarray]]] = [
    ("normal_250_80_clip_0_500", gen_normal),
    ("uniform_0_500", gen_uniform),
    ("exponential_80_clip_0_500", gen_exponential),
    ("outlier_mix_clip_0_500", gen_outlier),
]


# Same datatype/method set as the real-model slide (grouped by datatype:
# direct then transcode-to-MXFP6). NVFP4 shown at both block 32 and block 16.
METHODS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("Direct MXFP6", lambda x: quantize(x, E8M0, E2M3).deq),
    ("Direct MXFP4", lambda x: quantize(x, E8M0, E2M1).deq),
    ("Direct NVFP4 b16", lambda x: quantize(x, E4M3, E2M1, block=16).deq),
    ("Direct NVFP4", lambda x: quantize(x, E4M3, E2M1).deq),
    ("NVFP4 -> MXFP6", lambda x: storage_to_mxfp6(x, E4M3, "revised")),
    ("Direct FP4SE4", lambda x: quantize(x, E4M4, E2M1).deq),
    ("FP4SE4 -> MXFP6", lambda x: storage_to_mxfp6(x, E4M4, "fp4se")),
    ("Direct FP4SE5", lambda x: quantize(x, E5M3, E2M1).deq),
    ("FP4SE5 -> MXFP6", lambda x: storage_to_mxfp6(x, E5M3, "fp4se")),
    ("Direct FP4SE6", lambda x: quantize(x, E6M2, E2M1).deq),
    ("FP4SE6 -> MXFP6", lambda x: storage_to_mxfp6(x, E6M2, "fp4se")),
]


def pct(v: float) -> str:
    return f"{100.0 * v:8.3f}%"


def main() -> None:
    rng = np.random.default_rng(SEED)
    lines: list[str] = []
    lines.append("Block Vector Quantization Error")
    lines.append("================================")
    lines.append("")
    lines.append(f"seed:       {SEED}")
    lines.append(f"elements:   {N_ELEMS}")
    lines.append(f"block size: {BLOCK_SIZE}")
    lines.append("")
    lines.append("implementation: NumPy vectorized")
    lines.append("data range: [0, 500]")
    lines.append("Scale selection: smallest representable scale covering block amax.")
    lines.append("Primary metric: relative L2 error = ||q - x||_2 / ||x||_2.")
    lines.append("")

    agg: dict[str, list[float]] = {name: [] for name, _ in METHODS}
    # per-block rel-L2 collected at block=32 for every method (same granularity
    # as the whole-model analysis on the real-weights slide), pooled over tensors
    block_rel: dict[str, list[np.ndarray]] = {name: [] for name, _ in METHODS}
    for tensor_name, gen in TENSORS:
        values = gen(rng).astype(np.float64, copy=False)
        if float(values.min()) < 0.0 or float(values.max()) > 500.0:
            raise ValueError(f"{tensor_name} generated values outside [0, 500]")
        xb = values.reshape(-1, BLOCK_SIZE)
        x2 = np.einsum("ij,ij->i", xb, xb)
        nz = x2 > 0.0
        lines.append(f"Tensor: {tensor_name}")
        lines.append("-" * (8 + len(tensor_name)))
        lines.append(f"{'method':28s} {'rel_l2':>12s} {'rel_linf':>12s} {'max_abs_err':>14s}")
        for method_name, fn in METHODS:
            q = fn(values)
            m = metrics(values, q)
            agg[method_name].append(100.0 * m.rel_l2)
            eb = (q - values).reshape(-1, BLOCK_SIZE)
            e2 = np.einsum("ij,ij->i", eb, eb)
            block_rel[method_name].append(np.sqrt(e2[nz] / x2[nz]))
            lines.append(
                f"{method_name:28s} {pct(m.rel_l2):>12s} {pct(m.rel_linf):>12s} {m.max_abs:14.6g}"
            )
        lines.append("")

    # aggregate mean rel-L2 (%) across all tensors, in METHODS order
    lines.append("Aggregate mean rel-L2 (%) across tensors")
    lines.append("-" * 40)
    csv_lines = ["method,mean_rel_l2_pct"]
    for method_name, _ in METHODS:
        vals = agg[method_name]
        mean_v = sum(vals) / len(vals)
        lines.append(f"{method_name:28s} {mean_v:7.2f}")
        csv_lines.append(f"{method_name},{mean_v:.4f}")
    lines.append("")

    # per-block rel-L2 distribution (%), pooled over all tensors (block = 32)
    lines.append("Per-block rel-L2 distribution (%), block = 32, pooled over tensors")
    lines.append("-" * 66)
    lines.append(f"{'method':28s} {'p10':>8s} {'p50':>8s} {'p90':>8s} "
                 f"{'p99':>8s} {'max':>8s} {'mean':>8s}")
    dist_lines = ["method,p10,p50,p90,p99,max,mean"]
    for method_name, _ in METHODS:
        rel = 100.0 * np.concatenate(block_rel[method_name])
        p10, p50, p90, p99 = np.percentile(rel, [10, 50, 90, 99])
        rmax = float(rel.max())
        rmean = float(rel.mean())
        lines.append(f"{method_name:28s} {p10:8.2f} {p50:8.2f} {p90:8.2f} "
                     f"{p99:8.2f} {rmax:8.2f} {rmean:8.2f}")
        dist_lines.append(f"{method_name},{p10:.4f},{p50:.4f},{p90:.4f},"
                          f"{p99:.4f},{rmax:.4f},{rmean:.4f}")
    lines.append("")

    text = "\n".join(lines)
    OUT.write_text(text)
    Path("vector_quant_error_agg.csv").write_text("\n".join(csv_lines) + "\n")
    Path("vector_block_dist.csv").write_text("\n".join(dist_lines) + "\n")
    print(text)
    print(f"wrote {OUT}, vector_quant_error_agg.csv and vector_block_dist.csv")
    write_manifest()


def _book_endpoints() -> dict:
    """The clamp behaviour of every book this run used, since that is what a reader has to check."""
    return {name: [float(b.value[0]), float(b.value[-1]), int(b.value.size)]
            for name, b in (("E4M3", E4M3), ("E4M4/FP4SE4", E4M4), ("E5M3/FP4SE5", E5M3),
                            ("E6M2/FP4SE6", E6M2), ("E8M0", E8M0))}


def write_manifest() -> None:
    """Record how these three files were made, so a build can refuse a stale or edited copy.

    Written here rather than by a separate tool because a manifest produced by anything other than
    the run it describes is a guess. See kernel/provenance.py for what fails a build.
    """
    from kernel import provenance

    provenance.write(
        "vector_quant_error.manifest.json",
        dict(
            artifact_set="synthetic_block_sweep",
            description="Per-block and aggregate rel-L2 for 11 quantization paths over four "
                        "synthetic tensors. Source of slide 6.",
            generated_by="vector_quant_error.py",
            command="python vector_quant_error.py",
            generated_at_utc=None,
            model_repo=None,
            model_revision=None,
            tensors=[name for name, _ in TENSORS],
            n_elements=len(TENSORS) * N_ELEMS,
            n_quant_blocks=len(TENSORS) * (N_ELEMS // BLOCK_SIZE),
            seed=SEED,
            # One method stores at block 16 while every statistic is still pooled in 32-element
            # windows, so the two cannot be recorded as one number.
            quant_block={"default": BLOCK_SIZE, "Direct NVFP4 b16": 16},
            metric_window=BLOCK_SIZE,
            element_format="E2M1 (4-bit) or E2M3 (6-bit), per method",
            scale_format="E4M3FN / E4M4 / E5M3 / E6M2 / E8M0, per method",
            scale_book_endpoints=_book_endpoints(),
            subnormal={"E4M3": bool(E4M3_SUBNORMAL), "E4M4/FP4SE4": True,
                       "E5M3/FP4SE5": True, "E6M2/FP4SE6": True, "E8M0": False},
            global_scale_recipe=None,     # this experiment applies no tensor-level scale
            rounding_tie_rule="round-to-nearest over the positive magnitude, sign reapplied "
                              "afterwards (nearest_codes); ties resolve toward the lower magnitude",
            code_sha256=provenance.code_sha256(["vector_quant_error.py"]),
            environment=provenance.environment(),
            device="cpu",
            source_artifacts={},          # generated from the seeded RNG, no input files
        ),
        files=["vector_quant_error.txt", "vector_quant_error_agg.csv", "vector_block_dist.csv"],
    )
    print("wrote vector_quant_error.manifest.json")


if __name__ == "__main__":
    main()
