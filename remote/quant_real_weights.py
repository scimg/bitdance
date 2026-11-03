#!/usr/bin/env python3
"""MXFP6-emulation quantization error on real Qwen3.5-4B weights.

Ports the block-quant logic from vector_quant_error.py to real weight
matrices. bf16 tensors are read from safetensors shards with a small manual
parser (no torch dependency). Blocks of 32 run along the input (last) dim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

BLOCK_SIZE = 32


# --------------------------------------------------------------------------- #
# Codebooks (identical grids to vector_quant_error.py)
# --------------------------------------------------------------------------- #
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


def make_scale_book(ebits: int, mbits: int, bias: int, fn: bool = False) -> ScaleBook:
    """Positive normalized scales for an (ebits, mbits) format.

    `fn` selects the OCP "finite, NaN-only" convention that E4M3FN actually uses: the top
    exponent field is a normal field like any other, and only the single all-ones exponent AND
    all-ones mantissa code is taken for NaN. Without it this loop stops one exponent short and
    the E4M3 book tops out at 240 instead of 448 -- it borrows IEEE's reservation of the whole
    top exponent field for Inf/NaN, which E4M3FN does not do.

    On these weights the missing octave changes nothing (kernel/p137_error_recheck.py measures
    19.23 dB against 19.24 dB) because no weight block ever asks for a scale near the top of the
    book. It is still the wrong book, and it is the same error the shipped path had before
    kernel/p115_book_gate.py started checking it against torch.float8_e4m3fn.
    """
    values: list[float] = []
    exps: list[int] = []
    mants: list[float] = []
    top = (1 << ebits) if fn else (1 << ebits) - 1
    for exp_bits in range(1, top):
        exp = exp_bits - bias
        for mant_bits in range(0, 1 << mbits):
            if fn and exp_bits == top - 1 and mant_bits == (1 << mbits) - 1:
                continue                     # the one NaN code E4M3FN reserves
            mant = 1.0 + mant_bits / float(1 << mbits)
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


E4M3 = make_scale_book(ebits=4, mbits=3, bias=7, fn=True)
E6M2 = make_scale_book(ebits=6, mbits=2, bias=31)
E5M3 = make_scale_book(ebits=5, mbits=3, bias=15)
E4M4 = make_scale_book(ebits=4, mbits=4, bias=7)
E8M0 = make_e8m0_book()


@dataclass
class Quantized:
    deq: np.ndarray
    scale: np.ndarray
    exp: np.ndarray
    mant: np.ndarray
    data: np.ndarray


def nearest_codes(x: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Nearest code via a single searchsorted against bin midpoints, ties toward zero.

    codebook must be sorted ascending. One O(N log K) pass with only an index array + output as
    large temporaries (vs the O(N*K) broadcast argmin, or the multi-temporary abs/where variant)
    -- important for full-model sweeps.

    TIES. This used to be a bare searchsorted(side="left") documented as sending a tie to "the
    lower-magnitude (left) code". That is only true for positive x: the codebook is signed and
    ascending, so for negative x the left code is the LARGER magnitude. The rule was really
    round-half-toward-minus-infinity, which is asymmetric and therefore leaves a systematic
    MEAN error, not just a magnitude error.

    It mattered, and only in one place. Direct quantization essentially never lands exactly on a
    midpoint, so its bias stayed at ~1e-5. The transcode fold's input is mant * |d| / 2^bump with
    mant = 1 + k/8 and d from E2M1 -- small dyadic rationals that hit E2M3 midpoints constantly
    -- and under a two-level global scale its bias reached -1.9e-4, about 50x the direct
    quantization it comes from. In weight SQNR that is worth 0.03 dB and invisible. Against a
    NON-NEGATIVE input, where a coherent error grows as K instead of sqrt(K), it was worth 14 dB
    (kernel/p137_error_recheck.py), and `down_proj` takes a post-activation input.

    The device never had this: kernel/transcode_lut.build_luts rounds the positive magnitude and
    reapplies the sign afterwards, which is symmetric. So this function was modelling a fold the
    kernel does not execute, and modelling it as worse than it is. Rounding the magnitude and
    restoring the sign is what makes the two agree.
    """
    x = np.asarray(x)
    mids = 0.5 * (codebook[:-1] + codebook[1:])
    pos = codebook[codebook >= 0]
    pmids = 0.5 * (pos[:-1] + pos[1:])
    if pos.size and np.allclose(np.sort(-pos), codebook[codebook <= 0]):
        idx = np.searchsorted(pmids, np.abs(x).ravel(), side="left")
        return (np.sign(x).ravel() * pos[idx]).reshape(x.shape)
    return codebook[np.searchsorted(mids, x.ravel(), side="left")].reshape(x.shape)


def require_finite(values: np.ndarray, what: str = "input") -> None:
    """Refuse non-finite input instead of quantizing it into a plausible wrong number.

    Byte-for-byte the policy in vector_quant_error.py's `require_finite`, deliberately duplicated
    the way the codebooks above are: this module is imported by every remote sweep and must run on
    the node from `remote/` alone. kernel/p230_nonfinite_gate.py asserts the two implementations
    accept and reject exactly the same inputs, so the duplication cannot drift into two policies.

    Why it matters more here than there: this module reads real checkpoints, and the sweeps that
    import it accumulate into histograms. A NaN in one shard makes its block's amax NaN, the target
    lands past the end of the book, `np.clip` pins it to the top entry, and the block is scaled by
    the book maximum -- so its 31 finite elements all round to zero (measured, not inferred). The
    NaN itself is then dropped by `np.histogram` without being counted, which removes the block from
    the denominators instead of reporting it. That is a bias toward "no error" in the tail these
    sweeps exist to measure.
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


def choose_scales(blocks: np.ndarray, scales: ScaleBook, data_codebook: np.ndarray):
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
        raise ValueError(f"size must be divisible by {block}")
    return values.reshape(-1, block)


def quantize(values: np.ndarray, scales: ScaleBook, data_codebook: np.ndarray,
             block: int = BLOCK_SIZE) -> Quantized:
    require_finite(values, "quantize() input")
    blocks = as_blocks(values, block)
    scale, exp, mant = choose_scales(blocks, scales, data_codebook)
    normalized = blocks / scale[:, None]
    data = nearest_codes(normalized, data_codebook)
    deq = scale[:, None] * data
    return Quantized(deq=deq.reshape(values.shape), scale=scale, exp=exp, mant=mant, data=data)


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


E2M1_MAX = float(np.max(np.abs(E2M1)))


# Each method: (name, base_fn, global_scalebook_or_None).
# When a scalebook is given, an exact per-tensor FP32 scale is applied first
# (as in real NVFP4 two-level scaling / MXFP6 epilogue), so the per-block scale
# lands in the well-utilized part of that book instead of saturating at its
# floor. MX (E8M0) formats have a huge exponent range and need no global scale.
METHODS = [
    ("Direct NVFP4 (no-g)", lambda x: quantize(x, E4M3, E2M1).deq, None),
    ("Direct NVFP4 (2lvl)", lambda x: quantize(x, E4M3, E2M1).deq, E4M3),
    ("Direct NVFP4 b16 (2lvl)", lambda x: quantize(x, E4M3, E2M1, block=16).deq, E4M3),
    ("Direct FP4SE6 (2lvl)", lambda x: quantize(x, E6M2, E2M1).deq, E6M2),
    ("Direct FP4SE5 (2lvl)", lambda x: quantize(x, E5M3, E2M1).deq, E5M3),
    ("Direct FP4SE4 (2lvl)", lambda x: quantize(x, E4M4, E2M1).deq, E4M4),
    ("Direct MXFP4", lambda x: quantize(x, E8M0, E2M1).deq, None),
    ("Direct MXFP6", lambda x: quantize(x, E8M0, E2M3).deq, None),
    ("NVFP4->MXFP6 naive (no-g)", lambda x: storage_to_mxfp6(x, E4M3, "naive"), None),
    ("NVFP4->MXFP6 naive (2lvl)", lambda x: storage_to_mxfp6(x, E4M3, "naive"), E4M3),
    ("NVFP4->MXFP6 revised (2lvl)", lambda x: storage_to_mxfp6(x, E4M3, "revised"), E4M3),
    ("FP4SE6->MXFP6 (2lvl)", lambda x: storage_to_mxfp6(x, E6M2, "fp4se"), E6M2),
    ("FP4SE5->MXFP6 (2lvl)", lambda x: storage_to_mxfp6(x, E5M3, "fp4se"), E5M3),
    ("FP4SE4->MXFP6 (2lvl)", lambda x: storage_to_mxfp6(x, E4M4, "fp4se"), E4M4),
]


def apply_chunked(fn, x: np.ndarray, chunk_blocks: int = 30000) -> np.ndarray:
    step = chunk_blocks * BLOCK_SIZE
    out = np.empty_like(x)
    for i in range(0, x.size, step):
        out[i : i + step] = fn(x[i : i + step])
    return out


def run_method(fn, gbook, flat: np.ndarray) -> np.ndarray:
    """Apply a method, optionally with an exact per-tensor FP32 scale.

    s_t = amax_tensor / (book_max * E2M1_max) maps the largest block's target
    scale to the top of the scale book, keeping all block scales in range.
    """
    if gbook is None:
        return apply_chunked(fn, flat)
    amax = float(np.max(np.abs(flat)))
    if amax == 0.0:
        return apply_chunked(fn, flat)
    s_t = amax / (float(np.max(gbook.value)) * E2M1_MAX)
    scaled = flat / np.float32(s_t)
    return apply_chunked(fn, scaled) * np.float32(s_t)


@dataclass
class Metrics:
    rel_l2: float
    rel_linf: float
    max_abs: float


def metrics(true: np.ndarray, pred: np.ndarray) -> Metrics:
    err = pred - true
    err2 = float(np.dot(err, err))
    true2 = float(np.dot(true, true))
    max_err = float(np.max(np.abs(err)))
    max_true = float(np.max(np.abs(true))) or 1.0
    return Metrics(
        rel_l2=(err2 / true2) ** 0.5 if true2 else 0.0,
        rel_linf=max_err / max_true,
        max_abs=max_err,
    )


# --------------------------------------------------------------------------- #
# Minimal safetensors reader (bf16/f16/f32 -> float32)
# --------------------------------------------------------------------------- #
_DT = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # reinterpreted below
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


class SafeTensorsFile:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            self.header = json.loads(fh.read(hlen).decode("utf-8"))
        self.data_start = 8 + hlen

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def meta(self, name):
        return self.header[name]

    def get_float32(self, name: str) -> np.ndarray:
        info = self.header[name]
        dtype = info["dtype"]
        shape = tuple(info["shape"])
        begin, end = info["data_offsets"]
        with open(self.path, "rb") as fh:
            fh.seek(self.data_start + begin)
            raw = fh.read(end - begin)
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2")
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=_DT[dtype]).astype(np.float32)
        # Named here rather than at the quantizer, because "shard X tensor Y is not finite" is
        # actionable and "block 41,209,663 is not finite" is not.
        require_finite(arr, f"{os.path.basename(self.path)}:{name}")
        return arr.reshape(shape)


# --------------------------------------------------------------------------- #
def local_shard_paths(repo: str) -> list[str] | None:
    """Shard paths from an already-populated HF cache, or None if that is not possible.

    Needed to run this on a node with no network. `hf_hub_download` insists on
    model.safetensors.index.json, which is not always in the cache even when every shard is --
    the Gemma 4 12B snapshot here has the shards and not the index. Nothing downstream actually
    needs the index: each shard's own header lists the tensors it holds, so the weight map can
    be rebuilt from the files. Resolution only; the arithmetic below is untouched.
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from kernel.hfshards import shards_for
        return shards_for(repo)
    except Exception:
        return None


def resolve_shards(repo: str, prefix: str):
    """(tensor name -> local shard path, list of local shard paths) for one prefix."""
    try:
        idx_path = hf_hub_download(repo, "model.safetensors.index.json")
        with open(idx_path) as fh:
            wmap = json.load(fh)["weight_map"]
        targets = {k: hf_hub_download(repo, v)
                   for k, v in wmap.items() if k.startswith(prefix)}
    except Exception as exc:
        paths = local_shard_paths(repo)
        if not paths:
            raise
        print(f"hub lookup failed ({type(exc).__name__}); "
              f"rebuilding the weight map from {len(paths)} cached shards")
        targets = {}
        for p in paths:
            for name in SafeTensorsFile(p).names():
                if name.startswith(prefix):
                    targets[name] = p
    return targets, sorted(set(targets.values()))


# One regex per decoder-layer naming convention seen in the models this is run on, tried in order
# of how many tensors each one MATCHES rather than assumed. Qwen3.5 nests the decoder under
# `model.language_model.layers.N`; Gemma 4 12B uses `model.layers.N`.
LAYER_RES = [re.compile(p) for p in (
    r"^model\.language_model\.layers\.(\d+)\.",
    r"^model\.layers\.(\d+)\.",
    r"^language_model\.model\.layers\.(\d+)\.",
    r"^layers\.(\d+)\.",
)]


def build_layer_index(repo: str, verbose: bool = True):
    """{layer id: {tensor name: ABSOLUTE LOCAL shard path}} for any of the layouts above.

    The paths are already resolved on disk. A caller must NOT put them back through
    `hf_hub_download` -- that asks the hub to resolve an absolute path as a repo-relative filename
    and fails on exactly the offline, index-less cache this function exists to support.

    LIVES HERE BECAUSE TWO SWEEPS NEED IT AND ONLY ONE OF THEM WAS FIXED. remote/block_error_dist.py
    and remote/error_stats.py each carried their own copy that hard-coded Qwen3.5's prefix and went
    through `model.safetensors.index.json`. On Gemma 4 12B -- a single `model.safetensors` with no
    index, decoder at `model.layers.N` -- the index lookup fails and the regex matches nothing, so
    the copy that was not fixed returns ZERO layers and reports over an empty aggregate: a run that
    prints a finished-looking table of nothing. A crash would have been kinder. Both callers now
    resolve layout through this function, so a new model is one entry in LAYER_RES.
    """
    wmap, paths = resolve_shards(repo, "")
    if not wmap:
        raise SystemExit(f"{repo}: no tensors resolved (no index and no cached shards)")
    best, best_n = None, 0
    for rx in LAYER_RES:
        n = sum(1 for name in wmap if rx.match(name))
        if n > best_n:
            best, best_n = rx, n
    if best is None or best_n == 0:
        raise SystemExit(f"{repo}: no tensor name matches a known decoder-layer pattern; "
                         f"first few are {sorted(wmap)[:4]}")
    if verbose:
        print(f"{repo}: {len(wmap)} tensors in {len(paths)} shard(s); "
              f"layer pattern {best.pattern!r} matches {best_n}", flush=True)
    layers: dict[int, dict[str, str]] = {}
    for name, shard in wmap.items():
        m = best.match(name)
        if m:
            layers.setdefault(int(m.group(1)), {})[name] = shard
    return layers


def pct(v: float) -> str:
    return f"{100.0 * v:9.4f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prefix", default="model.language_model.layers.0.")
    ap.add_argument("--out", default="out/quant_real_layer0.txt")
    ap.add_argument("--min-elems", type=int, default=4096,
                    help="skip tensors smaller than this (norms/biases)")
    args = ap.parse_args()

    targets, shards = resolve_shards(args.repo, args.prefix)
    print(f"repo={args.repo} prefix={args.prefix}")
    print(f"target tensors: {len(targets)} across shards: {shards}")

    # Open only the needed shards. resolve_shards has already made these local paths.
    opened: dict[str, SafeTensorsFile] = {}
    for shard in shards:
        opened[shard] = SafeTensorsFile(shard)
        print(f"opened shard {os.path.basename(shard)}")

    lines: list[str] = []
    lines.append(f"Real-weight block quantization: {args.repo}")
    lines.append("=" * 60)
    lines.append(f"prefix: {args.prefix}")
    lines.append(f"block size: {BLOCK_SIZE}  (along input/last dim)")
    lines.append("metric: relative L2 error = ||q - x||_2 / ||x||_2")
    lines.append("")

    method_names = [m[0] for m in METHODS]

    # Accumulators for a layer-wide (concatenated) summary.
    agg_true2 = 0.0
    agg_err2 = {n: 0.0 for n in method_names}
    col = max(len(n) for n in method_names) + 1

    for name in sorted(targets):
        shard = targets[name]
        stf = opened[shard]
        info = stf.meta(name)
        shape = tuple(info["shape"])
        if len(shape) != 2:
            print(f"skip (ndim={len(shape)}): {name} {shape}")
            continue
        w = stf.get_float32(name)
        rows, cols = w.shape
        keep = (cols // BLOCK_SIZE) * BLOCK_SIZE
        if keep == 0:
            print(f"skip (cols<{BLOCK_SIZE}): {name} {shape}")
            continue
        if keep != cols:
            w = w[:, :keep]
        flat = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
        if flat.size < args.min_elems:
            print(f"skip (small): {name} {shape}")
            continue

        lines.append(f"Tensor: {name}")
        lines.append(f"  shape={shape} dtype={info['dtype']} used_cols={keep}")
        lines.append(f"  {'method':{col}s} {'rel_l2':>11s} {'rel_linf':>11s} {'max_abs':>12s}")
        true2 = float(np.dot(flat, flat))
        agg_true2 += true2
        for mname, fn, gbook in METHODS:
            pred = run_method(fn, gbook, flat)
            m = metrics(flat, pred)
            err = pred - flat
            agg_err2[mname] += float(np.dot(err, err))
            lines.append(f"  {mname:{col}s} {pct(m.rel_l2):>11s} {pct(m.rel_linf):>11s} {m.max_abs:12.5g}")
        lines.append("")
        print(f"done: {name} {shape}")

    lines.append("Layer-wide summary (all quantized tensors concatenated)")
    lines.append("-" * 55)
    lines.append(f"  {'method':{col}s} {'rel_l2':>11s}")
    for mname in method_names:
        rel = (agg_err2[mname] / agg_true2) ** 0.5 if agg_true2 else 0.0
        lines.append(f"  {mname:{col}s} {pct(rel):>11s}")

    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print("\n" + text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
