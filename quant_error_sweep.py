#!/usr/bin/env python3
"""Sweep scalar values and plot relative quantization error.

The script compares:
  1. Direct NVFP4: E4M3 block scale * E2M1 data
  2. Direct MXFP4: E8M0 power-of-two block scale * E2M1 data
  3. Direct MXFP6: E8M0 power-of-two block scale * E2M3 data
  4. NVFP4 -> MXFP6 naive transcode:
       NVFP4 scale E4M3 = mantissa * 2^exponent
       exponent -> MXFP6 E8M0 scale
       LUT(mantissa, E2M1 data) -> MXFP6 E2M3 data
  5. NVFP4 -> MXFP6 revised transcode:
       if mantissa > 1, use exponent + 1 as MXFP6 scale exponent
       LUT(mantissa / 2, E2M1 data) -> MXFP6 E2M3 data
  6. FP4SE6: positive E6M2 scale * E2M1 data
  7. FP4SE6 -> MXFP6: exponent -> E8M0, mantissa + E2M1 -> E2M3
  8. FP4SE4: positive E4M4 scale * E2M1 data
  9. FP4SE4 -> MXFP6: exponent -> E8M0, mantissa + E2M1 -> E2M3

The plot writer reuses the local FreeType-backed drawing utilities from
make_mxfp6_nvfp4_png.py so it does not require matplotlib/Pillow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import make_mxfp6_nvfp4_png as draw
except Exception:  # FreeType backend unavailable (e.g. on Windows)
    draw = None


OUT = "quant_error.png"
X_MIN = 0.0
X_MAX = 500.0
STEP = 0.1


E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


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


E2M3_POS = fp6_e2m3_positive_values()


@dataclass(frozen=True)
class Nvfp4Q:
    value: float
    scale: float
    scale_exp: int
    scale_mant: float
    data: float


@dataclass(frozen=True)
class Fp4se6Q:
    value: float
    scale: float
    scale_exp: int
    scale_mant: float
    data: float


@dataclass(frozen=True)
class Fp4se4Q:
    value: float
    scale: float
    scale_exp: int
    scale_mant: float
    data: float


def e4m3_positive_scales() -> list[tuple[float, int, float]]:
    """Positive normalized E4M3 scales.

    Uses exponent bias 7. The sweep range does not require E4M3 subnormal
    scales, and NVFP4 block scales are treated as positive scale values.
    """
    out: list[tuple[float, int, float]] = []
    bias = 7
    for exp_bits in range(1, 15):
        exp = exp_bits - bias
        for mant_bits in range(0, 8):
            mant = 1.0 + mant_bits / 8.0
            out.append((mant * (2.0**exp), exp, mant))
    return out


E4M3_SCALES = e4m3_positive_scales()


def e6m2_positive_scales() -> list[tuple[float, int, float]]:
    """Positive normalized unsigned E6M2 scales.

    The sign bit from NVFP4 scale is repurposed as an exponent bit, yielding
    E6M2 scale precision/range. Bias 31 follows the usual 2^(ebits-1)-1 rule.
    The 0 and all-ones exponent codes are excluded for this finite normalized
    sweep model.
    """
    out: list[tuple[float, int, float]] = []
    bias = 31
    for exp_bits in range(1, 63):
        exp = exp_bits - bias
        for mant_bits in range(0, 4):
            mant = 1.0 + mant_bits / 4.0
            out.append((mant * (2.0**exp), exp, mant))
    return out


E6M2_SCALES = e6m2_positive_scales()


def e4m4_positive_scales() -> list[tuple[float, int, float]]:
    """Positive normalized unsigned E4M4 scales.

    FP4SE4 uses an 8-bit positive scale with 4 exponent bits and 4 mantissa
    bits. Bias 7 follows the usual 2^(ebits-1)-1 rule. The 0 and all-ones
    exponent codes are excluded for this finite normalized sweep model.
    """
    out: list[tuple[float, int, float]] = []
    bias = 7
    for exp_bits in range(1, 15):
        exp = exp_bits - bias
        for mant_bits in range(0, 16):
            mant = 1.0 + mant_bits / 16.0
            out.append((mant * (2.0**exp), exp, mant))
    return out


E4M4_SCALES = e4m4_positive_scales()


def e5m3_positive_scales() -> list[tuple[float, int, float]]:
    """Positive normalized unsigned E5M3 scales (FP4SE5).

    8-bit positive scale with 5 exponent bits and 3 mantissa bits, bias 15.
    The 0 and all-ones exponent codes are excluded for this finite sweep model.
    """
    out: list[tuple[float, int, float]] = []
    bias = 15
    for exp_bits in range(1, 31):
        exp = exp_bits - bias
        for mant_bits in range(0, 8):
            mant = 1.0 + mant_bits / 8.0
            out.append((mant * (2.0**exp), exp, mant))
    return out


E5M3_SCALES = e5m3_positive_scales()


def nearest(target: float, values: list[float]) -> float:
    return min(values, key=lambda v: (abs(v - target), abs(v)))


def build_nvfp4_codebook() -> list[Nvfp4Q]:
    codebook = [Nvfp4Q(0.0, 0.0, 0, 0.0, 0.0)]
    for scale, exp, mant in E4M3_SCALES:
        for data in E2M1_POS[1:]:
            codebook.append(Nvfp4Q(scale * data, scale, exp, mant, data))
    codebook.sort(key=lambda q: q.value)
    return codebook


NVFP4_CODEBOOK = build_nvfp4_codebook()


def build_fp4se6_codebook() -> list[Fp4se6Q]:
    codebook = [Fp4se6Q(0.0, 0.0, 0, 0.0, 0.0)]
    for scale, exp, mant in E6M2_SCALES:
        for data in E2M1_POS[1:]:
            codebook.append(Fp4se6Q(scale * data, scale, exp, mant, data))
    codebook.sort(key=lambda q: q.value)
    return codebook


FP4SE6_CODEBOOK = build_fp4se6_codebook()


def build_fp4se4_codebook() -> list[Fp4se4Q]:
    codebook = [Fp4se4Q(0.0, 0.0, 0, 0.0, 0.0)]
    for scale, exp, mant in E4M4_SCALES:
        for data in E2M1_POS[1:]:
            codebook.append(Fp4se4Q(scale * data, scale, exp, mant, data))
    codebook.sort(key=lambda q: q.value)
    return codebook


FP4SE4_CODEBOOK = build_fp4se4_codebook()


def build_fp4se5_codebook() -> list[Fp4se4Q]:
    codebook = [Fp4se4Q(0.0, 0.0, 0, 0.0, 0.0)]
    for scale, exp, mant in E5M3_SCALES:
        for data in E2M1_POS[1:]:
            codebook.append(Fp4se4Q(scale * data, scale, exp, mant, data))
    codebook.sort(key=lambda q: q.value)
    return codebook


FP4SE5_CODEBOOK = build_fp4se5_codebook()


def q_nvfp4(x: float) -> Nvfp4Q:
    if x == 0:
        return NVFP4_CODEBOOK[0]
    return min(NVFP4_CODEBOOK, key=lambda q: (abs(q.value - x), abs(q.value)))


def q_fp4se6(x: float) -> Fp4se6Q:
    if x == 0:
        return FP4SE6_CODEBOOK[0]
    return min(FP4SE6_CODEBOOK, key=lambda q: (abs(q.value - x), abs(q.value)))


def q_fp4se4(x: float) -> Fp4se4Q:
    if x == 0:
        return FP4SE4_CODEBOOK[0]
    return min(FP4SE4_CODEBOOK, key=lambda q: (abs(q.value - x), abs(q.value)))


def q_fp4se5(x: float) -> Fp4se4Q:
    if x == 0:
        return FP4SE5_CODEBOOK[0]
    return min(FP4SE5_CODEBOOK, key=lambda q: (abs(q.value - x), abs(q.value)))


def q_fp4se5_transcoded_to_mxfp6(x: float) -> float:
    q = q_fp4se5(x)
    if q.value == 0:
        return 0.0
    bump = 1 if q.scale_mant > 1.0 else 0
    e2m3 = lut_mantissa_e2m1_to_e2m3(q.scale_mant, q.data, bump)
    return (2.0 ** (q.scale_exp + bump)) * e2m3


def q_mx(x: float, data_values: list[float]) -> float:
    if x == 0:
        return 0.0
    # Wide enough for this sweep and for small normalized values.
    best = 0.0
    best_err = float("inf")
    for exp in range(-20, 20):
        scale = 2.0**exp
        for data in data_values:
            value = scale * data
            err = abs(value - x)
            if err < best_err or (err == best_err and abs(value) < abs(best)):
                best = value
                best_err = err
    return best


def lut_mantissa_e2m1_to_e2m3(
    scale_mant: float, data_e2m1: float, exponent_bump: int
) -> float:
    """Model the transcode LUT.

    The conceptual LUT key is (scale_mantissa_bits, e2m1_code). The output is
    an E2M3 code. The caller determines whether the MXFP6 scale exponent is
    bumped, which divides the value presented to the data LUT by 2.

    Returns:
        e2m3_value
    """
    combined = scale_mant * data_e2m1 / (2.0**exponent_bump)
    if combined == 0:
        return 0.0
    return nearest(combined, E2M3_POS)


def q_nvfp4_transcoded_to_mxfp6(x: float, recipe: str) -> float:
    nv = q_nvfp4(x)
    if nv.value == 0:
        return 0.0
    if recipe == "naive":
        bump = 0
    elif recipe == "revised":
        bump = 1 if nv.scale_mant > 1.0 else 0
    else:
        raise ValueError(f"unknown transcode recipe: {recipe}")
    e2m3 = lut_mantissa_e2m1_to_e2m3(nv.scale_mant, nv.data, bump)
    return (2.0 ** (nv.scale_exp + bump)) * e2m3


def q_fp4se6_transcoded_to_mxfp6(x: float) -> float:
    q = q_fp4se6(x)
    if q.value == 0:
        return 0.0
    bump = 1 if q.scale_mant > 1.0 else 0
    e2m3 = lut_mantissa_e2m1_to_e2m3(q.scale_mant, q.data, bump)
    return (2.0 ** (q.scale_exp + bump)) * e2m3


def q_fp4se4_transcoded_to_mxfp6(x: float) -> float:
    q = q_fp4se4(x)
    if q.value == 0:
        return 0.0
    bump = 1 if q.scale_mant > 1.0 else 0
    e2m3 = lut_mantissa_e2m1_to_e2m3(q.scale_mant, q.data, bump)
    return (2.0 ** (q.scale_exp + bump)) * e2m3


def rel_error(q: float, true: float) -> float:
    if true == 0:
        return 0.0 if q == 0 else float("inf")
    return abs(q - true) / abs(true)


def sweep() -> dict[str, list[tuple[float, float]]]:
    n = int(round((X_MAX - X_MIN) / STEP))
    xs = [round(X_MIN + i * STEP, 10) for i in range(n + 1)]
    series = {
        "Direct NVFP4": [],
        "Direct FP4SE6": [],
        "Direct FP4SE5": [],
        "Direct FP4SE4": [],
        "Direct MXFP4": [],
        "Direct MXFP6": [],
        "NVFP4 -> MXFP6 naive": [],
        "NVFP4 -> MXFP6 revised": [],
        "FP4SE6 -> MXFP6": [],
        "FP4SE5 -> MXFP6": [],
        "FP4SE4 -> MXFP6": [],
    }
    for x in xs:
        nv = q_nvfp4(x).value
        fp4se6 = q_fp4se6(x).value
        fp4se5 = q_fp4se5(x).value
        fp4se4 = q_fp4se4(x).value
        mx4 = q_mx(x, E2M1_POS)
        mx6 = q_mx(x, E2M3_POS)
        trans_naive = q_nvfp4_transcoded_to_mxfp6(x, "naive")
        trans_revised = q_nvfp4_transcoded_to_mxfp6(x, "revised")
        trans_fp4se6 = q_fp4se6_transcoded_to_mxfp6(x)
        trans_fp4se5 = q_fp4se5_transcoded_to_mxfp6(x)
        trans_fp4se4 = q_fp4se4_transcoded_to_mxfp6(x)
        series["Direct NVFP4"].append((x, rel_error(nv, x)))
        series["Direct FP4SE6"].append((x, rel_error(fp4se6, x)))
        series["Direct FP4SE5"].append((x, rel_error(fp4se5, x)))
        series["Direct FP4SE4"].append((x, rel_error(fp4se4, x)))
        series["Direct MXFP4"].append((x, rel_error(mx4, x)))
        series["Direct MXFP6"].append((x, rel_error(mx6, x)))
        series["NVFP4 -> MXFP6 naive"].append((x, rel_error(trans_naive, x)))
        series["NVFP4 -> MXFP6 revised"].append((x, rel_error(trans_revised, x)))
        series["FP4SE6 -> MXFP6"].append((x, rel_error(trans_fp4se6, x)))
        series["FP4SE5 -> MXFP6"].append((x, rel_error(trans_fp4se5, x)))
        series["FP4SE4 -> MXFP6"].append((x, rel_error(trans_fp4se4, x)))
    return series


def write_trend_csv(series: dict[str, list[tuple[float, float]]],
                    path: str = "scalar_sweep_trend.csv",
                    step: float = 0.5, xmax: float = 64.0) -> None:
    """Downsampled per-value rel-error (%) for the deck's native trend chart.

    Uses a smoothed envelope: within each output bin, the mean relative error
    over the fine sweep samples. Restricted to [0, xmax] where the per-octave
    behavior is legible.
    """
    # (series key, csv column name)
    cols = [("Direct MXFP6", "MXFP6"), ("Direct MXFP4", "MXFP4"),
            ("Direct NVFP4", "NVFP4"), ("NVFP4 -> MXFP6 revised", "NVFP4->MXFP6"),
            ("Direct FP4SE4", "FP4SE4"), ("Direct FP4SE5", "FP4SE5"),
            ("Direct FP4SE6", "FP4SE6")]
    fine_step = STEP
    per_bin = max(1, int(round(step / fine_step)))
    n_out = int(xmax / step)
    header = "x," + ",".join(name for _, name in cols)
    rows = [header]
    for b in range(1, n_out + 1):
        x_hi = b * step
        i0 = int(round((x_hi - step) / fine_step))
        i1 = int(round(x_hi / fine_step))
        vals = []
        for key, _ in cols:
            pts = series[key][i0:i1]
            errs = [100.0 * e for _, e in pts if math.isfinite(e)]
            vals.append(sum(errs) / len(errs) if errs else 0.0)
        rows.append(f"{x_hi - step / 2:.3f}," + ",".join(f"{v:.4f}" for v in vals))
    __import__("pathlib").Path(path).write_text("\n".join(rows) + "\n")
    print(f"wrote {path}")


def pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def draw_plot(series: dict[str, list[tuple[float, float]]]) -> None:
    draw.WIDTH = 2400
    draw.HEIGHT = 1050
    draw.img = bytearray(draw.WHITE * draw.WIDTH * draw.HEIGHT)

    left, top = 135, 145
    plot_w, plot_h = 1500, 760
    right = left + plot_w
    bottom = top + plot_h

    max_err = max(err for points in series.values() for _, err in points if math.isfinite(err))
    y_max = max(0.01, math.ceil(max_err * 100.0) / 100.0)
    # Keep a little headroom so curves do not sit on the top border.
    y_max = min(1.0, max(y_max, max_err * 1.08))

    draw.draw_text("Relative Quantization Error Sweep", 70, 42, 44, draw.INK)
    draw.draw_text("True value from 0 to 500, step 0.1", 72, 96, 24, draw.MUTED)
    draw.draw_text("Relative error = abs(quantized - true) / true", 72, 126, 22, draw.MUTED)

    # Plot background and grid.
    draw.rounded_rect(left - 10, top - 10, plot_w + 20, plot_h + 20, 12, (250, 252, 255), (210, 218, 230), 2)
    for i in range(0, 11):
        x = left + i * plot_w // 10
        draw.line(x, top, x, bottom, (232, 237, 244), 1)
        tick = X_MIN + i * (X_MAX - X_MIN) / 10.0
        draw.draw_text(f"{tick:.0f}", x - 14, bottom + 18, 18, draw.MUTED)
    for i in range(0, 6):
        y_val = i * y_max / 5.0
        y = bottom - round(y_val / y_max * plot_h)
        draw.line(left, y, right, y, (232, 237, 244), 1)
        draw.draw_text(pct(y_val), 40, y - 12, 18, draw.MUTED)

    draw.line(left, bottom, right, bottom, draw.INK, 3)
    draw.line(left, top, left, bottom, draw.INK, 3)
    draw.draw_text("True value", left + plot_w // 2 - 55, bottom + 36, 24, draw.INK)
    draw.draw_text("Relative error", 26, top - 44, 24, draw.INK)

    colors = {
        "Direct NVFP4": (38, 119, 184),
        "Direct FP4SE6": (0, 112, 132),
        "Direct FP4SE5": (150, 90, 160),
        "Direct FP4SE4": (82, 151, 36),
        "Direct MXFP4": (211, 119, 35),
        "Direct MXFP6": (44, 145, 105),
        "NVFP4 -> MXFP6 naive": (117, 86, 171),
        "NVFP4 -> MXFP6 revised": (190, 68, 120),
        "FP4SE6 -> MXFP6": (82, 82, 82),
        "FP4SE5 -> MXFP6": (150, 120, 60),
        "FP4SE4 -> MXFP6": (115, 100, 0),
    }

    def map_xy(x: float, y: float) -> tuple[int, int]:
        px = left + round((x - X_MIN) / (X_MAX - X_MIN) * plot_w)
        py = bottom - round(min(y, y_max) / y_max * plot_h)
        return px, py

    for name, points in series.items():
        color = colors[name]
        prev = None
        for x, y in points:
            cur = map_xy(x, y)
            if prev is not None:
                draw.line(prev[0], prev[1], cur[0], cur[1], color, 2)
            prev = cur

    # Legend and summary.
    legend_x, legend_y = 1685, 150
    draw.rounded_rect(legend_x, legend_y, 660, 410, 14, draw.WHITE, (210, 218, 230), 2)
    draw.draw_text("Curves", legend_x + 24, legend_y + 22, 28, draw.INK)
    for i, name in enumerate(series):
        y = legend_y + 68 + i * 35
        draw.rounded_rect(legend_x + 24, y + 5, 30, 12, 5, colors[name])
        draw.draw_text(name, legend_x + 66, y - 1, 17, draw.INK)

    summary_x, summary_y = 1685, 590
    draw.rounded_rect(summary_x, summary_y, 660, 390, 14, draw.WHITE, (210, 218, 230), 2)
    draw.draw_text("Max / Mean", summary_x + 24, summary_y + 22, 27, draw.INK)
    for i, (name, points) in enumerate(series.items()):
        vals = [e for x, e in points if x > 0 and math.isfinite(e)]
        max_v = max(vals)
        mean_v = sum(vals) / len(vals)
        y = summary_y + 60 + i * 34
        draw.draw_text(name, summary_x + 24, y, 15, colors[name])
        draw.draw_text(f"{pct(max_v)} / {pct(mean_v)}", summary_x + 450, y, 16, draw.MUTED)

    note_x, note_y = 60, 965
    draw.rounded_rect(note_x, note_y, 1480, 60, 14, draw.WHITE, (210, 218, 230), 2)
    draw.draw_text("Transcode LUT input = (scale mantissa, E2M1 data); output = nearest E2M3 data. FP4SE4 uses E4M4 scale.", note_x + 22, note_y + 18, 18, draw.MUTED)

    draw.write_png(__import__("pathlib").Path(OUT))


def main() -> None:
    series = sweep()
    if draw is not None:
        draw_plot(series)
        print(f"wrote {OUT}")
    write_trend_csv(series)
    for name, points in series.items():
        vals = [err for x, err in points if x > 0 and math.isfinite(err)]
        print(f"{name:18s} max={pct(max(vals))} mean={pct(sum(vals) / len(vals))}")
    write_manifest()


def write_manifest() -> None:
    """Record how scalar_sweep_trend.csv was made. See kernel/provenance.py."""
    from kernel import provenance

    books = {"E4M3": E4M3_SCALES, "E4M4/FP4SE4": E4M4_SCALES,
             "E5M3/FP4SE5": E5M3_SCALES, "E6M2/FP4SE6": E6M2_SCALES}
    provenance.write(
        "scalar_sweep.manifest.json",
        dict(
            artifact_set="scalar_sweep",
            description="Relative error vs magnitude for a single scalar with an ideal per-scalar "
                        "scale. Source of slide 5.",
            generated_by="quant_error_sweep.py",
            command="python quant_error_sweep.py",
            generated_at_utc=None,
            model_repo=None,
            model_revision=None,
            tensors=[f"linear scalar sweep x in [{X_MIN}, {X_MAX}] step {STEP}"],
            n_elements=int(round((X_MAX - X_MIN) / STEP)) + 1,
            n_quant_blocks=None,          # a scalar sweep has no blocks
            seed=None,                    # deterministic, no RNG
            quant_block=None,
            metric_window=None,           # per-scalar, then averaged into 0.5-wide output bins
            element_format="E2M1 (4-bit) or E2M3 (6-bit), per series",
            scale_format="E4M3 / E4M4 / E5M3 / E6M2 / E8M0, per series",
            scale_book_endpoints={k: [float(min(s for s, _e, _m in v)),
                                      float(max(s for s, _e, _m in v)), len(v)]
                                  for k, v in books.items()},
            # Normals only, and NOT the same convention as vector_quant_error.py's FP4SE books.
            # kernel/p229_bookmatch.py measures that this cannot change any number here, because no
            # swept value is small enough to select a subnormal scale.
            subnormal={k: False for k in books},
            global_scale_recipe=None,
            rounding_tie_rule="nearest value in the full scale x data cross product; ties resolve "
                              "toward the smaller magnitude (nearest/min key)",
            code_sha256=provenance.code_sha256(["quant_error_sweep.py"]),
            environment=provenance.environment(),
            device="cpu",
            source_artifacts={},
        ),
        files=["scalar_sweep_trend.csv"],
    )
    print("wrote scalar_sweep.manifest.json")


if __name__ == "__main__":
    main()
