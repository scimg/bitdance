# AMD Technical Slide Style Guide

A reference for building professional, information-dense technical slide decks in
the **AMD dark corporate theme**, matching the style of `MXFP6_Emu.pptx`
(see `build_pptx.py`). The goal is consistency: any author following these rules
should produce slides that look like they belong in the same deck.

---

## 1. Core principles

1. **Native objects only.** Draw everything with `python-pptx` autoshapes,
   native charts, and native tables. **Do not paste rasterized images** of
   diagrams or charts — they blur, don't recolor, and break on rescale. If you
   can express it as a shape or a chart, do so.
2. **Data comes from real experiments.** Numbers on charts/tables should be
   exported from the project's own runs (scalar sweep, per-block sweep,
   whole-model stats), not hand-typed placeholders.
3. **Information density with calm aesthetics.** Pack meaning, but keep the
   palette muted and the borders subtle so attention lands on content, not
   frame lines.
4. **No emojis.** Use `✓` / `✗` glyphs (green/red) for pass/fail, not emoji.
5. **Reuse the template.** Start from the AMD template so master, fonts, and the
   dark background are inherited rather than re-created.

---

## 2. Canvas & template

- **Template:** `C:\Users\xuanwyin\code\LPT\Presentation.pptx` (dark background).
- **Aspect:** 16:9, `13.33" × 7.5"`.
- **Layouts used:**
  - `slide_layouts[0]` → title slide (`L_TITLE`).
  - `slide_layouts[7]` → blank/title-only content slide (`L_ONLY`), used for
    every content slide.
- Clear the template's stock slides first with `strip_slides(prs)`, then add
  your own.
- **Titles inherit template styling** — set text only via `set_title()`; never
  restyle the title font/color. `set_title` measures the title in the real Arial
  Bold and shrinks it, half a point at a time down to a floor of 22pt, only if it
  would otherwise wrap. **The title box holds exactly one line**: the master
  renders it at 28pt (1 line = 0.447") in a 0.47" box, so a second line is drawn
  outside the box and over whatever is under it. A title that still does not fit
  at 22pt is too long — shorten the words instead. The build prints every title it
  had to shrink.

---

## 3. Color palette

The background is dark, so all fills/text are chosen for legibility on black.
Saturation is medium (never neon).

### Text & surfaces

| Role | Name | Hex |
|---|---|---|
| Primary light text (on dark) | `LT` | `#ECEFF3` |
| Secondary / caption text | `MUTE` | `#B7C1CE` |
| Dark text (only on light fills) | `INK` | `#1F242B` |
| Card / panel fill | `PANEL` | `#1B222C` |
| Faint card border | `PANEL_EDGE` | `#3A4452` |
| Subtle chip border (recedes) | `BORDER_DIM` | `#3C4552` |
| Payoff bar fill (deep green) | `PAYOFF_FILL` | `#223A2E` |

### Datatype-family colors (separated hues, medium saturation)

Assign one hue per datatype family and use it **consistently** across every
chart, table highlight, and legend. `color_for(name)` centralizes this mapping.

| Family | Name | Hex | Meaning |
|---|---|---|---|
| MXFP6 | `BLUE` | `#5A8CC8` | reference / baseline target |
| MXFP4 | `AMBER` | `#DDA23C` | |
| NVFP4 (incl. →MXFP6) | `TEAL` | `#2EB3AC` | |
| FP4SE4 | `VIOLET` | `#8E7BD6` | |
| FP4SE5 | `MAUVE` | `#C882CE` | |
| FP4SE6 | `OLIVE` | `#94B854` | |

### Semantic accents

| Role | Name | Hex |
|---|---|---|
| Good / positive takeaway (text) | `GREEN` | `#7FBE97` |
| Warning / worst-case marker | `RED` | `#E07C72` |
| Green on light surfaces | `GREEN_D` | `#2E7D52` |
| Red on light surfaces | `RED_D` | `#B04547` |

### Pipeline / diagram stage colors (bright accent border+text, dark tinted fill)

| Stage | Accent | Fill |
|---|---|---|
| Storage (NVFP4) | `#7FB0E0` | `#1E2C3C` |
| Transcode | `#E0B25E` | `#322A18` |
| Compute (MXFP6) | `#74C79C` | `#183024` |
| Output | `#A99AE6` | `#27203E` |

**Rule of thumb:** border/text = bright accent, fill = dark tint of the same
hue. Borders always *recede* relative to text.

---

## 4. Typography

- **Font:** inherit template default for titles; use the same sans for body.
  Use **Consolas** (monospace) for worked examples, encodings, and code.
- **Sizes (pt):**
  - Captions / axis labels: `8.5–9.5` (`MUTE`).
  - Body bullets: `10.5–11.5`.
  - Section sub-headers: `12.5–13`, bold, `LT`.
  - Table header/body: `~10.5`.
- **Line spacing:** tight, `0.95–1.05`.
- **Bold** for the first line of a box, first column + header of a table, and
  key terms — not whole paragraphs.

---

## 5. Layout grid

- **Title** at the top (template placeholder), occupying `0.65"–1.12"`.
- **Content** starts ~`1.1"` from top — and on a slide whose title runs the full
  width, start it at `1.16"` instead. `1.02"` is above the title placeholder's own
  bottom edge, which is invisible under a short title and a collision under a long
  one.
- **Two-column split:** left ≈ `0.55"–6.6"`, right ≈ `6.75"–12.85"`.
- **Payoff bar:** full-width strip near the bottom (`y ≈ 6.6"`, `x` from
  `0.55"` to `12.8"`), summarizing the slide's one-line takeaway.
- Keep a consistent outer margin of ~`0.5"`.

---

## 6. Reusable components (helpers in `build_pptx.py`)

All coordinates/sizes are in **inches**.

### Panels / cards
```python
panel(slide, x, y, w, h, fill=PANEL, edge=PANEL_EDGE, edge_w=1.0, radius=0.06)
```
Rounded-rectangle card. Use for grouping related content (definitions, worked
examples). Keep `edge` subtle.

### Boxes & arrows (diagrams)
```python
add_box(slide, x, y, w, h, text, line_c, fill_c, text_c, size=11, bold=True)
add_arrow(slide, x, y, w, h, shape=MSO_SHAPE.RIGHT_ARROW, color=None)  # default MUTE
```
Box border is a thin `0.75pt`. Arrows default to `MUTE` so they don't compete
with labeled stages.

### Payoff bar
```python
payoff_bar(slide, x, y, w, text, h=0.6)   # green fill, GREEN border+text
```
One per slide, bottom strip, states the conclusion.

### Text
```python
tf = textbox(slide, left, top, width, height)
add_para(tf, [("run text", bold_bool)], first=?, size=12, bullet=?, color=LT,
         font=None, line=1.05)
```
`runs` is a list of `(text, bold)` tuples so a single line can mix weights.

### Tables (dark theme)
```python
add_table(slide, data, left, top, width, height, col_w=None,
          header_fill=TBL_HEAD, header_size=10.5, body_size=10.5,
          highlight_col=None)
```
- Header band `TBL_HEAD` `#2C3A4C`; zebra rows `TBL_A` `#171D25` /
  `TBL_B` `#1F2732`; optional highlighted column `TBL_HL` `#22342E` (green
  tint); subtle grid `TBL_EDGE` `#3C4756`.
- **No built-in PowerPoint table style** (borders drawn manually); header row
  and first column bold; text `LT`/`MUTE`.
- `height` is the height the table will actually **render** at: `add_table` states
  every row's height rather than letting PowerPoint split the frame evenly, which
  it then undoes by growing any row too small for its text. A table with a header
  larger than its body used to render below its own frame, and the slide's
  vertical budget did not know. If a table still comes out taller than asked,
  `p219_fit.py` reports it — the rows genuinely do not fit at that font.
- **Keep headers on one line.** The row heights are computed from the font size,
  not from the string, so a `"\n"` in a header cell renders two lines into one
  line's worth of height and pushes the whole table past its frame. Shorten the
  header or widen the column; do not wrap it by hand.

---

## 7. Charts (all native)

### Horizontal bar
```python
bar_chart(slide, x, y, w, h, cats, vals, colors, num_fmt="0.0",
          label_size=10, cat_size=10, maxscale=None)
```
- Legend/title off, value axis hidden, gridlines off.
- Data labels `OUTSIDE_END`, bold, `LT`.
- Per-point fills from the datatype palette; no point borders.
- `cats/vals` given top→bottom (reversed internally).

### XY trend (scatter-line)
```python
trend_chart(slide, x, y, w, h, xs, series, colors, xmax, ymax,
            y_ticks=None, line_w=1.5, dashes=None, widths=None)
```
- True numeric X axis; use for sweeps (e.g. rel-error vs scalar value).
- `dashes` distinguishes direct vs transcode series (e.g. NVFP4→MXFP6 = dashed).
- Distinct color per datatype from the palette.

### Box / whisker (per-method error distribution)
```python
box_whisker(slide, x, y, w, h, rows, lo, hi, ticks, log=False,
            axis_label="", lab_w=1.75, right_pad=0.62, lab_size=9.5)
```
- Each `row` carries `(label, color, box_lo, median, box_hi, cap, max)` — i.e.
  percentiles (p10/p50/p90/p99/max) drawn as autoshapes.
- Prefer this over a single "mean error" bar: it shows the **tail**, which is
  what matters for accuracy claims. Supports `log=True` for wide ranges.

### Legend
```python
legend_row(slide, x, y, items=[(label, color), ...], size=10)
```
Manual color-key dots — use when charts have legends disabled (they always do).

### Chart styling rules
- Gridlines: off, or dim (`~#3A4049`) at `0.5pt` when needed.
- Axis tick labels: `8.5pt`, `MUTE`.
- Never rely on PowerPoint's default legend/theme colors — set every fill.

---

## 8. Slide patterns (how the deck is composed)

1. **Title slide** — template title layout.
2. **Motivation / definitions** — card layout: a short rationale plus a
   datatype-definition table (scale/mantissa encodings). Include relevant
   hardware facts (e.g. "MI355 MXFP4 and MXFP6 have identical peak throughput").
3. **Idea / method** — pipeline diagram of stage chips
   (storage → transcode → compute → output) with muted arrows, plus a
   **worked example** card (mono font, `✓`/`✗`) demonstrating the numeric
   benefit (e.g. exponent-bump on a non-max value). Dark fills, dimmed borders.
4. **Scalar sweep** — `trend_chart` of relative error vs value, one line per
   datatype, dashed line for transcode variants, with a stat table + one-line
   readout.
5. **Synthetic block results** — `box_whisker` per method (same 11 methods as
   the real-model slide) over synthetic weights.
6. **Real-model block results** — identical layout/methods as slide 5, over
   real Qwen weights, so the two are directly comparable.

Every content slide ends with a **payoff bar**.

---

## 9. Do / Don't

**Do**
- Keep one hue per datatype, everywhere.
- Use `GREEN` for "our method wins", `RED` for failure/worst-case, `BLUE` for
  the reference baseline.
- Make borders recede (dim) and text lead (bright).
- Show distributions (box/whisker), not just means, for accuracy claims.
- Keep synthetic vs real slides structurally identical for comparison.

**Don't**
- Paste screenshots of charts/diagrams.
- Use neon/vivid primaries or emojis.
- Restyle the template title.
- Enable PowerPoint's default chart legends/table styles.
- Hand-type result numbers — export from experiments.

---

## 10. Build & preview workflow

```powershell
# 1. (re)generate result CSVs from experiments as needed
python quant_error_sweep.py        # scalar sweep + transcode
python vector_quant_error.py       # per-block distributions

# 2. build the deck (native objects only)
python build_pptx.py               # -> MXFP6_Emu.pptx

# 3. render read-only PNG previews to verify (never edits the live deck)
powershell -File preview_slides.ps1   # -> preview/*.png
```

- The preview script uses PowerPoint COM on a **read-only copy** so it won't
  disrupt an open editing session.
- On Windows, keep plotting/FreeType imports optional so the build never hard-
  depends on matplotlib.

---

_Reference implementation: `build_pptx.py` in this repo. When in doubt, copy an
existing helper call and adjust coordinates rather than inventing new styling._
