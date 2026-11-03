# Reproducing: NVFP4-weight → MXFP6 in-GEMM transcode on AMD MI355

This is the **single entry point** for reproducing the fused NVFP4→MXFP6 GEMM work:
runtime transcode of 4-bit NVFP4 weights to MXFP6 *inside* the production preshuffle
GEMM prologue, benchmarked against W4A4 / W4A6 / W6A6 on LLM and FLUX.1-schnell shapes.

For the full narrative + analysis see **`PROGRESS.md`**; per-attempt tuning log is
**`kernel/opt_log.md`**; MFMA fragment/scale semantics are **`kernel/W4A6_MFMA_RECIPE.md`**.
**`INDEX.md`** maps every file in the repo to what it is for, and separates the ~40 files
that are the live project from the ~1000 that are session-specific one-offs — start there
if you are looking for a file rather than a procedure.

The current best config for any (shape, arm) is not in this document: ask the registry with
`python3 kernel/bestcfg.py report`. Numbers written into prose go stale silently; the
registry has a gate that fails. See §3.3.

---

## 1. TL;DR result

On MI355 (gfx950), `v_mfma_scale_f32_16x16x128_f8f6f4` runs fp4/fp6 at the **same**
MFMA rate (fp8 is half), so MXFP6 math is "free" vs fp4 — the only cost is the transcode
itself. Measured directly with a register-resident microbench (`kernel/hip/mfma_peak.hip`,
no LDS/HBM): **fp4×fp4 = 9909, fp6×fp4 = 9899, fp6×fp6 = 9882, fp8×fp8 = 4962 TFLOP/s**.
Because real GEMMs reach only 19–33% of that ~9.9 PFLOPS peak, the large-M shapes are **not
MFMA-bound but operand-delivery (LDS-read) bound**: fp6 operands are 2× the on-chip bytes
of fp4, which is exactly why W4A6 (fp6 A + *fp4* B) ≫ W6A6 (fp6 A + *fp6* B) despite the
identical matrix-core instruction — see §"Why W6A6 < W4A4". The
kernel adds `in_dtype="w6a6_tc"`: A = native MXFP6, B stored NVFP4 (fp4 16 B/lane +
E4M3 scale), transcoded on-chip to MXFP6 via an **LDS-resident 12-bit LUT**, then fp6×fp6
scaled MFMA. Output is **bit-exact identical to W4A6**.

**The single controlling variable is M (the token/compute dimension):**

| Regime | Example | Best kernel | tc vs W4A6 |
|---|---|---|---|
| **Compute-bound** (large M ≥ ~1K) | LLM prefill, FLUX @1024px image stream (M≥4096) | **W4A6** | 0.61–0.79× (loses; VALU-saturated) |
| **Memory-bound** (small M ≤ ~256) | LLM decode (M=128), FLUX text stream (M=256) | **w4a6_tc** | **+22 – +50%** (wins; 4-bit B + free transcode) |

Mechanism of the small-M win (profiled): identical DRAM traffic and identical VALU count,
but w4a6_tc issues **~half the LDS instructions** (transcodes B once into registers), and
decode is issue-bound, so fewer LDS ops → fewer cycles.

**Two transcode implementations** are benchmarked: `w4a6_tc FlyDSL` (the psg-kernel
`w6a6_tc` path — the **decode/memory-bound champion**, register-B transcode with no
barrier) and `w4a6_tc HIP` (`kernel/hip/w4a6_tc_gemm_v31.hip`, hand-written, **CTA
M-major rasterization** for L2 reuse, default **GROUP=2**). The HIP kernel now runs in
**every regime** — a `BM=1024` prefill build for large M, and `BM=128`/`BM=256` +
**split-K** builds for decode/M256 (previously these were `n/a`, the gap in the slides).
On compute it beats W6A6 where its tiling wins (square 2042, FC1 1817, FC2 1832, LIN2 1835)
and on decode it beats W6A6 on the small-N shapes (DOWN +178%, QKV +36%); on huge-N GATE and
M256 the cooperative-pack barrier makes it trail the FlyDSL transcode (a barrier-free
register-B HIP variant, `-DREGB`, is bit-exact but slower, so the LDS pack stays the
default). A **best-of-GROUP** prefill dispatch (`max(GROUP=2, GROUP=4)`) flips **FC2** over
W6A6, bringing the win count to **8/12**.

> **Update (2026-07-13):** the HIP kernel now covers **all** regimes via a small-BM +
> split-K build (was BM=1024 prefill-only, hence the missing memory-bound cells). The
> FLUX **256-token text stream is memory-bound** (fp4 only 38% of peak): with thin-N
> tiles tc-Fly = 1786, beating W4A6 by +40%. On compute-bound shapes tc does the same
> fp6×fp6 MFMA *plus* a fp4→fp6 pack, so it beats W6A6 only where the HIP tiling
> out-classes FlyDSL-fp6 (square, FC1); W4A6 stays the compute ceiling. See
> `kernel/bench_full_out.txt`.

---

## 2. Environment

- **Host:** `ssh xuanwyin@smci355-ccs-aus-n12-25.cs-aus.dcgpu` (Conductor key auth). gfx950.
- **Container:** `nvfp4_mxfp6_emu` from `rocm/pytorch-private:rocm7.2_aiter_atom_flyDSL_20260626`,
  project mounted at `/work`, pinned to `HIP_VISIBLE_DEVICES=5`. In-container: ROCm 7.2,
  torch 2.9.1+rocm, **flydsl 0.1.9**. Full start command in `kernel/remote_setup.md`.
- The kernel under test is copied into the FlyDSL kernel package at run time:
  `cp /work/psg_preshuffle_gemm.py /root/psg/preshuffle_gemm.py` (all scripts below do this).

---

## 3. Entry point — the kernel

**`psg_preshuffle_gemm.py`** — an editable copy of the production preshuffle GEMM
(`compile_preshuffle_gemm_a8`). Supported `in_dtype`:

| `in_dtype` | Meaning | A operand | B operand |
|---|---|---|---|
| `fp4` | **W4A4** (ceiling) | MXFP4 dense 16 B | MXFP4 dense 16 B |
| `w4a6` | **W4A6** (accurate, no transcode) | MXFP6 32 B (cbsz=2) | MXFP4 dense 16 B (blgp=4) |
| `fp6` | **W6A6** | MXFP6 32 B | MXFP6 32 B |
| `w6a6_tc` / `w4a6_tc` | **NVFP4→MXFP6 weight transcode** (production) | MXFP6 32 B | NVFP4 fp4 16 B → transcoded to fp6 in-kernel |
| `w4a4_tc` | **NVFP4→MXFP6 both operands** (s18) | NVFP4 16 B → fp6 while staging to LDS | NVFP4 fp4 16 B → transcoded to fp6 in-kernel |

`w4a6_tc` is an accepted alias for `w6a6_tc` and selects the identical path. The name
`w6a6_tc` is historical and misleading: the kernel stores W as NVFP4, not fp6. There is no
kernel where both operands are *stored* fp6 and a transcode still runs — if both are already
fp6 there is nothing to transcode.

**Which reference to compare against.** `w4a6` is the fastest arm that also carries A in MXFP6, so
it is tempting to treat it as `w4a6_tc`'s target — but the two do not compute the same product.
`w4a6` feeds B to the MFMA as MXFP4 (a 16 B/lane operand); `w4a6_tc` must feed it as MXFP6
(32 B/lane), because the wide form is the only place NVFP4's E4M3 scale mantissa can go. **`w4a6`
is the less accurate kernel**, and `w4a6_tc`'s 0.59–0.75 ratio against it is mostly the operand
format, not the transcode. `fp6` is the arm that isolates this: it feeds B as fp6 with no transcode
at all, so `gap = (operand format) × (transcode arithmetic)` — e.g. square `0.706 = 0.792 × 0.891`.
Measured against the format it actually needs, `w4a6_tc` runs at **0.88–0.99** of `fp6` on compute
shapes while reading **half** of `fp6`'s B bytes. Quote `tc/fp6` when the question is kernel
quality and `tc/w4a6` only when the question is end-to-end throughput at differing accuracy.

Key regions in the file:
- LUT global→LDS cooperative copy + per-block LDS row view: search `lut_lds_ptr`, `_lut_row_view`.
- The transcode itself: `def _tc_bdata_fp4_to_fp6` (fp4→fp6 data) and `_tc_bscale_e4m3_to_e8m0` (scale).
- Transcode is hoisted **once per B-tile** (above the `mi` A-row loop): search `is_w6a6_tc`.
- A's transcode for `w4a4_tc` lives in the staging copy: search `store_a_tile_to_lds`.

### Tuning flags (environment variables, read at compile time)

| Variable | Default | Effect |
|---|---|---|
| `TC_HWCVT` | off | Use the hardware scale-converts instead of the 12-bit LDS LUT. Not bit-identical to the LUT; gated on rel-L2 vs fp32. Worth +7..32%. |
| `TC_SCALE_BYTEWISE` | **on** | SWAR the E4M3→E8M0 scale convert (all 4 bytes at once). Bit-identical. |
| `TC_MF_INT` | **on** | Build the `(1+m/8)` mantissa factor as an f16 bit pattern rather than f32 arithmetic. Bit-identical. |
| `TC_A_SERIAL` | **on** | `sched_barrier` between `w4a4_tc`'s staged A blocks. Keeps the two 16-VGPR `pk32` tuples from co-residing: 274→256 VGPR, 1→2 waves, +26%. |
| `TC_WRITE_PAD` | **on** | Write A's unread FP8 pad dword. Skipping it is numerically free but measures **−10%** (see `kernel/opt_trace_s18.md` §4); the flag exists to keep that result reproducible. |
| `WG512` | off | 512-thread (8-wave) workgroups. **Compute shapes only** — costs 20–30% on M=128 decode. Two configs are useful under it and they target different operands: `tile_n=512` (i.e. `128,512,128`) is the **A-side** lever, worth +22..55% to `w4a4_tc`; `256,256,128` is the **B-side** lever, worth a median +2.9% to `w4a6_tc` and best on 5 of 12 rows. See §3.1. |
| `TC_FOLDSCALE` | off | Diagnostic: fold the scale mantissa onto a convert's f32 scale operand. **Produces wrong results** — the hardware uses only that operand's exponent. Kept as evidence. |

`WG512` also switches the launch to `block=(512,1,1)` and adds `known_block_size=[512,1,1]`
to the kernel decorator; without the latter AMDGPU rejects the launch, since
`max_flat_workgroup_size` defaults to 256. The 256-thread paths are bit-for-bit unaffected.

### 3.1 Which tile to use, and which tiles are rejected

The two transcoding kernels pay different costs and therefore want different tiles.
`w4a6_tc` transcodes **B only** (its A is already MXFP6), so its cost law is
`VALU/MFMA = base + tcB/tile_m` — no A term. Raising `tile_m` is what helps it; raising `tile_n`
is what helps `w4a4_tc`, which stages a transcoded A through LDS. What decides whether a tile is
reachable is the accumulator budget, which must stay at 128 VGPRs:

    acc_VGPR = (tile_m/16) · (n_per_wave/16) · 4,   n_per_wave = tile_n / num_waves

| config | threads | n_per_wave | acc VGPR | helps |
|---|---|---|---|---|
| `128,256,128` | 256 | 64 | 128 | the general-purpose default |
| `256,128,128` | 256 | 32 | 128 | `w4a6_tc` B-side, 256 threads |
| `128,512,128` | 512 | 64 | 128 | `w4a4_tc` A-side |
| `256,256,128` | 512 | 32 | 128 | `w4a6_tc` B-side |
| `256,512,128` | 512 | 64 | **256** | nothing — spills 348 times, ~10× slower |

**`num_acc_n` must be a positive even number** on every 4-bit-B path (`fp4`, `w4a6`, `w6a6_tc`,
`w4a4_tc`). The packed-N MFMA consumes two accumulator blocks per step, so an odd count silently
drops the last block and a zero count writes no `C` at all — and **both failure modes measure as
speedups**. `tile_n=384` at 8 waves gives `num_acc_n=3` and reported 2957 TFLOP/s against a true
best of 2177 while computing two thirds of `N`. The compiler now rejects these at compile time:

```
NotImplementedError: 4-bit B (fp4/w4a6/w6a6_tc/w4a4_tc) needs an even number of N accumulator
blocks per wave, got num_acc_n=3 for tile_n=384 on 8 waves ...
```

Note that legality depends on `tile_n` **and** the wave count: `tile_n=384` is fine at 256 threads
(`num_acc_n=6`) and degenerate at 512 (`num_acc_n=3`). If you add a tile to a sweep, check
`tile_n // num_waves // 16` is even before trusting any number it produces.

### 3.2 Split-K — required on two rows, harmful everywhere else

The tile is only half the configuration. This GEMM parallelises across **output** tiles only:
workgroups = `ceil(M/tile_m) · ceil(N/tile_n)`, which scales with M×N and not at all with K. When
that product falls below the 256 CUs, the device idles no matter how good the tile is, and the
only remaining free dimension is the K reduction.

Pass `split_k=N` (or `--splits` to `bench_sweep.py`). Each of the `N` workgroup-z slices reduces
`K/N` and writes its own **partial** C plane, so the caller must allocate `split_k*M*N` and sum
over the leading axis. **That sum is not free and must be inside whatever you time** — see below.

| row | workgroups at `split_k=1` | best `split_k` | `w4a6_tc` effect |
|---|---|---|---|
| `LLM_DOWN_dec` 128×2560×9728 | 40 | **19** | 130 → **393** TFLOP/s (+202%) |
| `LLM_QKV_dec` 128×4096×4096 | 64 | **4** | 250 → **340** TFLOP/s (+36%) |
| `LLM_GATE_dec` 128×19456×2560 | 304 | 1 | −0.1% |
| `FLUX_FC1_M256` 256×12288×3072 | 384 | 1 | −0.8% |
| `LLM_square` 4096×4096×8192 | 512 | 1 | −0.2% |

Read the middle column, not M. `GATE_dec` and `DOWN_dec` are both M=128; only the one whose grid
cannot fill the device benefits. Split-K must be **selected per shape**, never applied globally.

Legal values are constrained three ways, and the third is easy to miss: `K/split_k` must be a
multiple of `tile_k`, **a multiple of 256** (the block-scaled 4-bit path addresses scales in
256-element K chunks), and at least `4*tile_k` so the pipelined K loop keeps its prologue and
drain. For `K=9728` that leaves only `{1, 2, 19}`. Illegal values raise at compile time.

**The reduction is currently a third of the cost on these rows.** On `DOWN_dec` at `split_k=19`
the kernel itself runs at 640 TFLOP/s and the `torch.sum` over 19 bf16 planes drags the usable
figure to 416. Every number in this document is the net one. Folding the reduction into the kernel
is the largest known piece of remaining headroom — see `kernel/opt_trace_s20.md` §4.

Verify with `kernel/s17_splitk_check.py`, which checks that the summed partials reproduce the
full-K output under per-K-chunk-varying block scales (so a skipped or duplicated slice cannot
hide). All arms and all splits currently return `rel ≈ 2.3e-03`, the expected bf16 re-rounding.

---

## 3.3 The best-known-config registry (read this before writing a new sweep driver)

`kernel/best_known.json` holds the best measured TFLOP/s per (shape, arm) together with the
full config that produced it — tile, thread count, `split_k` — plus the reference value, the
session and the date. `kernel/bestcfg.py` is the only interface to it, and it is also the only
place the shape suite is defined.

**Do not hardcode a shape list, a tile list, or a `--splits` value in a new driver.** That is
not a style preference; it is the specific mistake that cost this project a +177% result for
three sessions. `s18_dispatch.sh:19` and `s19_dispatch.sh:19` are byte-identical lines, both
carrying `--splits 1`, because the `go()` helper was copy-pasted forward and nobody re-reads a
helper they copied. `kernel/s21_bench.sh` is the model to follow: it asks `bestcfg.py space`
for its config space and `bestcfg.py shapes` for its rows, so there is no constant left to go
stale.

```bash
python3 kernel/bestcfg.py shapes --cls MEM           # grid-starved rows
python3 kernel/bestcfg.py space  --shape G4_DOWN_dec --arm w6a6_tc
python3 kernel/bestcfg.py report                     # everything on record
python3 kernel/bestcfg.py check  --shape LLM_DOWN_dec --arm w6a6_tc --tflops 130
#   REGRESSION  LLM_DOWN_dec/w6a6_tc: 130 vs record 392.0 (-66.8%, tol 3.0%)   -> exit 1
```

`check` exits nonzero when a cell falls more than `--tol` (default 3%) below its record, which
is above the run-to-run spread on a quiet card (0.37% across a clean run) and far below
anything worth calling a regression. When it fires, **reproduce the recorded config before
theorising** — the s19 failure was not missing the regression, it was explaining it with a
plausible mechanism and moving on.

Two further rules the registry enforces rather than documents:

- **A record is only taken against a quiet reference.** `s21_bench.sh` probes the reference
  before *and* after every row and records only if both land in the quiet band. Sampling only
  afterwards is not enough: the first Gemma pass recorded `G4_O/fp4` at 2167 against a clean
  trailing probe of 2197 while `G4_O/w4a6` scored 2656 on the same tile, which is not physical
  — contention had come and gone entirely inside the fp4 row. Use `bestcfg.py purge` to remove
  such a record; leaving it in place lowers the bar every future run is measured against.
- **A pinned axis carries its reason.** `SPLITS_CB = [1]` sits next to
  `SPLITS_CB_PINNED_BECAUSE`, which cites the control that justifies it (split-K is −0.2% on
  `LLM_square`, whose output grid already fills the device). A bare constant is how the last
  one got lost.

`split_k` candidates are **derived, not enumerated**: `bestcfg.py` computes the legal set from
K (`K/split_k` a multiple of `tile_k` and of 256, leaving ≥4 k-tiles) and keeps those landing
the grid between 0.5× and 6× the 256 CUs. The high end is generous deliberately —
`LLM_DOWN_dec` wins at `split_k=19`, which is 760 workgroups, 3× oversubscribed.

## 4. How to run

All scripts live in `kernel/`, are `scp`'d to the host, and executed with
`docker exec -w /work nvfp4_mxfp6_emu bash /work/kernel/<script>.sh`. Pattern used
throughout (PowerShell):

```powershell
scp -o BatchMode=yes kernel/<script>.sh xuanwyin@<host>:~/nvfp4_mxfp6_kernel/kernel/
ssh -o BatchMode=yes xuanwyin@<host> "docker exec -w /work nvfp4_mxfp6_emu bash /work/kernel/<script>.sh" > kernel/<script>_out.txt 2>&1
```

| Script | Purpose | Output |
|---|---|---|
| `kernel/bestcfg.py` | **Shape suite + best-known-config registry + regression gate.** Single source of truth; see §3.3 | `kernel/best_known.json` |
| `kernel/s21_bench.sh` | **Current benchmark driver.** Hardcodes nothing; sources shapes and config space from the registry, gates the reference before/after each row, records only when quiet. `SHAPES=G4_ CLS=MEM ARMS="fp6 w6a6_tc"` to subset | stdout + registry |
| `kernel/s21_wait_run.sh` | Polls the reference until two consecutive probes are quiet, then runs `s21_bench.sh`. Use on a shared node instead of hand-retrying (s20 burned 14 attempts doing that by hand) | stdout |
| `kernel/s21_table.py` | Renders the dispatch table from the registry — no numbers live in the script | stdout |
| `kernel/s21_atomic_check.py` / `s21_atomic_gate.sh` | Correctness + crossover for `split_reduce="atomic"`. Verdict: correct, more accurate than the default, and slower at every `split_k` worth using | `kernel/s21_atomic_gate.txt` |
| `kernel/seed_records.sh` | One-shot seeding of the registry from the s19/s20 quiet tables | — |
| `kernel/bench_full.sh` | **Unified benchmark** — all 5 kernels (FlyDSL fp4/w4a6/fp6/w6a6_tc **+ HIP v31 w4a6_tc**) on all LLM + FLUX shapes, one combined table | `kernel/bench_full_out.txt` |
| `kernel/seq_group_sweep.sh` | HIP v31 CTA-rasterization `GROUP` sweep (`{2,4,8,16,32}`) over the 6 large-M shapes → picks default. **GROUP=2** is the best single default (bit-exact); set as kernel default. | stdout |
| `kernel/bench_all.sh` | Legacy 4-kernel FlyDSL-only benchmark (W4A4/W4A6/W6A6/w6a6_tc) | `kernel/bench_all_out.txt` |
| `kernel/verify_decode.sh` | Correctness: cross-check `w6a6_tc == W4A6` (same A) bit-exact at winning tiles | `kernel/verify_decode_out.txt` |
| `kernel/decode_stable.sh` | High-iter re-measure of the decode win (stability, <1% spread) | `kernel/decode_stable_out.txt` |
| `kernel/prof_decode.sh` | rocprofv3 counters explaining the decode win (LDS/VALU/FETCH) | `kernel/prof_decode_out.txt` |
| `kernel/crossover.sh` | Original LLM decode/prefill crossover | `kernel/crossover_out.txt` |

Single-shot correctness check (any shape/tile):
```
python3 kernel/verify_w4a6.py --dtype w6a6_tc --m 128 --n 2560 --k 9728 --tm 128 --tn 64 --b_scale 2.0
# prints:  [cross-check vs W4A6 same-A] bit-exact match: True
```

### 4.1 The model-level pipeline (accuracy and end-to-end throughput)

Everything above measures a GEMM. Reproducing an *accuracy* number is a different sequence, and
the order matters because a cell is ~40 minutes of card time on a node that takes the GPU back
on its own schedule.

```bash
# 1. gates first -- both are minutes, and both have caught a wrong number that looked right
docker exec -w /work nvfp4_mxfp6_emu python3 -m kernel.qfused_gate         # quantizer, byte-exact
docker exec -w /work nvfp4_mxfp6_emu python3 -m kernel.p15_smooth_gate     # equalizer, 3 claims

# 2. plain arms, so one point per arm exists before any equalized delta does
PLAIN_ONLY=1 bash kernel/p15_run_eval.sh

# 3. equalized arms, calibrated IN DOMAIN. Do not skip the distinction: the same code
#    calibrated on prose scores fp6 at 33.6% and in domain at 54.4%.
bash kernel/p16_recal.sh                    # -> AMAX_FILE=..._v2.pt, EQ_SUFFIX=_eq2

# 4. record, then gate
docker exec -w /work nvfp4_mxfp6_emu python3 -m kernel.best_e2e record --session sNN --note "..."
docker exec -w /work nvfp4_mxfp6_emu python3 -m kernel.best_e2e check
```

Things that will otherwise cost a day:

- **`--perms 4` is not optional.** Scoring each item in its given option order measures option
  position as well as accuracy: bf16 put 67% of its GPQA answers on one slot against a 25% base
  rate, and the arm that looked best on GPQA was mostly the arm that was least position-biased.
  A run without it is not comparable to anything in `best_known_e2e.json`'s `@p4` cells.
- **The equalizer's calibration split must be disjoint from the scored split** (`--split`
  in `eval_bench.load_bench`; `p15_calibrate` uses MMLU-Pro validation). Calibrating on the
  scored items would be leakage; calibrating on generic prose is what broke it the first time.
- **New results go to their own directory, never into an existing one.** A resumable run treats
  an existing line as a finished item, so mixing scoring generations in one directory silently
  keeps the old answer. Equalized cells are tagged (`_eq`, `_eq2`) for the same reason. s22 wrote
  `e2e_results`, s23–s24 `e2e_results_v2`, s25 `e2e_results_s25`.
- **Write the manifest before the first cell.** `python3 -m kernel.p18_write_manifest --dir DIR`
  stamps the directory with the code state that is about to score it. Without it `best_e2e
  record` has to infer provenance from whether the accuracy moved, and a cell re-scored today
  keeps yesterday's session stamp.
- **`PASSES`/exit 75.** The driver retries cells in passes and exits 75 when a whole pass makes
  no progress, which tells `supervise.sh` the card is gone rather than the code broken.
- **vLLM throughput** (`kernel/p16_vllm_supervise.sh`) needs `HF_HUB_OFFLINE=1` — Hub 429s during
  engine init surface as "engine core initialization failed", which reads like a bug in the arm —
  and a **persistent** container driven by `docker exec`, because a `--rm` container per attempt
  gives the node's GPU reaper a fresh target during every 60–90 s weight load. s23 still failed
  to complete a sweep this way; see `kernel/opt_trace_s23.md` §8.

### 4.1b The generative protocol (s38) — what the reported accuracies now come from

Everything in 4.1 builds a cell from `kernel/eval_bench.py`, which reads one token's logprobs.
Every accuracy this project reports is now built the other way: the model is asked for an
answer and the answer is parsed. That is a different measurement, not a re-scoring — 12B bf16
on GPQA is 42.4% under the cloze harness and 59.60% here — so the two are kept in different
files and no slide or figure may mix them. `ACCURACY_DEBUG.md` §11–14 is the account.

> **The commands below are the record of how these cells were produced, not a recipe that can be
> run today.** The node's `/work/kernel/e2e_results_gen` is empty and nothing matching `*_nothink*`
> survives anywhere in the container — `kernel/e2e_gen/` in this repository is the only copy of
> every accuracy the deck reports. Re-running would not restore them either: at `T=1.0` a fresh run
> is a different draw, five points wide on GPQA at n=198 (§16.7). Treat this directory as primary
> data. `ACCURACY_DEBUG.md` §16.8.

```bash
# One arm per card. Thinking OFF, 16,384-token cap (see the cap note below -- 4,096 was the setting
# until s38 and it cost the damaged arms up to 14 points). MMLU-Pro is scored on 500 items and GPQA
# Diamond on all 198; `LIMIT=500` below is what makes the first of those a stratified subsample of
# the benchmark's 12,032, and every arm and both models share that identical sample.
REPO=google/gemma-4-12B-it BENCH=mmlu_pro SUFFIX=_nothink16k EXTRA=--no-think \
  ARMS='bf16 w6a6 w4a6_tc w4a6 w4a4_tc fp4 fp6' CARDS='0 1 2 3 4 5 6' \
  BATCH=8 MAXNEW=16384 bash kernel/p172_gen_run.sh

N=1 bash kernel/p181_tails.sh                      # progress AND row counts, see below
PYTHONPATH=. python3 -m kernel.p185_lattice        # the ladder, paired McNemar
bash kernel/p190_figs.sh                           # summaries -> figures -> 31B plane
PYTHONPATH=. python3 -m kernel.p192_deckcheck      # before building the deck on Windows
```

Everything downstream of the cells now rebuilds **on Windows**, which is where the deck has to be
built anyway (Windows template path, python-pptx) and is now also where the only copy of the cells
lives. `C:\Users\xuanwyin\AppData\Local\miniforge3\python.exe` has python-pptx 1.0.2 and
matplotlib 3.11, so the whole chain runs in one place and every step can be checked against the
next:

```powershell
$P = "C:\Users\xuanwyin\AppData\Local\miniforge3\python.exe"; $env:PYTHONPATH = "."
& $P -m kernel.p197_capaudit --suffix _nothink16k   # right cap, and n equal across cells
& $P -m kernel.p189_gen_summary --model gemma-4-12B-it --out kernel/gen_summary_12B.json
& $P -m kernel.p189_gen_summary --model gemma-4-31B-it --out kernel/gen_summary_31B.json
& $P -m kernel.p13_tradeoff --acc kernel/gen_summary_12B.json --model 12B --out kernel/tradeoff.png
& $P -m kernel.p202_captable                       # cap table; asserts it matches the summaries
& $P -m kernel.p192_deckcheck                      # every cell the deck asks for
& $P build_pptx.py; & $P check_pptx.py             # build, then geometry
& $P -m kernel.p199_deckaudit                      # accuracy literals the data cannot justify
```

If a 4,096 cell already exists, do not re-run the dataset. `CARRY=_nothink` copies every
untruncated record forward and regenerates only the items the old cap cut, which is both ~4x
cheaper and a paired test of the cap itself (`kernel/p195_capdelta.py` reads the pair):

```bash
REPO=google/gemma-4-12B-it BENCH=mmlu_pro SUFFIX=_nothink16k CARRY=_nothink \
  EXTRA=--no-think ARMS='fp4 fp6' CARDS='0 1' BATCH=8 MAXNEW=16384 \
  bash kernel/p172_gen_run.sh
PYTHONPATH=. python3 -m kernel.p195_capdelta       # what the cap was worth, per arm
```

Things that will otherwise cost a day, and did:

- **The psg overlay is the whole ballgame.** `/flydsl_pkg/psg` is a bind-mounted snapshot shared
  by every container and it is 189 lines behind `/work/psg_preshuffle_gemm.py`, with no `w6a6`
  in its `in_dtype` tuple. A launcher that sets `PYTHONPATH=/flydsl_pkg:/work` measures a
  different kernel than every recorded cell, and the 6-bit arm dies at layer zero claiming it
  does not exist — which reads exactly like an unimplemented arm. `p172_gen_run.sh` now builds
  `/tmp/psgover`, hash-checks it, refuses to launch on a mismatch, and writes
  `psg overlay: 9de48186c6f3e8c4` as the **first line of each cell's log**, because the question
  asked six months later is not "what did this batch use" but "what did this number use".
- **`--no-think` is a deliberate choice, not a default.** The canonical chat template appends an
  empty thought block unless `enable_thinking=True`, so every pre-s38 cell had thinking off
  without anyone deciding that. Thinking on is the card's protocol and scores higher; it is also
  ~26 min for 8 GPQA items on 31B against ~11 min for 64, which is why the reported lattice is
  no-think and says so in its protocol string.
- **The cap is a per-arm penalty, and 4,096 was priced on bf16.** `p183_tailvalue.py` showed
  accuracy falling monotonically with reply length and reaching zero past 8,192, so 4,096 looked
  like 0.8 points for a 3.2× wall-time saving. That price is bf16's, which truncates 1–4% of
  items; the arms being ranked truncate up to 64% and were charged 8–14 points. Raising the cap
  to 16,384 flipped one pair and dissolved three orderings the 4,096 cells had resolved at
  p<0.05. Report truncation per arm beside every number, and treat an arm that leads only because
  it truncated less as unranked. `ACCURACY_DEBUG.md` §16.
- **Raising the cap lowers the batch, and the worst arm is where it bites.** KV cache is
  batch × `max_new`. 31B needs `BATCH=2` at 16,384; at 8 the allocator OOMs and *retries*, so the
  cell crawls rather than failing. The arms that need the high cap are exactly the ones whose
  replies reach it, so a batch that survived six arms can still die on the seventh.
- **Watch row counts, not the progress line.** A cell that has written zero rows for 40 minutes
  prints the same last line as a healthy one. `p181_tails.sh` prints both.
- **Kill inside the container.** A host-side kill leaves the process holding the GPU;
  `PAT='...' bash kernel/p184_kill.sh` matches on the command line in-container.
- **New protocol, new suffix — and the suffix is not evidence.** Cells are tagged `_nothink16k`
  and loaded by suffix, for the same reason 4.1 tags `_eq2`: a resumable run treats an existing
  line as finished, so a directory that mixes protocols silently keeps the wrong answer. The cap
  is part of the protocol, which is why raising it got its own suffix rather than overwriting
  `_nothink` — the 4,096 cells are what makes §16's paired test possible and are kept. But a
  filename records an intention, not a measurement: one `_nothink` cell was run at 16,384 and sat
  in the directory for a session looking like every other one.
  `PYTHONPATH=. python3 -m kernel.p197_capaudit --suffix _nothink` recovers each cell's real cap
  from its own truncated rows and is worth running before any cross-cap comparison.
- **A torn last line costs an item, not a character.** Every loader here read records with
  `except Exception: continue`, so a run killed between the write and the flush left one cell at
  n=197 beside 198 for its 27 siblings — with no error anywhere, because `n` was reported per cell
  and never compared across them. `p189_gen_summary.py` and `p195_capdelta.py` now raise on an
  unparseable line, `p197_capaudit.py` compares `n` across every cell on a benchmark, and
  `PYTHONPATH=. python3 -m kernel.p201_mend --apply` recovers the record when the tear landed after
  the scored keys. `ACCURACY_DEBUG.md` §16.8.
- **A protocol string is a claim, and it outlives the protocol.** `p189_gen_summary.py` published
  `whole dataset`, which was true of the cloze runs (`p148_full.sh` scored all 12,032 MMLU-Pro
  items) and false of these, which sample 500 of them — and the phrase reached both figure captions,
  four slide footnotes and two appendix headings before anyone read it against the 12,032 the same
  deck reports four slides earlier. Coverage is now derived from `n` against the split size and
  published per benchmark as `_meta.items` / `_meta.full_split`; the figures and the deck read those
  instead of asserting coverage in prose. `ACCURACY_DEBUG.md` §16.9.
- **`--carry-from` needs the item set pinned to the source.** `--limit` and the carried set are
  two independent ways of saying which items a cell covers; when they disagreed, a 500-item
  MMLU-Pro cell grew to 11,700. `eval_gen` now restricts the pool to the source cell's ids.
  `kernel/p196_trim.py` repairs a cell already written that way, and must run **inside** the
  container because `e2e_gen/` is root-owned.

### 4.1c The end-to-end throughput sweep (and the three ways it lied before s25)

```bash
# One arm per line, on ONE verified-idle card, all three batch sizes per arm.
SERIAL=1 GPU=0 MXFP6_REQUIRE_IDLE=1 WORK=$PWD PKG=$HOME/flydsl_pkg \
  bash kernel/p18_sweep_m11.sh                       # -> kernel/vllm_m11_<arm>.json

python3 -m kernel.p18_kernel_parity                  # same kernel as the accuracy cells?
python3 -m kernel.p18_config_audit                   # measured tiles, not untimed defaults?
python3 -m kernel.p18_trace_audit --dirs 'trace_m11_*'   # what actually ran, per arm
```

Each of these exists because the sweep produced a plausible wrong answer without it:

- **The arm must be recorded, not requested.** `MXFP6_ARM` does not reach vLLM's EngineCore
  child process, and the plugin used to default to `fp4`, so s22–s24 measured fp4 under five
  arm names. The plugin now raises if no arm was passed by any of its three channels, and
  `kernel/p12_vllm_bench.py` refuses to file a row whose `built_arms` is not the requested arm.
  A row without `kernel_id` and `built_arms` predates this and cannot be trusted.
- **The card must be exclusively ours.** `MXFP6_REQUIRE_IDLE=1` refuses to start on a card
  that already holds someone else's memory. Mid-run arrivals are caught afterwards by the
  **control kernel**: every quantized arm runs one identical bf16 lm_head GEMM per decode step,
  so `control_us_per_step` is one quantity measured once per arm, and the sweep prints its
  spread. A co-tenant put one arm's control at 817 µs against 308–341 for the rest.
- **The kernel in the container must be the kernel in the repo.** `kernel/p12_vllm_run.sh`
  overlays `psg_preshuffle_gemm.py` onto the staged package and asserts the sha256 matches
  before anything runs; the staged snapshot had drifted 38 lines behind, missing an entire arm.

Batch 32 is the profiled batch (`--profile`), which is what fills `profile_us_per_step` and
writes the per-arm trace directory. Older rows call that field `profile_ms`; the values were
always microseconds, and readers accept both names.

### 4.2 The accuracy matrix, under one checkout

A table of arms is a set of *differences*, so every cell in it has to come from one build. The
deck's table did not: `w6a6` was added a session after the rest and carried a different
`qlayout.py`. Re-measure the family rather than the cell that looks wrong — the `bf16` control
moves too, and it is the reference every delta is taken against.

```bash
# On the node, one arm per GPU. Writes _code_state.json before the first cell is scored.
ARMS="bf16 fp4 w4a6 w6a6 w6a6_tc w4a4_tc fp6" bash kernel/p18_matrix_m11.sh

docker exec -w /work mxfp6_vllm_fp4 python3 -m kernel.eval_analyze \
    --repo google/gemma-4-12B-it --out-dir /work/kernel/e2e_results_s25 \
    --json-out /work/kernel/e2e_summary_s25.json

# Pull the per-item records back -- the summary alone cannot be re-audited.
tar czf ~/e2e_s25.tgz -C ~/mxfp6_gen/kernel e2e_results_s25

python3 -m kernel.best_e2e record --note "s25 single-checkout matrix"
bash kernel/p18_all_audits.sh
```

- **Pull the records, not just the summary.** Every paired test — the equalizer audit, the
  before/after diff, the bf16 control — needs per-item correctness, and a summary cannot supply
  it. This is how the `w6a6` records went missing and the equalization audit came to report a
  clean sweep over four pairs it had never run.
- **One equalizer calibration for the whole family.** `_eq2` cells are only comparable to each
  other if they share `act_amax__gemma-4-12B-it__v2.pt`. `p18_calib_determinism.py` shows a
  rebuild is bit-identical, so losing the file is recoverable — but only by re-measuring every
  cell that depended on it.
- **`best_e2e record` writes what the records say.** The 0.005 tolerance decides whether a move
  is flagged as a regression, and nothing else. It used to also decide whether the new number was
  stored, which left five cells at their pre-s25 values under records that disagreed.
(The `RESULT: FAIL` on the absolute check is a **pre-existing host fp6-A swizzle-mirror
gap** that hits W4A6 and w6a6_tc identically — hence the cross-check matches. It is a
verification-harness limitation, not a kernel bug; see `PROGRESS.md`.)

### 4.3 The same measurements on a second model (Gemma 4 31B)

Everything above is one model, and a codec result on one model is an anecdote. Repeating it on
31B costs three scripts and one prerequisite that is easy to skip:

```bash
# 0. PREREQUISITE, and it is not optional -- see the fourth bullet below.
python3 kernel/g31_tilecheck.py            # coverage matrix + what choose_config will pick
CARD=0 SESSION=s34 bash kernel/g31_tiles.sh       # the shapes with no row at all
bash kernel/g31_decode_tiles.sh                   # decode-scored tiles, one arm per card
MERGE_ONLY=1 bash kernel/g31_decode_tiles.sh      # -> best_known_decode_total.json

# 1. Accuracy: seven arms, one card each, four option orders, shipped recipe.
SESSION=s34 REPO=google/gemma-4-31B-it bash kernel/p98_g31_matrix.sh
bash kernel/g31_record_acc.sh              # -> best_known_e2e.json under gemma-4-31B-it/...@p4

# 2. Performance: profile, step wall, layer bench -- one card, sequential, idle-guarded.
CARD=0 bash kernel/p99_g31_perf.sh
python3 kernel/p95_reconcile.py --profiles 'kernel/g31_prof_*.json' \
    --walls 'kernel/g31_wall_*.json' --layers kernel/g31_layer_bench.json \
    --model-label "Gemma 4 31B" --out-json kernel/g31_reconcile.json \
    --out-pie kernel/bf16_breakdown_g31.png
```

Four things the 12B pass did not have to deal with:

- **A shape suite tuned for one model is not tuned for another, and the run will tell you.**
  vLLM's plugin prints `untuned_shapes` and the 31B run named two — the global-attention layers,
  which `bestcfg.py` never had because the public config does not pin their fused width. The
  checkpoint does, at load. Read that field before trusting any 31B timing: a shape missing from
  the registry does not fail, it silently takes 128×128×128/split_k 1, and the tuned decode
  configs here are split_k 3 and 8.
- **An arm added after a sweep has no rows in it, and nothing says so.** `w6a6` had no measured
  configuration for *any* 31B shape. Count the coverage matrix per (shape, arm) — a sweep log's
  tail cannot show you a row that was never attempted.
- **Two registries cover these shapes and only the second one decides.** `kdispatch` applies
  `best_known_decode_total.json` last, and every one of its 188 rows was a 12B shape, so the 31B
  GEMMs fell back to `best_known.json`'s M=128 row — tuned at a batch these shapes never see, and
  scored on GEMM kernel time rather than on the layer. All 36 (shape, arm) pairs want something
  else at M=32, and running `p14_decode_total --shapes 31B` first is worth **+16% and +19% of
  served throughput on the two transcode arms**. Without it the 31B and 12B columns are not the
  same measurement, and the difference reads as a fact about model size. `bash
  kernel/g31_decode_tiles.sh` is ~20 min on six cards; it is a pure kernel sweep, so it needs no
  weights and does not care about co-tenants beyond the timing noise.
- **Pull before you push.** `best_known*.json` is written on the node and read from the repo.
  `push.ps1` now refuses when the two differ; that guard exists because a push deleted 24 rows
  that had just been measured, and the only symptom was slower kernels.
- **`PLAIN_ONLY=1` for the matrix.** Seven parallel 31B evals each doing their own equalizer
  calibration is 7 × 60 GiB of bf16 weight loading for cells nothing ships.

---

## 5. Transcode bit-contract (for an independent re-implementation)

NVFP4 per-32 block = one **E4M3** scale byte + 32 **E2M1** 4-bit codes
(`sign=bit3`, `mag_index=code&7`, magnitudes {0,.5,1,1.5,2,3,4,6}).
- `idx = mant_bits*8 + mag_index`; `e2m3_code = (sign<<5) | CODE_LUT[idx]`
- `e8m0_byte = exp + BUMP_LUT[mant_bits] + 127`, where `exp=((scale>>3)&0xF)-7`, `mant_bits=scale&7`

The kernel packs the 12-bit `CODE_LUT` (2048 i32 = 8 mant rows × 256) via
`build_LUT12()` in `kernel/nvfp4_mxfp6_gemm_tiled.py`; it is verified bit-exact against
the reference `storage_to_mxfp6(...,"revised")` in `kernel/transcode_lut.py` /
`kernel/nvfp4_codec.py`.

---

## 6. Full materials index

| File | Role |
|---|---|
| `psg_preshuffle_gemm.py` | **The kernel** (fp4 / w4a6 / fp6 / w6a6_tc) |
| `kernel/bench_sweep.py` | TFLOPS harness (per `in_dtype`, tile sweep) — driven by the `.sh` scripts |
| `kernel/verify_w4a6.py` | Constant-value correctness + `w6a6_tc==W4A6` cross-check |
| `kernel/bench_all.sh` | Reproduce all headline numbers (LLM + FLUX) |
| `kernel/{verify_decode,decode_stable,prof_decode,crossover}.sh` | Verify / stability / profiling / crossover |
| `kernel/opt_log.md` | Per-attempt optimization log (phase 0 → LDS-LUT → micro-opt) |
| `opt_snapshots/psg_phase{0,1,2}_*.py` | Kernel snapshots for rollback at each optimization phase |
| `kernel/transcode_lut.py`, `kernel/nvfp4_codec.py`, `kernel/nvfp4_mxfp6_gemm_tiled.py` | LUT + bit-exact codec + `build_LUT12()` |
| `kernel/W4A6_MFMA_RECIPE.md` | Verified f8f6f4 fragment layout + per-lane E8M0 scale semantics |
| `kernel/remote_setup.md` | Container / host setup |
| `PROGRESS.md` | Full narrative: motivation, phases, profiling, LLM + FLUX conclusions |
| `MXFP6_Emu.pptx` / `build_pptx.py` | Slide deck + its generator (accuracy + performance) |

---

## 7. Headline numbers (MI355, best tile, TFLOPS — 2026-07-13 unified re-run)

> **Newer table available.** The current per-shape numbers for the FlyDSL arms, re-measured on a
> quiet node with the reference re-timed after every row (2177–2189, a 0.55% band) and every arm
> selecting over the same config space including the 512-thread configs, are in
> `kernel/s19_results/s19_dispatch_table.txt` and `kernel/opt_trace_s19.md` §5. That table adds the
> `fp6` column and the `tc/fp6` ratio, which is the comparison that isolates kernel quality from
> operand format. The 2026-07-13 table below remains the reference for the **HIP** arm, which s19
> did not re-measure.

Columns: **W4A4**(fp4) · **W4A6** · **W6A6**(fp6) · **tc-Fly**(w4a6_tc FlyDSL) ·
**tc-HIP**(w4a6_tc HIP v31). HIP now covers **all** regimes (small-BM + split-K):
`M%1024==0`→BM=1024 prefill, `M==128`→BM=128 decode, other `M%256==0`→BM=256.
**Bold** = beats W6A6; the memory-bound rows also beat W4A6. Full log:
`kernel/bench_full_out.txt`.

LLM (Qwen):

| Shape (M×N×K) | W4A4 | W4A6 | W6A6 | tc-Fly | tc-HIP |
|---|---:|---:|---:|---:|---:|
| square 4096×4096×8192 (compute) | 3252 | 2845 | 1877 | 1741 | **2004** |
| gate_up 4096×19456×2560 (compute) | 2995 | 2230 | 1759 | 1553 | 1622 |
| DOWN decode 128×2560×9728 (**mem**) | 345 | 239 | 117 | 308 | **328** |
| GATE decode 128×19456×2560 (**mem**) | 1596 | 1016 | 677 | **1525** | 558 |
| QKV decode 128×4096×4096 (**mem**) | 476 | 344 | 202 | **429** | **281** |

Memory-bound (decode) is where transcode wins: tc-Fly beats W6A6 by **1.5–2.6×** and
W4A6 by **+25–50%**. tc-HIP decode was retuned to **BN=32** (more CTAs → hides HBM
latency; +14–26% vs BN=64) and now **beats tc-Fly on DOWN (328 vs 308)** while beating
W6A6/W4A6 on all small-N decode shapes. On huge-N GATE/QKV and M256 tc-Fly still leads
~2.5×.

**Two more decode rewrites were tried to close the huge-N gap and both lost** (see
`kernel/opt_log.md` §7b): (1) a **barrier-free register-B with WM=1** (`hip_regb_decode.sh`,
bit-exact, fixes the old REGB's WM-fold redundant B loads) is still slower than the
cooperative LDS pack (GATE 468 < 558) — coalesced `uint4` loads + shared transcode beat
register-B even without the barrier; (2) a **`global_load_lds` async multi-stage pipeline**
(`hip_glds.sh`) was both incorrect (the builtin writes LDS wave-contiguously via `m0`, not to
the per-lane pointer) and slower, and was reverted. **Root cause of the residual:** with M=128
fixed, GATE has only ~608 CTAs (≈2.4/CU) yet its output C (10 MB) is 40% of B (25 MB), so
split-K multiplies output traffic and loses; each N-tile CTA also re-reads the full A
(~245 KB fp6) from L2 (~149 MB of A reads ≫ 25 MB B). Beating FlyDSL there needs a persistent +
async production pipeline, not tile tuning.

### Delivered: best-of-both dispatch (≥ FlyDSL on every shape)

Both transcode kernels are **bit-identical**, so the shipped path dispatches per-shape to the
faster one. `bench_full.sh` prints a `tc_BEST(win)` column = `max(tc_HIP, tc_Fly)`. **tc_BEST is
≥ FlyDSL on all 12 shapes**, and — after adding a **best-of-GROUP** prefill dispatch
(`max(GROUP=2, GROUP=4)`; a second 1024×64 binary with a different CTA→tile rasterization) —
**beats W6A6 on 8/12** (square, DOWN, GATE, QKV, FC1, **FC2**, LIN2, M256). FC2 was the flip
(1832 vs 1818). The 4 that stay under W6A6 (gate_up, imgQKV, imgPROJ, LIN1) are **all
compute-bound** — a *mathematical* cap, not a tuning miss (next section).

### Pack-tax decomposition (definitive evidence, `kernel/packtax_probe.sh`)

The SAME 1024×64 w8×1 tile built in three modes isolates exactly where the compute deficit
lives: **NOPACK** (skip transcode, constant-fed sB = my kernel's pure fp6-LDS ceiling),
**DUMMYPACK** (stream+write B to LDS every tile, skip the LUT), **full** (production).

| shape | W6A6 | full | NOPACK | DUMMY | tax% |
|---|---:|---:|---:|---:|---:|
| gate_up | 1763 | 1648 | **2075** | 1723 | 20.6 |
| imgQKV | 1632 | 1460 | **1837** | 1522 | 20.5 |
| imgPROJ | 1524 | 1421 | **1725** | 1469 | 17.7 |
| FC2 | 1818 | 1794 | **2304** | 1821 | 22.1 |
| LIN1 | 1747 | 1518 | **1954** | 1556 | 22.3 |

- **NOPACK ≫ W6A6 everywhere** (FC2 2304 vs 1818, +27%): my MFMA loop is already *faster*
  than FlyDSL-W6A6. The entire deficit is the pack tax, not the GEMM.
- The tax is **NOPACK→DUMMY = 17–22%** (per-tile B stream + LDS write + barrier) and
  **DUMMY→full = only 2–4%** (the LUT). The dominant cost is the fp6 **B→LDS stream**, which
  W6A6 must pay too (my DUMMY 1723 ≈ W6A6 1763). Even a *free* LUT would not flip the 4
  losers — the wall is the fundamental operand stream, confirming the ceiling below.

Geometry is exhausted: wide-BN collapses (`compute_widebn_sweep.sh`: 1024×128 → 259, a
VGPR/occupancy cliff; BN=256 exceeds 64 KB LDS), and barrier-free REGB is slower on compute
(`regb_compute_probe.sh`: WN=1 → 8× redundant B HBM + 8× LUT).

### Counter-driven diagnosis (rocprofv3, gfx950) — corrects the earlier "barrier tax" claim

An earlier version of this section attributed the compute deficit to "the per-tile barrier
(22–28%), an occupancy artifact," inferred from wall-clock `PROBE_BAR` deltas. **Hardware
counters overturn that attribution.** `kernel/prof_run.sh` + `prof2.sh` on the square prefill:

| metric | value | reading |
|---|---:|---|
| VGPRs / Occupancy | 255 / 2 waves/SIMD | 1 CTA/CU (VGPR-limited) |
| `MfmaUtil` | ~20% | MFMA pipe idle ~80% |
| `OccupancyPercent` | ~20% | **MfmaUtil tracks occupancy exactly** |
| `SALUBusy` | **4.7%** | the `s_barrier` instruction is **not** the cost |
| `SQ_INSTS_VALU : SQ_INSTS_MFMA` | **3.6 : 1** | transcode is VALU-heavy |
| `SQ_VALU_MFMA_COEXEC / BUSY` | ~6% | VALU and MFMA run **serially** |
| `SQ_LDS_BANK_CONFLICT / LDS_LOAD` | ~7.9 | real 8-way B-read conflict, but off critical path |

Five new levers, each **bit-exact** (`maxabs=0`) where run, each **measured** — all flat or worse:

| lever (probe) | Δ perf | rules out |
|---|---:|---|
| **XPOSE** conflict-free transposed B LDS (`xpose_probe.sh`) | −1..+1.3% | LDS bank conflict is off the critical path |
| **APIPE** software-prefetch A one `mi` ahead (`apipe_probe.sh`) | −0.3..+0.9% | A global-load latency |
| **ILVPACK** pack issued inside the MFMA loop (`ilvpack_probe.sh`) | −0.6..+0.9% | in-order issue won't co-execute here |
| **occupancy sweep** BM/WM for ≥2 CTA/CU (`occ_sweep.sh`) | worse; MfmaUtil ↓ | 256×4: occ 23.7% but util 15.1%, 1513 vs 1996 |
| **DUMMYPACK** drop ALL transcode VALU (`tax_decide.sh`) | **+2..+4% only** | transcode arithmetic; DUMMY imgQKV 1514 still < W6A6 1673 |

The occupancy sweep is decisive: **more occupancy *lowers* MfmaUtil and perf** — the 1024×64
w8×1 tile is already at the GEMM low-occupancy sweet spot (the HIP performance guide notes
MFMA GEMMs run best at low occupancy). And `DUMMYPACK` proves the transcode VALU is only ~4%
of the tax; removing it entirely still loses to W6A6 on huge-N.

**Corrected root cause:** MFMA utilization is pinned at ~20% by the **synchronous pipeline
structure** — `prefetch B → LDS store → __syncthreads → LDS read → MFMA`, one full-CTA
barrier per K-tile at 1 CTA/CU. It is **not** transcode VALU (~4%), **not** bank conflicts,
**not** A latency, **not** occupancy (more waves is worse), **not** the barrier instruction
(SALU 4.7%). The `PROBE_BAR` wall-clock drop is the *stall-at-barrier idle* of the synchronous
pipeline (there is no second CTA to overlap), which more occupancy provably does **not** fix.

The only candidate lever was a **barrier-free async-LDS pipeline** (`buffer/global_load_lds`
HBM→LDS + `s_waitcnt` multi-buffer). A **cheap de-risk** (`kernel/nobar_probe.sh`, `#ifdef
NOBAR` drops the per-tile `__syncthreads` — output incorrect, timing only) measured its ceiling:

| shape | baseline | NOBAR (barrier-free) | Δ |
|---|---:|---:|---:|
| square | 1992 | 2116 | +6% |
| imgQKV | 1457 | 1619 | +11% |
| FC2 | 1766 | 2018 | +14% |

NOBAR MfmaUtil ≈ 23% (vs ~20%). **The barrier-free ceiling is only +6–14%**, and a *correct*
async kernel **can't reach it**: B is shared across all 8 waves, so cross-wave LDS visibility
needs a CTA `__syncthreads` (`s_waitcnt` is per-wave); async can overlap only the HBM→LDS load
(~4–5%, the `+PF` term). So the async rewrite is **shelved** (low, mostly-unreachable ROI vs a
multi-day rewrite that risks the bit-exact kernel). The HIP kernel is at its **architectural
optimum** for the shared-B cooperative-pack design; the residual gap to W6A6 on compute-bound
huge-N is the mathematical fp6-operand-delivery cap, not a tuning miss. The prior tile-geometry
levers (KTILE2 −50%, BN=32 −22..−39%, WN=2 −73%, wide-BN −87%, COMPACTB −1%, REGB slower) remain
rejected, consistent with this diagnosis (all trade reuse/occupancy without lifting MFMA duty).

### Why no transcode kernel can beat W6A6 on compute-bound shapes

The NVFP4 weight scale is **E4M3 (has a mantissa)**; the MFMA's hardware per-block scale is
**E8M0 (pure exponent)**. So the scale mantissa *cannot* be applied by the matrix core — it
must be **merged into the B data**, which forces B up to **fp6** for the dot product. Therefore
transcode necessarily feeds the MFMA an **fp6 × fp6** product — the **same operand delivery as
W6A6** — *plus* the fp4→fp6 pack. Since `mfma_peak.hip` proves fp4/fp6 issue at the same rate,
large-M GEMMs are **operand-delivery (LDS) bound**, so on compute-bound shapes **tc's ceiling is
exactly W6A6**: it can only win when the HIP kernel's tiling out-classes FlyDSL's fp6 kernel by
more than the pack tax (square/FC1/LIN2 do; the other 5 don't). W4A6 sits above only because it
keeps B at **fp4 (half the operand bytes)** — unreachable by *any* transcode. Attempts to dodge
the pack tax on compute (register-B, `global_load_lds`, raw-fp4-in-LDS) were all slower or
incorrect (`kernel/opt_log.md` §7–7c). **Net: on compute-bound, W4A6 is the ceiling and W6A6 is
the transcode ceiling; on memory-bound, transcode wins big (fp4 B = half the HBM traffic).**

FLUX.1-schnell @1024px — the image stream (M≥4096) is compute-bound (**W4A6**), but the
**256-token text stream is memory-bound → transcode wins**:

| Shape (M×N×K) | W4A4 | W4A6 | W6A6 | tc-Fly | tc-HIP |
|---|---:|---:|---:|---:|---:|
| dual imgQKV 4096×9216×3072 | 2838 | 2150 | 1673 | 1497 | 1469 |
| dual imgPROJ 4096×3072×3072 | 2681 | 1926 | 1548 | 1344 | 1436 |
| dual FC1 4096×12288×3072 | 2999 | 2295 | 1758 | 1701 | **1817** |
| dual FC2 4096×3072×12288 | 3037 | 2214 | 1826 | 1546 | **1832** |
| single L1 4352×21504×3072 | 3150 | 2338 | 1746 | 1692 | 1606 |
| single L2 4352×3072×15360 | 3124 | 2227 | 1698 | 1201 | **1835** |
| text fc1 256×12288×3072 (**mem**) | 2044 | 1270 | 865 | **1786** | 701 |

(tc-HIP here = best-of-GROUP; **FC2 now beats W6A6** — the best-of-GROUP flip. Numbers are
the best clean reading per shape across runs, since the shared card throttles under
contention, which only depresses figures, so the max is the uncontended truth.)

**Partial-M win (M=4352):** LIN1/LIN2 aren't divisible by 1024, so they used to fall to the
BM=256 *decode* bin (LIN2 was 1407). The kernel now supports **partial-M tiles** (ceil-grid +
zero-padded A frags, output masked), so the big BM=1024 tile serves M=4352: **LIN2 jumps to
1772, beating W6A6 (1690) and tc-Fly (1205)**. LIN1 (huge-N, like gate_up) rises to 1604 but
stays under W6A6 — same compute-bound cap as gate_up.

> **M256 correction (2026-07-13):** the FLUX text stream is **memory-bound**, not
> compute-bound. With thin-N tiles tc-Fly hits **1786** (fp4 only 38% of peak here),
> beating W4A6 by +40% and W6A6 by +109%. The earlier "623 / compute-bound" figure
> used compute (CB) tiles and undersold tc; superseded.

---

## Why W6A6 < W4A4 on "compute-bound" shapes (it's operand delivery, not the MFMA)

A natural objection: if the large-M shapes are compute-bound and `f8f6f4` runs fp4/fp6 at
the same rate, W4A4/W4A6/W6A6 should tie. They don't (square: 3252 / 2845 / 1877). The
resolution — measured, not argued:

**1. The matrix core really is format-agnostic.** `kernel/hip/mfma_peak.hip` issues
back-to-back `v_mfma_scale_f32_16x16x128_f8f6f4` on **register-resident** operands (no LDS,
no HBM), 4 independent accumulator chains, 2048×4 waves:

```
fp4×fp4  = 9909 TFLOP/s
fp6×fp4  = 9899 TFLOP/s
fp6×fp6  = 9882 TFLOP/s     # fp4 == fp6, identical issue rate
fp8×fp8  = 4962 TFLOP/s     # exactly half — fp8 costs 2× cycles
```

Reproduce: `bash kernel/mfma_peak.sh` (inside the container).

**2. So the GEMMs are NOT MFMA-bound.** W4A4 at 3252 is only **33%** of the 9909 ceiling;
W6A6 at 1877 is **19%**. If they were MFMA-bound they'd all sit near 9900. They don't, so
the limiter is elsewhere: **operand delivery** — LDS-read bandwidth and VGPR footprint that
feed the matrix core each k-step.

**3. fp6 operands are 2× the on-chip bytes of fp4** (`i32x8` = 32 B/lane vs `i32x4` = 16 B).
The clincher is in the data itself: **W4A6 (fp6 A + *fp4* B) = 2845 ≫ W6A6 (fp6 A + *fp6* B)
= 1877.** Both do the *identical* fp6-containing MFMA; the **only** difference is B's LDS
width (16 B vs 32 B). A pure-MFMA limiter cannot explain a 1.5× gap here — an operand-
bandwidth limiter explains it exactly.

**Consequence for transcode:** tc expands B from fp4→fp6 *in LDS*, so on compute-bound shapes
it inherits W6A6's 2× B-read cost and tracks W6A6 (not W4A6). That is the structural
"transcode tax" on large M — and precisely why transcode instead **wins** on memory-bound M,
where HBM (fp4 B, same bytes as W4A4) dominates and LDS bandwidth is not the bottleneck.
