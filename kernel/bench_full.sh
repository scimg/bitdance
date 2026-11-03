#!/usr/bin/env bash
# UNIFIED benchmark: all 5 kernels on Qwen (LLM) + FLUX.1-schnell shapes.
#   FlyDSL psg:  fp4 (=W4A4) | w4a6 | fp6 (=W6A6) | w6a6_tc (=reference only)
#   HIP v31:     w4a6_tc HIP  (M-major CTA rasterization, GROUP=2)
# Dispatch column is HIP-only; FlyDSL transcode is measured but not selected.
# HIP now covers ALL regimes (N%64==0 && K%128==0):
#   M%1024==0 -> BM=1024 prefill; M==128 -> BM=128 decode + split-K sweep;
#   other M%256==0 -> BM=256 + split-K. Only non-64/128-divisible shapes are n/a.
# TFLOPS = 2*M*N*K/t for BOTH harnesses -> directly comparable.
set +e
rm -rf /root/psg; mkdir -p /root/psg
cp /aiter/aiter/ops/flydsl/kernels/*.py /root/psg/ 2>/dev/null
: > /root/psg/__init__.py
cp /work/psg_preshuffle_gemm.py /root/psg/preshuffle_gemm.py
cd /work
. /work/kernel/s14_lib.sh
# s14: the node is shared. Pick the quietest card rather than assuming card 5 is free.
CARD=${CARD:-$(pick_card)}; WARM=15; ITER=80
HREPS=${HREPS:-3}   # repeats per (binary,S) inside hbest; the S-sweep already adds spread
# s15: the FlyDSL side needs the same contention-robust best-of as HIP's best_of, else the
# table is biased on a shared node. --reps repeats the timed loop inside the process, so it
# costs no extra startup or JIT compile (unlike HREPS, which re-launches a binary).
FLYREPS=${FLYREPS:-3}
# s15: kill-resilience. The full sweep is long enough that a container kill (exit 137, seen
# repeatedly) used to cost the whole run. Each row is appended to RESULTS as it completes and
# a rerun skips rows already recorded, so the host can simply re-invoke until the table is
# full. Delete RESULTS to force a clean sweep.
RESULTS=${RESULTS:-/work/kernel/bench_full_results.tsv}
touch "$RESULTS"
echo "using card $CARD (HREPS=$HREPS) results=$RESULTS ($(grep -c '^ROW' "$RESULTS") rows cached)"
# Rebuilding 17 binaries takes minutes and is pure waste on a resumed run, so skip any whose
# output is already newer than both kernel sources. Implemented as a function shadowing the
# compiler, which leaves all 17 call sites below untouched. FORCE_BUILD=1 defeats it.
newest_src=$(ls -t /work/kernel/w4a6_tc_gemm_v42.hip /work/kernel/w4a6_tc_gemm_v31.hip 2>/dev/null | head -1)
hipcc() {
  local a out="" prev=""
  for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
  if [ "${FORCE_BUILD:-0}" != "1" ] && [ -n "$out" ] && [ -x "$out" ] \
     && [ -n "$newest_src" ] && [ "$out" -nt "$newest_src" ]; then
    return 0
  fi
  command hipcc "$@"
}
DTS="fp4 w4a6 fp6 w6a6_tc"
# s17: the compute list only ever offered tile_m=128, so every compute row in every table this
# project has produced was measured with a tile the kernel does not prefer. tile_m=256 with
# tile_n=128 -- the transposed aspect ratio -- is worth +4.4% to +29.8% and flips SQUARE, LIN2
# and imgPROJ from HIP to FlyDSL. HIP's own compute win came from BM=512, which should have been
# the hint. tile_k must stay 128 (w4a6/w6a6_tc reject anything else).
CB="128,256,128;128,128,128;256,128,128;256,256,128"
# M=256 rows can use tile_m=256 as well; the M=128 rows cannot (a 256-row tile would run half
# empty), so they keep their own list.
# s17c: every decode tile this project ever measured used tile_m=128, which at M=128 is exactly ONE
# M block -- so the grid was N/tile_n workgroups and DOWN decode ran 20 of 256 CUs. tile_m=64 splits
# M in two and doubles the grid; it is legal (m_repeat=4, packed 4//2=2, A-load 16 B/thread) whereas
# tile_m=32 is not (8 B/thread fails the 16-byte A load). Measured warm: QKV decode +29.5%, DOWN
# +14.1%, M256 FC1 +9.9% (which flips it to FlyDSL, 866 vs HIP 851). GATE decode is the exception at
# -5.7%: it already had 152 workgroups, and doubling overshoots to 119% of the CUs, so the second
# pass runs 19% full. Bit-identical to the incumbent over 36 configurations (s17_equiv64.sh); the
# list keeps tile_m=128 so dispatch takes the max per shape.
MEM256="128,128,128;128,256,128;128,512,128;256,128,128;256,256,128;64,128,128"
DEC="128,128,128;128,256,128;64,128,128"
# memory-bound rows (M<=256). tile_n=64 is REMOVED: on the fp4-family B path (fp4/w4a6/w6a6_tc)
# it leaves C entirely unwritten, so it timed as the fastest tile and won every decode row --
# worth a fabricated +20% to +224%. It was originally adopted here precisely because it "gave
# 1789 instead of 607", i.e. the defect sold itself as the optimization. fp6 is unaffected, which
# is why the reference columns never looked wrong. bench_sweep.py now refuses any tile that
# writes no output, so this cannot silently return.
MEM="128,128,128;128,256,128;128,512,128;64,128,128"

# --- build HIP v31 binaries once: prefill (BM=1024), decode (BM=128), M256 (BM=256) ---
# GROUP=2 is the kernel default best (seq_group_sweep). Decode/M256 use small-BM LDS
# path + split-K; the register-B (REGB) path was bit-exact but slower, so LDS pack stays.
HGRP=${HGRP:-2}
HPRE=/tmp/hip_pre;  HDEC=/tmp/hip_dec;  HM256=/tmp/hip_m256; HC256=/tmp/hip_c256
# best-of-GROUP: a second prefill binary with GROUP=4. Same 1024x64 tile, different CTA
# rasterization -> better A L2 locality on some compute shapes (flips FC2 over W6A6:
# 1832 vs 1818). Prefill dispatch takes the max of GROUP=2 and GROUP=4 (no regression).
HPRE4=/tmp/hip_pre4
HAPX=/tmp/hip_apx;  HAPX2=/tmp/hip_apx2;  HAPX3=/tmp/hip_apx3;  HAPX4=/tmp/hip_apx4
HDECF=/tmp/hip_decf; HM256F=/tmp/hip_m256f   # s14: fp32-output decode twins
# s14 winners, added as extra dispatch candidates (dispatch is max(), so they cannot
# regress anything): BARAMORT+PRIOPK barrier amortization, and the HWCVT transcode.
HAPX5=/tmp/hip_apx5;  HAPX6=/tmp/hip_apx6;  HAPX7=/tmp/hip_apx7
HDEC2=/tmp/hip_dec2;  HDEC3=/tmp/hip_dec3;  HM2562=/tmp/hip_m2562
# HKT2 (KTILE2+XCMPB) measured 60-65% of XCMPB+APIPE — shelved (2x pack per barrier hurts)
# Decode/M256 use BN=32 (more CTAs -> better HBM-latency hiding at tiny M; +14-26%
# over BN=64 from hip_decode_opt.sh). Prefill = BM=1024 (now partial-M capable, so it
# also serves M=4352 via ceil-grid + A masking). HC256 = BM=256 compute tile for large
# M not divisible by 1024 where the big tile's padding waste loses (e.g. LIN1).
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=1024 -DBN=64 -DWM=8 -DWN=1 -DGROUP=$HGRP /work/kernel/w4a6_tc_gemm_v31.hip -o $HPRE  2>/tmp/hb1.log
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=1024 -DBN=64 -DWM=8 -DWN=1 -DGROUP=4     /work/kernel/w4a6_tc_gemm_v31.hip -o $HPRE4 2>/tmp/hb5.log
# v43 PIPE G4: NSTG=3 LDS ring + s_setprio MFMA priority + ADDR32+ASMPACK2+EPILOG32.
# Enabled by the 160KB LDS finding. Beats prior best on ALL 7 shapes (+2-4.5%); gate_up
# now beats W6A6 -> 6/7. Primary APX binary. (CDNA4 GEMM blog scheduling method.)
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=512  -DBN=128 -DWM=8 -DWN=1 -DGROUP=4 -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=3 /work/kernel/w4a6_tc_gemm_v42.hip -o $HAPX  2>/tmp/hb6.log
# v43 PIPE G2: GROUP=2 rasterization variant.
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=512  -DBN=128 -DWM=8 -DWN=1 -DGROUP=2 -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=3 /work/kernel/w4a6_tc_gemm_v42.hip -o $HAPX2 2>/tmp/hb6b.log
# v43 PIPE G4 NSTG=4: deeper ring, wins some large-K shapes.
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=512  -DBN=128 -DWM=8 -DWN=1 -DGROUP=4 -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=4 /work/kernel/w4a6_tc_gemm_v42.hip -o $HAPX3 2>/tmp/hb6d.log
# v45 PIPE G2 + XCD-aware swizzle (Primus-Turbo PR410/424): marginal per-XCD L2 reuse.
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=512  -DBN=128 -DWM=8 -DWN=1 -DGROUP=2 -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=3 -DXCDSW /work/kernel/w4a6_tc_gemm_v42.hip -o $HAPX4 2>/tmp/hb6e.log
# KTILE2+XCMPB shelved: measured 60-65% of XCMPB+APIPE (2x pack per barrier is worse)
# hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=512 -DBN=128 -DWM=8 -DWN=1 -DGROUP=2 -DKTILE2 -DXCMPB /work/kernel/w4a6_tc_gemm_v41.hip -o $HKT2
# v43 PIPE decode: BN=32 + s_setprio + micro-opts. +8% vs v31 BN=32 on GATE/QKV decode.
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=128  -DBN=32 -DWM=4 -DWN=1 -DGROUP=$HGRP -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=3 /work/kernel/w4a6_tc_gemm_v42.hip -o $HDEC  2>/tmp/hb2.log
# v43 PIPE M256: BN=64 exact-M fit + s_setprio.
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=256  -DBN=64 -DWM=4 -DWN=1 -DGROUP=$HGRP -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=3 /work/kernel/w4a6_tc_gemm_v42.hip -o $HM256 2>/tmp/hb3.log
# s14: fp32-output twins of the decode/M256 binaries. Making decode bf16-ONLY had cost it
# the split-K lever (DOWN 352 -> 183). Both dtypes are now offered and dispatch takes the
# max, so whichever wins per shape is used and neither can regress the other.
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=128  -DBN=32 -DWM=4 -DWN=1 -DGROUP=$HGRP -DPIPE -DXCMPB -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=3 /work/kernel/w4a6_tc_gemm_v42.hip -o $HDECF  2>/tmp/hb2f.log
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=256  -DBN=64 -DWM=4 -DWN=1 -DGROUP=$HGRP -DPIPE -DXCMPB -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=3 /work/kernel/w4a6_tc_gemm_v42.hip -o $HM256F 2>/tmp/hb3f.log
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=256  -DBN=64 -DWM=4 -DWN=1 -DGROUP=$HGRP /work/kernel/w4a6_tc_gemm_v31.hip -o $HC256 2>/tmp/hb4.log
# --- s14 additions ---------------------------------------------------------------------
# BARAMORT: barrier once per D/2 k-tiles instead of every k-tile. Safe while the barrier
# interval R <= min(P, D-P) for ring depth D and pack-ahead P; the stock ring used P=D-1,
# whose WAR slack of 1 is what pinned a barrier to every tile. PRIOPK additionally holds
# MFMA priority through the pack, which is off the critical path in a deep ring.
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=512  -DBN=128 -DWM=8 -DWN=1 -DGROUP=4 -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=8 -DBARAMORT -DPRIOPK /work/kernel/w4a6_tc_gemm_v42.hip -o $HAPX5 2>/tmp/hb7.log
# HWCVT: hardware scale-convert transcode, replacing the LUT + 12-bit bit-pack entirely.
# Different rounding from the LUT, so it is gated on rel-L2 against the exact fp32 product
# rather than on bit-compare (Phase 5: ratio 1.002, i.e. indistinguishable).
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=512  -DBN=128 -DWM=8 -DWN=1 -DGROUP=4 -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DHWCVT   -DEPILOG32 -DNSTG=8 -DBARAMORT -DPRIOPK /work/kernel/w4a6_tc_gemm_v42.hip -o $HAPX6 2>/tmp/hb8.log
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=512  -DBN=128 -DWM=8 -DWN=1 -DGROUP=4 -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DHWCVT   -DEPILOG32 -DNSTG=4 -DBARAMORT -DPRIOPK /work/kernel/w4a6_tc_gemm_v42.hip -o $HAPX7 2>/tmp/hb9.log
# decode / M256 twins of the same two levers
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=128  -DBN=32 -DWM=4 -DWN=1 -DGROUP=$HGRP -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DASMPACK2 -DEPILOG32 -DNSTG=4 -DBARAMORT -DPRIOPK /work/kernel/w4a6_tc_gemm_v42.hip -o $HDEC2 2>/tmp/hb10.log
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=128  -DBN=32 -DWM=4 -DWN=1 -DGROUP=$HGRP -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DHWCVT   -DEPILOG32 -DNSTG=4 -DBARAMORT -DPRIOPK /work/kernel/w4a6_tc_gemm_v42.hip -o $HDEC3 2>/tmp/hb11.log
hipcc -O3 --offload-arch=gfx950 -I/work/kernel -DBM=256  -DBN=64 -DWM=4 -DWN=1 -DGROUP=$HGRP -DPIPE -DXCMPB -DOUTBF16 -DADDR32 -DHWCVT   -DEPILOG32 -DNSTG=4 -DBARAMORT -DPRIOPK /work/kernel/w4a6_tc_gemm_v42.hip -o $HM2562 2>/tmp/hb12.log
{ [ -x $HPRE ] && [ -x $HPRE4 ] && [ -x $HAPX ] && [ -x $HAPX2 ] && [ -x $HAPX3 ] && [ -x $HAPX4 ] && [ -x $HDEC ] && [ -x $HM256 ] && [ -x $HDECF ] && [ -x $HM256F ] && [ -x $HC256 ] \
  && [ -x $HAPX5 ] && [ -x $HAPX6 ] && [ -x $HAPX7 ] && [ -x $HDEC2 ] && [ -x $HDEC3 ] && [ -x $HM2562 ]; } \
  && echo "HIP builds OK (pre/pre4/apx/apx2/apx3/apx4/apx5/apx6/apx7/dec/dec2/dec3/decf/m256/m2562/m256f/c256)" \
  || { echo "HIP BUILD FAIL"; head -5 /tmp/hb1.log /tmp/hb5.log /tmp/hb6.log /tmp/hb6b.log /tmp/hb6d.log /tmp/hb6e.log /tmp/hb2.log /tmp/hb2f.log /tmp/hb3.log /tmp/hb3f.log /tmp/hb4.log /tmp/hb7.log /tmp/hb8.log /tmp/hb9.log /tmp/hb10.log /tmp/hb11.log /tmp/hb12.log; }

# s15: two measurement defects fixed here. Together they understated the FlyDSL column by
# 12-25% on every memory-bound row, and since the shipped policy is max(tc_HIP, tc_Fly) that
# mis-assigned DOWN decode to HIP: measured head-to-head on one card, HIP 340.1 vs FlyDSL
# 384.1. The HIP column reproduced its recorded numbers (340 vs 342.6, 280 vs 278, 710 vs
# 720), so only the FlyDSL side was wrong.
#   1. use_async_copy was never offered, though session 13 established it as the production
#      w6a6_tc front (bit-exact, +4..52% on memory-bound). It is regime-specific (-9% on
#      some compute shapes), so offer BOTH and take the max -- exactly the treatment the
#      HIP side already gets across its binaries.
#   2. ITER=80 does not reach steady state on small kernels. The QKV row is ~30us/iteration
#      and read anywhere in 285..526 TFLOPS at 200 iterations, but 519.5-519.9 (spread 0.1%)
#      at 800. Scale iterations with shape so every sample times >=25ms of steady state.
fly_iters() { awk -v m=$1 -v n=$2 -v k=$3 -v base=$ITER 'BEGIN{
  per  = 2*m*n*k/300e12          # seconds per iteration at a pessimistic 300 TFLOP/s
  want = int(0.025/per)+1        # aim for >=25ms of timed steady state
  if (want < base) want = base
  if (want > 2000) want = 2000
  print want }'; }
fly1() { HIP_VISIBLE_DEVICES=$CARD python3 kernel/bench_sweep.py --m $2 --n $3 --k $4 \
  --in_dtype $1 --tiles "$5" --warmup $WARM --iters $6 --reps ${FR:-1} \
  --splits "${SPLITS:-1}" $7 2>&1 \
  | grep -oE "BEST $1: [0-9.]+" | grep -oE "[0-9.]+$"; }
# For w6a6_tc, the transcode path is also a candidate: hardware scale-converts vs the 12-bit
# LDS LUT (s15 measured HWCVT ahead on every canonical shape, +1..+3% decode / +8..+10%
# compute, accuracy-gated by s15_accgate.sh at rel-L2 ratio 1.0124 <= 1.05). Offering both
# and taking the max is the same treatment the HIP side gives its own HWCVT binary.
fly() { local it=$(fly_iters $2 $3 $4) a b c d
  # Only the w6a6_tc column competes with HIP for dispatch, so only it needs HIP's repeat count.
  # fp4/w4a6/fp6 are reference columns: one sample each still reveals contention (they sag
  # together), at a fraction of the runtime.
  local FR=$FLYREPS; [ "$1" = "w6a6_tc" ] || FR=${FLYREPS_REF:-1}
  # s17c: split_k is swept only on the column that competes for dispatch. The reference columns
  # exist to show contention (they sag together), and multiplying every one of them by the split
  # list would multiply the JIT count for information the dispatch never reads.
  local SPLITS=1; [ "$1" = "w6a6_tc" ] && SPLITS="${ROWSPLITS:-1}"
  a=$(fly1 "$1" $2 $3 $4 "$5" "$it" "")
  b=$(fly1 "$1" $2 $3 $4 "$5" "$it" "--async_copy")
  if [ "$1" = "w6a6_tc" ]; then
    c=$(fly1 "$1" $2 $3 $4 "$5" "$it" "--hw_transcode on")
    d=$(fly1 "$1" $2 $3 $4 "$5" "$it" "--async_copy --hw_transcode on")
  fi
  maxof "${a:-0}" "${b:-0}" "${c:-0}" "${d:-0}"; }
# best TFLOP/s of a HIP binary over split-K choices (S divides K/128)
hbest() { local bin=$1 M=$2 N=$3 K=$4; shift 4; local best=0 v
  for S in "$@"; do [ $((K % (128*S))) -ne 0 ] && continue
    # s14: best_of repeats each config and takes the max (contention-robust) and drops any
    # run vetoed by the split-K correctness gate, so a wrong config can never post a time.
    v=$(best_of $bin $M $N $K $S $HREPS)
    [ -n "$v" ] && awk -v a="$v" -v b="$best" 'BEGIN{exit !(a>b)}' && best=$v
  done; echo "$best"; }
# s14: per-binary correctness gate. Each binary runs one small-shape gold verify before
# it may be used; a binary that fails is excluded from dispatch (prints its name).
# Retried 3x for the same reason as the split-K gate: on a contended node a run can die
# spuriously, and c256 was observed "failing" here while passing all 7 shapes on a quiet
# card. A genuinely broken binary fails every attempt.
gate() { local bin=$1 nm=$2 a
  [ -x "$bin" ] || { echo "GATE: $nm MISSING" >&2; return 1; }
  for a in 1 2 3; do
    HIP_VISIBLE_DEVICES=$CARD $bin 256 128 256 2>/dev/null | grep -q "RESULT: PASS" && return 0
  done
  echo "GATE: $nm FAILED VERIFY 3x" >&2; return 1; }
# s14: HWCVT rounds in hardware rather than through the LUT, so a bit-compare against a
# LUT-derived gold rejects it by construction and would say nothing about quality. Its gate
# is instead the accuracy criterion from Phase 5: rel-L2 against the EXACT fp32 dequantized
# product must be no worse than the LUT's own rel-L2 by more than 5%. Measured ratio 1.002.
gate_acc() { local bin=$1 nm=$2 r
  [ -x "$bin" ] || { echo "GATE: $nm MISSING" >&2; return 1; }
  r=$(HIP_VISIBLE_DEVICES=$CARD $bin 256 256 2048 acc 2>/dev/null | grep -aoE 'ratio=[0-9.]+' | cut -d= -f2)
  [ -z "$r" ] && { echo "GATE: $nm acc produced no ratio" >&2; return 1; }
  awk -v x="$r" 'BEGIN{exit !(x<=1.05)}' && return 0
  echo "GATE: $nm rel-L2 ratio $r > 1.05" >&2; return 1; }

# Run the gate now that the helpers exist, and only start timing on a quiet card.
# s17c: the gate costs ~60s of GPU verifies. That is cheap on a clean run and ruinous on a resumed
# one: this container is killed from outside mid-run (exit 137, the reason RESULTS resume exists),
# and attempts have died inside 80s -- i.e. entirely inside the gate, so no row could ever complete.
# The gate result is recorded when it passes, and it can only change if a binary is rebuilt, so a
# resumed attempt may trust a pass that is newer than every binary. Any rebuild invalidates it.
GATESTAMP="${RESULTS%.tsv}.gatepass"
gate_is_fresh() {
  [ -f "$GATESTAMP" ] || return 1
  local b
  for b in $HPRE $HPRE4 $HAPX $HAPX2 $HAPX3 $HAPX4 $HAPX5 $HAPX6 $HAPX7 \
           $HDEC $HDEC2 $HDEC3 $HDECF $HM256 $HM2562 $HM256F $HC256; do
    [ -e "$b" ] && [ "$b" -nt "$GATESTAMP" ] && return 1
  done
  return 0; }
if gate_is_fresh; then
  echo "GATE: skipped, 17/17 recorded $(cat "$GATESTAMP") and no binary rebuilt since"
else
  GATEFAIL=0
  for pair in "$HPRE pre" "$HPRE4 pre4" "$HAPX apx" "$HAPX2 apx2" "$HAPX3 apx3" "$HAPX4 apx4" \
              "$HAPX5 apx5" "$HDEC dec" "$HDEC2 dec2" "$HDECF decf" "$HM256 m256" \
              "$HM256F m256f" "$HC256 c256"; do
    set -- $pair; gate "$1" "$2" || GATEFAIL=$((GATEFAIL+1))
  done
  for pair in "$HAPX6 apx6" "$HAPX7 apx7" "$HDEC3 dec3" "$HM2562 m2562"; do
    set -- $pair; gate_acc "$1" "$2" || GATEFAIL=$((GATEFAIL+1))
  done
  echo "GATE: $((17-GATEFAIL))/17 binaries passed verify"
  [ "$GATEFAIL" -eq 0 ] && date -u +%Y-%m-%dT%H:%M:%SZ > "$GATESTAMP"
fi
wait_idle $CARD 900
# dispatch by M: 128->decode(BN32) | 256->M256 decode | M%1024==0->prefill |
# other large M (e.g. 4352, compute-bound)->best of {prefill partial-M, BM256 compute}.
# max over an arbitrary list of numbers; keeps the dispatch readable as candidates grow.
maxof() { printf '%s\n' "$@" | awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print m}'; }
hip() { local M=$1 N=$2 K=$3
  [ $((K%128)) -ne 0 ] && { echo "n/a"; return; }
  if   [ $M -eq 128 ]       && [ $((N%32)) -eq 0 ]; then
       maxof "$(hbest $HDEC  $M $N $K 1 2 4 8 16)" "$(hbest $HDECF $M $N $K 1 2 4 8 16)" \
             "$(hbest $HDEC2 $M $N $K 1 2 4 8 16)" "$(hbest $HDEC3 $M $N $K 1 2 4 8 16)"
  elif [ $M -eq 256 ]       && [ $((N%32)) -eq 0 ]; then
       maxof "$(hbest $HM256 $M $N $K 1 2 4)" "$(hbest $HM256F $M $N $K 1 2 4)" \
             "$(hbest $HM2562 $M $N $K 1 2 4)"
  elif [ $((M%1024)) -eq 0 ] && [ $((N%64)) -eq 0 ]; then
       local p=$(hbest $HPRE $M $N $K 1) q=$(hbest $HPRE4 $M $N $K 1)
       local a=0 a2=0 a3=0 a4=0 a5=0 a6=0 a7=0
       [ $((N%128)) -eq 0 ] && { a=$(hbest $HAPX $M $N $K 1); a2=$(hbest $HAPX2 $M $N $K 1)
         a3=$(hbest $HAPX3 $M $N $K 1); a4=$(hbest $HAPX4 $M $N $K 1); a5=$(hbest $HAPX5 $M $N $K 1)
         a6=$(hbest $HAPX6 $M $N $K 1); a7=$(hbest $HAPX7 $M $N $K 1); }
       maxof "$p" "$q" "$a" "$a2" "$a3" "$a4" "$a5" "$a6" "$a7"
  elif [ $((M%256)) -eq 0 ]  && [ $((N%64)) -eq 0 ]; then
       local p=$(hbest $HPRE $M $N $K 1) q=$(hbest $HPRE4 $M $N $K 1) c=$(hbest $HC256 $M $N $K 1)
       local a=0 a2=0 a3=0 a4=0 a5=0 a6=0 a7=0
       [ $((N%128)) -eq 0 ] && { a=$(hbest $HAPX $M $N $K 1); a2=$(hbest $HAPX2 $M $N $K 1)
         a3=$(hbest $HAPX3 $M $N $K 1); a4=$(hbest $HAPX4 $M $N $K 1); a5=$(hbest $HAPX5 $M $N $K 1)
         a6=$(hbest $HAPX6 $M $N $K 1); a7=$(hbest $HAPX7 $M $N $K 1); }
       maxof "$p" "$q" "$c" "$a" "$a2" "$a3" "$a4" "$a5" "$a6" "$a7"
  else echo "n/a"; fi; }
row() { # label M N K tiles [split_k list for the w6a6_tc column]
  local f4 wa f6 tc h best who cached
  # Split-K is a per-shape lever, like the tile: the decode rows cannot fill the GPU from their
  # M/N grid alone and gain hugely, while the rows that already fill it only pay the reduction.
  # bench_sweep rejects the illegal divisors itself, so a superset can be offered safely.
  local ROWSPLITS="${6:-1}"
  # resume: a row already in RESULTS is replayed, not re-measured
  cached=$(grep -F "$(printf 'ROW\t%s\t%d\t%d\t%d\t' "$1" $2 $3 $4)" "$RESULTS" | tail -1)
  if [ -n "$cached" ]; then printf '%s\t(cached)\n' "$cached"; return; fi
  f4=$(fly fp4 $2 $3 $4 "$5");    wa=$(fly w4a6 $2 $3 $4 "$5")
  f6=$(fly fp6 $2 $3 $4 "$5");    tc=$(fly w6a6_tc $2 $3 $4 "$5")
  h=$(hip $2 $3 $4)
  # s15: the shipped policy is max(tc_HIP, tc_Fly) per shape, so the table must apply it.
  # It previously hardcoded HIP and printed FlyDSL as an unselected reference column, which
  # hid the DOWN-decode mis-assignment: a row could show a faster FlyDSL number and still
  # report "(HIP)" as the winner.
  best="$h"; who="HIP"
  if [ -n "$tc" ] && awk -v a="$tc" -v b="${h:-0}" 'BEGIN{exit !(a>b)}'; then
    best="$tc"; who="FlyDSL"
  fi
  printf "ROW\t%s\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s(%s)\n" "$1" $2 $3 $4 \
    "${f4:-err}" "${wa:-err}" "${f6:-err}" "${tc:-err}" "$h" "${best:-$tc}" "$who" \
    | tee -a "$RESULTS"
}

# s17c: warm the card to its clock plateau before measuring anything. An idle MI355X ramps over
# roughly six invocations, not one: s17_cold.sh saw the first invocation read low in all four tile
# columns at once and climb monotonically until the last two agreed to 0.08%. Rows measured on the
# ramp are simply low -- the previous audit put the shipped SQUARE config at 2528 where it settles
# at 2565 -- and whichever rows run first absorb the whole penalty, which is worse than a uniform
# offset because it is not uniform. Warm-up is a discarded measurement, so discard it explicitly.
warm() { local prev="" v i
  echo "== warming card $CARD to plateau (stop when two consecutive samples agree within 0.5%)"
  for i in $(seq 1 12); do
    v=$(fly1 w6a6_tc 4096 4096 8192 "256,128,128" 200 "--async_copy --hw_transcode on")
    printf '   warm %2d: %s\n' "$i" "${v:-ERR}"
    if [ -n "$prev" ] && [ -n "$v" ] && awk -v a="$prev" -v b="$v" \
        'BEGIN{m=(a>b?a:b); exit !(100*(a>b?a-b:b-a)/m <= 0.5)}'; then
      echo "   plateau at sample $i"; return 0
    fi
    prev="$v"
  done
  echo "   WARNING: no plateau in 12 samples; rows below may be measured on the ramp" >&2; }
warm

echo "== HDR label M N K W4A4(fp4) W4A6 W6A6(fp6) tc_flydsl tc_HIP DISPATCH(win) =="
echo "==================== A) LLM (Qwen) ===================="
row "LLM square"        4096 4096  8192 "$CB"
row "LLM gate_up pref"  4096 19456 2560 "$CB"
row "LLM DOWN decode"   128  2560  9728 "$MEM" "1,2,19"
row "LLM GATE decode"   128  19456 2560 "$MEM" "1,2,5"
row "LLM QKV decode"    128  4096  4096 "$MEM" "1,2,4,8"
echo "==================== B) FLUX.1-schnell (1024px) ===================="
row "FLUX dbl imgQKV"   4096 9216  3072 "$CB"
row "FLUX dbl imgPROJ"  4096 3072  3072 "$CB"
row "FLUX dbl FC1"      4096 12288 3072 "$CB"
row "FLUX dbl FC2"      4096 3072  12288 "$CB"
row "FLUX sgl LIN1"     4352 21504 3072 "$CB"
row "FLUX sgl LIN2"     4352 3072  15360 "$CB"
row "FLUX txt FC1 M256" 256  12288 3072 "$MEM256" "1,2,3,4,6"
echo "=== END rc=$? ==="
