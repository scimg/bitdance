#!/usr/bin/env bash
# Phase 1: make native HIP W6A6 beat FlyDSL W6A6 on EVERY shape.
# Shared card is noisy (contention only DEPRESSES) -> best-of-N max = uncontended truth.
# Only gate_up/imgPROJ lose in the clean baseline; BN=128 halves A L2-refetch (A is
# re-read per N-slice, not LDS-staged) and stays at 128 acc-VGPR when BM<=512.
set -u
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd /work
CARD=${CARD:-5}; SRC=kernel/w4a6_tc_gemm_v31.hip; INC=kernel; G=2
REPS=${REPS:-10}; ITER=${ITER:-50}
TILES=( "1024 64 8 1" "512 128 8 1" "512 128 4 2" "256 128 4 1" "512 64 8 1" )
declare -A BIN
for t in "${TILES[@]}"; do set -- $t; b=/tmp/w6_${1}_${2}_${3}_${4}
  hipcc -O3 --offload-arch=gfx950 -I$INC -DBNATIVEFP6 -DBM=$1 -DBN=$2 -DWM=$3 -DWN=$4 -DGROUP=$G $SRC -o $b 2>/tmp/w6.log \
    || { echo "BUILDFAIL $t"; head -20 /tmp/w6.log; continue; }
  v=$(HIP_VISIBLE_DEVICES=$CARD $b 256 256 256 2>/dev/null | grep -oE 'maxabs=[0-9.eg+-]+ nbad=[0-9]+')
  echo "built $t  ($v)"; BIN["$t"]=$b
done
run(){ HIP_VISIBLE_DEVICES=$CARD $1 $2 $3 $4 bench $ITER 1 2>/dev/null | grep -oE '[0-9.]+ TFLOP' | grep -oE '[0-9.]+'; }
best(){ local z=0 v; for r in $(seq 1 $REPS); do v=$(run "$@"); awk -v a="$v" -v c="$z" 'BEGIN{exit !(a>c)}'&&z=$v; done; echo "$z"; }
echo "=== native W6A6 best tile per shape vs FlyDSL W6A6 (best-of-$REPS) ==="
printf "%-8s %8s  %-14s %8s  %s\n" "shape" "hipBest" "tile" "flydsl" "win?"
for shape in "square 4096 4096 8192 1890" "gate_up 4096 19456 2560 1765" "imgQKV 4096 9216 3072 1662" "imgPROJ 4096 3072 3072 1560" "FC2 4096 3072 12288 1828" "LIN1 4352 21504 3072 1747" "LIN2 4352 3072 15360 1688"; do
  set -- $shape; n=$1; M=$2; N=$3; K=$4; R=$5
  bz=0; bt=""; line=""
  for t in "${TILES[@]}"; do [ -n "${BIN[$t]:-}" ] || continue
    v=$(best "${BIN[$t]}" $M $N $K); line="$line ${t// /x}=$v"
    awk -v a="$v" -v c="$bz" 'BEGIN{exit !(a>c)}' && { bz=$v; bt="$t"; }
  done
  w=$(awk -v a="$bz" -v b="$R" 'BEGIN{print (a>b)?"WIN":"loss"}')
  printf "%-8s %8s  %-14s %8s  %s | %s\n" "$n" "$bz" "$bt" "$R" "$w" "$line"
done
echo "=== END ==="
