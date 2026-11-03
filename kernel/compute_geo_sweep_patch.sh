#!/usr/bin/env bash
# Post packed-scale patch geometry sweep on remaining compute-bound problem shapes.
set -u
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd /work
CARD=${CARD:-5}
SRC=kernel/w4a6_tc_gemm_v31.hip
INC=kernel
G=${G:-2}
CFGS=(
  "1024 64 8 1" "1024 128 8 2"
  "512 64 8 1" "512 128 8 2" "512 128 4 2" "512 256 4 4"
  "256 64 4 1" "256 128 4 2" "256 256 4 4"
)
declare -A BIN
for c in "${CFGS[@]}"; do
  set -- $c
  b=/tmp/cgp_${1}_${2}_${3}_${4}
  hipcc -O3 --offload-arch=gfx950 -I$INC -DBM=$1 -DBN=$2 -DWM=$3 -DWN=$4 -DGROUP=$G \
    $SRC -o $b 2>/tmp/cgp.log && BIN["$c"]=$b || { echo "BUILDFAIL $c"; head -4 /tmp/cgp.log; BIN["$c"]=NONE; }
done

run1() {
  local bin=$1 M=$2 N=$3 K=$4
  HIP_VISIBLE_DEVICES=$CARD $bin $M $N $K bench 130 1 2>/dev/null \
    | grep -oE '[0-9.]+ TFLOP' | grep -oE '[0-9.]+'
}

printf "%-34s" "shape W6A6"; for c in "${CFGS[@]}"; do printf "%14s" "$(echo $c|tr ' ' '_')"; done; echo
for shape in \
  "gate_up 4096 19456 2560 1765" \
  "imgQKV 4096 9216 3072 1662" \
  "imgPROJ 4096 3072 3072 1560" \
  "FC2 4096 3072 12288 1828" \
  "LIN1 4352 21504 3072 1747"; do
  set -- $shape; name=$1; M=$2; N=$3; K=$4; ref=$5
  printf "%-34s" "$name $ref"
  for c in "${CFGS[@]}"; do
    set -- $c; BM=$1; BN=$2; b=${BIN["$c"]}
    if [ "$b" = NONE ] || [ $((N%BN)) -ne 0 ]; then
      printf "%14s" "-"
    else
      printf "%14s" "$(run1 $b $M $N $K)"
    fi
  done
  echo
done
