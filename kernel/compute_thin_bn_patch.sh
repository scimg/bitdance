#!/usr/bin/env bash
# Try thin-BN compute tiles after packed-scale patch.
set -u
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd /work
CARD=${CARD:-5}
SRC=kernel/w4a6_tc_gemm_v31.hip
INC=kernel
G=${G:-2}
CFGS=(
  "1024 16 8 1" "1024 32 8 1" "1024 64 8 1"
  "512 16 8 1" "512 32 8 1" "512 64 8 1"
  "256 16 4 1" "256 32 4 1" "256 64 4 1"
)
declare -A BIN
for c in "${CFGS[@]}"; do
  set -- $c
  b=/tmp/tbn_${1}_${2}_${3}_${4}
  hipcc -O3 --offload-arch=gfx950 -I$INC -DBM=$1 -DBN=$2 -DWM=$3 -DWN=$4 -DGROUP=$G \
    $SRC -o $b 2>/tmp/tbn.log && BIN["$c"]=$b || { echo "BUILDFAIL $c"; head -4 /tmp/tbn.log; BIN["$c"]=NONE; }
done
run1(){ HIP_VISIBLE_DEVICES=$CARD $1 $2 $3 $4 bench 120 1 2>/dev/null | grep -oE '[0-9.]+ TFLOP' | grep -oE '[0-9.]+'; }
printf "%-28s" "shape W6A6"; for c in "${CFGS[@]}"; do printf "%14s" "$(echo $c|tr ' ' '_')"; done; echo
for shape in \
  "FC2 4096 3072 12288 1828" \
  "imgPROJ 4096 3072 3072 1560" \
  "imgQKV 4096 9216 3072 1662" \
  "LIN1 4352 21504 3072 1747"; do
  set -- $shape; name=$1; M=$2; N=$3; K=$4; ref=$5
  printf "%-28s" "$name $ref"
  for c in "${CFGS[@]}"; do set -- $c; BN=$2; b=${BIN["$c"]}
    if [ "$b" = NONE ] || [ $((N%BN)) -ne 0 ]; then printf "%14s" "-"; else printf "%14s" "$(run1 $b $M $N $K)"; fi
  done; echo
done
