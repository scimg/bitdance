#!/usr/bin/env bash
set -u
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd /work
CARD=${CARD:-5}; SRC=kernel/w4a6_tc_gemm_v31.hip; INC=kernel
MACROS=("-DXPOSE" "-DAPIPE" "-DAPIPE -DXPOSE" "")
GEOS=("1024 64 8 1" "512 64 8 1" "512 128 4 2" "256 64 4 1")
GRPS=(1 2 4 8 16)
declare -A BIN
for mi in ${!MACROS[@]}; do for gi in ${!GEOS[@]}; do for G in ${GRPS[@]}; do
  set -- ${GEOS[$gi]}; name=m${mi}_g${gi}_G${G}; defs="${MACROS[$mi]}"
  hipcc -O3 --offload-arch=gfx950 -I$INC $defs -DBM=$1 -DBN=$2 -DWM=$3 -DWN=$4 -DGROUP=$G $SRC -o /tmp/$name 2>/tmp/$name.log \
    && BIN[$name]=/tmp/$name || BIN[$name]=NONE
done; done; done
run(){ HIP_VISIBLE_DEVICES=$1 $2 $3 $4 $5 bench 70 1 2>/dev/null | grep -oE '[0-9.]+ TFLOP' | grep -oE '[0-9.]+'; }
echo "shape ref best variant"
for shape in "gate_up 4096 19456 2560 1765" "imgQKV 4096 9216 3072 1662" "imgPROJ 4096 3072 3072 1560" "FC2 4096 3072 12288 1828" "LIN1 4352 21504 3072 1747"; do
  set -- $shape; nm=$1; M=$2; N=$3; K=$4; ref=$5; best=0; bestname=NONE
  for mi in ${!MACROS[@]}; do for gi in ${!GEOS[@]}; do set -- ${GEOS[$gi]}; BM=$1; BN=$2
    [ $((N%BN)) -ne 0 ] && continue
    for G in ${GRPS[@]}; do name=m${mi}_g${gi}_G${G}; b=${BIN[$name]}; [ "$b" = NONE ] && continue
      v=$(run $CARD $b $M $N $K); [ -z "$v" ] && continue
      if awk -v a="$v" -v b="$best" 'BEGIN{exit !(a>b)}'; then best=$v; bestname="$name:${MACROS[$mi]}:${GEOS[$gi]}:G$G"; fi
    done
  done; done
  echo "$nm $ref $best $bestname"
done
