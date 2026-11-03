#!/usr/bin/env bash
set -u
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd /work
CARD=${CARD:-5}
SRC=kernel/w4a6_tc_gemm_v31.hip
INC=kernel

build(){ local name=$1 defs=$2 bm=$3 bn=$4 wm=$5 wn=$6 grp=$7
  hipcc -O3 --offload-arch=gfx950 -I$INC $defs -DBM=$bm -DBN=$bn -DWM=$wm -DWN=$wn -DGROUP=$grp \
    $SRC -o /tmp/$name 2>/tmp/${name}.log || { echo BUILDFAIL $name; head -8 /tmp/${name}.log; return 1; }
}
runS(){ local bin=$1 M=$2 N=$3 K=$4 it=${5:-120}; shift 5; local best=0 bestS=0 v
  for S in "$@"; do [ $((K%(128*S))) -ne 0 ] && continue
    v=$(HIP_VISIBLE_DEVICES=$CARD $bin $M $N $K bench $it $S 2>/dev/null | grep -oE '[0-9.]+ TFLOP' | grep -oE '[0-9.]+')
    if [ -n "$v" ] && awk -v a="$v" -v b="$best" 'BEGIN{exit !(a>b)}'; then best=$v; bestS=$S; fi
  done
  echo "$best@$bestS"
}

# A) BM1024/BN64 wave geometry variants not covered deeply before.
GEOS=("1024 64 8 1" "1024 64 4 1" "1024 64 2 1" "1024 64 16 1" "1024 32 8 1" "1024 128 8 1" "1024 128 8 2")
declare -A GBIN
for geo in "${GEOS[@]}"; do set -- $geo; name=g_${1}_${2}_${3}_${4}; build $name "" $1 $2 $3 $4 2 && GBIN["$geo"]=/tmp/$name || GBIN["$geo"]=NONE; done
printf "%-24s" "geo_shape"; for geo in "${GEOS[@]}"; do printf "%15s" "$(echo $geo|tr ' ' '_')"; done; echo
for shape in "gate_up 4096 19456 2560 1765" "imgQKV 4096 9216 3072 1662" "imgPROJ 4096 3072 3072 1560" "FC2 4096 3072 12288 1828" "LIN1 4352 21504 3072 1747"; do
  set -- $shape; nm=$1; M=$2; N=$3; K=$4; ref=$5; printf "%-24s" "$nm/$ref"
  for geo in "${GEOS[@]}"; do set -- $geo; BN=$2; b=${GBIN["$geo"]}; if [ "$b" = NONE ] || [ $((N%BN)) -ne 0 ]; then printf "%15s" -; else printf "%15s" "$(runS $b $M $N $K 100 1 2)"; fi; done; echo
done

# B) GROUP sweep for the current best geometry after packed-scale patch.
printf "\n%-24s" "group_shape"; for G in 1 2 4 8 16; do printf "%15s" "G$G"; done; echo
for G in 1 2 4 8 16; do build grp_$G "" 1024 64 8 1 $G >/dev/null || true; done
for shape in "gate_up 4096 19456 2560 1765" "imgQKV 4096 9216 3072 1662" "imgPROJ 4096 3072 3072 1560" "FC2 4096 3072 12288 1828" "LIN1 4352 21504 3072 1747"; do
  set -- $shape; nm=$1; M=$2; N=$3; K=$4; ref=$5; printf "%-24s" "$nm/$ref"
  for G in 1 2 4 8 16; do printf "%15s" "$(runS /tmp/grp_$G $M $N $K 100 1)"; done; echo
done

# C) split-K for close FC2/LIN1 cases with best geometry; include S>1 explicitly.
printf "\n%-24s" "split_shape"; for S in 1 2 4 8; do printf "%15s" "S$S"; done; echo
for shape in "FC2 4096 3072 12288 1828" "LIN1 4352 21504 3072 1747" "gate_up 4096 19456 2560 1765"; do
  set -- $shape; nm=$1; M=$2; N=$3; K=$4; ref=$5; printf "%-24s" "$nm/$ref"
  for S in 1 2 4 8; do printf "%15s" "$(runS /tmp/g_1024_64_8_1 $M $N $K 80 $S)"; done; echo
done
