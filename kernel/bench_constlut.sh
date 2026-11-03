#!/usr/bin/env bash
# Compare default LDS-resident LUT against constant-memory LUT in v31 pack path.
set -u
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd /work
CARD=${CARD:-5}
SRC=kernel/w4a6_tc_gemm_v31.hip
INC=kernel
G=${G:-2}

build() {
  local name=$1 defs=$2 bm=$3 bn=$4 wm=$5 wn=$6
  hipcc -O3 --offload-arch=gfx950 -I$INC $defs -DBM=$bm -DBN=$bn -DWM=$wm -DWN=$wn -DGROUP=$G \
    $SRC -o /tmp/$name 2>/tmp/${name}.log || { echo "BUILDFAIL $name"; head -20 /tmp/${name}.log; return 1; }
}

build def_pre "" 1024 64 8 1
build con_pre "-DCONSTLUT" 1024 64 8 1
build def_dec "" 128 32 4 1
build con_dec "-DCONSTLUT" 128 32 4 1

echo "== correctness CONSTLUT =="
HIP_VISIBLE_DEVICES=$CARD /tmp/con_dec 128 256 256 2>&1 | tail -4

best() {
  local bin=$1 M=$2 N=$3 K=$4 it=$5; shift 5
  local best=0 bestS=0 v
  for S in "$@"; do
    [ $((K % (128*S))) -ne 0 ] && continue
    v=$(HIP_VISIBLE_DEVICES=$CARD $bin $M $N $K bench $it $S 2>/dev/null | grep -oE '[0-9.]+ TFLOP' | grep -oE '[0-9.]+')
    if [ -n "$v" ] && awk -v a="$v" -v b="$best" 'BEGIN{exit !(a>b)}'; then best=$v; bestS=$S; fi
  done
  echo "$best@$bestS"
}

echo "shape default constlut"
echo "square $(best /tmp/def_pre 4096 4096 8192 80 1) $(best /tmp/con_pre 4096 4096 8192 80 1)"
echo "gate_up $(best /tmp/def_pre 4096 19456 2560 80 1) $(best /tmp/con_pre 4096 19456 2560 80 1)"
echo "DOWN_decode $(best /tmp/def_dec 128 2560 9728 180 1 2 4 8 16) $(best /tmp/con_dec 128 2560 9728 180 1 2 4 8 16)"
echo "QKV_decode $(best /tmp/def_dec 128 4096 4096 180 1 2 4 8 16) $(best /tmp/con_dec 128 4096 4096 180 1 2 4 8 16)"
