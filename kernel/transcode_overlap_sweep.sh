#!/usr/bin/env bash
set -u
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd /work
CARD=${CARD:-5}; SRC=kernel/w4a6_tc_gemm_v31.hip; INC=kernel; G=2
build(){ local name=$1 defs=$2
  hipcc -O3 --offload-arch=gfx950 -I$INC $defs -DBM=1024 -DBN=64 -DWM=8 -DWN=1 -DGROUP=$G $SRC -o /tmp/$name 2>/tmp/$name.log \
    && echo OK $name || { echo BUILDFAIL $name; head -8 /tmp/$name.log; }
}
build def ""
build apipe "-DAPIPE"
build ilv "-DILVPACK"
build xpose "-DXPOSE"
build kt2 "-DKTILE2"
build kt2_x "-DKTILE2 -DXPOSE"
build apipe_x "-DAPIPE -DXPOSE"
VARS=(def apipe ilv xpose kt2 kt2_x apipe_x)
run(){ local bin=/tmp/$1 M=$2 N=$3 K=$4; [ -x $bin ] || { echo -; return; }
  HIP_VISIBLE_DEVICES=$CARD $bin $M $N $K bench 100 1 2>/dev/null | grep -oE '[0-9.]+ TFLOP' | grep -oE '[0-9.]+'; }
echo correctness
HIP_VISIBLE_DEVICES=$CARD /tmp/apipe 128 256 256 2>&1 | tail -4 || true
printf "%-24s" "shape/W6"; for v in "${VARS[@]}"; do printf "%13s" "$v"; done; echo
for shape in "gate_up 4096 19456 2560 1765" "imgQKV 4096 9216 3072 1662" "imgPROJ 4096 3072 3072 1560" "FC2 4096 3072 12288 1828" "LIN1 4352 21504 3072 1747" "LIN2 4352 3072 15360 1688"; do
  set -- $shape; nm=$1; M=$2; N=$3; K=$4; ref=$5; printf "%-24s" "$nm/$ref"
  for v in "${VARS[@]}"; do printf "%13s" "$(run $v $M $N $K)"; done; echo
done
