#!/usr/bin/env bash
set -u
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd /work
CARD=${CARD:-5}; SRC=kernel/w4a6_tc_gemm_v31.hip; INC=kernel; G=2
build(){ local name=$1 defs=$2 bm=${3:-1024} bn=${4:-64} wm=${5:-8} wn=${6:-1} grp=${7:-2}
 hipcc -O3 --offload-arch=gfx950 -I$INC $defs -DBM=$bm -DBN=$bn -DWM=$wm -DWN=$wn -DGROUP=$grp $SRC -o /tmp/$name 2>/tmp/$name.log \
  && echo OK $name || { echo BUILDFAIL $name; head -30 /tmp/$name.log; }
}
build def ""
build asm "-DASMPACK"
build xpose "-DXPOSE"
build xpose_asm "-DXPOSE -DASMPACK"
build apx "-DAPIPE -DXPOSE" 512 128 8 1 4
build apx_asm "-DAPIPE -DXPOSE -DASMPACK" 512 128 8 1 4
run(){ local bin=/tmp/$1 M=$2 N=$3 K=$4; [ -x $bin ] || { echo -; return; }; HIP_VISIBLE_DEVICES=$CARD $bin $M $N $K bench 120 1 2>/dev/null | grep -oE '[0-9.]+ TFLOP' | grep -oE '[0-9.]+'; }
echo correctness
HIP_VISIBLE_DEVICES=$CARD /tmp/asm 128 256 256 2>&1 | tail -4 || true
echo "shape def asm xpose xpose_asm apx apx_asm W6A6"
for shape in "square 4096 4096 8192 1890" "gate_up 4096 19456 2560 1765" "imgQKV 4096 9216 3072 1662" "imgPROJ 4096 3072 3072 1560" "FC2 4096 3072 12288 1828" "LIN1 4352 21504 3072 1747" "LIN2 4352 3072 15360 1688"; do
 set -- $shape; n=$1; M=$2; N=$3; K=$4; R=$5
 echo "$n $(run def $M $N $K) $(run asm $M $N $K) $(run xpose $M $N $K) $(run xpose_asm $M $N $K) $(run apx $M $N $K) $(run apx_asm $M $N $K) $R"
done
