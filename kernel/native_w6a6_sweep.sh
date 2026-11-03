#!/usr/bin/env bash
set -u
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd /work
CARD=${CARD:-5}; SRC=kernel/w4a6_tc_gemm_v31.hip; INC=kernel; G=2
build(){ hipcc -O3 --offload-arch=gfx950 -I$INC -DBNATIVEFP6 -DBM=$1 -DBN=$2 -DWM=$3 -DWN=$4 -DGROUP=$G $SRC -o /tmp/w6_$1_$2_$3_$4 2>/tmp/w6.log || { echo BUILDFAIL $1 $2 $3 $4; head -30 /tmp/w6.log; exit 1; }; }
build 1024 64 8 1
build 256 64 4 1
run(){ HIP_VISIBLE_DEVICES=$CARD $1 $2 $3 $4 bench 120 1 2>/dev/null | grep -oE '[0-9.]+ TFLOP' | grep -oE '[0-9.]+'; }
echo correctness
HIP_VISIBLE_DEVICES=$CARD /tmp/w6_1024_64_8_1 128 256 256 2>&1 | tail -4
echo "shape hip_w6_1024 hip_w6_256 flydsl_w6"
for shape in "square 4096 4096 8192 1890" "gate_up 4096 19456 2560 1765" "imgQKV 4096 9216 3072 1662" "imgPROJ 4096 3072 3072 1560" "FC2 4096 3072 12288 1828" "LIN1 4352 21504 3072 1747" "LIN2 4352 3072 15360 1688"; do
 set -- $shape; n=$1; M=$2; N=$3; K=$4; R=$5
 echo "$n $(run /tmp/w6_1024_64_8_1 $M $N $K) $(run /tmp/w6_256_64_4_1 $M $N $K) $R"
done
