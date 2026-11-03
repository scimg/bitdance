// MFMA-rate microbenchmark: back-to-back v_mfma_scale_f32_16x16x128_f8f6f4 on
// REGISTER-RESIDENT operands (no LDS, no HBM). Isolates the raw matrix-core issue
// rate for a given operand format (CBSZ/BLGP) from any operand-delivery cost.
//   CBSZ/BLGP: 2=fp6(E2M3), 4=fp4(E2M1).  fp4xfp4 -> 4,4 ; fp6xfp4 -> 2,4 ; fp6xfp6 -> 2,2
// Build: hipcc -O3 --offload-arch=gfx950 -DCBSZ=x -DBLGP=y mfma_peak.hip -o peak
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef __attribute__((ext_vector_type(8))) int   i32x8;
typedef __attribute__((ext_vector_type(4))) float f32x4;
#ifndef CBSZ
#define CBSZ 2
#endif
#ifndef BLGP
#define BLGP 2
#endif
#ifndef UNROLL
#define UNROLL 64
#endif
#define MFMA(A,B,C) __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A,B,C,CBSZ,BLGP,0,127,0,127)

__global__ void peak(const int* __restrict__ seed, float* __restrict__ out, int reps){
  i32x8 a,b;
  #pragma unroll
  for(int i=0;i<8;++i){ a[i]=seed[i]; b[i]=seed[i+8]; }
  // 4 independent accumulator chains to hide MFMA pipeline latency
  f32x4 c0{}, c1{}, c2{}, c3{};
  for(int r=0;r<reps;++r){
    #pragma unroll
    for(int i=0;i<UNROLL;++i){
      c0=MFMA(a,b,c0); c1=MFMA(a,b,c1); c2=MFMA(a,b,c2); c3=MFMA(a,b,c3);
    }
  }
  float s=c0[0]+c1[1]+c2[2]+c3[3];
  if(s==-1.234e30f) out[threadIdx.x]=s;   // DCE guard, never taken
}

int main(int argc,char**argv){
  int blocks = argc>1?atoi(argv[1]):2048;
  int waves  = argc>2?atoi(argv[2]):4;      // waves per block
  int reps   = argc>3?atoi(argv[3]):2000;
  int threads=waves*64;
  int seed[16]; for(int i=0;i<16;++i) seed[i]=0x08080808; // valid fp6/fp4 codes
  int *dseed; float *dout;
  (void)hipMalloc(&dseed,sizeof(seed)); (void)hipMalloc(&dout,threads*sizeof(float));
  (void)hipMemcpy(dseed,seed,sizeof(seed),hipMemcpyHostToDevice);
  auto launch=[&](){ peak<<<blocks,threads>>>(dseed,dout,reps); };
  for(int w=0;w<5;++w) launch(); (void)hipDeviceSynchronize();
  hipEvent_t t0,t1;(void)hipEventCreate(&t0);(void)hipEventCreate(&t1);
  (void)hipEventRecord(t0); int iters=10; for(int i=0;i<iters;++i) launch();
  (void)hipEventRecord(t1);(void)hipEventSynchronize(t1);
  float ms=0;(void)hipEventElapsedTime(&ms,t0,t1);
  // wave-MFMAs = blocks*waves*reps*UNROLL*4 (4 chains); each 16x16x128 = 2*16*16*128 flop
  double wmfma=(double)blocks*waves*reps*UNROLL*4.0*iters;
  double flop=wmfma*2.0*16*16*128;
  double tf=flop/(ms*1e-3)*1e-12;
  printf("cbsz=%d blgp=%d  %.1f TFLOP/s  (%.3f ms, %g wave-mfmas)\n",CBSZ,BLGP,tf,ms/iters,wmfma/iters);
  return 0;
}
