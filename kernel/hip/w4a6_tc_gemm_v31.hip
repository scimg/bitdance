// w4a6_tc v15: v11 double-buffer + COALESCED pre-packed B ingress.
//   Diagnosis (v11 512x64w81): base 2302, +Bload 1477 (-825!), +pack 1190.
//   The B global load was UNCOALESCED (consecutive lanes -> different B rows, stride K/2).
//   Fix: pre-arrange NVFP4 B nibbles in DRAM in fragment order [gblk][kc][lane][16B]
//        so each lane loads ONE coalesced uint4 (16B). Still NVFP4, still transcoded
//        at runtime -> bandwidth savings preserved, just a fair offline weight re-layout
//        (same as we already do for A).
// Build: -DBM=512 -DBN=64 -DWM=8 -DWN=1 ; Run: ./v15 M N K bench ITERS SPLITS

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include "transcode_lut.h"

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP %s @%s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(1);} } while(0)

typedef __attribute__((ext_vector_type(8))) int   i32x8;
typedef __attribute__((ext_vector_type(6))) int   i32x6;
typedef __attribute__((ext_vector_type(4))) float f32x4;

__device__ __forceinline__ i32x8 load_i32x8(const int* __restrict__ p) {
#if defined(BNATIVEFP6) || defined(FORCE_VECLOAD)
  return *reinterpret_cast<const i32x8*>(p);
#else
  i32x8 v;
  #pragma unroll
  for(int i=0;i<8;++i) v[i]=p[i];
  return v;
#endif
}

__device__ __forceinline__ uint32_t asm_lshl_or(uint32_t x, int sh, uint32_t y) {
  uint32_t r;
  asm volatile("v_lshl_or_b32 %0, %1, %2, %3" : "=v"(r) : "v"(x), "I"(sh), "v"(y));
  return r;
}

// ASMPACK2: balanced-tree pack. Same 16 v_lshl_or work but split into independent
// sub-chains combined with a final v_or_b32, cutting the longest dependency chain from
// 4 (fully serial w1/w4) to 3. Latency-bound (MFMA util ~20%, few waves) so shorter
// chains matter more than raw op count. Same bit layout -> bit-exact.
__device__ __forceinline__ uint32_t asm_or(uint32_t x, uint32_t y) {
  uint32_t r; asm volatile("v_or_b32 %0, %1, %2" : "=v"(r) : "v"(x), "v"(y)); return r;
}
__device__ __forceinline__ i32x8 pack12_asm2_8x2(const uint16_t* __restrict__ lut,
                                                  const uint8_t* q0, const uint8_t* q1,
                                                  int b0, int b1) {
  uint32_t a0=lut[b0|q0[0]], a1=lut[b0|q0[1]], a2=lut[b0|q0[2]], a3=lut[b0|q0[3]];
  uint32_t a4=lut[b0|q0[4]], a5=lut[b0|q0[5]], a6=lut[b0|q0[6]], a7=lut[b0|q0[7]];
  uint32_t b_0=lut[b1|q1[0]], b_1=lut[b1|q1[1]], b_2=lut[b1|q1[2]], b_3=lut[b1|q1[3]];
  uint32_t b_4=lut[b1|q1[4]], b_5=lut[b1|q1[5]], b_6=lut[b1|q1[6]], b_7=lut[b1|q1[7]];
  // group a -> w0,w1,w2  (two parallel sub-chains per word, combined)
  uint32_t w0 = asm_lshl_or(a2, 24, asm_lshl_or(a1, 12, a0));
  uint32_t w1 = asm_or(asm_lshl_or(a3, 4, (a2 >> 8)), asm_lshl_or(a5, 28, (a4 << 16)));
  uint32_t w2 = asm_lshl_or(a7, 20, asm_lshl_or(a6, 8, (a5 >> 4)));
  // group b -> w3,w4,w5
  uint32_t w3 = asm_lshl_or(b_2, 24, asm_lshl_or(b_1, 12, b_0));
  uint32_t w4 = asm_or(asm_lshl_or(b_3, 4, (b_2 >> 8)), asm_lshl_or(b_5, 28, (b_4 << 16)));
  uint32_t w5 = asm_lshl_or(b_7, 20, asm_lshl_or(b_6, 8, (b_5 >> 4)));
  i32x8 v;
  v[0]=int(w0); v[1]=int(w1); v[2]=int(w2); v[3]=int(w3);
  v[4]=int(w4); v[5]=int(w5); v[6]=0; v[7]=0;
  return v;
}

__device__ __forceinline__ i32x8 pack12_asm_8x2(const uint16_t* __restrict__ lut,
                                                 const uint8_t* q0, const uint8_t* q1,
                                                 int b0, int b1) {
  uint32_t a0=lut[b0|q0[0]], a1=lut[b0|q0[1]], a2=lut[b0|q0[2]], a3=lut[b0|q0[3]];
  uint32_t a4=lut[b0|q0[4]], a5=lut[b0|q0[5]], a6=lut[b0|q0[6]], a7=lut[b0|q0[7]];
  uint32_t b_0=lut[b1|q1[0]], b_1=lut[b1|q1[1]], b_2=lut[b1|q1[2]], b_3=lut[b1|q1[3]];
  uint32_t b_4=lut[b1|q1[4]], b_5=lut[b1|q1[5]], b_6=lut[b1|q1[6]], b_7=lut[b1|q1[7]];
  uint32_t w0 = asm_lshl_or(a1, 12, a0);
  w0 = asm_lshl_or(a2, 24, w0);
  uint32_t w1 = (a2 >> 8);
  w1 = asm_lshl_or(a3, 4, w1);
  w1 = asm_lshl_or(a4, 16, w1);
  w1 = asm_lshl_or(a5, 28, w1);
  uint32_t w2 = (a5 >> 4);
  w2 = asm_lshl_or(a6, 8, w2);
  w2 = asm_lshl_or(a7, 20, w2);
  uint32_t w3 = asm_lshl_or(b_1, 12, b_0);
  w3 = asm_lshl_or(b_2, 24, w3);
  uint32_t w4 = (b_2 >> 8);
  w4 = asm_lshl_or(b_3, 4, w4);
  w4 = asm_lshl_or(b_4, 16, w4);
  w4 = asm_lshl_or(b_5, 28, w4);
  uint32_t w5 = (b_5 >> 4);
  w5 = asm_lshl_or(b_6, 8, w5);
  w5 = asm_lshl_or(b_7, 20, w5);
  i32x8 v;
  v[0]=int(w0); v[1]=int(w1); v[2]=int(w2); v[3]=int(w3);
  v[4]=int(w4); v[5]=int(w5); v[6]=0; v[7]=0;
  return v;
}

#ifndef BM
#define BM 512
#endif
#ifndef BN
#define BN 64
#endif
#ifndef WM
#define WM 8
#endif
#ifndef WN
#define WN 1
#endif
static constexpr int BK=128, BLK=32;
static constexpr int MB=BM/16, NB=BN/16;
static constexpr int NWAVE=WM*WN, NTHREAD=NWAVE*64;
static constexpr int MPW=MB/WM, NPW=NB/WN;
static constexpr int BFPT=(NB*64+NTHREAD-1)/NTHREAD;

__constant__ uint8_t d_code_lut[64];
__constant__ uint16_t d_lut16[2048];  // v21: (scale<<8|byte)->two 6-bit codes packed (12 bits)
#ifdef CONSTLUT
#define TC_LUT(idx_) d_lut16[(idx_)]
#else
#define TC_LUT(idx_) sLUT[(idx_)]
#endif
__host__ inline void pack32_h(const uint8_t* c,int* out){uint32_t w[8]={0,0,0,0,0,0,0,0};
  for(int t=0;t<32;++t){uint32_t v=c[t]&0x3F;int b=t*6,wi=b>>5,off=b&31;
    w[wi]|=(v<<off); if(off>26) w[wi+1]|=(v>>(32-off));} for(int i=0;i<8;++i) out[i]=int(w[i]);}

// v41 decode: transcode ONE lane's B fragment (16B fp4 + 2 scale bytes) -> i32x8 fp6,
// directly into registers (no LDS staging). Mirrors the packto() body exactly.
__device__ inline i32x8 tc_one(uint4 raw, int s0, int s1, const uint16_t* __restrict__ sLUT){
  uint2 P0=make_uint2(raw.x,raw.y), P1=make_uint2(raw.z,raw.w);
  const uint8_t* q0=reinterpret_cast<const uint8_t*>(&P0);
  const uint8_t* q1=reinterpret_cast<const uint8_t*>(&P1);
  const int b0=s0<<8, b1=s1<<8;
  uint32_t w[8]={0,0,0,0,0,0,0,0};
  #pragma unroll
  for(int i=0;i<8;++i){uint64_t x=(uint64_t)sLUT[b0|q0[i]]<<((i*12)&31);int wi=(i*12)>>5;
    w[wi]|=(uint32_t)x; w[wi+1]|=(uint32_t)(x>>32);}
  #pragma unroll
  for(int j=0;j<8;++j){uint64_t x=(uint64_t)sLUT[b1|q1[j]]<<((j*12)&31);int wi=3+((j*12)>>5);
    w[wi]|=(uint32_t)x; w[wi+1]|=(uint32_t)(x>>32);}
  i32x8 v;
  #pragma unroll
  for(int i=0;i<8;++i) v[i]=int(w[i]);
  return v;
}

__global__ __launch_bounds__(NTHREAD) void
kern(const int* __restrict__ Afrag, const uint8_t* __restrict__ Bpk,
     const uint8_t* __restrict__ Bspk, float* __restrict__ C, int M,int N,int K){
#if defined(LDSFP4)
  __shared__ uint4    sBraw[2][NB][64];   // fp4 B (16B/lane) -> half the LDS of fp6 pack
  __shared__ uint16_t sBsc [2][NB][64];   // two 3-bit block scales packed
#elif defined(BNATIVEFP6)
  __shared__ i32x8 sB[2][NB][64];          // native MXFP6 B, no transcode
#elif defined(KTILE2)
  // K-tile granularity 2: each double-buffer stage holds TWO consecutive 128-K blocks, so
  // the per-tile __syncthreads() fires once per 2 K-blocks -> HALF the barriers. The tax
  // probe showed the barrier is the dominant compute-bound cost (NOPACK+BAR -22..-28%).
  __shared__ i32x8 sB2[2][2][NB][64];     // [stage][ksub][bblk][lane]  (BN=64 -> 32 KB)
#elif !defined(REGB)
#ifdef XPOSE
  // dword-major B: [buf][bblk][dword][lane]. The per-lane MFMA fragment (8 dwords)
  // was stored lane-contiguous (stride 8 dwords) -> lanes {0,8,16..} alias the same
  // LDS bank => 8-way conflict (measured ~7.9 conflict-cyc/load). Storing dword-major
  // makes each dword's 64 lanes span all 64 banks => conflict-free B read.
  __shared__ int sBx[2][NB][8][64];
#elif defined(COMPACTB)
  __shared__ i32x6 sB[2][NB][64];   // compact FP6 payload: 24 real bytes, pad in registers
#else
  __shared__ i32x8 sB[2][NB][64];
#endif
#endif
#if !defined(BNATIVEFP6) && (!defined(CONSTLUT) || defined(REGB) || defined(LDSFP4))
#ifdef LUT32
  // uint32 LUT: one entry per LDS bank-word (vs 2 uint16/word) -> halves the LUT
  // bank-conflict rate under the BN=128 pack storm (512 concurrent divergent readers).
  __shared__ uint32_t sLUT[2048];
#else
  __shared__ uint16_t sLUT[2048];
#endif
#endif
  const int tid=threadIdx.x, wave=tid>>6, lane=tid&63;
  const int wm=wave/WN, wn=wave%WN;
  const int row=lane&15, g=lane>>4;
  // v31: CTA->tile rasterization (grouped-M ordering) for L2 locality on A (big operand).
// GROUP sweep (seq_group_sweep.sh, MI355 dev6, M=4096): GROUP=2 is best single
// default — wins square/gate_up/imgQKV/imgPROJ/FC1, loses only ~1.8% on FC2.
#ifndef GROUP
#define GROUP 2
#endif
  const int npm=gridDim.y, npn=gridDim.x;
  const int pid=blockIdx.y*npn+blockIdx.x;
  const int ppg=GROUP*npn; const int gid=pid/ppg;
  const int fm=gid*GROUP; const int gm=(npm-fm)<GROUP?(npm-fm):GROUP;
  const int pid_m=fm+((pid%ppg)%gm);
  const int pid_n=(pid%ppg)/gm;
  const int ctm=pid_m*BM, ctn=pid_n*BN, nkc=K/128;
  const int Sp=gridDim.z, kper=K/Sp, kbeg=blockIdx.z*kper;
  const int nkt=kper/BK, gblk0=ctn/16;
#if !defined(BNATIVEFP6) && (!defined(CONSTLUT) || defined(REGB) || defined(LDSFP4))
  for(int i=tid;i<2048;i+=NTHREAD) sLUT[i]=d_lut16[i];
#endif

#if defined(BNATIVEFP6)
  i32x8 PB[BFPT];
  auto prefetch=[&](int kc){ int idx=0;
    for(int task=tid; task<NB*64; task+=NTHREAD,++idx){
      int bblk=task>>6, ln=task&63; size_t base=((size_t)(gblk0+bblk)*nkc+kc)*64+ln;
      const int* p=reinterpret_cast<const int*>(Bpk+base*32);
      PB[idx]=load_i32x8(p);
    } };
  auto packto=[&](int buf){ int idx=0;
    for(int task=tid; task<NB*64; task+=NTHREAD,++idx) sB[buf][task>>6][task&63]=PB[idx]; };
#elif !defined(REGB) && !defined(LDSFP4) && !defined(KTILE2)
  uint2 P0[BFPT],P1[BFPT]; uint8_t SB0[BFPT],SB1[BFPT];
  auto prefetch=[&](int kc){ int idx=0;
    for(int task=tid; task<NB*64; task+=NTHREAD,++idx){
      int bblk=task>>6, ln=task&63; size_t base=((size_t)(gblk0+bblk)*nkc+kc)*64+ln;
      uint4 raw=*reinterpret_cast<const uint4*>(Bpk+base*16);   // COALESCED 16B
      P0[idx]=make_uint2(raw.x,raw.y); P1[idx]=make_uint2(raw.z,raw.w);
      const uint8_t sp=Bspk[base]; SB0[idx]=sp&7; SB1[idx]=(sp>>3)&7;
    } };
  auto packto=[&](int buf){ int idx=0;
    for(int task=tid; task<NB*64; task+=NTHREAD,++idx){
      int bblk=task>>6, ln=task&63;
#ifdef DUMMYPACK
      i32x8 vd; vd[0]=P0[idx].x; vd[1]=P0[idx].y; vd[2]=P1[idx].x; vd[3]=P1[idx].y;
      vd[4]=SB0[idx]; vd[5]=SB1[idx]; vd[6]=0x08080808; vd[7]=0x08080808;
#ifdef COMPACTB
      i32x6 cd; cd[0]=vd[0]; cd[1]=vd[1]; cd[2]=vd[2]; cd[3]=vd[3]; cd[4]=vd[4]; cd[5]=vd[5];
      sB[buf][bblk][ln]=cd;
#else
      sB[buf][bblk][ln]=vd; continue;
#endif
      continue;
#endif
      const uint8_t* q0=reinterpret_cast<const uint8_t*>(&P0[idx]);
      const uint8_t* q1=reinterpret_cast<const uint8_t*>(&P1[idx]);
      const int b0=SB0[idx]<<8, b1=SB1[idx]<<8;
#if defined(ASMPACK2)
      i32x8 v=pack12_asm2_8x2(sLUT, q0, q1, b0, b1);
#elif defined(ASMPACK)
      i32x8 v=pack12_asm_8x2(sLUT, q0, q1, b0, b1);
#else
      uint32_t w[8]={0,0,0,0,0,0,0,0};
      #pragma unroll
      for(int i=0;i<8;++i){uint64_t x=(uint64_t)TC_LUT(b0|q0[i])<<((i*12)&31);int wi=(i*12)>>5;
        w[wi]|=(uint32_t)x; w[wi+1]|=(uint32_t)(x>>32);}
      #pragma unroll
      for(int j=0;j<8;++j){uint64_t x=(uint64_t)TC_LUT(b1|q1[j])<<((j*12)&31);int wi=3+((j*12)>>5);
        w[wi]|=(uint32_t)x; w[wi+1]|=(uint32_t)(x>>32);}
      i32x8 v;
      #pragma unroll
      for(int i=0;i<8;++i) v[i]=int(w[i]);
#endif
#ifdef XPOSE
      #pragma unroll
      for(int i=0;i<8;++i) sBx[buf][bblk][i][ln]=v[i];
#elif defined(COMPACTB)
      i32x6 cv;
      #pragma unroll
      for(int i=0;i<6;++i) cv[i]=v[i];
      sB[buf][bblk][ln]=cv;
#else
      sB[buf][bblk][ln]=v;
#endif
    } };
#endif

  f32x4 acc[MPW][NPW]={};
#if defined(LDSFP4)
  // Stage RAW fp4 B in LDS (half the fp6 footprint -> higher occupancy), transcode at
  // consume in registers. On memory-bound the LUT reads hide under HBM; the occupancy
  // gain hides HBM latency better. Coalesced cooperative load + double-buffer as usual.
  uint4 Praw[BFPT]; uint16_t Psc[BFPT];
  auto prefetch=[&](int kc){ int idx=0;
    for(int task=tid; task<NB*64; task+=NTHREAD,++idx){
      int bblk=task>>6, ln=task&63; size_t base=((size_t)(gblk0+bblk)*nkc+kc)*64+ln;
      Praw[idx]=*reinterpret_cast<const uint4*>(Bpk+base*16);
      const uint8_t sp=Bspk[base]; Psc[idx]=(uint16_t)((sp&7)|(((sp>>3)&7)<<8));
    } };
  auto storeraw=[&](int buf){ int idx=0;
    for(int task=tid; task<NB*64; task+=NTHREAD,++idx){
      sBraw[buf][task>>6][task&63]=Praw[idx]; sBsc[buf][task>>6][task&63]=Psc[idx]; } };
  prefetch(kbeg/128); storeraw(0); __syncthreads();
  for(int t=0;t<nkt;++t){ int cur=t&1, kc=(kbeg/128)+t;
    if(t+1<nkt) prefetch(kc+1);
    i32x8 areg[MPW];
    #pragma unroll
    for(int mi=0;mi<MPW;++mi){int abg=ctm/16+wm*MPW+mi;
      const int* pa=Afrag+((size_t)abg*nkc+kc)*64*8+(size_t)lane*8;
      areg[mi]=load_i32x8(pa);}
    i32x8 breg[NPW];
    #pragma unroll
    for(int ni=0;ni<NPW;++ni){ int bblk=wn*NPW+ni;
      uint4 raw=sBraw[cur][bblk][lane]; uint16_t sc=sBsc[cur][bblk][lane];
      breg[ni]=tc_one(raw, sc&7, (sc>>8)&7, sLUT); }
    #pragma unroll
    for(int ni=0;ni<NPW;++ni)
      #pragma unroll
      for(int mi=0;mi<MPW;++mi)
        acc[mi][ni]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            areg[mi],breg[ni],acc[mi][ni],2,2,0,127,0,127);
    if(t+1<nkt) storeraw((t+1)&1);
    __syncthreads();
  }
#elif defined(REGB)
  // v41 decode path: per-wave register transcode of B, no LDS B-staging, no per-k barrier.
  // Wins the tiny-M (decode) regime where the cooperative-pack barrier dominates.
  __syncthreads();                       // sLUT ready (loaded above), once only
  for(int t=0;t<nkt;++t){ int kc=(kbeg/128)+t;
    i32x8 breg[NPW];
    #pragma unroll
    for(int ni=0;ni<NPW;++ni){
      int bblk=wn*NPW+ni; size_t base=((size_t)(gblk0+bblk)*nkc+kc)*64+lane;
      uint4 raw=*reinterpret_cast<const uint4*>(Bpk+base*16);
      const uint8_t sp=Bspk[base];
      breg[ni]=tc_one(raw, sp&7, (sp>>3)&7, sLUT);
    }
    i32x8 areg[MPW];
    #pragma unroll
    for(int mi=0;mi<MPW;++mi){int abg=ctm/16+wm*MPW+mi;
      const int* pa=Afrag+((size_t)abg*nkc+kc)*64*8+(size_t)lane*8;
      areg[mi]=load_i32x8(pa);}
    #pragma unroll
    for(int ni=0;ni<NPW;++ni)
      #pragma unroll
      for(int mi=0;mi<MPW;++mi)
        acc[mi][ni]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            areg[mi],breg[ni],acc[mi][ni],2,2,0,127,0,127);
  }
#elif defined(REGBPF)
  // Decode MLP lever: register-transcode B (no LDS B, no barrier) with a DEEP software
  // prefetch ring of raw NVFP4 B (uint4/lane) PFD k-tiles ahead. The plain REGB path loads
  // B in the same iteration it consumes -> exposed HBM latency; memory-bound decode is
  // MLP-limited. Issuing PFD*NPW outstanding B loads per lane hides the latency (mirrors
  // FlyDSL's async LDS ring). Same math as REGB -> bit-exact.
#ifndef PFD
#define PFD 4
#endif
  __syncthreads();                        // sLUT ready (loaded above), once only
  uint4 Braw[PFD][NPW]; uint8_t Bs0[PFD][NPW], Bs1[PFD][NPW];
  auto ldB=[&](int slot,int kc){
    #pragma unroll
    for(int ni=0;ni<NPW;++ni){ int bblk=wn*NPW+ni;
      size_t base=((size_t)(gblk0+bblk)*nkc+kc)*64+lane;
      Braw[slot][ni]=*reinterpret_cast<const uint4*>(Bpk+base*16);
      const uint8_t sp=Bspk[base]; Bs0[slot][ni]=sp&7; Bs1[slot][ni]=(sp>>3)&7;
    } };
  #pragma unroll
  for(int p=0;p<PFD;++p){ if(p<nkt) ldB(p,(kbeg/128)+p); }
  for(int t=0;t<nkt;++t){ int kc=(kbeg/128)+t; int slot=t%PFD;
    i32x8 breg[NPW];
    #pragma unroll
    for(int ni=0;ni<NPW;++ni)
      breg[ni]=tc_one(Braw[slot][ni],Bs0[slot][ni],Bs1[slot][ni],sLUT);
    if(t+PFD<nkt) ldB(slot,(kbeg/128)+t+PFD);   // refill this slot PFD ahead
    i32x8 areg[MPW];
    #pragma unroll
    for(int mi=0;mi<MPW;++mi){int abg=ctm/16+wm*MPW+mi;
      const int* pa=Afrag+((size_t)abg*nkc+kc)*64*8+(size_t)lane*8;
      areg[mi]=load_i32x8(pa);}
    #pragma unroll
    for(int ni=0;ni<NPW;++ni)
      #pragma unroll
      for(int mi=0;mi<MPW;++mi)
        acc[mi][ni]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            areg[mi],breg[ni],acc[mi][ni],2,2,0,127,0,127);
  }
#elif defined(NOPACK)
  { int idx=0; for(int task=tid;task<NB*64;task+=NTHREAD,++idx){ i32x8 v;
      #pragma unroll
      for(int i=0;i<8;++i) v[i]=int(0x08080808);
#ifdef COMPACTB
      i32x6 cv; for(int i=0;i<6;++i) cv[i]=v[i]; sB[0][task>>6][task&63]=cv; sB[1][task>>6][task&63]=cv;
#else
      sB[0][task>>6][task&63]=v; sB[1][task>>6][task&63]=v;
#endif
    } }
  __syncthreads();
  // PROBE flags decompose the NOPACK->DUMMY tax: PROBE_PF adds the per-tile B HBM prefetch
  // (into unused regs), PROBE_BAR adds the per-tile barrier. Isolates prefetch vs barrier.
#ifdef PROBE_PF
  prefetch(kbeg/128);
#endif
  for(int t=0;t<nkt;++t){ int cur=t&1, kc=(kbeg/128)+t;
#ifdef PROBE_PF
    if(t+1<nkt) prefetch(kc+1);
    asm volatile(""::"v"(P0[0].x),"v"(P1[0].x),"v"((int)SB0[0]));  // sink: keep loads live
#endif
    i32x8 areg[MPW];
    #pragma unroll
    for(int mi=0;mi<MPW;++mi){int abg=ctm/16+wm*MPW+mi;
      const int* pa=Afrag+((size_t)abg*nkc+kc)*64*8+(size_t)lane*8;
      areg[mi]=load_i32x8(pa);}
    #pragma unroll
    for(int ni=0;ni<NPW;++ni){
#ifdef COMPACTB
      i32x6 bc=sB[cur][wn*NPW+ni][lane]; i32x8 b;
      #pragma unroll
      for(int i=0;i<6;++i) b[i]=bc[i];
      b[6]=0; b[7]=0;
#else
      i32x8 b=sB[cur][wn*NPW+ni][lane];
#endif
      #pragma unroll
      for(int mi=0;mi<MPW;++mi)
        acc[mi][ni]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(areg[mi],b,acc[mi][ni],2,2,0,127,0,127);
    }
#ifdef PROBE_BAR
    __syncthreads();
#endif
  }
#elif defined(KTILE2)
  // Pack TWO consecutive 128-K blocks per double-buffer stage -> one barrier per 2 tiles.
  uint2 Q0[2][BFPT],Q1[2][BFPT]; uint8_t QS0[2][BFPT],QS1[2][BFPT];
  auto prefetch2=[&](int kc){
    #pragma unroll
    for(int sub=0;sub<2;++sub){ int idx=0;
      for(int task=tid; task<NB*64; task+=NTHREAD,++idx){
        int bblk=task>>6, ln=task&63; size_t base=((size_t)(gblk0+bblk)*nkc+kc+sub)*64+ln;
        uint4 raw=*reinterpret_cast<const uint4*>(Bpk+base*16);
        Q0[sub][idx]=make_uint2(raw.x,raw.y); Q1[sub][idx]=make_uint2(raw.z,raw.w);
        const uint8_t sp=Bspk[base]; QS0[sub][idx]=sp&7; QS1[sub][idx]=(sp>>3)&7;
      } } };
  auto packto2=[&](int st){
    #pragma unroll
    for(int sub=0;sub<2;++sub){ int idx=0;
      for(int task=tid; task<NB*64; task+=NTHREAD,++idx){
        int bblk=task>>6, ln=task&63;
        const uint8_t* q0=reinterpret_cast<const uint8_t*>(&Q0[sub][idx]);
        const uint8_t* q1=reinterpret_cast<const uint8_t*>(&Q1[sub][idx]);
        const int b0=QS0[sub][idx]<<8, b1=QS1[sub][idx]<<8;
#ifdef ASMPACK
        i32x8 v=pack12_asm_8x2(sLUT, q0, q1, b0, b1);
#else
        uint32_t w[8]={0,0,0,0,0,0,0,0};
        #pragma unroll
        for(int i=0;i<8;++i){uint64_t x=(uint64_t)TC_LUT(b0|q0[i])<<((i*12)&31);int wi=(i*12)>>5;
          w[wi]|=(uint32_t)x; w[wi+1]|=(uint32_t)(x>>32);}
        #pragma unroll
        for(int j=0;j<8;++j){uint64_t x=(uint64_t)TC_LUT(b1|q1[j])<<((j*12)&31);int wi=3+((j*12)>>5);
          w[wi]|=(uint32_t)x; w[wi+1]|=(uint32_t)(x>>32);}
        i32x8 v;
        #pragma unroll
        for(int i=0;i<8;++i) v[i]=int(w[i]);
#endif
        sB2[st][sub][bblk][ln]=v;
      } } };
  prefetch2(kbeg/128); packto2(0);
  __syncthreads();
  for(int tt=0; tt<nkt; tt+=2){ int cur=(tt>>1)&1;
    if(tt+2<nkt) prefetch2((kbeg/128)+tt+2);
    #pragma unroll
    for(int sub=0;sub<2;++sub){ int kc=(kbeg/128)+tt+sub;
      i32x8 areg[MPW];
      #pragma unroll
      for(int mi=0;mi<MPW;++mi){int abg=ctm/16+wm*MPW+mi;
        const int* pa=Afrag+((size_t)abg*nkc+kc)*64*8+(size_t)lane*8;
        areg[mi]=load_i32x8(pa);}
      #pragma unroll
      for(int ni=0;ni<NPW;++ni){ i32x8 b=sB2[cur][sub][wn*NPW+ni][lane];
        #pragma unroll
        for(int mi=0;mi<MPW;++mi)
          acc[mi][ni]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
              areg[mi],b,acc[mi][ni],2,2,0,127,0,127);
      }
    }
    if(tt+2<nkt) packto2(((tt>>1)+1)&1);
    __syncthreads();
  }
#elif defined(ILVPACK)
  // COEXEC(MFMA,VALU) ~6% => the transcode VALU (3.6x the MFMA instr count) runs
  // serially with MFMA. Interleave: issue the next-buffer pack AFTER the first ni's
  // MFMAs so the pack VALU overlaps the remaining ni's in-flight MFMAs (different
  // pipes co-execute). packto writes the OTHER buffer => no hazard with current reads.
  prefetch(kbeg/128); packto(0);
  __syncthreads();
  for(int t=0;t<nkt;++t){
    int cur=t&1, kc=(kbeg/128)+t;
    if(t+1<nkt) prefetch(kc+1);
    i32x8 areg[MPW];
    #pragma unroll
    for(int mi=0;mi<MPW;++mi){int abg=ctm/16+wm*MPW+mi;
      const int* pa=Afrag+((size_t)abg*nkc+kc)*64*8+(size_t)lane*8;
      areg[mi]=load_i32x8(pa);}
    #pragma unroll
    for(int ni=0;ni<NPW;++ni){ int bblk=wn*NPW+ni;
#ifdef XPOSE
      i32x8 b;
      #pragma unroll
      for(int i=0;i<8;++i) b[i]=sBx[cur][bblk][i][lane];
#else
      i32x8 b=sB[cur][bblk][lane];
#endif
      #pragma unroll
      for(int mi=0;mi<MPW;++mi)
        acc[mi][ni]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            areg[mi],b,acc[mi][ni],2,2,0,127,0,127);
      if(ni==0 && t+1<nkt) packto((t+1)&1);   // overlap pack VALU with ni=1..3 MFMAs
    }
    __syncthreads();
  }
#elif defined(APIPE)
  // MFMA util ~= occupancy (~20%) => latency-bound. The A operand was loaded in the
  // SAME iteration it feeds MFMA (exposed global-load latency, not hidden by only 2
  // waves/SIMD). Restructure: B fragments register-resident, A software-prefetched one
  // mi ahead so its load latency overlaps the current mi's MFMAs (ILP-based hiding).
  auto loadA=[&](int mi,int kc)->i32x8{ int abg=ctm/16+wm*MPW+mi;
    const int* pa=Afrag+((size_t)abg*nkc+kc)*64*8+(size_t)lane*8;
    return load_i32x8(pa); };
  prefetch(kbeg/128); packto(0);
  __syncthreads();
  for(int t=0;t<nkt;++t){
    int cur=t&1, kc=(kbeg/128)+t;
    if(t+1<nkt) prefetch(kc+1);
    i32x8 breg[NPW];
    #pragma unroll
    for(int ni=0;ni<NPW;++ni){ int bblk=wn*NPW+ni;
#ifdef XPOSE
      #pragma unroll
      for(int i=0;i<8;++i) breg[ni][i]=sBx[cur][bblk][i][lane];
#else
      breg[ni]=sB[cur][bblk][lane];
#endif
    }
    i32x8 a_next=loadA(0,kc);
    #pragma unroll
    for(int mi=0;mi<MPW;++mi){
      i32x8 a=a_next;
      if(mi+1<MPW) a_next=loadA(mi+1,kc);
      #pragma unroll
      for(int ni=0;ni<NPW;++ni)
        acc[mi][ni]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a,breg[ni],acc[mi][ni],2,2,0,127,0,127);
#ifdef ILVPACK
      // overlap the next-tile LUT pack VALU with mi=1..MPW-1's in-flight MFMAs
      if(mi==0 && t+1<nkt) packto((t+1)&1);
#endif
    }
#ifndef ILVPACK
    if(t+1<nkt) packto((t+1)&1);
#endif
    __syncthreads();
  }
#else
  prefetch(kbeg/128); packto(0);
  // NOBAR: DIAGNOSTIC ONLY — drop the per-K-tile CTA barrier to measure the barrier-free
  // MFMA-utilization ceiling. Output is INCORRECT (races the double buffer); use only for
  // perf/MfmaUtil to decide whether the async-LDS rewrite is worth it. Never ship.
#ifndef NOBAR
  __syncthreads();
#endif
  for(int t=0;t<nkt;++t){
    int cur=t&1, kc=(kbeg/128)+t;
    if(t+1<nkt) prefetch(kc+1);
    i32x8 areg[MPW];
    #pragma unroll
    for(int mi=0;mi<MPW;++mi){int abg=ctm/16+wm*MPW+mi;
      const int* pa=Afrag+((size_t)abg*nkc+kc)*64*8+(size_t)lane*8;
      areg[mi]=load_i32x8(pa);}
    #pragma unroll
    for(int ni=0;ni<NPW;++ni){
#ifdef XPOSE
      i32x8 b; int bblk=wn*NPW+ni;
      #pragma unroll
      for(int i=0;i<8;++i) b[i]=sBx[cur][bblk][i][lane];
#elif defined(COMPACTB)
      i32x6 bc=sB[cur][wn*NPW+ni][lane]; i32x8 b;
      #pragma unroll
      for(int i=0;i<6;++i) b[i]=bc[i];
      b[6]=0; b[7]=0;
#else
      i32x8 b=sB[cur][wn*NPW+ni][lane];
#endif
      #pragma unroll
      for(int mi=0;mi<MPW;++mi)
        acc[mi][ni]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            areg[mi],b,acc[mi][ni],2,2,0,127,0,127);
    }
    if(t+1<nkt) packto((t+1)&1);
#ifndef NOBAR
    __syncthreads();
#endif
  }
#endif

  const bool one=(gridDim.z==1);
  #pragma unroll
  for(int mi=0;mi<MPW;++mi)
    #pragma unroll
    for(int ni=0;ni<NPW;++ni){
      int out_col=ctn+(wn*NPW+ni)*16+row, out_row0=ctm+(wm*MPW+mi)*16+g*4;
      #pragma unroll
      for(int t=0;t<4;++t){int m=out_row0+t;
        if(m<M&&out_col<N){size_t idx=(size_t)m*N+out_col;
          if(one) C[idx]=acc[mi][ni][t]; else atomicAdd(&C[idx],acc[mi][ni][t]);}}
    }
}

static float fp6d(uint8_t c){int s=(c>>5)&1,e=(c>>3)&3,m=c&7;float mg=(e==0)?(m/8.f):((1.f+m/8.f)*float(1<<(e-1)));return s?-mg:mg;}
static uint8_t htc(uint8_t nib,uint8_t s){return uint8_t((((nib>>3)&1)<<5)|NVFP4_MXFP6_CODE_LUT[(s&7)*8+(nib&7)]);}
static void build_Afrag(const std::vector<uint8_t>&A,int M,int K,std::vector<int>&Af){
  int nabg=M/16,nkc=K/128; Af.assign((size_t)nabg*nkc*64*8,0);
  for(int abg=0;abg<nabg;++abg)for(int kc=0;kc<nkc;++kc)for(int ln=0;ln<64;++ln){
    int r=ln&15,g=ln>>4,row=abg*16+r; uint8_t c[32];
    for(int e=0;e<16;++e){c[e]=A[(size_t)row*K+kc*128+g*16+e]&0x3F;
      c[e+16]=A[(size_t)row*K+kc*128+g*16+64+e]&0x3F;}
    pack32_h(c,&Af[(((size_t)abg*nkc+kc)*64+ln)*8]);}
}
// Coalesced B pre-pack: [gblk][kc][lane][16 bytes raw NVFP4] + scales [.][2]
static void build_Bpk(const std::vector<uint8_t>&Bn,const std::vector<uint8_t>&Bs,int N,int K,
                      std::vector<uint8_t>&Bpk,std::vector<uint8_t>&Bspk){
  int Nb=N/16,nkc=K/128;
#ifdef BNATIVEFP6
  Bpk.assign((size_t)Nb*nkc*64*32,0); Bspk.clear();
  for(int gblk=0;gblk<Nb;++gblk)for(int kc=0;kc<nkc;++kc)for(int ln=0;ln<64;++ln){
    int r=ln&15,g=ln>>4,brow=gblk*16+r,k0=kc*128; size_t base=((size_t)gblk*nkc+kc)*64+ln;
    uint8_t c[32];
    for(int e=0;e<16;++e){
      c[e]    = Bn[(size_t)brow*K + k0 + g*16 + e] & 0x3F;
      c[e+16] = Bn[(size_t)brow*K + k0 + g*16 + 64 + e] & 0x3F;
    }
    int w[8]; pack32_h(c,w); memcpy(&Bpk[base*32], w, 32);
  }
#else
  Bpk.assign((size_t)Nb*nkc*64*16,0); Bspk.assign((size_t)Nb*nkc*64,0);
  for(int gblk=0;gblk<Nb;++gblk)for(int kc=0;kc<nkc;++kc)for(int ln=0;ln<64;++ln){
    int r=ln&15,g=ln>>4,brow=gblk*16+r,k0=kc*128; size_t base=((size_t)gblk*nkc+kc)*64+ln;
    memcpy(&Bpk[base*16],   &Bn[(size_t)brow*(K/2)+(k0+g*16)/2],    8);
    memcpy(&Bpk[base*16+8], &Bn[(size_t)brow*(K/2)+(k0+g*16+64)/2], 8);
    const uint8_t s0=Bs[(size_t)brow*(K/BLK)+(k0+g*16)/BLK]&7;
    const uint8_t s1=Bs[(size_t)brow*(K/BLK)+(k0+g*16+64)/BLK]&7;
    Bspk[base]=(uint8_t)(s0|(s1<<3));
  }
#endif
}

int main(int argc,char**argv){
  int M=BM,N=BN,K=256; bool bench=false; int iters=200,S=1;
  if(argc>=4){M=atoi(argv[1]);N=atoi(argv[2]);K=atoi(argv[3]);}
  if(argc>=5&&std::string(argv[4])=="bench") bench=true;
  if(argc>=6) iters=atoi(argv[5]);
  if(argc>=7) S=atoi(argv[6]);
  if(N%BN||K%BK||(K/S)%BK){fprintf(stderr,"bad dims\n");return 1;}
  // partial-M tiles allowed: pad M up to a BM multiple (A zero-padded, output masked).
  const int Mp=((M+BM-1)/BM)*BM;
  printf("v31 tile=%dx%d w%dx%d M=%d(pad%d) N=%d K=%d S=%d %s\n",BM,BN,WM,WN,M,Mp,N,K,S,bench?"bench":"verify");
  srand(1234);
  std::vector<uint8_t> A((size_t)M*K),
#ifdef BNATIVEFP6
    Bn((size_t)N*K),
#else
    Bn((size_t)N*(K/2)),
#endif
    Bs((size_t)N*(K/BLK));
  for(auto&x:A)x=rand()&0x3F; for(auto&x:Bn)x=rand()&0xFF;
  for(auto&x:Bs)x=(uint8_t)(0x38|(rand()&7));
  std::vector<int> Af; build_Afrag(A,M,K,Af);
  Af.resize((size_t)(Mp/16)*(K/128)*64*8, 0);   // zero-pad A fragments for partial tile
  std::vector<uint8_t> Bpk,Bspk; build_Bpk(Bn,Bs,N,K,Bpk,Bspk);
  HIP_CHECK(hipMemcpyToSymbol(d_code_lut,NVFP4_MXFP6_CODE_LUT,64));
  { static uint16_t lut16[2048]; for(int s=0;s<8;++s)for(int b=0;b<256;++b){
      int lo=b&0xF, hi=(b>>4)&0xF;
      uint8_t cl=(uint8_t)((((lo>>3)&1)<<5)|NVFP4_MXFP6_CODE_LUT[s*8+(lo&7)]);
      uint8_t ch=(uint8_t)((((hi>>3)&1)<<5)|NVFP4_MXFP6_CODE_LUT[s*8+(hi&7)]);
      lut16[(s<<8)|b]=(uint16_t)((cl&0x3F)|((ch&0x3F)<<6)); }
    HIP_CHECK(hipMemcpyToSymbol(d_lut16,lut16,sizeof(lut16))); }
  int*dAf; uint8_t*dBp,*dBs; float*dC;
  HIP_CHECK(hipMalloc(&dAf,Af.size()*sizeof(int)));HIP_CHECK(hipMalloc(&dBp,Bpk.size()));
  HIP_CHECK(hipMalloc(&dBs,Bspk.size()));HIP_CHECK(hipMalloc(&dC,(size_t)M*N*sizeof(float)));
  HIP_CHECK(hipMemcpy(dAf,Af.data(),Af.size()*sizeof(int),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dBp,Bpk.data(),Bpk.size(),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dBs,Bspk.data(),Bspk.size(),hipMemcpyHostToDevice));
  dim3 grid(N/BN,Mp/BM,S);
  auto launch=[&](){ if(S>1) HIP_CHECK(hipMemset(dC,0,(size_t)M*N*sizeof(float)));
    kern<<<grid,NTHREAD>>>(dAf,dBp,dBs,dC,M,N,K); };
  if(bench){
    for(int w=0;w<20;++w) launch(); HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0,t1; (void)hipEventCreate(&t0); (void)hipEventCreate(&t1);
    (void)hipEventRecord(t0); for(int i=0;i<iters;++i) launch();
    (void)hipEventRecord(t1); (void)hipEventSynchronize(t1);
    float ms=0; (void)hipEventElapsedTime(&ms,t0,t1);
    double per=ms/iters, tf=2.0*M*N*K/(per*1e-3)*1e-12;
    printf("bench: %.4f ms/iter  %.2f TFLOP/s\n",per,tf); return 0;
  }
  std::vector<float> gold((size_t)M*N,0.f);
  for(int m=0;m<M;++m)for(int n=0;n<N;++n){float a=0.f;
    for(int k=0;k<K;++k){
#ifdef BNATIVEFP6
      uint8_t bc=Bn[(size_t)n*K+k]&0x3F;
#else
      uint8_t nib=(Bn[(size_t)n*(K/2)+k/2]>>((k&1)*4))&0xF;
      uint8_t sb=Bs[(size_t)n*(K/BLK)+k/BLK]; uint8_t bc=htc(nib,sb);
#endif
      a+=fp6d(A[(size_t)m*K+k])*fp6d(bc);}
    gold[(size_t)m*N+n]=a;}
  launch(); HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> C((size_t)M*N);
  HIP_CHECK(hipMemcpy(C.data(),dC,C.size()*sizeof(float),hipMemcpyDeviceToHost));
  double mx=0; int nbad=0;
  for(size_t i=0;i<C.size();++i){double d=std::abs((double)C[i]-gold[i]);
    double r=d/(std::abs(gold[i])+1e-6); mx=std::max(mx,d); if(d>1e-2&&r>1e-3)++nbad;}
  printf("verify: maxabs=%.4g nbad=%d/%zu\n",mx,nbad,C.size());
  printf(nbad==0?"RESULT: PASS\n":"RESULT: FAIL\n"); return nbad==0?0:2;
}
