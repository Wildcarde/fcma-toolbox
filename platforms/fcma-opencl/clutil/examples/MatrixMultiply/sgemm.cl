#pragma once
#ifdef __DEVICE__
#define LDAS 97
#define LDBS 17

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void sgemmNN(int n, int m, int k, float alpha, __global float* a, int lda, __global float* b, int ldb, float beta, __global float* c, int ldc)
{
  float c0_0 = 0.0f;
  float c0_16 = 0.0f;
  float c0_32 = 0.0f;
  float c0_48 = 0.0f;
  float c0_64 = 0.0f;
  float c0_80 = 0.0f;
  float c16_0 = 0.0f;
  float c16_16 = 0.0f;
  float c16_32 = 0.0f;
  float c16_48 = 0.0f;
  float c16_64 = 0.0f;
  float c16_80 = 0.0f;
  float c32_0 = 0.0f;
  float c32_16 = 0.0f;
  float c32_32 = 0.0f;
  float c32_48 = 0.0f;
  float c32_64 = 0.0f;
  float c32_80 = 0.0f;
  float c48_0 = 0.0f;
  float c48_16 = 0.0f;
  float c48_32 = 0.0f;
  float c48_48 = 0.0f;
  float c48_64 = 0.0f;
  float c48_80 = 0.0f;
  float c64_0 = 0.0f;
  float c64_16 = 0.0f;
  float c64_32 = 0.0f;
  float c64_48 = 0.0f;
  float c64_64 = 0.0f;
  float c64_80 = 0.0f;
  float c80_0 = 0.0f;
  float c80_16 = 0.0f;
  float c80_32 = 0.0f;
  float c80_48 = 0.0f;
  float c80_64 = 0.0f;
  float c80_80 = 0.0f;

  float a0;
  float a16;
  float a32;
  float a48;
  float a64;
  float a80;

  float b0;
  float b16;
  float b32;
  float b48;
  float b64;
  float b80;

  __local float aShared[LDAS * 16];
  __local float bShared[LDBS * 96];

  const int rowBlock = get_local_id(0);
  const int colBlock = get_local_id(1);

  const int rowC = get_group_id(0) * 96;
  const int colC = get_group_id(1) * 96;

  __global float* blockC;

  for(int i = 0; i < k; i+= 16)
  {
    #pragma unroll 6
    for(int l = 0; l < 96; l+= 16)
    {
      aShared[rowBlock + l + LDAS * colBlock] = a[rowC + rowBlock + l + (colBlock + i) * lda];
      bShared[colBlock + LDBS * (rowBlock + l)] = b[colBlock + i + (colC + rowBlock + l) * ldb];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int l = 0; l <16; l++)
    {
      a0 = aShared[rowBlock + 0 + LDAS * l];
      a16 = aShared[rowBlock + 16 + LDAS * l];
      a32 = aShared[rowBlock + 32 + LDAS * l];
      a48 = aShared[rowBlock + 48 + LDAS * l];
      a64 = aShared[rowBlock + 64 + LDAS * l];
      a80 = aShared[rowBlock + 80 + LDAS * l];

      b0 = bShared[l + LDBS * (colBlock + 0)];
      b16 = bShared[l + LDBS * (colBlock + 16)];
      b32 = bShared[l + LDBS * (colBlock + 32)];
      b48 = bShared[l + LDBS * (colBlock + 48)];
      b64 = bShared[l + LDBS * (colBlock + 64)];
      b80 = bShared[l + LDBS * (colBlock + 80)];

      c0_0 = fma(a0, b0, c0_0);
      c0_16 = fma(a0, b16, c0_16);
      c0_32 = fma(a0, b32, c0_32);
      c0_48 = fma(a0, b48, c0_48);
      c0_64 = fma(a0, b64, c0_64);
      c0_80 = fma(a0, b80, c0_80);
      c16_0 = fma(a16, b0, c16_0);
      c16_16 = fma(a16, b16, c16_16);
      c16_32 = fma(a16, b32, c16_32);
      c16_48 = fma(a16, b48, c16_48);
      c16_64 = fma(a16, b64, c16_64);
      c16_80 = fma(a16, b80, c16_80);
      c32_0 = fma(a32, b0, c32_0);
      c32_16 = fma(a32, b16, c32_16);
      c32_32 = fma(a32, b32, c32_32);
      c32_48 = fma(a32, b48, c32_48);
      c32_64 = fma(a32, b64, c32_64);
      c32_80 = fma(a32, b80, c32_80);
      c48_0 = fma(a48, b0, c48_0);
      c48_16 = fma(a48, b16, c48_16);
      c48_32 = fma(a48, b32, c48_32);
      c48_48 = fma(a48, b48, c48_48);
      c48_64 = fma(a48, b64, c48_64);
      c48_80 = fma(a48, b80, c48_80);
      c64_0 = fma(a64, b0, c64_0);
      c64_16 = fma(a64, b16, c64_16);
      c64_32 = fma(a64, b32, c64_32);
      c64_48 = fma(a64, b48, c64_48);
      c64_64 = fma(a64, b64, c64_64);
      c64_80 = fma(a64, b80, c64_80);
      c80_0 = fma(a80, b0, c80_0);
      c80_16 = fma(a80, b16, c80_16);
      c80_32 = fma(a80, b32, c80_32);
      c80_48 = fma(a80, b48, c80_48);
      c80_64 = fma(a80, b64, c80_64);
      c80_80 = fma(a80, b80, c80_80);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  blockC = &c[rowC + ldc * (colC + colBlock)];

  if(rowC + 95 < m && colC + 95 < n)
  {
    blockC[rowBlock + 0] = alpha * c0_0 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_0 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_0 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_0 + beta * blockC[rowBlock + 48];
    blockC[rowBlock + 64] = alpha * c64_0 + beta * blockC[rowBlock + 64];
    blockC[rowBlock + 80] = alpha * c80_0 + beta * blockC[rowBlock + 80];

    blockC += 16 * ldc;

    blockC[rowBlock + 0] = alpha * c0_16 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_16 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_16 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_16 + beta * blockC[rowBlock + 48];
    blockC[rowBlock + 64] = alpha * c64_16 + beta * blockC[rowBlock + 64];
    blockC[rowBlock + 80] = alpha * c80_16 + beta * blockC[rowBlock + 80];

    blockC += 16 * ldc;

    blockC[rowBlock + 0] = alpha * c0_32 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_32 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_32 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_32 + beta * blockC[rowBlock + 48];
    blockC[rowBlock + 64] = alpha * c64_32 + beta * blockC[rowBlock + 64];
    blockC[rowBlock + 80] = alpha * c80_32 + beta * blockC[rowBlock + 80];

    blockC += 16 * ldc;

    blockC[rowBlock + 0] = alpha * c0_48 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_48 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_48 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_48 + beta * blockC[rowBlock + 48];
    blockC[rowBlock + 64] = alpha * c64_48 + beta * blockC[rowBlock + 64];
    blockC[rowBlock + 80] = alpha * c80_48 + beta * blockC[rowBlock + 80];

    blockC += 16 * ldc;

    blockC[rowBlock + 0] = alpha * c0_64 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_64 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_64 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_64 + beta * blockC[rowBlock + 48];
    blockC[rowBlock + 64] = alpha * c64_64 + beta * blockC[rowBlock + 64];
    blockC[rowBlock + 80] = alpha * c80_64 + beta * blockC[rowBlock + 80];

    blockC += 16 * ldc;

    blockC[rowBlock + 0] = alpha * c0_80 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_80 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_80 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_80 + beta * blockC[rowBlock + 48];
    blockC[rowBlock + 64] = alpha * c64_80 + beta * blockC[rowBlock + 64];
    blockC[rowBlock + 80] = alpha * c80_80 + beta * blockC[rowBlock + 80];

    blockC += 16 * ldc;

  }

}

#else
#define SGEMMTHREADROWS 16
#define SGEMMTHREADCOLS 16
#define SGEMMROWREP 6
#define SGEMMCOLREP 6
#endif
