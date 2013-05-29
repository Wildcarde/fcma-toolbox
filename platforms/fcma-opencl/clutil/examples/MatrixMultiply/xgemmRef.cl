#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void dgemmNN_reference(int m,
                                int n,
                                int k,
                                double alpha,
                                __global double* a,
                                int lda,
                                __global double* b,
                                int ldb,
                                double beta,
                                __global double* c,
                                int ldc)
{
  int row = get_global_id(0);
  int col = get_global_id(1);

  double cVal = 0.0;

  if(row < m && col < n)
  {
    for(int i = 0; i < k; i++)
    {
      cVal = fma(a[row + i * lda], b[i + col * ldb], cVal);
    }

    c[row + col * ldc] = alpha * cVal + beta * c[row + col * ldc];
  }
}

__kernel void sgemmNN_reference(int m,
                                int n,
                                int k,
                                float alpha,
                                __global float* a,
                                int lda,
                                __global float* b,
                                int ldb,
                                float beta,
                                __global float* c,
                                int ldc)
{
  int row = get_global_id(0);
  int col = get_global_id(1);

  float cVal = 0.0;

  if(row < m && col < n)
  {
    for(int i = 0; i < k; i++)
    {
      cVal = fma(a[row + i * lda], b[i + col * ldb], cVal);
    }

    c[row + col * ldc] = alpha * cVal + beta * c[row + col * ldc];
  }
}
