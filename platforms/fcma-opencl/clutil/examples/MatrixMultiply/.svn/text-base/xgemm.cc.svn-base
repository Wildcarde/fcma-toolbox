#include <memory>
#include "blas.h"

using namespace clUtil;
using namespace std;

void transpose(int m, int n, float* a, float* b)
{
  for(int i = 0; i < m; i++)
  {
    for(int j = 0; j < n; j++)
    {
      b[i * n + j] = a[j * m + i];
    }
  }
}

void dgemm_reference(char* transA,
                     char* transB,
                     int m,
                     int n,
                     int k,
                     double alpha,
                     Buffer& a,
                     int lda,
                     Buffer& b,
                     int ldb,
                     double beta,
                     Buffer& c,
                     int ldc)
{
  clUtilEnqueueKernel("dgemmNN_reference",
                      clUtilGrid(m, 8, n, 8),
                      m, 
                      n,
                      k, 
                      alpha, 
                      a, 
                      lda,
                      b, 
                      ldb, 
                      beta, 
                      c,
                      ldc);
  
}

void dgemm(char* transA,
           char* transB,
           int m,
           int n,
           int k,
           double alpha,
           Buffer& a,
           int lda,
           Buffer& b,
           int ldb,
           double beta,
           Buffer& c,
           int ldc)
{
  clUtilEnqueueKernel("dgemmNN",
                      clUtilGrid(m / DGEMMROWREP, 
                                 DGEMMTHREADROWS, 
                                 n / DGEMMCOLREP, 
                                 DGEMMTHREADCOLS),
                      m,
                      n,
                      k,
                      alpha,
                      a,
                      lda,
                      b,
                      ldb,
                      beta,
                      c,
                      ldc);

}


void sgemm_reference(char* transA,
                     char* transB,
                     int m,
                     int n,
                     int k,
                     float alpha,
                     Buffer& a,
                     int lda,
                     Buffer& b,
                     int ldb,
                     float beta,
                     Buffer& c,
                     int ldc)
{
  if(*transA == 'N' && *transB == 'N')
  {
    clUtilEnqueueKernel("sgemmNN_reference",
                        clUtilGrid(m, 8, n, 8),
                        m,
                        n,
                        k,
                        alpha,
                        a,
                        lda,
                        b,
                        ldb,
                        beta,
                        c,
                        ldc);
  }

}

void sgemm(char* transA,
           char* transB,
           int m,
           int n,
           int k,
           float alpha,
           Buffer& a,
           int lda,
           Buffer& b,
           int ldb,
           float beta,
           Buffer& c,
           int ldc)
{

  if(*transA == 'N' && *transB == 'N')
  {
    clUtilEnqueueKernel("sgemmNN",
                        clUtilGrid(m / SGEMMROWREP, 
                                   SGEMMTHREADROWS, 
                                   n / SGEMMCOLREP, 
                                   SGEMMTHREADCOLS),
                        m,
                        n,
                        k,
                        alpha,
                        a,
                        lda,
                        b,
                        ldb,
                        beta,
                        c,
                        ldc);
  }
}
