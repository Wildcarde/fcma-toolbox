#include "cblas.h"

float cblas_sdot(blasint n, float  *x, blasint incx, float  *y, blasint incy)
{
	return 1.0;
}

void cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB, blasint M, blasint N, blasint K,
		 float alpha, float *A, blasint lda, float *B, blasint ldb, float beta, float *C, blasint ldc)
{
}