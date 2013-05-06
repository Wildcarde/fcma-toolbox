#include "common.h"
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

int getBuf(int start_col, int end_col, int row, int col, double* mat, float* buf);
CorrMatrix* CorrMatrixComputation(Trial trial, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2);
CorrMatrix** ComputeAllTrialsCorrMatrices(Trial* trials, int nTrials, int sr, int step, RawMatrix** matrices1, RawMatrix** matrices2);