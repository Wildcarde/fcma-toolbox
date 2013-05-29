/*
 * opencl_blas3_gemm.h
 *
 *  Created on: Apr 15, 2011
 *      Author: ludwig
 */

/*
   Copyright 2011 PureSol Technologies

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

OpenCLStatus opencl_sgemm(const MatrixOrder order,
		const TransposeMatrix transposeA, const TransposeMatrix transposeB,
		const int m, const int n, const int k, const float alpha, float *a,
		const int lda, float *b, const int ldb, const float beta, float *c,
		const int ldc);

OpenCLStatus opencl_dgemm(const MatrixOrder order,
		const TransposeMatrix transposeA, const TransposeMatrix transposeB,
		const int m, const int n, const int k, const double alpha, double *a,
		const int lda, double *b, const int ldb, const double beta, double *c,
		const int ldc);

OpenCLStatus opencl_cgemm(const MatrixOrder order,
		const TransposeMatrix transposeA, const TransposeMatrix transposeB,
		const int m, const int n, const int k, const void *alpha, void *a,
		const int lda, void *b, const int ldb, const void *beta, void *c,
		const int ldc);

OpenCLStatus opencl_zgemm(const MatrixOrder order,
		const TransposeMatrix transposeA, const TransposeMatrix transposeB,
		const int m, const int n, const int k, const void *alpha, void *a,
		const int lda, void *b, const int ldb, const void *beta, void *c,
		const int ldc);

