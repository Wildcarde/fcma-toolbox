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

__kernel void sgemm(const int colMajor, const int transa, const int transb,
		const unsigned int m, const unsigned int n, const unsigned int k,
		const float alpha, __global float* a, const int lda, __global float* b,
		const int ldb, const float beta, __global float* c, const int ldc,
		const unsigned int submatrixSize, __local float* bufferA,
		__local float* bufferB)
{
	const int row = get_global_id(0);
	const int col = get_global_id(1);

	const int localRow = get_local_id(0);
	const int localCol = get_local_id(1);

	const int transc = 0;
	FloatMatrix A =
	{	m, k, colMajor, transa, lda, a};
	FloatMatrix B =
	{	k, n, colMajor, transb, ldb, b};
	FloatMatrix C =
	{	m, n, colMajor, transc, ldc, c};

	float r = 0.0;
	for (int i = 0; i < k; i += submatrixSize)
	{
		bufferA[localCol * submatrixSize + localRow] = getMatrixFloat(A, row,
				i + localCol);
		bufferB[localCol * submatrixSize + localRow] = getMatrixFloat(B,
				i + localRow, col);

		/*
		 * Synchronize to make sure the sub-matrices are loaded
		 * before starting the computation
		 */
		barrier(CLK_LOCAL_MEM_FENCE);

		/* #pragma unroll */
		for (int j = 0; j < submatrixSize; ++j)
		{
			r += bufferA[j * submatrixSize + localRow] * bufferB[localCol
			* submatrixSize + j];
		}

		/*
		 * Synchronize to make sure that the preceding
		 * computation is done before loading two new
		 * sub-matrices of A and B in the next iteration
		 */
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (beta == 0.0)
	{
		setMatrixFloat(C, row, col, alpha * r);
	}
	else
	{
		setMatrixFloat(C, row, col,
				alpha * r + beta * getMatrixFloat(C, row, col));
	}
}
