/*
 * opencl_datautils.c
 *
 *  Created on: Jun 17, 2011
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

#include "opencl_datautils.h"

cl_int createAndCopyMatrix(cl_context context, unsigned int size,
		void *data, cl_mem_flags flags, cl_mem *mem)
{
	cl_int ret;
	if (data != NULL)
	{
		flags |= CL_MEM_COPY_HOST_PTR;
	}
	*mem = clCreateBuffer(context, flags, size, data, &ret);
	return ret;
}

cl_int createAndCopyMatrixFloat(cl_context context,
		MatrixOrder matrixOrder, TransposeMatrix transpose, unsigned int m,
		unsigned int n, cl_float *data, unsigned int stride,
		cl_mem_flags flags, cl_mem *mem)
{
	unsigned int size = sizeof(cl_float);
	if (matrixOrder == ColumnMajor)
	{
		if (transpose == NoTranspose)
		{
			size *= stride * n;
		}
		else
		{
			size *= stride * m;
		}
	}
	else
	{
		if (transpose == NoTranspose)
		{
			size *= stride * m;
		}
		else
		{
			size *= stride * n;
		}
	}
	return createAndCopyMatrix(context, size, data, flags, mem);
}

cl_int createAndCopyMatrixDouble(cl_context context,
		MatrixOrder matrixOrder, TransposeMatrix transpose, unsigned int m,
		unsigned int n, cl_double *data, unsigned int stride,
		cl_mem_flags flags, cl_mem *mem)
{
	unsigned int size = sizeof(cl_double);
	if (matrixOrder == ColumnMajor)
	{
		if (transpose == NoTranspose)
		{
			size *= stride * n;
		}
		else
		{
			size *= stride * m;
		}
	}
	else
	{
		if (transpose == NoTranspose)
		{
			size *= stride * m;
		}
		else
		{
			size *= stride * n;
		}
	}
	return createAndCopyMatrix(context, size, data, flags, mem);
}
