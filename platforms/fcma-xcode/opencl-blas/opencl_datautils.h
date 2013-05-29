/*
 * opencl_datautils.h
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

#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "opencl_types.h"

#ifdef __cplusplus
extern "C" {
#endif
    
/**
 * This function copies a junk of data which contains a matrix into the GPU.
 * The buffer needed for this is created and the amount of memory to be copied
 * is calculated by the matrix dimensions.
 */
cl_int createAndCopyMatrixFloat(cl_context context, MatrixOrder matrixOrder,
		TransposeMatrix transpose, unsigned int m, unsigned int n,
		cl_float *data, unsigned int stride, cl_mem_flags flags, cl_mem *mem);

/**
 * This function copies a junk of data which contains a matrix into the GPU.
 * The buffer needed for this is created and the amount of memory to be copied
 * is calculated by the matrix dimensions.
 */
cl_int createAndCopyMatrixDouble(cl_context context, MatrixOrder matrixOrder,
		TransposeMatrix transpose, unsigned int m, unsigned int n,
		cl_double *data, unsigned int stride, cl_mem_flags flags, cl_mem *mem);

#ifdef __cplusplus
}
#endif
