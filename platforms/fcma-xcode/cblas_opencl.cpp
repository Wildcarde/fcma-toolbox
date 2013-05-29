//
//  cblas_opencl.cpp
//  fcma-xcode
//
//  Created by Benjamin Singer on 5/20/13.
//  Copyright (c) 2013 pni. All rights reserved.
//

#include <OpenCL/OpenCL.h>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include "cblas.h"
#include "opencl_blas.h"
#include "gemm.h"

float cblas_sdot(blasint n, float  *x, blasint incx, float  *y, blasint incy)
{
	return 1.0;
}

void cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB, blasint m, blasint n, blasint k, float alpha, float *a, blasint lda, float *b, blasint ldb, float beta, float *c, blasint ldc)
{
    MatrixOrder order = static_cast<MatrixOrder>( Order - CblasRowMajor );
    TransposeMatrix transposeA = static_cast<TransposeMatrix>( TransA - CblasNoTrans );
    TransposeMatrix transposeB = static_cast<TransposeMatrix>( TransB - CblasNoTrans );
    
	cl_int ret;
	OpenCLStatus status = opencl_failure;
    
	cl_device_id usedDevice = deviceIds[1][0];
    cl_command_queue usedCommandQueue;
    cl_program program;
    cl_context context;
    
    cl_mem memA = NULL;
    cl_mem memB = NULL;
    cl_mem memC = NULL;
    
    try
    {
        context = clCreateContext(NULL, 1, &usedDevice, NULL, NULL, &ret);
        ERROR_HANDLER(ret);
        
        usedCommandQueue = clCreateCommandQueue(context, usedDevice, 0, &ret);
        ERROR_HANDLER(ret);
        
        /* Create Kernel Program from the source */
        
        const int SUBMATRIX_SIZE = 16;
        
        const char *sources[] =
        { "FloatMatrix.cl", "GetMatrixFloat.cl", "SetMatrixFloat.cl", "SGEMM.cl" };
        
        program = createProgramWithSourceStrings(context, 4, sources,
                                                            &ret);
        if (ret != CL_SUCCESS)
        {
            printPlatformInformation(platformIds[0], stderr, "");
            ERROR_HANDLER(ret);
        }
        
        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &usedDevice, NULL, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
            printProgramBuildInfo(ret, program, usedDevice);
            ERROR_HANDLER(ret);
        }
                
        cl_kernel kernel = clCreateKernel(program, "sgemm", &ret);
        ERROR_HANDLER(ret);
        
        ERROR_HANDLER(createAndCopyMatrixFloat(context, order, transposeA, m, k, a,
                                               lda, CL_MEM_READ_ONLY, &memA));
        ERROR_HANDLER(createAndCopyMatrixFloat(context, order, transposeB, k, n, b,
                                               ldb, CL_MEM_READ_ONLY, &memB));
        if (beta != 0.0)
        {
            ERROR_HANDLER(createAndCopyMatrixFloat(context, order, NoTranspose, m, n,
                                                   c, ldc, CL_MEM_READ_WRITE, &memC));
        }
        else
        {
            ERROR_HANDLER(createAndCopyMatrixFloat(context, order, NoTranspose, m, n,
                                                   NULL, ldc, CL_MEM_WRITE_ONLY, &memC));
        }
        
        /* Set OpenCL Kernel Arguments */
        int colMajor = order == ColumnMajor ? 1 : 0;
        int transa = transposeA == NoTranspose ? 0 : 1;
        int transb = transposeB == NoTranspose ? 0 : 1;
        ERROR_HANDLER(clSetKernelArg(kernel, 0, sizeof(int), (void *) &colMajor));
        ERROR_HANDLER(clSetKernelArg(kernel, 1, sizeof(int), (void *) &transa));
        ERROR_HANDLER(clSetKernelArg(kernel, 2, sizeof(int), (void *) &transb));
        ERROR_HANDLER(clSetKernelArg(kernel, 3, sizeof(int), (void *) &m));
        ERROR_HANDLER(clSetKernelArg(kernel, 4, sizeof(int), (void *) &n));
        ERROR_HANDLER(clSetKernelArg(kernel, 5, sizeof(int), (void *) &k));
        ERROR_HANDLER(clSetKernelArg(kernel, 6, sizeof(float), (void *) &alpha));
        ERROR_HANDLER(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *) &memA));
        ERROR_HANDLER(clSetKernelArg(kernel, 8, sizeof(int), (void *) &lda));
        ERROR_HANDLER(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *) &memB));
        ERROR_HANDLER(clSetKernelArg(kernel, 10, sizeof(int), (void *) &ldb));
        ERROR_HANDLER(clSetKernelArg(kernel, 11, sizeof(float), (void *) &beta));
        ERROR_HANDLER(clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *) &memC));
        ERROR_HANDLER(clSetKernelArg(kernel, 13, sizeof(int), (void *) &ldc));
        ERROR_HANDLER(clSetKernelArg(kernel, 14, sizeof(unsigned int), &SUBMATRIX_SIZE));
        ERROR_HANDLER(clSetKernelArg(kernel, 15, sizeof(cl_float) * SUBMATRIX_SIZE * SUBMATRIX_SIZE, NULL));
        ERROR_HANDLER(clSetKernelArg(kernel, 16, sizeof(cl_float) * SUBMATRIX_SIZE * SUBMATRIX_SIZE, NULL));
        
        /* Execute OpenCL Kernel */
        size_t global_item_size[2];
        global_item_size[0] = m % SUBMATRIX_SIZE == 0 ? m
        : (m / SUBMATRIX_SIZE + 1) * SUBMATRIX_SIZE;
        global_item_size[1] = n % SUBMATRIX_SIZE == 0 ? n
        : (n / SUBMATRIX_SIZE + 1) * SUBMATRIX_SIZE;
        size_t local_item_size[2];
        local_item_size[0] = SUBMATRIX_SIZE;
        local_item_size[1] = SUBMATRIX_SIZE;
        
        ret = clEnqueueNDRangeKernel(usedCommandQueue, kernel, 2, NULL,
                                     global_item_size, local_item_size, 0, NULL, NULL);
        ERROR_HANDLER(ret);
        
        /* Finalization */
        clFlush(usedCommandQueue);
        clFinish(usedCommandQueue);
        
        /* Copy results from the memory buffer */
        if (order == ColumnMajor)
        {
            ret = clEnqueueReadBuffer(usedCommandQueue, memC, CL_TRUE, 0,
                                      ldc * n * sizeof(float), c, 0, NULL, NULL);
        }
        else
        {
            ret = clEnqueueReadBuffer(usedCommandQueue, memC, CL_TRUE, 0,
                                      ldc * m * sizeof(float), c, 0, NULL, NULL);
        }
        ERROR_HANDLER(ret);
        printMatrixS(order, m, n, c, ldc);
        
        status = opencl_success;
    }
    catch (std::exception const& e) {
        std::cout << "Exception: " << e.what() << "\n";
    }

    clReleaseProgram(program);
    
	if (usedCommandQueue != NULL)
	{
		clReleaseCommandQueue(usedCommandQueue);
		usedCommandQueue = NULL;
	}
	if (context != NULL)
	{
		clReleaseContext(context);
		context = NULL;
	}
    
	if (memA != NULL)
	{
		clReleaseMemObject(memA);
	}
	if (memB != NULL)
	{
		clReleaseMemObject(memB);
	}
	if (memC != NULL)
	{
		clReleaseMemObject(memC);
	}
    
    assert(status == opencl_success);
}