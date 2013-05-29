/*
 * opencl_kernel.c
 *
 *  Created on: May 2, 2011
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

#include "opencl_kernel.h"

void _printKernelWorkGroupInformationSizeT(cl_kernel kernel,
		cl_device_id deviceId, cl_device_info info, char *string)
{
	size_t result;
	size_t size;
	cl_int ret = clGetKernelWorkGroupInfo(kernel, deviceId, info,
			sizeof(result), &result, &size);
	if (ret == CL_SUCCESS)
	{
		fprintf(stdout, "%s: %i\n", string, (int) result);
	}
	else
	{
		printErrorMessage("clGetKernelWorkGroupInfo", __FILE__, __LINE__, ret);
	}
}

void _printKernelWorkGroupInformationSizeTArray(cl_kernel kernel,
		cl_device_id deviceId, cl_device_info info, char *string,
		unsigned int length)
{
	size_t *result = malloc(length * sizeof(size_t));
	size_t size;
	cl_int ret = clGetKernelWorkGroupInfo(kernel, deviceId, info, length
			* sizeof(*result), result, &size);
	if (ret == CL_SUCCESS)
	{
		fprintf(stdout, "%s\n", string);
		for (unsigned int i = 0; i < length; ++i)
		{
			fprintf(stdout, "\t%i\n", (int) result[i]);
		}
	}
	else
	{
		printErrorMessage("clGetKernelWorkGroupInfo", __FILE__, __LINE__, ret);
	}
	free(result);
}

void _printKernelWorkGroupInformationULong(cl_kernel kernel,
		cl_device_id deviceId, cl_device_info info, char *string)
{
	cl_ulong result;
	size_t size;
	cl_int ret = clGetKernelWorkGroupInfo(kernel, deviceId, info,
			sizeof(result), &result, &size);
	if (ret == CL_SUCCESS)
	{
		fprintf(stdout, "%s: %lli\n", string, result);
	}
	else
	{
		printErrorMessage("clGetKernelWorkGroupInfo", __FILE__, __LINE__, ret);
	}
}

void printKernelWorkGroupInformation(cl_kernel kernel, cl_device_id deviceId)
{
	_printKernelWorkGroupInformationSizeT(kernel, deviceId,
			CL_KERNEL_WORK_GROUP_SIZE, "Work group size");
	_printKernelWorkGroupInformationSizeTArray(kernel, deviceId,
			CL_KERNEL_COMPILE_WORK_GROUP_SIZE, "Compile work group size", 3);
	_printKernelWorkGroupInformationULong(kernel, deviceId,
			CL_KERNEL_LOCAL_MEM_SIZE, "Local mem size");
	_printKernelWorkGroupInformationSizeT(kernel, deviceId,
			CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
			"Preferred work group size multiple");
	_printKernelWorkGroupInformationULong(kernel, deviceId,
			CL_KERNEL_PRIVATE_MEM_SIZE, "Private memory size");
}
