/*
 * opencl_device.c
 *
 *  Created on: Apr 25, 2011
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

#include "opencl_device.h"

void _printDeviceInformationString(cl_device_id deviceId, cl_device_info info,
		char *string)
{
	char str[256];
	size_t size;
	cl_int ret = clGetDeviceInfo(deviceId, info, 256, &str, &size);
	if (ret == 0)
	{
		fprintf(stdout, "%s: %s\n", string, str);
	}
}

void _printDeviceInformationInt(cl_device_id deviceId, cl_device_info info,
		char *string)
{
	int i;
	size_t size;
	cl_int ret = clGetDeviceInfo(deviceId, info, sizeof(i), &i, &size);
	if (ret == 0)
	{
		fprintf(stdout, "%s: %i\n", string, i);
	}
}

void _printDeviceInformationLongInt(cl_device_id deviceId, cl_device_info info,
		char *string)
{
	long int i;
	size_t size;
	cl_int ret = clGetDeviceInfo(deviceId, info, sizeof(i), &i, &size);
	if (ret == 0)
	{
		fprintf(stdout, "%s: %li\n", string, i);
	}
}

void _printDeviceType(cl_device_id deviceId, char *string)
{
	char str[256];
	size_t size;
	cl_int ret = clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, 256, &str, &size);
	if (ret == 0)
	{
		if (*str == CL_DEVICE_TYPE_CPU)
		{
			fprintf(stdout, "%s: CPU\n", string);
		}
		else if (*str == CL_DEVICE_TYPE_GPU)
		{
			fprintf(stdout, "%s: GPU\n", string);
		}
		else if (*str == CL_DEVICE_TYPE_ACCELERATOR)
		{
			fprintf(stdout, "%s: ACCELERATOR\n", string);
		}
		else
		{
			fprintf(stdout, "%s: NOT_RECOGNIZED\n", string);
		}
	}
}

void printDeviceInformation(cl_device_id deviceId)
{
	_printDeviceInformationString(deviceId, CL_DEVICE_NAME, "Name");
	_printDeviceType(deviceId, "Type");
	_printDeviceInformationString(deviceId, CL_DEVICE_VENDOR, "Vendor");
	_printDeviceInformationString(deviceId, CL_DEVICE_VERSION, "Version");
	_printDeviceInformationString(deviceId, CL_DEVICE_PROFILE, "Profile");
	_printDeviceInformationString(deviceId, CL_DEVICE_EXTENSIONS, "Extensions");
	_printDeviceInformationInt(deviceId, CL_DEVICE_AVAILABLE, "Available");
	_printDeviceInformationInt(deviceId, CL_DEVICE_COMPILER_AVAILABLE,
			"Compiler Available");

	_printDeviceInformationInt(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS,
			"Maximum compute units");

	_printDeviceInformationInt(deviceId, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
			"Cache Type");
	_printDeviceInformationInt(deviceId, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
			"Cache Size");
	_printDeviceInformationLongInt(deviceId,
			CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "Cacheline Size");
	_printDeviceInformationLongInt(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE,
			"Global Memory Size");
	_printDeviceInformationInt(deviceId, CL_DEVICE_LOCAL_MEM_TYPE,
			"Local Memory Type");
	_printDeviceInformationLongInt(deviceId, CL_DEVICE_LOCAL_MEM_SIZE,
			"Local Memory Size");
}

cl_uint getNumberOfCPUS(cl_device_id device)
{
	cl_uint numCPU;
	cl_int ret = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 1,
			&numCPU, NULL);
	if (ret != CL_SUCCESS)
	{
		return 0;
	}
	return numCPU;
}

