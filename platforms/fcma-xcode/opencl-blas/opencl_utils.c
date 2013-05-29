/*
 * opencl_utils.c
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

#include "opencl_utils.h"

void getErrorMessage(cl_int returnValue, size_t maxLength, char *msg)
{
	switch (returnValue)
	{
	case CL_SUCCESS:
		strncpy(msg, "success", maxLength);
		break;
	case CL_DEVICE_NOT_FOUND:
		strncpy(msg, "device not found", maxLength);
		break;
	case CL_DEVICE_NOT_AVAILABLE:
		strncpy(msg, "device not available", maxLength);
		break;
	case CL_COMPILER_NOT_AVAILABLE:
		strncpy(msg, "compiler not available", maxLength);
		break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		strncpy(msg, "mem object allocation failure", maxLength);
		break;
	case CL_OUT_OF_RESOURCES:
		strncpy(msg, "out of resources", maxLength);
		break;
	case CL_OUT_OF_HOST_MEMORY:
		strncpy(msg, "out of host memory", maxLength);
		break;
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		strncpy(msg, "profiling info not available", maxLength);
		break;
	case CL_MEM_COPY_OVERLAP:
		strncpy(msg, "mem copy overlap", maxLength);
		break;
	case CL_IMAGE_FORMAT_MISMATCH:
		strncpy(msg, "image format mismatch", maxLength);
		break;
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		strncpy(msg, "image format not supported", maxLength);
		break;
	case CL_BUILD_PROGRAM_FAILURE:
		strncpy(msg, "build program failure", maxLength);
		break;
	case CL_MAP_FAILURE:
		strncpy(msg, "map failure", maxLength);
		break;
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		strncpy(msg, "misaligned sub buffer offset", maxLength);
		break;
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		strncpy(msg, "exec status error for events in wait list", maxLength);
		break;
		/* INVALIDS */
	case CL_INVALID_VALUE:
		strncpy(msg, "invalid value", maxLength);
		break;
	case CL_INVALID_DEVICE_TYPE:
		strncpy(msg, "invalid device type", maxLength);
		break;
	case CL_INVALID_PLATFORM:
		strncpy(msg, "invalid platform", maxLength);
		break;
	case CL_INVALID_DEVICE:
		strncpy(msg, "invalid device", maxLength);
		break;
	case CL_INVALID_CONTEXT:
		strncpy(msg, "invalid context", maxLength);
		break;
	case CL_INVALID_QUEUE_PROPERTIES:
		strncpy(msg, "invalid queue properties", maxLength);
		break;
	case CL_INVALID_COMMAND_QUEUE:
		strncpy(msg, "invalid command queue", maxLength);
		break;
	case CL_INVALID_HOST_PTR:
		strncpy(msg, "invalid host ptr", maxLength);
		break;
	case CL_INVALID_MEM_OBJECT:
		strncpy(msg, "invalid mem object", maxLength);
		break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		strncpy(msg, "invalid image format descriptor", maxLength);
		break;
	case CL_INVALID_IMAGE_SIZE:
		strncpy(msg, "invalid image size", maxLength);
		break;
	case CL_INVALID_SAMPLER:
		strncpy(msg, "invalid sampler", maxLength);
		break;
	case CL_INVALID_BINARY:
		strncpy(msg, "invalid binary", maxLength);
		break;
	case CL_INVALID_BUILD_OPTIONS:
		strncpy(msg, "invalid build options", maxLength);
		break;
	case CL_INVALID_PROGRAM:
		strncpy(msg, "invalid program", maxLength);
		break;
	case CL_INVALID_PROGRAM_EXECUTABLE:
		strncpy(msg, "invalid program executable", maxLength);
		break;
	case CL_INVALID_KERNEL_NAME:
		strncpy(msg, "invalid kernel name", maxLength);
		break;
	case CL_INVALID_KERNEL_DEFINITION:
		strncpy(msg, "invalid kernel definition", maxLength);
		break;
	case CL_INVALID_KERNEL:
		strncpy(msg, "invalid kernel", maxLength);
		break;
	case CL_INVALID_ARG_INDEX:
		strncpy(msg, "invalid arg index", maxLength);
		break;
	case CL_INVALID_ARG_VALUE:
		strncpy(msg, "invalid arg value", maxLength);
		break;
	case CL_INVALID_ARG_SIZE:
		strncpy(msg, "invalid arg size", maxLength);
		break;
	case CL_INVALID_KERNEL_ARGS:
		strncpy(msg, "invalud kernel args", maxLength);
		break;
	case CL_INVALID_WORK_DIMENSION:
		strncpy(msg, "invalid work dimension", maxLength);
		break;
	case CL_INVALID_WORK_GROUP_SIZE:
		strncpy(msg, "invalid work group size", maxLength);
		break;
	case CL_INVALID_WORK_ITEM_SIZE:
		strncpy(msg, "invalid work item size", maxLength);
		break;
	case CL_INVALID_GLOBAL_OFFSET:
		strncpy(msg, "invalid global offset", maxLength);
		break;
	case CL_INVALID_EVENT_WAIT_LIST:
		strncpy(msg, "invalid event wait list", maxLength);
		break;
	case CL_INVALID_EVENT:
		strncpy(msg, "invalid event", maxLength);
		break;
	case CL_INVALID_OPERATION:
		strncpy(msg, "invalid operation", maxLength);
		break;
	case CL_INVALID_GL_OBJECT:
		strncpy(msg, "invalid GL object", maxLength);
		break;
	case CL_INVALID_BUFFER_SIZE:
		strncpy(msg, "invalid buffer size", maxLength);
		break;
	case CL_INVALID_MIP_LEVEL:
		strncpy(msg, "invalid MIP level", maxLength);
		break;
	case CL_INVALID_GLOBAL_WORK_SIZE:
		strncpy(msg, "invalid global work size", maxLength);
		break;
	case CL_INVALID_PROPERTY:
		strncpy(msg, "invalid property", maxLength);
		break;

	default:
		strncpy(msg, "<unknown message>", maxLength);
		break;
	}
}

void printErrorMessage(const char *str, const char *file, int line, cl_int ret)
{
	char message[0x10000];
	getErrorMessage(ret, 0x10000, message);
	fprintf(stderr, "%s (%s:%i)\nmessage: %s\n", str, file, line, message);
}

void printProgramBuildInfo(cl_int ret, cl_program program, cl_device_id device)
{
	char buf[0x10000];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, buf,
			NULL);
	if (ret == CL_SUCCESS)
	{
		fprintf(stdout, "build info: \"%s\"\n", buf);
	}
	else
	{
		fprintf(stderr, "build error: \"%s\"\n", buf);
	}
}

cl_program createProgramWithSourceStrings(cl_context context,
		unsigned int numStrings, const char **strings, cl_int *ret)
{
	size_t *sizes = malloc(sizeof(size_t) * numStrings);
	for (unsigned int id = 0; id < numStrings; ++id)
	{
		sizes[id] = strlen(strings[id]);
	}
	return clCreateProgramWithSource(context, numStrings, strings, sizes, ret);
}
