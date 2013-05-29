/*
 * opencl.h
 *
 *  Created on: Apr 22, 2011
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

#include <string.h>
#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "opencl_datautils.h"
#include "opencl_utils.h"
#include "opencl_platform.h"
#include "opencl_device.h"
#include "opencl_kernel.h"
#include "opencl_types.h"
#include "opencl_output.h"

cl_uint numPlatforms;
cl_platform_id *platformIds;

cl_uint *numDevices;
cl_device_id **deviceIds;

void getErrorMessage(cl_int returnValue, size_t maxLength, char *msg);

OpenCLStatus
opencl_initPlatform();

OpenCLStatus
opencl_dropPlatform();

