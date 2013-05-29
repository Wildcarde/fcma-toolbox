/*
 * opencl_platform.h
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

#pragma once

#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
    
unsigned int getPlatformNum();
unsigned int getDeviceNum(cl_platform_id platformId, cl_device_type type);

char *getPlaformInformationString(cl_platform_id platformId,
		cl_platform_info info);
char *getPlatformName(cl_platform_id platformId);
char *getPlatformVersion(cl_platform_id platformId);
char *getPlatformVendor(cl_platform_id platformId);
char *getPlatformProfile(cl_platform_id platformId);
char *getPlatformExtensions(cl_platform_id platformId);

void printPlatformInformation(cl_platform_id platformId, FILE *output,
		const char *leading_comment);

#ifdef __cplusplus
}
#endif

