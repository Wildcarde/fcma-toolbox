#pragma once
#include "clUtilException.h"
#include "clUtilDevice.h"
#include "clUtilCommon.h"
#include "clUtilUtility.h"

#define kCLUtilMaxDevices 64
#define kCLUtilMaxKernelNameLength 128

//Timing
double clUtilGetTime();
