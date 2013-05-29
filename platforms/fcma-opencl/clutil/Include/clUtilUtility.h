#pragma once
#include "clUtilCommon.h"
#include "clUtilException.h"

namespace clUtil
{
  namespace Utility
  {
    std::vector<std::string> split(std::string str, char delimiter);
    cl_int fetchPlatformInfo(cl_platform_id platformID,
                             cl_platform_info paramName,
                             std::string& param);
    cl_int fetchDeviceInfo(cl_device_id deviceID,
                           cl_device_info paramName,
                           std::string& param);
    template <typename T> cl_int fetchDeviceInfo(cl_device_id deviceID,
                                                 cl_device_info paramName, 
                                                 T& param)
    {
      cl_int err;

      err = clGetDeviceInfo(deviceID, 
                            paramName,
                            sizeof(T), 
                            (void*)&param, 
                            NULL);
      clUtilCheckError(err);

      return CL_SUCCESS;
    }

    double getTime();
  
    bool directoryExists(const char* directoryName);
    bool fileExists(const char* fileName);
    bool makeDirectory(const char* directoryName);
  }
}
