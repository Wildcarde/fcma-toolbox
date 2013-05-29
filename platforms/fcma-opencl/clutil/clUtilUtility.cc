#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <exception>
#include "clUtilUtility.h"

using namespace std;

namespace clUtil
{
  namespace Utility
  {
    std::vector<std::string> split(std::string str, char delimiter)
    {
      std::istringstream strStream(str);
      std::vector<std::string> fields;

      while(strStream.good() == true)
      {
        std::string currentField;

        getline(strStream, currentField, delimiter);

        fields.push_back(currentField);
      }

      return fields;
    }

    cl_int fetchDeviceInfo(cl_device_id deviceID,
                           cl_device_info paramName,
                           std::string& param)
    {
      size_t propertySize;
      cl_int err;

      err = clGetDeviceInfo(deviceID,
                            paramName,
                            0,
                            NULL,
                            &propertySize);
      clUtilCheckError(err);

      std::unique_ptr<char[]> propertyString(new char[propertySize]);

      err = clGetDeviceInfo(deviceID,
                            paramName,
                            propertySize,
                            propertyString.get(),
                            NULL);
      clUtilCheckError(err);

      param = std::string(propertyString.get(), 
                          strnlen(propertyString.get(), propertySize));

      return CL_SUCCESS;
    }

    cl_int fetchPlatformInfo(cl_platform_id platformID,
                             cl_platform_info paramName,
                             std::string& param)
    {
      cl_int err;
      size_t paramSize;

      err = clGetPlatformInfo(platformID, paramName, 0, NULL, &paramSize);
      clUtilCheckError(err);

      std::unique_ptr<char[]> paramString(new char[paramSize]);

      err = clGetPlatformInfo(platformID,
                              paramName,
                              paramSize,
                              paramString.get(),
                              NULL);
      clUtilCheckError(err);

      param = std::string(paramString.get(), paramSize);

      return CL_SUCCESS;
    }

    double getTime()
    {
#if defined(__WIN32) || defined(WIN32)
      FILETIME fileTime;
      uint64_t result = 0;

      GetSystemTimeAsFileTime(&fileTime);
      result |= fileTime.dwHighDateTime;
      result <<= 32;
      result |= fileTime.dwLowDateTime;

      return result / 1e7;
#else
      struct timeval time;

      gettimeofday(&time, NULL);
      return (double)time.tv_sec + (double)time.tv_usec / 1e6;
#endif
    }

    bool directoryExists(const char* directoryName) 
    {
#if defined(__WIN32) || defined(WIN32)
      DWORD fileType = GetFileAttributes(directoryName);

      return fileType != INVALID_FILE_ATTRIBUTES && 
             fileType & FILE_ATTRIBUTE_DIRECTORY;
#else
      struct stat autotuningDir;

      int err = stat(directoryName, &autotuningDir);

      return err == 0 && S_ISDIR(autotuningDir.st_mode);
#endif
    }

    bool fileExists(const char* fileName)
    {
#if defined(__WIN32) || defined(WIN32)
      DWORD fileType = GetFileAttributes(fileName);

      return fileType != INVALID_FILE_ATTRIBUTES && 
             fileType  & FILE_ATTRIBUTE_NORMAL;
#else
      struct stat fileStat;

      return stat(fileName, &fileStat) == 0 && S_ISREG(fileStat.st_mode);
#endif
    }

    bool makeDirectory(const char* directoryName)
    {
#if defined(__WIN32) || defined(WIN32)
      return CreateDirectory(directoryName, NULL) != TRUE;
#else
      return mkdir(directoryName,  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1;
#endif
    }
  }
}
