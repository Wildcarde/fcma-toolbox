#include <clUtilPlatform.h>

using namespace clUtil;
using namespace std;

PlatformInfo::PlatformInfo(cl_platform_id platformID)
{
  Utility::fetchPlatformInfo(platformID, CL_PLATFORM_PROFILE, Profile);
  Utility::fetchPlatformInfo(platformID, CL_PLATFORM_VERSION, Version);
  Utility::fetchPlatformInfo(platformID, CL_PLATFORM_NAME, Name);
  Utility::fetchPlatformInfo(platformID, CL_PLATFORM_VENDOR, Vendor);
  
  string extensionsString;

  Utility::fetchPlatformInfo(platformID, 
                             CL_PLATFORM_EXTENSIONS, 
                             extensionsString);

  Extensions = Utility::split(extensionsString, ' ');
}

Platform::Platform(cl_platform_id platformID) :
  mPlatformID(platformID),
  mPlatformInfo(platformID)
{
}

vector<Platform> Platform::GetPlatforms()
{
  cl_int err;
  cl_uint numPlatforms;
  vector<Platform> usablePlatforms;

  //Get the number of platforms and allocate a buffer to hold the platforms
  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  clUtilCheckError(err);

  unique_ptr<cl_platform_id[]> platforms(new cl_platform_id[numPlatforms]);

  //Actually get the platforms
  clGetPlatformIDs(numPlatforms, platforms.get(), NULL);
  clUtilCheckError(err);

  for(size_t curPlatformID = 0; curPlatformID < numPlatforms; curPlatformID++)
  {
    Platform curPlatform(platforms.get()[curPlatformID]);
    unsigned int majorVersion;
    unsigned int minorVersion;

    vector<string>&& versionStrings = 
      Utility::split(curPlatform.mPlatformInfo.Version, ' ');

    if(sscanf(versionStrings[1].c_str(), 
              "%u.%u", 
              &majorVersion, 
              &minorVersion) != 2)
    {
      printf("Warning: platform %u (%s) has malformed platform version string."
             " As such, clUtil will not use it.\n", 
             (unsigned int)curPlatformID, 
             curPlatform.mPlatformInfo.Name.c_str());
      continue;
    }
    
    if(majorVersion < 1 || (majorVersion == 1 && minorVersion < 1))
    {
      printf("Warning: platform %u (%s) has version < 1.1."
             " As such, clUtil will not use it.\n", 
             (unsigned int)curPlatformID, 
             curPlatform.mPlatformInfo.Name.c_str());
      continue;
    }

    usablePlatforms.push_back(curPlatform);
  }

  return usablePlatforms;
}

vector<Device> Platform::getDevices()
{
  cl_uint numDevices;
  cl_int err;
  vector<Device> platformDevices;

  //Get the number of devices in the platform and allocate a buffer to hold
  //all the devices
  err = clGetDeviceIDs(mPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
  clUtilCheckError(err);

  unique_ptr<cl_device_id[]> devices(new cl_device_id[numDevices]);

  err = clGetDeviceIDs(mPlatformID, 
                       CL_DEVICE_TYPE_ALL, 
                       numDevices,
                       devices.get(), 
                       NULL);
  clUtilCheckError(err);

  for(cl_uint curDevice = 0; curDevice < numDevices; curDevice++)
  {
    platformDevices.push_back(Device(devices.get()[curDevice]));
  }

  return platformDevices;
}

