#pragma once
#include "clUtilCommon.h"
#include "clUtilDevice.h"

namespace clUtil
{
  struct PlatformInfo
  {
    public:
      std::string Profile;
      std::string Version;
      std::string Name;
      std::string Vendor;
      std::vector<std::string> Extensions;

      PlatformInfo(cl_platform_id platformID);
    private:
      PlatformInfo();
  };

  class Platform
  {
    public:
      Platform(cl_platform_id platformID);
      std::vector<Device> getDevices();

      static std::vector<Platform> GetPlatforms();

    private:
      cl_platform_id mPlatformID;
      PlatformInfo mPlatformInfo;
  };
}
