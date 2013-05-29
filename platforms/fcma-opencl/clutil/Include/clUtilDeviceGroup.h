#pragma once
#include <map>
#include <vector>
#include "clUtilDevice.h"

namespace clUtil
{
  class DeviceGroupInfo
  {
    private:
      std::vector<size_t> mDeviceGroups;
      std::size_t mNumDeviceGroups;

      static DeviceGroupInfo deviceGroupInfoSingleton;
      static bool singletonInitialized;

      DeviceGroupInfo() : mNumDeviceGroups(0)
      {
      }
      
      void initialize();
    public:
      const size_t operator[](size_t i) const
      {
        return mDeviceGroups[i];
      }

      size_t numGroups() const
      {
        return mNumDeviceGroups;
      }

      static const DeviceGroupInfo& Get() 
      {
        if(singletonInitialized == false)
        {
          deviceGroupInfoSingleton.initialize();
          singletonInitialized = true;
        }

        return deviceGroupInfoSingleton;
      }
  };
}
