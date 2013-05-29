#include "clUtilDeviceGroup.h"

using namespace std;
using namespace clUtil;

void DeviceGroupInfo::initialize()
{
  const vector<Device>& devices = Device::GetDevices();
  map<string, size_t> deviceGroupNames;

  for(auto curDevice = devices.begin(); curDevice < devices.end(); curDevice++)
  {
    string curDeviceName = (*curDevice).getDeviceInfo().Name;
    auto deviceGroupQuery = deviceGroupNames.find(curDeviceName);

    //If we haven't found this device yet, add it and increment the group 
    //counter
    if(deviceGroupQuery == deviceGroupNames.end())
    {
      deviceGroupNames[curDeviceName] = mNumDeviceGroups;
      mDeviceGroups.push_back(mNumDeviceGroups);
      mNumDeviceGroups++;
    }
    else //Otherwise, put the group ID into the vector
    {
      mDeviceGroups.push_back(deviceGroupQuery->second);
    }
  }
}
