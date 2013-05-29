#include <clUtil.h>
#include <iostream>

using namespace clUtil;
using namespace std;

int main(int argc, char** argv)
{
  Device::FetchDevices();
  size_t curDeviceNum = 0;

  for(size_t currentDevice = 0; 
      currentDevice < Device::GetDevices().size();
      currentDevice++)
  {
    const Device& curDevice = Device::GetDevices()[currentDevice];
    const DeviceInfo& deviceInfo = curDevice.getDeviceInfo();

    cout << "Device " << curDeviceNum++ << endl;
    cout << "\tName:" << deviceInfo.Name << endl;
    cout << "\tVendor:" << deviceInfo.Vendor << endl;
    cout << "\tDriver:" << deviceInfo.DriverVersion << endl;
    cout << "\tProfile:" << deviceInfo.OpenCLProfile << endl;
    cout << "\tVersion:" << deviceInfo.OpenCLVersion << endl;
    cout << "\tCVersion:" << deviceInfo.OpenCLCVersion << endl;
    cout << "\tNum Compute Units:" << deviceInfo.MaxComputeUnits << endl;
    cout << "\tMax Frequency:" << deviceInfo.MaxClockFrequency << endl;
  }

  return 0;
}
