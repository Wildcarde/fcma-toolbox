#include <clUtil.h>
#include <iostream>
#include <memory>

using namespace clUtil;
using namespace std;

const size_t kBigArraySize = 12345;

int main(int argc, char** argv)
{
  const char* filename[] = {"kernel.cl"};

  unique_ptr<float[]> a(new float[kBigArraySize]);
  unique_ptr<float[]> b(new float[kBigArraySize]);
  unique_ptr<float[]> c(new float[kBigArraySize]);

  for(size_t i = 0; i < kBigArraySize; i++)
  {
    a[i] = (float)i;
    b[i] = (float)i;
  }

  try
  {
    Device::InitializeDevices(filename, 1);

    Device::StartProfiling();
 
    size_t numDevices = Device::GetDevices().size();

    vector<unique_ptr<Buffer>> aDevice(numDevices);
    vector<unique_ptr<Buffer>> bDevice(numDevices);
    vector<unique_ptr<Buffer>> cDevice(numDevices);

    for(size_t currentDevice = 0; currentDevice < numDevices; currentDevice++)
    {
      Device::SetCurrentDevice(currentDevice);

      unique_ptr<Buffer> tmp1(new Buffer(sizeof(float) * kBigArraySize));
      aDevice[currentDevice] = move(tmp1);
      unique_ptr<Buffer> tmp2(new Buffer(sizeof(float) * kBigArraySize));
      bDevice[currentDevice] = move(tmp2);
      unique_ptr<Buffer> tmp3(new Buffer(sizeof(float) * kBigArraySize));
      cDevice[currentDevice] = move(tmp3);
    }

    ParallelFor(0, 1, kBigArraySize - 1, [&](size_t startIdx, size_t endIdx)
    { 
#if 0
      cout << "Device " << Device::GetCurrentDeviceNum()
           << " Start " << startIdx
           << " End " << endIdx << endl;
#else
#if 0
      unsigned int indexCount = endIdx - startIdx + 1;

      size_t deviceNum = Device::GetCurrentDeviceNum();
      
      aDevice[deviceNum]->put(&a[startIdx], indexCount * sizeof(float));
      aDevice[deviceNum]->get(&c[startIdx], indexCount * sizeof(float));
#else
      unsigned int indexCount = endIdx - startIdx + 1;

      size_t deviceNum = Device::GetCurrentDeviceNum();

      aDevice[deviceNum]->put(&a[startIdx], indexCount * sizeof(float));
      bDevice[deviceNum]->put(&b[startIdx], indexCount * sizeof(float));

      clUtilEnqueueKernel("vectorAdd", 
                          clUtilGrid(indexCount, 64),
                          *aDevice[deviceNum],
                          *bDevice[deviceNum],
                          *cDevice[deviceNum],
                          indexCount);

      cDevice[deviceNum]->get(&c[startIdx], indexCount * sizeof(float));
#endif
#endif
    },
    PINAScheduler("loop"));

    Device::DumpProfilingData();
  }
  catch(clUtilException& err)
  {
    cout << err.what() << endl;
  }

  for(unsigned int i = 0; i < kBigArraySize; i++)
  {
    if(c[i] != 2.0f * (float)i)
    {
      cout << "Element " << i << " is incorrect. Should be " 
           << 2.0f * (float)i << ". Is " << c[i] << endl;

      exit(1);
    }
  }

  cout << "Success!" << endl;

  return 0;
}
