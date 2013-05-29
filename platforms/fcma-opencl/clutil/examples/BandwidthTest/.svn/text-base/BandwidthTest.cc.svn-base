#include <clUtil.h>

using namespace clUtil;

int main(int argc, char** argv)
{
  char const* kernel = "kernel.cl";
  double time1;
  double time2;
  float* a;
  const unsigned int size = 1024*1024*20;

  Device::InitializeDevices(&kernel, 1);

  a = new float[size];
  
  for(unsigned int i = 0; i < size; i++)
  {
    a[i] = 0.0f;
  }

  Buffer aGPU(sizeof(a[0]) * size);

  aGPU.put(a);

  time1 = clUtilGetTime();

  aGPU.put(a);
  
  Device::Finish();
  
  time2 = clUtilGetTime();

  printf("%E B/s\n", 1 * sizeof(a[0]) * size / (time2 - time1));
}
