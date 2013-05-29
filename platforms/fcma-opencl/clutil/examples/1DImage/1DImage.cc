#include <clUtil.h>
#include <memory>
#include <iostream>

using namespace clUtil;
using namespace std;

const unsigned int kImageSize = 1234;

struct float4
{
  float x;
  float y;
  float z;
  float w;
};

int main(int argc, char** argv)
{
  unique_ptr<float4> result(new float4[kImageSize]);
  char const* kernel = "kernel.cl";

  try
  {
    Device::InitializeDevices(&kernel, 1);

    Image a(kImageSize, CL_RGBA, CL_FLOAT);
    Image b(kImageSize, CL_RGBA, CL_FLOAT);
    Image c(kImageSize, CL_RGBA, CL_FLOAT);

    clUtilEnqueueKernel("fillImage",
                        clUtilGrid(kImageSize, 64),
                        a,
                        kImageSize);

    clUtilEnqueueKernel("fillImage",
                        clUtilGrid(kImageSize, 64),
                        b,
                        kImageSize);

    clUtilEnqueueKernel("sumImages",
                        clUtilGrid(kImageSize, 64),
                        a,
                        b,
                        c,
                        kImageSize);

    c.get(result.get());
    Device::Finish();

    Device::DumpProfilingData();
  }
  catch(clUtilException& err)
  {
    cout << err.what() << endl;
    exit(1);
  }

  for(size_t i = 0; i < kImageSize; i++)
  {
    if(result.get()[i].w != 2.0f * i)
    {
      printf("Error: index %ld value: %f\n", i, result.get()[i].w);
    }
  }

  printf("Success!\n");
}
