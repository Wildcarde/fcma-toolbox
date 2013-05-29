#include <clUtil.h>
#include <stdlib.h>
#include <stdio.h>

using namespace clUtil;

void fillKeys(unsigned int* array, unsigned int length)
{
  for(unsigned int i = 0; i < length; i++)
  {
    array[i] = i % 10;
  }
}

void fillVals(unsigned int* array, unsigned int length)
{
  for(unsigned int i = 0; i < length; i++)
  {
    array[i] = i;
  }
}


void usage()
{
  printf("Usage:Sort rows cols [deviceNum]\n");
  exit(1);
}

int main(int argc, char** argv)
{
  unsigned int rows;
  unsigned int cols;
  char const* kernel = "kernel.cl";
  unsigned int* keys;
  unsigned int* vals;
  size_t size;
  unsigned int deviceNum = 0;

  if(argc < 3)
  {
    usage();
  }
  if(sscanf(argv[1], "%u", &rows) != 1)
  {
    usage();
  }
  if(sscanf(argv[2], "%u", &cols) != 1)
  {
    usage();
  }
  if(argc >= 4)
  {
    if(sscanf(argv[3], "%u", &deviceNum) != 1)
    {
      usage();
    }
  }

  keys = new unsigned int[rows * cols];
  vals = new unsigned int[rows * cols];

  //Fill the arrays
  for(unsigned int curRow = 0; curRow < rows; curRow++)
  {
    fillKeys(&keys[curRow * cols], cols);
    fillVals(&vals[curRow * cols], cols);
  }

  Device::InitializeDevices(&kernel, 1);

  Device::SetCurrentDevice(deviceNum);

  size = sizeof(unsigned int) * rows * cols;

  printf("Running on device %lu %s\n",
         Device::GetCurrentDeviceNum(),
         Device::GetCurrentDevice().getDeviceInfo().Name.c_str());

  Buffer keysDevice1(size);
  Buffer keysDevice2(size);
  Buffer valsDevice1(size);
  Buffer valsDevice2(size);
  
  keysDevice1.put(keys);
  valsDevice1.put(vals);

  clUtilEnqueueKernel("sort",
                      clUtilGrid(1, 1, rows, 1),
                      keysDevice1,
                      keysDevice2,
                      valsDevice1,
                      valsDevice2,
                      cols);

  keysDevice2.get(keys);
  valsDevice2.get(vals);

  Device::Finish();

  for(unsigned int curRow = 0; curRow < rows; curRow++)
  {
    for(unsigned int curCol = 0; curCol < cols; curCol++)
    {
      printf("{%u, %u} ", 
             keys[curRow * cols + curCol],
             vals[curRow * cols + curCol]);
#if VERIFY
      if(curCol < cols - 1 &&
         keys[curRow * cols + curCol] < keys[curRow * cols + curCol + 1])
      {
        printf("Error, element %u, %u\n", curRow, curCol);
        exit(1);
      }
#endif
    }

    printf("\n");
  }

  printf("Success!\n");


  return 0;
}
