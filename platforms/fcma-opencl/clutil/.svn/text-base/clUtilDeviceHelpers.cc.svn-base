#include "clUtilDevice.h"

using namespace std;
using namespace clUtil;

string Device::fileToString(const char* filename)
{
  ifstream fileStream(filename, ios::binary | ios::in | ios::ate);

  if(fileStream.is_open() == true)
  {
    size_t fileSize = fileStream.tellg();
    unique_ptr<char[]> memoryBuffer(new char[fileSize]);

    fileStream.seekg(0, ios::beg);
    fileStream.read(memoryBuffer.get(), fileSize);
 
    return string(memoryBuffer.get(), fileSize);
  }
  else
  {
    throw clUtilException(string("Error: could not open file ") +
                          string(filename));
  }
}

cl_int Device::loadBinary(const char* cachename)
{
  cl_int err;
  string binaryFile;

  try
  {
    string filename = string(cachename) + getDeviceInfo().Name + string(".bin");

    binaryFile = fileToString(filename.c_str());
  }
  catch(clUtilException err)
  {
    return CL_INVALID_BINARY;
  }

  size_t binaryFileSize = binaryFile.size();
  const char* binaryData = binaryFile.data();

  mProgram = 
    clCreateProgramWithBinary(mContext,
                              1,
                              &mDeviceID,
                              &binaryFileSize,
                              (const unsigned char**)&binaryData,
                              NULL,
                              &err);
  if(err != CL_SUCCESS)
  {
    return err;
  }

  err = clBuildProgram(mProgram,
                       1,
                       &mDeviceID,
                       NULL,
                       NULL,
                       NULL);

  if(err != CL_SUCCESS)
  {
    return err;
  }

  return CL_SUCCESS;
}

void Device::dumpBinary(const char* filename)
{
  try
  {
    size_t binarySize;
    cl_int err;

    err = clGetProgramInfo(mProgram,
                           CL_PROGRAM_BINARY_SIZES,
                           sizeof(binarySize),
                           &binarySize,
                           NULL);
    if(err != CL_SUCCESS)
    {
      printf("Warning: Could not dump program binary to disk for later use. "
             "Reason: %s\n",
             clUtilGetErrorCode(err));
      return;
    }

    unique_ptr<char[]> binary(new char[binarySize]);
    char* binaryPtr = binary.get();

    err = clGetProgramInfo(mProgram,
                           CL_PROGRAM_BINARIES,
                           sizeof(binaryPtr),
                           &binaryPtr,
                           NULL);
    if(err != CL_SUCCESS)
    {
      printf("Warning: Could not dump program binary to disk for later use. "
             "Reason: %s\n",
             clUtilGetErrorCode(err));
      return;
    }

    ofstream binaryFileStream(filename, ios::out | ios::binary);
    binaryFileStream.exceptions(ofstream::failbit | ofstream::badbit);

    binaryFileStream.write(binary.get(), binarySize);

    return;
  }
  catch(exception& err)
  {
    printf("Warning: Could not dump program binary to disk for later use. "
           "Reason: %s\n",
           err.what());
    return;
  }
}

cl_int Device::buildProgram(const char** filenames, 
                            size_t numFiles, 
                            const char* options)
{
  cl_int err;

  unique_ptr<const char*[]> filePointers(new const char*[numFiles]);
  vector<string> fileStrings;

  //Load the source files
  for(size_t curFile = 0; curFile < numFiles; curFile++)
  {
    fileStrings.push_back(fileToString(filenames[curFile]));

    filePointers.get()[curFile] = fileStrings[curFile].c_str();
  }

  mProgram = clCreateProgramWithSource(mContext,
                                       numFiles,
                                       filePointers.get(),
                                       NULL,
                                       &err);
  clUtilCheckError(err);

  ostringstream deviceFlags;

  deviceFlags << options << " -DDEVICE_NUM=" << mDeviceNumber;
#if defined(WIN32) || defined(__WIN32)
  deviceFlags << " -I\"C:\\Program Files (x86)\\clUtil\\Include\\kernels\"";
#else
  deviceFlags << " -I/usr/include/clutil/kernels";
#endif
  err = clBuildProgram(mProgram,
                       1,
                       &mDeviceID,
                       deviceFlags.str().c_str(),
                       NULL,
                       NULL);
  if(err != CL_SUCCESS)
  {
    char errString[32768];
    cl_int err1;

    err1 = clGetProgramBuildInfo(mProgram,
                                 mDeviceID,
                                 CL_PROGRAM_BUILD_LOG,
                                 sizeof(errString),
                                 errString,
                                 NULL);
    clUtilCheckError(err1);
    printf("%s", errString);
    return err;
  }

  return CL_SUCCESS;
}

void Device::getKernels()
{
  cl_uint numKernels;
  cl_int err;

  //Get the number of kernels in the program
  err = clCreateKernelsInProgram(mProgram, 0, NULL, &numKernels);
  clUtilCheckError(err);

  //Actually create the kernels
  unique_ptr<cl_kernel[]> kernels(new cl_kernel[numKernels]);

  err = clCreateKernelsInProgram(mProgram, 
                                 numKernels,
                                 kernels.get(),
                                 NULL);

  //Put all them there kernels in a hash table
  for(size_t curKernel = 0; curKernel < numKernels; curKernel++)
  {
    size_t kernelNameLength;

    //Get the length of the kernel's name
    err = clGetKernelInfo(kernels.get()[curKernel],
                          CL_KERNEL_FUNCTION_NAME,
                          0,
                          NULL,
                          &kernelNameLength);
    clUtilCheckError(err);

    unique_ptr<char[]> kernelName(new char[kernelNameLength]);

    err = clGetKernelInfo(kernels.get()[curKernel], 
                          CL_KERNEL_FUNCTION_NAME,
                          kernelNameLength,
                          kernelName.get(),
                          NULL);
    clUtilCheckError(err);

    mKernels[string(kernelName.get())] = kernels.get()[curKernel];
  }
}
