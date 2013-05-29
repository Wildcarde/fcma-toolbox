#include "clUtil.h"

void setArg_(const cl_kernel kernel, 
             const size_t argIndex, 
             clUtil::Memory* curArg,
             std::vector<clUtil::Memory*>& sources)
{
  cl_int err;
  cl_mem memoryHandle = curArg->getMemHandle();

  err = clSetKernelArg(kernel, argIndex, sizeof(memoryHandle), &memoryHandle);
  clUtilCheckError(err);

  sources.push_back(curArg);
}

void setArg_(const cl_kernel kernel, 
             const size_t argIndex, 
             clUtil::Image& curArg,
             std::vector<clUtil::Memory*>& sources)
{
  cl_int err;
  cl_mem memoryHandle = curArg.getMemHandle();

  err = clSetKernelArg(kernel, argIndex, sizeof(memoryHandle), &memoryHandle);
  clUtilCheckError(err);
  
  sources.push_back(&curArg);
}

void setArg_(const cl_kernel kernel, 
             const size_t argIndex, 
             clUtil::Buffer& curArg,
             std::vector<clUtil::Memory*>& sources)
{
  cl_int err;
  cl_mem memoryHandle = curArg.getMemHandle();

  err = clSetKernelArg(kernel, argIndex, sizeof(memoryHandle), &memoryHandle);
  clUtilCheckError(err);

  sources.push_back(&curArg);
}

void clUtilSetArgs(cl_kernel kernel,
                   const char* kernelName,
                   std::vector<clUtil::Memory*>& eventSources,
                   size_t argIndex)
{
}

void gridSetGlobalLocal(size_t* global, 
                        size_t* local, 
                        unsigned int curDim)
{
}
