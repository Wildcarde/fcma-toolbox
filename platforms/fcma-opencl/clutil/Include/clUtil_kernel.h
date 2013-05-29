#pragma once
#include <stdarg.h>

#include "clUtilCommon.h"
#include "clUtilDevice.h"
#include "clUtilMemory.h"

#define clUtilCheckKernelArg(kernelName, paramNum, err)\
  if(err != CL_SUCCESS)\
{\
  printf("__kernel %s(): %ld parameter: %s\n",\
         kernelName,\
         paramNum,\
         clUtilGetErrorCode(err));\
  raise(SIGTRAP);\
  return;\
}

void gridSetGlobalLocal(size_t* global, 
                        size_t* local, 
                        unsigned int curDim);

template<typename TGlobal, typename TLocal, typename...Args> 
void gridSetGlobalLocal(size_t* global, 
                        size_t* local, 
                        unsigned int curDim, 
                        TGlobal globalDimVal,
                        TLocal localDimVal,
                        Args... args)
{
  global[curDim] = globalDimVal % localDimVal == 0 ?
    globalDimVal :
    (globalDimVal / localDimVal + 1) * localDimVal;
  local[curDim] = localDimVal;

  gridSetGlobalLocal(global, local, curDim + 1, args...);
}

//Grid class definitions for defining work items
namespace clUtil
{
  //Base class. Exists for virtual function overloading.
  class Grid
  {
    public:
      virtual size_t getDim() = 0;
      virtual size_t* getGlobal() = 0;
      virtual size_t* getLocal() = 0;
  };

  //Created by clUtilGrid(). Actually contains work group information.
  template<typename... Args> class _Grid : public Grid
  {
    public:
      _Grid(Args... args)
      {
        gridSetGlobalLocal(mGlobal,
                           mLocal,
                           0,
                           args...);
      }

      size_t getDim(){return sizeof...(Args) / 2;}
      size_t* getGlobal(){return mGlobal;}
      size_t* getLocal(){return mLocal;}

    private:
      size_t mGlobal[sizeof...(Args) / 2];
      size_t mLocal[sizeof...(Args) / 2];

      _Grid();
  };
}

//Wrapper function to create a grid object of the correct type so you don't
//have to fuck with template arguments.
template<typename... Args> clUtil::_Grid<Args...> clUtilGrid(Args... args)
{
  return clUtil::_Grid<Args...>(args...);
}

void setArg_(const cl_kernel kernel, 
             const size_t argIndex, 
             clUtil::Memory* curArg,
             std::vector<clUtil::Memory*>& sources);
void setArg_(const cl_kernel kernel, 
             const size_t argIndex, 
             clUtil::Image& curArg,
             std::vector<clUtil::Memory*>& sources);
void setArg_(const cl_kernel kernel, 
             const size_t argIndex, 
             clUtil::Buffer& curArg,
             std::vector<clUtil::Memory*>& sources);

template<typename T> void setArg_(const cl_kernel kernel, 
                                  const size_t argIndex, 
                                  const T& curArg,
                                  std::vector<clUtil::Memory*>& sources)
{
  cl_int err;

  err = clSetKernelArg(kernel, argIndex, sizeof(curArg), &curArg);
  clUtilCheckError(err);
}

void clUtilSetArgs(cl_kernel kernel, 
                   const char* kernelName, 
                   std::vector<clUtil::Memory*>& eventSources,
                   size_t argIndex);

template<typename T, typename... Args> 
void clUtilSetArgs(cl_kernel kernel,
                   const char* kernelName,
                   std::vector<clUtil::Memory*>& eventSources,
                   size_t argIndex,
                   T& curArg,
                   Args&... args)
{
  setArg_(kernel, argIndex, curArg, eventSources);

  clUtilSetArgs(kernel, kernelName, eventSources, argIndex + 1, args...);
}

template<typename... Args> 
void clUtilEnqueueKernel(const char* kernelName,
                         clUtil::Grid&& workGrid,
                         Args&... args)
{
  cl_int err;
  clUtil::Device& currentDevice = clUtil::Device::GetCurrentDevice();
  cl_kernel kernel = 
    currentDevice.getKernel(std::move(std::string(kernelName)));
  std::vector<clUtil::Memory*> memories;

  clUtilSetArgs(kernel, kernelName, memories, 0, args...);
  
  std::map<cl_event, char> events;
  
  //Store unique events associated with memory objects
  for(auto i = memories.begin(); i < memories.end(); i++)
  {
    if((*i)->getLastAccess() != NULL)
    {
      events[(*i)->getLastAccess()] = '\0';
    }
  }

  std::unique_ptr<cl_event[]> eventArray(new cl_event[events.size()]);
  size_t curEvent = 0;

  //Put all the events into a C array so it can be passed off to OpenCL
  for(auto i = events.begin(); i != events.end(); i++)
  {
    eventArray[curEvent] = (*i).first;
    curEvent++;
  }

  cl_event outputEvent;

  err = clEnqueueNDRangeKernel(currentDevice.getCommandQueue(),
                               kernel,
                               workGrid.getDim(),
                               NULL,
                               workGrid.getGlobal(),
                               workGrid.getLocal(),
                               events.size(),
                               events.size() > 0 ? &eventArray[0] : NULL,
                               &outputEvent);
  clUtilCheckError(err);
 
  currentDevice.addProfilingEvent(outputEvent);

  //Update the last event for each Memory object passed to this kernel
  for(auto i = memories.begin(); i < memories.end(); i++)
  {
    (*i)->setLastAccess(outputEvent);
  }

  err = clReleaseEvent(outputEvent);
  clUtilCheckError(err);
}
