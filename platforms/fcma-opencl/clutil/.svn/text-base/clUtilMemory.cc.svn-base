#include "clUtilMemory.h"

using namespace clUtil;

void Image::initialize()
{
  cl_int err;

  if(mDimensions == 1) //1D image
  {
    //Map the 1D image to a 2D image
    mWidth = mDevice.getDeviceInfo().Image2DMaxWidth;
    mHeight = m1DWidth % mWidth == 0 ? 
      m1DWidth / mWidth : 
      m1DWidth / mWidth + 1;

    cl_image_format imageFormat = {mChannelOrder, mChannelType};

    mMemHandle = clCreateImage2D(mDevice.getContext(),
                                 CL_MEM_READ_WRITE,
                                 &imageFormat,
                                 mWidth,
                                 mHeight,
                                 0,  
                                 NULL,
                                 &err);
    clUtilCheckError(err);
  }
  else if(mDimensions == 2) //2D image
  {
    cl_image_format imageFormat = {mChannelOrder, mChannelType};

    mMemHandle = clCreateImage2D(mDevice.getContext(),
                                 CL_MEM_READ_WRITE,
                                 &imageFormat,
                                 mWidth,
                                 mHeight,
                                 0,  
                                 NULL,
                                 &err);
    clUtilCheckError(err);
  }
  else if(mDimensions == 3) //3D image
  {
    cl_image_format imageFormat = {mChannelOrder, mChannelType};

    mMemHandle = clCreateImage3D(mDevice.getContext(),
                                 CL_MEM_READ_WRITE,
                                 &imageFormat,
                                 mWidth,
                                 mHeight,
                                 mDepth,
                                 0,
                                 0,
                                 NULL,
                                 &err);
    clUtilCheckError(err);
  }
  else //This should never be reached
  {
    throw clUtilException("Invalid image dimensionality. This should never "
                          "happen; please report this as a bug.");
  }
}

void Image::put(void const* pointer, const size_t len)
{
  cl_int err;
  cl_event nextEvent = NULL;

  if(mDimensions == 1) //1D image
  {
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {mWidth, mHeight, 1};
    size_t imageElementSize;
    //size_t actualLength; //TODO allow fetching arbitrary regions

    //actualLength = len == 0 ? m1DWidth : len;

    err = clGetImageInfo(mMemHandle,
                         CL_IMAGE_ELEMENT_SIZE,
                         sizeof(imageElementSize),
                         &imageElementSize,
                         NULL);
    clUtilCheckError(err);

    //Copy all but the last row to the device
    if(mHeight > 1)
    {
      region[1] = mHeight - 1;

      err = clEnqueueWriteImage(mDevice.getCommandQueue(),
                                mMemHandle,
                                CL_FALSE,
                                origin,
                                region,
                                0,
                                0,
                                pointer,
                                mLastAccess != NULL ? 1 : 0,
                                mLastAccess != NULL ? &mLastAccess : NULL,
                                &nextEvent);

      clUtilCheckError(err);

      setLastAccess(nextEvent);
    
      mDevice.addProfilingEvent(nextEvent);

      err = clReleaseEvent(nextEvent);
      clUtilCheckError(err);
    }

    //Copy the last row to the device
    origin[1] = mHeight - 1;

    region[0] = m1DWidth % mWidth == 0 ? mWidth : m1DWidth % mWidth;
    region[1] = 1;

    err = clEnqueueWriteImage(mDevice.getCommandQueue(),
                              mMemHandle,
                              CL_FALSE,
                              origin,
                              region,
                              0,
                              0,
                              &((char*)pointer)[imageElementSize * 
                              (mHeight - 1) * mWidth],
                              mLastAccess != NULL ? 1 : 0,
                              mLastAccess != NULL ? &mLastAccess : NULL,
                              &nextEvent);
    clUtilCheckError(err);

    setLastAccess(nextEvent);   
    
    mDevice.addProfilingEvent(nextEvent);

    err = clReleaseEvent(nextEvent);
    clUtilCheckError(err);
  }
  else if(mDimensions == 2 || mDimensions == 3)
  {
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {mWidth, mHeight, mDepth};

    err = clEnqueueWriteImage(mDevice.getCommandQueue(),
                              mMemHandle,
                              CL_FALSE,
                              origin,
                              region,
                              0,
                              0,
                              pointer,
                              mLastAccess != NULL ? 1 : 0,
                              mLastAccess != NULL ? &mLastAccess : NULL,
                              &nextEvent);
    clUtilCheckError(err);

    setLastAccess(nextEvent);
    
    mDevice.addProfilingEvent(nextEvent);

    err = clReleaseEvent(nextEvent);
    clUtilCheckError(err);
  }
  else
  {
    throw clUtilException("Invalid image dimensionality. This should never "
                          "happen; please report this as a bug.");
  }
}

void Image::get(void* const pointer, const size_t len)
{
  cl_int err;
  cl_event nextEvent;

  if(mDimensions == 1) //1D image
  {
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {mWidth, mHeight, 1};
    size_t imageElementSize; 
    //size_t actualLength;

    //actualLength = len == 0 ? m1DWidth : len;

    err = clGetImageInfo(mMemHandle,
                         CL_IMAGE_ELEMENT_SIZE,
                         sizeof(imageElementSize),
                         &imageElementSize,
                         NULL);
    clUtilCheckError(err);

    //Copy all but the last row to the device
    if(mHeight > 1)
    {
      region[1] = mHeight - 1;

      err = clEnqueueReadImage(mDevice.getCommandQueue(),
                               mMemHandle,
                               CL_FALSE,
                               origin,
                               region,
                               0,
                               0,
                               pointer,
                               mLastAccess != NULL ? 1 : 0,
                               mLastAccess != NULL ? &mLastAccess : NULL,
                               &nextEvent);
      clUtilCheckError(err);

      setLastAccess(nextEvent);
    
      mDevice.addProfilingEvent(nextEvent);

      err = clReleaseEvent(nextEvent);
      clUtilCheckError(err);
    }

    //Copy the last row to the device
    origin[1] = mHeight - 1;

    region[0] = m1DWidth % mWidth == 0 ? mWidth : m1DWidth % mWidth;
    region[1] = 1;

    err = clEnqueueReadImage(mDevice.getCommandQueue(),
                             mMemHandle,
                             CL_FALSE,
                             origin,
                             region,
                             0,
                             0,
                             &((char*)pointer)[imageElementSize * 
                             (mHeight - 1) * mWidth],
                             mLastAccess != NULL ? 1 : 0,
                             mLastAccess != NULL ? &mLastAccess : NULL,
                             &nextEvent);
    clUtilCheckError(err);

    setLastAccess(nextEvent);   
    
    mDevice.addProfilingEvent(nextEvent);

    err = clReleaseEvent(nextEvent);
    clUtilCheckError(err);
  }
  else if(mDimensions == 2 || mDimensions == 3)
  {
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {mWidth, mHeight, mDepth};

    err = clEnqueueReadImage(mDevice.getCommandQueue(),
                             mMemHandle,
                             CL_FALSE,
                             origin,
                             region,
                             0,
                             0,
                             pointer,
                             mLastAccess != NULL ? 1 : 0,
                             mLastAccess != NULL ? &mLastAccess : NULL,
                             &nextEvent);
    clUtilCheckError(err);

    setLastAccess(nextEvent);   
  
    mDevice.addProfilingEvent(nextEvent);

    err = clReleaseEvent(nextEvent);
    clUtilCheckError(err);
  }
  else
  {
    throw clUtilException("Invalid image dimensionality. This should never "
                          "happen; please report this as a bug.");
  }
}

void Buffer::put(const void* const pointer, const size_t len)
{
  cl_int err;
  size_t length = len == 0 ? mLength : len;
  cl_event nextEvent;
  
#ifdef CLUTIL_MAPBUFFER //Stream Data through pinned memory
  size_t curCommandQueue = mDevice.mCurrentCommandQueue;

  for(size_t bytes = 0; bytes < length; bytes += Device::PinnedBufferSize)
  {
    size_t copyBytes = length - bytes < Device::PinnedBufferSize ? 
      length - bytes :
      Device::PinnedBufferSize;

    cl_event deviceTransfer;
    cl_event copyEvent;
#if 0
    cl_event mapEvent;
    cl_event unmapEvent;
    CopyTask newCopyTask;

    void* mappedPinnedMemory = clEnqueueMapBuffer(mDevice.getCommandQueue(),
                                                  mDevice.mPinnedBuffer,
                                                  CL_FALSE,
                                                  CL_MAP_WRITE,
                                                  0,
                                                  copyBytes,
                                                  mLastAccess != NULL ? 
                                                    1 : 0,
                                                  mLastAccess != NULL ? 
                                                    &mLastAccess : NULL,
                                                  &mapEvent,
                                                  &err);
    clUtilCheckError(err);

    //Create an event so we know when the memcpy from pinned memory completes
    copyEvent = clCreateUserEvent(mDevice.getContext(), &err);
    clUtilCheckError(err);

    newCopyTask.CopyDevice = &mDevice;
    newCopyTask.StartEvent = mapEvent;
    newCopyTask.CopyEvent = copyEvent;
    newCopyTask.HostPtr = &((char*)pointer)[bytes];
    newCopyTask.PinnedPtr = mappedPinnedMemory;
    newCopyTask.Bytes = copyBytes;
    newCopyTask.IsRead = false;

    EnqueueCopyTask(newCopyTask);

    err = clEnqueueUnmapMemObject(mDevice.getCommandQueue(),
                                  mDevice.mPinnedBuffer,
                                  mappedPinnedMemory,
                                  1,
                                  &copyEvent,
                                  &unmapEvent);
    clUtilCheckError(err);

    err = clEnqueueCopyBuffer(mDevice.getCommandQueue(),
                              mMemHandle,
                              mDevice.mPinnedBuffer,
                              bytes,
                              0,
                              copyBytes,
                              1,
                              &unmapEvent,
                              &deviceTransfer);
    clUtilCheckError(err);

    //When the copy completes, we can use the pinned memory again...
    err = clSetEventCallback(deviceTransfer,
                             CL_COMPLETE,
                             [=](cl_event event, cl_int status, void* userData)
                             {
                               *(volatile bool*)userData = false;
                             },
                             (void*)&mDevice.mBufferInUse);
    clUtilCheckError(err);

    err = clReleaseEvent(deviceTransfer);
    clUtilCheckError(err);
    err = clReleaseEvent(mapEvent);
    clUtilCheckError(err);
    err = clReleaseEvent(copyEvent);
    clUtilCheckError(err);
    err = clReleaseEvent(unmapEvent);
    clUtilCheckError(err);
#else
    err = clEnqueueWriteBuffer(mDevice.getCommandQueue(),
                               mDevice.mPinnedBuffer[curCommandQueue],
                               CL_FALSE,
                               0,
                               copyBytes,
                               &((char*)pointer)[bytes],
                               mLastAccess != NULL ? 1 : 0,
                               mLastAccess != NULL ? &mLastAccess : NULL,
                               &copyEvent);
    clUtilCheckError(err);

    err = clEnqueueCopyBuffer(mDevice.getCommandQueue(),
                              mDevice.mPinnedBuffer[curCommandQueue],
                              mMemHandle,
                              0,
                              bytes,
                              copyBytes,
                              1,
                              &copyEvent,
                              &deviceTransfer);
    clUtilCheckError(err);

    setLastAccess(deviceTransfer);
    
    mDevice.addProfilingEvent(deviceTransfer);
    mDevice.addProfilingEvent(copyEvent);
    
    err = clReleaseEvent(deviceTransfer);
    clUtilCheckError(err);
    err = clReleaseEvent(copyEvent);
    clUtilCheckError(err);

#endif
  }

  clEnqueueMarker(mDevice.getCommandQueue(), &nextEvent);
#else

  err = clEnqueueWriteBuffer(mDevice.getCommandQueue(),
                             mMemHandle,
                             CL_FALSE,
                             0,
                             length,
                             pointer,
                             mLastAccess != NULL ? 1 : 0,
                             mLastAccess != NULL ? &mLastAccess : NULL,
                             &nextEvent);
  clUtilCheckError(err);
#endif

  setLastAccess(nextEvent);

  mDevice.addProfilingEvent(nextEvent);

  err = clReleaseEvent(nextEvent);
  clUtilCheckError(err);
}

void Buffer::get(void* const pointer, const size_t len)
{
  cl_int err;
  size_t length = len == 0 ? mLength : len;
  cl_event nextEvent;

#ifdef CLUTIL_MAPBUFFER
  size_t curCommandQueue = mDevice.mCurrentCommandQueue;

  for(size_t bytes = 0; bytes < length; bytes += Device::PinnedBufferSize)
  {
    size_t copyBytes = length - bytes < Device::PinnedBufferSize ? 
      length - bytes :
      Device::PinnedBufferSize;

    cl_event deviceTransfer;
    cl_event copyEvent;
#if 0
    cl_event mapEvent;
    cl_event unmapEvent;
    CopyTask newCopyTask;

    err = clEnqueueCopyBuffer(mDevice.getCommandQueue(),
                              mMemHandle,
                              mDevice.mPinnedBuffer,
                              bytes,
                              0,
                              copyBytes,
                              mLastAccess != NULL ? 1 : 0,
                              mLastAccess != NULL ? &mLastAccess : NULL,
                              &deviceTransfer);
    clUtilCheckError(err);

    void* mappedPinnedMemory = clEnqueueMapBuffer(mDevice.getCommandQueue(),
                                                  mDevice.mPinnedBuffer,
                                                  CL_FALSE,
                                                  CL_MAP_READ,
                                                  0,
                                                  copyBytes,
                                                  1,
                                                  &deviceTransfer,
                                                  &mapEvent,
                                                  &err);
    clUtilCheckError(err);

    //Create an event so we know when the memcpy from pinned memory completes
    copyEvent = clCreateUserEvent(mDevice.getContext(), &err);
    clUtilCheckError(err);

    newCopyTask.CopyDevice = &mDevice;
    newCopyTask.StartEvent = mapEvent;
    newCopyTask.CopyEvent = copyEvent;
    newCopyTask.HostPtr = &((char*)pointer)[bytes];
    newCopyTask.PinnedPtr = mappedPinnedMemory;
    newCopyTask.Bytes = copyBytes;
    newCopyTask.IsRead = true;

    EnqueueCopyTask(newCopyTask);

    err = clEnqueueUnmapMemObject(mDevice.getCommandQueue(),
                                  mDevice.mPinnedBuffer,
                                  mappedPinnedMemory,
                                  1,
                                  &copyEvent,
                                  &unmapEvent);
    clUtilCheckError(err);

    //When the unmap completes, we can use the pinned memory again...
    err = clSetEventCallback(unmapEvent,
                             CL_COMPLETE,
                             [=](cl_event event, cl_int status, void* userData)
                             {
                               *(volatile bool*)userData = false;
                             },
                             (void*)&mDevice.mBufferInUse);

    err = clReleaseEvent(deviceTransfer);
    clUtilCheckError(err);
    err = clReleaseEvent(mapEvent);
    clUtilCheckError(err);
    err = clReleaseEvent(copyEvent);
    clUtilCheckError(err);
    err = clReleaseEvent(unmapEvent);
    clUtilCheckError(err);
#else
    err = clEnqueueCopyBuffer(mDevice.getCommandQueue(),
                              mMemHandle,
                              mDevice.mPinnedBuffer[curCommandQueue],
                              bytes,
                              0,
                              copyBytes,
                              mLastAccess != NULL ? 1 : 0,
                              mLastAccess != NULL ? &mLastAccess : NULL,
                              &deviceTransfer);
    clUtilCheckError(err);
    
    err = clEnqueueReadBuffer(mDevice.getCommandQueue(),
                              mDevice.mPinnedBuffer[curCommandQueue],
                              CL_FALSE,
                              0,
                              copyBytes,
                              &((char*)pointer)[bytes],
                              1,
                              &deviceTransfer,
                              &copyEvent);
    clUtilCheckError(err);

    setLastAccess(copyEvent);
    
    mDevice.addProfilingEvent(deviceTransfer);
    mDevice.addProfilingEvent(copyEvent);
    
    err = clReleaseEvent(deviceTransfer);
    clUtilCheckError(err);
    err = clReleaseEvent(copyEvent);
    clUtilCheckError(err);

#endif
  }

  clEnqueueMarker(mDevice.getCommandQueue(), &nextEvent);
#else
  err = clEnqueueReadBuffer(mDevice.getCommandQueue(),
                            mMemHandle,
                            CL_FALSE,
                            0,
                            length,
                            pointer,
                            mLastAccess != NULL ? 1 : 0,
                            mLastAccess != NULL ? &mLastAccess : NULL,
                            &nextEvent);
  clUtilCheckError(err);
#endif

  setLastAccess(nextEvent);

  mDevice.addProfilingEvent(nextEvent);

  err = clReleaseEvent(nextEvent);
  clUtilCheckError(err);
}

void Buffer::put(const Buffer& b)
{
}
