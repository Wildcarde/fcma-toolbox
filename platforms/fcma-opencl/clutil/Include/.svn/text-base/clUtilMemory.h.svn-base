#pragma once
#include "clUtilCommon.h"
#include "clUtilDevice.h"
#include "clUtilException.h"

#define kCLUtilAllocPinnedBuffer ((void*)0x1)

namespace clUtil
{
  class Memory
  {
    protected:
      cl_mem mMemHandle;
      Device& mDevice;
      cl_event mLastAccess;
    public:
      Memory(const Device& device = Device::GetCurrentDevice()) :
        mDevice(const_cast<Device&>(device)),
        mLastAccess(NULL)
      {
      }

      Memory(const Memory& b) :
        mMemHandle(b.mMemHandle),
        mDevice(b.mDevice),
        mLastAccess(b.mLastAccess)
      {
        clRetainMemObject(mMemHandle);
      }

      ~Memory()
      {
        clReleaseMemObject(mMemHandle);
      }

      cl_mem getMemHandle() const { return mMemHandle; }
      cl_event getLastAccess() const { return mLastAccess; }
      void setLastAccess(cl_event event) 
      { 
        cl_int err;

        if(mLastAccess != NULL)
        {
          err = clReleaseEvent(mLastAccess);
          clUtilCheckError(err);
        }

        mLastAccess = event; 
      
        err = clRetainEvent(mLastAccess);
        clUtilCheckError(err);
      }

      virtual void get(void* const pointer, const size_t len = 0) = 0;
      virtual void put(const void* const pointer, const size_t len = 0) = 0;
      virtual bool isImage() const = 0;
  };  

  class Image : public Memory
  {
    private:
      size_t mDimensions;
      size_t m1DWidth;
      size_t mWidth;
      size_t mHeight;
      size_t mDepth;
      cl_channel_order mChannelOrder;
      cl_channel_type mChannelType;

      void initialize();
    public:
      Image(size_t width, 
            cl_channel_order channelOrder,
            cl_channel_type channelType,
            Device& device = Device::GetCurrentDevice()) :
        Memory(device),
        mDimensions(1),
        m1DWidth(width),
        mWidth(0),
        mHeight(0),
        mDepth(0),
        mChannelOrder(channelOrder),
        mChannelType(channelType)
      {
        initialize();
      }

      Image(const size_t width,
            const size_t height,
            const cl_channel_order channelOrder,
            const cl_channel_type channelType,
            const Device& device = Device::GetCurrentDevice()) :
        Memory(device),
        mDimensions(2),
        m1DWidth(0),
        mWidth(width),
        mHeight(height),
        mDepth(1),
        mChannelOrder(channelOrder),
        mChannelType(channelType)
      {
        initialize();
      }

      Image(const size_t width,
            const size_t height,
            const size_t depth,
            const cl_channel_order channelOrder,
            const cl_channel_type channelType,
            const Device& device = Device::GetCurrentDevice()) :
        Memory(device),
        mDimensions(3),
        m1DWidth(0),
        mWidth(width),
        mHeight(height),
        mDepth(depth),
        mChannelOrder(channelOrder),
        mChannelType(channelType)
      {
        initialize();
      }

      Image(const Image& b) : 
        Memory(b),
        mDimensions(b.mDimensions),
        m1DWidth(b.m1DWidth),
        mWidth(b.mWidth),
        mHeight(b.mHeight),
        mDepth(b.mDepth),
        mChannelOrder(b.mChannelOrder),
        mChannelType(b.mChannelType)
      {
      }

      Image operator=(const Image& b) const
      {
        return Image(b);
      }

      virtual void put(const void* const pointer, const size_t len = 0);
      virtual void get(void* const pointer, const size_t len = 0);
      virtual bool isImage() const { return true; }
  };

  class Buffer : public Memory
  {
    private:
      cl_mem mParentBuffer;
      size_t mLength;
    public:

      Buffer(const size_t len, 
             void* hostPtr = NULL,
             const Device& device = Device::GetCurrentDevice()) : 
        Memory(device),
        mParentBuffer(0),
        mLength(len)
      {
        cl_int err;
        cl_mem_flags flags = CL_MEM_READ_WRITE;

        if(hostPtr == kCLUtilAllocPinnedBuffer)
        {
          flags |= CL_MEM_ALLOC_HOST_PTR;
          hostPtr = NULL;
        }
        else if(hostPtr != NULL)
        {
          flags |= CL_MEM_USE_HOST_PTR;
        }

        mMemHandle = clCreateBuffer(mDevice.getContext(),
                                    flags,
                                    len,
                                    hostPtr,
                                    &err);
        clUtilCheckError(err);
      }

      Buffer(const Buffer& b) : 
        Memory(b),
        mParentBuffer(b.mParentBuffer),
        mLength(b.mLength)
      {
      }

      Buffer operator=(const Buffer& b) const
      {
        return Buffer(b);
      }

      virtual void put(const void* const pointer, const size_t len = 0);
      virtual void get(void* const pointer, const size_t len = 0);
      void put(const Buffer& b);
      void get(const Buffer& b);
      virtual bool isImage() const { return false; }
  };
}

