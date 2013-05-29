#pragma once
#include "clUtilCommon.h"
#include "clUtilUtility.h"
//#define CLUTIL_ENABLE_PROFILING
//#define CLUTIL_MAPBUFFER

namespace clUtil
{
  struct EventTime
  {
    double Start;
    double End;
  };

#ifdef CLUTIL_MAPBUFFER
  class Device;

  struct CopyTask
  {
    Device* CopyDevice;
    cl_event StartEvent;
    cl_event CopyEvent;
    void* HostPtr;
    void* PinnedPtr;
    size_t Bytes;
    bool IsRead;
  };

  cl_mem GetDevicePinnedMemory(size_t deviceID);
  void EnqueueCopyTask(CopyTask& task);
  void* copyThread(void* data);
#endif

  struct DeviceInfo
  {
    public:
      cl_device_type Type;
      cl_uint VendorID;
      cl_uint MaxComputeUnits;
      std::vector<size_t> MaxWorkItemSizes;
      size_t MaxWorkGroupSize;
      cl_uint PreferredVectorWidthChar;
      cl_uint PreferredVectorWidthShort;
      cl_uint PreferredVectorWidthInt;
      cl_uint PreferredVectorWidthLong;
      cl_uint PreferredVectorWidthFloat;
      cl_uint PreferredVectorWidthDouble;
      cl_uint PreferredVectorWidthHalf;
      cl_uint NativeVectorWidthChar;
      cl_uint NativeVectorWidthShort;
      cl_uint NativeVectorWidthInt;
      cl_uint NativeVectorWidthLong;
      cl_uint NativeVectorWidthFloat;
      cl_uint NativeVectorWidthDouble;
      cl_uint NativeVectorWidthHalf;
      cl_uint MaxClockFrequency;
      cl_uint AddressBits;
      cl_ulong MaxMemAllocSize;
      cl_bool ImageSupport;
      cl_uint MaxReadImageArgs;
      cl_uint MaxWriteImageArgs;
      size_t Image2DMaxWidth;
      size_t Image2DMaxHeight;
      size_t Image3DMaxWidth;
      size_t Image3DMaxHeight;
      size_t Image3DMaxDepth;
      cl_uint MaxSamplers;
      size_t MaxParameterSize;
      cl_uint MemBaseAddrAlign;
      cl_uint MinDataTypeAlignSize;
      cl_device_fp_config SingleFPConfig;
      cl_device_mem_cache_type GlobalMemCacheType;
      cl_uint GlobalMemCachelineSize;
      cl_ulong GlobalMemCacheSize;
      cl_ulong GlobalMemSize;
      cl_ulong MaxConstantBufferSize;
      cl_uint MaxConstantArgs;
      cl_device_local_mem_type LocalMemType;
      cl_ulong LocalMemSize;
      cl_bool ErrorCorrectionSupport;
      cl_bool HostUnifiedMemory;
      size_t ProfilingTimerResolution;
      cl_bool EndianLittle;
      cl_bool DeviceAvailable;
      cl_bool CompilerAvailable;
      cl_device_exec_capabilities ExecutionCapabilities;
      cl_command_queue_properties QueueProperties;
      std::string Name;
      std::string Vendor;
      std::string DriverVersion;
      std::string OpenCLProfile;
      std::string OpenCLVersion;
      std::string OpenCLCVersion;
      std::vector<std::string> Extensions;

      void initialize(cl_device_id deviceID);
    private:
  };

  class Device
  {
    friend class Buffer;
    friend void* copyThread(void* data);

    private:
      cl_device_id mDeviceID;
      cl_context mContext;
      cl_program mProgram;
      std::map<std::string, cl_kernel> mKernels;
      DeviceInfo mDeviceInfo;
      bool mInfoInitialized;
      size_t mDeviceNumber;     
      std::vector<std::vector<cl_event>> mProfileEvents;
      std::vector<cl_command_queue> mCommandQueues;
      size_t mCurrentCommandQueue;
#ifdef CLUTIL_MAPBUFFER
      std::vector<cl_mem> mPinnedBuffer;
      std::vector<bool> mBufferInUse;
#endif

      static size_t CurrentDevice;
      static bool DevicesInitialized;
      static bool DevicesFetched;
      static std::vector<Device> Devices;
      static const size_t NumCommandQueues;
      static const size_t PinnedBufferSize;
      static bool ProfilingStarted;

      std::string fileToString(const char* filename);
      cl_int loadBinary(const char* cachename);
      cl_int buildProgram(const char** filenames, 
                          size_t numFiles,
                          const char* options);
      void initialize(const char** sourceFiles,
                      size_t numFiles,
                      const char* cachename,
                      const char* options);
      void dumpBinary(const char* filename);
      void getKernels();

    public:
      Device(cl_device_id deviceID);
      const DeviceInfo& getDeviceInfo() const {return mDeviceInfo;}
      cl_device_id getDeviceID() const {return mDeviceID;}
      cl_context getContext() const {return mContext;}
      void flush();
      void finish();
      void addProfilingEvent(cl_event event);

      size_t getNumCommandQueues() const { return mCommandQueues.size(); }
      size_t getCommandQueueID() const { return mCurrentCommandQueue; }

      cl_command_queue getCommandQueue() const 
      {
        return mCommandQueues[mCurrentCommandQueue];
      }

      void setCommandQueue(size_t id)
      {
        if(id > mCommandQueues.size())
        {
          throw clUtilException("Invalid command queue ID");
        }

        mCurrentCommandQueue = id;
      }

      cl_kernel getKernel(const std::string& kernelName) const;
      size_t getDeviceNumber() const {return mDeviceNumber;}

      static void Flush();
      static void Finish();
      static const std::vector<Device>& GetDevices() {return Devices; }
      static void FetchDevices();
      static void InitializeDevices(const char** sourceFiles, 
                                    size_t numFiles,
                                    const char* cachename = NULL,
                                    const char* options = "");
      static Device& GetCurrentDevice() { return Devices[CurrentDevice]; }
      static size_t GetCurrentDeviceNum() { return CurrentDevice; }
      static void DumpProfilingData();
      static void SetCurrentDevice(size_t deviceNum) 
      {
        if(deviceNum > Devices.size())
        {
          throw clUtilException("SetCurrentDevice():Invalid device.");
        }

        CurrentDevice = deviceNum;
      };

      static void StartProfiling();
      static void Finalize();
  };
}
