#include "clUtilDevice.h"
#include "clUtilPlatform.h"

using namespace clUtil;
using namespace std;

size_t Device::CurrentDevice = 0;
bool Device::DevicesInitialized = false;
bool Device::DevicesFetched = false;
vector<Device> Device::Devices;
const size_t Device::NumCommandQueues = 2;
const size_t Device::PinnedBufferSize = 128 * 1024 * 1024;
bool Device::ProfilingStarted = false;

#ifdef CLUTIL_MAPBUFFER
#include <pthread.h>

template <typename T> class SRSWQueue
{
  public:
    size_t mQueueSize;
    unique_ptr<T[]> mQueue;
    volatile size_t mHead;
    volatile size_t mTail;

    bool enqueue(T& data)
    {
      size_t nextTail = (mTail + 1) % mQueueSize;

      if(nextTail == mHead)
      {
        return false;
      }

      mQueue[mTail] = data;
      mTail = nextTail;

      return true;
    }

    bool dequeue(T& data)
    {
      if(mHead == mTail)
      {
        return false;
      }

      data = mQueue[mHead];

      mHead = (mHead + 1) % mQueueSize;
      return true;
    }

    SRSWQueue(size_t len = 1024) : mQueueSize(len), mHead(0), mTail(0)
    {
      unique_ptr<T[]> buffer(new T[len]);

      mQueue = move(buffer);
    }
  private:
};

static SRSWQueue<CopyTask> gCopyTaskQueue;
static vector<CopyTask> gCopyTasks;
static vector<cl_mem> gPinnedBuffers;
static volatile bool gShutdown = false;
static pthread_t gCopyThread;

void* clUtil::copyThread(void* data)
{
#if 0
  while(gShutdown == false)
  {
    CopyTask newTask;

    //Move tasks out of the task queue and into our private array
    while(gCopyTaskQueue.dequeue(newTask) == true) 
    {
      gCopyTasks.push_back(newTask);
    }

    //Find tasks that have their map() called and copy the data to or from
    //a pinned buffer and signal copy completion
    for(size_t curTaskID = 0; curTaskID < gCopyTasks.size(); curTaskID++)
    {
      cl_int err;
      cl_int eventStatus;
      CopyTask& curTask = gCopyTasks[curTaskID];

      if(curTask.CopyDevice->mBufferInUse == false)
      {

        err = clGetEventInfo(curTask.StartEvent,
                             CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(eventStatus),
                             &eventStatus,
                             NULL);
        clUtilCheckError(err);

        if(eventStatus == CL_COMPLETE)
        {
          if(curTask.IsRead == false)
          {
            memcpy(curTask.PinnedPtr, curTask.HostPtr, curTask.Bytes);
          }
          else
          {
            memcpy(curTask.HostPtr, curTask.PinnedPtr, curTask.Bytes);
          }

          clSetUserEventStatus(curTask.CopyEvent, CL_COMPLETE);
          curTask.CopyDevice->mBufferInUse = true;
          gCopyTasks.erase(gCopyTasks.begin() + curTaskID);
        }

      }
    }
  }

#endif
  return NULL;
}

void clUtil::EnqueueCopyTask(CopyTask& task)
{
  if(gCopyTaskQueue.enqueue(task) == false)
  {
    Device::Finish();

    if(gCopyTaskQueue.enqueue(task) == false)
    {
      throw clUtilException("Could not enqueue memcpy task. This is a bug.");
    }
  }
}
#endif

static void initHelpers()
{
#ifdef CLUTIL_MAPBUFFER
  gShutdown = false;

  if(pthread_create(&gCopyThread, NULL, copyThread, NULL) != 0)
  {
    throw clUtilException("Could not spawn clUtil helper thread.");
  }
#endif
}

static void finalizeHelpers()
{
#ifdef CLUTIL_MAPBUFFER
  gShutdown = true;

  pthread_join(gCopyThread, NULL);
#endif
}

#ifdef CLUTIL_ENABLE_PROFILING
static const unsigned int kQueueGraphicHeight = 50;
static const unsigned int kTextAreaWidth = 150;
static const unsigned int kGraphicAreaWidth = 1250;
static const unsigned int kTextYOffset = 25;
static const unsigned int kTextXOffset = 10;
static const char* kKernelColor = "#FF0000";
static const char* kBufferColor = "#00FF00";
static const char* kImageColor = "#0000FF";
static const char* kOtherColor = "#FFFFFF";
#endif

void DeviceInfo::initialize(cl_device_id deviceID)
{
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_TYPE, 
                           Type);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_VENDOR_ID, 
                           VendorID);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_MAX_COMPUTE_UNITS, 
                           MaxComputeUnits);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                           MaxWorkGroupSize);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, 
                           PreferredVectorWidthChar);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, 
                           PreferredVectorWidthShort);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, 
                           PreferredVectorWidthInt);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, 
                           PreferredVectorWidthLong);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, 
                           PreferredVectorWidthFloat);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, 
                           PreferredVectorWidthDouble);
  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, 
                           PreferredVectorWidthHalf);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
                           NativeVectorWidthChar);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
                           NativeVectorWidthShort);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
                           NativeVectorWidthInt);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
                           NativeVectorWidthLong);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
                           NativeVectorWidthFloat);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
                           NativeVectorWidthDouble);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
                           NativeVectorWidthHalf);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MAX_CLOCK_FREQUENCY,
                           MaxClockFrequency);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_ADDRESS_BITS,
                           AddressBits);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                           MaxMemAllocSize);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_IMAGE_SUPPORT,
                           ImageSupport);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MAX_READ_IMAGE_ARGS,
                           MaxReadImageArgs);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
                           MaxWriteImageArgs);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_IMAGE2D_MAX_WIDTH,
                           Image2DMaxWidth);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                           Image2DMaxHeight);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_IMAGE3D_MAX_WIDTH,
                           Image3DMaxWidth);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                           Image3DMaxHeight);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_IMAGE3D_MAX_DEPTH,
                           Image3DMaxDepth);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MAX_SAMPLERS,
                           MaxSamplers);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MAX_PARAMETER_SIZE,
                           MaxParameterSize);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                           MemBaseAddrAlign);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
                           MinDataTypeAlignSize);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_SINGLE_FP_CONFIG,
                           SingleFPConfig);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                           GlobalMemCacheType);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                           GlobalMemCacheSize);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_GLOBAL_MEM_SIZE,
                           GlobalMemSize);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                           MaxConstantBufferSize);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_MAX_CONSTANT_ARGS,
                           MaxConstantArgs);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_LOCAL_MEM_TYPE,
                           LocalMemType);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_LOCAL_MEM_SIZE,
                           LocalMemSize);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                           ErrorCorrectionSupport);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_HOST_UNIFIED_MEMORY,
                           HostUnifiedMemory);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                           ProfilingTimerResolution);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_ENDIAN_LITTLE,
                           EndianLittle);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_AVAILABLE,
                           DeviceAvailable);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_COMPILER_AVAILABLE,
                           CompilerAvailable);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_EXECUTION_CAPABILITIES,
                           ExecutionCapabilities);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_QUEUE_PROPERTIES,
                           QueueProperties);

  //Now for the more exciting properties...
  //Work item dimensions
  cl_uint maxWorkItemDimensions;

  Utility::fetchDeviceInfo(deviceID, 
                           CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 
                           maxWorkItemDimensions);

  unique_ptr<size_t[]> maxWorkDimensions(new size_t[maxWorkItemDimensions]);

  clGetDeviceInfo(deviceID, 
                  CL_DEVICE_MAX_WORK_ITEM_SIZES,
                  sizeof(size_t) * maxWorkItemDimensions,
                  maxWorkDimensions.get(),
                  NULL);

  for(cl_uint curDimension = 0; 
      curDimension < maxWorkItemDimensions; 
      curDimension++)
  {
    this->MaxWorkItemSizes.push_back(maxWorkDimensions[curDimension]);
  }

  //String information
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_NAME,
                           Name);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_VENDOR,
                           Vendor);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DRIVER_VERSION,
                           DriverVersion);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_PROFILE,
                           OpenCLProfile);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_VERSION,
                           OpenCLVersion);
  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_OPENCL_C_VERSION,
                           OpenCLCVersion);

  string extensions;

  Utility::fetchDeviceInfo(deviceID,
                           CL_DEVICE_EXTENSIONS,
                           extensions);

  Extensions = Utility::split(extensions, ' ');

}

Device::Device(cl_device_id deviceID) : 
  mDeviceID(deviceID),
  mContext(NULL),
  mProgram(NULL),
  mKernels(),
  mInfoInitialized(false),
  mDeviceNumber(0),
  mProfileEvents(Device::NumCommandQueues),
  mCommandQueues(),
  mCurrentCommandQueue(0)
#ifdef CLUTIL_MAPBUFFER 
  , mPinnedBuffer(Device::NumCommandQueues),
  mBufferInUse(Device::NumCommandQueues)
#endif
{
  mDeviceInfo.initialize(mDeviceID);
}

void Device::FetchDevices()
{
  vector<size_t> embargoList;
  char* embargoEnvString;
  vector<Device> allDevices;

  //Get the device embargo list
  embargoEnvString = getenv("CLUTIL_DEVICE_EMBARGO");

  if(embargoEnvString != NULL)
  {
    vector<string> embargoStrings = 
      Utility::split(string(embargoEnvString), ':');

    for(auto curString = embargoStrings.begin();
        curString != embargoStrings.end();
        curString++)
    {
      size_t embargoValue;

      istringstream(*curString) >> embargoValue;
      
      embargoList.push_back(embargoValue);
    }
  }

  vector<Platform> platforms = Platform::GetPlatforms();
  
  //Collect all devices from all platforms
  for(auto curPlatform = platforms.begin(); 
      curPlatform != platforms.end(); 
      curPlatform++)
  {
    vector<Device> platformDevices = (*curPlatform).getDevices();
    
    allDevices.insert(allDevices.end(), 
                      platformDevices.begin(), 
                      platformDevices.end());

  }

  //Embargo devices
  for(size_t curDevice = 0; curDevice < allDevices.size(); curDevice++)
  {
    bool deviceIsEmbargoed = false;

    for(auto curEmbargo = embargoList.begin();
        curEmbargo != embargoList.end();
        curEmbargo++)
    {
      if(curDevice == *curEmbargo)
      {
        deviceIsEmbargoed = true;
        break;
      }
    }

    if(deviceIsEmbargoed == false)
    {
      Device::Devices.push_back(allDevices[curDevice]);
    }
  }

  //Number each device
  for(size_t curDevice = 0; curDevice < Device::Devices.size(); curDevice++)
  {
    Device::Devices[curDevice].mDeviceNumber = curDevice;
  }

  Device::DevicesFetched = true;

  initHelpers();
}

void Device::InitializeDevices(const char** filenames, 
                               size_t numFiles,
                               const char* cachename,
                               const char* options)
{
  //If we haven't initialized the devices, initialize them
  if(Device::DevicesInitialized == false)
  {
    //If we haven't fetched the devices, fetch them
    if(Device::DevicesFetched == false)
    {
      Device::FetchDevices();
      Device::DevicesFetched = true;
    }
    
    for(size_t curDevice = 0; curDevice < Device::Devices.size(); curDevice++)
    {
      Device::Devices[curDevice].initialize(filenames,
                                            numFiles,
                                            cachename,
                                            options);
    }

    Device::DevicesInitialized = true;
  }
}

void Device::StartProfiling()
{
  ProfilingStarted = true;

  //Put the starting event into each queue so we know when the profiling 
  //starts
  for(size_t curDeviceID = 0; curDeviceID < Devices.size(); curDeviceID++)
  {
    Device& curDevice = Devices[curDeviceID];
   
    for(size_t curQueueID = 0; 
        curQueueID < curDevice.mCommandQueues.size();
        curQueueID++)
    { 
      cl_int err;
      cl_event event;

      cl_command_queue queue = curDevice.mCommandQueues[curQueueID];

      err = clEnqueueMarker(queue, &event);
      clUtilCheckError(err);

      err = clFlush(queue);
      clUtilCheckError(err);

      curDevice.addProfilingEvent(event);

      err = clReleaseEvent(event);
      clUtilCheckError(err);
    }
  }
}


void Device::initialize(const char** filenames,
                        size_t numFiles,
                        const char* cachename,
                        const char* options)
{
  cl_int err;

  //Create context and command queue
  mContext = clCreateContext(NULL, 1, &mDeviceID, NULL, NULL, &err);
  clUtilCheckError(err);

  for(size_t curQueue = 0; curQueue < Device::NumCommandQueues; curQueue++)
  {
    cl_command_queue queue = clCreateCommandQueue(mContext, 
                                                  mDeviceID, 
#ifdef CLUTIL_ENABLE_PROFILING
                                                  CL_QUEUE_PROFILING_ENABLE,
#else
                                                  0, 
#endif
                                                  &err);
    clUtilCheckError(err);

    mCommandQueues.push_back(queue);
  }

  if(cachename != NULL)
  {
    err = this->loadBinary(cachename);
    
    //If we loaded the binary, then good. We'll use it.
    if(err == CL_SUCCESS)
    {
      return;
    }
    else
    {
      printf("Info: clUtil could not load binaries due to error: %s. Building "
             "program from source.\n",
             clUtilGetErrorCode(err));
    }
  }

  this->buildProgram(filenames, numFiles, options);
  this->getKernels();

#ifdef CLUTIL_MAPBUFFER
  for(size_t curQueue = 0; curQueue < Device::NumCommandQueues; curQueue++)
  {
    cl_mem pinnedBuffer = clCreateBuffer(mContext,
                                         CL_MEM_ALLOC_HOST_PTR,
                                         Device::PinnedBufferSize,
                                         NULL,
                                         &err);
    clUtilCheckError(err);

    mPinnedBuffer[curQueue] = pinnedBuffer;
    mBufferInUse[curQueue] = false;
  }

#endif
}

cl_kernel Device::getKernel(const std::string& kernelName) const
{
  auto kernel = mKernels.find(kernelName);

  if(kernel == mKernels.end())
  {
    clUtilCheckError(CL_INVALID_KERNEL_NAME);
  }

  return kernel->second;
}

void Device::addProfilingEvent(cl_event event)
{
#ifdef CLUTIL_ENABLE_PROFILING
  if(ProfilingStarted == true)
  {
    size_t commandQueueID = mCommandQueues.size();
    cl_int err;
    cl_command_queue queue;

    err = clGetEventInfo(event,
                         CL_EVENT_COMMAND_QUEUE,
                         sizeof(queue),
                         &queue,
                         NULL);
    clUtilCheckError(err);

    for(size_t curQueue = 0; curQueue < mCommandQueues.size(); curQueue++)
    {
      if(queue == mCommandQueues[curQueue])
      {
        commandQueueID = curQueue;
        break;
      }
    }

    if(commandQueueID == mCommandQueues.size())
    {
      throw clUtilException("Bad command queue for profiling event");
    }

    err = clRetainEvent(event);
    clUtilCheckError(err);

    mProfileEvents[commandQueueID].push_back(event);
  }
#endif
}

void Device::DumpProfilingData()
{
#ifdef CLUTIL_ENABLE_PROFILING
  if(ProfilingStarted == false)
  {
    throw clUtilException("Can't dump profiling data: not started");
  }

  //Put the ending event into each queue so we know when the profiling 
  //stops
  for(size_t curDeviceID = 0; curDeviceID < Devices.size(); curDeviceID++)
  {
    Device& curDevice = Devices[curDeviceID];
   
    for(size_t curQueueID = 0; 
        curQueueID < curDevice.mCommandQueues.size();
        curQueueID++)
    { 
      cl_int err;
      cl_event event;

      cl_command_queue queue = curDevice.mCommandQueues[curQueueID];

      err = clEnqueueMarker(queue, &event);
      clUtilCheckError(err);

      err = clFlush(queue);
      clUtilCheckError(err);

      curDevice.addProfilingEvent(event);

      err = clReleaseEvent(event);
      clUtilCheckError(err);
    }
  }

  Finish();

  ofstream outputFile("clUtilProfile.svg");

  outputFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
             << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\""
             << " \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n";
  outputFile << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\"" 
             << " xmlns:xlink=\"http://www.w3.org/1999/xlink\""
             << " width=\""
             << kTextAreaWidth + kGraphicAreaWidth 
             << "\" height=\""
             << kQueueGraphicHeight * Devices.size() * NumCommandQueues 
             << "\">\n";
 
  outputFile << "<rect width='"
             << kTextAreaWidth + kGraphicAreaWidth
             << "' height='"
             << kQueueGraphicHeight * Devices.size() * NumCommandQueues
             << "' style='fill:#FFFFFF'/>\n";

  for(size_t curDeviceID = 0; curDeviceID < Devices.size(); curDeviceID++)
  {
    Device& curDevice = Devices[curDeviceID];

    for(size_t curQueueID = 0; 
        curQueueID < curDevice.mProfileEvents.size(); 
        curQueueID++)
    {
      cl_int err;

      outputFile << "\t<text x='"
                 << kTextXOffset
                 << "' y='" 
                 << (curDeviceID * NumCommandQueues + curQueueID) 
                    * kQueueGraphicHeight + kTextYOffset
                 << "'>";
  
      outputFile << curDevice.mDeviceInfo.Name << " queue " << curQueueID;
      outputFile << "</text>" << endl;

      vector<cl_event>& profileSet = curDevice.mProfileEvents[curQueueID];

      cl_ulong queueStartTime;
      cl_ulong queueEndTime;
      bool shouldRenderEvents = true;

      //Get the queue's start time and release that event
      err = clGetEventProfilingInfo(profileSet[0],
                                    CL_PROFILING_COMMAND_START,
                                    sizeof(queueStartTime),
                                    &queueStartTime,
                                    NULL);
      
      if(err == CL_PROFILING_INFO_NOT_AVAILABLE)
      {
        shouldRenderEvents = false;
      }
      else
      {
        clUtilCheckError(err);
      }

      err = clReleaseEvent(profileSet[0]);
      clUtilCheckError(err);

      //Get the queue's stopping time and relase that event
      err = clGetEventProfilingInfo(profileSet[profileSet.size() - 1],
                                    CL_PROFILING_COMMAND_END,
                                    sizeof(queueEndTime),
                                    &queueEndTime,
                                    NULL);

      if(err == CL_PROFILING_INFO_NOT_AVAILABLE)
      {
        shouldRenderEvents = false;
      }
      else
      {
        clUtilCheckError(err);
      }
      
      err = clReleaseEvent(profileSet[profileSet.size() - 1]);
      clUtilCheckError(err);

      cl_ulong queueElapsedTime = queueEndTime - queueStartTime;

      for(size_t curEventNum = 1; 
          curEventNum < profileSet.size() - 1; 
          curEventNum++)
      {
        cl_event curEvent = profileSet[curEventNum];
        
        if(shouldRenderEvents == true)
        {
          cl_ulong startTime;
          cl_ulong stopTime;
          const char* eventColor;
          cl_command_type commandType;

          err = clGetEventProfilingInfo(curEvent,
                                        CL_PROFILING_COMMAND_START,
                                        sizeof(startTime),
                                        &startTime,
                                        NULL);
          clUtilCheckError(err);

          err = clGetEventProfilingInfo(curEvent,
                                        CL_PROFILING_COMMAND_END,
                                        sizeof(stopTime),
                                        &stopTime,
                                        NULL);
          clUtilCheckError(err);

          err = clGetEventInfo(curEvent,
                               CL_EVENT_COMMAND_TYPE,
                               sizeof(commandType),
                               &commandType,
                               NULL);
          clUtilCheckError(err);

          switch(commandType)
          {
            case CL_COMMAND_NDRANGE_KERNEL:
            case CL_COMMAND_NATIVE_KERNEL:
              eventColor = kKernelColor;
              break;
            case CL_COMMAND_READ_BUFFER:
            case CL_COMMAND_WRITE_BUFFER:
            case CL_COMMAND_COPY_BUFFER:
              eventColor = kBufferColor;
              break;
            case CL_COMMAND_READ_IMAGE:
            case CL_COMMAND_WRITE_IMAGE:
              eventColor = kImageColor;
              break;
            default:
              eventColor = kOtherColor;
              break;
          }

          outputFile << "\t\t<rect x='"
                     << kTextAreaWidth + (double)(startTime - queueStartTime) / 
                        (double)queueElapsedTime * kGraphicAreaWidth
                     << "' y='"
                     << (curDeviceID * NumCommandQueues + curQueueID) *
                        kQueueGraphicHeight
                     << "' width='"
                     << (double)(stopTime - startTime) / 
                        (double)queueElapsedTime * kGraphicAreaWidth
                     << "' height='"
                     << kQueueGraphicHeight
                     << "' style='stroke:#FFFFFF;fill:" << eventColor
                     << "'/>" << endl;
        }

        err = clReleaseEvent(curEvent);
        clUtilCheckError(err);
      }

      profileSet.clear();
    }
  }

  outputFile << "</svg>" << endl;
#endif
}

void Device::flush()
{
  for(size_t curQueue = 0; curQueue < mCommandQueues.size(); curQueue++)
  {
    cl_int err;

    err = clFlush(mCommandQueues[curQueue]);
    clUtilCheckError(err);
  }
}

void Device::finish()
{
  for(size_t curQueue = 0; curQueue < mCommandQueues.size(); curQueue++)
  {
    cl_int err;

    err = clFinish(mCommandQueues[curQueue]);
    clUtilCheckError(err);
  }
}

void Device::Flush()
{
  for(auto device = Devices.begin(); device < Devices.end(); device++)
  {
    device->flush();
  }
}

void Device::Finish()
{
  for(auto device = Devices.begin(); device < Devices.end(); device++)
  {
    device->finish();
  }
}

void Device::Finalize()
{
  finalizeHelpers();
}

