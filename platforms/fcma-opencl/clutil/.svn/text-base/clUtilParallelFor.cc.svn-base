#include "clUtilParallelFor.h"
#include <math.h>

#define STRINGIFY(arg) #arg
#define __WHERE__ __FILE__ ":" STRINGIFY(__LINE__)

using namespace std;
using namespace clUtil;
using namespace clUtil::Utility;

bool DeviceGroupInfo::singletonInitialized = false;
DeviceGroupInfo DeviceGroupInfo::deviceGroupInfoSingleton;

void clUtil::ParallelFor(const size_t start, 
                         const size_t stride, 
                         const size_t end, 
                         function<void (size_t, size_t)> loopBody,
                         IScheduler&& model)
{
  size_t oldDeviceNum = Device::GetCurrentDeviceNum();
  size_t iterationsRemaining = end - start + 1;

  IndexRange range;
  range.Start = start;
  range.End = end;

  model.setRange(range);

  //Initialize device statuses
  vector<DeviceStatus> deviceStatuses(Device::GetDevices().size());

  for(size_t curDeviceID = 0; 
      curDeviceID < deviceStatuses.size(); 
      curDeviceID++)
  {
    deviceStatuses[curDeviceID].DeviceID = curDeviceID;
  }

  //Parallel for scheduling loop
  while(iterationsRemaining > 0)
  {
    for(size_t curDeviceID = 0; 
        curDeviceID < Device::GetDevices().size(); 
        curDeviceID++)
    {
      DeviceStatus& curDeviceStatus = deviceStatuses[curDeviceID];
      size_t deviceGroup = DeviceGroupInfo::Get()[curDeviceID];

      //If this device isn't busy, get some work from the model and run it
      if(curDeviceStatus.IsBusy == false && 
         model.workRemains(deviceGroup) == true)
      {
        IndexRange work;

        work = model.getWork(deviceGroup);

        Device::SetCurrentDevice(curDeviceID);
        Device& curDevice = Device::GetCurrentDevice();

        curDeviceStatus.Range = work;
        curDeviceStatus.Time1 = getTime();
        curDeviceStatus.IsBusy = true;

        loopBody(work.Start, work.End);

        //We indicate this device is finished by enqueueing markers into
        //every queues, then enqueueing a waitForEvents which depends on the
        //markers. Then enqueue one more marker so we can capture its event
        size_t prevCommandQueue = curDevice.getCommandQueueID();

        unique_ptr<cl_event[]> 
          markerList(new cl_event[curDevice.getNumCommandQueues()]); 

        for(size_t curQueueID = 0; 
            curQueueID < curDevice.getNumCommandQueues();
            curQueueID++)
        {
          curDevice.setCommandQueue(curQueueID);

          clEnqueueMarker(curDevice.getCommandQueue(),
                          &markerList[curQueueID]);
        }

        curDevice.setCommandQueue(prevCommandQueue);

        clEnqueueWaitForEvents(curDevice.getCommandQueue(),
                               curDevice.getNumCommandQueues(),
                               markerList.get());

        clEnqueueMarker(curDevice.getCommandQueue(), 
                        &curDeviceStatus.WaitEvent);

        //Release the horses from the gates
        curDevice.flush();
      }

      if(curDeviceStatus.WaitEvent != NULL) //If device has a valid event...
      {
        //Poll the event for completion
        cl_int eventStatus;

        cl_int err = clGetEventInfo(curDeviceStatus.WaitEvent,
                                    CL_EVENT_COMMAND_EXECUTION_STATUS,
                                    sizeof(eventStatus),
                                    &eventStatus,
                                    NULL);
        if(err != CL_SUCCESS)
        {
          throw clUtilException("ParallelFor internal error: could not get "
                                "event info " __WHERE__ "\n");
        }

        //If done, mark this device as available, notify the model, and
        //reset this device for more work
        if(eventStatus == CL_COMPLETE)
        {
          curDeviceStatus.Time2 = getTime();
          model.updateModel(curDeviceStatus);

          curDeviceStatus.IsBusy = false;

          clReleaseEvent(curDeviceStatus.WaitEvent);
          curDeviceStatus.WaitEvent = NULL;

          iterationsRemaining -= curDeviceStatus.Range.End - 
                                 curDeviceStatus.Range.Start + 
                                 1;

          //cout << "Iterations remaining " << iterationsRemaining << endl;
        }
      }
    }
  }

  Device::SetCurrentDevice(oldDeviceNum);
}
