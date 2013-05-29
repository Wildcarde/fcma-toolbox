#include "clUtilScheduler.h"

using namespace std;
using namespace clUtil;
using namespace clUtil::Utility;

EGSScheduler::EGSScheduler() :
  IScheduler(),
  mTasks(clUtil::DeviceGroupInfo::Get().numGroups()),
  mTasksValid(clUtil::DeviceGroupInfo::Get().numGroups(), false),
  mPerformanceRank(clUtil::DeviceGroupInfo::Get().numGroups()),
  mNextIteration(0)
{
  size_t numDeviceGroups = clUtil::DeviceGroupInfo::Get().numGroups();

  std::vector<unsigned int> performance(numDeviceGroups);

  const DeviceGroupInfo& groupInfo = DeviceGroupInfo::Get();

  //Compute a heuristic (e.g. bullshit) performance metric for each
  //device group
  for(size_t curDeviceNum = 0; 
      curDeviceNum < Device::GetDevices().size();
      curDeviceNum++)
  {
    const Device& curDevice = Device::GetDevices()[curDeviceNum];
    const DeviceInfo& deviceInfo = curDevice.getDeviceInfo();

    unsigned int estimatedPerformance = 1;

    estimatedPerformance *= deviceInfo.Type == CL_DEVICE_TYPE_GPU ? 16 : 1;
    estimatedPerformance *= deviceInfo.MaxComputeUnits;

    performance[groupInfo[curDeviceNum]] = estimatedPerformance;
  }

  //Now actually rank the performances
  for(size_t performanceRank = 0;
      performanceRank < numDeviceGroups;
      performanceRank++)
  {
    unsigned int maxPerformance = 0;
    unsigned int maxPerformanceIdx = 0;

    //Find the maximum performance
    for(size_t i = 0; i < performance.size(); i++)
    {
      if(performance[i] > maxPerformance)
      {
        maxPerformance = performance[i];
        maxPerformanceIdx = i;
      }
    }

    mPerformanceRank[maxPerformanceIdx] = performanceRank;
    performance[maxPerformanceIdx] = 0;
  }
}

IndexRange EGSScheduler::getWork(const size_t deviceGroup)
{
  IndexRange work = mTasks[mPerformanceRank[deviceGroup]];

  //Move the rest of the tasks up in line
  for(size_t curTask = mPerformanceRank[deviceGroup]; 
      curTask < mTasks.size() - 1;
      curTask++)
  {
    mTasks[curTask] = mTasks[curTask + 1];
  }

  //Replace the last task if work remains
  if(mNextIteration <= mEnd)
  {
    size_t count = (mEnd - mNextIteration + 1) / Device::GetDevices().size(); 
    IndexRange newWork;

    count = count == 0 ? 1 : count;

    newWork.Start = mNextIteration;
    newWork.End = mNextIteration + count - 1;

    mNextIteration = newWork.End + 1;

    mTasks[mTasks.size() - 1] = newWork;
  }
  else
  {
    size_t lastTrueIdx = mTasksValid.size() - 1;

    while(1)
    {
      if(lastTrueIdx == 0)
      {
        mTasksValid[lastTrueIdx] = false;
        break;
      }

      if(mTasksValid[lastTrueIdx] == true)
      {
        mTasksValid[lastTrueIdx] = false;
        break;
      }

      lastTrueIdx--;
    }
  }

  return work;
}

void EGSScheduler::setRange(IndexRange& range)
{
  IScheduler::setRange(range);

  size_t end = range.Start;
  size_t remaining = range.End - range.Start + 1;
  size_t numDevices = Device::GetDevices().size();

  //Assign initial work
  for(unsigned int curWorkIdx = 0; curWorkIdx < mTasks.size(); curWorkIdx++)
  {
    IndexRange work;
    size_t count = remaining / numDevices;
    count = count == 0 ? 1 : count;

    work.Start = end;
    work.End = work.Start + count - 1;

    mTasks[curWorkIdx] = work;
    mTasksValid[curWorkIdx] = true;

    end = work.End + 1;

    remaining = range.End - end + 1;

    if(end > range.End)
    {
      break;
    }
  }

  mNextIteration = end;
}

void EGSScheduler::updateModel(const DeviceStatus& status)
{
}

bool EGSScheduler::workRemains(size_t deviceGroup) const
{
  return mTasksValid[mPerformanceRank[deviceGroup]];
}
