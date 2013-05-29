#include "clUtilScheduler.h"
#include <math.h>

using namespace clUtil;
using namespace clUtil::Utility;
using namespace std;

map<string, vector<size_t>> PINAScheduler::ChunkSizeCache;
bool PINAScheduler::AutotuningInProgress = false;
static double kPerformanceCutoff = 0.9;
static double kAutotuningCutoffSpeedup = 1.1;
static const char* autotuningDirName = "autotuning";

struct AutotuningField
{
  size_t Size;
  double Time;
};

PINAScheduler::PINAScheduler(const char* loopName, size_t numSamples) :
  mTasksRemaining(numSamples),
  mCurrentSample(DeviceGroupInfo::Get().numGroups(), 0),
  mModel(DeviceGroupInfo::Get().numGroups(), vector<Sample>(numSamples)),
  mMeanIterationTime(DeviceGroupInfo::Get().numGroups()),
  mIterationsRemaining(0),
  mNumSamples(numSamples),
  mChunkSize(DeviceGroupInfo::Get().numGroups(), 1),
  mLoopName(loopName)
{
  
  size_t numDeviceGroups = DeviceGroupInfo::Get().numGroups();

  //If there doesn't exist an autotuning directory, make one and begin 
  //autotuning
  if(directoryExists(autotuningDirName))
  {
    //Make directory with 775 permissions. Throw exception if you can't
    if(makeDirectory(autotuningDirName))
    {
      throw clUtilException("Could not make 'autotuning' directory. Please "
                            "ensure you have write access to the current "
                            "directory");
    }
    
    if(directoryExists(autotuningDirName) == false)
    {
      throw clUtilException("Could not stat directory we just made. Please "
                            "report this as a WTF.");
    }
  }

  //Find the device that needs tuning
  for(size_t curDeviceGroupID = 0; 
      curDeviceGroupID < numDeviceGroups;
      curDeviceGroupID++)
  {
    stringstream filename;

#if defined (__WIN32) || defined(WIN32)
    filename << "autotuning\\" << loopName << curDeviceGroupID;
#else
    filename << "autotuning/" << loopName << curDeviceGroupID;
#endif
    //Check for autotuning/deviceID
    if(fileExists(filename.str().c_str()) == false)
    {
      mAutotuningMode = true;
      mAutotuningDeviceGroup = curDeviceGroupID;

      //Check for the partial autotuning (autotuning/deviceID.part)
      stringstream partialFilename;

#if defined (__WIN32) || defined(WIN32)
      partialFilename << "autotuning\\" << loopName 
        << curDeviceGroupID << ".part";
#else
      partialFilename << "autotuning/" << loopName 
        << curDeviceGroupID << ".part";
#endif

      //If not found, write a new file with exponent value 0
      if(fileExists(partialFilename.str().c_str()))
      {
        fstream partialFile(partialFilename.str().c_str(), 
                            fstream::out | fstream::binary);

        size_t lastPow2 = 1;

        partialFile.write((char*)&lastPow2, sizeof(lastPow2));

        if(partialFile.fail() == true)
        {
          throw clUtilException("Couldn't write autotuning file: malformed");
        }

        partialFile.close();
      }
      else //If file is found, read the last power of 2 we benchmarked
      {
        fstream partialFile(partialFilename.str().c_str(), 
                            fstream::in | fstream::binary);

        size_t lastPow2;

        partialFile.read((char*)&lastPow2, sizeof(lastPow2));

        if(partialFile.fail())
        {
          throw clUtilException("Couldn't read autotuning file: malformed");
        }

        partialFile.close();

        mChunkSize[curDeviceGroupID] = lastPow2 * 2;

      }

      return;
    }
  }

  //If we're done autotuning, load the tuning data and compute the best chunk
  //size
  mAutotuningMode = false;

  string loopNameStr = string(loopName);
  
  //If we've already cached the preferred chunk sizes, load them from cache
  auto cachedSizes = ChunkSizeCache.find(loopNameStr);

  if(cachedSizes != ChunkSizeCache.end())
  {
    mChunkSize = cachedSizes->second;
  }
  else //If loop chunk sizes aren't cached, load them then cache them
  {
    for(size_t curDeviceGroupID = 0; 
        curDeviceGroupID < numDeviceGroups; 
        curDeviceGroupID++)
    {
      stringstream filename;
      filename << "autotuning/" << loopName << curDeviceGroupID;

      mChunkSize[curDeviceGroupID] = computeChunkSize(filename.str());
    }
  }
}

size_t PINAScheduler::computeChunkSize(string filename)
{
  vector<AutotuningField> tuningData;
  AutotuningField curField;
  size_t lastPow2;

  fstream file(filename.c_str(), fstream::in | fstream::binary);

  file.read((char*)&lastPow2, sizeof(lastPow2));

  while(file.eof() == false)
  {
    file.read((char*)&curField, sizeof(curField));

#if 0
    if(file.fail() == true)
    {
      throw clUtilException("Malformed autotuning file. Please delete "
                            "files in 'autotuning/' and rerun tuning "
                            "procedure.");
    }
#endif

    tuningData.push_back(curField);

  }

  //Normalize the data to the maximum field
  double maxRate = 0.0;

  for(size_t curField = 0; curField < tuningData.size(); curField++)
  {
    maxRate = tuningData[curField].Time > maxRate ? 
      tuningData[curField].Time : 
      maxRate;
  }

  for(size_t curField = 0; curField < tuningData.size(); curField++)
  {
    tuningData[curField].Time /= maxRate;
  }

  //Compute rate constant via simple linear regression with no intercept
  double meanXY = 0.0;
  double meanXSq = 0.0;

  for(size_t curField = 0; curField < tuningData.size(); curField++)
  {
    double curC = 1 << curField;

    meanXSq += curC * curC;
    meanXY += curC * log(1.001 - tuningData[curField].Time);
  }

  double rate = meanXY / meanXSq;

  cout << rate << endl;

  return log(1.0 - kPerformanceCutoff) / rate;
}

void PINAScheduler::setRange(IndexRange& range)
{
  IScheduler::setRange(range);

  mIterationsRemaining = range.End - range.Start + 1;

  size_t sampleSize = mIterationsRemaining / mNumSamples;

  //This isn't so straightforward because of edge cases
  for(size_t curSample = 0; curSample < mNumSamples; curSample++)
  {
    IndexRange work;
   
    //The first sample needs 0 for its start index
    if(curSample == 0)
    {
      work.Start = range.Start;
    }
    else //Other samples pick up where the last one left off
    {
      work.Start = mTasksRemaining[curSample - 1].End + 1;
    }

    //The last sample needs the remaining iterations. Alternatively, if
    //previous sample consumed all iterations, then we need Start > End for
    //this one (e.g. NULL work)
    if(curSample == mNumSamples - 1 || 
       (curSample > 0 && mTasksRemaining[curSample - 1].End == range.End))
    {
      work.End = range.End;
    }
    else
    {
      if(work.Start + sampleSize < range.End)
      {
        work.End = work.Start + sampleSize;
      }
      else
      {
        work.End = range.End;
      }
    }

    mTasksRemaining[curSample] = work;
  }
}

bool PINAScheduler::workRemains(size_t deviceGroup) const
{
  //If we're autotuning, only give work to the device we're tuning
  if(mAutotuningMode == true && deviceGroup != mAutotuningDeviceGroup)
  {
    return false;
  }
  
  return mIterationsRemaining > 0;
}

IndexRange PINAScheduler::getWork(const size_t deviceGroup)
{
  IndexRange work;

  size_t& currentSampleRef = mCurrentSample[deviceGroup];
  size_t chunkSize = mChunkSize[deviceGroup];

  //If we haven't sampled everything with the current device, pull work from
  //The next sampleing region
  while(currentSampleRef < mNumSamples)
  {
    //If this work region has no more work, move on
    if(mTasksRemaining[currentSampleRef].End < 
       mTasksRemaining[currentSampleRef].Start)
    {
      currentSampleRef++;
      continue;
    }
    else
    {
      size_t iterationsRemaining = mTasksRemaining[currentSampleRef].End -
                                   mTasksRemaining[currentSampleRef].Start + 1;

      work.Start = mTasksRemaining[currentSampleRef].Start;
      work.End = chunkSize < iterationsRemaining ?
                 work.Start + chunkSize - 1:
                 mTasksRemaining[currentSampleRef].End;

      mTasksRemaining[currentSampleRef].Start = work.End + 1;

      currentSampleRef++;

      mIterationsRemaining -= work.End - work.Start + 1;

      return work;
    }
  }

  //Find the region of work where this device's norm speedup is greatest
  size_t bestSample = 0;
  double bestSpeedup = -1.0;

  for(size_t curSample = 0; curSample < mNumSamples; curSample++)
  {
    //Ignore this sample if there's no work left
    if(mTasksRemaining[curSample].Start > mTasksRemaining[curSample].End)
    {
      continue;
    }

    const Sample& thisSample = mModel[deviceGroup][curSample];
    double normSpeedup = 0.0;

    for(size_t curGroupID = 0; 
        curGroupID < DeviceGroupInfo::Get().numGroups();
        curGroupID++)
    {
      //Don't look at speedup over ourself
      if(curGroupID == deviceGroup) { continue; }
      
      double thatTime;

      if(curSample < mNumSamples - 1)
      {
        thatTime = interpolate(mModel[curGroupID][curSample],
                               mModel[curGroupID][curSample + 1],
                               thisSample.Index);
      }
      else //Last sample region uses the mean as the right sample
      {
        Sample meanSample;
        double meanTime = mMeanIterationTime[curGroupID].TotalTime /
                          mMeanIterationTime[curGroupID].IterationsCompleted;

        meanSample.Index = mTasksRemaining[curSample].End;
        meanSample.Time = meanTime;

        thatTime = interpolate(mModel[curGroupID][curSample],
                               meanSample,
                               thisSample.Index);
      }

      double speedup = thatTime / thisSample.Time;

      normSpeedup += speedup * speedup;
    }

    normSpeedup = sqrt(normSpeedup);

    if(normSpeedup > bestSpeedup)
    {
      bestSpeedup = normSpeedup;
      bestSample = curSample;
    }
  }

  //Get work from the best sample
  work.Start = mTasksRemaining[bestSample].Start;
  work.End = work.Start + chunkSize <= mTasksRemaining[bestSample].End ?
    work.Start + chunkSize :
    mTasksRemaining[bestSample].End;

  mTasksRemaining[bestSample].Start = work.End + 1;

  mIterationsRemaining -= work.End - work.Start + 1;

  return work;
}

double PINAScheduler::interpolate(const Sample& s0, 
                                  const Sample& s1, 
                                  size_t index)
{
  double slope = (s1.Time - s0.Time) / (double)(s1.Index - s0.Index);

  //If slope is NAN because of 0.0 / 0.0...
  if(slope != slope)
  {
    slope = 0.0;
  }

  return ((double)index - (double)s0.Index) * slope + s0.Index;
}

void PINAScheduler::updateModel(const DeviceStatus& status)
{
  size_t groupID = DeviceGroupInfo::Get()[status.DeviceID];
  size_t count = status.Range.End - status.Range.Start + 1;
  double time = status.Time2 - status.Time1;

  vector<Sample>& deviceModel = mModel[groupID];

  //Find the slot where our work went
  for(size_t currentSample = 0; 
      currentSample < mTasksRemaining.size(); 
      currentSample++)
  {
    if(status.Range.End <= mTasksRemaining[currentSample].End)
    {
      deviceModel[currentSample].Index = 
        (status.Range.End + status.Range.Start) / 2;

      deviceModel[currentSample].Time = time / count;

      break;
    }
  }

  GroupTimingInfo& meanTime = mMeanIterationTime[groupID];

  meanTime.TotalTime += time;
  meanTime.IterationsCompleted += count;
}

PINAScheduler::~PINAScheduler()
{
  //If we're in autotuning mode, dump our performance data
  if(mAutotuningMode == true)
  {
    stringstream filename;

    filename << "autotuning/" << mLoopName << mAutotuningDeviceGroup << ".part";

    //Read the previous performance
    double lastPerformance = 0.0;
    fstream file;

    if(mChunkSize[mAutotuningDeviceGroup] > 1)
    {
      file.open(filename.str().c_str(), fstream::binary | fstream::in);

      file.seekg(-8, fstream::end);
      file.clear();

      file.read((char*)&lastPerformance, sizeof(lastPerformance));

      file.close();
    }

    double rate = 1.0 / mMeanIterationTime[mAutotuningDeviceGroup].TotalTime;

    if(mChunkSize[mAutotuningDeviceGroup] == 1 ||
       rate / lastPerformance > kAutotuningCutoffSpeedup)
    {
      //Write the chunk size we tested
      file.open(filename.str().c_str(), 
                fstream::binary | fstream::out | fstream::ate | fstream::in);

      file.seekg(0, fstream::beg);
      file.clear();

      file.write((char*)&mChunkSize[mAutotuningDeviceGroup], 
                 sizeof(mChunkSize[0]));

      file.close();

      //Close the file, reopen as append, and add our timing data
      file.open(filename.str().c_str(), 
                fstream::binary | fstream::app | fstream::out);

      file.write((char*)&mChunkSize[mAutotuningDeviceGroup], 
                 sizeof(mChunkSize[0]));
      file.write((char*)&rate, sizeof(rate));

      file.close();
    }

    //If this chunk size yielded negligible performance improvement, stop
    //autotuning
    if(mChunkSize[mAutotuningDeviceGroup] > 1 && 
       rate / lastPerformance < kAutotuningCutoffSpeedup)
    {
      stringstream finalName;

      finalName << "autotuning/" << mLoopName << mAutotuningDeviceGroup;

      rename(filename.str().c_str(), finalName.str().c_str()); 
    }
  }
}
