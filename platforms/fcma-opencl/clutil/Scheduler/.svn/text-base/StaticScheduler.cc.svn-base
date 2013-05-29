#include "clUtilScheduler.h"

using namespace std;
using namespace clUtil;
using namespace clUtil::Utility;

IndexRange StaticScheduler::getWork(const size_t deviceGroup)
{
  IndexRange range;

  range.Start = mNextIteration;
  range.End = range.Start + mChunkSize > mEnd ? mEnd : range.Start + mChunkSize;

  mNextIteration += mChunkSize + 1;

  return range;
}

void StaticScheduler::updateModel(const Utility::DeviceStatus& status)
{
}

bool StaticScheduler::workRemains(size_t deviceGroup) const
{
  return mNextIteration <= mEnd;
}

void StaticScheduler::setRange(IndexRange& range)
{
  IScheduler::setRange(range);

  mChunkSize = (range.End - range.Start) / mNumChunks;

  if(mChunkSize < 1)
  {
    mChunkSize = 1;
  }
}
