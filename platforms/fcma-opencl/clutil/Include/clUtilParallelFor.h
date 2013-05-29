#pragma once
#include <map>
#include <memory>
#include <math.h>
#include <functional>
#include "clUtilDeviceGroup.h"
#include "clUtilDevice.h"
#include "clUtilUtility.h"
#include "clUtilScheduler.h"

namespace clUtil
{
  void ParallelFor(const size_t start,
                   const size_t stride,
                   const size_t end,
                   std::function<void (size_t, size_t)> loopBody,
                   IScheduler&& model = StaticScheduler(30));
}
