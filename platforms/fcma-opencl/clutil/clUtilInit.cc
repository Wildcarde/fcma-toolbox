#include "clUtilCommon.h"
#include "clUtilDevice.h"

using namespace clUtil;
using namespace std;

static cl_command_queue_properties commandQueueFlags = 0;

void clUtilEnableOOOExecution()
{
  commandQueueFlags |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
}
