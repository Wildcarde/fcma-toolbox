#include <clUtil.h>
#include <vector>

using namespace std;
using namespace clUtil;

const char* clUtilGetErrorCode(cl_int err)
{
  switch(err)
  {
    case CL_SUCCESS:
      return "No Error.";
    case CL_INVALID_MEM_OBJECT:
      return "Invalid memory object.";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "Invalid image format descriptor.";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "Image format not supported.";
    case CL_INVALID_IMAGE_SIZE:
      return "Invalid image size.";
    case CL_INVALID_ARG_INDEX:
      return "Invalid argument index for this kernel.";
    case CL_INVALID_ARG_VALUE:
      return "Invalid argument value.";
    case CL_INVALID_SAMPLER:
      return "Invalid sampler.";
    case CL_INVALID_ARG_SIZE:
      return "Invalid argument size.";
    case CL_INVALID_BUFFER_SIZE:
      return "Invalid buffer size.";
    case CL_INVALID_HOST_PTR:
      return "Invalid host pointer.";
    case CL_INVALID_DEVICE:
      return "Invalid device.";
    case CL_INVALID_VALUE:
      return "Invalid value.";
    case CL_INVALID_CONTEXT:
      return "Invalid Context.";
    case CL_INVALID_KERNEL:
      return "Invalid kernel.";
    case CL_INVALID_PROGRAM:
      return "Invalid program object.";
    case CL_INVALID_BINARY:
      return "Invalid program binary.";
    case CL_INVALID_OPERATION:
      return "Invalid operation.";
    case CL_INVALID_BUILD_OPTIONS:
      return "Invalid build options.";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "Invalid executable.";
    case CL_INVALID_COMMAND_QUEUE:
      return "Invalid command queue.";
    case CL_INVALID_KERNEL_ARGS:
      return "Invalid kernel arguments.";
    case CL_INVALID_WORK_DIMENSION:
      return "Invalid work dimension.";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "Invalid work group size.";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "Invalid work item size.";
    case CL_INVALID_GLOBAL_OFFSET:
      return "Invalid global offset (should be NULL).";
    case CL_OUT_OF_RESOURCES:
      return "Insufficient resources.";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "Could not allocate mem object.";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "Invalid event wait list.";
    case CL_OUT_OF_HOST_MEMORY:
      return "Out of memory on host.";
    case CL_INVALID_KERNEL_NAME:
      return "Invalid kernel name.";
    case CL_INVALID_KERNEL_DEFINITION:
      return "Invalid kernel definition.";
    case CL_BUILD_PROGRAM_FAILURE:
      return "Failed to build program.";
    case CL_MAP_FAILURE:
      return "Failed to map buffer/image";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "Profiling info not available.";
    case -1001: //This is CL_PLATFORM_NOT_FOUND_KHR
      return "No platforms found. (Did you put ICD files in /etc/OpenCL?)";
    default:
      return "Unknown error.";
  }
}