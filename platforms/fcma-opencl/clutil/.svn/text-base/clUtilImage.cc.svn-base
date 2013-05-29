#include "clUtilDevice.h"
#include "clUtilCommon.h"
#include "clUtilException.h"
#include "clUtil_kernel.h"

using namespace clUtil;

cl_int clUtilPutImage1D(cl_mem image,
                        const size_t offset,
                        const size_t region,
                        void* ptr)
{
  void* mappedImage;
  size_t imageWidth;
  size_t imageHeight;
  size_t origin3D[3] = {0, 0, 0};
  size_t region3D[3] = {imageWidth, imageHeight, 1};
  size_t pitch;
  cl_int err;
  void* hostStartAddress;
  void* mappedImageStartAddress;
  size_t pixelSize;

  err = clGetImageInfo(image,
                       CL_IMAGE_WIDTH,
                       sizeof(imageWidth),
                       &imageWidth,
                       NULL);
  clUtilCheckError(err);

  err = clGetImageInfo(image,
                       CL_IMAGE_HEIGHT,
                       sizeof(imageHeight),
                       &imageHeight,
                       NULL);
  clUtilCheckError(err);

  err = clGetImageInfo(image,
                       CL_IMAGE_ELEMENT_SIZE,
                       sizeof(pixelSize),
                       &pixelSize,
                       NULL);
  clUtilCheckError(err);

  mappedImage = clEnqueueMapImage(Device::GetCurrentDevice().getCommandQueue(),
                                  image,
                                  CL_TRUE,
                                  CL_MAP_WRITE,
                                  origin3D,
                                  region3D,
                                  &pitch,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  &err);
  clUtilCheckError(err);

  hostStartAddress = (void*)((size_t)ptr + offset * pixelSize);
  mappedImageStartAddress = 
    (void*)((size_t)mappedImage + offset * pixelSize);

  memcpy(mappedImageStartAddress, 
         hostStartAddress, 
         region * pixelSize);

  err = clEnqueueUnmapMemObject(Device::GetCurrentDevice().getCommandQueue(),
                                image,
                                mappedImage,
                                0,
                                NULL,
                                NULL);
  clUtilCheckError(err);

  err = clEnqueueBarrier(Device::GetCurrentDevice().getCommandQueue());
  clUtilCheckError(err);

  return CL_SUCCESS;
}

cl_int clUtilGetImage1D(cl_mem image,
                        const size_t offset,
                        const size_t region,
                        void* ptr)
{
  void* mappedImage;
  size_t imageWidth;
  size_t imageHeight;
  size_t origin3D[3] = {0, 0, 0};
  size_t region3D[3] = {imageWidth, imageHeight, 1};
  size_t pitch = 0;
  cl_int err;
  void* hostStartAddress;
  void* mappedImageStartAddress;
  size_t pixelSize;

  err = clGetImageInfo(image,
                       CL_IMAGE_WIDTH,
                       sizeof(imageWidth),
                       &imageWidth,
                       NULL);
  clUtilCheckError(err);

  err = clGetImageInfo(image,
                       CL_IMAGE_HEIGHT,
                       sizeof(imageHeight),
                       &imageHeight,
                       NULL);
  clUtilCheckError(err);

  err = clGetImageInfo(image,
                       CL_IMAGE_ELEMENT_SIZE,
                       sizeof(pixelSize),
                       &pixelSize,
                       NULL);
  clUtilCheckError(err);

  mappedImage = clEnqueueMapImage(Device::GetCurrentDevice().getCommandQueue(),
                                  image,
                                  CL_TRUE,
                                  CL_MAP_READ,
                                  origin3D,
                                  region3D,
                                  &pitch,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  &err);
  clUtilCheckError(err);

  hostStartAddress = (void*)((size_t)ptr + offset * pixelSize);
  mappedImageStartAddress = 
    (void*)((size_t)mappedImage + offset * pixelSize);

  memcpy(hostStartAddress, 
         mappedImageStartAddress,
         region * pixelSize);

  err = clEnqueueUnmapMemObject(Device::GetCurrentDevice().getCommandQueue(),
                                image,
                                mappedImage,
                                0,
                                NULL,
                                NULL);
  clUtilCheckError(err);

  err = clEnqueueBarrier(Device::GetCurrentDevice().getCommandQueue());
  clUtilCheckError(err);

  return CL_SUCCESS;
}

cl_int clUtilCreateImage1D(size_t numPixels, 
                           cl_channel_order order,
                           cl_channel_type type,
                           cl_mem* image)
{
  size_t imageWidth;
  size_t imageHeight;
  cl_int err;
  cl_image_format format;

  format.image_channel_order = order;
  format.image_channel_data_type = type;

  //Create an image buffer with the next highest perfect square # elements
#if 1
  imageWidth = 8192;
  imageHeight = numPixels % imageWidth == 0 ? 
    numPixels / imageWidth :
    numPixels / imageWidth + 1;
#else
  for(unsigned int i = 0; i < 14; i++)
  {
    unsigned int size = 1 << i;

    if(size * size > numPixels)
    {
      imageHeight = size;
      imageWidth = size;
      break;
    }
  }

#endif

  *image = clCreateImage2D(Device::GetCurrentDevice().getContext(),
                           CL_MEM_READ_WRITE,
                           &format,
                           imageWidth,
                           imageHeight,
                           0,
                           NULL,
                           &err);
  clUtilCheckError(err);

  return CL_SUCCESS;
}

cl_int clUtilCopyToImageFloat(cl_mem buffer, 
                                int offset, 
                                int m, 
                                int n, 
                                int ld,
                                cl_mem* image)
{
  cl_int err;
  cl_image_format format;
  int imageWidth;

  imageWidth = m;

  //Set image format
  format.image_channel_order = CL_A;
  format.image_channel_data_type = CL_FLOAT;

  *image = clCreateImage2D(Device::GetCurrentDevice().getContext(),
                           CL_MEM_READ_WRITE,
                           &format,
                           imageWidth,
                           n,
                           0,
                           NULL,
                           &err);
  clUtilCheckError(err);
  
  clUtilEnqueueKernel("copyToImageFloat",
                      clUtilGrid(imageWidth, 8, n, 8),
                      buffer,
                      offset,
                      m,
                      n,
                      ld,
                      *image);
  
  return CL_SUCCESS;
}

cl_int clUtilCopyToImageTransposeFloat(cl_mem buffer, 
                                         int offset, 
                                         int m, 
                                         int n, 
                                         int ld,
                                         cl_mem* image)
{
  cl_int err;
  cl_image_format format;
  int imageWidth;

  imageWidth = n;

  //Set image format
  format.image_channel_order = CL_A;
  format.image_channel_data_type = CL_FLOAT;

  *image = clCreateImage2D(Device::GetCurrentDevice().getContext(),
                           CL_MEM_READ_WRITE,
                           &format,
                           imageWidth,
                           m,
                           0,
                           NULL,
                           &err);
  clUtilCheckError(err);
  
  clUtilEnqueueKernel("copyToImageTransposeFloat",
                      clUtilGrid(imageWidth, 8, m, 8),
                      buffer,
                      offset,
                      m,
                      n,
                      ld,
                      *image);
  
  return CL_SUCCESS;
}

cl_int clUtilCopyToImageDouble(cl_mem buffer, 
                                 int offset, 
                                 int m, 
                                 int n, 
                                 int ld,
                                 cl_mem* image)
{
  cl_int err;
  cl_image_format format;
  int imageWidth;

  imageWidth = m;

  //Set image format
  format.image_channel_order = CL_RG;
  format.image_channel_data_type = CL_FLOAT;

  *image = clCreateImage2D(Device::GetCurrentDevice().getContext(),
                           CL_MEM_READ_WRITE,
                           &format,
                           imageWidth,
                           n,
                           0,
                           NULL,
                           &err);
  clUtilCheckError(err);

  clUtilEnqueueKernel("copyToImageDouble",
                      clUtilGrid(imageWidth, 8, n, 8),
                      buffer,
                      offset,
                      m,
                      n,
                      ld,
                      *image);

  return CL_SUCCESS;
}

cl_int clUtilCopyToImageTransposeDouble(cl_mem buffer, 
                                          int offset, 
                                          int m, 
                                          int n, 
                                          int ld,
                                          cl_mem* image)
{
  cl_int err;
  cl_image_format format;
  int imageWidth;

  imageWidth = n;

  //Set image format
  format.image_channel_order = CL_RG;
  format.image_channel_data_type = CL_FLOAT;

  *image = clCreateImage2D(Device::GetCurrentDevice().getContext(),
                           CL_MEM_READ_WRITE,
                           &format,
                           imageWidth,
                           m,
                           0,
                           NULL,
                           &err);
  clUtilCheckError(err);
  
  clUtilEnqueueKernel("copyToImageTransposeDouble",
                      clUtilGrid(imageWidth, 8, m, 8),
                      buffer,
                      offset,
                      m,
                      n,
                      ld,
                      *image);



  return CL_SUCCESS;
}

cl_int clUtilCopyToImageFloat4(cl_mem buffer, 
                                 int offset, 
                                 int m, 
                                 int n, 
                                 int ld,
                                 cl_mem* image)
{
  cl_int err;
  cl_image_format format;
  int imageWidth;

  imageWidth = m % 4 == 0 ? m / 4 : m / 4 + 1;

  //Set image format
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = CL_FLOAT;

  *image = clCreateImage2D(Device::GetCurrentDevice().getContext(),
                           CL_MEM_READ_WRITE,
                           &format,
                           imageWidth,
                           n,
                           0,
                           NULL,
                           &err);
  clUtilCheckError(err);

  clUtilEnqueueKernel("copyToImageFloat4",
                      clUtilGrid(imageWidth, 8, n, 8),
                      buffer,
                      offset,
                      m,
                      n,
                      ld,
                      *image);
  
  return CL_SUCCESS;
}

cl_int clUtilCopyToImageTransposeFloat4(cl_mem buffer, 
                                          int offset, 
                                          int m, 
                                          int n, 
                                          int ld,
                                          cl_mem* image)
{
  cl_int err;
  cl_image_format format;
  int imageWidth;

  imageWidth = n % 4 == 0 ? n / 4 : n / 4 + 1;

  //Set image format
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = CL_FLOAT;

  *image = clCreateImage2D(Device::GetCurrentDevice().getContext(),
                           CL_MEM_READ_WRITE,
                           &format,
                           imageWidth,
                           m,
                           0,
                           NULL,
                           &err);
  clUtilCheckError(err);

  clUtilEnqueueKernel("copyToImageTransposeFloat4",
                      clUtilGrid(imageWidth, 8, m, 8),
                      buffer,
                      offset,
                      m,
                      n,
                      ld,
                      *image);

  return CL_SUCCESS;
}

cl_int clUtilCopyToImageDouble2(cl_mem buffer, 
                                  int offset, 
                                  int m, 
                                  int n, 
                                  int ld,
                                  cl_mem* image)
{
  cl_int err;
  cl_image_format format;
  int imageWidth;

  imageWidth = m % 2 == 0 ? m / 2 : m / 2 + 1;

  //Set image format
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = CL_FLOAT;

  *image = clCreateImage2D(Device::GetCurrentDevice().getContext(),
                           CL_MEM_READ_WRITE,
                           &format,
                           imageWidth,
                           n,
                           0,
                           NULL,
                           &err);
  clUtilCheckError(err);
  
  clUtilEnqueueKernel("copyToImageDouble2",
                      clUtilGrid(imageWidth, 8, n, 8),
                      buffer,
                      offset,
                      m,
                      n,
                      ld,
                      *image);

  return CL_SUCCESS;
}

cl_int clUtilCopyToImageTransposeDouble2(cl_mem buffer, 
                                           int offset, 
                                           int m, 
                                           int n, 
                                           int ld,
                                           cl_mem* image)
{
  cl_int err;
  cl_image_format format;
  int imageWidth;

  imageWidth = n % 2 == 0 ? n / 2 : n / 2 + 1;

  //Set image format
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = CL_FLOAT;

  *image = clCreateImage2D(Device::GetCurrentDevice().getContext(),
                           CL_MEM_READ_WRITE,
                           &format,
                           imageWidth,
                           m,
                           0,
                           NULL,
                           &err);
  clUtilCheckError(err);

  clUtilEnqueueKernel("copyToImageTransposeDouble2",
                      clUtilGrid(imageWidth, 8, m, 8),
                      buffer,
                      offset,
                      m,
                      n,
                      ld,
                      *image);
  
  return CL_SUCCESS;
}
