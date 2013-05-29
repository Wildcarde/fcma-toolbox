#ifndef CLUTIL_IMAGE1D_CL
#define CLUTIL_IMAGE1D_CL

#if 0
#ifndef NVIDIA_CORPORATION
typedef image2d_t image1d_t;
#else
#define image1d_t image2d_t
#endif
#endif

#define image1d_t image2d_t

float4 read_1Dimagef(read_only image1d_t image,
                     sampler_t sampler,
                     int coord)
{
  int2 imageDim = get_image_dim(image);
  int2 sampleCoord;

  sampleCoord.x = coord % imageDim.x;
  sampleCoord.y = coord / imageDim.x;

  return read_imagef(image, sampler, sampleCoord);
}

int4 read_1Dimagei(read_only image1d_t image,
                   sampler_t sampler,
                   int coord)
{
  int2 imageDim = get_image_dim(image);
  int2 sampleCoord;

  sampleCoord.x = coord % imageDim.x;
  sampleCoord.y = coord / imageDim.x;

  return read_imagei(image, sampler, sampleCoord);
}

uint4 read_1Dimageui(read_only image1d_t image,
                     sampler_t sampler,
                     int coord)
{
  int2 imageDim = get_image_dim(image);
  int2 sampleCoord;

  sampleCoord.x = coord % imageDim.x;
  sampleCoord.y = coord / imageDim.x;

  return read_imageui(image, sampler, sampleCoord);
}

void write_1Dimagef(write_only image1d_t image,
                    int coord,
                    float4 color)
{
  int2 imagedim = get_image_dim(image);
  int2 samplecoord;

  samplecoord.x = coord % imagedim.x;
  samplecoord.y = coord / imagedim.x;

  write_imagef(image, samplecoord, color);
}

void write_1Dimagei(write_only image1d_t image,
                    int coord,
                    int4 color)
{
  int2 imagedim = get_image_dim(image);
  int2 samplecoord;

  samplecoord.x = coord % imagedim.x;
  samplecoord.y = coord / imagedim.x;

  write_imagei(image, samplecoord, color);
}

void write_1Dimageui(write_only image1d_t image,
                     int coord,
                     uint4 color)
{
  int2 imagedim = get_image_dim(image);
  int2 samplecoord;

  samplecoord.x = coord % imagedim.x;
  samplecoord.y = coord / imagedim.x;

  write_imageui(image, samplecoord, color);
}

#endif

