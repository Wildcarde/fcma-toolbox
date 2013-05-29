#include <clUtil.cl>

__constant sampler_t s0 = CLK_NORMALIZED_COORDS_FALSE | 
                     CLK_ADDRESS_CLAMP |
                     CLK_FILTER_NEAREST;

__constant sampler_t s1 = CLK_NORMALIZED_COORDS_FALSE | 
                     CLK_ADDRESS_CLAMP |
                     CLK_FILTER_NEAREST;

__kernel void fillImage(__write_only image1d_t theImage, int n)
{
  float4 color = {0.0f, 0.0f, 0.0f, 0.0f};

  if(get_global_id(0) < n)
  {
    color.w = get_global_id(0);

    write_1Dimagef(theImage, get_global_id(0), color);
  }
}


__kernel void sumImages(__read_only image1d_t a,
                        __read_only image1d_t b,
                        __write_only image1d_t c,
                        int n)
{
  float4 aVal;
  float4 bVal;
  float4 cVal;

  if(get_global_id(0) < n)
  {
    aVal = read_1Dimagef(a, s0, get_global_id(0));
    bVal = read_1Dimagef(b, s1, get_global_id(0));
    
    cVal = aVal + bVal;

    write_1Dimagef(c, get_global_id(0), cVal);
  }
}

