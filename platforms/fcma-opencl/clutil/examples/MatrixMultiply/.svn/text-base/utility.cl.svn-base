#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void bzero(__global float* a, unsigned int count)
{
  if(get_global_id(0) < count)
  {
    a[get_global_id(0)] = 0.0f;
  }
}

__kernel void acc(__global float* a, __global float* b, unsigned int count)
{
  if(get_global_id(0) < count)
  {
    a[get_global_id(0)] += b[get_global_id(0)];
  }
}

__kernel void bzero_double(__global double* a, unsigned int count)
{
  if(get_global_id(0) < count)
  {
    a[get_global_id(0)] = 0.0;
  }
}

__kernel void acc_double(__global double* a, 
                         __global double* b, 
                         unsigned int count)
{
  if(get_global_id(0) < count)
  {
    a[get_global_id(0)] += b[get_global_id(0)];
  }
}
