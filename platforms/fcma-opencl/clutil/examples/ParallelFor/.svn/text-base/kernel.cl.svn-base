__kernel void vectorAdd(__global float* a, 
                        __global float* b,
                        __global float* c,
                        unsigned int length)
{
  unsigned int id = get_global_id(0);

  if(id < length)
  {
    c[id] = a[id] + b[id];
  }
}
