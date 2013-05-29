#include <clUtil.cl>

__kernel void sort(__global unsigned int* keys1,
                   __global unsigned int* keys2,
                   __global unsigned int* vals1,
                   __global unsigned int* vals2,
                   unsigned int size)
{
  __local unsigned int sortTmp1;
  __local unsigned int sortTmp2;
  __local unsigned int sortTmp3[64];
  __local unsigned int sortTmp4[64];

  keys1 = &keys1[size * get_global_id(1)];
  keys2 = &keys2[size * get_global_id(1)];
  vals1 = &vals1[size * get_global_id(1)];
  vals2 = &vals2[size * get_global_id(1)];

  radixSortLG(keys1, 
              vals1, 
              keys2, 
              vals2,
              size,
              &sortTmp1,
              &sortTmp2,
              sortTmp3,
              sortTmp4);
  
  radixSortGL(keys2, 
              vals2, 
              keys1, 
              vals1,
              size,
              &sortTmp1,
              &sortTmp2,
              sortTmp3,
              sortTmp4);
}
