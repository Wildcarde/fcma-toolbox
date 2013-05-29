#ifndef CLUTIL_SORT
#define CLUTIL_SORT
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

unsigned int _log2(unsigned int num)
{
  switch(num)
  {
    case 1:
      return 0;
    case 2:
      return 1;
    case 4:
      return 2;
    case 8:
      return 3;
    case 16:
      return 4;
    case 32:
      return 5;
    case 64:
      return 6;
    case 128:
      return 7;
    case 256:
      return 8;
    case 512:
      return 9;
    default:
      return 0;
  }
}

void scanSum(__local unsigned int* array)
{
  if(get_local_id(0) == 0)
  {
    for(unsigned int i = 1; i < get_local_size(0); i++)
    {
      array[i] += array[i - 1];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
#if 0
  //Don't know why but this code crashes Radeon cards

  unsigned int logThreads = _log2(get_local_size(0));

  for(unsigned int i = 0; i < logThreads; i++)
  {
    unsigned int newVal = array[get_local_id(0)];

    //If threadNum > 2^i
    if(get_local_id(0) >= (1 << i))
    {
      newVal += array[get_local_id(0) - (1 << i)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    array[get_local_id(0)] = newVal;
    
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#endif
}

void sum(__global unsigned int* array, 
         __local unsigned int* totalSum,
         unsigned int len)
{
  unsigned int partialSum = 0;

  if(get_global_id(0) == 0)
  {
    *totalSum = 0;
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  for(unsigned int i = get_local_id(0); i < len; i += get_local_size(0))
  {
    partialSum += array[i];
  }

  //Sum partial results
  atomic_add(totalSum, partialSum);

  barrier(CLK_LOCAL_MEM_FENCE);
}

void arrayMax(__global unsigned int* array,
         __local unsigned int* totalMax,
         unsigned int len)
{
  unsigned int partialMax = 0;

  if(get_global_id(0) == 0)
  {
    *totalMax = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for(unsigned int i = get_local_id(0); i < len; i += get_local_size(0))
  {
    unsigned int curVal = array[i];

    if(curVal > partialMax)
    {
      partialMax = curVal;
    }
  }

  //Put the max of the maxes in the return value
  atomic_max(totalMax, partialMax);

  barrier(CLK_LOCAL_MEM_FENCE);
}

//Radix sort array from least to greatest
void radixSortLG(__global unsigned int* key,
                 __global unsigned int* val,
                 __global unsigned int* keyTmp,
                 __global unsigned int* valTmp,
                 unsigned int len,
                 __local unsigned int* zerosIndex,
                 __local unsigned int* onesIndex,
                 __local unsigned int* zerosScan,
                 __local unsigned int* onesScan)
{
  for(unsigned int i = 0; i < 32; i++)
  {
    unsigned int mask = 1 << i;
    __global unsigned int* curKeyArrayRead = i % 2 == 0 ? key : keyTmp;
    __global unsigned int* curKeyArrayWrite = i % 2 == 0 ? keyTmp : key;
    __global unsigned int* curValArrayRead = i % 2 == 0 ? val : valTmp;
    __global unsigned int* curValArrayWrite = i % 2 == 0 ? valTmp : val;

    if(get_local_id(0) == 0)
    {
      *zerosIndex = 0;
      *onesIndex = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //Calculate number of zeros so 
    for(unsigned int j = get_local_id(0); 
        j < len; 
        j += get_local_size(0))
    {
      if((curKeyArrayRead[j] & mask) == 0x0)
      {
        atomic_inc(onesIndex);
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //Round to next multiple of local size so we can still execute barriers
    unsigned int size = len % get_local_size(0) == 0 ?
      len :
      (len / get_local_size(0) + 1) * get_local_size(0);

    //Put each element where it needs to go
    for(unsigned int j = get_local_id(0);
        j < size;
        j += get_local_size(0))
    {
      unsigned int curKey;
      unsigned int curVal;

      //Don't read beyond the array
      if(j < len)
      {
        curKey = curKeyArrayRead[j];
        curVal = curValArrayRead[j];

        //Mark the scan array if the current digit is a 1 or a 0
        zerosScan[get_local_id(0)] = (curKey & mask) == 0x0 ? 1 : 0;
        onesScan[get_local_id(0)] = (curKey & mask) != 0x0 ? 1 : 0;
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      //Compute prefix scan of ones and zeros array
      scanSum(zerosScan);
      scanSum(onesScan);

      //Don't write beyond the array
      if(j < len)
      {
        //If our key has a zero in the current digit
        if((curKey & mask) == 0x0)
        {
          curKeyArrayWrite[*zerosIndex + zerosScan[get_local_id(0)] - 1] = 
            curKey;
          curValArrayWrite[*zerosIndex + zerosScan[get_local_id(0)] - 1] = 
            curVal;
        }
        else
        {
          curKeyArrayWrite[*onesIndex + onesScan[get_local_id(0)] - 1] = 
            curKey;
          curValArrayWrite[*onesIndex + onesScan[get_local_id(0)] - 1] = 
            curVal;
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      if(get_local_id(0) == 0)
      {
        *onesIndex += onesScan[get_local_size(0) - 1];
        *zerosIndex += zerosScan[get_local_size(0) - 1];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//Radix sort array from greatest to least
void radixSortGL(__global unsigned int* key,
                 __global unsigned int* val,
                 __global unsigned int* keyTmp,
                 __global unsigned int* valTmp,
                 unsigned int len,
                 __local unsigned int* zerosIndex,
                 __local unsigned int* onesIndex,
                 __local unsigned int* zerosScan,
                 __local unsigned int* onesScan)
{
  for(unsigned int i = 0; i < 32; i++)
  {
    unsigned int mask = 1 << i;
    __global unsigned int* curKeyArrayRead = i % 2 == 0 ? key : keyTmp;
    __global unsigned int* curKeyArrayWrite = i % 2 == 0 ? keyTmp : key;
    __global unsigned int* curValArrayRead = i % 2 == 0 ? val : valTmp;
    __global unsigned int* curValArrayWrite = i % 2 == 0 ? valTmp : val;

    if(get_local_id(0) == 0)
    {
      *zerosIndex = 0;
      *onesIndex = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //Calculate number of zeros so 
    for(unsigned int j = get_local_id(0); 
        j < len; 
        j += get_local_size(0))
    {
      if((curKeyArrayRead[j] & mask) != 0x0)
      {
        atomic_inc(zerosIndex);
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //Round to next multiple of local size so we can still execute barriers
    unsigned int size = len % get_local_size(0) == 0 ?
      len :
      (len / get_local_size(0) + 1) * get_local_size(0);

    //Put each element where it needs to go
    for(unsigned int j = get_local_id(0);
        j < size;
        j += get_local_size(0))
    {
      unsigned int curKey;
      unsigned int curVal;

      //Don't read beyond the array
      if(j < len)
      {
        curKey = curKeyArrayRead[j];
        curVal = curValArrayRead[j];

        //Mark the scan array if the current digit is a 1 or a 0
        zerosScan[get_local_id(0)] = (curKey & mask) == 0x0 ? 1 : 0;
        onesScan[get_local_id(0)] = (curKey & mask) != 0x0 ? 1 : 0;
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      //Compute prefix scan of ones and zeros array
      scanSum(zerosScan);
      scanSum(onesScan);

      //Don't write beyond the array
      if(j < len)
      {
        //If our key has a zero in the current digit
        if((curKey & mask) == 0x0)
        {
          curKeyArrayWrite[*zerosIndex + zerosScan[get_local_id(0)] - 1] = 
            curKey;
          curValArrayWrite[*zerosIndex + zerosScan[get_local_id(0)] - 1] = 
            curVal;
        }
        else
        {
          curKeyArrayWrite[*onesIndex + onesScan[get_local_id(0)] - 1] = 
            curKey;
          curValArrayWrite[*onesIndex + onesScan[get_local_id(0)] - 1] = 
            curVal;
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      if(get_local_id(0) == 0)
      {
        *onesIndex += onesScan[get_local_size(0) - 1];
        *zerosIndex += zerosScan[get_local_size(0) - 1];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

#endif

