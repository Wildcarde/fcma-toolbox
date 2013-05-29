#include "Matrix.h"

bool _IMatrix::RuntimeInitialized = false;

#ifdef RAII
std::unique_ptr<std::shared_ptr<clUtil::Buffer>[]> _IMatrix::aDevice;
std::unique_ptr<std::shared_ptr<clUtil::Buffer>[]> _IMatrix::bDevice;
std::unique_ptr<std::shared_ptr<clUtil::Buffer>[]> _IMatrix::cDevice;
#else
clUtil::Buffer** _IMatrix::aDevice;
clUtil::Buffer** _IMatrix::bDevice;
clUtil::Buffer** _IMatrix::cDevice;
#endif

void multiply(BlockedMatrix<float>& c,
              const BlockedMatrix<float>& a, 
              const BlockedMatrix<float>& b)
{
  using namespace clUtil;
  using namespace std;

  if((a.mRowsPerBlock != b.mRowsPerBlock) ||
     (b.mRowsPerBlock != c.mRowsPerBlock))
  {
    throw BlockedMatrixBlockMismatch();
  }

  if((a.mColsPerBlock != b.mColsPerBlock) ||
     (b.mColsPerBlock != c.mColsPerBlock))
  {
    throw BlockedMatrixBlockMismatch();
  }

  if((a.mRows != c.mRows) || (b.mCols != c.mCols) || (a.mCols != b.mRows))
  {
    throw MatrixBadDimensions();
  }

  char transA = 'N';
  char transB = 'N';
 
    //Use double buffering to overlap transfer and compute
  unsigned int k = a.getCols();

  ParallelFor(0, 1, c.numRowBlocks() * c.numColBlocks() - 1, 
  [&](size_t startIdx, size_t endIdx)
  {
    size_t curDeviceNum = Device::GetCurrentDeviceNum();

#ifdef RAII
    shared_ptr<Buffer> curA = _IMatrix::aDevice[2 * curDeviceNum];
    shared_ptr<Buffer> altA = _IMatrix::aDevice[2 * curDeviceNum + 1];
    shared_ptr<Buffer> curB = _IMatrix::bDevice[2 * curDeviceNum];
    shared_ptr<Buffer> altB = _IMatrix::bDevice[2 * curDeviceNum + 1];
    shared_ptr<Buffer> curC = _IMatrix::cDevice[2 * curDeviceNum];
    shared_ptr<Buffer> altC = _IMatrix::cDevice[2 * curDeviceNum + 1];
#else
    Buffer* curA = _IMatrix::aDevice[2 * curDeviceNum];
    Buffer* altA = _IMatrix::aDevice[2 * curDeviceNum + 1];
    Buffer* curB = _IMatrix::bDevice[2 * curDeviceNum];
    Buffer* altB = _IMatrix::bDevice[2 * curDeviceNum + 1];
    Buffer* curC = _IMatrix::cDevice[2 * curDeviceNum];
    Buffer* altC = _IMatrix::cDevice[2 * curDeviceNum + 1];
#endif

    Device& curDevice = Device::GetCurrentDevice();

    size_t curQueue = 0;

    for(size_t curBlockID = startIdx; curBlockID <= endIdx; curBlockID++)
    {
      unsigned int blockRow = curBlockID % c.numRowBlocks();
      unsigned int blockCol = curBlockID / c.numRowBlocks();
      unsigned int blockSize = c.mRowsPerBlock * c.mColsPerBlock;

      //Zero our output buffers
      curDevice.setCommandQueue(0);

      clUtilEnqueueKernel("bzero", 
                          clUtilGrid(blockSize, 64),
                          *curC,
                          blockSize);

      curDevice.setCommandQueue(1);

      clUtilEnqueueKernel("bzero",
                          clUtilGrid(blockSize, 64),
                          *altC,
                          blockSize);
      
      curDevice.setCommandQueue(0);


      //Go back and forth between the two queues, multiplying blocks of
      //A and B. This should give us transfer/kernel interleaving
      for(unsigned int curK = 0; curK < k / c.mRowsPerBlock; curK++)
      {
        curA->put(a.getBlock(blockRow, curK), sizeof(float) * blockSize);
        curB->put(b.getBlock(curK, blockCol), sizeof(float) * blockSize);
        
        sgemm(&transA,
              &transB, 
              c.mRowsPerBlock, 
              c.mColsPerBlock, 
              a.mColsPerBlock, 
              1.0,
              *curA, 
              a.mRowsPerBlock, 
              *curB, 
              b.mRowsPerBlock, 
              1.0, 
              *curC, 
              c.mRowsPerBlock);

        //Swap the buffers
#ifdef RAII
        shared_ptr<Buffer> tmp;
#else
        Buffer* tmp;
#endif

        tmp = curA;
        curA = altA;
        altA = tmp;
        
        tmp = curB;
        curB = altB;
        altB = tmp;
        
        tmp = curC;
        curC = altC;
        altC = tmp;

        //Change the queue the next batch of tasks will go into
        curQueue = curQueue == 0 ? 1 : 0;

        curDevice.setCommandQueue(curQueue);
      }

      //Add the two c buffers
      clUtilEnqueueKernel("acc",
                          clUtilGrid(blockSize, 64),
                          *curC,
                          *altC,
                          blockSize);
      
      //Finally, write c out to its correct place
      curC->get(c.getBlock(curBlockID), sizeof(float) * blockSize);
    }
  });

  //Device::Finish();

}

void multiply(BlockedMatrix<double>& c,
              const BlockedMatrix<double>& a, 
              const BlockedMatrix<double>& b)
{
  using namespace clUtil;
  using namespace std;

  if((a.mRowsPerBlock != b.mRowsPerBlock) ||
     (b.mRowsPerBlock != c.mRowsPerBlock))
  {
    throw BlockedMatrixBlockMismatch();
  }

  if((a.mColsPerBlock != b.mColsPerBlock) ||
     (b.mColsPerBlock != c.mColsPerBlock))
  {
    throw BlockedMatrixBlockMismatch();
  }

  if((a.mRows != c.mRows) || (b.mCols != c.mCols) || (a.mCols != b.mRows))
  {
    throw MatrixBadDimensions();
  }

  char transA = 'N';
  char transB = 'N';
 
    //Use double buffering to overlap transfer and compute
  unsigned int k = a.getCols();

  ParallelFor(0, 1, c.numRowBlocks() * c.numColBlocks() - 1, 
  [&](size_t startIdx, size_t endIdx)
  {
    size_t curDeviceNum = Device::GetCurrentDeviceNum();

#ifdef RAII
    shared_ptr<Buffer> curA = _IMatrix::aDevice[2 * curDeviceNum];
    shared_ptr<Buffer> altA = _IMatrix::aDevice[2 * curDeviceNum + 1];
    shared_ptr<Buffer> curB = _IMatrix::bDevice[2 * curDeviceNum];
    shared_ptr<Buffer> altB = _IMatrix::bDevice[2 * curDeviceNum + 1];
    shared_ptr<Buffer> curC = _IMatrix::cDevice[2 * curDeviceNum];
    shared_ptr<Buffer> altC = _IMatrix::cDevice[2 * curDeviceNum + 1];
#else
    Buffer* curA = _IMatrix::aDevice[2 * curDeviceNum];
    Buffer* altA = _IMatrix::aDevice[2 * curDeviceNum + 1];
    Buffer* curB = _IMatrix::bDevice[2 * curDeviceNum];
    Buffer* altB = _IMatrix::bDevice[2 * curDeviceNum + 1];
    Buffer* curC = _IMatrix::cDevice[2 * curDeviceNum];
    Buffer* altC = _IMatrix::cDevice[2 * curDeviceNum + 1];
#endif

    Device& curDevice = Device::GetCurrentDevice();

    size_t curQueue = 0;

    for(size_t curBlockID = startIdx; curBlockID <= endIdx; curBlockID++)
    {
      unsigned int blockRow = curBlockID % c.numRowBlocks();
      unsigned int blockCol = curBlockID / c.numRowBlocks();
      unsigned int blockSize = c.mRowsPerBlock * c.mColsPerBlock;

      //Zero our output buffers
      curDevice.setCommandQueue(0);

      clUtilEnqueueKernel("bzero_double", 
                          clUtilGrid(blockSize, 64),
                          *curC,
                          blockSize);

      curDevice.setCommandQueue(1);

      clUtilEnqueueKernel("bzero_double",
                          clUtilGrid(blockSize, 64),
                          *altC,
                          blockSize);
      
      curDevice.setCommandQueue(0);


      //Go back and forth between the two queues, multiplying blocks of
      //A and B. This should give us transfer/kernel interleaving
      for(unsigned int curK = 0; curK < k / c.mRowsPerBlock; curK++)
      {
        curA->put(a.getBlock(blockRow, curK), sizeof(double) * blockSize);
        curB->put(b.getBlock(curK, blockCol), sizeof(double) * blockSize);
        
        dgemm(&transA,
              &transB, 
              c.mRowsPerBlock, 
              c.mColsPerBlock, 
              a.mColsPerBlock, 
              1.0,
              *curA, 
              a.mRowsPerBlock, 
              *curB, 
              b.mRowsPerBlock, 
              1.0, 
              *curC, 
              c.mRowsPerBlock);

        //Swap the buffers
#ifdef RAII
        shared_ptr<Buffer> tmp;
#else
        Buffer* tmp;
#endif

        tmp = curA;
        curA = altA;
        altA = tmp;
        
        tmp = curB;
        curB = altB;
        altB = tmp;
        
        tmp = curC;
        curC = altC;
        altC = tmp;

        //Change the queue the next batch of tasks will go into
        curQueue = curQueue == 0 ? 1 : 0;

        curDevice.setCommandQueue(curQueue);
      }

      //Add the two c buffers
      clUtilEnqueueKernel("acc_double",
                          clUtilGrid(blockSize, 64),
                          *curC,
                          *altC,
                          blockSize);
      
      //Finally, write c out to its correct place
      curC->get(c.getBlock(curBlockID), sizeof(double) * blockSize);
    }
  });

  //Device::Finish();

}
