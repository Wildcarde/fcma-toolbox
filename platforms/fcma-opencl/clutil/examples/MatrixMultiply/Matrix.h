#pragma once
#include <memory>
#include <exception>
#include <stdlib.h>
#include <clUtil.h>

#include "blas.h"

#define RAII

const unsigned int kMaxRowsPerBlock = 5760u;
const unsigned int kMaxColsPerBlock = 5760u;
const unsigned int kDefaultRowsPerBlock = 5760u;
const unsigned int kDefaultColsPerBlock = 5760u;

class BlockedMatrixBlockMismatch : std::exception
{
  const char* what()
  {
    return "Attempted matrix operation with mismatched block sizes";
  }
};

class MatrixBadDimensions : std::exception
{
  const char* what()
  {
    return "Bad matrix dimensions";
  }
};

template <typename T> class Matrix;
template <typename T> class BlockedMatrix;

class _IMatrix
{
  friend void multiply(BlockedMatrix<float>& c,
                       const BlockedMatrix<float>& a,
                       const BlockedMatrix<float>& b);
  
  friend void multiply(BlockedMatrix<double>& c,
                       const BlockedMatrix<double>& a,
                       const BlockedMatrix<double>& b);

  protected:
    static bool RuntimeInitialized;
#ifdef RAII
    static std::unique_ptr<std::shared_ptr<clUtil::Buffer>[]> aDevice;
    static std::unique_ptr<std::shared_ptr<clUtil::Buffer>[]> bDevice;
    static std::unique_ptr<std::shared_ptr<clUtil::Buffer>[]> cDevice;
#else 
    static clUtil::Buffer** aDevice;
    static clUtil::Buffer** bDevice;
    static clUtil::Buffer** cDevice;
#endif

    static void Initialize()
    {
      using namespace clUtil;
      using namespace std;

      size_t numDevices = Device::GetDevices().size();
      unsigned int blockSize = kMaxRowsPerBlock * kMaxColsPerBlock;

#ifdef RAII
      unique_ptr<shared_ptr<Buffer>[]> 
        aGPU(new shared_ptr<Buffer>[2 * numDevices]);
      unique_ptr<shared_ptr<Buffer>[]> 
        bGPU(new shared_ptr<Buffer>[2 * numDevices]);
      unique_ptr<shared_ptr<Buffer>[]> 
        cGPU(new shared_ptr<Buffer>[2 * numDevices]);
#else
      aDevice = new Buffer*[2 * numDevices];
      bDevice = new Buffer*[2 * numDevices];
      cDevice = new Buffer*[2 * numDevices];
#endif

      for(unsigned int i = 0; i < 2 * numDevices; i++)
      {
        Device::SetCurrentDevice(i / 2);

#ifdef RAII
        aGPU[i] = shared_ptr<Buffer>(new Buffer(sizeof(double) * blockSize));
        bGPU[i] = shared_ptr<Buffer>(new Buffer(sizeof(double) * blockSize));
        cGPU[i] = shared_ptr<Buffer>(new Buffer(sizeof(double) * blockSize));
#else
        aDevice[i] = new Buffer(sizeof(double) * blockSize);
        bDevice[i] = new Buffer(sizeof(double) * blockSize);
        cDevice[i] = new Buffer(sizeof(double) * blockSize);

#define aGPU aDevice
#define bGPU bDevice
#define cGPU cDevice
#endif

        clUtilEnqueueKernel("bzero", 
                            clUtilGrid(blockSize, 64),
                            *aGPU[i],
                            blockSize);

        clUtilEnqueueKernel("bzero", 
                            clUtilGrid(blockSize, 64),
                            *bGPU[i],
                            blockSize);

        clUtilEnqueueKernel("bzero", 
                            clUtilGrid(blockSize, 64),
                            *cGPU[i],
                            blockSize);
      }

      Device::Finish();

#ifdef RAII
      aDevice = move(aGPU);
      bDevice = move(bGPU);
      cDevice = move(cGPU);
#endif

      RuntimeInitialized = true;
    }

    _IMatrix()
    {
      if(RuntimeInitialized == false)
      {
        Initialize();
      }
    }

};

//Interface for the matrix classes
template <typename T> class IMatrix : public _IMatrix
{
  protected:
    std::unique_ptr<T[]> mData;
    unsigned int mRows;
    unsigned int mCols;

    IMatrix(unsigned int rows, unsigned int cols) :
      _IMatrix(),
      mData((T*)NULL),
      mRows(rows),
      mCols(cols)
    {
    }

    IMatrix(const IMatrix& b) :
      _IMatrix(),
      mData((T*)NULL),
      mRows(b.mRows),
      mCols(b.mCols)
    {
    }

  public:
    bool operator==(const IMatrix<T>& b)
    {
      if(mRows != b.mRows || mCols != b.mCols) { return false; }

      for(unsigned int i = 0; i < mRows; i++)
      {
        for(unsigned int j = 0; j < mCols; j++)
        {
          if(*(this->getPtr(i, j)) != *(b.getPtr(i, j)))
          {
            return false;
          }
        }
      }

      return true;
    }

    bool operator!=(const IMatrix<T>& b)
    {
      if(mRows != b.mRows || mCols != b.mCols) { return true; }

      for(unsigned int i = 0; i < mRows; i++)
      {
        for(unsigned int j = 0; j < mCols; j++)
        {
          if(*(this->getPtr(i, j)) != *(b.getPtr(i, j)))
          {
            return true;
          }
        }
      }

      return false;
    }

    T* operator&()
    {
      return mData.get();
    }

    T& operator()(const unsigned int i, const unsigned int j)
    {
      return *this->getPtr(i, j);
    }

    const T& operator()(const unsigned int i, const unsigned int j) const
    {
      return *this->getPtr(i, j);
    }

    T& operator[](const size_t i) 
    {
      return mData[i];
    }
    
    const T& operator[](const size_t i) const
    {
      return mData[i];
    }

    unsigned int getRows() const { return mRows; }
    unsigned int getCols() const { return mCols; }
    std::unique_ptr<T[]>& getData(){ return mData; };

    virtual T* getPtr(unsigned int row, unsigned int col) const = 0;
};

//A blocked column major storage matrix
template <typename T> class BlockedMatrix : public IMatrix<T>
{
  friend void multiply(BlockedMatrix<float>& c,
                       const BlockedMatrix<float>& a,
                       const BlockedMatrix<float>& b);
  
  friend void multiply(BlockedMatrix<double>& c,
                       const BlockedMatrix<double>& a,
                       const BlockedMatrix<double>& b);

  protected:
    using IMatrix<T>::mRows;
    using IMatrix<T>::mCols;
    using IMatrix<T>::mData;

  private:
    unsigned int mRowBlocks;
    unsigned int mColBlocks;
    unsigned int mRowsPerBlock;
    unsigned int mColsPerBlock;
    size_t mBlockSize;

  public:
    BlockedMatrix(const unsigned int rowBlocks, 
                  const unsigned int colBlocks, 
                  const unsigned int rowsPerBlock, 
                  const unsigned int colsPerBlock) :
      IMatrix<T>(rowBlocks * rowsPerBlock, colBlocks * colsPerBlock),
      mRowBlocks(rowBlocks),
      mColBlocks(colBlocks),
      mRowsPerBlock(rowsPerBlock),
      mColsPerBlock(colsPerBlock),
      mBlockSize(mRowsPerBlock * mColsPerBlock)
    {
      std::unique_ptr<T[]> data(new T[IMatrix<T>::mRows * IMatrix<T>::mCols]);
      IMatrix<T>::mData = std::move(data);
    }

    BlockedMatrix(const unsigned int rowsPerBlock,
                  const unsigned int colsPerBlock) :
      IMatrix<T>(0, 0),
      mRowBlocks(0),
      mColBlocks(0),
      mRowsPerBlock(rowsPerBlock),
      mColsPerBlock(colsPerBlock),
      mBlockSize(mRowsPerBlock * mColsPerBlock)
    {
    }

    BlockedMatrix() = delete;

    //Generic copy constructor
    BlockedMatrix(const IMatrix<T>& b) : 
      IMatrix<T>(b),
      mRowBlocks(IMatrix<T>::mRows / kDefaultRowsPerBlock),
      mColBlocks(IMatrix<T>::mCols / kDefaultColsPerBlock),
      mRowsPerBlock(kDefaultRowsPerBlock),
      mColsPerBlock(kDefaultColsPerBlock),
      mBlockSize(mRowsPerBlock * mColsPerBlock)
    {
      std::unique_ptr<T[]> data(new T[IMatrix<T>::mRows * IMatrix<T>::mCols]);
      IMatrix<T>::mData = move(data);

      for(unsigned int i = 0; i < IMatrix<T>::mRows; i++)
      {
        for(unsigned int j = 0; j < IMatrix<T>::mCols; j++)
        {
          *this->getPtr(i, j) = *b.getPtr(i, j);
        }
      }
    }

    //Copy from Matrix
    BlockedMatrix(const Matrix<T>& b, 
                  size_t blockRows = kDefaultRowsPerBlock,
                  size_t blockCols = kDefaultColsPerBlock) :
      IMatrix<T>(static_cast<const IMatrix<T>&>(b)),
      mRowBlocks(IMatrix<T>::mRows / blockRows),
      mColBlocks(IMatrix<T>::mCols / blockCols),
      mRowsPerBlock(blockRows),
      mColsPerBlock(blockCols),
      mBlockSize(mRowsPerBlock * mColsPerBlock)
    {
      std::unique_ptr<T[]> data(new T[IMatrix<T>::mRows * IMatrix<T>::mCols]);
      IMatrix<T>::mData = move(data);
      
      for(unsigned int rowBlock = 0; rowBlock < mRowBlocks; rowBlock++)
      {
        for(unsigned int colBlock = 0; colBlock < mColBlocks; colBlock++)
        {
          for(unsigned int curCol = 0; curCol < mColsPerBlock; curCol++)
          {
            T* aPtr = this->getPtr(rowBlock * mRowsPerBlock, 
                                   colBlock * mColsPerBlock + curCol);
            T* bPtr = b.getPtr(rowBlock * mRowsPerBlock,
                               colBlock * mColsPerBlock + curCol);

            memcpy(aPtr, bPtr, sizeof(T) * mRowsPerBlock);
          }
        }
      }
    }

    BlockedMatrix& operator=(const IMatrix<T>& b)
    {
      if(this != &b)
      {
        this->mRows = b.getRows();
        this->mCols = b.getCols();
        std::unique_ptr<T[]> data(new T[this->mRows * this->mCols]);
        this->mData = move(data);
        mRowBlocks = this->mRows / mRowsPerBlock;
        mColBlocks = this->mCols / mColsPerBlock;

        for(unsigned int i = 0; i < IMatrix<T>::mRows; i++)
        {
          for(unsigned int j = 0; j < IMatrix<T>::mCols; j++)
          {
            *getPtr(i, j) = *b.getPtr(i, j);
          }
        }
      }
      return *this;
    }

    virtual T* getPtr(unsigned int row, unsigned int col) const
    {
      unsigned int rowBlock = row / mRowsPerBlock;
      unsigned int colBlock = col / mColsPerBlock;
      unsigned int subRow = row % mRowsPerBlock;
      unsigned int subCol = col % mColsPerBlock;

      return &(getBlock(rowBlock, colBlock)[subRow + subCol * mRowsPerBlock]);
    }

    unsigned int numRowBlocks() const { return mRowBlocks; }
    unsigned int numColBlocks() const { return mColBlocks; }

    T* getBlock(unsigned int rowBlock, unsigned colBlock) const
    {
      return &IMatrix<T>::mData[mBlockSize * rowBlock +
                                mBlockSize * colBlock * mRowBlocks];
    }

    T* getBlock(unsigned int blockID) const
    {
      unsigned int rowBlock = blockID % mRowBlocks;
      unsigned int colBlock = blockID / mRowBlocks;

      return &IMatrix<T>::mData[mBlockSize * rowBlock +
                                mBlockSize * colBlock * mRowBlocks];
    }
  
    BlockedMatrix operator*(const BlockedMatrix& b);

    static BlockedMatrix Rand(unsigned int rowBlocks, 
                              unsigned int colBlocks,
                              unsigned int rowBlockSize,
                              unsigned int colBlockSize)
    {
      BlockedMatrix rand(rowBlocks, colBlocks, rowBlockSize, colBlockSize);

      for(unsigned int i = 0; i < rand.mRows; i++)
      {
        for(unsigned int j = 0; j < rand.mCols; j++)
        {
          rand.mData[i + j * rand.mCols] = (T)(i * j);
          //rand.mData[i + j * rand.mCols] = (T)drand48();
        }
      }

      return rand;
    }
};

void multiply(BlockedMatrix<float>& c,
              const BlockedMatrix<float>& a, 
              const BlockedMatrix<float>& b);

#if 0
template <> BlockedMatrix<float> 
BlockedMatrix<float>::operator*(const BlockedMatrix<float>& b)
{
  BlockedMatrix<float> c(mRowsPerBlock, 
                         b.mRowsPerBlock, 
                         mRowBlocks, 
                         b.mColBlocks);

  return c;
}
#endif



//A column major matrix
template <typename T> class Matrix : public IMatrix<T>
{
  public:
    Matrix(unsigned int rows, unsigned int cols) : IMatrix<T>(rows, cols)
    {
      std::unique_ptr<T[]> data(new T[rows * cols]);
      IMatrix<T>::mData = move(data);
    }

    Matrix(const IMatrix<T>& b) : IMatrix<T>(b.mRows, b.mCols)
    {
      std::unique_ptr<T[]> data(new T[IMatrix<T>::mRows * IMatrix<T>::mCols]);
      IMatrix<T>::mData = move(data);

      for(unsigned int i = 0; i < IMatrix<T>::mRows; i++)
      {
        for(unsigned int j = 0; j < IMatrix<T>::mCols; j++)
        {
          *this->getPtr(i, j) = *b.getPtr(i, j);
        }
      }
    }

    virtual T* getPtr(unsigned int row, unsigned int col) const
    {
      return &IMatrix<T>::mData[row + IMatrix<T>::mRows * col];
    }

    static Matrix Rand(unsigned int rows, unsigned int cols)
    {
      Matrix ret(rows, cols);

      for(unsigned int i = 0; i < rows; i++)
      {
        for(unsigned int j = 0; j < cols; j++)
        {
          *ret.getPtr(i, j) = (T)(i * j);
          //*ret.getPtr(i, j) = (T)drand48();
        }
      }

      return ret;
    }
};
