#include <clUtil.h>
#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <exception>
#include "Matrix.h"
#include "blas.h"

using namespace clUtil;
using namespace std;

unsigned int m;
unsigned int n;
unsigned int k;

const unsigned int kColBlockSize = 5120u;
const unsigned int kRowBlockSize = 5120u;
const unsigned int kKBlockSize = 5120u;

#define IDX(row, col, ld) (row + col * ld)
//#define VERIFY

static double time()
{
  timeval curTime;

  gettimeofday(&curTime, NULL);

  return (double)curTime.tv_sec + (double)curTime.tv_usec / 1e6;
}

static void usage()
{
  cout << "Usage: gemm m n k" << endl;
  exit(1);
}

int main(int argc, char** argv)
{
  const char* kernels[] = {"dgemm.cl", "utility.cl"};

  if(argc < 4)
  {
    usage();
  }

  m = (unsigned int)atoi(argv[1]);
  n = (unsigned int)atoi(argv[2]);
  k = (unsigned int)atoi(argv[3]);

  if((m % kRowBlockSize != 0) || 
     (n % kColBlockSize != 0) || 
     (k % 16 != 0))
  {
    cout << "m must be a multiple of " << kRowBlockSize << endl;
    cout << "n must be a multiple of " << kColBlockSize << endl;
    cout << "k must be a multiple of " << kRowBlockSize << endl;
    
    exit(1);
  }

  try
  {

    Device::InitializeDevices(kernels, 2, NULL, "-D__DEVICE__");

    cout << "Generating matrices..." << endl;

    Matrix<double> a = Matrix<double>::Rand(m, k);
    Matrix<double> b = Matrix<double>::Rand(k, n);
    Matrix<double> c = Matrix<double>::Rand(m, n);

    cout << "Converting matrices..." << endl;

    BlockedMatrix<double> aBlocked(a, kRowBlockSize, kKBlockSize);
    BlockedMatrix<double> bBlocked(b, kKBlockSize, kColBlockSize);
    BlockedMatrix<double> cBlocked(c, kRowBlockSize, kColBlockSize);

    cout << "Starting gemm..." << endl;

    Device::StartProfiling();

    double time1 = time();

    multiply(cBlocked, aBlocked, bBlocked);

    double time2 = time();

#ifdef VERIFY
    cout << "Validation gemm..." << endl;

    Buffer aDev(sizeof(a[0]) * m * k);
    Buffer bDev(sizeof(b[0]) * k * n);
    Buffer cRefDev(sizeof(c[0]) * m * n);

    aDev.put(&a(0, 0));
    bDev.put(&b(0, 0));

    cRefDev.put(&c(0, 0));
    
    char transA = 'N';
    char transB = 'N';

    dgemm(&transA,
          &transB, 
          m, 
          n, 
          k, 
          1.0,
          aDev, 
          m, 
          bDev, 
          k, 
          0.0, 
          cRefDev, 
          m);
    
    cRefDev.get(&c(0, 0));
#endif


    Device::Finish();

    Device::DumpProfilingData();

#ifdef VERIFY
    cout << "Checking results..." << endl;

    for(unsigned int i = 0; i < m; i++)
    {
      for(unsigned int j = 0; j < n; j++)
      {
        float c1 = c(i, j);
        float c2 = cBlocked(i, j);

        if(fabs((c1-c2) / c1) > 1e-5)
        {
          cout << "Error row " << i << " col " << j 
            << " calc " << c1 << " ref " << c2 << endl;
          exit(1);
        }
      }
    }
#endif

    printf("%d\t%E\n", m, 2.0 * m * n * k / (time2 - time1));

  }
  catch(clUtilException& err)
  {
    cout << err.what() << endl;
    exit(1);
  }
  return 0;
}
