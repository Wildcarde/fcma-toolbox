#pragma once

#include <string>
#define OMPI_SKIP_MPICXX
#include <mpi.h>

// Matrix multiplication parameters
#define MAXTRIAL 10000
#define MAXTASK 10000
#define TINYNUM 1e-4

typedef unsigned long long uint64;
typedef unsigned short uint16;

typedef struct raw_matrix_t
{
  std::string sname;  // subject name (file name without extension)
  int sid;  // subject id
  int row;
  int col;
  int nx, ny, nz;
  double* matrix;
}RawMatrix;

/*typedef struct avg_matrix_t
{
  int sid;  // subject id
  int row;
  int col;
  float* matrix;
}AvgMatrix;*/

typedef struct corr_matrix_t
{
  int sid;  // subject id
  int tlabel;  // trial label
  int sr; // starting row id
  int step; // row of this matrix
  int nVoxels;  // col of this matrix
  float* matrix;
}CorrMatrix;

typedef struct param_t
{
  const char* fmri_directory;
  const char* fmri_file_type;
  const char* block_information_file;
  const char* block_information_directory;
  const char* mask_file1;
  const char* mask_file2;
  int step;
  const char* output_file;
  int leave_out_id;
  int taskType;
  int nHolds;
  int nFolds;
  bool isTestMode;
  bool isUsingMaskFile;
}Param;

typedef struct trial_t  //data structure for the start and end point of a trial
{
  int tid;
  int sid;
  int label;
  int sc, ec;
}Trial;

typedef struct voxel_Score_t
{
  int vid;
  float score;
}VoxelScore;

typedef struct point_t  //voxel's 3-d coordinates
{
  int x, y, z;
}Point;

extern unsigned long long counter;

#define COMPUTATIONTAG 1
#define LENGTHTAG 2
#define VOXELCLASSIFIERTAG 3
#define ELAPSETAG 4
