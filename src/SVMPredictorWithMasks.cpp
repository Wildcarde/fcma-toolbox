#include <iostream>
#include "nifti1_io.h"
#include "common.h"
#include "svm.h"
#include "MatComputation.h"
#include "Preprocessing.h"
#include "SVMPredictor.h"
#include "SVMPredictorWithMasks.h"

using namespace std;

/***************************************
Get two parts of the brain to compute the correlation and then use the correlation vectors to predict
input: the raw activation matrix array, the number of voxels, the number of subjects, the first mask file, the second mask file, the number of blocks(trials), the blocks, the number of test samples
output: the results are displayed on the screen and returned
****************************************/
int SVMPredictCorrelationWithMasks(RawMatrix** r_matrices, int nSubs, const char* maskFile1, const char* maskFile2, int nTrials, Trial* trials, int nTests)
{
  int i, j;
  svm_set_print_string_function(&print_null);
  RawMatrix** masked_matrices1=NULL;
  RawMatrix** masked_matrices2=NULL;
  if (maskFile1!=NULL)
    masked_matrices1 = GetMaskedMatrices(r_matrices, nSubs, maskFile1);
  else
    masked_matrices1 = r_matrices;
  if (maskFile2!=NULL)
    masked_matrices2 = GetMaskedMatrices(r_matrices, nSubs, maskFile2);
  else
    masked_matrices2 = r_matrices;
  cout<<"masked matrices generating done!"<<endl;
  cout<<"#voxels for mask1: "<<masked_matrices1[0]->row<<" #voxels for mask2: "<<masked_matrices2[0]->row<<endl;
  float* simMatrix = new float[nTrials*nTrials];
  int corrRow = masked_matrices1[0]->row;
  //int corrCol = masked_matrices2[0]->row; // no use here
  memset((void*)simMatrix, 0, nTrials*nTrials*sizeof(float));
  int sr = 0, rowLength = 100;
  int result = 0;
  while (sr<corrRow)
  {
    if (rowLength >= corrRow - sr)
    {
      rowLength = corrRow - sr;
    }
    float* tempSimMatrix = GetPartialInnerSimMatrixWithMasks(nSubs, nTrials, sr, rowLength, trials, masked_matrices1, masked_matrices2);
    for (i=0; i<nTrials*nTrials; i++) simMatrix[i] += tempSimMatrix[i];
    delete tempSimMatrix;
    sr += rowLength;
  }
  SVMParameter* param = SetSVMParameter(4); // precomputed
  SVMProblem* prob = GetSVMTrainingSet(simMatrix, nTrials, trials, nTrials-nTests);
  struct svm_model *model = svm_train(prob, param);
  int nTrainings = nTrials-nTests;
  SVMNode* x = new SVMNode[nTrainings+2];
  for (i=nTrainings; i<nTrials; i++)
  {
    x[0].index = 0;
    x[0].value = static_cast<float>( i-nTrainings+1 );
    for (j=0; j<nTrainings; j++)
    {
      x[j+1].index = j+1;
      x[j+1].value = simMatrix[i*nTrials+j];
    }
    x[j+1].index = -1;
    double predict_label = svm_predict(model, x);
    if ((double)trials[i].label == predict_label)
    {
      result++;
    }
  }
  svm_free_and_destroy_model(&model);
  delete x;
  delete prob->y;
  for (i=0; i<nTrainings; i++)
  {
    delete prob->x[i];
  }
  delete prob->x;
  delete prob;
  svm_destroy_param(param);
  delete simMatrix;
  for (i=0; i<nSubs; i++)
  {
    delete masked_matrices1[i]->matrix;
    delete masked_matrices2[i]->matrix;
  }
  delete masked_matrices1;
  delete masked_matrices2;
  return result;
}

/***********************************************
Get the inner product of vectors from start row(sr), last rowLength-length
input: the number of subjects, the number of blocks, the start row, the number of voxels of masked matrix one that involved in the computing, the first masked data array, the second masked data array
output: the partial similarity matrix based on the selected rows of first matrices and the whole second matrices
************************************************/
float* GetPartialInnerSimMatrixWithMasks(int nSubs, int nTrials, int sr, int rowLength, Trial* trials, RawMatrix** masked_matrices1, RawMatrix** masked_matrices2) // compute the correlation between masked matrices
{
  int i;
  int row1 = masked_matrices1[0]->row;
  int row2 = masked_matrices2[0]->row;  //rows should be the same across subjects since we are using the same mask file to filter out voxels
  float* values= new float[nTrials*rowLength*row2];
  float* simMatrix = new float[nTrials*nTrials];
  memset((void*)simMatrix, 0, nTrials*nTrials*sizeof(float));
  for (i=0; i<nTrials; i++)
  {
    int sc = trials[i].sc;
    int ec = trials[i].ec;
    int sid = trials[i].sid;
    int col = masked_matrices1[sid]->col; // the column of 1 and 2 should be the same, i.e. the number of TRs of a block; columns may be different, since different subjects have different TRs
    double* mat1 = masked_matrices1[sid]->matrix;
    double* mat2 = masked_matrices2[sid]->matrix;
    float* buf1 = new float[row1*col]; // col is more than what really need, just in case
    float* buf2 = new float[row2*col]; // col is more than what really need, just in case
    int ml1 = getBuf(sc, ec, row1, col, mat1, buf1);  // get the normalized matrix, return the length of time points to be computed
    int ml2 = getBuf(sc, ec, row2, col, mat2, buf2);  // get the normalized matrix, return the length of time points to be computed, m1==m2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rowLength, row2, ml1, 1.0, buf1+sr*ml1, ml1, buf2, ml2, 0.0, values+i*rowLength*row2, row2);
    delete buf1;
    delete buf2;
  }
  NormalizeCorrValues(values, nTrials, rowLength, row2, nSubs);
  GetDotProductUsingMatMul(simMatrix, values, nTrials, rowLength, row2);
  delete values;
  return simMatrix;
}


/***************************************
Get one part of the brain to compute the averaged activation and then use the normalized activation vectors to predict
input: the raw activation matrix array, the number of voxels, the number of subjects, the ROI mask file, the number of blocks(trials), the blocks, the number of test samples
output: the results are displayed on the screen and returned
****************************************/
int SVMPredictActivationWithMasks(RawMatrix** avg_matrices, int nSubs, const char* maskFile, int nTrials, Trial* trials, int nTests)
{
  int i, j;
  int nTrainings = nTrials-nTests;
  SVMParameter* param = SetSVMParameter(0); // linear
  SVMProblem* prob = new SVMProblem();
  prob->l = nTrainings;
  prob->y = new double[nTrainings];
  prob->x = new SVMNode*[nTrainings];
  svm_set_print_string_function(&print_null);
  
  RawMatrix** masked_matrices=NULL;
  if (maskFile!=NULL)
    masked_matrices = GetMaskedMatrices(avg_matrices, nSubs, maskFile);
  else
    masked_matrices = avg_matrices;
  cout<<"masked matrices generating done!"<<endl;
  cout<<"#voxels for mask "<<masked_matrices[0]->row<<endl;
  int nVoxels = masked_matrices[0]->row;
  for (i=0; i<nTrainings; i++)
  {
    int sid = trials[i].sid;
    prob->y[i] = trials[i].label;
    prob->x[i] = new SVMNode[nVoxels+1];
    for (j=0; j<nVoxels; j++)
    {
      prob->x[i][j].index = j+1;
      int col = masked_matrices[sid]->col;
      int offset = (trials[i].sc-4)/20; //ad hoc here, for this dataset only!!!! from the block number (0 based) from starting TR
      prob->x[i][j].value = static_cast<float>( masked_matrices[sid]->matrix[j*col+offset] );
    }
    prob->x[i][j].index = -1;
  }
  struct svm_model *model = svm_train(prob, param);
  SVMNode* x = new SVMNode[nVoxels+1];
  int result = 0;
  for (i=nTrainings; i<nTrials; i++)
  {
    int sid = trials[i].sid;
    for (j=0; j<nVoxels; j++)
    {
      x[j].index = j+1;
      int col = masked_matrices[sid]->col;
      int offset = (trials[i].sc-4)/20; //ad hoc here, for this dataset only!!!! from the block number (0 based) from starting TR
      x[j].value = static_cast<float>( masked_matrices[sid]->matrix[j*col+offset] );
    }
    x[j].index = -1;
    double predict_label = svm_predict(model, x);
    if ((double)trials[i].label == predict_label)
    {
      result++;
    }
  }
  return result;
}
