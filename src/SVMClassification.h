#pragma once

typedef struct svm_problem SVMProblem;
typedef struct svm_parameter SVMParameter;
typedef struct svm_node SVMNode;

void print_null(const char* s);
SVMParameter* SetSVMParameter(int kernel_type);
VoxelScore* GetSVMPerformance(int me, CorrMatrix** c_matrices, int nTrainings, int nFolds);
void print_null(const char* s);
SVMProblem* GetSVMProblem(CorrMatrix** c_matrices, int row, int startIndex, int nTrainings);
SVMProblem* GetSVMProblemWithPreKernel(CorrMatrix** c_matrices, int row, int startIndex, int nTrainings);
float DoSVM(int nFolds, SVMProblem* prob, SVMParameter* param);
