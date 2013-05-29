#pragma once
#include <clUtil.h>
#include "sgemm.cl"
#include "dgemm.cl"

void sgemm_reference(char* transA,
                     char* transB,
                     int m,
                     int n,
                     int k,
                     float alpha,
                     clUtil::Buffer& a,
                     int lda,
                     clUtil::Buffer& b,
                     int ldb,
                     float beta,
                     clUtil::Buffer& c,
                     int ldc);

void sgemm(char* transA,
           char* transB,
           int m,
           int n,
           int k,
           float alpha,
           clUtil::Buffer& a,
           int lda,
           clUtil::Buffer& b,
           int ldb,
           float beta,
           clUtil::Buffer& c,
           int ldc);

void dgemm_reference(char* transA,
                     char* transB,
                     int m,
                     int n,
                     int k,
                     double alpha,
                     clUtil::Buffer& a,
                     int lda,
                     clUtil::Buffer& b,
                     int ldb,
                     double beta,
                     clUtil::Buffer& c,
                     int ldc);

void dgemm(char* transA,
           char* transB,
           int m,
           int n,
           int k,
           double alpha,
           clUtil::Buffer& a,
           int lda,
           clUtil::Buffer& b,
           int ldb,
           double beta,
           clUtil::Buffer& c,
           int ldc);

