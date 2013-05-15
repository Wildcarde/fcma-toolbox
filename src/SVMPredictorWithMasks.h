#pragma once

int SVMPredictCorrelationWithMasks(RawMatrix** r_matrices, int nSubs, const char* maskFile1, const char* maskFile2, int nTrials, Trial* trials, int nTests);
float* GetPartialInnerSimMatrixWithMasks(int nSubs, int nTrials, int sr, int rowLength, Trial* trials, RawMatrix** masked_matrices1, RawMatrix** masked_matrices2);
int SVMPredictActivationWithMasks(RawMatrix** avg_matrices, int nSubs, const char* maskFile, int nTrials, Trial* trials, int nTests);
