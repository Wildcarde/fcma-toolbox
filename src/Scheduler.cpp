#include <iostream>
#include <algorithm>
#include <fstream>
#include "common.h"
#include "MatComputation.h"
#include "CorrMatAnalysis.h"
#include "Classification.h"
#include "Preprocessing.h"
#include "SVMClassification.h"
#include "Scheduler.h"

using namespace std;

void Scheduler(int me, int nprocs, int step, RawMatrix** r_matrices, int taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds, const char* output_file, const char* mask_file1, const char* mask_file2)
{
  RawMatrix** masked_matrices1=NULL;
  RawMatrix** masked_matrices2=NULL;
  if (mask_file1!=NULL)
    masked_matrices1 = GetMaskedMatrices(r_matrices, nSubs, mask_file1);
  else
    masked_matrices1 = r_matrices;
  if (mask_file2!=NULL)
    masked_matrices2 = GetMaskedMatrices(r_matrices, nSubs, mask_file2);
  else
    masked_matrices2 = r_matrices;
  if (me == 0)
  {
    int row1 = masked_matrices1[0]->row;
    int row2 = masked_matrices2[0]->row;
    cout<<"#voxels for mask 1: "<<row1<<endl;
    cout<<"#voxels for mask 2: "<<row2<<endl;
    DoMaster(nprocs, step, row1, output_file);
  }
  else
  {
    DoSlave(me, 0, masked_matrices1, masked_matrices2, taskType, trials, nTrials, nHolds, nSubs, nFolds);  // 0 for master id
  }
}

// function for sort
bool cmp(VoxelScore w1, VoxelScore w2)
{
  return w1.score>w2.score;
}

/* the master node splits the tasks and assign them to slave nodes. 
When a slave nodes return, the master node would send another tasks to it. 
The master node finally collects all results and sort them to write to a file. */
void DoMaster(int nprocs, int step, int row, const char* output_file)
{
  int curSr = 0;
  int i, j;
  int total = row / step;
  if (row%step != 0)
  {
    total += 1;
  }
  cout<<"total task: "<<total<<endl;
  int sentCount = 0;
  int doneCount = 0;
  int sendMsg[2];
  for (i=1; i<nprocs; i++)  // fill up the processes first
  {
    sendMsg[0] = curSr;
    sendMsg[1] = step;
    if (curSr+step>row) // overflow, so only get the left
    {
      sendMsg[1] = row - curSr;
    }
    curSr += sendMsg[1];
    MPI_Send(sendMsg,  /* message buffer, the correlation vector */
           2,                  /* number of data to send */
           MPI_INT,                       /* data item is float */
           i,                            /* destination process rank */
           COMPUTATIONTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    sentCount++;
  }
  VoxelScore* scores = new VoxelScore[row];  // one voxel one score
  int totalLength = 0;
  MPI_Status status;
  while (sentCount < total)
  {
    int curLength;
    float elapse;
    // get the elapse time
    MPI_Recv(&elapse,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           MPI_ANY_SOURCE,                       /* receive from any sender */
           ELAPSETAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the length of message first
    MPI_Recv(&curLength,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_INT,          /* of type float real */
           status.MPI_SOURCE,       /* receive from any sender */
           LENGTHTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the classifier array
    MPI_Recv(scores+totalLength,      /* message buffer */
           curLength*2,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           status.MPI_SOURCE,                       /* receive from the previous sender */
           VOXELCLASSIFIERTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    totalLength += curLength;
    doneCount++;
    cout.precision(4);
    cout<<doneCount<<'\t'<<elapse<<"s\t"<<flush;
    if (doneCount%6==0)
    {
      cout<<endl;
    }
    sendMsg[0] = curSr;
    sendMsg[1] = step;
    if (curSr+step>row) // overflow, so only get the left
    {
      sendMsg[1] = row - curSr;
    }
    curSr += sendMsg[1];
    MPI_Send(sendMsg,  /* message buffer, the correlation vector */
           2,                  /* number of data to send */
           MPI_INT,                       /* data item is float */
           status.MPI_SOURCE,             /* destination process rank */
           COMPUTATIONTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    sentCount++;
  }
  while (doneCount < total)
  {
    int curLength;
    float elapse;
    // get the elapse time
    MPI_Recv(&elapse,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           MPI_ANY_SOURCE,                       /* receive from any sender */
           ELAPSETAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the length of message first
    MPI_Recv(&curLength,      /* message buffer */
           1,              /* numbers of data to receive */
           MPI_INT,          /* of type float real */
           status.MPI_SOURCE,      /* receive from any sender */
           LENGTHTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    // get the classifier array
    MPI_Recv(scores+totalLength,      /* message buffer */
           curLength*2,              /* numbers of data to receive */
           MPI_FLOAT,          /* of type float real */
           status.MPI_SOURCE,                       /* receive from any sender */
           VOXELCLASSIFIERTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    totalLength += curLength;
    doneCount++;
    cout.precision(4);
    cout<<doneCount<<'\t'<<elapse<<"s\t"<<flush;
    if (doneCount%6==0)
    {
      cout<<endl;
    }
  }
  for (i=1; i<nprocs; i++)  // tell all processes to stop
  {
    sendMsg[0] = -1;
    sendMsg[1] = -1;
    curSr += step;
    MPI_Send(sendMsg,  /* message buffer, the correlation vector */
           2,                  /* number of data to send */
           MPI_INT,                       /* data item is float */
           i,                            /* destination process rank */
           COMPUTATIONTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
  }
  sort(scores, scores+totalLength, cmp);
  cout<<"Total length: "<<totalLength<<endl;
  ofstream ofile(output_file);
  for (j=0; j<totalLength; j++)
  {
    ofile<<scores[j].vid<<" "<<scores[j].score<<endl;
  }
  ofile.close();
}

/* the slave node listens to the master node and does the task that the master node assigns. 
A task is a matrix multiplication to get the correlation vectors of some voxels. 
The slave node also does some preprocessing on the correlation vectors then 
analyzes the correlatrion vectors (either do classification or compute the average correlation coefficients.*/
void DoSlave(int me, int masterId, RawMatrix** matrices1, RawMatrix** matrices2, int taskType, Trial* trials, int nTrials, int nHolds, int nSubs, int nFolds)
{
  int recvMsg[2];
  MPI_Status status;
  while (true)
  {
    MPI_Recv(recvMsg,      /* message buffer */
           2,              /* numbers of data to receive */
           MPI_INT,          /* of type float real */
           masterId,                       /* receive from any sender */
           COMPUTATIONTAG,              /* user chosen message tag */
           MPI_COMM_WORLD,          /* default communicator */
           &status);                /* info about the received message */
    double tstart = MPI_Wtime();
    int sr = recvMsg[0];
    int step = recvMsg[1];
    if (sr == -1) // finish flag
    {
      break;
    }
    CorrMatrix** c_matrices = ComputeAllTrialsCorrMatrices(trials, nTrials, sr, step, matrices1, matrices2);
    VoxelScore* scores = NULL;
    switch (taskType)
    {
      case 0:
        // Fisher transform and z-score (across the blocks) the data here
        corrMatPreprocessing(c_matrices, nTrials, nSubs);
        scores = GetSVMPerformance(me, c_matrices, nTrials-nHolds, nFolds);
        break;
      case 1:
        // Fisher transform and z-score (across the blocks) the data here
        corrMatPreprocessing(c_matrices, nTrials, nSubs);
        scores = GetDistanceRatio(me, c_matrices, nTrials-nHolds);
        break;
      case 3:
				scores = GetCorrVecSum(me, c_matrices, nTrials);
				break;
      default:
        cerr<<"unknown task type"<<endl;
        exit(1);
    }
    double tstop = MPI_Wtime();
    float elapse = float(tstop-tstart);
    MPI_Send(&elapse,  /* message buffer, the correlation vector */
           1,                  /* number of data to send */
           MPI_FLOAT,                       /* data item is float */
           masterId,                            /* destination process rank */
           ELAPSETAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    MPI_Send(&(c_matrices[0]->step),  /* message buffer, the correlation vector */
           1,                  /* number of data to send */
           MPI_INT,                       /* data item is float */
           masterId,                            /* destination process rank */
           LENGTHTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    MPI_Send(scores,  /* message buffer, the correlation vector */
           c_matrices[0]->step*2,                  /* number of data to send */
           MPI_FLOAT,                       /* data item is float */
           masterId,                            /* destination process rank */
           VOXELCLASSIFIERTAG,                      /* user chosen message tag */
           MPI_COMM_WORLD);                 /* default communicator */
    delete scores;
    int i;
    for (i=0; i<nTrials; i++)
    {
      delete c_matrices[i]->matrix;
    }
    delete c_matrices;
  }
}
