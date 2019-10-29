#pragma once

#ifndef MPIANDOPENMPMETHODS_H
#define MPIANDOPENMPMETHODS_H

#define ALPHA_TAG 1
#define Q_TAG 2
#define W_TAG 3
#define POINTS_TAG 4
#define POINTS_GROUPS_TAG 5
#define START_ALPHA_TAG 6
#define LAST_ALPHA_TAG 7
#define NUM_OF_POINTS_TAG 8
#define NUM_OF_DIMENSIONS_TAG 9
#define LIMIT_ITER_TAG 10
#define QC_TAG 11
#define ALPHA_ZERO_TAG 12
#define ALPHA_MAX_TAG 13

void sendInitDataToSlaves(int numOfProcs, int numOfPoints, int numOfDimensions,
	int limitIter, double qc, double alphaZero, double alphaMax);
void recvInitDataFromMaster(int *numOfPoints, int *numOfDimensions, int *limitIter,
	double *qc, double *alphaZero, double *alphaMax, MPI_Status status);
void sendPointsToSlaves(int numOfProcs, int numOfPoints, int numOfDimensions, double *points, int *pointGroups);
void recvPointsFromMaster(int numOfPoints, int numOfDimensions, double **points, int **pointGroups, MPI_Status status);
void shareAlphasToSlaves(double *currentAlpha, double alphaZero, double alphaMax, int chunkSize, int numOfProcs, int *numOfProcsUsed);
void getAlphasFromMaster(double *startAlpha, double *lastAlpha, MPI_Status status);
void sendChunkResultToMaster(double q, double alpha, double *w, int numOfW);
void getSmallestChunkResultFromSlaves(double *q, double qc, double *alpha, double alphaMax, double **w, int numOfW, int numOfProcs, MPI_Status status);
void updateWeightsWithOpenMP(double **w, int numOfDimensions, int firstFlaggedPointID, int sign, double alpha, double *points);
double getQualityWithOpenMP(int numOfPoints, double *points, int *pointGroups, double **w, int numOfDimensions);
void copyArr(double ***arr, double *arrToCopy, int size);
#endif
