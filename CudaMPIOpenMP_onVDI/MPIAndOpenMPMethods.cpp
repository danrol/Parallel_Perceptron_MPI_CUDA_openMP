#include "mainMethods.h"
#include <mpi.h>
#include "MPIAndOpenMPMethods.h"
#include "Sequential.h"

//send initialize data to slaves
void sendInitDataToSlaves(int numOfProcs, int numOfPoints, int numOfDimensions,
	int limitIter, double qc, double alphaZero, double alphaMax)
{
	int i;
	for (i = 1; i < numOfProcs; i++)
	{
			MPI_Send(&numOfPoints, 1, MPI_INT, i, NUM_OF_POINTS_TAG, MPI_COMM_WORLD);
			MPI_Send(&numOfDimensions, 1, MPI_INT, i, NUM_OF_DIMENSIONS_TAG, MPI_COMM_WORLD);
			MPI_Send(&limitIter, 1, MPI_INT, i, LIMIT_ITER_TAG, MPI_COMM_WORLD);
			MPI_Send(&qc, 1, MPI_DOUBLE, i, QC_TAG, MPI_COMM_WORLD);
			MPI_Send(&alphaZero, 1, MPI_DOUBLE, i, ALPHA_ZERO_TAG, MPI_COMM_WORLD);
			MPI_Send(&alphaMax, 1, MPI_DOUBLE, i, ALPHA_MAX_TAG, MPI_COMM_WORLD);
	}
}

//get all initialize data from master
void recvInitDataFromMaster(int *numOfPoints, int *numOfDimensions, int *limitIter,
	double *qc, double *alphaZero, double *alphaMax, MPI_Status status)
{
	MPI_Recv(numOfPoints, 1, MPI_INT, MASTER, NUM_OF_POINTS_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(numOfDimensions, 1, MPI_INT, MASTER, NUM_OF_DIMENSIONS_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(limitIter, 1, MPI_INT, MASTER, LIMIT_ITER_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(qc, 1, MPI_DOUBLE, MASTER, QC_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(alphaZero, 1, MPI_DOUBLE, MASTER, ALPHA_ZERO_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(alphaMax, 1, MPI_DOUBLE, MASTER, ALPHA_MAX_TAG, MPI_COMM_WORLD, &status);
}

//send points values to slaves
void sendPointsToSlaves(int numOfProcs, int numOfPoints, int numOfDimensions, double *points, int *pointGroups)
{
	int i;
	for (i = 1; i < numOfProcs; i++)
	{
		MPI_Send(points, numOfPoints*numOfDimensions, MPI_DOUBLE, i, POINTS_TAG, MPI_COMM_WORLD);
		MPI_Send(pointGroups, numOfPoints, MPI_INT, i, POINTS_GROUPS_TAG, MPI_COMM_WORLD);
	}
}

//get points from master
void recvPointsFromMaster(int numOfPoints, int numOfDimensions, double **points, int **pointGroups, MPI_Status status)
{
	double *tempPointsArr = (double*)malloc(numOfDimensions*numOfPoints * sizeof(double));
	MPI_Recv(tempPointsArr, numOfPoints*numOfDimensions, MPI_DOUBLE, MASTER, POINTS_TAG, MPI_COMM_WORLD, &status);
	(*points) = tempPointsArr;
	int *tempPointGroupsArr = (int*)malloc(numOfPoints * sizeof(int));
	MPI_Recv(tempPointGroupsArr, numOfPoints, MPI_INT, MASTER, POINTS_GROUPS_TAG, MPI_COMM_WORLD, &status);
	(*pointGroups) = tempPointGroupsArr;
}

//send alphas to slaves
void shareAlphasToSlaves(double *currentAlpha, double alphaZero, double alphaMax, int chunkSize, int numOfProcs, int *numOfProcsUsed)
{
	int i, j;
	double startAlpha;

	for (i = 1; i < numOfProcs; i++)
	{
		startAlpha = *currentAlpha;

		if (startAlpha > alphaMax)
			startAlpha = HUGE_NUMBER;
		else
		{
			*numOfProcsUsed = *numOfProcsUsed + 1;
			for (j = 0, *currentAlpha = startAlpha; *currentAlpha < alphaMax && j < chunkSize - 1; j++)
				*currentAlpha += alphaZero;
		}

		MPI_Send(&startAlpha, 1, MPI_DOUBLE, i, START_ALPHA_TAG, MPI_COMM_WORLD);
		MPI_Send(currentAlpha, 1, MPI_DOUBLE, i, LAST_ALPHA_TAG, MPI_COMM_WORLD);
		*currentAlpha += alphaZero;
	}
}

//receive alphas from master
void getAlphasFromMaster(double *startAlpha, double *lastAlpha, MPI_Status status)
{
	MPI_Recv(startAlpha, 1, MPI_DOUBLE, MASTER, START_ALPHA_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(lastAlpha, 1, MPI_DOUBLE, MASTER, LAST_ALPHA_TAG, MPI_COMM_WORLD, &status);
}


//Update weights using openMP
void updateWeightsWithOpenMP(double **w, int numOfDimensions, int firstFlaggedPointID, int sign, double alpha, double *dimensions)
{
	int i;
	(*w)[numOfDimensions] += (-sign)*alpha;

#pragma omp parallel for shared(w, dimensions) private(i)
		for (i = 0; i < numOfDimensions; i++)
			(*w)[i] += (-sign)*alpha*dimensions[i];
}

//Get quality using openMP
double getQualityWithOpenMP(int numOfPoints, double *points, int *pointGroups, double **w, int numOfDimensions )
{
//printf("###### started getQualityWithOpenMP");
int i, numOfMiss = 0, sign = 0, signToCheck = 0;
double fVal = 0;
#pragma omp parallel for reduction(+ : numOfMiss) shared(pointGroups) private (i, fVal, sign, signToCheck)
for (i = 0; i < numOfPoints; i++)
{
	fVal = f(w, &(points[i*numOfDimensions]), numOfDimensions + 1);
	sign = getFSign(fVal);
	signToCheck = pointGroups[i];
	if (sign != signToCheck)
		numOfMiss++;
}
//printf("##### finished getQualityWithOpenMP");
return (double)numOfMiss / numOfPoints;
}

//copy array
void copyArr(double ***arr, double *arrToCopy, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		(**arr)[i] = arrToCopy[i];
	}
}

//get smallest chunk result from every slave
void getSmallestChunkResultFromSlaves(double *q,double qc, double *alpha, double alphaMax, double **w, int numOfW,int numOfProcs, MPI_Status status)
{
	int i;
	double smallestAlpha = HUGE_NUMBER, tempQ = qc+1;
	double *tempW = (double*)malloc(numOfW * sizeof(double));

	for (i = 1; i < numOfProcs; i++)
	{
			MPI_Recv(q, 1, MPI_DOUBLE, i, Q_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(alpha, 1, MPI_DOUBLE, i, ALPHA_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(tempW, numOfW, MPI_DOUBLE, i, W_TAG, MPI_COMM_WORLD, &status);
			if (*alpha < smallestAlpha && *q < qc)
			{
				smallestAlpha = *alpha;
				tempQ = *q;
				copyArr(&w, tempW, numOfW);
			}
	}
	*alpha = smallestAlpha;
	*q = tempQ;
}

//send result of current chunk to master
void sendChunkResultToMaster(double q, double alpha, double *w, int numOfW)
{
	MPI_Send(&q, 1, MPI_DOUBLE, MASTER, Q_TAG, MPI_COMM_WORLD);
	MPI_Send(&alpha, 1, MPI_DOUBLE, MASTER, ALPHA_TAG, MPI_COMM_WORLD);
	MPI_Send(w, numOfW, MPI_DOUBLE, MASTER, W_TAG, MPI_COMM_WORLD);
}

