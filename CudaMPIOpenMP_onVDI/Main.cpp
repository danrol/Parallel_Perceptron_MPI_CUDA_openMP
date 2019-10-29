#include <mpi.h>
#include "mainMethods.h"
#include "FileMethods.h"
#include "Sequential.h"
#include "MPIAndOpenMPMethods.h"
#include "PrintMethods.h"

void allocPointsAndGroups(double **points, int **pointGroups, int numOfDimensions, int numOfPoints)
{
	*points = (double*)malloc(numOfDimensions*numOfPoints * sizeof(double));
	*pointGroups = (int*)malloc(numOfPoints * sizeof(int));
}

void cudaPredifinitions(double **deviceW, double **devicePoints, int **devicePointsSigns, int **deviceNumOfDimensions, int numOfDimensions, int numOfPoints, double *points)
{
	*deviceW = allocateWOnDevice(numOfDimensions);
	*devicePoints = allocateAndCopyPointsToDevice(numOfPoints, numOfDimensions, points);
	*devicePointsSigns = allocateDevicePointsSigns(numOfPoints);
	*deviceNumOfDimensions = allocateAndCopyNumOfDimensionsToDevice(numOfDimensions);
}

void changeArrToZeros(double **arr, int size)
{
	int i;
	for (i = 0; i < size; i++)
		(*arr)[i] = 0;
}

//Daniil Rolnik 334018009
int main(int argc, char *argv[])
{
	int numOfPoints, numOfDimensions, iterLimit, rank, numOfProcs, i, numOfW;
	int iterCount = 0, numOfMiss = 0, *pointGroups = 0, stopRecvAlphas = CONTINUE_RECV_ALPHAS;
	double alphaZero, alpha = 0, startAlpha, lastAlpha, alphaMax, q, qc, *w = 0, *points = 0;
	cudaError_t cudaStatus;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);

	if (numOfProcs > 2)
	{
		if (rank == MASTER)
		{
			readInputFromFile(&numOfPoints, &numOfDimensions, &alphaZero,
				&alphaMax, &iterLimit, &qc, &points, &pointGroups);
			alpha = alphaZero;

			perceptronSequentialSolution(numOfPoints, numOfDimensions, alphaZero,
				alphaMax, iterLimit, qc, points, pointGroups, &w);
			changeArrToZeros(&w, numOfDimensions + 1);
			sendInitDataToSlaves(numOfProcs, numOfPoints, numOfDimensions, iterLimit,
				qc, alphaZero, alphaMax);
			sendPointsToSlaves(numOfProcs, numOfPoints, numOfDimensions, points, pointGroups);
		}
		else
		{
			recvInitDataFromMaster(&numOfPoints, &numOfDimensions, &iterLimit,
				&qc, &alphaZero, &alphaMax, status);

			allocPointsAndGroups(&points, &pointGroups, numOfDimensions, numOfPoints);
			recvPointsFromMaster(numOfPoints, numOfDimensions, &points, &pointGroups, status);
		}

		if (rank == MASTER)
		{
			double currentAlpha = alphaZero;
			int numOfProcsUsed;
			q = HUGE_NUMBER;
			numOfW = numOfDimensions + 1;
			changeArrToZeros(&w, numOfW);
			//while solution is not found
			while (q > qc && currentAlpha < alphaMax)
			{
				numOfProcsUsed = 0;
				shareAlphasToSlaves(&currentAlpha, alphaZero, alphaMax, DEFAULT_ALPHA_CHUNK_SIZE, numOfProcs, &numOfProcsUsed);
				getSmallestChunkResultFromSlaves(&q, qc, &alpha, alphaMax, &w, numOfW, numOfProcsUsed, status);
				MPI_Bcast(&stopRecvAlphas, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
			}

			//check if proper solution wasn't found
			if (q > qc && currentAlpha >= alphaMax)
				outputAlphaNotFound();
			else
				outputResult(alpha, q, w, numOfW);
			printf("\n**** Algorithm finished. Final Result:");
			printResults(numOfPoints, numOfDimensions, iterLimit, alpha, alphaMax, q, qc, w);
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaDeviceReset failed!");
				return 1;
			}
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		else
		{
			numOfW = numOfDimensions + 1;
			int* cudaPointsGroups = (int*)malloc(numOfPoints * sizeof(int)), *devicePointsSigns, *deviceNumOfDimensions;
			double *deviceW, *devicePoints;
			cudaPredifinitions(&deviceW, &devicePoints, &devicePointsSigns, &deviceNumOfDimensions, numOfDimensions, numOfPoints, points);
			while (stopRecvAlphas == CONTINUE_RECV_ALPHAS)
			{
				getAlphasFromMaster(&startAlpha, &lastAlpha, status);

				//processes that didn't get chunk will stop
				if (startAlpha == HUGE_NUMBER)
					break;
				q = HUGE_NUMBER;

				for (alpha = startAlpha; alpha <= lastAlpha && q > qc; alpha += alphaZero)
				{
					w = (double*)calloc(numOfW, sizeof(double));
					iterCount = 0;
					numOfMiss = 0;
					int flaggedPointID = -1;
					bool allPointsCorrect = false;
					while (iterCount < iterLimit && allPointsCorrect == false)
					{
						//find pointsGroups with f
						cudaStatus = findPointGroupsforWByCuda(w, deviceW, devicePoints, devicePointsSigns, deviceNumOfDimensions, numOfPoints, numOfDimensions, &cudaPointsGroups);
						if (cudaStatus != cudaSuccess)
						{
							fprintf(stderr, "findPointGroupsforWByCuda failed!");
							return 1;
						}

						//Cycle through all given points Xi in the order as it is defined in the input file
						for (i = 0; i < numOfPoints; i++)
						{
							if (cudaPointsGroups[i] != pointGroups[i])
							{
								flaggedPointID = i;
								break;
							}
						}
						//Check if all given points are classified properly
						if (flaggedPointID < 0)
						{
							printf("\n****** All points correct");
							allPointsCorrect = true;
							break;
						}
						else
							updateWeightsWithOpenMP(&w, numOfDimensions, flaggedPointID, cudaPointsGroups[i], alpha, &(points[flaggedPointID*numOfDimensions]));

						iterCount++;
					}
					//check quality using openMP
					q = getQualityWithOpenMP(numOfPoints, points, pointGroups, &w, numOfDimensions);

				}
				alpha -= alphaZero;
				sendChunkResultToMaster(q, alpha, w, numOfW);
			}
			freeCudaValues(deviceW, devicePoints, devicePointsSigns, deviceNumOfDimensions);
		}
	}
	else
	{
		printf("number of processes must be bigger than 2");
	}

	free(points);
	free(w);
	free(pointGroups);

	MPI_Finalize();
	return 0;
}