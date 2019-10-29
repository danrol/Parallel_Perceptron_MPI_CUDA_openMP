#include "mainMethods.h"
#include "PrintMethods.h"

void printW(double *w, int numOfW)
{
	int i;
	printf("\n");
	for (i = 0; i < numOfW; i++)
	{
		printf("w%d = %1f, ", i, w[i]);
	}
}

void printPointGroups(int *pointGroups, int numOfPoints)
{
	int i;
	printf("\n**** point groups:\n");
	for (i = 0; i < numOfPoints; i++)
	{
		printf("%d ", pointGroups[i]);
	}
}

void printResults(int numOfPoints, int numOfDimensions, int iterLimit, double alpha,
	double alphaMax, double q, double qc, double* w)
{
	int i;
	printf("\n");
	if (alpha >= alphaMax && q > qc)
		printf("Alpha is not found!\n");
	else 
	{
		printf("Alpha minimum = %f   q = %f\n", alpha, q);
		for (i = 0; i < numOfDimensions+1; i++)
			printf("W[%d] = %1f\n", i, w[i]);
	}
	printf("numOfPoints = %d, numOfDimensions = %d, limit of iterations = %d, qc = %1f\n\n",
		numOfPoints, numOfDimensions, iterLimit, qc);
}

void printPointsData(double *points, int numOfPoints, int numOfDimensions, int *pointsGroups)
{
	int i, j;
	for (i = 0; i < numOfPoints; i++)
	{
		printf("\n");
		printf("point #%d: ", i);
		for (j = 0; j < numOfDimensions; j++)
			printf("%1f ", points[i*numOfDimensions+j]);
		printf("%d", pointsGroups[i]);
	}
}