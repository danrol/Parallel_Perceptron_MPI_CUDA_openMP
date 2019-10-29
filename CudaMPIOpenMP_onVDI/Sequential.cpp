#include "mainMethods.h"
#include "Sequential.h"

void resetArrToZeros(double ***arr, int arrSize)
{
	int i;
	for (i = 0; i < arrSize; i++)
		(**arr)[i] = 0;
}

double f(double **w, double* dimensions, int numOfW)
{
	int i;
	double result = (*w)[numOfW - 1];

	for (i = 0; i < numOfW - 1; i++)
		result += (*w)[i] * dimensions[i];

	return result;
}

int getFSign(double fVal)
{
	if (fVal >= 0)
		return GROUP1;
	else
		return GROUP2;
}

double getQuality(int numOfPoints, int numOfDimensions, double *points, int* pointGroups, double **w)
{
	int i, sign, signToCheck, numOfMiss = 0;
	double fVal;
	for (i = 0; i < numOfPoints; i++)
	{
		fVal = f(w, &(points[i*numOfDimensions]), numOfDimensions+1);
		sign = getFSign(fVal);
		signToCheck = pointGroups[i];
		if (sign != signToCheck)
			numOfMiss++;
	}
	return (double)numOfMiss / numOfPoints;
}

void perceptronSequentialSolution(int numOfPoints, int numOfDimensions,
	double alphaZero, double alphaMax, int iterLimit, double qc,
	double *points, int *pointGroups, double **w)
{
	int numOfW = numOfDimensions + 1;
	int i = 0, sign, iterCount;
	double q = qc + 1, alpha = 0, numOfMiss, fVal;
	*w = (double*)calloc(numOfW, sizeof(double));
	//double *optimalW = (double*)calloc(numOfDimensions + 1, sizeof(double));

	while (q > qc && alpha < alphaMax)
	{
		iterCount = 0;
		numOfMiss = 0;
		alpha += alphaZero;
		resetArrToZeros(&w, numOfW);
		while (iterCount < iterLimit)
		{
			i = 0;
			while (i < numOfPoints)
			{
				fVal = f(w, &(points[i*numOfDimensions]), numOfW);
				sign = getFSign(fVal);

				if (sign != pointGroups[i])
				{
					updateW(&w, numOfW, alpha, &(points[i*numOfDimensions]), sign);
					break;
				}
				i++;
			}
			iterCount++;
		}

		//Check if all points suit conditions
		if (i >= numOfPoints)
			q = 0;
		else
			q = getQuality(numOfPoints, numOfDimensions, points, pointGroups, w);
		//free(points);
		//free(pointGroups);
	}
}

void updateW(double ***w, int numOfW, double alpha, double* dimensions, int sign)
{
	int i;
	(**w)[numOfW - 1] += (-sign)*alpha;
	for (i = 0; i < numOfW - 1; i++)
		(**w)[i] += (-sign)*alpha*dimensions[i];
}