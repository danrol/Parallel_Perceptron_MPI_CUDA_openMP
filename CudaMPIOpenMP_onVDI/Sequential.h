#pragma once

#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

void resetArrToZeros(double ***arr, int arrSize);
double getQuality(int numOfPoints, int numOfDimensions, double *points, int* pointGroups, double **w);
void perceptronSequentialSolution(int numOfPoints, int numOfDimensions, double alphaZero, 
	double alphaMax, int iterLimit, double qc, double *points, int *pointGroups, double **w);
double f(double **w, double* dimensions, int numOfW);
int getFSign(double fVal);
void updateW(double ***w, int numOfW, double alpha, double* dimensions, int sign);

#endif