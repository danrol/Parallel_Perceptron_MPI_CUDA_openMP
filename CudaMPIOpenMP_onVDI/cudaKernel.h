#pragma once
#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H
#define CHECK_ERRORS(status, msg, retValue) if ((status) != cudaSuccess) { fprintf(stderr, (msg)); return (retValue); }
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__device__ int getFSignCuda(double fVal);

__global__ void fOnGPUKernel(double *w, double* devicePoints, int *numOfDimensions, int *cudaPointsSigns);

double* allocateWOnDevice(int numOfDimensions);
double *allocateAndCopyPointsToDevice(int numOfPoints, int numOfDimensions, double *points);
int* allocateDevicePointsSigns(int numOfPoints);
int* allocateAndCopyNumOfDimensionsToDevice(int numOfDimensions);
void freeCudaValues(double *deviceW, double *devicePoints, int *devicePointsSigns, int *deviceNumOfDimensions);

cudaError_t findPointGroupsforWByCuda(double *w, double *deviceW, double *devicePoints, int *devicePointsSigns, int *deviceNumOfDimensions,
	int numOfPoints, int numOfDimensions, int **pointsGroupsResult);
#endif