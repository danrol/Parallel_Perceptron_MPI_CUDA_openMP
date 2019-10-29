#pragma once
#pragma warning (disable : 4996) //stop warnings for strtok
#ifndef MYAPP_H
#define myAPP_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mainMethods.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <time.h> 

#define INPUT_FILE_PATH "C:\\Users\\Public\\Desktop\\SQL Server Management Studio\\logs\\CudaMPIOpenMP_onVDI\\CudaMPIOpenMP_onVDI\\inputOutpuFiles\\data1.txt"
#define OUTPUT_FILE_PATH "C:\\Users\\Public\\Desktop\\SQL Server Management Studio\\logs\\CudaMPIOpenMP_onVDI\\CudaMPIOpenMP_onVDI\\inputOutpuFiles\\output.txt"
#define FILE_OPEN_ERROR "Error opening file"
#define INTGR "int"
#define DBL "double"
#define GROUP1 1
#define GROUP2 -1
#define MASTER 0
#define HUGE_NUMBER 1000000000
#define DEFAULT_ALPHA_CHUNK_SIZE 3
#define STOP_RECV_ALPHAS 0
#define CONTINUE_RECV_ALPHAS 1


cudaError_t findPointGroupsforWByCuda(double *w, double *deviceW, double *devicePoints, int *devicePointsSigns, int *deviceNumOfDimensions,
	int numOfPoints, int numOfDimensions, int **pointsGroupsResult);
double* allocateWOnDevice(int numOfDimensions);
double *allocateAndCopyPointsToDevice(int numOfPoints, int numOfDimensions, double *points);
int* allocateDevicePointsSigns(int numOfPoints);
int* allocateAndCopyNumOfDimensionsToDevice(int numOfDimensions);
void freeCudaValues(double *deviceW, double *devicePoints, int *devicePointsSigns, int *deviceNumOfDimensions);
#endif

