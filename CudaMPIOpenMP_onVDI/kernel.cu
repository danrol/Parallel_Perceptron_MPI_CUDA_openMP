#include "mainMethods.h"
#include "cudaKernel.h"
#define MAX_THREADS_PER_BLOCK 1000

//check with wich group point associated by f value
__device__ int getFSignCuda(double fVal)
{
	if (fVal >= 0)
		return GROUP1;
	else
		return GROUP2;
}

//one thread finds f for one point and the sign
__global__ void fOnGPUKernel(double *w, double *devicePoints, int *numOfDimensions, int *cudaPointsSigns)
{
	int i;
	int index = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	int fval = w[*numOfDimensions];

	for (i = 0; i < *numOfDimensions; i++)
	{
		fval += w[i] * devicePoints[index*(*numOfDimensions) + i];
	}
	cudaPointsSigns[index] = getFSignCuda(fval);
}

double* allocateWOnDevice(int numOfDimensions)
{
	double *deviceW = 0;
	cudaError cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to set Device inside allocateWOnDevice\n");

	cudaStatus = cudaMalloc((void**)&deviceW, (numOfDimensions + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to cudaMalloc of w\n");

	return deviceW;
}

double *allocateAndCopyPointsToDevice(int numOfPoints, int numOfDimensions, double *points)
{
	cudaError cudaStatus;
	double *devicePoints;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to set Device inside allocateAndCopyPointsToDevice\n");

	//start Allocate and copy points to CUDA
	cudaStatus = cudaMalloc((void**)&devicePoints, numOfPoints*numOfDimensions * sizeof(double));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to cudaMalloc of devicePoints\n");

	cudaStatus = cudaMemcpy(devicePoints, points, numOfPoints*numOfDimensions * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to memCpy of points to device\n");
	//finish Allocate and copy points to CUDA
	return devicePoints;
}

int* allocateDevicePointsSigns(int numOfPoints)
{
	cudaError cudaStatus;
	int *devicePointsSigns;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to set Device inside allocateDevicePointsSigns\n");

	//Allocate GPU buffer for device points signs
	cudaStatus = cudaMalloc((void**)&devicePointsSigns, numOfPoints * sizeof(int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to cudaMalloc of devicePointsSigns\n");

	return devicePointsSigns;
}

int* allocateAndCopyNumOfDimensionsToDevice(int numOfDimensions)
{
	int *deviceNumOfDimensions;
	cudaError cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to set Device inside allocateAndCopyNumOfDimensionsToDevice\n");

	//start Allocate and copy numOfDimensions to CUDA
	cudaStatus = cudaMalloc((void**)&deviceNumOfDimensions, sizeof(int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to cudaMalloc of deviceNumOfDimensions\n");

	cudaStatus = cudaMemcpy(deviceNumOfDimensions, &numOfDimensions, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Failed to memCpy of numOfDimensions to device\n");
	//finish Allocate and copy numOfDimensions to CUDA

	return deviceNumOfDimensions;
}

//free cuda values at the end
void freeCudaValues(double *deviceW, double *devicePoints, int *devicePointsSigns, int *deviceNumOfDimensions)
{
	cudaFree(deviceW);
	cudaFree(devicePoints);
	cudaFree(devicePointsSigns);
	cudaFree(deviceNumOfDimensions);
}

cudaError_t findPointGroupsforWByCuda(double *w, double *deviceW, double *devicePoints, int *devicePointsSigns, int *deviceNumOfDimensions,
	int numOfPoints, int numOfDimensions, int **pointsGroupsResult)
{
	int numOfBlocks = numOfPoints / MAX_THREADS_PER_BLOCK + 1;
	int *tempPointsGroupResult = 0;

	cudaError_t cudaStatus = cudaSuccess;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	//start Copy weights to CUDA
	cudaStatus = cudaMemcpy(deviceW, w, (numOfDimensions + 1) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "CudaMemCpy of deviceW failed!");
	//finish Copy w to CUDA

	/***********************************************************************************/
	//perform f on gpu using number of blocks with 1000 threads
	fOnGPUKernel << <numOfBlocks, MAX_THREADS_PER_BLOCK >> > (deviceW, devicePoints, deviceNumOfDimensions, devicePointsSigns);
	//printf("\nfinished perform f on gpU using number of blocks with 1000 threads\n");
	/***********************************************************************************/

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "fOnGPUKernel launch failed");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Cuda Sync10launch failed");

	// Copy devicePointsSigns from GPU buffer to host memory.
	tempPointsGroupResult = (int*)malloc(numOfPoints * sizeof(int));
	cudaStatus = cudaMemcpy((void *)tempPointsGroupResult, (void *)(devicePointsSigns), numOfPoints * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "CudaMemCpy of tempPointsGroupResult failed");
	(*pointsGroupsResult) = tempPointsGroupResult;
	// End Copy devicePointsSigns from GPU buffer to host memory.

	return cudaStatus;
}