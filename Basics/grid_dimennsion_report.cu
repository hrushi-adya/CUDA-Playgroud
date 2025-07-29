#include<stdio.h>
#include<cuda_runtime.h>

__global__ void reportGridDimensions(int blocksPerGrid, int threadsPerBlock) {
	printf("Block:%d Thread:%d Dimension:%d", blockIdx.x, threadIdx.x, grimDim.x);
}

int main(void) {
	int blocksPerGrid = 3;
	int threadsPerBlock = 8;
	
	reportGridDimension << <blocksPerGrid, threadsPerBlock >> > (blocksPerGrid, threadsPerBlock);
	cudaDeviceSynchronize();
	
	return 0;
}
