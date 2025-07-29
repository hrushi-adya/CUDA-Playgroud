#include<stdio.h>
#inlude<cuda_runtime.h>

__global__ void doubleArray(int* arr, int num) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) {
		arr[tid] = arr[tid] * 2;
	}
}

int main() {
	const int num = 20;
	int arr[num];
	int* d_arr;

	for (int i = 0; i < num; i++) {
		arr[i] = i;
	}

	cudaMalloc((void**)&d_arr, num * sizeof(int));
	cudaMemcpy(d_arr, arr, num * sizeof(int), cudaMemcpyHostToDevice);

	int threadsPerBlock = 8;
	int blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;

	// Launch kernel to fill the array
	doubleArray<<<blocksPerGrid, threadsPerBlock >>>(d_arr, num);
	cudaMemcpy(arr, d_arr, num * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Doubled Array:\n");
	for (int i = 0; i < num; i++) {
		printf("%d\t", arr[i]);
	}

	cudaFree(d_arr);

	return 0;
}
