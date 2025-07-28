#include<stdio.h>

__global__ void fillArray(int* arr) {
	int tid = threadIdx.x;
	arr[tid] = tid * 10;
}

int main() {
	const int num = 5;
	int h_arr[num]; 
	int* d_arr;

	cudaMalloc((void**)&d_arr, num * sizeof(int));

	fillArray <<<1, num>>> (d_arr);
	cudaMemcpy(h_arr, d_arr, num * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < num; i++) {
		printf("h_arr[%d] = %d\n", i, h_arr[i]);
	}

	cudaFree(d_arr);

	return 0;
}
