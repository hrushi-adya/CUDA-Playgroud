#include<stdio.h>

__global__ void scaleArray(int* arr, int num) {
	int tid = threadIdx.x;
	arr[tid] = arr[tid] * num;
}
int main() {
	const int num = 5;
	int h_arr[num] = {1, 2, 3, 4, 5};
	int* d_arr;

	cudaMalloc((void**)&d_arr, num * sizeof(int));
	cudaMemcpy(d_arr, h_arr, num * sizeof(int), cudaMemcpyHostToDevice);

	scaleArray <<<1, num >>>(d_arr, 2);
	
	cudaMemcpy(h_arr, d_arr, num * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Scaled Array: \n");
	for (int i = 0; i < num; i++) {
		printf("%d \t", h_arr[i]);
	}

	cudaFree(d_arr);

	return 0; 
} 

/*
 * Output:
 * Scaled Array:
 * 2       4       6       8       10
 */ 
