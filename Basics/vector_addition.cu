#include<stdio.h>

__global__ void vectorAdd(int* d_a, int* d_b, int* d_result, int num) {
	int tid = threadIdx.x;
	if (tid < num) {
		d_result[tid] = d_a[tid] + d_b[tid];
	}
}

int main() {
	// declare vectors and result variable
	const int num = 5;
	int h_a[num] = {1, 2, 3, 4, 5};
	int h_b[num] = {10, 20, 30, 40, 50};
	int h_result[num];

	// declare device (GPU) pointers
	int* d_a;
	int* d_b;
	int* d_result;

	// allocate memory on gpu for them 
	cudaMalloc((void**)&d_a, num * sizeof(int));
	cudaMalloc((void**)&d_b, num * sizeof(int));
	cudaMalloc((void**)&d_result, num * sizeof(int));

	// copy data from host (CPU) to device (GPU)
	cudaMemcpy(d_a, h_a, num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, num * sizeof(int), cudaMemcpyHostToDevice);

	// launch kernel
	vectorAdd<<<1, num >>>(d_a, d_b, d_result, num);
	cudaDeviceSynchronize();

	// copy data from gpu to cpu with memcpy function
	cudaMemcpy(h_result, d_result, num * sizeof(int), cudaMemcpyDeviceToHost);

	// print the output
	printf("Result of vector addition: \n");
	for (int i = 0; i < num; i++) {
		printf("h_result[%d]: %d\n", i, h_result[i]);
	}

	// free the memory pointers
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);

	// reset the device
	// cudaDeviceReset();

	return 0;
}

/*
Output:
Result of vector addition :
h_result[0] : 11
h_result[1] : 22
h_result[2] : 33
h_result[3] : 44
h_result[4] : 55
*/ 
