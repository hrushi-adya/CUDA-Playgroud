#include<stdio.h>

__global__ void reverseArray(int* arr, int num) {
	int tid = threadIdx.x;
	if (tid < num / 2) {
		int temp = arr[tid];
		arr[tid] = arr[num - 1 - tid];
		arr[num - 1 - tid] = temp;
	}
}

int main() {
	int num = 5;
	int h_arr[5] = { 10, 20, 30, 40, 50 };
	int *d_arr;

	cudaMalloc((void**)&d_arr, num * sizeof(int));
	cudaMemcpy(d_arr, h_arr, num * sizeof(int), cudaMemcpyHostToDevice);

	// Launch kernel to reverse the array
	reverseArray << <1, num >> > (d_arr, num);

	cudaMemcpy(h_arr, d_arr, num * sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("Reversed array: \n");
	for (int i = 0; i < num; i++) {
		printf("%d\t", h_arr[i]);
	}

	cudaFree(d_arr);

	return 0;
}

/*
* Output:
* Reversed array:
* 50      40      30      20      10
*/
