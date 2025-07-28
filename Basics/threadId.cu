#include<stdio.h>

__global__ void printThreadId() {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int gid = tid + bid * blockDim.x;
	printf("Thread Id: %d\n", tid);
	printf("Block Id: %d\n", bid);
	printf("Global id: %d\n", gid);
}

int main() {
	printThreadId <<<3, 4>>> ();
	cudaDeviceSynchronize();
	return 0;
}
