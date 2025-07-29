#include<stdio.h>
#include<cuda_runtime.h>

__global__ void fillArrayInBatch(int* arr, int num) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = gridDim.x * blockDim.x;

	for (int i = tid; i < num; i = i + totalThreads) {
		arr[i] = i * 10;
	}
}

int main(void) {
	const int num = 100;
	int h_arr[num];
	int* d_arr;

	cudaMalloc((void**)&d_arr, num * sizeof(int));

	int threadsPerBlock = 32;
	int blocksPerGrid = 2;

	fillArrayInBatch << <blocksPerGrid, threadsPerBlock >> > (d_arr, num);

	cudaMemcpy(h_arr, d_arr, num * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Filled Array: \n");
	for (int i = 0; i < num; i++) {
		printf("%d\t", h_arr[i]);
	}

	cudaFree(d_arr);

	return 0;
}

/*
* Output:
* Filled Array:
* 0       10      20      30      40      50      60      70      80      90      100     110     120     130     140    150      160     170     180     190     200     210     220     230     240     250     260     270     280     290    300      310     320     330     340     350     360     370     380     390     400     410     420     430     440    450      460     470     480     490     500     510     520     530     540     550     560     570     580     590    600      610     620     630     640     650     660     670     680     690     700     710     720     730     740    750      760     770     780     790     800     810     820     830     840     850     860     870     880     890    900      910     920     930     940     950     960     970     980     990
*/
