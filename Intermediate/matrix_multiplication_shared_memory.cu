// ------------------------------------------------------------------------
// Matrix multiplication using shared memory in CUDA
// Author: Hrushikesh Adya
// Description: Matrix multiplication using shared memory for optimization.
// ------------------------------------------------------------------------

#include<stdio.h>
#include<cuda_runtime.h>

// ------------------------------------------------------------------------
// Macro Definitions
// ------------------------------------------------------------------------

#define BLOCK_SIZE	16
#define MATRIX_DIM	16

// ------------------------------------------------------------------------
// Forward Declarations
// ------------------------------------------------------------------------

__global__ void matrixMultiplicationKernel(const float* A, const float* B, float* C, int width);

// ------------------------------------------------------------------------
// Kernel Function: Matrix Multiplication using Shared Memory
// ------------------------------------------------------------------------q

__global__ void matrixMultiplicationKernel(const float* A, const float* B, float* C, int width) {
	__shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	float sum = 0.0f;

	for (int m = 0; m < width / BLOCK_SIZE; m++) {

		tileA[threadIdx.y][threadIdx.x] = A[row * width + (m * BLOCK_SIZE + threadIdx.x)];
		tileB[threadIdx.y][threadIdx.x] = B[(m * BLOCK_SIZE + threadIdx.y) * width + col];

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; k++) {
			sum = sum + tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
		}
		__syncthreads();
	}
	C[row * width + col] = sum;

}

// ------------------------------------------------------------------------
// Main Function
// ------------------------------------------------------------------------

int main(void) {
	const int size = MATRIX_DIM * MATRIX_DIM;
	const int bytes = size * sizeof(float);

	float* h_A = (float*)malloc(bytes);
	float* h_B = (float*)malloc(bytes);
	float* h_C = (float*)malloc(bytes);

	float* d_A;
	float* d_B;
	float* d_C;

	for (int i = 0; i < size; i++) {
		h_A[i] = 1.0f;
		h_B[i] = 2.0f;
	}

	cudaMalloc((void**)&d_A, bytes);
	cudaMalloc((void**)&d_B, bytes);
	cudaMalloc((void**)&d_C, bytes);

	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid(MATRIX_DIM / BLOCK_SIZE, MATRIX_DIM / BLOCK_SIZE);

	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, MATRIX_DIM);

	cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	printf("Result Matrix C:\n");
	for (int i = 0; i < MATRIX_DIM; i++) {
		for (int j = 0; j < MATRIX_DIM; j++) {
			printf("%.2f\t", h_C[i * MATRIX_DIM + j]);
		}
		printf("\n");
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

/*
Output: 
Result Matrix C:
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00   32.00
*/
