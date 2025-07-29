// ---------------------------------------------------------------------------
// matricx_multiplication_basic.cu - Basic Matrix Multiplication using CUDA
// Author: Hrushikesh Adya
// Descrition: Basic Matrix Multiplication (no shared memory).
// ---------------------------------------------------------------------------

#include<stdio.h>
#include<cuda_runtime.h>

// ---------------------------------------------------------------------------
// Macro Definitions
// ---------------------------------------------------------------------------
#define MATRIX_DIM	16
#define BLOCK_SIZE	16

// ---------------------------------------------------------------------------
// Function Prototypes
// ---------------------------------------------------------------------------
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int width);

// ---------------------------------------------------------------------------
// Kernel Function: Matrix Multiplication
// ---------------------------------------------------------------------------

__global__ void matrixMultiplicationKernel(const float* A, const float* B, float* C, int width) {
	// Compute global row and column indices
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Each thread computes one element of C
	if (row < width && col < width) {
		float sum = 0.0f;
		for (int i = 0; i < width; i++) {
			sum = sum + A[row * width + i] * B[i * width + col];
		}
		C[row * width + col] = sum;
	}
}

// ---------------------------------------------------------------------------
// Main Function
// ---------------------------------------------------------------------------

int main(void) {

	const int SIZE = MATRIX_DIM * MATRIX_DIM;
	const int BYTES = SIZE * sizeof(float);

	float* h_A = (float*)malloc(BYTES);
	float* h_B = (float*)malloc(BYTES);
	float* h_C = (float*)malloc(BYTES);

	float* d_A;
	float* d_B;
	float* d_C;

	for (int i = 0; i < SIZE; i++) {
		h_A[i] = 1.0f;
		h_B[i] = 2.0f;
	}

	cudaMalloc((void**)&d_A, BYTES);
	cudaMalloc((void**)&d_B, BYTES);
	cudaMalloc((void**)&d_C, BYTES);

	cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);

	// Kernel launch configuration
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid(MATRIX_DIM/BLOCK_SIZE, MATRIX_DIM/BLOCK_SIZE);

	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, MATRIX_DIM);

	cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost);

	printf("Result Matrix C:\n");
	for(int i = 0; i < MATRIX_DIM; i++) {
		for (int j = 0; j < MATRIX_DIM; j++) {
			printf("%.2f \t", h_C[i * MATRIX_DIM + j]);
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
