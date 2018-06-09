
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "andor_cuda_kernel.h"

dim3 cuda_gridsize(int n)
{
    int k = (n - 1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if(x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    return d;
}

template <int TILE_DIM> __global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

  float CValue = 0;
  int Row = blockIdx.y*TILE_DIM + threadIdx.y;
  int Col = blockIdx.x*TILE_DIM + threadIdx.x;

  __shared__ float As[TILE_DIM][TILE_DIM];
  __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

      if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows) As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
      else As[threadIdx.y][threadIdx.x] = 0.0;

      if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)  Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
      else Bs[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();

      for (int n = 0; n < TILE_DIM; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

      __syncthreads();

  }

  if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;

}

/**
cuda binding for the kernel
*/
void andor_cuda(float *c, float *a, float *b, int dimsAx, int dimsAy, int dimsBx, int dimsBy, cudaStream_t stream)
{
    cudaError_t err;

    // Setup execution parameters
    dim3 threads(BLOCK_SIZE_USED, BLOCK_SIZE_USED);
    dim3 grid(ceil((float)dimsBx / threads.x), ceil((float)dimsAy / threads.y));

    MatMul<BLOCK_SIZE_USED> <<< grid, threads >>>(a, b, c, dimsAy, dimsAx, dimsBy, dimsBx, dimsAy, dimsBx);

    fprintf(stdout, "grid: %d (BLOCK_SIZE_USED) %d x %d, threads %d x %d", BLOCK_SIZE_USED, grid.x, grid.y, threads.x, threads.y);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}