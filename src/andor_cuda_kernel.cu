
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

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
//template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
//                                                        float *B, int wA,
//                                                        int wB) {
//    // Block index
//    int bx = blockIdx.x;
//    int by = blockIdx.y;
//
//    // Thread index
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//
//    // Index of the first sub-matrix of A processed by the block
//    int aBegin = wA * BLOCK_SIZE * by;
//
//    // Index of the last sub-matrix of A processed by the block
//    int aEnd   = aBegin + wA - 1;
//
//    // Step size used to iterate through the sub-matrices of A
//    int aStep  = BLOCK_SIZE;
//
//    // Index of the first sub-matrix of B processed by the block
//    int bBegin = BLOCK_SIZE * bx;
//
//    // Step size used to iterate through the sub-matrices of B
//    int bStep  = BLOCK_SIZE * wB;
//
//    // Csub is used to store the element of the block sub-matrix
//    // that is computed by the thread
//    float Csub = 0;
//
//    // Loop over all the sub-matrices of A and B
//    // required to compute the block sub-matrix
//    for (int a = aBegin, b = bBegin;
//            a <= aEnd;
//            a += aStep, b += bStep) {
//        // Declaration of the shared memory array As used to
//        // store the sub-matrix of A
//        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
//
//        // Declaration of the shared memory array Bs used to
//        // store the sub-matrix of B
//        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
//
//        // Load the matrices from device memory
//        // to shared memory; each thread loads
//        // one element of each matrix
//        As[ty][tx] = A[a + wA * ty + tx];
//        Bs[ty][tx] = B[b + wB * ty + tx];
//
//        // Synchronize to make sure the matrices are loaded
//        __syncthreads();
//
//        // Multiply the two matrices together;
//        // each thread computes one element
//        // of the block sub-matrix
//#pragma unroll
//
////        printf("thread calcs sum\n");
//        for (int k = 0; k < BLOCK_SIZE; ++k) {
////            printf("%d %d %d %d %d %d %f %f\n", BLOCK_SIZE, bx, by, ty, tx, k, As[ty][k], Bs[k][tx]);
//            Csub += As[ty][k] * Bs[k][tx];
//        }
//
//        // Synchronize to make sure that the preceding
//        // computation is done before loading two new
//        // sub-matrices of A and B in the next iteration
//        __syncthreads();
//    }
//
//    // Write the block sub-matrix to device memory;
//    // each thread writes one element
//    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
////    printf("cid = %d out %d\n", c, c + wB * ty + tx);
//    C[c + wB * ty + tx] = Csub;
//}

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

//    MatrixMulCUDA<BLOCK_SIZE_USED> <<< grid, threads >>>(c, a, b, dimsAx, dimsBx);
    MatMul<BLOCK_SIZE_USED> <<< grid, threads >>>(a, b, c, dimsAy, dimsAx, dimsBy, dimsBx, dimsAy, dimsBx);

    fprintf(stdout, "grid: %d (BLOCK_SIZE_USED) %d x %d, threads %d x %d", BLOCK_SIZE_USED, grid.x, grid.y, threads.x, threads.y);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}