#ifndef _MATHUTIL_CUDA_KERNEL
#define _MATHUTIL_CUDA_KERNEL

#define IDX2D(i, j, dj) (dj * i + j)
#define IDX3D(i, j, k, dj, dk) (IDX2D(IDX2D(i, j, dj), k, dk))

#define BLOCK 512
#define MAX_STREAMS 512
#define BLOCK_SIZE_USED 32

#ifdef __cplusplus
extern "C" {
#endif

void andor_cuda(float *c, float *a, float *b, int dimsAx, int dimsAy, int dimsBx, int dimsBy, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif