#ifndef _CUDA_FUNC_H_
#define _CUDA_FUNC_H_

#include <stdint.h>

void run_kernel_merge_height_intersect(
    uint32_t gridSize,
    uint32_t blockSize,
    float* matOne,
    float *matTwo,
    float *d_result,
    int rows,
    int cols,
    float fDiffThreshold);

#endif