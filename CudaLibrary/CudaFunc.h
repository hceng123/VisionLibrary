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

void run_kernel_select_cmp_point(
    uint32_t gridSize,
    uint32_t blockSize,
    float* dMap,
    float* dPhase,
    uint8_t* matResult,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int span);

#endif