#ifndef _CUDA_FUNC_H_
#define _CUDA_FUNC_H_

#include <stdint.h>
#include "cuda_runtime.h"

void run_kernel_calc_phase(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    const unsigned char* pInput0,
    const unsigned char* pInput1,
    const unsigned char* pInput2,
    const unsigned char* pInput3,
    float* pResult,
    const int ROWS,
    const int COLS,
    const int step);

void run_kernel_calc_phase_and_dark_mask(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    const unsigned char* pInput0,
    const unsigned char* pInput1,
    const unsigned char* pInput2,
    const unsigned char* pInput3,
    float* pResult,
    unsigned char* pMask,
    float fMinimumAlpitudeSquare,
    const int ROWS,
    const int COLS,
    const int step);

void run_kernel_select_cmp_point(
    uint32_t gridSize,
    uint32_t blockSize,
    cudaStream_t cudaStream,
    float* dMap,
    float* dPhase,
    uint8_t* matResult,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int span);

void run_kernel_phase_correction(
    uint32_t gridSize,
    uint32_t blockSize,
    cudaStream_t cudaStream,
    float* phaseDiff,
    float* phase,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int span);

void run_kernel_floor(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    float *data,
    uint32_t step,
    const int ROWS,
    const int COLS);

void run_kernel_interval_average(
    cudaStream_t cudaStream,
    float *data,
    uint32_t step,
    const int ROWS,
    const int COLS,
    int interval,
    float *d_result,
    float *result);

void run_kernel_range_interval_average(
    cudaStream_t cudaStream,
    const float *data,
    uint32_t step,
    const int ROWS,
    const int COLS,
    int interval,
    const float rangeStart,
    const float rangeEnd,
    float* d_result,
    float *result);

void run_kernel_phase_to_height_3d(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    const float* pZInRow, // len x 1
    const float* pPolyParamsInput, // 1 x ss
    const float* xxt, // len x ss
    const int xxtStep,
    const int len,
    const int ss,
    float* pResult // len x ss
    );

void run_kernel_calc_sum_and_convert_matrix(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    float* pInputData,
    const int ss,
    float* pOutput,
    const int ROWS,
    const int COLS,
    const int step);

void run_kernel_merge_height_intersect(
    uint32_t gridSize,
    uint32_t blockSize,
    cudaStream_t cudaStream,
    float* matOne,
    float *matTwo,
    float *d_result,
    int rows,
    int cols,
    float fDiffThreshold);

void run_median_filter(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    const float *src,
    float *dst,
    const int ROWS,
    const int COLS,
    const int step,
    const int winSize);

#endif