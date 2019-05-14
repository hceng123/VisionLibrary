#include <stdio.h>

#include "device_launch_parameters.h"
#include "CudaFunc.h"

template<typename T>
__device__ __host__
void sort2Value(T& a, T& b) {
    T t = a;
    a = min(a, b);
    b = max(b, t);
}

template<typename T>
__device__ __host__
T median9(T *p) {
    sort2Value(p[1], p[2]); sort2Value(p[4], p[5]); sort2Value(p[7], p[8]); sort2Value(p[0], p[1]);
    sort2Value(p[3], p[4]); sort2Value(p[6], p[7]); sort2Value(p[1], p[2]); sort2Value(p[4], p[5]);
    sort2Value(p[7], p[8]); sort2Value(p[0], p[3]); sort2Value(p[5], p[8]); sort2Value(p[4], p[7]);
    sort2Value(p[3], p[6]); sort2Value(p[1], p[4]); sort2Value(p[2], p[5]); sort2Value(p[4], p[7]);
    sort2Value(p[4], p[2]); sort2Value(p[6], p[4]); sort2Value(p[4], p[2]);
    return p[4];
}

template<typename T>
__device__ __host__
T median25(T *p) {
    sort2Value(p[1], p[2]); sort2Value(p[0], p[1]); sort2Value(p[1], p[2]); sort2Value(p[4], p[5]); sort2Value(p[3], p[4]);
    sort2Value(p[4], p[5]); sort2Value(p[0], p[3]); sort2Value(p[2], p[5]); sort2Value(p[2], p[3]); sort2Value(p[1], p[4]);
    sort2Value(p[1], p[2]); sort2Value(p[3], p[4]); sort2Value(p[7], p[8]); sort2Value(p[6], p[7]); sort2Value(p[7], p[8]);
    sort2Value(p[10], p[11]); sort2Value(p[9], p[10]); sort2Value(p[10], p[11]); sort2Value(p[6], p[9]); sort2Value(p[8], p[11]);
    sort2Value(p[8], p[9]); sort2Value(p[7], p[10]); sort2Value(p[7], p[8]); sort2Value(p[9], p[10]); sort2Value(p[0], p[6]);
    sort2Value(p[4], p[10]); sort2Value(p[4], p[6]); sort2Value(p[2], p[8]); sort2Value(p[2], p[4]); sort2Value(p[6], p[8]);
    sort2Value(p[1], p[7]); sort2Value(p[5], p[11]); sort2Value(p[5], p[7]); sort2Value(p[3], p[9]); sort2Value(p[3], p[5]);
    sort2Value(p[7], p[9]); sort2Value(p[1], p[2]); sort2Value(p[3], p[4]); sort2Value(p[5], p[6]); sort2Value(p[7], p[8]);
    sort2Value(p[9], p[10]); sort2Value(p[13], p[14]); sort2Value(p[12], p[13]); sort2Value(p[13], p[14]); sort2Value(p[16], p[17]);
    sort2Value(p[15], p[16]); sort2Value(p[16], p[17]); sort2Value(p[12], p[15]); sort2Value(p[14], p[17]); sort2Value(p[14], p[15]);
    sort2Value(p[13], p[16]); sort2Value(p[13], p[14]); sort2Value(p[15], p[16]); sort2Value(p[19], p[20]); sort2Value(p[18], p[19]);
    sort2Value(p[19], p[20]); sort2Value(p[21], p[22]); sort2Value(p[23], p[24]); sort2Value(p[21], p[23]); sort2Value(p[22], p[24]);
    sort2Value(p[22], p[23]); sort2Value(p[18], p[21]); sort2Value(p[20], p[23]); sort2Value(p[20], p[21]); sort2Value(p[19], p[22]);
    sort2Value(p[22], p[24]); sort2Value(p[19], p[20]); sort2Value(p[21], p[22]); sort2Value(p[23], p[24]); sort2Value(p[12], p[18]);
    sort2Value(p[16], p[22]); sort2Value(p[16], p[18]); sort2Value(p[14], p[20]); sort2Value(p[20], p[24]); sort2Value(p[14], p[16]);
    sort2Value(p[18], p[20]); sort2Value(p[22], p[24]); sort2Value(p[13], p[19]); sort2Value(p[17], p[23]); sort2Value(p[17], p[19]);
    sort2Value(p[15], p[21]); sort2Value(p[15], p[17]); sort2Value(p[19], p[21]); sort2Value(p[13], p[14]); sort2Value(p[15], p[16]);
    sort2Value(p[17], p[18]); sort2Value(p[19], p[20]); sort2Value(p[21], p[22]); sort2Value(p[23], p[24]); sort2Value(p[0], p[12]);
    sort2Value(p[8], p[20]); sort2Value(p[8], p[12]); sort2Value(p[4], p[16]); sort2Value(p[16], p[24]); sort2Value(p[12], p[16]);
    sort2Value(p[2], p[14]); sort2Value(p[10], p[22]); sort2Value(p[10], p[14]); sort2Value(p[6], p[18]); sort2Value(p[6], p[10]);
    sort2Value(p[10], p[12]); sort2Value(p[1], p[13]); sort2Value(p[9], p[21]); sort2Value(p[9], p[13]); sort2Value(p[5], p[17]);
    sort2Value(p[13], p[17]); sort2Value(p[3], p[15]); sort2Value(p[11], p[23]); sort2Value(p[11], p[15]); sort2Value(p[7], p[19]);
    sort2Value(p[7], p[11]); sort2Value(p[11], p[13]); sort2Value(p[11], p[12]);
    return p[12];
}

template<typename T>
__global__
void kernel_median_filter(
    const T* src,
    T* dst,
    const int ROWS,
    const int COLS,
    const int step,
    const int winSize) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= ROWS || c >= COLS)
        return;
    if (3 == winSize) {
        const T* row0 = src + max(r - 1, 0) * step;
        const T* row1 = src + r * step;
        const T* row2 = src + min(r + 1, ROWS - 1) * step;

        int j0 = c >= 1 ? c - 1 : c;
        int j2 = c < COLS - 1 ? c + 1 : c;

        T p[9];
        p[0] = row0[j0], p[1] = row0[c], p[2] = row0[j2];
        p[3] = row1[j0], p[4] = row1[c], p[5] = row1[j2];
        p[6] = row2[j0], p[7] = row2[c], p[8] = row2[j2];

        dst[r * step + c] = median9(p);
    }
    else if (5 == winSize) {
        const T* row[5];
        row[0] = src + max(r - 2, 0) * step;
        row[1] = src + max(r - 1, 0) * step;
        row[2] = src + r * step;
        row[3] = src + min(r + 1, ROWS - 1) * step;
        row[4] = src + min(r + 2, ROWS - 1) * step;

        T p[25];
        int j1 = c >= 1 ? c - 1 : c;
        int j0 = c >= 2 ? c - 2 : j1;
        int j3 = c < COLS - 1 ? c + 1 : c;
        int j4 = c < COLS - 2 ? c + 2 : j3;
        for (int k = 0; k < 5; k++)
        {
            const T* rowk = row[k];
            p[k * 5] = rowk[j0]; p[k * 5 + 1] = rowk[j1];
            p[k * 5 + 2] = rowk[c]; p[k * 5 + 3] = rowk[j3];
            p[k * 5 + 4] = rowk[j4];
        }

        dst[r * step + c] = median25(p);
    }
}

void run_median_filter(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    const float *src,
    float *dst,
    const int ROWS,
    const int COLS,
    const int step,
    const int winSize) {
    kernel_median_filter<<<grid, threads, 0, cudaStream>>>(src, dst, ROWS, COLS, step, winSize);
}
