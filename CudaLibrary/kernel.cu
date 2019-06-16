#include <stdio.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include "device_launch_parameters.h"
#include "CudaFunc.h"

#define MAX_DEPTH       16
#define INSERTION_SORT  32
#define PI 3.141592654f

__constant__ float ONE_HALF_CYCLE = 1.f;
__constant__ float ONE_CYCLE = 2.f;

__global__
void kernel_calc_phase(
    const unsigned char* pInput0,
    const unsigned char* pInput1,
    const unsigned char* pInput2,
    const unsigned char* pInput3,
    float* pResult,
    const int ROWS,
    const int COLS,
    const int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < COLS && j < ROWS) {
        int index = j * step + i;
        float x = (float)pInput0[index] - (float)pInput2[index];  // x = 2*cos(beta)
        float y = (float)pInput3[index] - (float)pInput1[index];  // y = 2*sin(beta)
        pResult[index] = atan2(y, x) / PI;
    }
}

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
    const int step) {
    kernel_calc_phase<<<grid, threads, 0, cudaStream>>>(pInput0, pInput1, pInput2, pInput3, pResult, ROWS, COLS, step);
}

__global__
void kernel_calc_phase_and_dark_mask(
    const unsigned char* pInput0,
    const unsigned char* pInput1,
    const unsigned char* pInput2,
    const unsigned char* pInput3,
    float* pResult,
    unsigned char* pMask,
    float fMinimumAlpitudeSquare,
    const int ROWS,
    const int COLS,
    const int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < COLS && j < ROWS) {
        int index = j * step + i;
        float x = (float)pInput0[index] - (float)pInput2[index];  // x = 2*cos(beta)
        float y = (float)pInput3[index] - (float)pInput1[index];  // y = 2*sin(beta)
        pResult[index] = atan2(y, x) / PI;

        float fAmplitudeSquare = (x*x + y*y) / 4.f;
        if (fAmplitudeSquare < fMinimumAlpitudeSquare)
            pMask[index] = 255;
    }
}

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
    const int step) {
    kernel_calc_phase_and_dark_mask<<<grid, threads, 0, cudaStream>>>(pInput0, pInput1, pInput2, pInput3,
        pResult, pMask, fMinimumAlpitudeSquare, ROWS, COLS, step);
}

template<typename T>
__device__ void selection_sort(T *data, int left, int right)
{
    for (int i = left ; i <= right ; ++i)
    {
        unsigned min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            unsigned val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
template<typename T>
__device__ void cdp_simple_quicksort(T *data, int left, int right, int depth)
{
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)
    {
        selection_sort(data, left, right);
        return;
    }

    T *lptr = data+left;
    T *rptr = data+right;
    T  pivot = data[(left+right)/2];

    // Do the partitioning.
    while (lptr <= rptr)
    {
        // Find the next left- and right-hand values to swap
        T lval = *lptr;
        T rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot)
        {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot)
        {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data))
    {
        cdp_simple_quicksort(data, left, nright, depth+1);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cdp_simple_quicksort(data, nleft, right, depth+1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void kernel_qsort(T *data, unsigned int nitems)
{
    // Launch on device
    int left = 0;
    int right = nitems - 1;
    cdp_simple_quicksort(data, left, right, 0);
}

// The dMap and dPhase is the X diff of Map and Phase
// Their size suppose to be ROW x (COL - 1)
// But to reuse the GPU buffer, their size is not redefined
// So their size still ROW X COL, so the COL index should start
// to from 1 to make the result same as ROW x (COL - 1)
__global__
void kernel_select_cmp_point(
    float* dMap,
    float* dPhase,
    uint8_t* matResult,
    uint32_t step,
    const int ROWS,
    const int COLS,
    int* pBufferArrayIdx1,
    int* pBufferArrayIdx2,
    const int span) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (start >= ROWS)
        return;

    for (int row = start; row < ROWS; row += stride) {
        int rowDataOffset = row * step;
        int* arrayIdx1 = pBufferArrayIdx1 + row * 1024;
        int* arrayIdx2 = pBufferArrayIdx2 + row * 1024;

        float* dMapRow = dMap + rowDataOffset;
        float* dPhaseRow = dPhase + rowDataOffset;
        uint8_t* matResultRow = matResult + rowDataOffset;

        int countOfIdx1 = 0, countOfIdx2 = 0;
        for (int i = 0; i < COLS - 1; ++ i) {
            const auto& value = dMapRow[i + 1];
            if (value == 1.f)
                arrayIdx1[countOfIdx1++] = i;
            else if (value == -1.f)
                arrayIdx2[countOfIdx2++] = i;
        }

        if (countOfIdx1 > 0 && countOfIdx2 > 0) {
            // If start of index1 larger than index 2, then remove beginning element of index2
            if (arrayIdx1[0] > arrayIdx2[0]) {
                countOfIdx2--;
                for (int i = 0; i < countOfIdx2; ++i)
                    arrayIdx2[i] = arrayIdx2[i + 1];
            }

            if (countOfIdx1 > countOfIdx2) {
                countOfIdx1--;
            }
        }

        for (int i = 0; i < countOfIdx1 && i < countOfIdx2; ++i) {
            if (arrayIdx2[i] - arrayIdx1[i] < span) {
                if (fabs(dPhaseRow[arrayIdx1[i] + 1]) > 1.f && fabs(dPhaseRow[arrayIdx2[i] + 1]) > 1.f) {
                    for (int k = arrayIdx1[i]; k <= arrayIdx2[i]; ++k) {
                        matResultRow[k] = 255;
                    }
                }
            }
        }
    }
}

void test_kernel_select_cmp_point(
    float* dMap,
    float* dPhase,
    uint8_t* matResult,
    const int ROWS,
    const int COLS,
    const int span) {
    int start = 0;
    int stride = 1;

    int *arrayIdx1 = (int *)malloc(COLS / 4 * sizeof(int));
    int *arrayIdx2 = (int *)malloc(COLS / 4 * sizeof(int));

    for (int row = start; row < ROWS; row += stride) {
        int offsetOfInput = row * COLS;
        int offsetOfResult = row * COLS;

        float* dMapRow = dMap + offsetOfInput;
        float* dPhaseRow = dPhase + offsetOfInput;
        uint8_t* matResultRow = matResult + offsetOfResult;

        int countOfIdx1 = 0, countOfIdx2 = 0;
        for (int i = 0; i < COLS - 1; ++ i) {
            const auto& value = dMapRow[i];
            if (value == 1.f)
                arrayIdx1[countOfIdx1++] = i;
            else if (value == -1.f)
                arrayIdx2[countOfIdx2++] = i;
        }

        if (countOfIdx1 > 0 && countOfIdx2 > 0) {
            // If start of index1 larger than index 2, then remove beginning element of index2
            if (arrayIdx1[0] > arrayIdx2[0]) {
                countOfIdx2--;
                for (int i = 0; i < countOfIdx2; ++i)
                    arrayIdx2[i] = arrayIdx2[i + 1];
            }

            if (countOfIdx1 > countOfIdx2) {
                countOfIdx1--;
            }
        }

        for (int i = 0; i < countOfIdx1 && i < countOfIdx2; ++i) {
            if (arrayIdx2[i] - arrayIdx1[i] < span) {
                if (fabs(dPhaseRow[arrayIdx1[i]]) > 1.f && fabs(dPhaseRow[arrayIdx2[i]]) > 1.f) {
                    for (int k = arrayIdx1[i]; k <= arrayIdx2[i]; ++k)
                        matResultRow[k] = 255;
                }
            }
        }
    }

    free(arrayIdx1);
    free(arrayIdx2);
}

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
    int* pBufferArrayIdx1,
    int* pBufferArrayIdx2,
    const int span) {
    kernel_select_cmp_point<<<gridSize, blockSize, 0, cudaStream>>>(
        dMap,
        dPhase,
        matResult,
        step,
        ROWS,
        COLS,
        pBufferArrayIdx1,
        pBufferArrayIdx2,
        span);
    //test_kernel_select_cmp_point(dMap, dPhase, matResult, ROWS, COLS, span);
}

__device__
void swapValue(int& value1, int &value2) {
    value1 = value1 + value2;
    value2 = value1 - value2;
    value1 = value1 - value2;
}

__device__
void kernel_sort_index_value(int* values, int size, int *sortedIndex) {
    for (int i = 0; i < size; ++ i)
        sortedIndex[i] = i;

    for(int i = 0; i < size - 1; ++ i) {
        for (int j = i + 1; j < size; ++j) {
            if (values[i] > values[j]) {
                swapValue(values[i], values[j]);
                swapValue(sortedIndex[i], sortedIndex[j]);
            }
        }
    }
}

__global__
void kernel_phase_correction(
    float* phaseDiff,
    float* phase,
    char* pBufferSign,
    char* pBufferAmpl,
    int* pBufferJumpSpan,
    int* pBufferJumpStart,
    int* pBufferJumpEnd,
    int* pBufferSortedJumpSpanIdx,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int spanThres) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= COLS - 1 || r >= ROWS)
        return;

    int dataRowOffset = r * step;
    float* phaseDiffRow = phaseDiff + dataRowOffset;
    char* vecSignOfRow = pBufferSign + dataRowOffset;
    char* vecAmplOfRow = pBufferAmpl + dataRowOffset;

    auto value = phaseDiffRow[c + 1];
    if (value > ONE_HALF_CYCLE) {
        vecSignOfRow[c] = 1;
        vecAmplOfRow[c] = static_cast<char> (std::ceil(std::fabs(value) / 2.f) * 2);
    }
    else if (value < -ONE_HALF_CYCLE) {
        vecSignOfRow[c] = -1;
        vecAmplOfRow[c] = static_cast<char> (std::ceil(std::abs(value) / 2.f) * 2);
    }

    cooperative_groups::grid_group grp = cooperative_groups::this_grid();
    grp.sync();

    if (c > 0)
        return;

    int offsetOfBuffer = r * 1024;
    int* vecJumpSpan = pBufferJumpSpan + offsetOfBuffer;
    int* vecJumpStart = pBufferJumpStart + offsetOfBuffer;
    int* vecJumpEnd = pBufferJumpEnd + offsetOfBuffer;
    int* vecSortedJumpSpanIdx = pBufferSortedJumpSpanIdx + offsetOfBuffer;

    float* phaseRow = phase + dataRowOffset;

    for (int kk = 0; kk < 2; ++kk) {
        int jumpSpanCount = 0;
        char lastSign = 0;
        int lastCol = 0;
        for (int col = 0; col < COLS - 1; ++col) {
            if (vecSignOfRow[col] != 0) {
                if (col - lastCol < spanThres && vecSignOfRow[col] * lastSign == -1) {
                    vecJumpSpan[jumpSpanCount] = col - lastCol;
                    vecJumpStart[jumpSpanCount] = lastCol;
                    vecJumpEnd[jumpSpanCount] = col;
                    ++jumpSpanCount;
                }

                lastCol = col;
                lastSign = vecSignOfRow[col];
            }
        }

        if (jumpSpanCount <= 0)
            break;

        kernel_sort_index_value(vecJumpSpan, jumpSpanCount, vecSortedJumpSpanIdx);

        for (int jj = 0; jj < jumpSpanCount; ++jj) {
            auto nStart = vecJumpStart[vecSortedJumpSpanIdx[jj]];
            auto nEnd = vecJumpEnd[vecSortedJumpSpanIdx[jj]];
            char chSignFirst = vecSignOfRow[nStart];        //The index is hard to understand. Use the sorted span index to find the original column.
            char chSignSecond = vecSignOfRow[nEnd];

            if (chSignFirst * chSignSecond == -1) { //it is a pair
                char chAmplFirst = vecAmplOfRow[nStart];
                char chAmplSecond = vecAmplOfRow[nEnd];
                char chTurnAmpl = min(chAmplFirst, chAmplSecond) / 2;

                char chAmplNew = chAmplFirst - 2 * chTurnAmpl;
                vecAmplOfRow[nStart] = chAmplNew;
                if (chAmplNew <= 0)
                    vecSignOfRow[nStart] = 0;  // Remove the sign of jump flag.

                chAmplNew = chAmplSecond - 2 * chTurnAmpl;
                vecAmplOfRow[nEnd] = chAmplNew;
                if (chAmplNew <= 0)
                    vecSignOfRow[nEnd] = 0;

                auto startValue = phaseRow[nStart];
                for (int col = nStart + 1; col <= nEnd; ++col) {
                    phaseRow[col] -= chSignFirst * ONE_CYCLE * chTurnAmpl;
                    if (chSignFirst > 0 && phaseRow[col] < startValue) { //Jump up, need to roll down, but can not over roll
                        phaseRow[col] = startValue;
                    }
                    else if (chSignFirst < 0 && phaseRow[col] > startValue) { //Jump down, need to roll up
                        phaseRow[col] = startValue;
                    }
                }
            }
        }
    }
}

void cpuSwapValue(int& value1, int &value2) {
    value1 = value1 + value2;
    value2 = value1 - value2;
    value1 = value1 - value2;
}

void cpu_sort_index_value(int* values, int size, int *sortedIndex) {
    for (int i = 0; i < size; ++ i)
        sortedIndex[i] = i;

    for(int i = 0; i < size - 1; ++ i) {
        for (int j = i + 1; j < size; ++j) {
            if (values[i] > values[j]) {
                cpuSwapValue(values[i], values[j]);
                cpuSwapValue(sortedIndex[i], sortedIndex[j]);
            }
        }
    }
}

void cpu_kernel_phase_correction(
    float* phaseDiff,
    float* phase,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int span) {
    int start = 0;
    int stride = 1;

    const float ONE_HALF_CYCLE = 1.f;
    const float ONE_CYCLE = 2.f;

    char* vecSignOfRow = (char*)malloc(COLS * sizeof(char));
    char* vecAmplOfRow = (char*)malloc(COLS * sizeof(char));
    int* vecJumpCol = (int*)malloc(COLS / 4 * sizeof(int));
    int* vecJumpSpan = (int*)malloc(COLS / 4 * sizeof(int));
    int* vecSortedJumpSpanIdx = (int*)malloc(COLS / 4 * sizeof(int));
    int* vecJumpIdxNeedToHandle = (int*)malloc(COLS / 4 * sizeof(int));

    const int DEBUG_ROW = 1588;

    for (int row = start; row < ROWS; row += stride) {
        int offsetOfData = row * step;

        bool bRowWithPosJump = false, bRowWithNegJump = false;
        memset(vecSignOfRow, 0, COLS);
        memset(vecAmplOfRow, 0, COLS);

        float* phaseDiffRow = phaseDiff + offsetOfData;
        float* phaseRow = phase + row * COLS;

        for (int col = 0; col < COLS - 1; ++col) {
            auto value = phaseDiffRow[col];
            if (value > ONE_HALF_CYCLE) {
                vecSignOfRow[col] = 1;
                bRowWithPosJump = true;

                char nJumpAmplitude = static_cast<char> (ceil(fabs(value) / 2.f) * 2);
                vecAmplOfRow[col] = nJumpAmplitude;
            }

            else if (value < -ONE_HALF_CYCLE) {
                vecSignOfRow[col] = -1;
                bRowWithNegJump = true;

                char nJumpAmplitude = static_cast<char> (std::ceil(std::abs(value) / 2.f) * 2);
                vecAmplOfRow[col] = nJumpAmplitude;
            }
        }

        if (!bRowWithPosJump || !bRowWithNegJump)
            continue;

        for (int kk = 0; kk < 2; ++ kk) {
            int jumpColCount = 0, jumpSpanCount = 0, jumpIdxNeedToHandleCount = 0;
            for (int col = 0; col < COLS; ++col) {
                if (vecSignOfRow[col] != 0)
                    vecJumpCol[jumpColCount++] = col;
            }
            if (jumpColCount < 2)
                continue;

            for (size_t i = 1; i < jumpColCount; ++i)
                vecJumpSpan[jumpSpanCount++] = vecJumpCol[i] - vecJumpCol[i - 1];
            cpu_sort_index_value(vecJumpSpan, jumpSpanCount, vecSortedJumpSpanIdx);
            for (int i = 0; i < jumpSpanCount; ++i) {
                if (vecJumpSpan[i] < span)
                    vecJumpIdxNeedToHandle[jumpIdxNeedToHandleCount++] = i;
            }

            if (DEBUG_ROW == row) {
                printf("jumpColCount = %d, jumpSpanCount = %d, jumpIdxNeedToHandleCount = %d\n", jumpColCount, jumpSpanCount, jumpIdxNeedToHandleCount);
                for (size_t jj = 0; jj < jumpIdxNeedToHandleCount; ++jj) {
                    auto nStart = vecJumpCol[vecSortedJumpSpanIdx[jj]];
                    auto nEnd = vecJumpCol[vecSortedJumpSpanIdx[jj] + 1];
                    printf("%d %d\n", nStart, nEnd);
                }
            }

            for (size_t jj = 0; jj < jumpIdxNeedToHandleCount; ++jj) {
                auto nStart = vecJumpCol[vecSortedJumpSpanIdx[jj]];
                auto nEnd = vecJumpCol[vecSortedJumpSpanIdx[jj] + 1];
                char chSignFirst = vecSignOfRow[nStart];        //The index is hard to understand. Use the sorted span index to find the original column.
                char chSignSecond = vecSignOfRow[nEnd];

                char chAmplFirst = vecAmplOfRow[nStart];
                char chAmplSecond = vecAmplOfRow[nEnd];
                char chTurnAmpl = min(chAmplFirst, chAmplSecond) / 2;
                if (chSignFirst * chSignSecond == -1) { //it is a pair
                    char chAmplNew = chAmplFirst - 2 * chTurnAmpl;
                    vecAmplOfRow[nStart] = chAmplNew;
                    if (chAmplNew <= 0)
                        vecSignOfRow[nStart] = 0;  // Remove the sign of jump flag.

                    chAmplNew = chAmplSecond - 2 * chTurnAmpl;
                    vecAmplOfRow[nEnd] = chAmplNew;
                    if (chAmplNew <= 0)
                        vecSignOfRow[nEnd] = 0;

                    auto startValue = phaseRow[nStart];
                    for (int col = nStart + 1; col <= nEnd; ++col) {
                        phaseRow[col] -= chSignFirst * ONE_CYCLE * chTurnAmpl;
                        if (chSignFirst > 0 && phaseRow[col] < startValue) { //Jump up, need to roll down, but can not over roll
                            phaseRow[col] = startValue;
                        }
                        else if (chSignFirst < 0 && phaseRow[col] > startValue) { //Jump down, need to roll up
                            phaseRow[col] = startValue;
                        }
                    }
                }
            }
        }
    }

    free(vecSignOfRow);
    free(vecAmplOfRow);
    free(vecJumpCol);
    free(vecJumpSpan);
    free(vecSortedJumpSpanIdx);
    free(vecJumpIdxNeedToHandle);
}

void run_kernel_phase_correction(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    float* phaseDiff,
    float* phase,
    char* pBufferSign,
    char* pBufferAmpl,
    int* pBufferJumpSpan,
    int* pBufferJumpStart,
    int* pBufferJumpEnd,
    int* pBufferSortedJumpSpanIdx,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int span) {
    kernel_phase_correction<<<grid, threads, 0, cudaStream>>>(phaseDiff, phase, pBufferSign, pBufferAmpl,
        pBufferJumpSpan,
        pBufferJumpStart,
        pBufferJumpEnd,
        pBufferSortedJumpSpanIdx,
        step, ROWS, COLS, span);
}

__global__
void kernel_floor(
     float *data,
     uint32_t step,
     const int ROWS,
     const int COLS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < COLS && j < ROWS) {
        auto pValue = data + j * step + i;
        *pValue = floor(*pValue);
    }
}

void run_kernel_floor(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    float *data,
    uint32_t step,
    const int ROWS,
    const int COLS) {
    kernel_floor<<<grid, threads, 0, cudaStream>>>(data, step, ROWS, COLS);
}

__global__
void kernel_interval_average(
    float *data,
    uint32_t step,
    const int ROWS,
    const int COLS,
    int interval,
    float *result) {
    float fSum = 0.f, fCount = 0.f;
    for (int row = 0; row < ROWS; row += interval) {
        for (int col = 0; col < COLS; col += interval) {
            auto pValue = data + row * step + col;
            fSum += *pValue;
            ++ fCount;
        }
    }
    *result = fSum / fCount;
}

void run_kernel_interval_average(
    cudaEvent_t& eventDone,
    cudaStream_t cudaStream,
    float *data,
    uint32_t step,
    const int ROWS,
    const int COLS,
    int interval,
    float *d_result,
    float *result) {
    kernel_interval_average<<<1, 1, 0, cudaStream>>>(data, step, ROWS, COLS, interval, d_result);
    cudaMemcpyAsync(result, d_result, sizeof(float), cudaMemcpyDeviceToHost, cudaStream);

    cudaEventRecord(eventDone, cudaStream);
    cudaEventSynchronize(eventDone);
}

//__device__ unsigned int count = 0;
//__shared__ bool isLastBlockDone;
//
//__global__
//void kernel_range_interval_average(
//    float *data,
//    uint32_t step,
//    const int ROWS,
//    const int COLS,
//    const int interval,
//    const float rangeStart,
//    const float rangeEnd,
//    float *result) {
//    extern __shared__ float sharedMemory[];
//
//    const int startX = threadIdx.x * interval;
//    const int strideX = blockDim.x * interval;
//    const int startY = threadIdx.y * interval;
//    const int strideY = blockDim.y * interval;
//    const int RESULT_COLS = COLS / interval;
//    const int size = (ROWS / interval) * (COLS / interval);
//
//    if (threadIdx.x == 4 && threadIdx.y == 4) {
//        printf("startX = %d, strideX = %d, startY = %d, strideY = %d\n", startX, strideX, startY, strideY);
//        printf("ROWS = %d, COLS = %d\n", ROWS, COLS);
//    }
//
//    for (int i = startX; i < COLS; i += strideX) {
//        for (int j = startY; j < ROWS; j += strideY) {
//            auto pValue = data + j * step + i;
//
//            int resultOffset = j / interval * RESULT_COLS + i / interval;
//            auto pResult = sharedMemory + resultOffset;
//            *pResult = *pValue;
//
//            // Thread 0 signals that it is done.
//            unsigned int value = atomicInc(&count, size);
//
//            // Thread 0 determines if its block is the last
//            // block to be done.
//            isLastBlockDone = (value == (size - 1));
//
//            if (threadIdx.x == 4 && threadIdx.y == 4)
//                printf("i = %d, j = %d, result offset %d, value %.2f\n", i, j, resultOffset, *pResult);
//        }
//    }
//    __syncthreads();
//
//    if (threadIdx.x == 0 && threadIdx.y == 0 && isLastBlockDone) {
//        printf("Before sort\n");
//
//        //kernel_sort(sharedMemory, size);
//        int start = static_cast<int>(rangeStart * size);
//        int end = static_cast<int>(rangeEnd * size);
//        printf("start = %d, end = %d\n", start, end);
//        float fSum = 0;
//        for (int k = start; k < end; ++k)
//            fSum += sharedMemory[k];
//        *result = fSum / (end - start);
//    }
//}

__global__
void kernel_interval_pick_data(
    const float *data,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int interval,
    float *result) {
    int count = 0;
    for (int row = 0; row < ROWS; row += interval) {
        for (int col = 0; col < COLS; col += interval) {
            auto pValue = data + row * step + col;
            result[count ++] = *pValue;
        }
    }
}

__global__
void kernel_range_average(float *data, const int size, float rangeStart, float rangeEnd) {
    int start = static_cast<int>(rangeStart * size);
    int end = static_cast<int>(rangeEnd * size);
    float fSum = 0;
    for (int k = start; k < end; ++k)
        fSum += data[k];
    data[0] = fSum / (end - start);
}

void run_kernel_range_interval_average(
    cudaEvent_t& eventDone,
    cudaStream_t cudaStream,
    const float *data,
    uint32_t step,
    const int ROWS,
    const int COLS,
    int interval,
    const float rangeStart,
    const float rangeEnd,
    float* d_result,
    float* result) {
    const int RESULT_ROWS = static_cast<int>(ceil((float)ROWS / interval));
    const int RESULT_COLS = static_cast<int>(ceil((float)COLS / interval));
    const int SIZE = RESULT_ROWS * RESULT_COLS;

    kernel_interval_pick_data <<<1, 1, 0, cudaStream>>>(data, step, ROWS, COLS, interval, d_result);

    kernel_qsort<<<1, 1, 0, cudaStream >>>(d_result, SIZE);
    kernel_range_average<<<1, 1, 0, cudaStream>>>(d_result, SIZE, rangeStart, rangeEnd);
    cudaMemcpyAsync(result, d_result, sizeof(float), cudaMemcpyDeviceToHost, cudaStream);

    cudaEventRecord(eventDone, cudaStream);
    cudaEventSynchronize(eventDone);
}

__global__
void kernel_phase_to_height_3d(
    const float* pZInRow, // len x 1
    const float* pPolyParamsInput, // 1 x ss
    const float* xxt, // len x ss
    const int xxtStep,
    const int len,
    const int ss,
    float* pResult// len x ss
    ) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int col = threadIdx.y;

    if (col >= ss)
        return;

    for(int row = start; row < len; row += stride) {
        float* pResultRow = pResult + row * ss;
        const float* xxtRow = xxt + row * xxtStep;
        float value = pZInRow[row];
        float ps3Power = pow(value, col);
        float ps4Power = pow(1.f - value, ss - col - 1);
        pResultRow[col] = ps3Power * ps4Power * pPolyParamsInput[col] * xxtRow[col];
    }
}

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
    )
{
    kernel_phase_to_height_3d<<<grid, threads, 0, cudaStream>>>(pZInRow, pPolyParamsInput, xxt, xxtStep, len, ss, pResult);
}

__global__
void kernel_calc_sum_and_convert_matrix(
    float* pInputData,
    const int ss,
    float* pOutput,
    const int step,
    const int ROWS,
    const int COLS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < COLS && j < ROWS) {
        int index = j * COLS + i;
        float* pSum = pOutput + j * step + i;
        int start = index * ss;
        int end = start + ss;
        for (int k = start; k < end; ++ k) {
            *pSum += pInputData[k];
        }
    }
}

void run_kernel_calc_sum_and_convert_matrix(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    float* pInputData,
    const int ss,
    float* pOutput,
    const int step,
    const int ROWS,
    const int COLS) {
    kernel_calc_sum_and_convert_matrix<<<grid, threads, 0, cudaStream>>>(pInputData, ss, pOutput, step, ROWS, COLS);
}

__device__
bool patch_up_side_value(
    float *pInOutData,
    const unsigned char* pNanMask,
    int step,
    int row,
    int col,
    const int ROWS,
    const int COLS) {
    int patchRow = row - 1;
    unsigned char maskValue = *(pNanMask + patchRow * step + col);
    while (maskValue != 0 && patchRow > 0) {
        patchRow--;
        maskValue = *(pNanMask + patchRow * step + col);
    }

    if (maskValue == 0 && patchRow >= 0) {
        *(pInOutData + row * step + col) = *(pInOutData + patchRow * step + col);
        return true;
    }
    return false;
}

__device__
bool patch_down_side_value(
    float *pInOutData,
    const unsigned char* pNanMask,
    int step,
    int row,
    int col,
    const int ROWS,
    const int COLS) {
    int patchRow = row + 1;
    unsigned char maskValue = *(pNanMask + patchRow * step + col);
    while (maskValue != 0 && patchRow < ROWS - 1) {
        patchRow++;
        maskValue = *(pNanMask + patchRow * step + col);
    }
    if (maskValue == 0 && patchRow < ROWS) {
        *(pInOutData + row * step + col) = *(pInOutData + patchRow * step + col);
        return true;
    }
    return false;
}

__device__
bool patch_left_side_value(
    float *pInOutData,
    const unsigned char* pNanMask,
    int step,
    int row,
    int col,
    const int ROWS,
    const int COLS) {
    int patchCol = col - 1;
    unsigned char maskValue = *(pNanMask + row * step + patchCol);
    while (maskValue != 0 && patchCol > 0) {
        patchCol--;
        maskValue = *(pNanMask + row * step + patchCol);
    }

    if (maskValue == 0 && patchCol >= 0) {
        *(pInOutData + row * step + col) = *(pInOutData + row * step + patchCol);
        return true;
    }
    return false;
}

__device__
bool patch_right_side_value(
    float *pInOutData,
    const unsigned char* pNanMask,
    int step,
    int row,
    int col,
    const int ROWS,
    const int COLS) {
    int patchCol = col + 1;
    unsigned char maskValue = *(pNanMask + row * step + patchCol);
    while (maskValue != 0 && patchCol < COLS - 1) {
        patchCol++;
        maskValue = *(pNanMask + row * step + patchCol);
    }

    if (maskValue == 0 && patchCol < COLS) {
        *(pInOutData + row * step + col) = *(pInOutData + row * step + patchCol);
        return true;
    }
    return false;
}

__global__
void kernel_phase_patch(float *pInOutData,
    const unsigned char* pNanMask,
    const int ROWS,
    const int COLS,
    int step,
    PR_DIRECTION enProjDir) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= COLS || row >= ROWS)
        return;

    int dataOffset = step * row;
    auto pRowOfNanMask = pNanMask + dataOffset;
    if (pRowOfNanMask[col] == 0)
        return;

    if (row == ROWS - 1 && col == COLS - 1) {
        patch_up_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
        return;
    }else if (row == 0 && col == 0) {
        patch_down_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
        return;
    }
    // Projection up to down, patch down to up
    if (PR_DIRECTION::UP == enProjDir) {
        if (row < ROWS - 1) {
            patch_down_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
        }else if (col < COLS - 1) {
            patch_right_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
        }
    }
    // Projection down to up, patch up to down
    else if (PR_DIRECTION::DOWN == enProjDir) {
        if (row > 0)
            patch_up_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
        else if (col > 0)
            patch_left_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
    }
    // Projection left to right, patch right to left
    else if (PR_DIRECTION::LEFT == enProjDir) {
        if (col < COLS - 1)
            patch_right_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
        else if (row < ROWS - 1)
            patch_down_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
    }
    // Projection right to left, patch left to right
    else if (PR_DIRECTION::RIGHT == enProjDir) {
        if (col > 0)
            patch_left_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
        else if (row > 0)
            patch_up_side_value(pInOutData, pNanMask, step, row, col, ROWS, COLS);
    }
}

void run_kernel_phase_patch(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    float *pInOutData,
    const unsigned char* pNanMask,
    const int ROWS,
    const int COLS,
    int step,
    PR_DIRECTION enProjDir) {
    kernel_phase_patch<<<grid, threads, 0, cudaStream>>>(pInOutData, pNanMask, ROWS, COLS, step, enProjDir);
}

__global__
void kernel_merge_height_intersect(
    float*         matOne,
    float*         matTwo,
    float*         matTre,
    float*         matFor,
    unsigned char* matMask,
    float*         matMaskDiffResult,
    const int      ROWS,
    const int      COLS,
    const int      step,
    int*           pMergeIndexBuffer,
    float*         pCmpTargetBuffer,
    float          fDiffThreshold) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int row = start; row < ROWS; row += stride) {
        int offset = row * step;
        auto matOneRow = matOne + offset;
        auto matTwoRow = matTwo + offset;
        auto matTreRow = matTre + offset;
        auto matForRow = matFor + offset;
        auto matNanRow = matMask + offset;
        auto matMaskDiffRow = matMaskDiffResult + offset;

        int* arrayIndexs = pMergeIndexBuffer + offset;
        float* arrayTargetH = pCmpTargetBuffer + offset;

        int count = 0;
        if (matNanRow[0] > 0)
            arrayIndexs[count++] = 0;

        for (int col = 0; col < COLS - 1; ++col) {
            if (matMaskDiffRow[col + 1] > 0)
                arrayIndexs[count++] = col;
        }

        if (matNanRow[COLS - 1] > 0)
            arrayIndexs[count++] = COLS - 1;

        for (int i = 0; i < count / 2; ++i) {
            int startIndex = arrayIndexs[i * 2];
            int endIndex = arrayIndexs[i * 2 + 1];

            if (0 == startIndex && endIndex < COLS - 1) {
                for (int i = 0; i <= endIndex - startIndex; ++i)
                    arrayTargetH[i] = matForRow[endIndex + 1];
            }
            else if (COLS - 1 == endIndex && startIndex >= 0) {
                for (int i = 0; i <= endIndex - startIndex; ++i)
                    arrayTargetH[i] = matForRow[startIndex];
            }
            else if (startIndex >= 0 && endIndex < COLS - 1) {
                float fInterval = (matForRow[endIndex + 1] - matForRow[startIndex]) / (endIndex - startIndex + 1);
                for (int i = 0; i <= endIndex - startIndex; ++i) {
                    arrayTargetH[i] = matForRow[startIndex] + fInterval * i;
                }
            }

            float fAbsDiffSumOne = 0.f, fAbsDiffSumTwo = 0.f, fAbsDiffSumTre = 0.f;
            for (int index = startIndex, k = 0; index <= endIndex; ++index, ++k) {
                fAbsDiffSumOne += fabsf(matOneRow[index] - arrayTargetH[k]);
                fAbsDiffSumTwo += fabsf(matTwoRow[index] - arrayTargetH[k]);
                fAbsDiffSumTre += fabsf(matTreRow[index] - arrayTargetH[k]);
            }

            float *matChoose;
            if (fAbsDiffSumOne <= fAbsDiffSumTwo && fAbsDiffSumOne <= fAbsDiffSumTre)
                matChoose = matOneRow;
            else if (fAbsDiffSumTwo <= fAbsDiffSumOne && fAbsDiffSumTwo <= fAbsDiffSumTre)
                matChoose = matTwoRow;
            else
                matChoose = matTreRow;

            for (int col = startIndex; col <= endIndex; ++col) {
                matForRow[col] = matChoose[col];
            }
        }
    }
}

void cpu_kernel_merge_height_intersect(
    float*         matOne,
    float*         matTwo,
    float*         matTre,
    float*         matFor,
    unsigned char* matMask,
    float*         matMaskDiffResult,
    const int      ROWS,
    const int      COLS,
    const int      step,
    float          fDiffThreshold) {
    int start = 0;
    int stride = 1;

    int *arrayIndexs = (int *)malloc(COLS / 2 * sizeof(int));
    float *arrayTargetH = (float *)malloc(COLS * sizeof(float));

    for (int row = start; row < ROWS; row += stride) {
        int offset = row * step;
        auto matOneRow = matOne + offset;
        auto matTwoRow = matTwo + offset;
        auto matTreRow = matTre + offset;
        auto matForRow = matFor + offset;
        auto matNanRow = matMask + offset;
        auto matMaskDiffRow = matMaskDiffResult + row * (step - 1);

        int count = 0;
        if (matNanRow[0] > 0)
            arrayIndexs[count++] = 0;

        for (int col = count; col < COLS - 1; ++col) {
            if (matMaskDiffRow[col] > 0)
                arrayIndexs[count++] = col;
        }

        if (matNanRow[COLS - 1] > 0)
            arrayIndexs[count++] = COLS - 1;

        printf("row = %d count = %d\n", row, count);

        for (int i = 0; i < count / 2; ++i) {
            int startIndex = arrayIndexs[i * 2];
            int endIndex = arrayIndexs[i * 2 + 1];

            if (0 == startIndex && endIndex < COLS - 1) {
                for (int i = 0; i <= endIndex - startIndex; ++i)
                    arrayTargetH[i] = matForRow[endIndex + 1];
            }
            else if (COLS - 1 == endIndex && startIndex >= 0) {
                for (int i = 0; i <= endIndex - startIndex; ++i)
                    arrayTargetH[i] = matForRow[startIndex];
            }
            else if (startIndex >= 0 && endIndex < COLS - 1) {
                float fInterval = (matForRow[endIndex + 1] - matForRow[startIndex]) / (endIndex - startIndex + 1);
                for (int i = 0; i <= endIndex - startIndex; ++i) {
                    arrayTargetH[i] = matForRow[startIndex] + fInterval * i;
                }
            }

            float fAbsDiffSumOne = 0.f, fAbsDiffSumTwo = 0.f, fAbsDiffSumTre = 0.f;
            for (int index = startIndex, k = 0; index <= endIndex; ++index, ++k) {
                fAbsDiffSumOne += fabsf(matOneRow[index] - arrayTargetH[k]);
                fAbsDiffSumTwo += fabsf(matTwoRow[index] - arrayTargetH[k]);
                fAbsDiffSumTre += fabsf(matTreRow[index] - arrayTargetH[k]);
            }

            float *matChoose;
            if (fAbsDiffSumOne <= fAbsDiffSumTwo && fAbsDiffSumOne <= fAbsDiffSumTre)
                matChoose = matOneRow;
            else if (fAbsDiffSumTwo <= fAbsDiffSumOne && fAbsDiffSumTwo <= fAbsDiffSumTre)
                matChoose = matTwoRow;
            else
                matChoose = matTreRow;

            for (int col = startIndex; col <= endIndex; ++col) {
                matForRow[col] = matChoose[col];
            }
        }
    }

    free(arrayTargetH);
    free(arrayIndexs);
}

void run_kernel_merge_height_intersect(
    uint32_t       gridSize,
    uint32_t       blockSize,
    cudaStream_t   cudaStream,
    float*         matOne,
    float*         matTwo,
    float*         matTre,
    float*         matFor,
    unsigned char* matMask,
    float*         matMaskDiffResult,
    const int      ROWS,
    const int      COLS,
    const int      step,
    int*           pMergeIndexBuffer,
    float*         pCmpTargetBuffer,
    float          fDiffThreshold) {
    kernel_merge_height_intersect<<<gridSize, blockSize, 0, cudaStream>>>(
        matOne,
        matTwo,
        matTre,
        matFor,
        matMask,
        matMaskDiffResult,
        ROWS,
        COLS,
        step,
        pMergeIndexBuffer,
        pCmpTargetBuffer,
        fDiffThreshold);
    //cpu_kernel_merge_height_intersect(
    //    matOne,
    //    matTwo,
    //    matTre,
    //    matFor,
    //    matMask,
    //    matMaskDiffResult,
    //    ROWS,
    //    COLS,
    //    step,
    //    fDiffThreshold);
}

__global__
void kernel_choose_min_value_for_mask(
    float *matH1,
    float *matH2,
    float *matH3,
    unsigned char* matMask,
    const int ROWS,
    const int COLS,
    const int step) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= COLS || row >= ROWS)
        return;

    int dataRowOffset = row * step;
    auto matMaskRow = matMask + dataRowOffset;
    if (matMaskRow[col] == 0)
        return;

    auto matH1Row = matH1 + dataRowOffset;
    auto matH2Row = matH2 + dataRowOffset;
    auto matH3Row = matH3 + dataRowOffset;

    if (matH2Row[col] < matH1Row[col])
        matH1Row[col] = matH2Row[col];
    
    if (matH3Row[col] < matH1Row[col])
        matH1Row[col] = matH3Row[col];
}

void run_kernel_choose_min_value_for_mask(
    dim3 grid,
    dim3 threads,
    cudaStream_t cudaStream,
    float *matH1,
    float *matH2,
    float *matH3,
    unsigned char* matMask,
    const int ROWS,
    const int COLS,
    const int step) {
    kernel_choose_min_value_for_mask<<<grid, threads, 0, cudaStream>>>(
        matH1,
        matH2,
        matH3,
        matMask,
        ROWS,
        COLS,
        step);
}
