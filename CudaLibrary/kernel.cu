#include <stdio.h>

#include "device_launch_parameters.h"
#include "CudaFunc.h"

#define MAX_DEPTH       16
#define INSERTION_SORT  32

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
__global__ void cdp_simple_quicksort(T *data, int left, int right, int depth)
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
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
template<typename T>
void run_qsort(T *data, unsigned int nitems)
{
    // Launch on device
    int left = 0;
    int right = nitems-1;
    cdp_simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
}

__global__
void kernel_merge_height_intersect(float* matOne, float *matTwo, float *d_result, int rows, int cols, float fDiffThreshold) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int *arrayIndexs = (int *)malloc(cols / 2 * sizeof(int));
    //assert(arrayIndexs != NULL);

    float *arrayTargetH = (float *)malloc(cols / 2 * sizeof(float));
    //assert(arrayTargetH != NULL);

    float *matTreRow = (float *)malloc(cols * sizeof(float));
    //assert(matTreRow != NULL);

    //float *matForRow = (float *)malloc(cols * sizeof(float));
    //assert(matForRow != NULL);

    char *matNanMaskRow = (char *)malloc(cols);
    //assert(matNanMaskRow != NULL);

    memset(matNanMaskRow, 0, cols);

    for (int row = start; row < rows; row += stride) {
        int offset = row * cols;
        float *matOneRow = matOne + offset;
        float *matTwoRow = matTwo + offset;
        float *matForRow = d_result + offset;

        for (int i = 0; i < cols; ++i) {
            matTreRow[i] = (matOneRow[i] + matTwoRow[i]) / 2;
            matForRow[i] = matTreRow[i];
            float absDiff = fabs(matOneRow[i] - matTwoRow[i]);
            if (isnan(absDiff) || absDiff > fDiffThreshold)
                matNanMaskRow[i] = 1;
        }

        //memcpy(matForRow, matTreRow, cols * sizeof(float));

        int count = 0;
        if (matNanMaskRow[0] > 0)
            arrayIndexs[count++] = 0;

        for (int col = 0; col < cols - 1; ++col) {
            if (fabsf(matNanMaskRow[col + 1] - matNanMaskRow[col]) > 0)
                arrayIndexs[count++] = col;
        }

        if (matNanMaskRow[cols - 1] > 0)
            arrayIndexs[count++] = cols - 1;

        for (int i = 0; i < count / 2; ++i) {
            int startIndex = arrayIndexs[i * 2];
            int endIndex = arrayIndexs[i * 2 + 1];

            if (0 == startIndex && endIndex < cols - 1) {
                for (int i = 0; i < endIndex - startIndex; ++i)
                    arrayTargetH[i] = matForRow[endIndex + 1];
            }
            else if (cols - 1 == endIndex && startIndex >= 0) {
                for (int i = 0; i < endIndex - startIndex; ++i)
                    arrayTargetH[i] = matForRow[startIndex];
            }
            else if (startIndex >= 0 && endIndex < cols - 1) {
                float fInterval = (matForRow[endIndex + 1] - matForRow[startIndex]) / (endIndex - startIndex + 1);
                for (int i = 0; i < endIndex - startIndex; ++i) {
                    arrayTargetH[i] = matForRow[startIndex] + fInterval * i;
                }
            }

            float fAbsDiffSumOne = 0.f, fAbsDiffSumTwo = 0.f, fAbsDiffSumTre = 0.f;
            for (int index = startIndex, k = 0; index < endIndex; ++index, ++k) {
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

        //memcpy(matOneRow, matForRow, cols * sizeof(float));
    }

    free(arrayTargetH);
    free(arrayIndexs);
    free(matTreRow);
    //free(matForRow);
    free(matNanMaskRow);
}

void run_kernel_merge_height_intersect(
    uint32_t gridSize,
    uint32_t blockSize,
    float* matOne,
    float *matTwo,
    float *d_result,
    int rows,
    int cols,
    float fDiffThreshold) {
    kernel_merge_height_intersect<<<gridSize, blockSize>>>(matOne, matTwo, d_result, rows, cols, fDiffThreshold);
}

__global__
void kernel_select_cmp_point(
    float* dMap,
    float* dPhase,
    uint8_t* matResult,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int span) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int *arrayIdx1 = (int *)malloc(COLS / 4 * sizeof(int));
    int *arrayIdx2 = (int *)malloc(COLS / 4 * sizeof(int));

    for (int row = start; row < ROWS; row += stride) {
        int offsetOfInput = row * step;
        int offsetOfResult = row * step;

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
                    for (int k = arrayIdx1[i]; k <= arrayIdx2[i]; ++k) {
                        matResultRow[k] = 255;
                    }

                }
            }
        }
    }

    free(arrayIdx1);
    free(arrayIdx2);
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
    float* dMap,
    float* dPhase,
    uint8_t* matResult,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int span) {
    kernel_select_cmp_point<<<gridSize, blockSize>>>(dMap, dPhase, matResult, step, ROWS, COLS, span);
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
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int span) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float ONE_HALF_CYCLE = 1.f;
    const float ONE_CYCLE = 2.f;

    char* vecSignOfRow = (char*)malloc(COLS * sizeof(char));
    char* vecAmplOfRow = (char*)malloc(COLS * sizeof(char));
    int* vecJumpCol = (int*)malloc(COLS / 4 * sizeof(int));
    int* vecJumpSpan = (int*)malloc(COLS / 4 * sizeof(int));
    int* vecSortedJumpSpanIdx = (int*)malloc(COLS / 4 * sizeof(int));
    int* vecJumpIdxNeedToHandle = (int*)malloc(COLS / 4 * sizeof(int));

    for (int row = start; row < ROWS; row += stride) {
        int offsetOfData = row * step;

        bool bRowWithPosJump = false, bRowWithNegJump = false;
        memset(vecSignOfRow, 0, COLS);
        memset(vecAmplOfRow, 0, COLS);

        float* phaseDiffRow = phaseDiff + offsetOfData;
        float* phaseRow = phase + offsetOfData;

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
            kernel_sort_index_value(vecJumpSpan, jumpSpanCount, vecSortedJumpSpanIdx);
            for (int i = 0; i < jumpSpanCount; ++i) {
                if (vecJumpSpan[i] < span)
                    vecJumpIdxNeedToHandle[jumpIdxNeedToHandleCount++] = i;
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
    uint32_t gridSize,
    uint32_t blockSize,
    float* phaseDiff,
    float* phase,
    uint32_t step,
    const int ROWS,
    const int COLS,
    const int span) {
    kernel_phase_correction<<<gridSize, blockSize>>>(phaseDiff, phase, step, ROWS, COLS, span);
    //cpu_kernel_phase_correction(phaseDiff, phase, step, ROWS, COLS, span);
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
    float *data,
    uint32_t step,
    const int ROWS,
    const int COLS) {
    kernel_floor<<<grid, threads >>>(data, step, ROWS, COLS);
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
    float *data,
    uint32_t step,
    const int ROWS,
    const int COLS,
    int interval,
    float *result) {

    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    kernel_interval_average <<<1, 1 >>>(data, step, ROWS, COLS, interval, d_result);

    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
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
    const float *data,
    uint32_t step,
    const int ROWS,
    const int COLS,
    int interval,
    const float rangeStart,
    const float rangeEnd,
    float *result) {

    float* d_result;
    const int RESULT_ROWS = static_cast<int>(ceil((float)ROWS / interval));
    const int RESULT_COLS = static_cast<int>(ceil((float)COLS / interval));
    const int SIZE = RESULT_ROWS * RESULT_COLS;

    cudaMalloc(&d_result, SIZE * sizeof(float));
    kernel_interval_pick_data <<<1, 1>>>(data, step, ROWS, COLS, interval, d_result);

    run_qsort(d_result, SIZE);
    kernel_range_average<<<1, 1>>>(d_result, SIZE, rangeStart, rangeEnd);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}
