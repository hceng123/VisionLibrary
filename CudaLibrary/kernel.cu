#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaFunc.h"

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