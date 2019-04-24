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
    float* dPhase,
    float* dMap,
    float* matResult,
    const int ROWS,
    const int COLS,
    const int span) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int *arrayIdx1 = (int *)malloc(COLS / 4 * sizeof(int));
    int *arrayIdx2 = (int *)malloc(COLS / 4 * sizeof(int));

    for (int row = start; row < ROWS; row += stride) {
        int offset = row * COLS;
        int offsetOfDiffResult = row * (COLS - 1);

        float *dMapRow = dMap + offsetOfDiffResult;
        float *dPhaseRow = dPhase + offsetOfDiffResult;
        float *matResultRow = matResult + offset;

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
                        matResultRow[k] = 1;
                }
            }
        }
    }
}

void run_kernel_select_cmp_point(
    uint32_t gridSize,
    uint32_t blockSize,
    float* dPhase,
    float* dMap,
    const int ROWS,
    const int COLS) {

}