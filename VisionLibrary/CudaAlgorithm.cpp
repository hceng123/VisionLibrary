#include <cuda_runtime.h>

#include "CudaAlgorithm.h"
#include "../VisionLibrary/StopWatch.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/highgui.hpp"
#include "CudaFunc.h"
#include "TimeLog.h"
#include "CalcUtils.hpp"
#include "Constants.h"

namespace AOI
{
namespace Vision
{

///*static*/ cv::cuda::Stream CudaAlgorithm::m_cudaStreams[NUM_OF_DLP];

//cv::cuda::Stream m_cudaStreams[NUM_OF_DLP];

static int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

/*static*/ bool CudaAlgorithm::initCuda() {
    size_t size = 0;
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, size * 10);
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);

    //cudaDeviceGetLimit(&size, cudaLimitStackSize);
    //cudaDeviceSetLimit(cudaLimitStackSize, size * 100);
    //cudaDeviceGetLimit(&size, cudaLimitStackSize);
    //std::cout  << "cudaLimitStackSize " << size << std::endl;

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16);

    int value = 0;
    cudaDeviceGetAttribute(&value, cudaDeviceAttr::cudaDevAttrKernelExecTimeout, 0);
    std::cout << "value of cudaDevAttrKernelExecTimeout " << value << std::endl;
    return true;
}

/*static*/ void CudaAlgorithm::floor(cv::cuda::GpuMat &matInOut) {
    dim3 threads(16, 16);
    dim3 grid(divUp(matInOut.cols, threads.x), divUp(matInOut.rows, threads.y));
    run_kernel_floor(grid, threads, 
        reinterpret_cast<float *>(matInOut.data),
        matInOut.step1(),
        matInOut.rows,
        matInOut.cols);
}

/*static*/ float CudaAlgorithm::intervalAverage(const cv::cuda::GpuMat &matInput, int interval) {
    float result = 0;
    run_kernel_interval_average( reinterpret_cast<float *>(matInput.data),
        matInput.step1(),
        matInput.rows,
        matInput.cols,
        interval,
        &result);
    return result;
}

/*static*/ float CudaAlgorithm::intervalRangeAverage(const cv::cuda::GpuMat& matInput, int interval, float rangeStart, float rangeEnd) {
    float result = 0;
    run_kernel_range_interval_average(
        reinterpret_cast<float *>(matInput.data),
        matInput.step1(),
        matInput.rows,
        matInput.cols,
        interval,
        rangeStart,
        rangeEnd,
        &result);
    return result;
}

/*static*/ void CudaAlgorithm::phaseWrapBuffer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matBuffer, float fShift/* = 0.f*/) {
    CStopWatch stopWatch;
    // matBuffer = matPhase / ONE_CYCLE + 0.5f- fShift;
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matBuffer, 0.f, 0.5f - fShift, matBuffer);
    TimeLog::GetInstance()->addTimeLog("phaseWrapBuffer Divide and add.", stopWatch.Span());

    floor(matBuffer);

    TimeLog::GetInstance()->addTimeLog("phaseWrapBuffer floor takes.", stopWatch.Span());

    //matBuffer = matPhase - matBuffer * ONE_CYCLE;
    cv::cuda::addWeighted(matPhase, 1, matBuffer, -ONE_CYCLE, 0, matBuffer);

    TimeLog::GetInstance()->addTimeLog("_phaseWrap multiply and subtract takes.", stopWatch.Span());

    float fMean = intervalAverage(matBuffer, 20);

    //matBuffer = matPhase / ONE_CYCLE + 0.5 - fShift - fMean / 2.f;
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matBuffer, 0.f, 0.5f - fShift - fMean / 2.f, matBuffer);
    floor(matBuffer);

    TimeLog::GetInstance()->addTimeLog("_phaseWrap floor takes.", stopWatch.Span());

    //matPhase = matPhase - matBuffer * ONE_CYCLE;
    cv::cuda::addWeighted(matPhase, 1, matBuffer, -ONE_CYCLE, 0, matPhase);
}

/*static*/ void CudaAlgorithm::phaseWrapByRefer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matRef) {
    CStopWatch stopWatch;
    //matRef = (matPhase - matRef) / ONE_CYCLE + 0.5;  //Use matResult to store the substract result to reduce memory allocate time.
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matRef, -1.f / ONE_CYCLE, 0.5, matRef);
    TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer Divide and add.", stopWatch.Span());

    floor(matRef);
    TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer floor takes.", stopWatch.Span());

    //matPhase = matPhase - matRef * ONE_CYCLE;
    cv::cuda::addWeighted(matPhase, 1.f, matRef, -ONE_CYCLE, 0, matPhase);
    TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer multiply and subtract takes.", stopWatch.Span());
}

/*static*/ void CudaAlgorithm::phaseWarpNoAutoBase(const cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matResult, float fShift) {
    CStopWatch stopWatch;
    //cv::Mat matResult = matPhase / ONE_CYCLE + 0.5 - fShift;
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matResult, 0, 0.5 - fShift, matResult);
    TimeLog::GetInstance()->addTimeLog("_phaseWrap Divide and add.", stopWatch.Span());

    floor(matResult);

    TimeLog::GetInstance()->addTimeLog("_phaseWrap floor takes.", stopWatch.Span());

    //matResult = matPhase - matResult * ONE_CYCLE;
    cv::cuda::addWeighted(matPhase, 1.f, matResult, -ONE_CYCLE, 0, matResult);
}

/*static*/ void CudaAlgorithm::phaseCorrectionCmp(cv::cuda::GpuMat& matPhase, const cv::cuda::GpuMat& matPhase1, cv::cuda::GpuMat& matPhaseT, int span) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    static int time = 0;
    ++ time;

    cv::cuda::GpuMat matIdx;
    cv::cuda::compare(matPhase, matPhase1, matIdx, cv::CmpTypes::CMP_NE);
    cv::cuda::GpuMat matMap(ROWS, COLS, CV_32FC1);
    matMap.setTo(1, matIdx);

    // Note even the size of Mat ix ROW * (COL - 1), but actually the background memory size is still ROW * COL
    cv::cuda::GpuMat dMap1 = CalcUtils::diff(matMap, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::cuda::GpuMat dxPhase = CalcUtils::diff(matPhase, 1, CalcUtils::DIFF_ON_X_DIR);

    cv::cuda::GpuMat idxl1(ROWS, COLS, CV_8UC1);
    idxl1.setTo(0);

    TimeLog::GetInstance()->addTimeLog("prepare data for phaseCorrectionCmp X", stopWatch.Span());

    // Select in X direction
    run_kernel_select_cmp_point(2, 256,
        reinterpret_cast<float *>(dMap1.data),
        reinterpret_cast<float *>(dxPhase.data),
        idxl1.data,
        dMap1.step1(),
        ROWS, COLS, span);

    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point X", stopWatch.Span());

    // Select in Y direction, transpose the matrix to accelerate
    cv::cuda::transpose(matMap, matMap);
    cv::cuda::transpose(matPhase, matPhaseT);

    cv::cuda::GpuMat dMap2 = CalcUtils::diff(matMap, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::cuda::GpuMat dyPhase = CalcUtils::diff(matPhaseT, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::cuda::GpuMat idxl2(COLS, ROWS, CV_8UC1);    // Note the indl2 is tranposed result, so the ROWS and COLS are changed
    idxl2.setTo(0);

    TimeLog::GetInstance()->addTimeLog("prepare data for phaseCorrectionCmp Y", stopWatch.Span());

    // Select in Y direction
    run_kernel_select_cmp_point(2, 256,
        reinterpret_cast<float *>(dMap2.data),
        reinterpret_cast<float *>(dyPhase.data),
        idxl2.data,
        dMap2.step1(),
        COLS, ROWS, span);

    cv::cuda::transpose(idxl2, idxl2);

    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point Y", stopWatch.Span());

    cv::cuda::bitwise_and(idxl1, idxl2, idxl1);
    matPhase1.copyTo(matPhase, idxl1);

    TimeLog::GetInstance()->addTimeLog("bitwise_and and copy data", stopWatch.Span());
}

/*static*/ void CudaAlgorithm::phaseCorrectionCmp(cv::Mat& matPhase, const cv::Mat& matPhase1, int span) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    static int time = 0;
    ++ time;

    cv::Mat matIdx;
    cv::compare(matPhase, matPhase1, matIdx, cv::CmpTypes::CMP_NE);
    cv::Mat matMap = cv::Mat::zeros(ROWS, COLS, CV_32FC1);
    matMap.setTo(1, matIdx);

    // Note even the size of Mat ix ROW * (COL - 1), but actually the background memory size is still ROW * COL
    cv::Mat dMap1 = CalcUtils::diff(matMap, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::Mat dxPhase = CalcUtils::diff(matPhase, 1, CalcUtils::DIFF_ON_X_DIR);

    if (time == 1) {
        CalcUtils::saveMatToCsv(dMap1, "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpDMap1Cpu.csv");
        CalcUtils::saveMatToCsv(dxPhase, "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpDXPhaseCpu.csv");
    }

    cv::Mat idxl1(ROWS, COLS, CV_8UC1);
    idxl1.setTo(0);

    TimeLog::GetInstance()->addTimeLog("prepare for phaseCorrectionCmp X", stopWatch.Span());

    // Select in X direction
    run_kernel_select_cmp_point(2, 256,
        reinterpret_cast<float *>(dMap1.data),
        reinterpret_cast<float *>(dxPhase.data),
        idxl1.data,
        dMap1.step1(),
        ROWS, COLS, span);
    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point X", stopWatch.Span());

    cv::imwrite(std::string("C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpMask1_") + std::to_string(time) + ".png", idxl1);
    int nonZeroCount1 = cv::countNonZero(idxl1);

    // Select in Y direction, transpose the matrix to accelerate
    cv::Mat matMapT;
    cv::transpose(matMap, matMapT);
    cv::Mat matPhaseT;
    cv::transpose(matPhase, matPhaseT);

    cv::Mat dMap2 = CalcUtils::diff(matMapT, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::Mat dyPhase = CalcUtils::diff(matPhaseT, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::Mat idxl2(COLS, ROWS, CV_8UC1);    // Note the indl2 is tranposed result, so the ROWS and COLS are changed
    idxl2.setTo(0);

    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point Y", stopWatch.Span());

    if (time == 1) {
        CalcUtils::saveMatToCsv(dMap2, "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpDMap2.csv");
        CalcUtils::saveMatToCsv(dyPhase, "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpDYPhase.csv");
    }

    // Select in Y direction
    run_kernel_select_cmp_point(2, 256,
        reinterpret_cast<float *>(dMap2.data),
        reinterpret_cast<float *>(dyPhase.data),
        idxl2.data,
        dMap2.step1(),
        COLS, ROWS, span);

    cv::imwrite(std::string("C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpMask2_") + std::to_string(time) + ".png", idxl2);
    int nonZeroCount2 = cv::countNonZero(idxl2);

    cv::transpose(idxl2, idxl2);

    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point Y", stopWatch.Span());

    cv::bitwise_and(idxl1, idxl2, idxl1);
    int nonZeroFinal = cv::countNonZero(idxl1);

    std::cout << "phaseCorrectionCmp point count1 " << nonZeroCount1 << " count2 " << nonZeroCount2 << " final cout " << nonZeroFinal << std::endl;

    matPhase1.copyTo(matPhase, idxl1);
}

/*static*/ void CudaAlgorithm::phaseCorrection(cv::cuda::GpuMat &matPhase, cv::cuda::GpuMat& matPhaseT, int nJumpSpanX, int nJumpSpanY) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    
    //X direction
    if (nJumpSpanX > 0) {
        cv::cuda::GpuMat matPhaseDiff = CalcUtils::diff(matPhase, 1, CalcUtils::DIFF_ON_X_DIR);
        run_kernel_phase_correction(2, 1024,
            reinterpret_cast<float *>(matPhaseDiff.data),
            reinterpret_cast<float *>(matPhase.data),
            matPhaseDiff.step1(),
            ROWS, COLS, nJumpSpanX);
        cudaDeviceSynchronize();
        TimeLog::GetInstance()->addTimeLog("run_kernel_phase_correction for X", stopWatch.Span());
    }

    // Y direction
    if (nJumpSpanY > 0) {
        cv::cuda::transpose(matPhase, matPhaseT);
        TimeLog::GetInstance()->addTimeLog("cv::cuda::transpose for Y", stopWatch.Span());
        cv::cuda::GpuMat matPhaseDiff = CalcUtils::diff(matPhaseT, 1, CalcUtils::DIFF_ON_X_DIR);
        run_kernel_phase_correction(2, 1024,
            reinterpret_cast<float *>(matPhaseDiff.data),
            reinterpret_cast<float *>(matPhaseT.data),
            matPhaseDiff.step1(),
            COLS, ROWS, nJumpSpanY);
        cudaDeviceSynchronize();
        TimeLog::GetInstance()->addTimeLog("run_kernel_phase_correction for Y", stopWatch.Span());

        cv::cuda::transpose(matPhaseT, matPhase);
        TimeLog::GetInstance()->addTimeLog("cv::cuda::transpose back", stopWatch.Span());
    }
}

///*static*/ void CudaAlgorithm::phaseCorrection(cv::Mat &matPhase, int nJumpSpanX, int nJumpSpanY) {
//    CStopWatch stopWatch;
//    const int ROWS = matPhase.rows;
//    const int COLS = matPhase.cols;
//    
//    static int time = 0;
//    time ++;
//
//    //X direction
//    if (nJumpSpanX > 0) {
//        cv::Mat matPhaseClone = matPhase.clone();
//
//        cv::Mat matPhaseDiff = CalcUtils::diff(matPhase, 1, CalcUtils::DIFF_ON_X_DIR);
//        std::cout << "matPhaseDiff step " << matPhaseDiff.step1() << ", matPhase step " << matPhase.step1() << std::endl;
//        run_kernel_phase_correction(2, 256,
//            reinterpret_cast<float *>(matPhaseDiff.data),
//            reinterpret_cast<float *>(matPhase.data),
//            matPhaseDiff.step1(),
//            ROWS, COLS, nJumpSpanX);
//        TimeLog::GetInstance()->addTimeLog("phaseCorrection in x direction.", stopWatch.Span());
//
//        if (time == 1) {
//            cv::Mat matDiff;
//            cv::absdiff(matPhase, matPhaseClone, matDiff);
//            cv::Mat matMask = matDiff > 0.01f;
//            cv::imwrite("C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/CompareXPhaseCorrectionResultCpu1.png", matMask);
//        }
//    }
//
//    // Y direction
//    if (nJumpSpanY > 0) {
//        cv::Mat matPhaseClone = matPhase.clone();
//
//        cv::transpose(matPhase, matPhase);
//        cv::Mat matPhaseDiff = CalcUtils::diff(matPhase, 1, CalcUtils::DIFF_ON_X_DIR);
//        std::cout << "matPhaseDiff step " << matPhaseDiff.step1() << ", matPhase step " << matPhase.step1() << std::endl;
//        run_kernel_phase_correction(2, 256,
//            reinterpret_cast<float *>(matPhaseDiff.data),
//            reinterpret_cast<float *>(matPhase.data),
//            matPhaseDiff.step1(),
//            COLS, ROWS, nJumpSpanY);
//        cv::transpose(matPhase, matPhase);
//        TimeLog::GetInstance()->addTimeLog("phaseCorrection in y direction.", stopWatch.Span());
//
//        if (time == 1) {
//            cv::Mat matDiff;
//            cv::absdiff(matPhase, matPhaseClone, matDiff);
//            cv::Mat matMask = matDiff > 0.01f;
//            cv::imwrite("C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/CompareYPhaseCorrectionResultCpu1.png", matMask);
//        }
//    }
//}

/*static*/ void CudaAlgorithm::phaseCorrection(cv::Mat &matPhase, int nJumpSpanX, int nJumpSpanY) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;

    const int DEBUG_ROW = 1588;

    static int time = 0;
    time++;

    //X direction
    if (nJumpSpanX > 0) {
        cv::Mat matPhaseClone = matPhase.clone();

        cv::Mat matPhaseDiff = CalcUtils::diff(matPhase, 1, 2);
        std::vector<char> vecSignOfRow(COLS, 0);
        std::vector<char> vecAmplOfRow(COLS, 0);
#ifdef _DEBUG
        //CalcUtils::saveMatToCsv ( matPhase, "./data/HaoYu_20171114/test1/NewLens2/BeforePhaseCorrectionX.csv");
        auto vecVecPhase = CalcUtils::matToVector<float>(matPhase);
        auto vecVecPhaseDiff = CalcUtils::matToVector<float>(matPhaseDiff);
#endif
        std::vector<int> vecRowsWithJump;
        for (int row = 0; row < matPhaseDiff.rows; ++ row) {
            bool bRowWithPosJump = false, bRowWithNegJump = false;
            std::fill(vecSignOfRow.begin(), vecSignOfRow.end(), 0);
            std::fill(vecAmplOfRow.begin(), vecAmplOfRow.end(), 0);

            for (int col = 0; col < matPhaseDiff.cols; ++col) {
                auto value = matPhaseDiff.at<float>(row, col);
                if (value > ONE_HALF_CYCLE) {
                    vecSignOfRow[col] = 1;
                    bRowWithPosJump = true;

                    char nJumpAmplitude = static_cast<char> (std::ceil(std::abs(value) / 2.f) * 2);
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
                std::vector<int> vecJumpCol, vecJumpSpan, vecJumpIdxNeedToHandle;
                vecJumpCol.reserve(20);
                for (int col = 0; col < COLS; ++col) {
                    if (vecSignOfRow[col] != 0)
                        vecJumpCol.push_back(col);
                }
                if (vecJumpCol.size() < 2)
                    continue;
                vecJumpSpan.reserve(vecJumpCol.size() - 1);
                for (size_t i = 1; i < vecJumpCol.size(); ++ i)
                    vecJumpSpan.push_back(vecJumpCol[i] - vecJumpCol[i - 1]);
                auto vecSortedJumpSpanIdx = CalcUtils::sort_index_value<int>(vecJumpSpan);
                for (int i = 0; i < ToInt32(vecJumpSpan.size()); ++i) {
                    if (vecJumpSpan[i] < nJumpSpanX)
                        vecJumpIdxNeedToHandle.push_back(i);
                }

                if (DEBUG_ROW == row) {
                    printf("jumpColCount = %d, jumpSpanCount = %d, jumpIdxNeedToHandleCount = %d\n", vecJumpCol.size(), vecJumpSpan.size(), vecJumpIdxNeedToHandle.size());
                    for (size_t jj = 0; jj < vecJumpIdxNeedToHandle.size(); ++jj) {
                        auto nStart = vecJumpCol[vecSortedJumpSpanIdx[jj]];
                        auto nEnd = vecJumpCol[vecSortedJumpSpanIdx[jj] + 1];
                        printf("%d %d\n", nStart, nEnd);
                    }
                }

                for (size_t jj = 0; jj < vecJumpIdxNeedToHandle.size(); ++ jj) {
                    auto nStart = vecJumpCol[vecSortedJumpSpanIdx[jj]];
                    auto nEnd = vecJumpCol[vecSortedJumpSpanIdx[jj] + 1];
                    char chSignFirst = vecSignOfRow[nStart];        //The index is hard to understand. Use the sorted span index to find the original column.
                    char chSignSecond = vecSignOfRow[nEnd];

                    char chAmplFirst = vecAmplOfRow[nStart];
                    char chAmplSecond = vecAmplOfRow[nEnd];
                    char chTurnAmpl = std::min(chAmplFirst, chAmplSecond) / 2;
                    if (chSignFirst * chSignSecond == -1) { //it is a pair
                        char chAmplNew = chAmplFirst - 2 * chTurnAmpl;
                        vecAmplOfRow[nStart] = chAmplNew;
                        if (chAmplNew <= 0)
                            vecSignOfRow[nStart] = 0;  // Remove the sign of jump flag.

                        chAmplNew = chAmplSecond - 2 * chTurnAmpl;
                        vecAmplOfRow[nEnd] = chAmplNew;
                        if (chAmplNew <= 0)
                            vecSignOfRow[nEnd] = 0;

                        auto startValue = matPhase.at<float>(row, nStart);
                        for (int col = nStart + 1; col <= nEnd; ++ col) {
                            auto &value = matPhase.at<float>(row, col);
                            value -= chSignFirst * ONE_CYCLE * chTurnAmpl;
                            if (chSignFirst > 0 && value < startValue) { //Jump up, need to roll down, but can not over roll
                                value = startValue;
                            }
                            else if (chSignFirst < 0 && value > startValue) { //Jump down, need to roll up
                                value = startValue;
                            }
                            else {
                                auto i = value; //Just for debug, no usage.
                            }
                        }
                    }
                }
            }
        }

        TimeLog::GetInstance()->addTimeLog("phaseCorrection in x direction.", stopWatch.Span());

        if (time == 1) {
            cv::Mat matDiff;
            cv::absdiff(matPhase, matPhaseClone, matDiff);
            cv::Mat matMask = matDiff > 0.01f;
            cv::imwrite("C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/CompareXPhaseCorrectionResultCpu.png", matMask);
        }
    }

    // Y direction
    if (nJumpSpanY > 0) {
        cv::Mat matPhaseClone = matPhase.clone();

        cv::Mat matPhaseDiff = CalcUtils::diff(matPhase, 1, 1);
#ifdef _DEBUG
        //CalcUtils::saveMatToCsv ( matPhase, "./data/HaoYu_20171114/test5/NewLens2/BeforePhaseCorrectionY.csv");
        auto vecVecPhase = CalcUtils::matToVector<float>(matPhase);
        auto vecVecPhaseDiff = CalcUtils::matToVector<float>(matPhaseDiff);
#endif
        cv::Mat matDiffSign = cv::Mat::zeros(ROWS, COLS, CV_8SC1);
        cv::Mat matDiffAmpl = cv::Mat::zeros(ROWS, COLS, CV_8SC1);
        std::vector<int> vecColsWithJump;
        for (int col = 0; col < matPhaseDiff.cols; ++ col) {
            bool bColWithPosJump = false, bColWithNegJump = false;
            for (int row = 0; row < matPhaseDiff.rows; ++row) {
                auto value = matPhaseDiff.at<float>(row, col);
                if (value > ONE_HALF_CYCLE) {
                    matDiffSign.at<char>(row, col) = 1;
                    bColWithPosJump = true;
                }
                else if (value < -ONE_HALF_CYCLE) {
                    matDiffSign.at<char>(row, col) = -1;
                    bColWithNegJump = true;
                }

                if (std::abs(value) > ONE_HALF_CYCLE) {
                    char nJumpAmplitude = static_cast<char> (std::ceil(std::abs(value) / 2.f) * 2);
                    matDiffAmpl.at<char>(row, col) = nJumpAmplitude;
                }
            }

            if (!bColWithPosJump && !bColWithNegJump)
                continue;

            cv::Mat matSignOfCol = cv::Mat(matDiffSign, cv::Range::all(), cv::Range(col, col + 1)).clone();
            char *ptrSignOfCol = matSignOfCol.ptr<char>(0);
            cv::Mat matAmplOfCol = cv::Mat(matDiffAmpl, cv::Range::all(), cv::Range(col, col + 1)).clone();
            char *ptrAmplOfCol = matAmplOfCol.ptr<char>(0);

            for (int kk = 0; kk < 2; ++ kk) {
                std::vector<int> vecJumpRow, vecJumpSpan, vecJumpIdxNeedToHandle;
                vecJumpRow.reserve(20);
                for (int row = 0; row < ROWS; ++row) {
                    if (ptrSignOfCol[row] != 0)
                        vecJumpRow.push_back(row);
                }
                if (vecJumpRow.size() < 2)
                    continue;
                vecJumpSpan.reserve(vecJumpRow.size() - 1);
                for (size_t i = 1; i < vecJumpRow.size(); ++ i)
                    vecJumpSpan.push_back(vecJumpRow[i] - vecJumpRow[i - 1]);
                auto vecSortedJumpSpanIdx = CalcUtils::sort_index_value<int>(vecJumpSpan);
                for (int i = 0; i < ToInt32(vecJumpSpan.size()); ++i) {
                    if (vecJumpSpan[i] < nJumpSpanY)
                        vecJumpIdxNeedToHandle.push_back(i);
                }

                for (size_t jj = 0; jj < vecJumpIdxNeedToHandle.size(); ++ jj) {
                    auto nStart = vecJumpRow[vecSortedJumpSpanIdx[jj]];
                    auto nEnd = vecJumpRow[vecSortedJumpSpanIdx[jj] + 1];
                    char chSignFirst = ptrSignOfCol[nStart];        //The index is hard to understand. Use the sorted span index to find the original column.
                    char chSignSecond = ptrSignOfCol[nEnd];

                    char chAmplFirst = ptrAmplOfCol[nStart];
                    char chAmplSecond = ptrAmplOfCol[nEnd];
                    char chTurnAmpl = std::min(chAmplFirst, chAmplSecond) / 2;
                    if (chSignFirst * chSignSecond == -1) { //it is a pair
                        char chAmplNew = chAmplFirst - 2 * chTurnAmpl;
                        ptrAmplOfCol[nStart] = chAmplNew;
                        if (chAmplNew <= 0)
                            ptrSignOfCol[nStart] = 0;  // Remove the sign of jump flag.

                        chAmplNew = chAmplSecond - 2 * chTurnAmpl;
                        ptrAmplOfCol[nEnd] = chAmplNew;
                        if (chAmplNew <= 0)
                            ptrSignOfCol[nEnd] = 0;

                        auto startValue = matPhase.at<float>(nStart, col);
                        for (int row = nStart + 1; row <= nEnd; ++row) {
                            auto &value = matPhase.at<float>(row, col);
                            value -= chSignFirst * ONE_CYCLE * chTurnAmpl;
                            if (chSignFirst > 0 && value < startValue) { //Jump up, need to roll down, but can not over roll
                                value = startValue;
                            }
                            else if (chSignFirst < 0 && value > startValue) { //Jump down, need to roll up
                                value = startValue;
                            }
                            else {
                                auto i = value; //Just for debug, no usage.
                            }
                        }
                    }
                }
            }
        }
        TimeLog::GetInstance()->addTimeLog("phaseCorrection in y direction.", stopWatch.Span());

         if (time == 1) {
            cv::Mat matDiff;
            cv::absdiff(matPhase, matPhaseClone, matDiff);
            cv::Mat matMask = matDiff > 0.01f;
            cv::imwrite("C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/CompareYPhaseCorrectionResultCpu.png", matMask);
        }
    }
}

/*static*/ cv::Mat CudaAlgorithm::mergeHeightIntersect(cv::Mat matHeightOne, cv::Mat matNanMaskOne, cv::Mat matHeightTwo, cv::Mat matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir) {
    CStopWatch stopWatch;
    auto start = stopWatch.AbsNowInMicro();

    matHeightTwo.copyTo(matHeightOne, matNanMaskOne);
    matHeightOne.copyTo(matHeightTwo, matNanMaskTwo);

    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
        cv::transpose(matHeightOne, matHeightOne);
        cv::transpose(matHeightTwo, matHeightTwo);

        TimeLog::GetInstance()->addTimeLog("_mergeHeightIntersect. Transpose image.", stopWatch.Span());
    }

    //#ifdef _DEBUG
    //    auto vecVecHeightOne = matToVector<float>(matHeightOne);
    //    auto vecVecHeightTwo = matToVector<float>(matHeightTwo);
    //    auto vecVecHeightTre = matToVector<float>(matHeightTre);
    //    auto vecVecHeightFor = matToVector<float>(matHeightFor);
    //#endif

    std::cout << "Prepare data " << stopWatch.SpanInMicro() << std::endl;

    //matHeightOne = matHeightOne.reshape(1, 1);
    //matHeightTwo = matHeightTwo.reshape(1, 1);
    //matHeightTre = matHeightTre.reshape(1, 1);
    //matHeightFor = matHeightFor.reshape(1, 1);
    cv::cuda::GpuMat matHeightOneGpu, matHeightTwoGpu;
    //float *d_result = NULL;
    //cudaMalloc(&d_result, matHeightOne.step * matHeightOne.rows);
    //auto size = matHeightOne.step * matHeightOne.rows;

    //float *d_One = NULL, *d_Two = NULL, *d_Tre = NULL, *d_For = NULL, *d_Mask = NULL;

    //int status = cudaHostRegister(matHeightOne.data, size, cudaHostRegisterPortable);
    cv::cuda::registerPageLocked(matHeightOne);
    std::cout << "Register matrix one " << stopWatch.SpanInMicro() << std::endl;

    matHeightOneGpu.upload(matHeightOne);
    //printf("Copy input data from the host memory to the CUDA device\n");
    //err = cudaMemcpy(d_One, matHeightOne.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector one from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}
    std::cout << "Upload matrix one " << stopWatch.SpanInMicro() << std::endl;

    cv::cuda::registerPageLocked(matHeightTwo);
    //cudaHostRegister(matHeightTwo.data, matHeightTwo.step * matHeightTwo.rows, cudaHostRegisterPortable);
    std::cout << "Register matrix two " << stopWatch.SpanInMicro() << std::endl;

    matHeightTwoGpu.upload(matHeightTwo);
    //err = cudaMemcpy(d_Two, matHeightTwo.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector two from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}
    std::cout << "Upload matrix two " << stopWatch.SpanInMicro() << std::endl;

    //cv::cuda::registerPageLocked(matHeightTre); matHeightTreGpu.upload(matHeightTre);
    //cv::cuda::registerPageLocked(matNanMask);   matNanMaskGpu.upload(matNanMask);

    //cv::cuda::registerPageLocked(matHeightFor); matHeightForGpu.upload(matHeightFor);

    //err = cudaMemcpy(d_Tre, matHeightTre.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector three from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}

    //err = cudaMemcpy(d_For, matHeightTre.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector four from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}

    //err = cudaMemcpy(d_Mask, matNanMask.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector four from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}
    // matHeightTwoGpu.upload(matHeightTwo);
    // matHeightTreGpu.upload(matHeightTre);
    // matHeightForGpu.upload(matHeightFor);
    //matNanMaskGpu.upload(matNanMask);

    cv::cuda::GpuMat matHeightForGpu(matHeightOne.size(), matHeightOne.type());

    std::cout << "Upload data " << stopWatch.SpanInMicro() << std::endl;
    std::cout << "rows " << matHeightOne.rows << " cols " << matHeightOne.cols << std::endl;
    int gnBlockNumber = 2;
    int gnThreadNumber = 256;
    run_kernel_merge_height_intersect(
        gnBlockNumber,
        gnThreadNumber,
        reinterpret_cast<float *>(matHeightOneGpu.data),
        reinterpret_cast<float *>(matHeightTwoGpu.data),
        reinterpret_cast<float *>(matHeightForGpu.data),
        matHeightOne.rows,
        matHeightOne.cols,
        fDiffThreshold);

    std::cout << "kernel_merge_height " << stopWatch.SpanInMicro() << std::endl;

    cv::Mat matResult;
    matHeightForGpu.download(matResult);
    //err = cudaMemcpy(matHeightFor.data, d_For, size, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector four from device to host (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}

    std::cout << "Download data " << stopWatch.SpanInMicro() << std::endl;

    cv::cuda::unregisterPageLocked(matHeightOne);
    cv::cuda::unregisterPageLocked(matHeightTwo);

    //if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
    //    cv::transpose(matHeightFor, matHeightFor);
    //    TimeLog::GetInstance()->addTimeLog("_mergeHeightIntersect. Transpose image back.", stopWatch.Span());
    //}

    auto total = stopWatch.AbsNowInMicro() - start;
    std::cout << "Total time: " << total << std::endl;

    return matResult;
}

}
}