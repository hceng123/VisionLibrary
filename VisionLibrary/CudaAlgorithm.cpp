#include <cuda_runtime.h>

#include "CudaAlgorithm.h"
#include "../VisionLibrary/StopWatch.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
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

/*static*/ DlpCalibResult CudaAlgorithm::m_dlpCalibData[NUM_OF_DLP];
/*static*/ Calc3DHeightVars CudaAlgorithm::m_arrCalc3DHeightVars[NUM_OF_DLP];

static int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

/*static*/ bool CudaAlgorithm::initCuda() {
    size_t size = 0;
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, size * 40);
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16);
    return true;
}

/*static*/ VisionStatus CudaAlgorithm::setDlpParams(const PR_SET_DLP_PARAMS_TO_GPU_CMD* const pstCmd, PR_SET_DLP_PARAMS_TO_GPU_RPY* const pstRpy) {
    pstRpy->enStatus = VisionStatus::OK;

    const size_t MEM_SIZE = pstCmd->vec3DBezierSurface[0].total() * sizeof(float);
    float* h_buffer = (float*)malloc(MEM_SIZE);
    const int ROWS = pstCmd->vec3DBezierSurface[0].rows;
    const int COLS = pstCmd->vec3DBezierSurface[0].cols;

    for (int dlp = 0; dlp < NUM_OF_DLP; ++dlp) {
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&m_dlpCalibData[dlp].pDlp3DBezierSurface), MEM_SIZE);
        if (err != cudaSuccess) {
            pstRpy->enStatus = VisionStatus::CUDA_MEMORY_ERROR;
            return pstRpy->enStatus;
        }

        for (int row = 0; row < ROWS; ++ row) {
            for (int col = 0; col < COLS; ++ col) {
                h_buffer[row * COLS + col] = pstCmd->vec3DBezierSurface[dlp].at<float>(row, col);
            }
        }

        cudaMemcpy(m_dlpCalibData[dlp].pDlp3DBezierSurface, h_buffer, MEM_SIZE, cudaMemcpyHostToDevice);

        m_dlpCalibData[dlp].matAlphaBase.upload(pstCmd->vecMatAlphaBase[dlp]);
        m_dlpCalibData[dlp].matBetaBase.upload(pstCmd->vecMatBetaBase[dlp]);
        m_dlpCalibData[dlp].matGammaBase.upload(pstCmd->vecMatGammaBase[dlp]);

        err = cudaMalloc(reinterpret_cast<void **>(&m_arrCalc3DHeightVars[dlp].d_p3), MEM_SIZE);
        if (err != cudaSuccess) {
            pstRpy->enStatus = VisionStatus::CUDA_MEMORY_ERROR;
            return pstRpy->enStatus;
        }

        err = cudaMalloc(reinterpret_cast<void **>(&m_arrCalc3DHeightVars[dlp].d_tmpResult), sizeof(float));
        if (err != cudaSuccess) {
            pstRpy->enStatus = VisionStatus::CUDA_MEMORY_ERROR;
            return pstRpy->enStatus;
        }

        m_arrCalc3DHeightVars[dlp].matAlpha = cv::cuda::GpuMat(2048, 2040, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matBeta = cv::cuda::GpuMat(2048, 2040, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matGamma = cv::cuda::GpuMat(2048, 2040, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matGamma1 = cv::cuda::GpuMat(2048, 2040, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matAvgUnderTolIndex = cv::cuda::GpuMat(2048, 2040, CV_8UC1);
        m_arrCalc3DHeightVars[dlp].matBufferGpu = cv::cuda::GpuMat(2048, 2040, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matBufferGpuT = cv::cuda::GpuMat(2040, 2048, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matMaskGpu = cv::cuda::GpuMat(2048, 2040, CV_8UC1);
        m_arrCalc3DHeightVars[dlp].matMaskGpuT = cv::cuda::GpuMat(2040, 2048, CV_8UC1);
        m_arrCalc3DHeightVars[dlp].matDiffResult = cv::cuda::GpuMat(2048, 2040, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matDiffResultT = cv::cuda::GpuMat(2040, 2048, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matDiffResult_1 = cv::cuda::GpuMat(2048, 2040, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matDiffResultT_1 = cv::cuda::GpuMat(2040, 2048, CV_32FC1);
    }
    free(h_buffer);

    return pstRpy->enStatus;
}

/*static*/ void CudaAlgorithm::floor(cv::cuda::GpuMat &matInOut, cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    dim3 threads(16, 16);
    dim3 grid(divUp(matInOut.cols, threads.x), divUp(matInOut.rows, threads.y));
    run_kernel_floor(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float *>(matInOut.data),
        matInOut.step1(),
        matInOut.rows,
        matInOut.cols);
}

/*static*/ void CudaAlgorithm::calcPhase(
    const cv::cuda::GpuMat& matInput0,
        const cv::cuda::GpuMat& matInput1,
        const cv::cuda::GpuMat& matInput2,
        const cv::cuda::GpuMat& matInput3,
        cv::cuda::GpuMat& matPhase,
        cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    dim3 threads(16, 16);
    dim3 grid(divUp(matInput0.cols, threads.x), divUp(matInput0.rows, threads.y));
    run_kernel_calc_phase(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        matInput0.data,
        matInput1.data,
        matInput2.data,
        matInput3.data,
        reinterpret_cast<float *>(matPhase.data),
        matInput0.rows,
        matInput0.cols,
        matInput0.step1());
}

/*static*/ void CudaAlgorithm::calcPhaseAndMask(
    const cv::cuda::GpuMat& matInput0,
    const cv::cuda::GpuMat& matInput1,
    const cv::cuda::GpuMat& matInput2,
    const cv::cuda::GpuMat& matInput3,
    cv::cuda::GpuMat& matPhase,
    cv::cuda::GpuMat& matMask,
    float fMinimumAlpitudeSquare,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    dim3 threads(16, 16);
    dim3 grid(divUp(matInput0.cols, threads.x), divUp(matInput0.rows, threads.y));
    run_kernel_calc_phase_and_dark_mask(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        matInput0.data,
        matInput1.data,
        matInput2.data,
        matInput3.data,
        reinterpret_cast<float *>(matPhase.data),
        matMask.data,
        fMinimumAlpitudeSquare,
        matInput0.rows,
        matInput0.cols,
        matInput0.step1());
}

/*static*/ cv::cuda::GpuMat CudaAlgorithm::diff(const cv::cuda::GpuMat& matInput, int nRecersiveTime, int nDimension,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    assert(CalcUtils::DIFF_ON_X_DIR == nDimension || CalcUtils::DIFF_ON_Y_DIR == nDimension);
    if (nRecersiveTime > 1)
        return diff(diff(matInput, nRecersiveTime - 1, nDimension), 1, nDimension);

    cv::Mat matKernel;
    if (CalcUtils::DIFF_ON_X_DIR == nDimension)
        matKernel = (cv::Mat_<float>(1, 2) << -1, 1);
    else if (CalcUtils::DIFF_ON_Y_DIR == nDimension)
        matKernel = (cv::Mat_<float>(2, 1) << -1, 1);
    auto filter = cv::cuda::createLinearFilter(CV_32FC1, CV_32FC1, matKernel, cv::Point(-1, -1), cv::BORDER_CONSTANT);
    cv::cuda::GpuMat matResult;
    filter->apply(matInput, matResult, stream);
    if (CalcUtils::DIFF_ON_X_DIR == nDimension)
        return cv::cuda::GpuMat(matResult, cv::Rect(1, 0, matResult.cols - 1, matResult.rows));
    else if (CalcUtils::DIFF_ON_Y_DIR == nDimension)
        return cv::cuda::GpuMat(matResult, cv::Rect(0, 1, matResult.cols, matResult.rows - 1));
    return cv::cuda::GpuMat();
}

/*static*/ void CudaAlgorithm::diff(
    const cv::cuda::GpuMat& matInput,
    cv::cuda::GpuMat& matResult,
    int nDimension,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    assert(CalcUtils::DIFF_ON_X_DIR == nDimension || CalcUtils::DIFF_ON_Y_DIR == nDimension);

    cv::Mat matKernel;
    if (CalcUtils::DIFF_ON_X_DIR == nDimension)
        matKernel = (cv::Mat_<float>(1, 2) << -1, 1);
    else if (CalcUtils::DIFF_ON_Y_DIR == nDimension)
        matKernel = (cv::Mat_<float>(2, 1) << -1, 1);
    auto filter = cv::cuda::createLinearFilter(CV_32FC1, CV_32FC1, matKernel, cv::Point(-1, -1), cv::BORDER_CONSTANT);
    filter->apply(matInput, matResult, stream);
    if (CalcUtils::DIFF_ON_X_DIR == nDimension)
        matResult = cv::cuda::GpuMat(matResult, cv::Rect(1, 0, matResult.cols - 1, matResult.rows));
    else if (CalcUtils::DIFF_ON_Y_DIR == nDimension)
        matResult = cv::cuda::GpuMat(matResult, cv::Rect(0, 1, matResult.cols, matResult.rows - 1));
}

/*static*/ float CudaAlgorithm::intervalAverage(const cv::cuda::GpuMat &matInput, int interval, float *d_result,  cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    float result = 0;
    run_kernel_interval_average(
        cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float *>(matInput.data),
        matInput.step1(),
        matInput.rows,
        matInput.cols,
        interval,
        d_result,
        &result);
    return result;
}

/*static*/ float CudaAlgorithm::intervalRangeAverage(const cv::cuda::GpuMat& matInput, int interval, float rangeStart, float rangeEnd, float* d_result,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    float result = 0;
    run_kernel_range_interval_average(
        cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float *>(matInput.data),
        matInput.step1(),
        matInput.rows,
        matInput.cols,
        interval,
        rangeStart,
        rangeEnd,
        d_result,
        &result);
    return result;
}

/*static*/ void CudaAlgorithm::phaseWrapBuffer(cv::cuda::GpuMat& matPhase,
    cv::cuda::GpuMat& matBuffer,
    float* d_tmpVar,
    float fShift/* = 0.f*/,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    CStopWatch stopWatch;
    // matBuffer = matPhase / ONE_CYCLE + 0.5f- fShift;
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matBuffer, 0.f, 0.5f - fShift, matBuffer, -1, stream); // -1 means the dest type is same as src type
    TimeLog::GetInstance()->addTimeLog("phaseWrapBuffer Divide and add.", stopWatch.Span());

    floor(matBuffer, stream);

    TimeLog::GetInstance()->addTimeLog("phaseWrapBuffer floor takes.", stopWatch.Span());

    cv::cuda::addWeighted(matPhase, 1, matBuffer, -ONE_CYCLE, 0, matBuffer, -1, stream);

    TimeLog::GetInstance()->addTimeLog("_phaseWrap multiply and subtract takes.", stopWatch.Span());

    float fMean = intervalAverage(matBuffer, 20, d_tmpVar, stream);

    //matBuffer = matPhase / ONE_CYCLE + 0.5 - fShift - fMean / 2.f;
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matBuffer, 0.f, 0.5f - fShift - fMean / 2.f, matBuffer, -1, stream);
    floor(matBuffer, stream);

    TimeLog::GetInstance()->addTimeLog("_phaseWrap floor takes.", stopWatch.Span());

    //matPhase = matPhase - matBuffer * ONE_CYCLE;
    cv::cuda::addWeighted(matPhase, 1, matBuffer, -ONE_CYCLE, 0, matPhase, -1, stream);
}

/*static*/ void CudaAlgorithm::phaseWrapByRefer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matRef, cv::cuda::Stream& stream/* = cv::cuda::Stream::Null()*/) {
    CStopWatch stopWatch;
    //matRef = (matPhase - matRef) / ONE_CYCLE + 0.5;  //Use matResult to store the substract result to reduce memory allocate time.
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matRef, -1.f / ONE_CYCLE, 0.5, matRef, -1, stream);
    TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer Divide and add.", stopWatch.Span());

    floor(matRef, stream);
    TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer floor takes.", stopWatch.Span());

    //matPhase = matPhase - matRef * ONE_CYCLE;
    cv::cuda::addWeighted(matPhase, 1.f, matRef, -ONE_CYCLE, 0, matPhase, -1, stream);
    TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer multiply and subtract takes.", stopWatch.Span());
}

/*static*/ void CudaAlgorithm::phaseWarpNoAutoBase(const cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matResult, float fShift,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    CStopWatch stopWatch;
    //cv::Mat matResult = matPhase / ONE_CYCLE + 0.5 - fShift;
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matResult, 0, 0.5 - fShift, matResult, -1, stream);
    TimeLog::GetInstance()->addTimeLog("_phaseWrap Divide and add.", stopWatch.Span());

    floor(matResult, stream);

    TimeLog::GetInstance()->addTimeLog("_phaseWrap floor takes.", stopWatch.Span());

    //matResult = matPhase - matResult * ONE_CYCLE;
    cv::cuda::addWeighted(matPhase, 1.f, matResult, -ONE_CYCLE, 0, matResult, -1, stream);
}

// matBufferGpu, matBufferGpuT, matDiffMapX, matDiffMapY, matDiffPhaseX, matDiffPhaseY, matMaskGpu, matMaskGpuT are reusable buffers,
// use them to avoid allocating memory every time. They are not passed out for any usage.
/*static*/ void CudaAlgorithm::phaseCorrectionCmp(
    cv::cuda::GpuMat& matPhase,
    const cv::cuda::GpuMat& matPhase1,
    cv::cuda::GpuMat& matBufferGpu,
    cv::cuda::GpuMat& matBufferGpuT,
    cv::cuda::GpuMat& matDiffMapX,
    cv::cuda::GpuMat& matDiffMapY,
    cv::cuda::GpuMat& matDiffPhaseX,
    cv::cuda::GpuMat& matDiffPhaseY,
    cv::cuda::GpuMat& matMaskGpu,
    cv::cuda::GpuMat& matMaskGpuT,
    int span,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    const int gridSize = 64;
    const int threadSize = 32;

    cv::cuda::compare(matPhase, matPhase1, matMaskGpu, cv::CmpTypes::CMP_NE, stream);
    matBufferGpu.setTo(0, stream);
    matBufferGpu.setTo(1, matMaskGpu, stream);

    // Note even the size of Mat ix ROW * (COL - 1), but actually the background memory size is still ROW * COL
    diff(matBufferGpu, matDiffMapX, CalcUtils::DIFF_ON_X_DIR, stream);
    diff(matPhase, matDiffPhaseX, CalcUtils::DIFF_ON_X_DIR, stream);

    matMaskGpu.setTo(0, stream);

    // Select in X direction
    run_kernel_select_cmp_point(gridSize, threadSize, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float *>(matDiffMapX.data),
        reinterpret_cast<float *>(matDiffPhaseX.data),
        matMaskGpu.data,
        matDiffMapX.step1(),
        ROWS, COLS, span);

    // Select in Y direction, transpose the matrix to accelerate
    cv::cuda::transpose(matBufferGpu, matBufferGpu, stream);
    cv::cuda::transpose(matPhase, matBufferGpuT, stream);

    diff(matBufferGpu, matDiffMapY, CalcUtils::DIFF_ON_X_DIR, stream);
    diff(matBufferGpuT, matDiffPhaseY, CalcUtils::DIFF_ON_X_DIR, stream);
    matMaskGpuT.setTo(0, stream);

    // Select in Y direction
    run_kernel_select_cmp_point(gridSize, threadSize, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float *>(matDiffMapY.data),
        reinterpret_cast<float *>(matDiffPhaseY.data),
        matMaskGpuT.data,
        matDiffMapY.step1(),
        COLS, ROWS, span);

    cv::cuda::transpose(matMaskGpuT, matMaskGpuT, stream);

    cv::cuda::bitwise_and(matMaskGpu, matMaskGpuT, matMaskGpu, cv::cuda::GpuMat(), stream);
    matPhase1.copyTo(matPhase, matMaskGpu, stream);
}

/*static*/ void CudaAlgorithm::phaseCorrection(
    cv::cuda::GpuMat &matPhase,
    cv::cuda::GpuMat& matPhaseT,
    cv::cuda::GpuMat& matDiffResult,
    cv::cuda::GpuMat& matDiffResultT,
    int nJumpSpanX,
    int nJumpSpanY,
    cv::cuda::Stream& stream/* = cv::cuda::Stream::Null()*/) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    const int gridSize = 16;
    const int threadSize = 32;
    
    //X direction
    if (nJumpSpanX > 0) {
        diff(matPhase, matDiffResult, CalcUtils::DIFF_ON_X_DIR, stream);
        run_kernel_phase_correction(gridSize, threadSize, cv::cuda::StreamAccessor::getStream(stream),
            reinterpret_cast<float *>(matDiffResult.data),
            reinterpret_cast<float *>(matPhase.data),
            matDiffResult.step1(),
            ROWS, COLS, nJumpSpanX);
        TimeLog::GetInstance()->addTimeLog("run_kernel_phase_correction for X", stopWatch.Span());
    }

    // Y direction
    if (nJumpSpanY > 0) {
        cv::cuda::transpose(matPhase, matPhaseT, stream);
        TimeLog::GetInstance()->addTimeLog("cv::cuda::transpose for Y", stopWatch.Span());
        diff(matPhaseT, matDiffResultT, CalcUtils::DIFF_ON_X_DIR, stream);
        run_kernel_phase_correction(gridSize, threadSize, cv::cuda::StreamAccessor::getStream(stream),
            reinterpret_cast<float *>(matDiffResultT.data),
            reinterpret_cast<float *>(matPhaseT.data),
            matDiffResultT.step1(),
            COLS, ROWS, nJumpSpanY);
        TimeLog::GetInstance()->addTimeLog("run_kernel_phase_correction for Y", stopWatch.Span());

        cv::cuda::transpose(matPhaseT, matPhase, stream);
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
    }
}

/*static*/ void CudaAlgorithm::calculateSurfaceConvert3D(
    const cv::cuda::GpuMat& matPhase,
    cv::cuda::GpuMat& matPhaseT,
    cv::cuda::GpuMat& matBufferGpu,
    const VectorOfFloat& param,
    const int ss,
    const int nDlpNo,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/)
{
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    const int TOTAL = ROWS * COLS;

    auto zmin = param[4];
    auto zmax = param[5];
    cv::cuda::transpose(matPhase, matPhaseT, stream);

    cv::cuda::addWeighted(matPhaseT, 1 / (zmax - zmin), matPhaseT, 0, - zmin / (zmax - zmin), matPhaseT, -1, stream);

    cv::cuda::GpuMat zInLine = matPhaseT.reshape(1, 1);
    
    int len = zInLine.cols;

    assert(len == TOTAL);

    auto matPolyParass = CalcUtils::paraFromPolyNomial(ss);
    cv::cuda::GpuMat matPolyParassGpu; matPolyParassGpu.upload(matPolyParass, stream);

    dim3 threads(16, 4);
    dim3 grid(256, 1, 1);
    float* d_p3 = m_arrCalc3DHeightVars[nDlpNo].d_p3;

    run_kernel_phase_to_height_3d(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float*>(zInLine.data),
        reinterpret_cast<float*>(matPolyParassGpu.data),
        m_dlpCalibData[nDlpNo].pDlp3DBezierSurface,
        ss,
        len, ss,
        d_p3);

    calcSumAndConvertMatrix(d_p3, ss, matPhaseT, stream);

    TimeLog::GetInstance()->addTimeLog("calculateSurfaceConvert3D run kernel", stopWatch.Span());
    cv::cuda::transpose(matPhaseT, matBufferGpu, stream);
    TimeLog::GetInstance()->addTimeLog("cv::cuda::transpose back data", stopWatch.Span());
}

/*static*/ void CudaAlgorithm::phaseToHeight3D(
    const cv::cuda::GpuMat& zInLine,
    const cv::cuda::GpuMat& matPolyParassGpu,
    const float* d_xxt,
    const int xxtStep,
    const int len,
    const int ss,
    float* d_P3,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    dim3 threads(16, 4);
    dim3 grid(256, 1, 1);
    run_kernel_phase_to_height_3d(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float*>(zInLine.data),
        reinterpret_cast<float*>(matPolyParassGpu.data),
        d_xxt,
        xxtStep,
        len, ss,
        d_P3);
}

/*static*/ void CudaAlgorithm::calcSumAndConvertMatrix(
    float* d_pInputData,
    int ss,
    cv::cuda::GpuMat& matResult,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    dim3 threads(16, 16);
    dim3 grid(divUp(matResult.cols, threads.x), divUp(matResult.rows, threads.y));
    matResult.setTo(0, stream);
    run_kernel_calc_sum_and_convert_matrix(
        grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        d_pInputData,
        ss,
        reinterpret_cast<float*>(matResult.data),
        matResult.step1(),
        matResult.rows,
        matResult.cols);
}

/*static*/ void CudaAlgorithm::medianFilter(
        const cv::cuda::GpuMat& matInput,
        cv::cuda::GpuMat& matOutput,
        const int windowSize,
        cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    dim3 threads(16, 16);
    dim3 grid(divUp(matInput.cols, threads.x), divUp(matInput.rows, threads.y));
    run_median_filter(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float*>(matInput.data),
        reinterpret_cast<float*>(matOutput.data),
        matInput.rows,
        matInput.cols,
        matInput.step1(),
        windowSize);
}

/*static*/ cv::Mat CudaAlgorithm::mergeHeightIntersect(cv::Mat matHeightOne,
    cv::Mat matNanMaskOne, cv::Mat matHeightTwo,
    cv::Mat matNanMaskTwo,
    float fDiffThreshold,
    PR_DIRECTION enProjDir,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
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
        cv::cuda::StreamAccessor::getStream(stream),
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