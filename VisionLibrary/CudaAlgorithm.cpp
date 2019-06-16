#include "CudaAlgorithm.h"
#include "../VisionLibrary/StopWatch.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/highgui.hpp"
#include "CudaFunc.h"
#include "TimeLog.h"
#include "CalcUtils.hpp"
#include "Constants.h"
#include "Log.h"

namespace AOI
{
namespace Vision
{

/*static*/ DlpCalibResult CudaAlgorithm::m_dlpCalibData[NUM_OF_DLP];
/*static*/ Calc3DHeightVars CudaAlgorithm::m_arrCalc3DHeightVars[NUM_OF_DLP];
/*static*/ cv::Ptr<cv::cuda::Filter> CudaAlgorithm::m_ptrXDiffFilter;
/*static*/ cv::Ptr<cv::cuda::Filter> CudaAlgorithm::m_ptrYDiffFilter;
/*static*/ cv::Ptr<cv::cuda::Filter> CudaAlgorithm::m_ptrGaussianFilter;
/*static*/ VectorOfGpuMat CudaAlgorithm::m_arrVecGpuMat[NUM_OF_DLP];
/*static*/ bool CudaAlgorithm::m_bParamsInitialized = false;

static int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

/*static*/ bool CudaAlgorithm::initCuda() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= 0)
        return false;

    size_t size = 0;
    cudaDeviceReset();

    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, size * 60);
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16);

    cv::Mat matKernelX = (cv::Mat_<float>(1, 2) << -1, 1);
    m_ptrXDiffFilter.release();
    m_ptrXDiffFilter = cv::cuda::createLinearFilter(CV_32FC1, CV_32FC1, matKernelX, cv::Point(-1, -1), cv::BORDER_CONSTANT);

    cv::Mat matKernelY = (cv::Mat_<float>(2, 1) << -1, 1);
    m_ptrYDiffFilter.release();
    m_ptrYDiffFilter = cv::cuda::createLinearFilter(CV_32FC1, CV_32FC1, matKernelY, cv::Point(-1, -1), cv::BORDER_CONSTANT);

    m_ptrGaussianFilter.release();
    m_ptrGaussianFilter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(5, 5), 5, 5, cv::BorderTypes::BORDER_REPLICATE);
    return true;
}

/*static*/ int CudaAlgorithm::getNanCount(const cv::cuda::GpuMat& matInput) {
    cv::cuda::GpuMat matMask;
    cv::cuda::compare(matInput, matInput, matMask, cv::CmpTypes::CMP_EQ);

    return matInput.rows * matInput.cols - cv::cuda::countNonZero(matMask);
}

/*static*/ VisionStatus CudaAlgorithm::setDlpParams(const PR_SET_DLP_PARAMS_TO_GPU_CMD* const pstCmd, PR_SET_DLP_PARAMS_TO_GPU_RPY* const pstRpy) {
    pstRpy->enStatus = VisionStatus::OK;
    if (m_bParamsInitialized) {
        WriteLog("Dlp params already set, no need set again");
        return pstRpy->enStatus;
    }

    const size_t MEM_SIZE = pstCmd->vec3DBezierSurface[0].total() * sizeof(float);
    float* h_buffer;
    cudaMallocHost(&h_buffer, MEM_SIZE);
    const int ROWS = pstCmd->vec3DBezierSurface[0].rows;
    const int COLS = pstCmd->vec3DBezierSurface[0].cols;

    for (int dlp = 0; dlp < NUM_OF_DLP; ++dlp) {
        if (pstCmd->vec3DBezierSurface[dlp].empty()) {
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            WriteLog("3D bezier surface is empty");
            return pstRpy->enStatus;
        }

        if (pstCmd->vecMatAlphaBase[dlp].empty()) {
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            WriteLog("3D alpha base is empty");
            return pstRpy->enStatus;
        }

        if (pstCmd->vecMatBetaBase[dlp].empty()) {
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            WriteLog("3D beta base is empty");
            return pstRpy->enStatus;
        }

        if (pstCmd->vecMatGammaBase[dlp].empty()) {
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            WriteLog("3D gamma base is empty");
            return pstRpy->enStatus;
        }

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

        const int PhaseCorrectionBufferSize = 2048 * 1024 * sizeof(int);
        err = cudaMalloc(reinterpret_cast<void **>(&m_arrCalc3DHeightVars[dlp].pBufferJumpSpan), PhaseCorrectionBufferSize);
        if (err != cudaSuccess) {
            pstRpy->enStatus = VisionStatus::CUDA_MEMORY_ERROR;
            return pstRpy->enStatus;
        }

        err = cudaMalloc(reinterpret_cast<void **>(&m_arrCalc3DHeightVars[dlp].pBufferJumpStart), PhaseCorrectionBufferSize);
        if (err != cudaSuccess) {
            pstRpy->enStatus = VisionStatus::CUDA_MEMORY_ERROR;
            return pstRpy->enStatus;
        }

        err = cudaMalloc(reinterpret_cast<void **>(&m_arrCalc3DHeightVars[dlp].pBufferJumpEnd), PhaseCorrectionBufferSize);
        if (err != cudaSuccess) {
            pstRpy->enStatus = VisionStatus::CUDA_MEMORY_ERROR;
            return pstRpy->enStatus;
        }

        err = cudaMalloc(reinterpret_cast<void **>(&m_arrCalc3DHeightVars[dlp].pBufferSortedJumpSpanIdx), PhaseCorrectionBufferSize);
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
        m_arrCalc3DHeightVars[dlp].matBufferGpuT_1 = cv::cuda::GpuMat(2040, 2048, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matMaskGpu = cv::cuda::GpuMat(2048, 2040, CV_8UC1);
        m_arrCalc3DHeightVars[dlp].matMaskGpuT = cv::cuda::GpuMat(2040, 2048, CV_8UC1);
        m_arrCalc3DHeightVars[dlp].matMaskGpu_1 = cv::cuda::GpuMat(2048, 2040, CV_8UC1);
        m_arrCalc3DHeightVars[dlp].matMaskGpuT_1 = cv::cuda::GpuMat(2040, 2048, CV_8UC1);
        m_arrCalc3DHeightVars[dlp].matMaskGpu_2 = cv::cuda::GpuMat(2048, 2040, CV_8UC1);
        m_arrCalc3DHeightVars[dlp].matMaskGpuT_2 = cv::cuda::GpuMat(2040, 2048, CV_8UC1);
        m_arrCalc3DHeightVars[dlp].matDiffResult = cv::cuda::GpuMat(2048, 2040, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matDiffResultT = cv::cuda::GpuMat(2040, 2048, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matDiffResult_1 = cv::cuda::GpuMat(2048, 2040, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matDiffResultT_1 = cv::cuda::GpuMat(2040, 2048, CV_32FC1);
        m_arrCalc3DHeightVars[dlp].matBufferSign = cv::cuda::GpuMat(2048, 2048, CV_8SC1);
        m_arrCalc3DHeightVars[dlp].matBufferAmpl = cv::cuda::GpuMat(2048, 2048, CV_8SC1);

        err = cudaEventCreate(&m_arrCalc3DHeightVars[dlp].eventDone);
        if (err != cudaSuccess) {
            pstRpy->enStatus = VisionStatus::CUDA_MEMORY_ERROR;
            return pstRpy->enStatus;
        }

        for (int i = 0; i < 12; ++i) {
            m_arrVecGpuMat[dlp].push_back(cv::cuda::GpuMat(2048, 2040, CV_8UC1));
        }
    }
    cudaFreeHost(h_buffer);
    m_bParamsInitialized = true;
    WriteLog("Set dlp params success");
    return pstRpy->enStatus;
}

/*static*/ VisionStatus CudaAlgorithm::clearDlpParams() {
    if (!m_bParamsInitialized) {
        WriteLog("Dlp params already cleared, no need clear again");
        return VisionStatus::OK;
    }

    for (int dlp = 0; dlp < NUM_OF_DLP; ++dlp) {
        cudaError_t err = cudaFree(m_dlpCalibData[dlp].pDlp3DBezierSurface);
        if (err != cudaSuccess) {
            WriteLog("cudaFree failed, error code " + std::to_string(err));
            return VisionStatus::CUDA_MEMORY_ERROR;
        }

        m_dlpCalibData[dlp].matAlphaBase.release();
        m_dlpCalibData[dlp].matBetaBase.release();
        m_dlpCalibData[dlp].matGammaBase.release();

        err = cudaFree(m_arrCalc3DHeightVars[dlp].d_p3);
        if (err != cudaSuccess) {
            WriteLog("cudaFree failed, error code " + std::to_string(err));
            return VisionStatus::CUDA_MEMORY_ERROR;
        }

        err = cudaFree(m_arrCalc3DHeightVars[dlp].d_tmpResult);
        if (err != cudaSuccess) {
            WriteLog("cudaFree failed, error code " + std::to_string(err));
            return VisionStatus::CUDA_MEMORY_ERROR;
        }

        err = cudaFree(m_arrCalc3DHeightVars[dlp].pBufferJumpSpan);
        if (err != cudaSuccess) {
            WriteLog("cudaFree failed, error code " + std::to_string(err));
            return VisionStatus::CUDA_MEMORY_ERROR;
        }

        err = cudaFree(m_arrCalc3DHeightVars[dlp].pBufferJumpStart);
        if (err != cudaSuccess) {
            WriteLog("cudaFree failed, error code " + std::to_string(err));
            return VisionStatus::CUDA_MEMORY_ERROR;
        }

        err = cudaFree(m_arrCalc3DHeightVars[dlp].pBufferJumpEnd);
        if (err != cudaSuccess) {
            WriteLog("cudaFree failed, error code " + std::to_string(err));
            return VisionStatus::CUDA_MEMORY_ERROR;
        }

        err = cudaFree(m_arrCalc3DHeightVars[dlp].pBufferSortedJumpSpanIdx);
        if (err != cudaSuccess) {
            WriteLog("cudaFree failed, error code " + std::to_string(err));
            return VisionStatus::CUDA_MEMORY_ERROR;
        }

        m_arrCalc3DHeightVars[dlp].matAlpha.release();
        m_arrCalc3DHeightVars[dlp].matBeta.release();
        m_arrCalc3DHeightVars[dlp].matGamma.release();
        m_arrCalc3DHeightVars[dlp].matGamma1.release();
        m_arrCalc3DHeightVars[dlp].matAvgUnderTolIndex.release();
        m_arrCalc3DHeightVars[dlp].matBufferGpu.release();
        m_arrCalc3DHeightVars[dlp].matBufferGpuT.release();
        m_arrCalc3DHeightVars[dlp].matBufferGpuT_1.release();
        m_arrCalc3DHeightVars[dlp].matMaskGpu.release();
        m_arrCalc3DHeightVars[dlp].matMaskGpuT.release();
        m_arrCalc3DHeightVars[dlp].matMaskGpu_1.release();
        m_arrCalc3DHeightVars[dlp].matMaskGpuT_1.release();
        m_arrCalc3DHeightVars[dlp].matMaskGpu_2.release();
        m_arrCalc3DHeightVars[dlp].matMaskGpuT_2.release();
        m_arrCalc3DHeightVars[dlp].matDiffResult.release();
        m_arrCalc3DHeightVars[dlp].matDiffResultT.release();
        m_arrCalc3DHeightVars[dlp].matDiffResult_1.release();
        m_arrCalc3DHeightVars[dlp].matDiffResultT_1.release();
        m_arrCalc3DHeightVars[dlp].matBufferSign.release();
        m_arrCalc3DHeightVars[dlp].matBufferAmpl.release();

        err = cudaEventDestroy(m_arrCalc3DHeightVars[dlp].eventDone);
        if (err != cudaSuccess) {
            WriteLog("cudaEventDestroy failed, error code " + std::to_string(err));
            return VisionStatus::CUDA_MEMORY_ERROR;
        }

        m_arrVecGpuMat[dlp].clear();
    }
    m_bParamsInitialized = false;
    WriteLog("Clear dlp params success");
    return VisionStatus::OK;
}

/*static*/ void CudaAlgorithm::floor(cv::cuda::GpuMat &matInOut, cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    dim3 threads(16, 16);
    dim3 grid(divUp(matInOut.cols, threads.x), divUp(matInOut.rows, threads.y));
    run_kernel_floor(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float *>(matInOut.data),
        ToInt32(matInOut.step1()),
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
        ToInt32(matInput0.step1()));
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
        ToInt32(matInput0.step1()));
}

/*static*/ cv::cuda::GpuMat CudaAlgorithm::diff(const cv::cuda::GpuMat& matInput, int nRecersiveTime, int nDimension,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    assert(CalcUtils::DIFF_ON_X_DIR == nDimension || CalcUtils::DIFF_ON_Y_DIR == nDimension);
    if (nRecersiveTime > 1)
        return diff(diff(matInput, nRecersiveTime - 1, nDimension), 1, nDimension);

    cv::cuda::GpuMat matResult;
    if (CalcUtils::DIFF_ON_X_DIR == nDimension)
        m_ptrXDiffFilter->apply(matInput, matResult, stream);
    else if (CalcUtils::DIFF_ON_Y_DIR == nDimension)
        m_ptrYDiffFilter->apply(matInput, matResult, stream);

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

    if (CalcUtils::DIFF_ON_X_DIR == nDimension)
        m_ptrXDiffFilter->apply(matInput, matResult, stream);
    else if (CalcUtils::DIFF_ON_Y_DIR == nDimension)
        m_ptrYDiffFilter->apply(matInput, matResult, stream);
}

/*static*/ float CudaAlgorithm::intervalAverage(const cv::cuda::GpuMat &matInput, int interval, float *d_result, 
    cudaEvent_t& eventDone, cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    float result = 0;
    run_kernel_interval_average(
        eventDone,
        cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float *>(matInput.data),
        ToInt32(matInput.step1()),
        matInput.rows,
        matInput.cols,
        interval,
        d_result,
        &result);
    return result;
}

/*static*/ float CudaAlgorithm::intervalRangeAverage(const cv::cuda::GpuMat& matInput, int interval, float rangeStart, float rangeEnd, float* d_result,
    cudaEvent_t& eventDone,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    float result = 0;
    run_kernel_range_interval_average(
        eventDone,
        cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float *>(matInput.data),
        ToInt32(matInput.step1()),
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
    float fShift,
    cudaEvent_t& eventDone,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    // matBuffer = matPhase / ONE_CYCLE + 0.5f- fShift;
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matBuffer, 0.f, 0.5f - fShift, matBuffer, -1, stream); // -1 means the dest type is same as src type

    floor(matBuffer, stream);

    cv::cuda::addWeighted(matPhase, 1, matBuffer, -ONE_CYCLE, 0, matBuffer, -1, stream);

    float fMean = intervalAverage(matBuffer, 40, d_tmpVar, eventDone, stream);

    //matBuffer = matPhase / ONE_CYCLE + 0.5 - fShift - fMean / 2.f;
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matBuffer, 0.f, 0.5f - fShift - fMean / 2.f, matBuffer, -1, stream);
    floor(matBuffer, stream);

    //matPhase = matPhase - matBuffer * ONE_CYCLE;
    cv::cuda::addWeighted(matPhase, 1, matBuffer, -ONE_CYCLE, 0, matPhase, -1, stream);
}

/*static*/ void CudaAlgorithm::phaseWrapByRefer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matRef, cv::cuda::Stream& stream/* = cv::cuda::Stream::Null()*/) {
    //matRef = (matPhase - matRef) / ONE_CYCLE + 0.5;  //Use matResult to store the substract result to reduce memory allocate time.
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matRef, -1.f / ONE_CYCLE, 0.5, matRef, -1, stream);

    floor(matRef, stream);

    //matPhase = matPhase - matRef * ONE_CYCLE;
    cv::cuda::addWeighted(matPhase, 1.f, matRef, -ONE_CYCLE, 0, matPhase, -1, stream);
}

/*static*/ void CudaAlgorithm::phaseWarpNoAutoBase(const cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matResult, float fShift,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    //cv::Mat matResult = matPhase / ONE_CYCLE + 0.5 - fShift;
    cv::cuda::addWeighted(matPhase, 1.f / ONE_CYCLE, matResult, 0, 0.5 - fShift, matResult, -1, stream);

    floor(matResult, stream);

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
    cv::cuda::GpuMat& matBufferGpuT_1,
    cv::cuda::GpuMat& matDiffMapX,
    cv::cuda::GpuMat& matDiffMapY,
    cv::cuda::GpuMat& matDiffPhaseX,
    cv::cuda::GpuMat& matDiffPhaseY,
    cv::cuda::GpuMat& matMaskGpu,
    cv::cuda::GpuMat& matMaskGpuT,
    cv::cuda::GpuMat& matMaskGpu_1,
    int* pBufferArrayIdx1,
    int* pBufferArrayIdx2,
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
        ToInt32(matDiffMapX.step1()),
        ROWS, COLS,
        pBufferArrayIdx1,
        pBufferArrayIdx2,
        span);

    // Select in Y direction, transpose the matrix to accelerate
    cv::cuda::transpose(matBufferGpu, matBufferGpuT_1, stream);
    cv::cuda::transpose(matPhase, matBufferGpuT, stream);

    diff(matBufferGpuT_1, matDiffMapY, CalcUtils::DIFF_ON_X_DIR, stream);
    diff(matBufferGpuT, matDiffPhaseY, CalcUtils::DIFF_ON_X_DIR, stream);
    matMaskGpuT.setTo(0, stream);

    // Select in Y direction
    run_kernel_select_cmp_point(gridSize, threadSize, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float *>(matDiffMapY.data),
        reinterpret_cast<float *>(matDiffPhaseY.data),
        matMaskGpuT.data,
        ToInt32(matDiffMapY.step1()),
        COLS, ROWS,
        pBufferArrayIdx1, pBufferArrayIdx2, span);

    cv::cuda::transpose(matMaskGpuT, matMaskGpu_1, stream);

    cv::cuda::bitwise_and(matMaskGpu, matMaskGpu_1, matMaskGpu, cv::cuda::GpuMat(), stream);
    matPhase1.copyTo(matPhase, matMaskGpu, stream);
}

/*static*/ void CudaAlgorithm::phaseCorrection(
    cv::cuda::GpuMat& matPhase,
    cv::cuda::GpuMat& matPhaseT,
    cv::cuda::GpuMat& matDiffResult,
    cv::cuda::GpuMat& matDiffResultT,
    cv::cuda::GpuMat& matBufferSign,
    cv::cuda::GpuMat& matBufferAmpl,
    int* pBufferJumpSpan,
    int* pBufferJumpStart,
    int* pBufferJumpEnd,
    int* pBufferSortedJumpSpanIdx,
    int nJumpSpanX,
    int nJumpSpanY,
    cv::cuda::Stream& stream/* = cv::cuda::Stream::Null()*/) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    
    //X direction
    if (nJumpSpanX > 0) {
        diff(matPhase, matDiffResult, CalcUtils::DIFF_ON_X_DIR, stream);
        dim3 threads(16, 16);
        dim3 grid(divUp(matPhase.cols, threads.x), divUp(matPhase.rows, threads.y));
        matBufferSign.setTo(0, stream);
        matBufferAmpl.setTo(0, stream);
        run_kernel_phase_correction(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
            reinterpret_cast<float *>(matDiffResult.data),
            reinterpret_cast<float *>(matPhase.data),
            reinterpret_cast<char *>(matBufferSign.data),
            reinterpret_cast<char *>(matBufferAmpl.data),
            pBufferJumpSpan,
            pBufferJumpStart,
            pBufferJumpEnd,
            pBufferSortedJumpSpanIdx,
            ToInt32(matDiffResult.step1()),
            ROWS, COLS, nJumpSpanX);
        //TimeLog::GetInstance()->addTimeLog("run_kernel_phase_correction for X", stopWatch.Span());
    }

    // Y direction
    if (nJumpSpanY > 0) {
        cv::cuda::transpose(matPhase, matPhaseT, stream);
        TimeLog::GetInstance()->addTimeLog("cv::cuda::transpose for Y", stopWatch.Span());
        diff(matPhaseT, matDiffResultT, CalcUtils::DIFF_ON_X_DIR, stream);
        dim3 threads(16, 16);
        dim3 grid(divUp(matPhaseT.cols, threads.x), divUp(matPhaseT.rows, threads.y));
        matBufferSign.setTo(0, stream);
        matBufferAmpl.setTo(0, stream);
        run_kernel_phase_correction(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
            reinterpret_cast<float *>(matDiffResultT.data),
            reinterpret_cast<float *>(matPhaseT.data),
            reinterpret_cast<char *>(matBufferSign.data),
            reinterpret_cast<char *>(matBufferAmpl.data),
            pBufferJumpSpan,
            pBufferJumpStart,
            pBufferJumpEnd,
            pBufferSortedJumpSpanIdx,
            ToInt32(matDiffResultT.step1()),
            COLS, ROWS, nJumpSpanY);
        //TimeLog::GetInstance()->addTimeLog("run_kernel_phase_correction for Y", stopWatch.Span());

        cv::cuda::transpose(matPhaseT, matPhase, stream);
        //TimeLog::GetInstance()->addTimeLog("cv::cuda::transpose back", stopWatch.Span());
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
        ToInt32(matResult.step1()),
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
        ToInt32(matInput.step1()),
        windowSize);
}

/*static*/ void CudaAlgorithm::phasePatch(
        cv::cuda::GpuMat& matInOut,
        const cv::cuda::GpuMat& matNanMask,
        PR_DIRECTION enProjDir,
        cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    dim3 threads(32, 32);
    dim3 grid(divUp(matInOut.cols, threads.x), divUp(matInOut.rows, threads.y));
    run_kernel_phase_patch(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float*>(matInOut.data),
        matNanMask.data,
        matInOut.rows,
        matInOut.cols,
        ToInt32(matInOut.step1()),
        enProjDir);
}

/*static*/ void CudaAlgorithm::getNanMask(
    cv::cuda::GpuMat& matInput,
    cv::cuda::GpuMat& matNanMask,
    cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    cv::cuda::compare(matInput, matInput, matNanMask, cv::CmpTypes::CMP_EQ, stream);
    cv::cuda::bitwise_not(matNanMask, matNanMask, cv::cuda::GpuMat(), stream);
}

/*static*/ void CudaAlgorithm::mergeHeightIntersectGpu(
        cv::cuda::GpuMat& matHGpu1,
        cv::cuda::GpuMat& matHGpu2,
        cv::cuda::GpuMat& matHGpu3,
        cv::cuda::GpuMat& matHGpu4,
        cv::cuda::GpuMat& matNanMask,
        cv::cuda::GpuMat& matNanMaskDiff,
        int*           pMergeIndexBuffer,
        float*         pCmpTargetBuffer,
        float fDiffThreshold,
        cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    uint32_t threads = 32;
    uint32_t grid = divUp(matHGpu1.rows, threads);
    run_kernel_merge_height_intersect(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float*>(matHGpu1.data),
        reinterpret_cast<float*>(matHGpu2.data),
        reinterpret_cast<float*>(matHGpu3.data),
        reinterpret_cast<float*>(matHGpu4.data),
        matNanMask.data,
        reinterpret_cast<float*>(matNanMaskDiff.data),
        matHGpu1.rows,
        matHGpu1.cols,
        ToInt32(matHGpu1.step1()),
        pMergeIndexBuffer,
        pCmpTargetBuffer,
        fDiffThreshold);
}

/*static*/ void CudaAlgorithm::mergeHeightIntersect(
        cv::cuda::GpuMat& matHeightOneInput,
        cv::cuda::GpuMat& matNanMaskOne,
        cv::cuda::GpuMat& matHeightTwoInput,
        cv::cuda::GpuMat& matNanMaskTwo,
        cv::cuda::GpuMat& matBufferGpu1,
        cv::cuda::GpuMat& matBufferGpu2,
        cv::cuda::GpuMat& matHeightOne,
        cv::cuda::GpuMat& matHeightTwo,
        cv::cuda::GpuMat& matHeightTre,
        cv::cuda::GpuMat& matHeightFor,
        cv::cuda::GpuMat& matMaskGpu1,
        cv::cuda::GpuMat& matMaskGpu2,
        cv::cuda::GpuMat& matMaskGpu3,
        cv::cuda::GpuMat& matMaskGpu4,
        cv::cuda::GpuMat& matAbsDiff, //Float
        cv::cuda::GpuMat& matBigDiffMask,
        cv::cuda::GpuMat& matBigDiffMaskFloat,
        cv::cuda::GpuMat& matDiffResultDiff,
        int*           pMergeIndexBuffer,
        float*         pCmpTargetBuffer,
        float fDiffThreshold,
        PR_DIRECTION enProjDir,
        cv::cuda::GpuMat& matMergeResult,
        cv::cuda::GpuMat& matResultNan,
        cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    PR_DIRECTION enProjDir2;
    if (ToInt32(enProjDir) >= 2)
        enProjDir2 = static_cast<PR_DIRECTION>(5 - ToInt32(enProjDir));
    else
        enProjDir2 = static_cast<PR_DIRECTION>(1 - ToInt32(enProjDir));

    matHeightOneInput.copyTo(matBufferGpu1, stream); // matBufferGpu1 is matH13
    CudaAlgorithm::phasePatch(matBufferGpu1, matNanMaskOne, enProjDir2, stream);

    matHeightTwoInput.copyTo(matBufferGpu2, stream); // matBufferGpu2 is matH23
    CudaAlgorithm::phasePatch(matBufferGpu2, matNanMaskTwo, enProjDir, stream);

    // Matlab: idxh2 = find(H22(idxnan1) - H13(idxnan1) > 0.05);
    //         idxh1 = find(H11(idxnan2) - H23(idxnan2) > 0.05);
    cv::cuda::add(matBufferGpu1, 0.05f, matBufferGpu1, cv::cuda::GpuMat(), -1, stream);
    cv::cuda::add(matBufferGpu2, 0.05f, matBufferGpu2, cv::cuda::GpuMat(), -1, stream);
    
    //idxh2 = (matHeightTwo - matH13) > 0.05f; idxh2 = idxh2 & matNanMaskOne;
    //idxh1 = (matHeightOne - matH23) > 0.05f; idxh1 = idxh1 & matNanMaskTwo;
    cv::cuda::compare(matHeightTwoInput, matBufferGpu1, matMaskGpu2, cv::CmpTypes::CMP_GT, stream);
    cv::cuda::bitwise_and(matMaskGpu2, matNanMaskOne, matMaskGpu2, cv::cuda::GpuMat(), stream);

    cv::cuda::compare(matHeightOneInput, matBufferGpu2, matMaskGpu1, cv::CmpTypes::CMP_GT, stream);
    cv::cuda::bitwise_and(matMaskGpu1, matNanMaskTwo, matMaskGpu1, cv::cuda::GpuMat(), stream);

    //cv::Mat idxnant1 = matNanMaskOne.clone(); idxnant1.setTo(255, idxh1);
    //auto idxnant1 = matNanMaskOne.clone(); idxnant1.setTo(255, matMaskGpu1, stream);
    //cv::Mat idxnant2 = matNanMaskTwo.clone(); idxnant2.setTo(255, idxh2);
    //auto idxnant2 = matNanMaskTwo.clone(); idxnant2.setTo(255, matMaskGpu2, stream);

    matNanMaskOne.setTo(PR_MAX_GRAY_LEVEL, matMaskGpu1, stream);
    matNanMaskTwo.setTo(PR_MAX_GRAY_LEVEL, matMaskGpu2, stream);
    
    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
        cv::cuda::transpose(matHeightOneInput, matHeightOne, stream);
        cv::cuda::transpose(matHeightTwoInput, matHeightTwo, stream);
        cv::cuda::transpose(matNanMaskOne, matMaskGpu3, stream);
        cv::cuda::transpose(matNanMaskTwo, matMaskGpu4, stream);
    }else {
        matHeightOneInput.copyTo(matHeightOne, stream);
        matHeightTwoInput.copyTo(matHeightTwo, stream);
        matNanMaskOne.copyTo(matMaskGpu3, stream);
        matNanMaskTwo.copyTo(matMaskGpu4, stream);
    }

    cv::cuda::addWeighted(matHeightOne, 0.5f, matHeightTwo, 0.5f, 0, matHeightTre, -1, stream); // matBufferGpu1 is matHeightTre
    matHeightTwo.copyTo(matHeightTre, matMaskGpu3, stream);
    matHeightOne.copyTo(matHeightTre, matMaskGpu4, stream);

    matHeightTre.copyTo(matHeightFor, stream); // Equal to cv::Mat matHeightFor = matHeightTre.clone();

    cv::cuda::absdiff(matHeightOne, matHeightTwo, matAbsDiff, stream);
    cv::cuda::compare(matAbsDiff, fDiffThreshold, matBigDiffMask, cv::CmpTypes::CMP_GT, stream);
    matHeightFor.setTo(NAN, matBigDiffMask, stream);

    getNanMask(matHeightFor, matBigDiffMask, stream);
    matBigDiffMask.convertTo(matBigDiffMaskFloat, CV_32FC1, stream);
    diff(matBigDiffMaskFloat, matDiffResultDiff, CalcUtils::DIFF_ON_X_DIR, stream);
    cv::cuda::abs(matDiffResultDiff, matDiffResultDiff, stream);

    //static int times = 0;
    //++ times;
    //cv::Mat matCpuH1, matCpuH2, matCpuH3, matCpuH4, matCpuMask, matCpuMaskDiffResult;
    //matHeightOne.download(matCpuH1);
    //matHeightTwo.download(matCpuH2);
    //matHeightTre.download(matCpuH3);
    //matHeightFor.download(matCpuH4);
    //matBigDiffMask.download(matCpuMask);
    //matDiffResultLocal.download(matCpuMaskDiffResult);
    //char fileName[200];
    //_snprintf(fileName, sizeof(fileName), "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/Gpu_BeforeMergeHeightResult_%d.yml", times);
    //cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
    //if (!fs.isOpened())
    //    return matBufferGpu2;

    //cv::write(fs, "H1", matCpuH1);
    //cv::write(fs, "H2", matCpuH2);
    //cv::write(fs, "H3", matCpuH3);
    //cv::write(fs, "H4", matCpuH4);
    //cv::write(fs, "Mask", matCpuMask);
    //cv::write(fs, "MaskDiff", matCpuMaskDiffResult);
    //fs.release();

    mergeHeightIntersectGpu(
        matHeightOne,
        matHeightTwo,
        matHeightTre,
        matHeightFor,
        matBigDiffMask,
        matDiffResultDiff,
        pMergeIndexBuffer,
        pCmpTargetBuffer,
        fDiffThreshold,
        stream);

    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
        cv::cuda::transpose(matHeightFor, matMergeResult, stream);
    }else {
        matHeightFor.copyTo(matMergeResult, stream);
    }

    // matResultNan = idxnant1 & idxnant2;
    cv::cuda::bitwise_and(matNanMaskOne, matNanMaskTwo, matResultNan, cv::cuda::GpuMat(), stream);
}

/*static*/ cv::cuda::GpuMat CudaAlgorithm::runMergeHeightIntersect(
        cv::cuda::GpuMat& matHeightOneInput,
        cv::cuda::GpuMat& matNanMaskOne,
        cv::cuda::GpuMat& matHeightTwoInput,
        cv::cuda::GpuMat& matNanMaskTwo,
        Calc3DHeightVars& calc3DHeightVar0,
        Calc3DHeightVars& calc3DHeightVar1,
        float fDiffThreshold,
        PR_DIRECTION enProjDir,
        cv::cuda::GpuMat& matResultNan,
        cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    cv::cuda::GpuMat& matBufferGpu1 = calc3DHeightVar0.matAlpha;
    cv::cuda::GpuMat& matBufferGpu2 = calc3DHeightVar0.matBeta;
    cv::cuda::GpuMat& matMaskGpu1 = calc3DHeightVar0.matMaskGpu;
    cv::cuda::GpuMat& matMaskGpu2 = calc3DHeightVar1.matMaskGpu;
    cv::cuda::GpuMat& matMergeResult = calc3DHeightVar1.matGamma;
    matResultNan = calc3DHeightVar0.matMaskGpu;

    cv::cuda::GpuMat matHeightOne, matHeightTwo, matHeightTre, matHeightFor, matMaskGpu3, matMaskGpu4;
    cv::cuda::GpuMat matAbsDiff, matBigDiffMask, matBigDiffMaskFloat, matDiffResultDiff;
    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
        matHeightOne = calc3DHeightVar0.matBufferGpuT;
        matHeightTwo = calc3DHeightVar1.matBufferGpuT;
        matHeightTre = calc3DHeightVar0.matBufferGpuT_1;
        matHeightFor = calc3DHeightVar1.matBufferGpuT_1;
        matMaskGpu3 = calc3DHeightVar0.matMaskGpuT_1;
        matMaskGpu4 = calc3DHeightVar1.matMaskGpuT_1;
        matAbsDiff = calc3DHeightVar0.matDiffResultT;
        matBigDiffMask = calc3DHeightVar0.matMaskGpuT_2;
        matBigDiffMaskFloat = calc3DHeightVar1.matDiffResultT;
        matDiffResultDiff = calc3DHeightVar0.matDiffResultT_1;
    }
    else {
        matHeightOne = calc3DHeightVar0.matGamma;
        matHeightTwo = calc3DHeightVar0.matGamma1;
        matHeightTre = calc3DHeightVar0.matBufferGpu;
        matHeightFor = calc3DHeightVar1.matBufferGpu;
        matMaskGpu3 = calc3DHeightVar0.matMaskGpu_1;
        matMaskGpu4 = calc3DHeightVar1.matMaskGpu_1;
        matAbsDiff = calc3DHeightVar0.matDiffResult;
        matBigDiffMask = calc3DHeightVar0.matMaskGpu_2;
        matBigDiffMaskFloat = calc3DHeightVar1.matDiffResult;
        matDiffResultDiff = calc3DHeightVar0.matDiffResult_1;
    }
    int* pMergeIndexBuffer = reinterpret_cast<int*>(calc3DHeightVar0.d_p3);
    float* pCmpTargetBuffer = calc3DHeightVar1.d_p3;

    mergeHeightIntersect(
        matHeightOneInput,
        matNanMaskOne,
        matHeightTwoInput,
        matNanMaskTwo,
        matBufferGpu1,
        matBufferGpu2,
        matHeightOne,
        matHeightTwo,
        matHeightTre,
        matHeightFor,
        matMaskGpu1,
        matMaskGpu2,
        matMaskGpu3,
        matMaskGpu4,
        matAbsDiff, //Float
        matBigDiffMask,
        matBigDiffMaskFloat,
        matDiffResultDiff,
        pMergeIndexBuffer,
        pCmpTargetBuffer,
        fDiffThreshold,
        enProjDir,
        matMergeResult,
        matResultNan,
        stream);
    return matMergeResult;
}

/*static*/ void CudaAlgorithm::chooseMinValueForMask(
        cv::cuda::GpuMat&       matH1,
        cv::cuda::GpuMat&       matH2,
        cv::cuda::GpuMat&       matH3,
        const cv::cuda::GpuMat& matMask,
        cv::cuda::Stream&       stream /*= cv::cuda::Stream::Null()*/) {
    dim3 threads(32, 32);
    dim3 grid(divUp(matH1.cols, threads.x), divUp(matH1.rows, threads.y));
    run_kernel_choose_min_value_for_mask(grid, threads, cv::cuda::StreamAccessor::getStream(stream),
        reinterpret_cast<float*>(matH1.data),
        reinterpret_cast<float*>(matH2.data),
        reinterpret_cast<float*>(matH3.data),
        matMask.data,
        matH1.rows,
        matH1.cols,
        ToInt32(matH1.step1()));
}

/*static*/ void CudaAlgorithm::patchNanBeforeMergeHeight(
        cv::cuda::GpuMat&       matHeightOneInput,
        const cv::cuda::GpuMat& matNanMaskOne,
        cv::cuda::GpuMat&       matHeightTwoInput,
        const cv::cuda::GpuMat& matNanMaskTwo,
        cv::cuda::GpuMat&       matBufferGpu1,
        cv::cuda::GpuMat&       matBufferGpu2,
        cv::cuda::Stream&       stream /*= cv::cuda::Stream::Null()*/) {
    matHeightOneInput.copyTo(matBufferGpu1, stream); // matBufferGpu1 is matH11
    CudaAlgorithm::phasePatch(matBufferGpu1,      matNanMaskOne, PR_DIRECTION::UP,   stream);
    CudaAlgorithm::phasePatch(matHeightOneInput,  matNanMaskOne, PR_DIRECTION::DOWN, stream); // matHeightOneInput is matH10

    matHeightTwoInput.copyTo(matBufferGpu2, stream); // matBufferGpu2 is matH22
    CudaAlgorithm::phasePatch(matBufferGpu2,      matNanMaskTwo, PR_DIRECTION::LEFT,  stream);
    CudaAlgorithm::phasePatch(matHeightTwoInput,  matNanMaskTwo, PR_DIRECTION::RIGHT, stream); // matHeightTwoInput is matH20

    chooseMinValueForMask(matBufferGpu1, matHeightOneInput, matBufferGpu2, matNanMaskOne, stream);
    chooseMinValueForMask(matBufferGpu2, matHeightTwoInput, matBufferGpu1, matNanMaskTwo, stream);
}

/*static*/ void CudaAlgorithm::mergeHeightIntersect06(
        const cv::cuda::GpuMat& matBufferGpu1,
        const cv::cuda::GpuMat& matBufferGpu2,
        cv::cuda::GpuMat& matHeightOne,
        cv::cuda::GpuMat& matHeightTwo,
        cv::cuda::GpuMat& matHeightTre,
        cv::cuda::GpuMat& matHeightFor,
        cv::cuda::GpuMat& matAbsDiff, //Float
        cv::cuda::GpuMat& matBigDiffMask,
        cv::cuda::GpuMat& matBigDiffMaskFloat,
        cv::cuda::GpuMat& matDiffResultDiff,
        int*              pMergeIndexBuffer,
        float*            pCmpTargetBuffer,
        float             fDiffThreshold,
        PR_DIRECTION      enProjDir,
        cv::cuda::GpuMat& matMergeResult,
        cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    //static int times = 0;
    //++ times;
    //if (1 == times) {
    //    cv::Mat matH11, matH10, matH22, matH20;
    //    matBufferGpu1.download(matH11);
    //    matHeightOneInput.download(matH10);
    //    matBufferGpu2.download(matH22);
    //    matHeightTwoInput.download(matH20);
    //    char fileName[200];
    //    _snprintf(fileName, sizeof(fileName), "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/Gpu_PatchPhaseResult_%d.yml", times);
    //    cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
    //    if (fs.isOpened()) {
    //        cv::write(fs, "H11", matH11);
    //        cv::write(fs, "H10", matH10);
    //        cv::write(fs, "H22", matH22);
    //        cv::write(fs, "H20", matH20);
    //        fs.release();
    //    }
    //}

    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
        cv::cuda::transpose(matBufferGpu1, matHeightOne, stream);
        cv::cuda::transpose(matBufferGpu2, matHeightTwo, stream);
    }else {
        matBufferGpu1.copyTo(matHeightOne, stream);
        matBufferGpu2.copyTo(matHeightTwo, stream);
    }

    cv::cuda::addWeighted(matHeightOne, 0.5f, matHeightTwo, 0.5f, 0, matHeightTre, -1, stream); // matBufferGpu1 is matHeightTre
    matHeightTre.copyTo(matHeightFor, stream); // Equal to cv::Mat matHeightFor = matHeightTre.clone();

    cv::cuda::absdiff(matHeightOne, matHeightTwo, matAbsDiff, stream);
    cv::cuda::compare(matAbsDiff, fDiffThreshold, matBigDiffMask, cv::CmpTypes::CMP_GT, stream);
    matHeightFor.setTo(NAN, matBigDiffMask, stream);

    getNanMask(matHeightFor, matBigDiffMask, stream);
    matBigDiffMask.convertTo(matBigDiffMaskFloat, CV_32FC1, stream);
    diff(matBigDiffMaskFloat, matDiffResultDiff, CalcUtils::DIFF_ON_X_DIR, stream);
    cv::cuda::abs(matDiffResultDiff, matDiffResultDiff, stream);

    mergeHeightIntersectGpu(
        matHeightOne,
        matHeightTwo,
        matHeightTre,
        matHeightFor,
        matBigDiffMask,
        matDiffResultDiff,
        pMergeIndexBuffer,
        pCmpTargetBuffer,
        fDiffThreshold,
        stream);

    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
        cv::cuda::transpose(matHeightFor, matMergeResult, stream);
    }else {
        matHeightFor.copyTo(matMergeResult, stream);
    }
}

/*static*/ cv::cuda::GpuMat CudaAlgorithm::runMergeHeightIntersect06(
        const cv::cuda::GpuMat& matBufferGpu1,
        const cv::cuda::GpuMat& matBufferGpu2,
        Calc3DHeightVars& calc3DHeightVar0,
        Calc3DHeightVars& calc3DHeightVar1,
        float fDiffThreshold,
        PR_DIRECTION enProjDir,
        cv::cuda::Stream& stream /*= cv::cuda::Stream::Null()*/) {
    //cv::cuda::GpuMat& matBufferGpu1 = calc3DHeightVar0.matAlpha;
    //cv::cuda::GpuMat& matBufferGpu2 = calc3DHeightVar0.matBeta;
    cv::cuda::GpuMat& matMergeResult = calc3DHeightVar1.matGamma1;

    cv::cuda::GpuMat matHeightOne, matHeightTwo, matHeightTre, matHeightFor;
    cv::cuda::GpuMat matAbsDiff, matBigDiffMask, matBigDiffMaskFloat, matDiffResultDiff;
    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
        matHeightOne = calc3DHeightVar0.matBufferGpuT;
        matHeightTwo = calc3DHeightVar1.matBufferGpuT;
        matHeightTre = calc3DHeightVar0.matBufferGpuT_1;
        matHeightFor = calc3DHeightVar1.matBufferGpuT_1;
        matAbsDiff = calc3DHeightVar0.matDiffResultT;
        matBigDiffMask = calc3DHeightVar0.matMaskGpuT_2;
        matBigDiffMaskFloat = calc3DHeightVar1.matDiffResultT;
        matDiffResultDiff = calc3DHeightVar0.matDiffResultT_1;
    }
    else {
        matHeightOne = calc3DHeightVar0.matGamma;
        matHeightTwo = calc3DHeightVar0.matGamma1;
        matHeightTre = calc3DHeightVar0.matBufferGpu;
        matHeightFor = calc3DHeightVar1.matBufferGpu;
        matAbsDiff = calc3DHeightVar0.matDiffResult;
        matBigDiffMask = calc3DHeightVar0.matMaskGpu_2;
        matBigDiffMaskFloat = calc3DHeightVar1.matDiffResult;
        matDiffResultDiff = calc3DHeightVar0.matDiffResult_1;
    }

    int* pMergeIndexBuffer = reinterpret_cast<int*>(calc3DHeightVar0.d_p3);
    float* pCmpTargetBuffer = calc3DHeightVar1.d_p3;

    mergeHeightIntersect06(
        matBufferGpu1,
        matBufferGpu2,
        matHeightOne,
        matHeightTwo,
        matHeightTre,
        matHeightFor,
        matAbsDiff, //Float
        matBigDiffMask,
        matBigDiffMaskFloat,
        matDiffResultDiff,
        pMergeIndexBuffer,
        pCmpTargetBuffer,
        fDiffThreshold,
        enProjDir,
        matMergeResult,
        stream);
    return matMergeResult;
}

}
}
