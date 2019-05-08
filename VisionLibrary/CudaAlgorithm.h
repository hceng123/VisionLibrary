#pragma once
#include "opencv2/core.hpp"
#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

class CudaAlgorithm
{
    CudaAlgorithm() = delete;
    ~CudaAlgorithm() = delete;
    static float* m_dlp3DBezierSurface[NUM_OF_DLP];

public:
    static bool initCuda();
    static VisionStatus setDlpParams(const PR_SET_DLP_PARAMS_TO_GPU_CMD* const pstCmd, PR_SET_DLP_PARAMS_TO_GPU_RPY* const pstRpy);
    static void floor(cv::cuda::GpuMat &matInOut);
    static float intervalAverage(const cv::cuda::GpuMat& matInput, int interval);
    static float intervalRangeAverage(const cv::cuda::GpuMat& matInput, int interval, float rangeStart, float rangeEnd);
    static void phaseWrapBuffer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matBuffer, float fShift = 0.f);
    static void phaseWrapByRefer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matRef);
    static void phaseWarpNoAutoBase(const cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matResult, float fShift);
    static void phaseCorrectionCmp(cv::cuda::GpuMat& matPhase, const cv::cuda::GpuMat& matPhase1, cv::cuda::GpuMat& matPhaseT, int span);
    static void phaseCorrectionCmp(cv::Mat& matPhase, const cv::Mat& matPhase1, int span);
    static void phaseCorrection(cv::cuda::GpuMat &matPhase, cv::cuda::GpuMat& matPhaseT, int nJumpSpanX, int nJumpSpanY);
    static void phaseCorrection(cv::Mat &matPhase, int nJumpSpanX, int nJumpSpanY);
    static void calculateSurfaceConvert3D(
        const cv::cuda::GpuMat& matPhase,
        cv::cuda::GpuMat& matPhaseT,
        cv::cuda::GpuMat& matBufferGpu,
        const VectorOfFloat& param,
        const int ss,
        const int nDlpNo);

    static void phaseToHeight3D(
        const cv::cuda::GpuMat& zInLine,
        const cv::cuda::GpuMat& matPolyParassGpu,
        const float* d_xxt,
        const int xxtStep,
        const int len,
        const int ss,
        float* d_P3);

    static void calcSumAndConvertMatrix(
        float* d_pInputData,
        int ss,
        cv::cuda::GpuMat& matResult);

    static cv::Mat mergeHeightIntersect(cv::Mat matHeightOne, cv::Mat matNanMaskOne, cv::Mat matHeightTwo, cv::Mat matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir);
};

}
}
