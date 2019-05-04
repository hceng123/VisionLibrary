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

public:
    static bool initCuda();
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
    static cv::Mat mergeHeightIntersect(cv::Mat matHeightOne, cv::Mat matNanMaskOne, cv::Mat matHeightTwo, cv::Mat matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir);
};

}
}
