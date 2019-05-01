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
    //static cv::cuda::Stream m_cudaStreams[NUM_OF_DLP];

public:
    static void phaseCorrectionCmp(cv::cuda::GpuMat& matPhase, const cv::cuda::GpuMat& matPhase1, int span);
    static void phaseCorrectionCmp(cv::Mat& matPhase, const cv::Mat& matPhase1, int span);
    static void phaseCorrection(cv::cuda::GpuMat &matPhase, int nJumpSpanX, int nJumpSpanY);
    static void phaseCorrection(cv::Mat &matPhase, int nJumpSpanX, int nJumpSpanY);
    static cv::Mat mergeHeightIntersect(cv::Mat matHeightOne, cv::Mat matNanMaskOne, cv::Mat matHeightTwo, cv::Mat matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir);
};

}
}
