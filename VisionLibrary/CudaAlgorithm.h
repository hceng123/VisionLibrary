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
    cv::Mat mergeHeightIntersect(cv::Mat matHeightOne, cv::Mat matNanMaskOne, cv::Mat matHeightTwo, cv::Mat matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir);
};

}
}
