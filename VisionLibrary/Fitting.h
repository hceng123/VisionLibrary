#ifndef _AOI_VISION_FITTING_H_
#define _AOI_VISION_FITTING_H_

#include <vector>
#include "opencv2/core/core.hpp"
#include "VisionStruct.h"

namespace AOI
{
namespace Vision
{

class Fitting
{
public:
    static cv::RotatedRect fitCircle(const std::vector<cv::Point2f> &vecPoints);
    static void fitLine(const std::vector<cv::Point2f> &vecPoints, float &fSlope, float &fIntercept, bool reverseFit = false);
    static void fitParallelLine(const std::vector<cv::Point2f> &vecPoints1,
                                const std::vector<cv::Point2f> &vecPoints2,
                                float                          &fSlope,
                                float                          &fIntercept1,
                                float                          &fIntercept2);
    static void fitRect(VectorOfVectorOfPoint &vecVecPoint, float &fSlope1, float &fSlope2, std::vector<float> &vecIntercept);
};

}
}

#endif