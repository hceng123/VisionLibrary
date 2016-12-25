#ifndef _AOI_VISION_FITTING_H_
#define _AOI_VISION_FITTING_H_

#include <vector>
#include "opencv2/core.hpp"
#include "VisionStruct.h"

namespace AOI
{
namespace Vision
{

class Fitting
{
public:
    static cv::RotatedRect fitCircle(const std::vector<cv::Point2f> &vecPoints);
    static cv::RotatedRect fitCircle(const ListOfPoint &listPoint);
    static void fitLine(const std::vector<cv::Point2f> &vecPoints, float &fSlope, float &fIntercept, bool reverseFit = false);
    static void fitLine(const ListOfPoint &listPoint, float &fSlope, float &fIntercept, bool reverseFit = false);
    static void fitParallelLine(const std::vector<cv::Point2f> &vecPoints1,
                                const std::vector<cv::Point2f> &vecPoints2,
                                float                          &fSlope,
                                float                          &fIntercept1,
                                float                          &fIntercept2,
                                bool                            reverseFit = false);
    static void fitParallelLine(const ListOfPoint &listPoints1,
                                const ListOfPoint &listPoints2,
                                float             &fSlope,
                                float             &fIntercept1,
                                float             &fIntercept2,
                                bool               reverseFit = false);
    static void fitRect(VectorOfVectorOfPoint &vecVecPoint, 
                        float                 &fSlope1, 
                        float                 &fSlope2, 
                        std::vector<float>    &vecIntercept,
                        bool                   bLineOneReversedFit,
                        bool                   bLineTwoReversedFit);
    static void fitRect(VectorOfListOfPoint &vecListPoint, 
                        float               &fSlope1, 
                        float               &fSlope2, 
                        std::vector<float>  &vecIntercept,
                        bool                 bLineOneReversedFit,
                        bool                 bLineTwoReversedFit);
};

}
}

#endif