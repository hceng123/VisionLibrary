#ifndef _AOI_VISION_ALGORITHM_H
#define _AOI_VISION_ALGORITHM_H

#include "BaseType.h"
#include "opencv2/core/core.hpp"

using namespace cv;

namespace AOI
{
namespace Vision
{

class VisionAlgorithm : private Uncopyable
{
public:
    explicit VisionAlgorithm(const Mat &mat);
    int learn(const Rect &rect, const Mat &mask, int nAlgorithm);
protected:
    const int   _constMinHessian = 40;
    cv::Mat     _mat;
};

}
}
#endif