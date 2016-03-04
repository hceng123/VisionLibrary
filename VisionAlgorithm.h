#ifndef _AOI_VISION_ALGORITHM_H
#define _AOI_VISION_ALGORITHM_H

#include "BaseType.h"
#include "opencv2/core/core.hpp"
#include "VisionDataStruct.h"

using namespace cv;

namespace AOI
{
namespace Vision
{

class VisionAlgorithm : private Uncopyable
{
public:
    explicit VisionAlgorithm(const Mat &mat);
    explicit VisionAlgorithm();
    int learn(const Rect &rect, const Mat &mask, int const nAlgorithm);
    int learn(PR_LearnTmplCmd * const pLearnTmplCmd, PR_LearnTmplRpy *pLearnTmplRpy);
    int align(PR_AlignCmd *const pAlignCmd, PR_AlignRpy *pAlignRpy);
protected:
    const int       _constMinHessian    =      40;
    const UInt32    _constLeastFeatures =      3;
    cv::Mat         _mat;
};

}
}
#endif