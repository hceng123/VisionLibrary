#include "VisionAlgorithm.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2\xfeatures2d\nonfree.hpp"

using namespace cv::xfeatures2d;

namespace AOI
{
namespace Vision
{

VisionAlgorithm::VisionAlgorithm(const Mat &mat)
{
    _mat = mat;
}

int VisionAlgorithm::learn(const Rect &rect, const Mat &mask, int nAlgorithm)
{
    if ( _mat.empty() )
        return -1;
    
    Ptr<SURF> detector = SURF::create(_constMinHessian);
    Mat matROI(_mat, rect);
    std::vector<KeyPoint> keypoints;
    Mat matDescritor;
    detector->detectAndCompute ( matROI, mask, keypoints, matDescritor );
    return 0;
}

}
}