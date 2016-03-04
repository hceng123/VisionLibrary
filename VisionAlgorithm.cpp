#include "VisionAlgorithm.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv::xfeatures2d;

namespace AOI
{
namespace Vision
{

VisionAlgorithm::VisionAlgorithm(const Mat &mat):_mat(mat)
{
}

VisionAlgorithm::VisionAlgorithm()
{
}

int VisionAlgorithm::learn(const Rect &rect, const Mat &mask, int const nAlgorithm)
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

int VisionAlgorithm::learn(PR_LearnTmplCmd * const pLearnTmplCmd, PR_LearnTmplRpy *pLearnTmplRpy)
{
    if ( NULL == pLearnTmplCmd || NULL == pLearnTmplRpy )
        return -1;

    if ( pLearnTmplCmd->mat.empty() )
        return -1;
    
    Ptr<SURF> detector = SURF::create(_constMinHessian);
    Mat matROI( pLearnTmplCmd->mat, pLearnTmplCmd->rectLrn );
    detector->detectAndCompute ( matROI, pLearnTmplCmd->mask, pLearnTmplRpy->vecKeyPoint, pLearnTmplRpy->vecKeyPoint );
    if ( pLearnTmplRpy->vecKeyPoint.size() > _constLeastFeatures )
        pLearnTmplRpy->nStatus = 0;

    return 0;
}

int VisionAlgorithm::align(PR_AlignCmd *const pAlignCmd, PR_AlignRpy *pAlignRpy)
{
    if ( NULL == pAlignCmd || NULL == pAlignRpy )
        return -1;

    if ( pAlignCmd->mat.empty() )
        return -1;

    if ( pAlignCmd->vecModelKeyPoint.empty() )
        return -1;

    cv::Mat matSrchROI(pAlignCmd->mat, pAlignCmd->rectSrchWindow);
    Ptr<SURF> detector = SURF::create(_constMinHessian);
    
    std::vector<KeyPoint> vecKeypointScene;
    cv::Mat matDescriptorScene;
    detector->detectAndCompute(matSrchROI, NULL, vecKeypointScene, matDescriptorScene);

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match ( pAlignCmd->vecModelKeyPoint, vecKeypointScene, matches );

    double max_dist = 0; double min_dist = 100.;

    //-- Quick calculation of max and min distances between keypoints
    for ( int i = 0; i < pAlignCmd->matModelDescritor.rows; ++ i )
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for (int i = 0; i < pAlignCmd->matModelDescritor.rows; ++ i)
    {
        if (matches[i].distance < 3 * min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (int i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back( pAlignCmd->vecModelKeyPoint[good_matches[i].queryIdx].pt);
        scene.push_back(vecKeypointScene[good_matches[i].trainIdx].pt);
    }
    
    //getHomographyMatrixFromMatchedFeatures
    Mat H = findHomography(obj, scene, CV_RANSAC);

    return 0;
}

}
}