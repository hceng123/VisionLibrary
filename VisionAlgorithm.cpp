#include "VisionAlgorithm.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"

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
    
    Ptr<Feature2D> detector;
    if (ALIGN_ALGORITHM_SIFT == pLearnTmplCmd->nAlgorithm)
        detector = SIFT::create(_constMinHessian);
    else
        detector = SURF::create(_constMinHessian);
    Mat matROI( pLearnTmplCmd->mat, pLearnTmplCmd->rectLrn );
    detector->detectAndCompute ( matROI, pLearnTmplCmd->mask, pLearnTmplRpy->vecKeyPoint, pLearnTmplRpy->matDescritor );
    if ( pLearnTmplRpy->vecKeyPoint.size() > _constLeastFeatures )
        pLearnTmplRpy->nStatus = 0;
    pLearnTmplRpy->ptCenter.x = pLearnTmplCmd->rectLrn.x + pLearnTmplCmd->rectLrn.width / 2.0f;
    pLearnTmplRpy->ptCenter.y = pLearnTmplCmd->rectLrn.y + pLearnTmplCmd->rectLrn.height / 2.0f;
    matROI.copyTo ( pLearnTmplRpy->matTmpl );
    //pLearnTmplRpy->matTmpl = matROI;
    return 0;
}

int VisionAlgorithm::findObject(PR_FindObjCmd *const pFindObjCmd, PR_FindObjRpy *pFindObjRpy)
{
    if ( NULL == pFindObjCmd || NULL == pFindObjRpy )
        return -1;

    if ( pFindObjCmd->mat.empty() )
        return -1;

    if ( pFindObjCmd->vecModelKeyPoint.empty() )
        return -1;

    cv::Mat matSrchROI( pFindObjCmd->mat, pFindObjCmd->rectSrchWindow);
    
    Ptr<Feature2D> detector;
    if (ALIGN_ALGORITHM_SIFT == pFindObjCmd->nAlgorithm)
        detector = SIFT::create(_constMinHessian);
    else
        detector = SURF::create(_constMinHessian);
    
    std::vector<KeyPoint> vecKeypointScene;
    cv::Mat matDescriptorScene;
    Mat mask;
    detector->detectAndCompute ( matSrchROI, mask, vecKeypointScene, matDescriptorScene );

    VectorOfDMatch good_matches;

    Ptr<flann::IndexParams> indexParams = new flann::KDTreeIndexParams;
    Ptr<flann::SearchParams> searchParams = new flann::SearchParams;

    FlannBasedMatcher matcher;

    VectorOfVectorOfDMatch vecVecMatches;
    vecVecMatches.reserve(2000);
    matcher.knnMatch ( pFindObjCmd->matModelDescritor, matDescriptorScene, vecVecMatches, 2 );

    for ( VectorOfVectorOfDMatch::const_iterator it = vecVecMatches.begin();  it != vecVecMatches.end(); ++ it )
    {
        VectorOfDMatch vecDMatch = *it;
        if ( (vecDMatch[0].distance / vecDMatch[1].distance ) < 0.8 )
        {
            good_matches.push_back ( vecDMatch[0] );
        }
    }

#ifdef NEARIST_NEIGHBOUR_MATCH
    std::vector<DMatch> matches;
    matches.reserve(1000);
    matcher.match ( pAlignCmd->matModelDescritor, matDescriptorScene, matches );

    double max_dist = 0; double min_dist = 100.;

    //-- Quick calculation of max and min distances between keypoints
    for ( int i = 0; i < pAlignCmd->matModelDescritor.rows; ++ i )
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )    

    for (int i = 0; i < pAlignCmd->matModelDescritor.rows; ++ i)
    {
        if (matches[i].distance < 3 * min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }    
#endif

    if ( good_matches.size() <= 0 )
        return -1;
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); ++ i )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( pFindObjCmd->vecModelKeyPoint[good_matches[i].queryIdx].pt);
        scene.push_back( vecKeypointScene[good_matches[i].trainIdx].pt);
    }    
    
    //getHomographyMatrixFromMatchedFeatures
    pFindObjRpy->matHomography = cv::findHomography ( obj, scene, CV_LMEDS );
    cv::Mat matOriginalPos ( 3, 1, CV_64F );
    matOriginalPos.at<double>(0, 0) = pFindObjCmd->rectLrn.width / 2.f;
    matOriginalPos.at<double>(1, 0) = pFindObjCmd->rectLrn.height / 2.f;
    matOriginalPos.at<double>(2, 0) = 1.0;

    Mat destPos ( 3, 1, CV_64F );

    destPos = pFindObjRpy->matHomography * matOriginalPos;
    pFindObjRpy->ptObjPos.x = (float) destPos.at<double>(0, 0);
    pFindObjRpy->ptObjPos.y = (float) destPos.at<double>(1, 0);
    pFindObjRpy->szOffset.width = pFindObjRpy->ptObjPos.x - pFindObjCmd->ptExpectedPos.x;
    pFindObjRpy->szOffset.height = pFindObjRpy->ptObjPos.y - pFindObjCmd->ptExpectedPos.y;
    pFindObjRpy->fRotation = (float)( pFindObjRpy->matHomography.at<double>( 1, 0 ) - pFindObjRpy->matHomography.at<double>(0, 1) ) / 2.0f;
    
    return 0;
}

int VisionAlgorithm::align(PR_AlignCmd *const pAlignCmd, PR_AlignRpy *pAlignRpy )
{
     if ( NULL == pAlignCmd || NULL == pAlignRpy )
        return -1;

    if ( pAlignCmd->matInput.empty() )
        return -1;

    if ( pAlignCmd->matTmpl.empty() )
        return -1;

    cv::Mat matAffine ( 2, 3, CV_32F );
    matAffine.setTo(Scalar(0));
    matAffine.at<float>(0,0) = 1.0;
    matAffine.at<float>(1,1) = 1.0;
   
    cv::Mat matSrchROI( pAlignCmd->matInput, pAlignCmd->rectSrchWindow );
    cv::findTransformECC ( pAlignCmd->matTmpl, matSrchROI, matAffine, MOTION_AFFINE );
    pAlignRpy->matAffine = matAffine;
    pAlignRpy->szOffset.width = matAffine.at<float> ( 0, 2 );
    pAlignRpy->szOffset.height = matAffine.at<float> ( 1, 2 );
    pAlignRpy->fRotation = ( matAffine.at<float>( 1, 0 ) - matAffine.at<float>(0, 1) ) / 2.0f;
    pAlignRpy->ptObjPos.x = pAlignCmd->ptOriginalPos.x + pAlignRpy->szOffset.width;
    pAlignRpy->ptObjPos.y = pAlignCmd->ptOriginalPos.y + pAlignRpy->szOffset.height;

    return 0;
}

}
}