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
    detector->detectAndCompute ( matROI, pLearnTmplCmd->mask, pLearnTmplRpy->vecKeyPoint, pLearnTmplRpy->matDescritor );
    if ( pLearnTmplRpy->vecKeyPoint.size() > _constLeastFeatures )
        pLearnTmplRpy->nStatus = 0;
    pLearnTmplRpy->ptCenter.x = pLearnTmplCmd->rectLrn.x + pLearnTmplCmd->rectLrn.width / 2;
    pLearnTmplRpy->ptCenter.y = pLearnTmplCmd->rectLrn.y + pLearnTmplCmd->rectLrn.height / 2;
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
    Ptr<SURF> detector = SURF::create(_constMinHessian);
    
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

    for ( VectorOfVectorOfDMatch::iterator it = vecVecMatches.begin();  it != vecVecMatches.end(); ++ it )
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

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back( pFindObjCmd->vecModelKeyPoint[good_matches[i].queryIdx].pt);
        scene.push_back( vecKeypointScene[good_matches[i].trainIdx].pt);
    }    
    
    //getHomographyMatrixFromMatchedFeatures
    pFindObjRpy->matHomography = cv::findHomography ( obj, scene, CV_LMEDS);

    return 0;
}

int VisionAlgorithm::align(PR_AlignCmd *const pAlignCmd, PR_AlignRpy *pAlignRpy )
{
     if ( NULL == pAlignCmd || NULL == pAlignRpy )
        return -1;

    if ( pAlignCmd->mat.empty() )
        return -1;

    if ( pAlignCmd->matTmpl.empty() )
        return -1;

    return 0;
}

}
}