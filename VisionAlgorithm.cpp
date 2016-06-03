#include "VisionAlgorithm.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2\highgui.hpp"
#include <vector>
#include "StopWatch.h"
#include "VisionStatus.h"

using namespace cv::xfeatures2d;
using namespace std;

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

    cv::Mat matSrchROI ( pFindObjCmd->mat, pFindObjCmd->rectSrchWindow );
    
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

	CStopWatch stopWatch;
    matcher.knnMatch ( pFindObjCmd->matModelDescritor, matDescriptorScene, vecVecMatches, 2 );
	//printf("knnMatch time consume %d us \n", stopWatch.NowInMicro() );

	double max_dist = 0; double min_dist = 100.;
	int nScoreCount = 0;
    for ( VectorOfVectorOfDMatch::const_iterator it = vecVecMatches.begin();  it != vecVecMatches.end(); ++ it )
    {
        VectorOfDMatch vecDMatch = *it;
        if ( (vecDMatch[0].distance / vecDMatch[1].distance ) < 0.8 )
        {
            good_matches.push_back ( vecDMatch[0] );
        }

		double dist = vecDMatch[0].distance;
		if (dist < 0.5)
			++nScoreCount;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
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

	pFindObjRpy->fMatchScore = (float)nScoreCount * 100.f / vecVecMatches.size();

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
	pFindObjRpy->matHomography.at<double>(0, 2) += pFindObjCmd->rectSrchWindow.x;
	pFindObjRpy->matHomography.at<double>(1, 2) += pFindObjCmd->rectSrchWindow.y;

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
	float fRotationValue = (float)(pFindObjRpy->matHomography.at<double>(1, 0) - pFindObjRpy->matHomography.at<double>(0, 1)) / 2.0f;
	pFindObjRpy->fRotation = (float) RADIAN_2_DEGREE ( asin(fRotationValue) );
    
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
	matAffine.at<float>(1, 1) = 1.0;
   
    cv::Mat matSrchROI( pAlignCmd->matInput, pAlignCmd->rectSrchWindow );
    cv::findTransformECC ( pAlignCmd->matTmpl, matSrchROI, matAffine, MOTION_AFFINE );	//The result only support 32F in Opencv
	matAffine.at<float>(0, 2) += pAlignCmd->rectSrchWindow.x;
	matAffine.at<float>(1, 2) += pAlignCmd->rectSrchWindow.y;

    pAlignRpy->matAffine = matAffine;

	cv::Mat matOriginalPos(3, 1, CV_32F);
	matOriginalPos.at<float>(0, 0) = pAlignCmd->rectLrn.width / 2.f;
	matOriginalPos.at<float>(1, 0) = pAlignCmd->rectLrn.height / 2.f;
	matOriginalPos.at<float>(2, 0) = 1.0;

	Mat destPos(3, 1, CV_32F);

	destPos = pAlignRpy->matAffine * matOriginalPos;
	pAlignRpy->ptObjPos.x = destPos.at<float>(0, 0);
	pAlignRpy->ptObjPos.y = destPos.at<float>(1, 0);

	pAlignRpy->szOffset.width = pAlignRpy->ptObjPos.x - pAlignCmd->ptExpectedPos.x;
	pAlignRpy->szOffset.height = pAlignRpy->ptObjPos.y - pAlignCmd->ptExpectedPos.y;
	
	pAlignRpy->szOffset.width = pAlignRpy->ptObjPos.x - pAlignCmd->ptExpectedPos.x;
	pAlignRpy->szOffset.height = pAlignRpy->ptObjPos.y - pAlignCmd->ptExpectedPos.y;
	float fRotationValue = ( matAffine.at<float>(1, 0) - matAffine.at<float>(0, 1) ) / 2.0f;
	pAlignRpy->fRotation = (float) RADIAN_2_DEGREE(asin(fRotationValue));

    return 0;
}

int VisionAlgorithm::inspect(PR_InspCmd *const pInspCmd, PR_InspRpy *pInspRpy)
{
	if ( NULL == pInspCmd || NULL == pInspRpy )
		return -1;

	cv::Mat matTmplROI( pInspCmd->matTmpl, pInspCmd->rectLrn );

	cv::Mat matTmplFullRange ( pInspCmd->matInsp.size(), pInspCmd->matInsp.type() );
	matTmplFullRange.setTo(cv::Scalar(0));

	cv::Mat matMask ( pInspCmd->matInsp.size(), pInspCmd->matInsp.type() );
	matMask.setTo ( cv::Scalar(0) );
	
	cv::Rect2f rect2fInspROI(pInspCmd->ptObjPos.x - pInspCmd->rectLrn.width / 2.f, pInspCmd->ptObjPos.y - pInspCmd->rectLrn.height / 2.f,
		pInspCmd->rectLrn.width, pInspCmd->rectLrn.height );
	cv::Mat matInspRoi ( matTmplFullRange, rect2fInspROI );
	matTmplROI.copyTo ( matInspRoi );

	cv::Mat matMaskRoi ( matMask, rect2fInspROI );
	matMaskRoi.setTo ( cv::Scalar(255,255,255) );

	Mat matRotation = getRotationMatrix2D ( pInspCmd->ptObjPos, -pInspCmd->fRotation, 1 );
	Mat matRotatedTmpl;
	warpAffine( matTmplFullRange, matRotatedTmpl, matRotation, matTmplFullRange.size() );

	Mat matRotatedMask;
	warpAffine( matMask, matRotatedMask, matRotation, matMask.size() );
	matRotatedMask =  cv::Scalar::all(255) - matRotatedMask;
	imshow("Mask", matRotatedMask );

	pInspCmd->matInsp.setTo(cv::Scalar(0), matRotatedMask );
	imshow("Masked Inspection", pInspCmd->matInsp );

	cv::Mat matCmpResult = pInspCmd->matInsp - matRotatedTmpl;

	imshow("Rotated template", matRotatedTmpl);

	imshow("Compare result", matCmpResult );
	
	pInspRpy->n16NDefect = 0;
	_findBlob(matCmpResult, pInspCmd, pInspRpy);
	_findLine(matCmpResult, pInspCmd, pInspRpy );

	waitKey(0);

	return 0;
}

int VisionAlgorithm::_findBlob(const Mat &mat, PR_InspCmd *const pInspCmd, PR_InspRpy *pInspRpy)
{
	for (short nIndex = 0; nIndex < pInspCmd->u16NumOfDefectCriteria; ++nIndex)
	{
		PR_DefectCriteria stDefectCriteria = pInspCmd->astDefectCriteria[nIndex];
		if ( stDefectCriteria.fArea <= 0.f )
			continue;

		SimpleBlobDetector::Params params;

		// Change thresholds
		params.blobColor = 0;

		params.minThreshold = 0;
		params.maxThreshold = stDefectCriteria.ubContrast;

		// Filter by Area.
		params.filterByArea = true;
		params.minArea = stDefectCriteria.fArea;

		// Filter by Circularity
		//params.filterByCircularity = true;
		//params.minCircularity = 0.1;

		// Filter by Convexity
		//params.filterByConvexity = true;
		//params.minConvexity = 0.87;

		// Filter by Inertia
		//params.filterByInertia = true;
		//params.minInertiaRatio = 0.01;

		// Set up detector with params
		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

		vector<Mat> vecInput;
		vecInput.push_back(mat);

		VectorOfVectorKeyPoint keyPoints;
		detector->detect(vecInput, keyPoints);

		VectorOfKeyPoint vecKeyPoint = keyPoints[0];
		if ( vecKeyPoint.size() > 0 && pInspRpy->n16NDefect < MAX_NUM_OF_DEFECT_RESULT )	{
			for ( VectorOfKeyPoint::const_iterator it = vecKeyPoint.begin(); it != vecKeyPoint.end(); ++ it )
			{
				pInspRpy->astDefect[pInspRpy->n16NDefect].n16Type = PR_DEFECT_BLOB;
				float fRadius = it->size / 2;
				pInspRpy->astDefect[pInspRpy->n16NDefect].fRadius = fRadius;
				pInspRpy->astDefect[pInspRpy->n16NDefect].fArea = (float)CV_PI * fRadius * fRadius; //Area = PI * R * R
				++ pInspRpy->n16NDefect;
			}
		}
	}

	return 0;
}

int VisionAlgorithm::_findLine(const Mat &mat, PR_InspCmd *const pInspCmd, PR_InspRpy *pInspRpy)
{
	for (short nIndex = 0; nIndex < pInspCmd->u16NumOfDefectCriteria; ++nIndex)
	{
		PR_DefectCriteria stDefectCriteria = pInspCmd->astDefectCriteria[nIndex];
		if ( stDefectCriteria.fLength <= 0.f )
			continue;

		cv::Mat matThreshold;
		cv::threshold ( mat, matThreshold, stDefectCriteria.ubContrast, 255, THRESH_BINARY );

		Mat matSobelX, matSobelY;
		Sobel ( matThreshold, matSobelX, mat.depth(), 1, 0 );
		Sobel ( matThreshold, matSobelY, mat.depth(), 0, 1 );

		short depth = matSobelX.depth();

		Mat matSobelResult(mat.size(), CV_8U);
		for (short row = 0; row < mat.rows; ++row)
		{
			for (short col = 0; col < mat.cols; ++col)
			{
				unsigned char x = (matSobelX.at<unsigned char>(row, col));
				unsigned char y = (matSobelY.at<unsigned char>(row, col));
				matSobelResult.at<unsigned char>(row, col) = x + y;
			}
		}

		imshow("Sobel Result", matSobelResult);
		cv::Mat color_dst;
		cvtColor(matSobelResult, color_dst, CV_GRAY2BGR);

		vector<Vec4i> lines;
		HoughLinesP ( matSobelResult, lines, 1, CV_PI / 180, 30, stDefectCriteria.fLength, 10 );

		//_mergeLines ( )
		vector<PR_Line2f> vecLines, vecLinesAfterMerge;
		for (size_t i = 0; i < lines.size(); i++)
		{
			Point2f pt1 ( (float)lines[i][0], (float)lines[i][1] );
			Point2f pt2 ( (float)lines[i][2], (float)lines[i][3] );
			PR_Line2f line(pt1, pt2 );
			vecLines.push_back(line);
			//line(color_dst, pt1, pt2, Scalar(0, 0, 255), 1, 8);

			//if ( pInspRpy->n16NDefect < MAX_NUM_OF_DEFECT_RESULT )	{
			//	pInspRpy->astDefect[pInspRpy->n16NDefect].n16Type = PR_DEFECT_LINE;
			//	pInspRpy->astDefect[pInspRpy->n16NDefect].fLength = distanceOf2Point (pt1, pt2);
			//	++pInspRpy->n16NDefect;
			//}
		}
		_mergeLines ( vecLines, vecLinesAfterMerge );
		for(const auto it:vecLinesAfterMerge)
		{
			line(color_dst, it.pt1, it.pt2, Scalar(0, 0, 255), 1, 8);
		}
		imshow("Detected Lines", color_dst);
		waitKey(0);
	}
	return 0;
}

float VisionAlgorithm::distanceOf2Point(const cv::Point &pt1, const cv::Point &pt2)
{
    float fOffsetX = (float)pt1.x - pt2.x;
    float fOffsetY = (float)pt1.y - pt2.y;
    float fDistance = sqrt ( fOffsetX * fOffsetX + fOffsetY * fOffsetY );
    return fDistance;
}

float VisionAlgorithm::distanceOf2Point(const cv::Point2f &pt1, const cv::Point2f &pt2)
{
    float fOffsetX = (float)pt1.x - pt2.x;
    float fOffsetY = (float)pt1.y - pt2.y;
    float fDistance = sqrt ( fOffsetX * fOffsetX + fOffsetY * fOffsetY );
    return fDistance;
}


int VisionAlgorithm::_merge2Line(const PR_Line2f &line1, const PR_Line2f &line2, PR_Line2f &lineResult)
{
	float fLineSlope1 = ( line1.pt2.y - line1.pt1.y ) / ( line1.pt2.x - line1.pt1.x );
	float fLineSlope2 = ( line2.pt2.y - line2.pt1.y ) / ( line2.pt2.x - line2.pt1.x );
	if ( fabs ( fLineSlope1 - fLineSlope2 ) > PARALLEL_LINE_SLOPE_DIFF_LMT )	{
		printf("Can not merge lines not parallel");
		return -1;
	}

	float fLineCrossWithY1 = fLineSlope1 * ( - line1.pt1.x) + line1.pt1.y;
	float fLineCrossWithY2 = fLineSlope2 * ( - line2.pt1.x) + line2.pt1.y;

	float fPerpendicularLineSlope =  - 1.f / fLineSlope1;
	Point2f ptCrossPointWithLine1, ptCrossPointWithLine2;

	//Find perpendicular line, get the cross points with two lines, then can get the distance of two lines.
	ptCrossPointWithLine1.x = fLineCrossWithY1 / ( fPerpendicularLineSlope - fLineSlope1 );
	ptCrossPointWithLine1.y = fPerpendicularLineSlope * ptCrossPointWithLine1.x;

	ptCrossPointWithLine2.x = fLineCrossWithY2 / ( fPerpendicularLineSlope - fLineSlope2 );
	ptCrossPointWithLine2.y = fPerpendicularLineSlope * ptCrossPointWithLine2.x;

	float fDistanceOfLine = distanceOf2Point ( ptCrossPointWithLine1, ptCrossPointWithLine2 );
	if ( fDistanceOfLine > PARALLEL_LINE_MERGE_DIST_LMT ){
		printf("Distance of the lines are too far, can not merge");
		return -1;
	}

	vector<Point> vecPoint;
	vecPoint.push_back ( line1.pt1 );
	vecPoint.push_back ( line1.pt2 );
	vecPoint.push_back ( line2.pt1 );
	vecPoint.push_back ( line2.pt2 );

	vector<float> vecPerpendicularLineB;
	//Find the expression of perpendicular lines y = a * x + b
	float fPerpendicularLineB1 = line1.pt1.y - fPerpendicularLineSlope * line1.pt1.x;
	vecPerpendicularLineB.push_back ( fPerpendicularLineB1 );
	float fPerpendicularLineB2 = line1.pt2.y - fPerpendicularLineSlope * line1.pt2.x;
	vecPerpendicularLineB.push_back ( fPerpendicularLineB2 );
	float fPerpendicularLineB3 = line2.pt1.y - fPerpendicularLineSlope * line2.pt1.x;
	vecPerpendicularLineB.push_back ( fPerpendicularLineB3 );
	float fPerpendicularLineB4 = line2.pt2.y - fPerpendicularLineSlope * line2.pt2.x;
	vecPerpendicularLineB.push_back ( fPerpendicularLineB4 );

	int nMaxIndex = 0, nMinIndex = 0;
	float fMaxB = fPerpendicularLineB1;
	float fMinB = fPerpendicularLineB1;
	for ( size_t i = 1; i < vecPerpendicularLineB.size(); ++ i )	{
		if ( vecPerpendicularLineB[i] > fMaxB )	{
			fMaxB = vecPerpendicularLineB[i];
			nMaxIndex = i;
		}

		if ( vecPerpendicularLineB[i] < fMinB )	{
			fMinB = vecPerpendicularLineB[i];
			nMinIndex = i;
		}
	}
	Point2f ptMidOfResult;
	ptMidOfResult.x = ( vecPoint[nMinIndex].x + vecPoint[nMaxIndex].x ) / 2.f;
	ptMidOfResult.y = ( vecPoint[nMinIndex].y + vecPoint[nMaxIndex].y ) / 2.f;
	float fLineLength = distanceOf2Point ( vecPoint[nMinIndex], vecPoint[nMaxIndex] );
	float fAverageSlope = ( fLineSlope1 + fLineSlope2 ) / 2;
	float fHypotenuse = sqrt( 1 + fAverageSlope * fAverageSlope );
	float fSin = fAverageSlope / fHypotenuse;
	float fCos = 1.f / fHypotenuse;

	lineResult.pt1.x = ptMidOfResult.x + fLineLength * fCos / 2.f;
	lineResult.pt1.y = ptMidOfResult.y + fLineLength * fSin / 2.f;
	lineResult.pt2.x = ptMidOfResult.x - fLineLength * fCos / 2.f;
	lineResult.pt2.y = ptMidOfResult.y - fLineLength * fSin / 2.f;
	return static_cast<int>(VisionStatus::OK);
}

int VisionAlgorithm::_mergeLines(const vector<PR_Line2f> &vecLines, vector<PR_Line2f> &vecResultLines)
{
	if ( vecLines.size() < 2 )
		return 0;

	vector<int> vecMerged(vecLines.size(), false);

	//vector<float> listLineSlope;
	//for ( vector<PR_Line>::const_iterator it = vecLines.begin(); it != vecLines.end(); ++ it )	{
	//	PR_Line line = *it;
	//	listLineSlope.push_back ( (float)( line.pt2.y - line.pt1.y ) / ( line.pt2.x - line.pt1.x ) );
	//}
	
	for ( size_t i = 0; i < vecLines.size(); ++ i )	{
		if ( vecMerged[i] )
			continue;

		PR_Line2f stLineResult = vecLines[i];
		for ( size_t j = i + 1; j < vecLines.size(); ++ j )	{
			if ( ! vecMerged[j] )	{
				if (  0 == _merge2Line ( stLineResult, vecLines[j], stLineResult ) )
				{
					vecMerged[i] = vecMerged[j] = true;
				}
			}
		}
		vecResultLines.push_back ( stLineResult );
	}
	return 0;
}

int VisionAlgorithm::_findLineCrossPoint(const PR_Line2f &line1, const PR_Line2f &line2, Point2f ptResult)
{
	float fLineSlope1 = ( line1.pt2.y - line1.pt1.y ) / ( line1.pt2.x - line1.pt1.x );
	float fLineSlope2 = ( line2.pt2.y - line2.pt1.y ) / ( line2.pt2.x - line2.pt1.x );
	if ( fabs ( fLineSlope1 - fLineSlope2 ) < PARALLEL_LINE_SLOPE_DIFF_LMT )	{
		printf("No cross point for parallized lines");
		return -1;
	}
	float fLineCrossWithY1 = fLineSlope1 * ( - line1.pt1.x) + line1.pt1.y;
	float fLineCrossWithY2 = fLineSlope2 * ( - line2.pt1.x) + line2.pt1.y;

	//Line1 can be expressed as y = fLineSlope1 * x + fLineCrossWithY1
	//Line2 can be expressed as y = fLineSlope2 * x + fLineCrossWithY2
	ptResult.x = ( fLineCrossWithY2 - fLineCrossWithY1 ) / (fLineSlope1 - fLineSlope2);
	ptResult.y = fLineSlope1 * ptResult.x + fLineCrossWithY1;
	return 0;
}

}
}