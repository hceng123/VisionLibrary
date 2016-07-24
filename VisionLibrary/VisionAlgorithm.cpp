#include "VisionAlgorithm.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2\highgui.hpp"
#include "TimeLog.h"
#include "logcase.h"
#include "boost/filesystem.hpp"
#include "RecordManager.h"
#include "Config.h"

using namespace cv::xfeatures2d;
using namespace std;
namespace bfs = boost::filesystem;

namespace AOI
{
namespace Vision
{

#define MARK_FUNCTION_START_TIME    __int64 functionStart = _stopWatch.AbsNow()
#define MARK_FUNCTION_END_TIME      TimeLog::GetInstance()->addTimeLog( __FUNCTION__, _stopWatch.AbsNow() - functionStart )

VisionAlgorithm::VisionAlgorithm()
{
}

/*static*/ shared_ptr<VisionAlgorithm> VisionAlgorithm::create()
{
    return make_shared<VisionAlgorithm>();
}

VisionStatus VisionAlgorithm::lrnTmpl(PR_LEARN_TMPL_CMD * const pLearnTmplCmd, PR_LEARN_TMPL_RPY *pLearnTmplRpy)
{
    if ( NULL == pLearnTmplCmd || NULL == pLearnTmplRpy )
        return VisionStatus::INVALID_PARAM;

    if ( pLearnTmplCmd->mat.empty() )
        return VisionStatus::INVALID_PARAM;

    LogCaseLrnTmpl logCase(Config::GetInstance()->getLogCaseDir());
    logCase.WriteCmd(pLearnTmplCmd);

    cv::Ptr<cv::Feature2D> detector;
    if ( PR_ALIGN_ALGORITHM::SIFT == pLearnTmplCmd->enAlgorithm)
        detector = SIFT::create (_constMinHessian, _constOctave, _constOctaveLayer );
    else
        detector = SURF::create (_constMinHessian, _constOctave, _constOctaveLayer );
    cv::Mat matROI( pLearnTmplCmd->mat, pLearnTmplCmd->rectLrn );
    detector->detectAndCompute ( matROI, pLearnTmplCmd->mask, pLearnTmplRpy->vecKeyPoint, pLearnTmplRpy->matDescritor );
    if ( pLearnTmplRpy->vecKeyPoint.size() > _constLeastFeatures )
        pLearnTmplRpy->nStatus = static_cast<int>(VisionStatus::OK);
    pLearnTmplRpy->ptCenter.x = pLearnTmplCmd->rectLrn.x + pLearnTmplCmd->rectLrn.width / 2.0f;
    pLearnTmplRpy->ptCenter.y = pLearnTmplCmd->rectLrn.y + pLearnTmplCmd->rectLrn.height / 2.0f;
    matROI.copyTo ( pLearnTmplRpy->matTmpl );
    _writeLrnTmplRecord(pLearnTmplRpy);

    logCase.WriteRpy(pLearnTmplRpy);
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::srchTmpl(PR_SRCH_TMPL_CMD *const pFindObjCmd, PR_SRCH_TMPL_RPY *pFindObjRpy)
{
    if ( NULL == pFindObjCmd || NULL == pFindObjRpy )
        return VisionStatus::INVALID_PARAM;

    if ( pFindObjCmd->mat.empty() )
        return VisionStatus::INVALID_PARAM;

    if ( pFindObjCmd->nRecordID <= 0 )
        return VisionStatus::INVALID_PARAM;

    TmplRecord *pRecord = (TmplRecord *)RecordManager::getInstance()->get(pFindObjCmd->nRecordID);

    //__int64 functionStart = _stopWatch.AbsNow();
    MARK_FUNCTION_START_TIME;
    cv::Mat matSrchROI ( pFindObjCmd->mat, pFindObjCmd->rectSrchWindow );
    
    cv::Ptr<cv::Feature2D> detector;
    if (PR_ALIGN_ALGORITHM::SIFT == pFindObjCmd->enAlgorithm)
        detector = SIFT::create(_constMinHessian, _constOctave, _constOctaveLayer );
    else
        detector = SURF::create(_constMinHessian, _constOctave, _constOctaveLayer );
    
    std::vector<cv::KeyPoint> vecKeypointScene;
    cv::Mat matDescriptorScene;
    cv::Mat mask;
    
    _stopWatch.Start();
    detector->detectAndCompute ( matSrchROI, mask, vecKeypointScene, matDescriptorScene );
    TimeLog::GetInstance()->addTimeLog("detectAndCompute");

    VectorOfDMatch good_matches;

    cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::KDTreeIndexParams;
    cv::Ptr<cv::flann::SearchParams> searchParams = new cv::flann::SearchParams;

    cv::FlannBasedMatcher matcher;

    VectorOfVectorOfDMatch vecVecMatches;
    vecVecMatches.reserve(2000);
	
    _stopWatch.Start();
    matcher.knnMatch ( pRecord->getModelDescriptor(), matDescriptorScene, vecVecMatches, 2 );
	TimeLog::GetInstance()->addTimeLog("knnMatch");

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
        return VisionStatus::SRCH_TMPL_FAIL;
    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); ++ i )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( pRecord->getModelKeyPoint()[good_matches[i].queryIdx].pt);
        scene.push_back( vecKeypointScene[good_matches[i].trainIdx].pt);
    }    
    
    pFindObjRpy->matHomography = cv::findHomography ( obj, scene, CV_LMEDS );
    TimeLog::GetInstance()->addTimeLog("cv::findHomography");

	pFindObjRpy->matHomography.at<double>(0, 2) += pFindObjCmd->rectSrchWindow.x;
	pFindObjRpy->matHomography.at<double>(1, 2) += pFindObjCmd->rectSrchWindow.y;

    cv::Mat matOriginalPos ( 3, 1, CV_64F );
    matOriginalPos.at<double>(0, 0) = pFindObjCmd->rectLrn.width / 2.f;
    matOriginalPos.at<double>(1, 0) = pFindObjCmd->rectLrn.height / 2.f;
    matOriginalPos.at<double>(2, 0) = 1.0;

    cv::Mat destPos ( 3, 1, CV_64F );

    destPos = pFindObjRpy->matHomography * matOriginalPos;
    pFindObjRpy->ptObjPos.x = (float) destPos.at<double>(0, 0);
    pFindObjRpy->ptObjPos.y = (float) destPos.at<double>(1, 0);
    pFindObjRpy->szOffset.width = pFindObjRpy->ptObjPos.x - pFindObjCmd->ptExpectedPos.x;
    pFindObjRpy->szOffset.height = pFindObjRpy->ptObjPos.y - pFindObjCmd->ptExpectedPos.y;
	float fRotationValue = (float)(pFindObjRpy->matHomography.at<double>(1, 0) - pFindObjRpy->matHomography.at<double>(0, 1)) / 2.0f;
	pFindObjRpy->fRotation = (float) RADIAN_2_DEGREE ( asin(fRotationValue) );
    
    MARK_FUNCTION_END_TIME;
    return VisionStatus::OK;
}

namespace
{
    int _calculateVerticalProjection(const cv::Mat &mat, cv::Mat &output)
    {
        if (mat.empty())
            return 0;

        output.create(1, mat.rows, CV_32FC1);
        for (int row = 0; row < mat.rows; ++row)
        {
            auto whitePixelCount = 0.f;
            for (int col = 0; col < mat.cols; ++col)
            {
                auto value = mat.at<unsigned char>(row, col);
                if (value > 0)
                    ++whitePixelCount;
            }
            output.at<float>(0, row) = whitePixelCount;
        }
        return 0;
    }

    int _calculateHorizontalProjection(const cv::Mat &mat, cv::Mat &output)
    {
        if (mat.empty())
            return 0;
        cv::Mat matTranspose;
        cv::transpose(mat, matTranspose);
        _calculateVerticalProjection(matTranspose, output);
        return 0;
    }

    template<typename T>
    void _expSmooth(const cv::Mat &matInput, cv::Mat &matOutput, float alpha)
    {
        matInput.copyTo(matOutput);
        for (int col = 1; col < matInput.cols; ++col)
        {
            matOutput.at<T>(0, col) = alpha * matInput.at<T>(0, col) + (1.f - alpha) * matOutput.at<T>(0, col - 1);
        }
    }

    int _findEdge(const cv::Mat &matInput, int nObjectLength, int &nObjectEdgeResult)
    {
        if (matInput.cols < nObjectLength)
            return -1;
        auto fMaxSumOfProjection = 0.f;
        nObjectEdgeResult = 0;
        for (int nObjectStart = 0; nObjectStart < matInput.cols - nObjectLength; ++nObjectStart)
        {
            auto fSumOfProjection = 0.f;
            for (int nIndex = nObjectStart; nIndex < nObjectStart + nObjectLength; ++nIndex)
                fSumOfProjection += matInput.at<float>(0, nIndex);
            if (fSumOfProjection > fMaxSumOfProjection) {
                fMaxSumOfProjection = fSumOfProjection;
                nObjectEdgeResult = nObjectStart;
            }
        }
        return 0;
    }

    int _findEdgeReverse(const cv::Mat &matInput, int nObjectLength, int &nObjectEdgeResult)
    {
        if (matInput.cols < nObjectLength)
            return -1;
        auto fMaxSumOfProjection = 0.f;
        nObjectEdgeResult = 0;
        for (int nObjectStart = 0; nObjectStart < matInput.cols - nObjectLength; ++nObjectStart)
        {
            auto fSumOfProjection = 0.f;
            for (int nIndex = nObjectStart; nIndex < nObjectStart + nObjectLength; ++nIndex)
                fSumOfProjection += matInput.at<float>(0, nIndex);
            if (fSumOfProjection > fMaxSumOfProjection) {
                fMaxSumOfProjection = fSumOfProjection;
                nObjectEdgeResult = nObjectStart;
            }
        }
        return 0;
    }

    struct AOI_COUNTOUR {
        Point2f     ptCtr;
        float       fArea;
    };

    float distanceOf2Point(const cv::Point2f &pt1, const cv::Point2f &pt2)
    {
        float fOffsetX = (float)pt1.x - pt2.x;
        float fOffsetY = (float)pt1.y - pt2.y;
        float fDistance = sqrt(fOffsetX * fOffsetX + fOffsetY * fOffsetY);
        return fDistance;
    }

    bool _isSimilarContourExist(const std::vector<AOI_COUNTOUR> &vecAoiCountour, const AOI_COUNTOUR &stCountourInput)
    {
        for (auto stContour : vecAoiCountour)
        {
            if (distanceOf2Point(stContour.ptCtr, stCountourInput.ptCtr) < 3
                && fabs(stContour.fArea - stCountourInput.fArea) < 10)
                return true;
        }
        return false;
    }
}

VisionStatus VisionAlgorithm::inspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy)
{
    if ( pstInspDeviceCmd == nullptr || pstInspDeivceRpy == nullptr )
        return VisionStatus::INVALID_PARAM;
    if ( pstInspDeviceCmd->matInput.empty() )
        return VisionStatus::INVALID_PARAM;

    for (Int32 i = 0; i < pstInspDeviceCmd->nDeviceCount; ++i)   {
        cv::Mat matROI(pstInspDeviceCmd->matInput, pstInspDeviceCmd->astDeviceInfo[i].rectSrchWindow);
        cv::Mat matGray, matThreshold, matHorizontalProjection, matVerticalProjection;
        Int32 nLeftEdge = 0, nRightEdge = 0, nTopEdge = 0, nBottomEdge = 0;

        cv::cvtColor( matROI, matGray, CV_BGR2GRAY);

        cv::threshold(matGray, matThreshold, pstInspDeviceCmd->nElectrodeThreshold, 255, cv::THRESH_BINARY);

        _calculateHorizontalProjection(matThreshold, matHorizontalProjection);
        _expSmooth<float>(matHorizontalProjection, matHorizontalProjection, _constExpSmoothRatio);
        _findEdge(matHorizontalProjection, static_cast<Int32>(pstInspDeviceCmd->astDeviceInfo[i].stSize.width), nLeftEdge);

        _calculateVerticalProjection(matThreshold, matVerticalProjection);
        _expSmooth<float>(matVerticalProjection, matVerticalProjection, _constExpSmoothRatio );
        _findEdge(matHorizontalProjection, static_cast<Int32>(pstInspDeviceCmd->astDeviceInfo[i].stSize.height), nTopEdge);

    }
    return VisionStatus::OK;
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
    matAffine.setTo(cv::Scalar(0));
    matAffine.at<float>(0,0) = 1.0;
	matAffine.at<float>(1, 1) = 1.0;
   
    cv::Mat matSrchROI( pAlignCmd->matInput, pAlignCmd->rectSrchWindow );
    cv::findTransformECC ( pAlignCmd->matTmpl, matSrchROI, matAffine, cv::MOTION_AFFINE );	//The result only support 32F in Opencv
	matAffine.at<float>(0, 2) += pAlignCmd->rectSrchWindow.x;
	matAffine.at<float>(1, 2) += pAlignCmd->rectSrchWindow.y;

    pAlignRpy->matAffine = matAffine;

	cv::Mat matOriginalPos(3, 1, CV_32F);
	matOriginalPos.at<float>(0, 0) = pAlignCmd->rectLrn.width / 2.f;
	matOriginalPos.at<float>(1, 0) = pAlignCmd->rectLrn.height / 2.f;
	matOriginalPos.at<float>(2, 0) = 1.0;

	cv::Mat destPos(3, 1, CV_32F);

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

	cv::Mat matRotation = getRotationMatrix2D ( pInspCmd->ptObjPos, -pInspCmd->fRotation, 1 );
	cv::Mat matRotatedTmpl;
	warpAffine( matTmplFullRange, matRotatedTmpl, matRotation, matTmplFullRange.size() );

	cv::Mat matRotatedMask;
	warpAffine( matMask, matRotatedMask, matRotation, matMask.size() );
	matRotatedMask =  cv::Scalar::all(255) - matRotatedMask;
	cv::imshow("Mask", matRotatedMask );

	pInspCmd->matInsp.setTo(cv::Scalar(0), matRotatedMask );
	cv::imshow("Masked Inspection", pInspCmd->matInsp );

	cv::Mat matCmpResult = pInspCmd->matInsp - matRotatedTmpl;
	cv::Mat matRevsCmdResult = matRotatedTmpl - pInspCmd->matInsp;

	cv::imshow("Rotated template", matRotatedTmpl);

	cv::imshow("Compare result", matCmpResult );
    cv::imshow("Compare result Reverse", matRevsCmdResult );
	
	pInspRpy->n16NDefect = 0;
	_findBlob(matCmpResult, matRevsCmdResult, pInspCmd, pInspRpy);
	_findLine(matCmpResult, pInspCmd, pInspRpy );

	cv::waitKey(0);

	return 0;
}

int VisionAlgorithm::_findBlob(const cv::Mat &mat, const cv::Mat &matRevs, PR_InspCmd *const pInspCmd, PR_InspRpy *pInspRpy)
{
	cv::Mat im_with_keypoints(mat);
	mat.copyTo(im_with_keypoints);

	for (short nIndex = 0; nIndex < pInspCmd->u16NumOfDefectCriteria; ++nIndex)
	{
		PR_DefectCriteria stDefectCriteria = pInspCmd->astDefectCriteria[nIndex];
		if ( stDefectCriteria.fArea <= 0.f )
			continue;

		cv::SimpleBlobDetector::Params params;

		// Change thresholds
		params.blobColor = 0;

		params.minThreshold = stDefectCriteria.ubContrast;
		params.maxThreshold = 255.f;

		// Filter by Area.
		params.filterByArea = true;
		params.minArea = stDefectCriteria.fArea;

		// Filter by Circularity
		//params.filterByCircularity = true;
		//params.minCircularity = 0.1;

		// Filter by Convexity
		params.filterByConvexity = false;
		//params.minConvexity = 0.87;

		// Filter by Inertia
		params.filterByInertia = true;
		params.minInertiaRatio = 0.05f;

        params.filterByColor = false;

		// Set up detector with params
		cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

		vector<cv::Mat> vecInput;
		if ( PR_DEFECT_ATTRIBUTE::BRIGHT == stDefectCriteria.enAttribute )	{
			vecInput.push_back(mat);
		}else if ( PR_DEFECT_ATTRIBUTE::DARK == stDefectCriteria.enAttribute )	{
			vecInput.push_back(matRevs);
		}else if(PR_DEFECT_ATTRIBUTE::BOTH == stDefectCriteria.enAttribute ){
            vecInput.push_back(mat);
            vecInput.push_back(matRevs);
		}	
		
		VectorOfVectorKeyPoint keyPoints;
		detector->detect ( vecInput, keyPoints, pInspCmd->matMask );

        //VectorOfKeyPoint vecKeyPoint = keyPoints[0];
        for (const auto &vecKeyPoint : keyPoints)    {
            if (vecKeyPoint.size() > 0 && pInspRpy->n16NDefect < MAX_NUM_OF_DEFECT_RESULT)	{
                for ( const auto &keyPoint : vecKeyPoint )
                {
                    pInspRpy->astDefect[pInspRpy->n16NDefect].enType = PR_DEFECT_TYPE::BLOB;
                    float fRadius = keyPoint.size / 2;
                    pInspRpy->astDefect[pInspRpy->n16NDefect].fRadius = fRadius;
                    pInspRpy->astDefect[pInspRpy->n16NDefect].fArea = (float)CV_PI * fRadius * fRadius; //Area = PI * R * R
                    ++pInspRpy->n16NDefect;
                }
                drawKeypoints( im_with_keypoints, vecKeyPoint, im_with_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );	
            }
        }			
	}

	cv::imshow("Find blob result", im_with_keypoints);
	cv::waitKey(0);
	return 0;
}

int VisionAlgorithm::_findLine(const cv::Mat &mat, PR_InspCmd *const pInspCmd, PR_InspRpy *pInspRpy)
{
    MARK_FUNCTION_START_TIME;
	for (short nIndex = 0; nIndex < pInspCmd->u16NumOfDefectCriteria; ++nIndex)
	{
		PR_DefectCriteria stDefectCriteria = pInspCmd->astDefectCriteria[nIndex];
		if ( stDefectCriteria.fLength <= 0.f )
			continue;

		cv::Mat matThreshold;
		cv::threshold ( mat, matThreshold, stDefectCriteria.ubContrast, 255, cv::THRESH_BINARY );

		cv::Mat matSobelX, matSobelY;
		Sobel ( matThreshold, matSobelX, mat.depth(), 1, 0 );
		Sobel ( matThreshold, matSobelY, mat.depth(), 0, 1 );

		short depth = matSobelX.depth();

		cv::Mat matSobelResult(mat.size(), CV_8U);
		for (short row = 0; row < mat.rows; ++row)
		{
			for (short col = 0; col < mat.cols; ++col)
			{
				unsigned char x = (matSobelX.at<unsigned char>(row, col));
				unsigned char y = (matSobelY.at<unsigned char>(row, col));
				matSobelResult.at<unsigned char>(row, col) = x + y;
			}
		}

        if (PR_DEBUG_MODE::SHOW_IMAGE == _enDebugMode)
            cv::imshow("Sobel Result", matSobelResult);

		cv::Mat color_dst;
		cvtColor(matSobelResult, color_dst, CV_GRAY2BGR);

		vector<cv::Vec4i> lines;
		HoughLinesP ( matSobelResult, lines, 1, CV_PI / 180, 30, stDefectCriteria.fLength, 10 );

		//_mergeLines ( )
		vector<PR_Line2f> vecLines, vecLinesAfterMerge;
		for (size_t i = 0; i < lines.size(); i++)
		{
			cv::Point2f pt1 ( (float)lines[i][0], (float)lines[i][1] );
			cv::Point2f pt2 ( (float)lines[i][2], (float)lines[i][3] );
			PR_Line2f line(pt1, pt2 );
			vecLines.push_back(line);
		}
		_mergeLines ( vecLines, vecLinesAfterMerge );
		for(const auto it:vecLinesAfterMerge)
		{
			line(color_dst, it.pt1, it.pt2, cv::Scalar(0, 0, 255), 3, 8);
			if ( pInspRpy->n16NDefect < MAX_NUM_OF_DEFECT_RESULT )	{
				pInspRpy->astDefect[pInspRpy->n16NDefect].enType = PR_DEFECT_TYPE::LINE;
				pInspRpy->astDefect[pInspRpy->n16NDefect].fLength = distanceOf2Point(it.pt1, it.pt2);
				++pInspRpy->n16NDefect;
			}
		}
        if (PR_DEBUG_MODE::SHOW_IMAGE == _enDebugMode)
            cv::imshow("Detected Lines", color_dst);
	}
    MARK_FUNCTION_END_TIME;
	return 0;
}

/*static*/ float VisionAlgorithm::distanceOf2Point(const cv::Point &pt1, const cv::Point &pt2)
{
    float fOffsetX = (float)pt1.x - pt2.x;
    float fOffsetY = (float)pt1.y - pt2.y;
    float fDistance = sqrt ( fOffsetX * fOffsetX + fOffsetY * fOffsetY );
    return fDistance;
}

/*static*/ float VisionAlgorithm::distanceOf2Point(const cv::Point2f &pt1, const cv::Point2f &pt2)
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
	cv::Point2f ptCrossPointWithLine1, ptCrossPointWithLine2;

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

	vector<cv::Point> vecPoint;
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
	cv::Point2f ptMidOfResult;
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

int VisionAlgorithm::_findLineCrossPoint(const PR_Line2f &line1, const PR_Line2f &line2, cv::Point2f ptResult)
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

VisionStatus VisionAlgorithm::_writeLrnTmplRecord(PR_LEARN_TMPL_RPY *pLearnTmplRpy)
{
    TmplRecord *pRecord = new TmplRecord(PR_RECORD_TYPE::ALIGNMENT);
    pRecord->setModelKeyPoint(pLearnTmplRpy->vecKeyPoint);
    pRecord->setModelDescriptor(pLearnTmplRpy->matDescritor);
    RecordManager::getInstance()->add(pRecord, pLearnTmplRpy->nRecordID);
    return VisionStatus::OK;
}

}
}