#include "VisionAlgorithm.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "TimeLog.h"
#include "logcase.h"
#include "boost/filesystem.hpp"
#include "RecordManager.h"
#include "Config.h"
#include "CalcUtils.h"
#include "Log.h"
#include "FileUtils.h"
#include "Fitting.h"
#include <iostream>
#include <limits>
#include <algorithm>

using namespace cv::xfeatures2d;
namespace bfs = boost::filesystem;

namespace AOI
{
namespace Vision
{

#define MARK_FUNCTION_START_TIME    CStopWatch      stopWatch; __int64 functionStart = stopWatch.AbsNow()
#define MARK_FUNCTION_END_TIME      TimeLog::GetInstance()->addTimeLog( __FUNCTION__, stopWatch.AbsNow() - functionStart )

#define SETUP_LOGCASE(classname) \
std::unique_ptr<classname> pLogCase; \
if ( ! bReplay )    {   \
    pLogCase = std::make_unique<classname>( Config::GetInstance()->getLogCaseDir() );    \
    if ( PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode() ) \
        pLogCase->WriteCmd ( pstCmd );  \
}

#define FINISH_LOGCASE \
if ( ! bReplay )    {   \
    if ( PR_DEBUG_MODE::LOG_FAIL_CASE == Config::GetInstance()->getDebugMode() && pstRpy->enStatus != VisionStatus::OK )    {   \
        pLogCase->WriteCmd ( pstCmd );  \
        pLogCase->WriteRpy ( pstRpy );  \
    }   \
    if ( PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode() ) \
        pLogCase->WriteRpy ( pstRpy );  \
}

/*static*/ OcrTesseractPtr VisionAlgorithm::_ptrOcrTesseract;
/*static*/ const cv::Scalar VisionAlgorithm::_constRedScalar   = cv::Scalar(0, 0, 255);
/*static*/ const cv::Scalar VisionAlgorithm::_constBlueScalar  = cv::Scalar(255, 0, 0);
/*static*/ const cv::Scalar VisionAlgorithm::_constGreenScalar = cv::Scalar(0, 255, 0);
/*static*/ const cv::Scalar VisionAlgorithm::_constYellowScalar(0, 255, 255);
/*static*/ const String VisionAlgorithm::_strRecordLogPrefix   = "tmplDir.";
/*static*/ const float VisionAlgorithm::_constExpSmoothRatio   = 0.3f;

VisionAlgorithm::VisionAlgorithm()
{
}

/*static*/ std::unique_ptr<VisionAlgorithm> VisionAlgorithm::create()
{
    return std::make_unique<VisionAlgorithm>();
}

/*static*/ VisionStatus VisionAlgorithm::lrnObj(const PR_LRN_OBJ_CMD *const pstCmd, PR_LRN_OBJ_RPY *const pstRpy, bool bReplay /*= false*/ )
{
    assert(NULL != pstCmd || NULL != pstRpy);

    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    char charrMsg[1000];
    if (pstCmd->rectLrn.x < 0 || pstCmd->rectLrn.y < 0 ||
        pstCmd->rectLrn.width <= 0 || pstCmd->rectLrn.height <= 0 ||
        ( pstCmd->rectLrn.x + pstCmd->rectLrn.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectLrn.y + pstCmd->rectLrn.height ) > pstCmd->matInputImg.rows ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The learn rect (%d, %d, %d, %d) is invalid.",
            pstCmd->rectLrn.x, pstCmd->rectLrn.y, pstCmd->rectLrn.width, pstCmd->rectLrn.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseLrnObj);

    cv::Ptr<cv::Feature2D> detector;
    if ( PR_SRCH_OBJ_ALGORITHM::SIFT == pstCmd->enAlgorithm)
        detector = SIFT::create (_constMinHessian, _constOctave );
    else
        detector = SURF::create (_constMinHessian, _constOctave, _constOctaveLayer );
    cv::Mat matROI( pstCmd->matInputImg, pstCmd->rectLrn );
    detector->detectAndCompute ( matROI, pstCmd->matMask, pstRpy->vecKeyPoint, pstRpy->matDescritor );

    if ( pstRpy->vecKeyPoint.size() < _constLeastFeatures ) {
        WriteLog("detectAndCompute can not find feature.");
        pstRpy->enStatus = VisionStatus::LEARN_OBJECT_FAIL;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    pstRpy->matResultImg = matROI.clone();
    cv::drawKeypoints( pstRpy->matResultImg, pstRpy->vecKeyPoint, pstRpy->matResultImg, _constRedScalar, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    pstRpy->enStatus = VisionStatus::OK;
    pstRpy->ptCenter.x = pstCmd->rectLrn.x + pstCmd->rectLrn.width  / 2.0f;
    pstRpy->ptCenter.y = pstCmd->rectLrn.y + pstCmd->rectLrn.height / 2.0f;
    matROI.copyTo(pstRpy->matTmpl);
    _writeLrnObjRecord(pstRpy);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::srchObj(const PR_SRCH_OBJ_CMD *const pstCmd, PR_SRCH_OBJ_RPY *const pstRpy, bool bReplay /*= false*/ ) {
    assert ( NULL != pstCmd || NULL != pstRpy );    
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    char charrMsg[1000];
    if ( pstCmd->nRecordId <= 0 ) {
        _snprintf(charrMsg, sizeof (charrMsg), "The input record ID %d is invalid.", pstCmd->nRecordId );
        WriteLog ( charrMsg );
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    ObjRecordPtr ptrRecord = std::static_pointer_cast<ObjRecord> ( RecordManager::getInstance()->get ( pstCmd->nRecordId ) );
    if ( nullptr == ptrRecord ) {
        _snprintf(charrMsg, sizeof (charrMsg), "Failed to get record ID %d in system.", pstCmd->nRecordId );
        WriteLog ( charrMsg );
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( ptrRecord->getType() != PR_RECORD_TYPE::OBJECT ) {
        _snprintf(charrMsg, sizeof (charrMsg), "The type of record ID %d is not OBJECT.", pstCmd->nRecordId );
        WriteLog ( charrMsg );
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseSrchObj);

    cv::Mat matSrchROI ( pstCmd->matInputImg, pstCmd->rectSrchWindow );
    
    cv::Ptr<cv::Feature2D> detector;
    if (PR_SRCH_OBJ_ALGORITHM::SIFT == pstCmd->enAlgorithm)
        detector = SIFT::create(_constMinHessian, _constOctave );
    else
        detector = SURF::create(_constMinHessian, _constOctave, _constOctaveLayer );
    
    std::vector<cv::KeyPoint> vecKeypointScene;
    cv::Mat matDescriptorScene;
    cv::Mat mask;
    
    detector->detectAndCompute ( matSrchROI, mask, vecKeypointScene, matDescriptorScene );
    TimeLog::GetInstance()->addTimeLog("detectAndCompute");

    if ( vecKeypointScene.size() < _constLeastFeatures ) {
        WriteLog("detectAndCompute can not find feature.");
        WriteLog ( charrMsg );
        pstRpy->enStatus = VisionStatus::SRCH_OBJ_FAIL;
        return pstRpy->enStatus;
    }

    VectorOfDMatch good_matches;

    cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::KDTreeIndexParams;
    cv::Ptr<cv::flann::SearchParams> searchParams = new cv::flann::SearchParams;

    cv::FlannBasedMatcher matcher;

    VectorOfVectorOfDMatch vecVecMatches;
    vecVecMatches.reserve(2000);
	
    matcher.knnMatch ( ptrRecord->getModelDescriptor(), matDescriptorScene, vecVecMatches, 2 );
	TimeLog::GetInstance()->addTimeLog("knnMatch");

	double max_dist = 0; double min_dist = 100.;
	int nScoreCount = 0;
    for ( VectorOfVectorOfDMatch::const_iterator it = vecVecMatches.begin();  it != vecVecMatches.end(); ++ it ) {
        VectorOfDMatch vecDMatch = *it;
        if ( (vecDMatch[0].distance / vecDMatch[1].distance ) < 0.8 )
            good_matches.push_back ( vecDMatch[0] );

		double dist = vecDMatch[0].distance;
		if (dist < 0.5)
			++ nScoreCount;
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

	pstRpy->fMatchScore = (float)nScoreCount * 100.f / vecVecMatches.size();

    if ( good_matches.size() <= 0 ) {
        pstRpy->enStatus = VisionStatus::SRCH_OBJ_FAIL;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); ++ i ) {
        //-- Get the keypoints from the good matches
        obj.push_back( ptrRecord->getModelKeyPoint()[good_matches[i].queryIdx].pt);
        scene.push_back( vecKeypointScene[good_matches[i].trainIdx].pt);
    }    
    
    pstRpy->matHomography = cv::findHomography ( obj, scene, CV_RANSAC );
    TimeLog::GetInstance()->addTimeLog("cv::findHomography");

	pstRpy->matHomography.at<double>(0, 2) += pstCmd->rectSrchWindow.x;
	pstRpy->matHomography.at<double>(1, 2) += pstCmd->rectSrchWindow.y;

    auto vevVecHomography = CalcUtils::matToVector<double>(pstRpy->matHomography);

    cv::Point2f ptLrnObjCenter = ptrRecord->getObjCenter();
    cv::Rect2f rectLrn(0, 0, ptLrnObjCenter.x * 2.f, ptLrnObjCenter.y * 2.f );
    
    VectorOfPoint2f vecPoint2f = CalcUtils::warpRect<double>( pstRpy->matHomography, rectLrn );
    VectorOfVectorOfPoint vecVecPoint(1);
    for ( const auto &point : vecPoint2f )
        vecVecPoint[0].push_back ( point );
    cv::cvtColor ( pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR );
    cv::polylines ( pstRpy->matResultImg, vecVecPoint, true, _constRedScalar );
    cv::drawKeypoints( pstRpy->matResultImg, vecKeypointScene, pstRpy->matResultImg, _constRedScalar, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    pstRpy->ptObjPos = CalcUtils::warpPoint<double>( pstRpy->matHomography, ptLrnObjCenter );
    pstRpy->szOffset.width = pstRpy->ptObjPos.x - pstCmd->ptExpectedPos.x;
    pstRpy->szOffset.height = pstRpy->ptObjPos.y - pstCmd->ptExpectedPos.y;
	float fRotationValue = (float)(pstRpy->matHomography.at<double>(1, 0) - pstRpy->matHomography.at<double>(0, 1)) / 2.0f;
	pstRpy->fRotation = (float) CalcUtils::radian2Degree ( asin ( fRotationValue ) );
    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;    
    return pstRpy->enStatus;
}

namespace
{
    int _calculateVerticalProjection(const cv::Mat &mat, cv::Mat &output)
    {
        if (mat.empty())
            return 0;

        output = cv::Mat::zeros(1, mat.rows, CV_32FC1);
        VectorOfPoint vecPoints;
        cv::findNonZero(mat, vecPoints);
        for ( const auto &point : vecPoints )   {
            ++ output.at<float>(0, point.y);
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

    //The exponent smooth algorithm come from pater "A Visual Inspection System for Surface Mounted Components Based on Color Features".
    template<typename T>
    void _expSmooth(const cv::Mat &matInputImg, cv::Mat &matOutput, float alpha)
    {
        matInputImg.copyTo(matOutput);
        for (int col = 1; col < matInputImg.cols; ++col)
        {
            matOutput.at<T>(0, col) = alpha * matInputImg.at<T>(0, col) + (1.f - alpha) * matOutput.at<T>(0, col - 1);
        }
    }

    //To find edge from left to right, or from top to bottom.
    VisionStatus _findEdge(const cv::Mat &matProjection, int nObjectSize, int &nObjectEdgeResult)
    {
        if (matProjection.cols < nObjectSize )
            return VisionStatus::INVALID_PARAM;

        auto fMaxSumOfProjection = 0.f;
        nObjectEdgeResult = 0;
        for (int nObjectStart = 0; nObjectStart < matProjection.cols - nObjectSize; ++ nObjectStart)
        {
            auto fSumOfProjection = 0.f;
            for (int nIndex = nObjectStart; nIndex < nObjectStart + nObjectSize; ++ nIndex)
                fSumOfProjection += matProjection.at<float>(0, nIndex);
            if (fSumOfProjection > fMaxSumOfProjection) {
                fMaxSumOfProjection = fSumOfProjection;
                nObjectEdgeResult = nObjectStart;
            }
        }
        return VisionStatus::OK;
    }

    //To find edge from right to left, or from bottom to top.
    VisionStatus _findEdgeReverse(const cv::Mat &matProjection, int nObjectSize, int &nObjectEdgeResult)
    {
        if (matProjection.cols < nObjectSize)
            return VisionStatus::INVALID_PARAM;

        auto fMaxSumOfProjection = 0.f;
        nObjectEdgeResult = 0;
        for (int nObjectStart = matProjection.cols - 1; nObjectStart > nObjectSize; --nObjectStart)
        {
            auto fSumOfProjection = 0.f;
            for (int nIndex = nObjectStart - 1; nIndex > nObjectStart - nObjectSize; --nIndex)
                fSumOfProjection += matProjection.at<float>(0, nIndex);
            if (fSumOfProjection > fMaxSumOfProjection) {
                fMaxSumOfProjection = fSumOfProjection;
                nObjectEdgeResult = nObjectStart;
            }
        }
        return VisionStatus::OK;
    }

    struct AOI_COUNTOUR {
        cv::Point2f     ptCtr;
        float           fArea;
        VectorOfPoint   contour;
    };

    bool _isSimilarContourExist(const std::vector<AOI_COUNTOUR> &vecAoiCountour, const AOI_COUNTOUR &stCountourInput)
    {
        for (auto stContour : vecAoiCountour)
        {
            if (CalcUtils::distanceOf2Point<float>(stContour.ptCtr, stCountourInput.ptCtr) < 3.f
                && fabs(stContour.fArea - stCountourInput.fArea) < 10)
                return true;
        }
        return false;
    }
}

/*******************************************************************************
* Purpose:   Auto find the threshold to distinguish the feature need to extract.
* Algorithm: The method is come from paper: "A Fast Algorithm for Multilevel 
*            Thresholding".
* Input:     mat - The input image.
             N - Number of therehold to calculate. 1 <= N <= 4
* Output:    The calculated threshold.
* Note:      Mu is name of greek symbol μ.
*            Omega and Eta are also greek symbol.
*******************************************************************************/
std::vector<Int16> VisionAlgorithm::_autoMultiLevelThreshold(const cv::Mat &matInputImg, const cv::Mat &matMask, int N)
{
    assert(N >= 1 && N <= 4);

    const int sbins = 128;
    const float MIN_P = 0.001f;
    std::vector<Int16> vecThreshold;
    int channels[] = { 0 };
    int histSize[] = { sbins };

    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { sranges };

    cv::MatND hist;
    cv::calcHist(&matInputImg, 1, channels, matMask,
        hist, 1, histSize, ranges,
        true, // the histogram is uniform
        false);

    std::vector<float> P, S;
    P.push_back(0.f);
    S.push_back(0.f);
    float fTotalPixel = (float)matInputImg.rows * matInputImg.cols;
    for (int i = 0; i < sbins; ++i)   {
        float fPi = hist.at<float>(i, 0) / fTotalPixel;
        P.push_back(fPi + P[i]);
        S.push_back(i * fPi + S[i]);
    }

    std::vector<std::vector<float>> vecVecOmega(sbins + 1, std::vector<float>(sbins + 1, 0));
    std::vector<std::vector<float>> vecVecMu(sbins + 1, std::vector<float>(sbins + 1, 0));
    std::vector<std::vector<float>> vecVecEta(sbins + 1, std::vector<float>(sbins + 1, -1.f));
    for (int i = 0; i <= sbins; ++ i)
    for (int j = i; j <= sbins; ++ j)
    {
        vecVecOmega[i][j] = P[j] - P[i];
        vecVecMu[i][j]    = S[j] - S[i];
    }

    const int START = 1;
    if ( 1 == N ) {
        float fMaxSigma = 0;
        int nResultT1 = 0;
        for (int T = START; T < sbins; ++ T) {
            float fSigma = 0.f;
            if (vecVecOmega[START][T] > MIN_P)
                fSigma += vecVecMu[START][T] * vecVecMu[1][T] / vecVecOmega[START][T];
            if (vecVecOmega[T][sbins] > MIN_P)
                fSigma += vecVecMu[T][sbins] * vecVecMu[T][sbins] / vecVecOmega[T][sbins];
            if (fMaxSigma < fSigma)    {
                nResultT1 = T;
                fMaxSigma = fSigma;
            }
        }
        vecThreshold.push_back(nResultT1 * PR_MAX_GRAY_LEVEL / sbins);
    }else if ( 2 == N ) {
        float fMaxSigma = 0;
        int nResultT1 = 0, nResultT2 = 0;
        for (int T1 = START;  T1 < sbins - 1; ++ T1)
        for (int T2 = T1 + 1; T2 < sbins;     ++ T2)
        {
            float fSigma = 0.f;

            if (vecVecEta[START][T1] >= 0.f)
                fSigma += vecVecEta[START][T1];
            else
            {
                if (vecVecOmega[START][T1] > MIN_P)
                    vecVecEta[START][T1] = vecVecMu[START][T1] * vecVecMu[START][T1] / vecVecOmega[START][T1];
                else
                    vecVecEta[START][T1] = 0.f;
                fSigma += vecVecEta[START][T1];
            }

            if (vecVecEta[T1][T2] >= 0.f)
                fSigma += vecVecEta[T1][T2];
            else
            {
                if (vecVecOmega[T1][T2] > MIN_P)
                    vecVecEta[T1][T2] = vecVecMu[T1][T2] * vecVecMu[T1][T2] / vecVecOmega[T1][T2];
                else
                    vecVecEta[T1][T2] = 0.f;
                fSigma += vecVecEta[T1][T2];
            }

            if (vecVecEta[T2][sbins] >= 0.f)
                fSigma += vecVecEta[T2][sbins];
            else
            {
                if (vecVecOmega[T2][sbins] > MIN_P)
                    vecVecEta[T2][sbins] = vecVecMu[T2][sbins] * vecVecMu[T2][sbins] / vecVecOmega[T2][sbins];
                else
                    vecVecEta[T2][sbins] = 0.f;
                fSigma += vecVecEta[T2][sbins];
            }

            if (fMaxSigma < fSigma)    {
                nResultT1 = T1;
                nResultT2 = T2;
                fMaxSigma = fSigma;
            }
        }
        vecThreshold.push_back(nResultT1 * PR_MAX_GRAY_LEVEL / sbins);
        vecThreshold.push_back(nResultT2 * PR_MAX_GRAY_LEVEL / sbins);
    }else if ( 3 == N ) {
        float fMaxSigma = 0;
        int nResultT1 = 0, nResultT2 = 0, nResultT3 = 0;
        for (int T1 = START;  T1 < sbins - 2; ++ T1)
        for (int T2 = T1 + 1; T2 < sbins - 1; ++ T2)
        for (int T3 = T2 + 1; T3 < sbins;     ++ T3)
        {

            float fSigma = 0.f;
            if (vecVecEta[START][T1] >= 0.f)
                fSigma += vecVecEta[START][T1];
            else
            {
                if (vecVecOmega[START][T1] > MIN_P)
                    vecVecEta[START][T1] = vecVecMu[START][T1] * vecVecMu[START][T1] / vecVecOmega[START][T1];
                else
                    vecVecEta[START][T1] = 0.f;
                fSigma += vecVecEta[START][T1];
            }

            if (vecVecEta[T1][T2] >= 0.f)
                fSigma += vecVecEta[T1][T2];
            else
            {
                if (vecVecOmega[T1][T2] > MIN_P)
                    vecVecEta[T1][T2] = vecVecMu[T1][T2] * vecVecMu[T1][T2] / vecVecOmega[T1][T2];
                else
                    vecVecEta[T1][T2] = 0.f;
                fSigma += vecVecEta[T1][T2];
            }

            if (vecVecEta[T2][T3] >= 0.f)
                fSigma += vecVecEta[T2][T3];
            else
            {
                if (vecVecOmega[T2][T3] > MIN_P)
                    vecVecEta[T2][T3] = vecVecMu[T2][T3] * vecVecMu[T2][T3] / vecVecOmega[T2][T3];
                else
                    vecVecEta[T2][T3] = 0.f;
                fSigma += vecVecEta[T2][T3];
            }

            if (vecVecEta[T3][sbins] >= 0.f)
                fSigma += vecVecEta[T3][sbins];
            else
            {
                if (vecVecOmega[T3][sbins] > MIN_P)
                    vecVecEta[T3][sbins] = vecVecMu[T3][sbins] * vecVecMu[T3][sbins] / vecVecOmega[T3][sbins];
                else
                    vecVecEta[T3][sbins] = 0.f;
                fSigma += vecVecEta[T3][sbins];
            }

            if (fMaxSigma < fSigma) {
                nResultT1 = T1;
                nResultT2 = T2;
                nResultT3 = T3;
                fMaxSigma = fSigma;
            }
        }
        vecThreshold.push_back(nResultT1 * PR_MAX_GRAY_LEVEL / sbins);
        vecThreshold.push_back(nResultT2 * PR_MAX_GRAY_LEVEL / sbins);
        vecThreshold.push_back(nResultT3 * PR_MAX_GRAY_LEVEL / sbins);
    }else if ( 4 == N ) {
        float fMaxSigma = 0;
        int nResultT1 = 0, nResultT2 = 0, nResultT3 = 0, nResultT4 = 0;
        for (int T1 = START;  T1 < sbins - 3; ++ T1)
        for (int T2 = T1 + 1; T2 < sbins - 2; ++ T2)
        for (int T3 = T2 + 1; T3 < sbins - 1; ++ T3)
        for (int T4 = T3 + 1; T4 < sbins;     ++ T4)
        {
            float fSigma = 0.f;

            if (vecVecEta[START][T1] >= 0.f)
                fSigma += vecVecEta[START][T1];
            else
            {
                if (vecVecOmega[START][T1] > MIN_P)
                    vecVecEta[START][T1] = vecVecMu[START][T1] * vecVecMu[START][T1] / vecVecOmega[START][T1];
                else
                    vecVecEta[START][T1] = 0.f;
                fSigma += vecVecEta[START][T1];
            }

            if (vecVecEta[T1][T2] >= 0.f)
                fSigma += vecVecEta[T1][T2];
            else
            {
                if (vecVecOmega[T1][T2] > MIN_P)
                    vecVecEta[T1][T2] = vecVecMu[T1][T2] * vecVecMu[T1][T2] / vecVecOmega[T1][T2];
                else
                    vecVecEta[T1][T2] = 0.f;
                fSigma += vecVecEta[T1][T2];
            }

            if (vecVecEta[T2][T3] >= 0.f)
                fSigma += vecVecEta[T2][T3];
            else
            {
                if (vecVecOmega[T2][T3] > MIN_P)
                    vecVecEta[T2][T3] = vecVecMu[T2][T3] * vecVecMu[T2][T3] / vecVecOmega[T2][T3];
                else
                    vecVecEta[T2][T3] = 0.f;
                fSigma += vecVecEta[T2][T3];
            }

            if (vecVecEta[T3][T4] >= 0.f)
                fSigma += vecVecEta[T3][T4];
            else
            {
                if (vecVecOmega[T3][T4] > MIN_P)
                    vecVecEta[T3][T4] = vecVecMu[T3][T4] * vecVecMu[T3][T4] / vecVecOmega[T3][T4];
                else
                    vecVecEta[T3][T4] = 0.f;
                fSigma += vecVecEta[T3][T4];
            }

            if (vecVecEta[T4][sbins] >= 0.f)
                fSigma += vecVecEta[T4][sbins];
            else
            {
                if (vecVecOmega[T4][sbins] > MIN_P)
                    vecVecEta[T4][sbins] = vecVecMu[T4][sbins] * vecVecMu[T4][sbins] / vecVecOmega[T4][sbins];
                else
                    vecVecEta[T4][sbins] = 0.f;
                fSigma += vecVecEta[T4][sbins];
            }

            if (fMaxSigma < fSigma)    {
                nResultT1 = T1;
                nResultT2 = T2;
                nResultT3 = T3;
                nResultT4 = T4;
                fMaxSigma = fSigma;
            }
        }
        vecThreshold.push_back(nResultT1 * PR_MAX_GRAY_LEVEL / sbins);
        vecThreshold.push_back(nResultT2 * PR_MAX_GRAY_LEVEL / sbins);
        vecThreshold.push_back(nResultT3 * PR_MAX_GRAY_LEVEL / sbins);
        vecThreshold.push_back(nResultT4 * PR_MAX_GRAY_LEVEL / sbins);
    }

    return vecThreshold;
}

VisionStatus VisionAlgorithm::autoThreshold(PR_AUTO_THRESHOLD_CMD *pstCmd, PR_AUTO_THRESHOLD_RPY *pstRpy, bool bReplay) {
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char charrMsg[1000];
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows ) {
        WriteLog("The auto threshold range is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if ( ! pstCmd->matMask.empty() ) {
        if ( pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols ) {
            _snprintf(charrMsg, sizeof(charrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d ).",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(charrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if ( pstCmd->matMask.channels() != 1 )  {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseAutoThreshold);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    cv::Mat matMaskROI;
    if ( ! pstCmd->matMask.empty() )
        matMaskROI = cv::Mat( pstCmd->matMask, pstCmd->rectROI).clone();
    pstRpy->vecThreshold = _autoMultiLevelThreshold ( matROI, matMaskROI, pstCmd->nThresholdNum);

    pstRpy->enStatus = VisionStatus::OK;
    
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

int VisionAlgorithm::_autoThreshold(const cv::Mat &mat, const cv::Mat &matMask/* = cv::Mat()*/ ) {
    auto vecThreshold = _autoMultiLevelThreshold ( mat, matMask, 1 );
    return vecThreshold[0];
}

//Need to use the in tolerance method to check.
/*static*/ bool VisionAlgorithm::_isContourRectShape(const VectorOfPoint &vecPoint) {
    cv::Rect rect = cv::boundingRect ( vecPoint );
    if ( rect.width <= 0 || rect.height <= 0 )
        return false;

    auto rectHead = cv::minAreaRect ( vecPoint );

    float fTolerance = 4.f;
    if ( rectHead.size.width <= 2 * fTolerance || rectHead.size.height <= 2 * fTolerance ) {
        return false;
    }

    if ( fabs ( rectHead.angle ) < 5 ) {
        if ( ! _isContourRectShapeProjection ( vecPoint ) )
            return false;
    }

    cv::Mat matVerify = cv::Mat::zeros ( rect.size() + cv::Size(rect.x, rect.y), CV_8UC1 );
    auto rectEnlarge = rectHead;
    rectEnlarge.size.width  += 2 * fTolerance;
    rectEnlarge.size.height += 2 * fTolerance;
    VectorOfVectorOfPoint vevVecPoint;
    vevVecPoint.push_back ( CalcUtils::getCornerOfRotatedRect( rectEnlarge ) );
    cv::fillPoly ( matVerify, vevVecPoint, cv::Scalar::all ( PR_MAX_GRAY_LEVEL ) );

    auto rectDeflate = rectHead;
    rectDeflate.size.width  -= 2 * fTolerance;
    rectDeflate.size.height -= 2 * fTolerance;
    vevVecPoint.clear();
    vevVecPoint.push_back ( CalcUtils::getCornerOfRotatedRect( rectDeflate ) );
    cv::fillPoly ( matVerify, vevVecPoint, cv::Scalar::all(0) );
    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode() )
        showImage ( "Verify Mat", matVerify );

    int nInTolPointCount = 0;
    for ( const auto &point : vecPoint ) {
        if ( matVerify.at<uchar>(point) > 0 )
            ++ nInTolPointCount;
    }

    if ( nInTolPointCount < ToInt32 ( vecPoint.size() * 0.5 ) )
        return false;
    return true;
}

/*static*/ bool VisionAlgorithm::_isContourRectShapeProjection ( const VectorOfPoint &vecPoint ) {
    cv::Rect rect = cv::boundingRect ( vecPoint );
    if (rect.width <= 0 || rect.height <= 0)
        return false;

    VectorOfVectorOfPoint vevVecPoint;
    cv::Mat matBlack = cv::Mat::zeros ( rect.size () + cv::Size ( rect.x, rect.y ), CV_8UC1 );
    vevVecPoint.push_back ( vecPoint );
    cv::fillPoly ( matBlack, vevVecPoint, cv::Scalar::all ( PR_MAX_GRAY_LEVEL ) );

    cv::Mat matROI ( matBlack, rect );
    cv::Mat matOneRow, matOneCol;
    cv::reduce ( matROI, matOneRow, 0, cv::ReduceTypes::REDUCE_SUM, CV_32SC1 );
    cv::reduce ( matROI, matOneCol, 1, cv::ReduceTypes::REDUCE_SUM, CV_32SC1 );

    float fAverageHeight = ToFloat ( cv::sum ( matOneRow )[0] / matOneRow.cols );
    int nConsecutiveOverTolerance = 0;
    const float FLUCTUATE_TOL = 0.3f;
    const int FLUCTUATE_WIDTH_TOL = 10;
    for (int index = 0; index < matOneRow.cols; ++ index) {
        if ( fabs ( matOneRow.at<int> ( 0, index ) - fAverageHeight ) / fAverageHeight > FLUCTUATE_TOL )
            ++ nConsecutiveOverTolerance;
        else
            nConsecutiveOverTolerance = 0;

        if ( nConsecutiveOverTolerance > FLUCTUATE_WIDTH_TOL )
            return false;        
    }
    return true;
}

/*static*/ inline float VisionAlgorithm::getContourRectRatio( const VectorOfPoint &vecPoint) {
    auto rotatedRect = cv::minAreaRect ( vecPoint );
    float fMaxDim = std::max ( rotatedRect.size.width, rotatedRect.size.height );
    float fMinDim = std::min ( rotatedRect.size.width, rotatedRect.size.height );
    return fMaxDim / fMinDim;
}

/*static*/ VisionStatus VisionAlgorithm::_findDeviceElectrode ( const cv::Mat &matDeviceROI, VectorOfPoint &vecElectrodePos, VectorOfSize2f &vecElectrodeSize, VectorOfVectorOfPoint &vevVecContour ) {
    VectorOfVectorOfPoint vecContours;
    std::vector<cv::Vec4i> hierarchy;
    vecElectrodePos.clear();
    const float MAX_TWO_DIM_RATIO = 4.f;

    cv::Mat matDraw = cv::Mat::zeros ( matDeviceROI.size(), CV_8UC3 );
    // Find contours
    cv::findContours ( matDeviceROI, vecContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    // Filter contours
    std::vector<AOI_COUNTOUR> vecAoiContour;
    int index = 0;
    for ( auto contour : vecContours ) {
        auto area = cv::contourArea ( contour );
        if ( area > 200 && getContourRectRatio ( contour ) < MAX_TWO_DIM_RATIO && _isContourRectShape ( contour ) ) {
            AOI_COUNTOUR stAoiContour;
            cv::Moments moment = cv::moments( contour );
            stAoiContour.ptCtr.x = static_cast< float >(moment.m10 / moment.m00);
            stAoiContour.ptCtr.y = static_cast< float >(moment.m01 / moment.m00);
            stAoiContour.fArea = ToFloat ( area );
            if ( ! _isSimilarContourExist ( vecAoiContour, stAoiContour ) ) {
                stAoiContour.contour = contour;
                vecAoiContour.push_back ( stAoiContour );

                if ( Config::GetInstance()->getDebugMode () == PR_DEBUG_MODE::SHOW_IMAGE ) {
                    cv::RNG rng ( 12345 );
                    cv::Scalar color = cv::Scalar ( rng.uniform ( 0, 255 ), rng.uniform ( 0, 255 ), rng.uniform ( 0, 255 ) );
                    cv::drawContours ( matDraw, vecContours, index, color );
                }
            }
        }
        ++ index;
    }
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Contours", matDraw );

    if ( vecAoiContour.size() < PR_ELECTRODE_COUNT )
        return VisionStatus::FIND_ELECTRODE_FAIL;

    if ( vecAoiContour.size() > PR_ELECTRODE_COUNT ) {
        std::sort ( vecAoiContour.begin(), vecAoiContour.end(), [](const AOI_COUNTOUR &stA, const AOI_COUNTOUR &stB ) {
            return stB.fArea < stA.fArea;
        });
        vecAoiContour = std::vector<AOI_COUNTOUR>( vecAoiContour.begin(), vecAoiContour.begin() + PR_ELECTRODE_COUNT );
    }

    float fArea1 = vecAoiContour[0].fArea;
    float fArea2 = vecAoiContour[1].fArea;
    if ( fabs ( fArea1 - fArea2 ) / fArea1 > 0.2 )
        return VisionStatus::FIND_ELECTRODE_FAIL;

    for ( const auto &stAOIContour : vecAoiContour ) {
        vecElectrodePos.push_back ( stAOIContour.ptCtr );
        auto rotatedRect = cv::minAreaRect ( stAOIContour.contour );
        vecElectrodeSize.push_back ( rotatedRect.size );
        vevVecContour.push_back ( stAOIContour.contour );
    }

    return VisionStatus::OK;
}

//The alogorithm come from paper "A Visual Inspection System for Surface Mounted Components Based on Color Features"
/*static*/ VisionStatus VisionAlgorithm::_findDeviceEdge(const cv::Mat &matDeviceROI, const cv::Size2f &size, cv::Rect &rectDeviceResult)
{
    cv::Mat matGray, matThreshold, matHorizontalProjection, matVerticalProjection;
    Int32 nLeftEdge = 0, nRightEdge = 0, nTopEdge = 0, nBottomEdge = 0;

    _calculateHorizontalProjection(matDeviceROI, matHorizontalProjection);
    _expSmooth<float>(matHorizontalProjection, matHorizontalProjection, _constExpSmoothRatio);
    _findEdge(matHorizontalProjection, static_cast< Int32 >(size.width), nLeftEdge);
    _findEdgeReverse(matHorizontalProjection, static_cast< Int32 >(size.width), nRightEdge);
    if ( ( nRightEdge - nLeftEdge ) < ( size.width / 2 ) )
        return VisionStatus::FIND_DEVICE_EDGE_FAIL;

    _calculateVerticalProjection(matThreshold, matVerticalProjection);
    _expSmooth<float>(matVerticalProjection, matVerticalProjection, _constExpSmoothRatio);
    _findEdge(matHorizontalProjection, static_cast< Int32 >(size.height), nTopEdge);
    _findEdgeReverse(matHorizontalProjection, static_cast< Int32 >(size.height), nBottomEdge);
    if ( (nBottomEdge - nTopEdge ) < size.height / 2 )
        return VisionStatus::FIND_DEVICE_EDGE_FAIL;

    rectDeviceResult = cv::Rect ( cv::Point ( nLeftEdge, nTopEdge ), cv::Point ( nRightEdge, nBottomEdge ) );
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::lrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy)
{
    char charrMsg[1000];
    if ( nullptr == pstLrnDeviceCmd || nullptr == pstLrnDeivceRpy ) {
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstLrnDeviceCmd = %d, pstLrnDeivceRpy = %d", (int)pstLrnDeviceCmd, (int)pstLrnDeivceRpy);
        WriteLog(charrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    if ( pstLrnDeviceCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty");
        return VisionStatus::INVALID_PARAM;
    }

    cv::Rect rectDeviceROI = CalcUtils::resizeRect<float> ( pstLrnDeviceCmd->rectDevice, pstLrnDeviceCmd->rectDevice.size() + cv::Size2f(20, 20) );
    if ( rectDeviceROI.x < 0  ) rectDeviceROI.x = 0;
    if ( rectDeviceROI.y < 0  ) rectDeviceROI.y = 0;
    if ( ( rectDeviceROI.x + rectDeviceROI.width ) > pstLrnDeviceCmd->matInputImg.cols ) rectDeviceROI.width  = pstLrnDeviceCmd->matInputImg.cols - rectDeviceROI.x;
    if ( ( rectDeviceROI.y + rectDeviceROI.height) > pstLrnDeviceCmd->matInputImg.rows ) rectDeviceROI.height = pstLrnDeviceCmd->matInputImg.rows - rectDeviceROI.y;

    cv::Mat matDevice ( pstLrnDeviceCmd->matInputImg, rectDeviceROI );
    cv::Mat matGray, matThreshold, matHorizontalProjection, matVerticalProjection;

    std::vector<cv::Mat> vecChannels;
    cv::split( matDevice, vecChannels );

    matGray = vecChannels [ Config::GetInstance()->getDeviceInspChannel() ];
    
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage ( "Learn channel image", matGray);

    cv::Mat matBlur;
    cv::blur ( matGray, matBlur, cv::Size(2, 2) );
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Blur image", matBlur);

    if ( pstLrnDeviceCmd->bAutoThreshold )    
        pstLrnDeviceCmd->nElectrodeThreshold = _autoThreshold ( matBlur );

    cv::threshold ( matBlur, matThreshold, pstLrnDeviceCmd->nElectrodeThreshold, 255, cv::THRESH_BINARY );    

    VectorOfPoint vecPtElectrode;
    VectorOfSize2f vecElectrodeSize;
    VectorOfVectorOfPoint vecContours;
    VisionStatus enStatus = _findDeviceElectrode ( matThreshold, vecPtElectrode, vecElectrodeSize, vecContours );
    if (enStatus != VisionStatus::OK) {
        pstLrnDeivceRpy->nStatus = ToInt32(enStatus);
        return enStatus;
    }    

    cv::Rect rectFindDeviceResult;
    enStatus = _findDeviceEdge ( matThreshold, cv::Size2f( pstLrnDeviceCmd->rectDevice.width, pstLrnDeviceCmd->rectDevice.height), rectFindDeviceResult );
    if (enStatus != VisionStatus::OK) {
        pstLrnDeivceRpy->nStatus = ToInt32(enStatus);
        return enStatus;
    }

    float fScaleX = ToFloat(rectFindDeviceResult.width) / pstLrnDeviceCmd->rectDevice.width;
    float fScaleY = ToFloat(rectFindDeviceResult.height) / pstLrnDeviceCmd->rectDevice.height;
    float fScaleMax = std::max<float>(fScaleX, fScaleY);
    float fScaleMin = std::min<float>(fScaleX, fScaleY);
    if ( fScaleMax > 1.2f || fScaleMin < 0.8f )  {
        pstLrnDeivceRpy->nStatus = ToInt32 ( VisionStatus::OBJECT_SCALE_FAIL );
        return VisionStatus::OBJECT_SCALE_FAIL;
    }

    pstLrnDeivceRpy->nStatus = ToInt32 ( VisionStatus::OK );
    pstLrnDeivceRpy->sizeDevice = cv::Size2f ( ToFloat ( rectFindDeviceResult.width ), ToFloat ( rectFindDeviceResult.height ) );
    pstLrnDeivceRpy->nElectrodeThreshold = pstLrnDeviceCmd->nElectrodeThreshold;
    _writeDeviceRecord ( pstLrnDeivceRpy );
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::inspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy)
{
    MARK_FUNCTION_START_TIME;
    char charrMsg[1000];
    if ( pstInspDeviceCmd == nullptr || pstInspDeivceRpy == nullptr ) {
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstInspDeviceCmd = %d, pstInspDeivceRpy = %d", (int)pstInspDeviceCmd, (int)pstInspDeivceRpy);
        WriteLog(charrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    if ( pstInspDeviceCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty");
        return VisionStatus::INVALID_PARAM;
    }

    pstInspDeivceRpy->nDeviceCount = pstInspDeviceCmd->nDeviceCount;
    for (Int32 i = 0; i < pstInspDeviceCmd->nDeviceCount; ++ i) {
        if ( pstInspDeviceCmd->astDeviceInfo[i].stSize.area() < 1)   {
            _snprintf(charrMsg, sizeof(charrMsg), "Device[ %d ] size is invalid", i );
            WriteLog(charrMsg);
            return VisionStatus::INVALID_PARAM;
        }

        cv::Rect rectDeviceROI = pstInspDeviceCmd->astDeviceInfo[i].rectSrchWindow;
        if (rectDeviceROI.x < 0) rectDeviceROI.x = 0;
        if (rectDeviceROI.y < 0) rectDeviceROI.y = 0;
        if ((rectDeviceROI.x + rectDeviceROI.width)  > pstInspDeviceCmd->matInputImg.cols) rectDeviceROI.width  = pstInspDeviceCmd->matInputImg.cols - rectDeviceROI.x;
        if ((rectDeviceROI.y + rectDeviceROI.height) > pstInspDeviceCmd->matInputImg.rows) rectDeviceROI.height = pstInspDeviceCmd->matInputImg.rows - rectDeviceROI.y;

        cv::Mat matROI(pstInspDeviceCmd->matInputImg, rectDeviceROI );
        cv::Mat matGray, matThreshold, matHorizontalProjection, matVerticalProjection;

        {   //Init inspection result
            pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(VisionStatus::OK);
            pstInspDeivceRpy->astDeviceResult[i].fRotation = 0.f;
            pstInspDeivceRpy->astDeviceResult[i].fOffsetX = 0.f;
            pstInspDeivceRpy->astDeviceResult[i].fOffsetY = 0.f;
            pstInspDeivceRpy->astDeviceResult[i].fScale = 1.f;
            pstInspDeivceRpy->astDeviceResult[i].ptPos = cv::Point2f(0.f, 0.f);
        }

        std::vector<cv::Mat> vecChannels;
        cv::split( matROI, vecChannels );

        matGray = vecChannels [ Config::GetInstance()->getDeviceInspChannel() ];

        cv::Mat matBlur;
        cv::blur ( matGray, matBlur, cv::Size(3, 3) );
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
            showImage("Blur image", matBlur );

        //cv::cvtColor( matROI, matGray, CV_BGR2GRAY);
        cv::threshold( matBlur, matThreshold, pstInspDeviceCmd->nElectrodeThreshold, 255, cv::THRESH_BINARY);
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
            cv::imshow("Threshold image", matThreshold );
            cv::waitKey(0);
        }

        VectorOfPoint vecPtElectrode;
        VectorOfSize2f vecElectrodeSize;
        VectorOfVectorOfPoint vecContours;
        VisionStatus enStatus = _findDeviceElectrode ( matThreshold, vecPtElectrode, vecElectrodeSize, vecContours );
        if ( enStatus != VisionStatus::OK ) {
            pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(enStatus);
            continue;
        }

        if (pstInspDeviceCmd->astDeviceInfo [ i ].stInspItem.bCheckRotation 
            || pstInspDeviceCmd->astDeviceInfo [ i ].stInspItem.bCheckShift
            || pstInspDeviceCmd->astDeviceInfo [ i ].stInspItem.bCheckScale )
        {
            if (0 > pstInspDeviceCmd->astDeviceInfo [ i ].nCriteriaNo || pstInspDeviceCmd->astDeviceInfo [ i ].nCriteriaNo >= PR_MAX_CRITERIA_COUNT) {
                _snprintf(charrMsg, sizeof(charrMsg), "Device[ %d ] Input nCriteriaNo %d is invalid", i, pstInspDeviceCmd->astDeviceInfo [ i ].nCriteriaNo);
                WriteLog(charrMsg);
                return VisionStatus::INVALID_PARAM;
            }
        }

        double dRotate = CalcUtils::calcSlopeDegree ( vecPtElectrode [ 0 ], vecPtElectrode [ 1 ] );
        pstInspDeivceRpy->astDeviceResult [ i ].fRotation = ToFloat ( dRotate );
        if (pstInspDeviceCmd->astDeviceInfo [ i ].stInspItem.bCheckRotation) {            
            if ( fabs ( dRotate ) > pstInspDeviceCmd->astCriteria [ pstInspDeviceCmd->astDeviceInfo [ i ].nCriteriaNo ].fMaxRotate) {
                pstInspDeivceRpy->astDeviceResult[ i ].nStatus = ToInt32(VisionStatus::OBJECT_ROTATE);
                continue;
            }
        }

        cv::Point2f ptDeviceCtr = CalcUtils::midOf2Points<float>( vecPtElectrode [ 0 ], vecPtElectrode [ 1 ] );
        //Map the find result from ROI to the whole image.
        ptDeviceCtr.x += rectDeviceROI.x;
        ptDeviceCtr.y += rectDeviceROI.y;

        pstInspDeivceRpy->astDeviceResult[i].ptPos = ptDeviceCtr;
        pstInspDeivceRpy->astDeviceResult[i].fOffsetX = pstInspDeivceRpy->astDeviceResult[i].ptPos.x - pstInspDeviceCmd->astDeviceInfo [ i ].stCtrPos.x;
        pstInspDeivceRpy->astDeviceResult[i].fOffsetY = pstInspDeivceRpy->astDeviceResult[i].ptPos.y - pstInspDeviceCmd->astDeviceInfo [ i ].stCtrPos.y;
        if (pstInspDeviceCmd->astDeviceInfo [ i ].stInspItem.bCheckShift) {
            if (  fabs ( pstInspDeivceRpy->astDeviceResult[i].fOffsetX ) > pstInspDeviceCmd->astCriteria [ pstInspDeviceCmd->astDeviceInfo [ i ].nCriteriaNo ].fMaxOffsetX )  {
                pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(VisionStatus::OBJECT_SHIFT);
                continue;
            }
            
            if (  fabs ( pstInspDeivceRpy->astDeviceResult[i].fOffsetY ) > pstInspDeviceCmd->astCriteria [ pstInspDeviceCmd->astDeviceInfo [ i ].nCriteriaNo ].fMaxOffsetY )  {
                pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(VisionStatus::OBJECT_SHIFT);
                continue;
            }
        }

        cv::Rect rectDevice;
        enStatus = _findDeviceEdge ( matThreshold, pstInspDeviceCmd->astDeviceInfo[i].stSize, rectDevice );
        if ( enStatus != VisionStatus::OK ) {
            pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(enStatus);
            continue;
        }                

        float fScaleX = ToFloat ( rectDevice.width ) / pstInspDeviceCmd->astDeviceInfo [ i ].stSize.width;
        float fScaleY = ToFloat ( rectDevice.height ) / pstInspDeviceCmd->astDeviceInfo [ i ].stSize.height;
        pstInspDeivceRpy->astDeviceResult[i].fScale = ( fScaleX + fScaleY ) / 2.f;
        if ( pstInspDeviceCmd->astDeviceInfo [ i ].stInspItem.bCheckScale ) {            
            //float fScaleMax = std::max<float> ( fScaleX, fScaleY );
            //float fScaleMin = std::min<float> ( fScaleX, fScaleY );
            if ( pstInspDeivceRpy->astDeviceResult[i].fScale > pstInspDeviceCmd->astCriteria [ pstInspDeviceCmd->astDeviceInfo [ i ].nCriteriaNo ].fMaxScale 
                || pstInspDeivceRpy->astDeviceResult[i].fScale < pstInspDeviceCmd->astCriteria [ pstInspDeviceCmd->astDeviceInfo [ i ].nCriteriaNo ].fMinScale )  {
                pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(VisionStatus::OBJECT_SCALE_FAIL);
                continue;
            }
        }
    }
    MARK_FUNCTION_END_TIME;
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::matchTemplate(PR_MATCH_TEMPLATE_CMD *const pstCmd, PR_MATCH_TEMPLATE_RPY *pstRpy, bool bReplay)
{
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    
    if ( pstCmd->matTmpl.empty() ) {
        WriteLog("Template image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectSrchWindow.x < 0 || pstCmd->rectSrchWindow.y < 0 ||
        pstCmd->rectSrchWindow.width <= 0 || pstCmd->rectSrchWindow.height <= 0 ||
        ( pstCmd->rectSrchWindow.x + pstCmd->rectSrchWindow.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectSrchWindow.y + pstCmd->rectSrchWindow.height ) > pstCmd->matInputImg.rows ) {
        WriteLog("The search window is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( pstCmd->matTmpl.rows > pstCmd->rectSrchWindow.height || pstCmd->matTmpl.cols > pstCmd->rectSrchWindow.width ) {
        WriteLog("The template is bigger than search window");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
   
    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseMatchTmpl);

    cv::Mat matSrchROI( pstCmd->matInputImg, pstCmd->rectSrchWindow );
    if ( matSrchROI.channels() > 1 )
        cv::cvtColor ( matSrchROI, matSrchROI, cv::COLOR_BGR2GRAY );

    cv::Mat matTmplGray;
    if ( pstCmd->matTmpl.channels() > 1 )
        cv::cvtColor ( pstCmd->matTmpl, matTmplGray, cv::COLOR_BGR2GRAY );
    else
        matTmplGray = pstCmd->matTmpl.clone();

    float fCorrelation;
    pstRpy->enStatus = _matchTemplate ( matSrchROI, matTmplGray, pstCmd->enMotion, pstRpy->ptObjPos, pstRpy->fRotation, fCorrelation );

    pstRpy->ptObjPos.x += pstCmd->rectSrchWindow.x;
    pstRpy->ptObjPos.y += pstCmd->rectSrchWindow.y;
    pstRpy->fMatchScore = fCorrelation * ConstToPercentage;
    if ( pstCmd->matInputImg.channels() > 1 )
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor ( pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);

    cv::Mat matWarp = cv::getRotationMatrix2D( cv::Point(0, 0), pstRpy->fRotation, 1. );
    float fCrossSize = 10.f;
    cv::Point2f crossLineOnePtOne(-fCrossSize, 0), crossLineOnePtTwo(fCrossSize, 0), crossLineTwoPtOne(0, -fCrossSize), crossLineTwoPtTwo(0, fCrossSize);
    crossLineOnePtOne = pstRpy->ptObjPos + CalcUtils::warpPoint<double>( matWarp, crossLineOnePtOne );
    crossLineOnePtTwo = pstRpy->ptObjPos + CalcUtils::warpPoint<double>( matWarp, crossLineOnePtTwo );
    crossLineTwoPtOne = pstRpy->ptObjPos + CalcUtils::warpPoint<double>( matWarp, crossLineTwoPtOne );
    crossLineTwoPtTwo = pstRpy->ptObjPos + CalcUtils::warpPoint<double>( matWarp, crossLineTwoPtTwo );

    cv::circle ( pstRpy->matResultImg, pstRpy->ptObjPos, 2, cv::Scalar(0, 0, 255), 1 );
    cv::line ( pstRpy->matResultImg, crossLineOnePtOne, crossLineOnePtTwo, cv::Scalar(0, 0, 255), 1 );
    cv::line ( pstRpy->matResultImg, crossLineTwoPtOne, crossLineTwoPtTwo, cv::Scalar(0, 0, 255), 1 );

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

VisionStatus VisionAlgorithm::inspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy)
{
    MARK_FUNCTION_START_TIME;
	if ( NULL == pInspCmd || NULL == pInspRpy )
		return VisionStatus::INVALID_PARAM;

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

	cv::Mat matRotation = cv::getRotationMatrix2D ( pInspCmd->ptObjPos, -pInspCmd->fRotation, 1 );
	cv::Mat matRotatedTmpl;
	cv::warpAffine( matTmplFullRange, matRotatedTmpl, matRotation, matTmplFullRange.size() );

	cv::Mat matRotatedMask;
	cv::warpAffine( matMask, matRotatedMask, matRotation, matMask.size() );
	matRotatedMask =  cv::Scalar::all(255) - matRotatedMask;

	pInspCmd->matInsp.setTo(cv::Scalar(0), matRotatedMask );

	cv::Mat matCmpResult = pInspCmd->matInsp - matRotatedTmpl;
	cv::Mat matRevsCmdResult = matRotatedTmpl - pInspCmd->matInsp;

    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
        cv::imshow("Mask", matRotatedMask );
        cv::imshow("Masked Inspection", pInspCmd->matInsp );
        cv::imshow("Rotated template", matRotatedTmpl);
        cv::imshow("Compare result", matCmpResult);
        cv::imshow("Compare result Reverse", matRevsCmdResult);
        cv::waitKey(0);
    }

	pInspRpy->n16NDefect = 0;
	_findBlob(matCmpResult, matRevsCmdResult, pInspCmd, pInspRpy);
	_findLine(matCmpResult, pInspCmd, pInspRpy );
    MARK_FUNCTION_END_TIME;
	return VisionStatus::OK;
}

int VisionAlgorithm::_findBlob(const cv::Mat &mat, const cv::Mat &matRevs, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy)
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

		std::vector<cv::Mat> vecInput;
		if ( PR_DEFECT_ATTRIBUTE::BRIGHT == stDefectCriteria.enAttribute ) {
			vecInput.push_back(mat);
		}else if ( PR_DEFECT_ATTRIBUTE::DARK == stDefectCriteria.enAttribute ) {
			vecInput.push_back(matRevs);
		}else if(PR_DEFECT_ATTRIBUTE::BOTH == stDefectCriteria.enAttribute ) {
            vecInput.push_back(mat);
            vecInput.push_back(matRevs);
		}	
		
		VectorOfVectorKeyPoint keyPoints;
		detector->detect ( vecInput, keyPoints, pInspCmd->matMask );

        //VectorOfKeyPoint vecKeyPoint = keyPoints[0];
        for (const auto &vecKeyPoint : keyPoints) {
            if (vecKeyPoint.size() > 0 && pInspRpy->n16NDefect < MAX_NUM_OF_DEFECT_RESULT) {
                for ( const auto &keyPoint : vecKeyPoint )
                {
                    pInspRpy->astDefect[pInspRpy->n16NDefect].enType = PR_DEFECT_TYPE::BLOB;
                    float fRadius = keyPoint.size / 2;
                    pInspRpy->astDefect[pInspRpy->n16NDefect].fRadius = fRadius;
                    pInspRpy->astDefect[pInspRpy->n16NDefect].fArea = (float)CV_PI * fRadius * fRadius; //Area = PI * R * R
                    ++ pInspRpy->n16NDefect;
                }
                drawKeypoints( im_with_keypoints, vecKeyPoint, im_with_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );	
            }
        }
	}

	cv::imshow("Find blob result", im_with_keypoints);
	cv::waitKey(0);
	return 0;
}

int VisionAlgorithm::_findLine(const cv::Mat &mat, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy)
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

        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )   {
            cv::imshow("Sobel Result", matSobelResult);
            cv::waitKey(0);
        }

		cv::Mat color_dst;
		cv::cvtColor(matSobelResult, color_dst, CV_GRAY2BGR);

		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP ( matSobelResult, lines, 1, CV_PI / 180, 30, stDefectCriteria.fLength, 10 );

		//_mergeLines ( )
		std::vector<PR_Line2f> vecLines, vecLinesAfterMerge;
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
				pInspRpy->astDefect[pInspRpy->n16NDefect].fLength = CalcUtils::distanceOf2Point<float>(it.pt1, it.pt2);
				++pInspRpy->n16NDefect;
			}
		}
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
            showImage("Detected Lines", color_dst);
	}
    MARK_FUNCTION_END_TIME;
	return 0;
}

/*static*/ int VisionAlgorithm::_merge2Line(const PR_Line2f &line1, const PR_Line2f &line2, PR_Line2f &lineResult) {
	float fLineSlope1 = ( line1.pt2.y - line1.pt1.y ) / ( line1.pt2.x - line1.pt1.x );
	float fLineSlope2 = ( line2.pt2.y - line2.pt1.y ) / ( line2.pt2.x - line2.pt1.x );

	if ( ( fabs(fLineSlope1) < 10.f || fabs(fLineSlope2) < 10.f )
        && fabs ( fLineSlope1 - fLineSlope2 ) > PARALLEL_LINE_SLOPE_DIFF_LMT )
		return -1;
	else if ( ( fabs(fLineSlope1) > 10.f || fabs(fLineSlope2) > 10.f )
        && fabs ( 1 / fLineSlope1 - 1 / fLineSlope2 ) > PARALLEL_LINE_SLOPE_DIFF_LMT )
		return -1;

    //Find the intersection point with Y axis.
	float fLineCrossWithY1 = fLineSlope1 * ( - line1.pt1.x) + line1.pt1.y;
	float fLineCrossWithY2 = fLineSlope2 * ( - line2.pt1.x) + line2.pt1.y;

	float fPerpendicularLineSlope1 =  - 1.f / fLineSlope1;
    float fPerpendicularLineSlope2 =  - 1.f / fLineSlope2;
    float fDistanceOfLine = 0.f;
    const float HORIZONTAL_LINE_SLOPE = 0.1f;
    if (fabs(fPerpendicularLineSlope1) < HORIZONTAL_LINE_SLOPE || fabs(fPerpendicularLineSlope2) < HORIZONTAL_LINE_SLOPE )
        fDistanceOfLine = fabs ( ( line1.pt1.x + line1.pt2.x ) / 2.f - ( line2.pt1.x + line2.pt2.x ) / 2.f );
    else if (fabs(fPerpendicularLineSlope1) > 100 || fabs(fPerpendicularLineSlope2) > 100 )
        fDistanceOfLine = fabs ( ( line1.pt1.y + line1.pt2.y ) / 2.f - ( line2.pt1.y + line2.pt2.y ) / 2.f );
    else {
        cv::Point2f ptCrossPointWithLine1, ptCrossPointWithLine2;

        //Find perpendicular line, get the cross points with two lines, then can get the distance of two lines.
        //The perpendicular line assume always cross origin(0, 0), so it can express as y = k * x.
        ptCrossPointWithLine1.x = fLineCrossWithY1 / (fPerpendicularLineSlope1 - fLineSlope1);
        ptCrossPointWithLine1.y = fPerpendicularLineSlope1 * ptCrossPointWithLine1.x;

        ptCrossPointWithLine2.x = fLineCrossWithY2 / (fPerpendicularLineSlope2 - fLineSlope2);
        ptCrossPointWithLine2.y = fPerpendicularLineSlope2 * ptCrossPointWithLine2.x;

        fDistanceOfLine = CalcUtils::distanceOf2Point<float>(ptCrossPointWithLine1, ptCrossPointWithLine2);
    }
    
	if ( fDistanceOfLine > PARALLEL_LINE_MERGE_DIST_LMT )
		return -1;

    std::vector<cv::Point> vecPoint;
    vecPoint.push_back ( line1.pt1 );
    vecPoint.push_back ( line1.pt2 );
    vecPoint.push_back ( line2.pt1 );
    vecPoint.push_back ( line2.pt2 );
    //{line1.pt1, line1.pt2, line2.pt1, line2.pt2 }
    if ( fabs ( fLineSlope1 ) < HORIZONTAL_LINE_SLOPE && fabs ( fLineSlope2 ) < HORIZONTAL_LINE_SLOPE ) {
        lineResult.pt1 = *std::min_element ( vecPoint.begin(), vecPoint.end(), []( const cv::Point &pt1, const cv::Point &pt2 ) { return pt1.x < pt2.x; } );
        lineResult.pt2 = *std::max_element ( vecPoint.begin(), vecPoint.end(), []( const cv::Point &pt1, const cv::Point &pt2 ) { return pt1.x < pt2.x; } );
        float fAverageY = ( line1.pt1.y + line1.pt2.y + line2.pt1.y + line2.pt2.y ) / 4.f;
        lineResult.pt1.y = fAverageY;
        lineResult.pt2.y = fAverageY;
        return 0;
    }

	std::vector<float> vecPerpendicularLineB;
	//Find the expression of perpendicular lines y = a * x + b
	float fPerpendicularLineB1 = line1.pt1.y - fPerpendicularLineSlope1 * line1.pt1.x;
	vecPerpendicularLineB.push_back ( fPerpendicularLineB1 );
	float fPerpendicularLineB2 = line1.pt2.y - fPerpendicularLineSlope1 * line1.pt2.x;
	vecPerpendicularLineB.push_back ( fPerpendicularLineB2 );
	float fPerpendicularLineB3 = line2.pt1.y - fPerpendicularLineSlope2 * line2.pt1.x;
	vecPerpendicularLineB.push_back ( fPerpendicularLineB3 );
	float fPerpendicularLineB4 = line2.pt2.y - fPerpendicularLineSlope2 * line2.pt2.x;
	vecPerpendicularLineB.push_back ( fPerpendicularLineB4 );

	size_t nMaxIndex = 0, nMinIndex = 0;
	float fMaxB = fPerpendicularLineB1;
	float fMinB = fPerpendicularLineB1;
	for ( size_t i = 1; i < vecPerpendicularLineB.size(); ++ i ) {
		if ( vecPerpendicularLineB[i] > fMaxB )	{
			fMaxB = vecPerpendicularLineB[i];
			nMaxIndex = i;
		}

		if ( vecPerpendicularLineB[i] < fMinB )	{
			fMinB = vecPerpendicularLineB[i];
			nMinIndex = i;
		}
	}
    lineResult.pt1 = vecPoint[nMinIndex];
    lineResult.pt2 = vecPoint[nMaxIndex];
	return static_cast<int>(VisionStatus::OK);
}

/*static*/ int VisionAlgorithm::_mergeLines(const std::vector<PR_Line2f> &vecLines, std::vector<PR_Line2f> &vecResultLines)
{
	if ( vecLines.size() < 2 )  {
        vecResultLines = vecLines;
        return 0;
    }		

	std::vector<int> vecMerged(vecLines.size(), false);
	
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

/*static*/ VisionStatus VisionAlgorithm::_writeLrnObjRecord(PR_LRN_OBJ_RPY *const pstRpy)
{
    ObjRecordPtr ptrRecord = std::make_shared<ObjRecord>(PR_RECORD_TYPE::OBJECT);
    ptrRecord->setModelKeyPoint(pstRpy->vecKeyPoint);
    ptrRecord->setModelDescriptor(pstRpy->matDescritor);
    ptrRecord->setObjCenter ( cv::Point2f ( ToFloat( pstRpy->matTmpl.cols ) / 2.f , ToFloat( pstRpy->matTmpl.rows ) / 2.f ) );
    RecordManager::getInstance()->add ( ptrRecord, pstRpy->nRecordId );
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::_writeDeviceRecord(PR_LRN_DEVICE_RPY *pLrnDeviceRpy)
{
    DeviceRecordPtr ptrRecord = std::make_shared<DeviceRecord>( PR_RECORD_TYPE::DEVICE );
    ptrRecord->setElectrodeThreshold ( pLrnDeviceRpy->nElectrodeThreshold );
    ptrRecord->setSize ( pLrnDeviceRpy->sizeDevice );
    RecordManager::getInstance()->add( ptrRecord, pLrnDeviceRpy->nRecordId );
    return VisionStatus::OK;
}

/*static*/ void VisionAlgorithm::showImage(String windowName, const cv::Mat &mat)
{
    static Int32 nImageCount = 0;
    String strWindowName = windowName + "_" + std::to_string(nImageCount ++ );
    cv::namedWindow ( strWindowName );
    cv::imshow( strWindowName, mat );
    cv::waitKey(0);
    cv::destroyWindow ( strWindowName );
}

/*static*/ VisionStatus VisionAlgorithm::runLogCase(const String &strFilePath)
{
    if ( strFilePath.length() < 2 )
        return VisionStatus::INVALID_PARAM;

    if (  ! bfs::exists ( strFilePath ) )
        return VisionStatus::PATH_NOT_EXIST;

    VisionStatus enStatus = VisionStatus::OK;
    auto strLocalPath = LogCase::unzip ( strFilePath );

    char chFolderToken = '\\';
    if ( strLocalPath.find('/') != std::string::npos )
        chFolderToken = '/';

    if ( strLocalPath[ strLocalPath.length() - 1 ] != chFolderToken )
        strLocalPath.append ( std::string { chFolderToken } );

    auto pos = strLocalPath.rfind ( chFolderToken, strLocalPath.length() - 2 );
    if ( String::npos == pos )
        return VisionStatus::INVALID_PARAM;

    auto folderName = strLocalPath.substr ( pos + 1 );
    pos = folderName.find('_');
    auto folderPrefix = folderName.substr(0, pos);

    std::unique_ptr<LogCase> pLogCase = nullptr;
    if ( LogCaseLrnObj::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique <LogCaseLrnObj>( strLocalPath, true );
    else if (LogCaseSrchObj::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique <LogCaseSrchObj>( strLocalPath, true );
    else if (LogCaseFitCircle::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique <LogCaseFitCircle>( strLocalPath, true );
    else if (LogCaseFitLine::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseFitLine>( strLocalPath, true );
    else if (LogCaseCaliper::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseCaliper>( strLocalPath, true );
    else if (LogCaseFitParallelLine::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseFitParallelLine>( strLocalPath, true );
    else if (LogCaseFitRect::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseFitRect>( strLocalPath, true );
    else if (LogCaseFindEdge::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseFindEdge>( strLocalPath, true );
    else if (LogCaseSrchFiducial::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseSrchFiducial>( strLocalPath, true );
    else if (LogCaseOcr::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseOcr>( strLocalPath, true );
    else if (LogCaseRemoveCC::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseRemoveCC>( strLocalPath, true );
    else if (LogCaseDetectEdge::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseDetectEdge>( strLocalPath, true );
    else if (LogCaseInspCircle::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseInspCircle>( strLocalPath, true );
    else if (LogCaseAutoThreshold::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseAutoThreshold>( strLocalPath, true );
    else if (LogCaseFillHole::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseFillHole>( strLocalPath, true );
    else if (LogCaseMatchTmpl::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseMatchTmpl>( strLocalPath, true );
    else if (LogCasePickColor::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCasePickColor>( strLocalPath, true );
    else if (LogCaseCalibrateCamera::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseCalibrateCamera>( strLocalPath, true );
    else if (LogCaseRestoreImg::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseRestoreImg>( strLocalPath, true );
    else if (LogCaseAutoLocateLead::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseAutoLocateLead>( strLocalPath, true );
    else if (LogCaseInspBridge::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseInspBridge>( strLocalPath, true );
    else if (LogCaseInspChip::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseInspChip>( strLocalPath, true );
    else if (LogCaseLrnContour::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseLrnContour>( strLocalPath, true );
    else if (LogCaseInspContour::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseInspContour>( strLocalPath, true );

    if ( nullptr != pLogCase )
        enStatus = pLogCase->RunLogCase();
    else
        enStatus = VisionStatus::INVALID_LOGCASE;
    return enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_refineSrchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation, float &fCorrelation)
{
    cv::Mat matWarp = cv::Mat::eye(2, 3, CV_32FC1);
    matWarp.at<float>(0,2) = ptResult.x;
    matWarp.at<float>(1,2) = ptResult.y;
    int number_of_iterations = 200;
    double termination_eps = 0.001;

    fCorrelation = ToFloat ( cv::findTransformECC ( matTmpl, mat, matWarp, ToInt32(enMotion), cv::TermCriteria ( cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
        number_of_iterations, termination_eps) ) );
    ptResult.x = matWarp.at<float>(0, 2);
    ptResult.y = matWarp.at<float>(1, 2);
    fRotation = ToFloat ( CalcUtils::radian2Degree ( asin ( matWarp.at<float>( 0, 1 ) ) ) );

    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_matchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation, float &fCorrelation)
{
    cv::Mat img_display, matResultImg;
    const int match_method = CV_TM_SQDIFF;
    fRotation = 0.f;

    mat.copyTo(img_display);

    /// Create the result matrix
    int result_cols = mat.cols - matTmpl.cols + 1;
    int result_rows = mat.rows - matTmpl.rows + 1;

    matResultImg.create(result_rows, result_cols, CV_32FC1);

    /// Do the Matching and Normalize
    cv::matchTemplate(mat, matTmpl, matResultImg, match_method);
    cv::normalize ( matResultImg, matResultImg, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal;
    cv::Point minLoc, maxLoc, matchLoc;

    cv::minMaxLoc(matResultImg, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
        matchLoc = minLoc;
    else
        matchLoc = maxLoc;

    ptResult.x = (float)matchLoc.x;
    ptResult.y = (float)matchLoc.y;
    try {
        _refineSrchTemplate ( mat, matTmpl, enMotion, ptResult, fRotation, fCorrelation );
    }catch (std::exception) {
        return VisionStatus::OPENCV_EXCEPTION;
    }

    ptResult.x += (float)( matTmpl.cols / 2 + 0.5 );
    ptResult.y += (float)( matTmpl.rows / 2 + 0.5 );
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::srchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy, bool bReplay)
{
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    cv::Point2f ptResult;

    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty");
        return VisionStatus::INVALID_PARAM;
    }

    if ( pstCmd->rectSrchRange.x < 0 ||
         pstCmd->rectSrchRange.y < 0 ||
        (pstCmd->rectSrchRange.x + pstCmd->rectSrchRange.width) > pstCmd->matInputImg.cols || 
        (pstCmd->rectSrchRange.y + pstCmd->rectSrchRange.height) > pstCmd->matInputImg.rows )
    {
        _snprintf(charrMsg, sizeof(charrMsg), "Search range is invalid, rect x = %d, y = %d, width = %d, height = %d", 
                  pstCmd->rectSrchRange.x, pstCmd->rectSrchRange.y, pstCmd->rectSrchRange.width, pstCmd->rectSrchRange.height );
        WriteLog(charrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseSrchFiducial);    

    auto nTmplSize = ToInt32 ( pstCmd->fSize + pstCmd->fMargin * 2 );
    cv::Mat matTmpl = cv::Mat::zeros(nTmplSize, nTmplSize, CV_8UC1);
    if ( PR_FIDUCIAL_MARK_TYPE::SQUARE == pstCmd->enType )   {
        cv::rectangle(matTmpl,
                      cv::Rect2f(pstCmd->fMargin, pstCmd->fMargin, pstCmd->fSize, pstCmd->fSize),
                      cv::Scalar::all(256),
                      CV_FILLED );
    }
    else if ( PR_FIDUCIAL_MARK_TYPE::CIRCLE == pstCmd->enType )
    {
        auto radius = ToInt32 ( pstCmd->fSize / 2.f );
        cv::circle ( matTmpl, cv::Point( ToInt32 ( pstCmd->fMargin ) + radius, ToInt32 ( pstCmd->fMargin ) + radius ), radius, cv::Scalar::all(256), -1);
    }
    else
    {
        WriteLog("Not supported fiducial mark type");
        return VisionStatus::INVALID_PARAM;
    }  

    cv::Mat matSrchROI( pstCmd->matInputImg, pstCmd->rectSrchRange );
    if ( matSrchROI.channels() > 1 )
        cv::cvtColor(matSrchROI, matSrchROI, cv::COLOR_BGR2GRAY);

    float fRotation = 0.f, fCorrelation = 0.f;
    pstRpy->enStatus = _matchTemplate ( matSrchROI, matTmpl, PR_OBJECT_MOTION::TRANSLATION, ptResult, fRotation, fCorrelation );

    pstRpy->ptPos.x = ptResult.x + pstCmd->rectSrchRange.x;
    pstRpy->ptPos.y = ptResult.y + pstCmd->rectSrchRange.y;
    pstRpy->fScore = fCorrelation * ConstToPercentage;

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::circle(pstRpy->matResultImg, pstRpy->ptPos, 2, cv::Scalar(255, 0, 0), 2);
    cv::rectangle(pstRpy->matResultImg, cv::Rect(ToInt32(pstRpy->ptPos.x) - nTmplSize / 2, ToInt32(pstRpy->ptPos.y) - nTmplSize / 2, nTmplSize, nTmplSize), cv::Scalar(255, 0, 0), 1);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VectorOfPoint VisionAlgorithm::_findPointInRegionOverThreshold(const cv::Mat &mat, const cv::Rect &rect, int nThreshold)
{
    VectorOfPoint vecPoint;
    if ( mat.empty() )
        return vecPoint;

    cv::Mat matROI(mat, rect);
    cv::Mat matThreshold;
   
    cv::threshold ( matROI, matThreshold, nThreshold, 256, CV_THRESH_BINARY );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("threshold result", matThreshold);
    std::vector<cv::Point> vecPoints;
    vecPoints.reserve(100000);
    cv::findNonZero( matThreshold, vecPoints );
    for ( const auto &point : vecPoints )   {
        cv::Point point2f ( point.x + rect.x, point.y + rect.y );
        vecPoint.push_back(point2f);
    }
    return vecPoint;
}

/*static*/ ListOfPoint VisionAlgorithm::_findPointsInRegionByThreshold(const cv::Mat &mat, const cv::Rect &rect, int nThreshold, PR_OBJECT_ATTRIBUTE enAttribute)
{
    ListOfPoint listPoint;
    if ( mat.empty() )
        return listPoint;

    cv::Mat matROI(mat, rect);
    cv::Mat matThreshold;

    cv::ThresholdTypes enThresType = cv::THRESH_BINARY;
    if ( PR_OBJECT_ATTRIBUTE::DARK == enAttribute )
        enThresType = cv::THRESH_BINARY_INV;
    cv::threshold ( matROI, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, enThresType );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("threshold result", matThreshold);
    std::vector<cv::Point> vecPoints;
    vecPoints.reserve(100000);
    cv::findNonZero( matThreshold, vecPoints );
    for ( const auto &point : vecPoints )   {
        cv::Point point2f ( point.x + rect.x, point.y + rect.y );
        listPoint.push_back(point2f);
    }
    return listPoint;
}

/*static*/ ListOfPoint VisionAlgorithm::_findPointsInRegion(const cv::Mat &mat, const cv::Rect &rect)
{
    ListOfPoint listPoint;
    if ( mat.empty() )
        return listPoint;

    cv::Mat matROI(mat, rect);
    cv::Mat matThreshold;
    
    std::vector<cv::Point> vecPoints;
    vecPoints.reserve(100000);
    cv::findNonZero( matThreshold, vecPoints );
    for ( const auto &point : vecPoints )   {
        cv::Point point2f ( point.x + rect.x, point.y + rect.y );
        listPoint.push_back(point2f);
    }
    return listPoint;
}

/*static*/ std::vector<size_t> VisionAlgorithm::_findPointOverLineTol(const VectorOfPoint   &vecPoint,
                                                                      bool                   bReversedFit,
                                                                      const float            fSlope,
                                                                      const float            fIntercept,
                                                                      PR_RM_FIT_NOISE_METHOD method,
                                                                      float                  tolerance)
{
    std::vector<size_t> vecResult;
    for ( size_t i = 0;i < vecPoint.size(); ++ i )  {
        cv::Point2f point = vecPoint[i];
        auto disToLine = CalcUtils::ptDisToLine ( point, bReversedFit, fSlope, fIntercept );
        if ( PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR == method && fabs ( disToLine ) > tolerance )  {
            vecResult.push_back(i);
        }else if ( PR_RM_FIT_NOISE_METHOD::POSITIVE_ERR == method && disToLine > tolerance ) {
            vecResult.push_back(i);
        }else if ( PR_RM_FIT_NOISE_METHOD::NEGATIVE_ERR == method && disToLine < -tolerance ) {
            vecResult.push_back(i);
        }
    }
    return vecResult;
}

/*static*/ std::vector<ListOfPoint::const_iterator> VisionAlgorithm::_findPointOverLineTol(const ListOfPoint       &listPoint,
                                                                                           bool                     bReversedFit,
                                                                                           const float              fSlope,
                                                                                           const float              fIntercept,
                                                                                           PR_RM_FIT_NOISE_METHOD   method,
                                                                                           float                    tolerance)
{
    std::vector<ListOfPoint::const_iterator> vecResult;
    for ( ListOfPoint::const_iterator it = listPoint.begin(); it != listPoint.end(); ++ it )  {
        cv::Point2f point = *it;
        auto disToLine = CalcUtils::ptDisToLine ( point, bReversedFit, fSlope, fIntercept );
        if ( PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR == method && fabs ( disToLine ) > tolerance )  {
            vecResult.push_back(it);
        }else if ( PR_RM_FIT_NOISE_METHOD::POSITIVE_ERR == method && disToLine > tolerance ) {
            vecResult.push_back(it);
        }else if ( PR_RM_FIT_NOISE_METHOD::NEGATIVE_ERR == method && disToLine < -tolerance ) {
            vecResult.push_back(it);
        }
    }
    return vecResult;
}

/*static*/ VisionStatus VisionAlgorithm::fitLine(const PR_FIT_LINE_CMD *const pstCmd, PR_FIT_LINE_RPY *const pstRpy, bool bReplay /*= false*/ )
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char charrMsg[1000];

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    
    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The ROI rect (%d, %d, %d, %d) is invalid",
            pstCmd->rectROI.x, pstCmd->rectROI.y, pstCmd->rectROI.width, pstCmd->rectROI.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( ! pstCmd->matMask.empty() ) {
        if ( pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols ) {
            _snprintf(charrMsg, sizeof(charrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d )",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(charrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if ( pstCmd->matMask.channels() != 1 )  {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFitLine);   

    cv::Mat matROI ( pstCmd->matInputImg, pstCmd->rectROI);  
    cv::Mat matGray, matThreshold;
    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();

    if ( pstCmd->bPreprocessed ) {
        matThreshold = matGray;
    }else {
        cv::ThresholdTypes enThresType = cv::THRESH_BINARY;
        if ( PR_OBJECT_ATTRIBUTE::DARK == pstCmd->enAttribute )
            enThresType = cv::THRESH_BINARY_INV;

        cv::threshold(matGray, matThreshold, pstCmd->nThreshold, PR_MAX_GRAY_LEVEL, enThresType);
        CV_Assert(matThreshold.channels() == 1);

        if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
            showImage("Threshold image", matThreshold);
    }

    if ( ! pstCmd->matMask.empty() ) {
        cv::Mat matROIMask ( pstCmd->matMask, pstCmd->rectROI );
        matROIMask = cv::Scalar(255) - matROIMask;
        matThreshold.setTo(cv::Scalar(0), matROIMask);
    }    

    VectorOfPoint vecPoints;
    cv::findNonZero( matThreshold, vecPoints );
    if ( vecPoints.size() < 2 )  {
        WriteLog("Not enough points to fit line");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    
    if ( vecPoints.size() > PR_FIT_LINE_MAX_POINT_COUNT ) {
        _snprintf ( charrMsg, sizeof(charrMsg), "Too much noise to fit line, points number %d", vecPoints.size() );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    std::vector<int> vecX, vecY;
    for ( auto &point : vecPoints ) {
        vecX.push_back ( point.x );
        vecY.push_back ( point.y );
        point = cv::Point ( point.x + pstCmd->rectROI.x, point.y + pstCmd->rectROI.y );
    }

    //For y = kx + b, the k is the slope, when it is very large, the error is quite big, so change
    //to get the x = k1 + b1, and get the k = 1 / k1, b = -b1 / k1. Then the result is more accurate.
    pstRpy->bReversedFit = false;
    if ( CalcUtils::calcStdDeviation ( vecY ) > CalcUtils::calcStdDeviation ( vecX ) )
        pstRpy->bReversedFit = true;

    if ( PR_FIT_METHOD::RANSAC == pstCmd->enMethod ) {
        _fitLineRansac ( vecPoints, pstCmd->fErrTol, pstCmd->nMaxRansacTime, pstCmd->nNumOfPtToFinishRansac, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->stLine );
    }else if ( PR_FIT_METHOD::LEAST_SQUARE_REFINE == pstCmd->enMethod ) {
        pstRpy->enStatus = _fitLineRefine ( vecPoints, pstCmd->enRmNoiseMethod, pstCmd->fErrTol, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->stLine );
        if ( VisionStatus::OK != pstRpy->enStatus ) {
            FINISH_LOGCASE;
            return pstRpy->enStatus;
        }
    }else if ( PR_FIT_METHOD::LEAST_SQUARE == pstCmd->enMethod ) {
        Fitting::fitLine ( vecPoints, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->bReversedFit );
        pstRpy->stLine = CalcUtils::calcEndPointOfLine ( vecPoints, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept );
    }
    
    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::line ( pstRpy->matResultImg, pstRpy->stLine.pt1, pstRpy->stLine.pt2, cv::Scalar(255,0,0), 2 );
    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;    
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VectorOfPoint VisionAlgorithm::_findPointInLineTol(const VectorOfPoint   &vecPoint,
                                                              bool                   bReversedFit,
                                                              const float            fSlope,
                                                              const float            fIntercept,
                                                              float                  fTolerance)
{
    VectorOfPoint vecPointInTol;
    for ( const auto &point : vecPoint ) {
        auto disToLine = CalcUtils::ptDisToLine ( point, bReversedFit, fSlope, fIntercept );
        if ( fabs ( disToLine ) <= fTolerance )
            vecPointInTol.push_back( point );
    }       
    return vecPointInTol;
}

/*static*/ void VisionAlgorithm::_fitLineRansac(const VectorOfPoint &vecPoints,
                                                float                fTolerance,
                                                int                  nMaxRansacTime,
                                                size_t               nFinishThreshold,
                                                bool                 bReversedFit,
                                                float               &fSlope,
                                                float               &fIntercept,
                                                PR_Line2f           &stLine) {
    int nRansacTime = 0;
    const int LINE_RANSAC_POINT = 2;
    size_t nMaxConsentNum = 0;

    while ( nRansacTime < nMaxRansacTime ) {
        VectorOfPoint vecSelectedPoints = _randomSelectPoints ( vecPoints, LINE_RANSAC_POINT );
        Fitting::fitLine ( vecSelectedPoints, fSlope, fIntercept, bReversedFit );
        vecSelectedPoints = _findPointInLineTol ( vecPoints, bReversedFit, fSlope, fIntercept, fTolerance );

        if ( vecSelectedPoints.size() >= nFinishThreshold ) {
            Fitting::fitLine( vecSelectedPoints, fSlope, fIntercept, bReversedFit );
            stLine = CalcUtils::calcEndPointOfLine ( vecSelectedPoints, bReversedFit, fSlope, fIntercept );
            return;
        }
        
        if ( vecSelectedPoints.size() > nMaxConsentNum ) {
            Fitting::fitLine( vecSelectedPoints, fSlope, fIntercept, bReversedFit );
            stLine = CalcUtils::calcEndPointOfLine ( vecSelectedPoints, bReversedFit, fSlope, fIntercept );
            nMaxConsentNum = vecSelectedPoints.size();
        }
        ++ nRansacTime;
    }
}

/*static*/ VisionStatus VisionAlgorithm::_fitLineRefine(const VectorOfPoint    &vecPoints,
                                                        PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod,
                                                        float                   fTolerance,
                                                        bool                    bReversedFit,
                                                        float                  &fSlope,
                                                        float                  &fIntercept,
                                                        PR_Line2f              &stLine)
{
    ListOfPoint listPoint;
    for ( const auto &point : vecPoints )
        listPoint.push_back ( point );

    int nIteratorNum = 0;
    std::vector<ListOfPoint::const_iterator> vecOverTolIter;
    do {
        for (const auto &it : vecOverTolIter )
            listPoint.erase( it );

        Fitting::fitLine ( listPoint, fSlope, fIntercept, bReversedFit );
        vecOverTolIter = _findPointOverLineTol(listPoint, bReversedFit, fSlope, fIntercept, enRmNoiseMethod, fTolerance);
    }while ( ! vecOverTolIter.empty() && nIteratorNum < 20 );

    if ( listPoint.size() < 2 )
        return VisionStatus::TOO_MUCH_NOISE_TO_FIT; 

    stLine = CalcUtils::calcEndPointOfLine ( listPoint, bReversedFit, fSlope, fIntercept );
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::caliper(const PR_CALIPER_CMD *const pstCmd, PR_CALIPER_RPY *const pstRpy, bool bReplay /* = false*/ ) {
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char charrMsg [ 1000 ];

    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    
    if (pstCmd->rectRotatedROI.size.width <= 0 || pstCmd->rectRotatedROI.size.height <= 0 ) {
        WriteLog("The caliper ROI size is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( ! pstCmd->matMask.empty() ) {
        if ( pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols ) {
            _snprintf(charrMsg, sizeof(charrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d )",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(charrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if ( pstCmd->matMask.channels() != 1 ) {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if ( pstCmd->bCheckLinerity && ( pstCmd->fPointMaxOffset <= 0.f || pstCmd->fMinLinerity <= 0.f ) ) {
        _snprintf(charrMsg, sizeof(charrMsg), "Linerity check parameter is invalid. PointMaxOffset = %.2f, MinLinerity = %.2f",
            pstCmd->fPointMaxOffset, pstCmd->fMinLinerity);
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseCaliper);   

    cv::Mat matROI;
    cv::Rect2f rectNewROI ( pstCmd->rectRotatedROI.center.x -  pstCmd->rectRotatedROI.size.width / 2.f, pstCmd->rectRotatedROI.center.y -  pstCmd->rectRotatedROI.size.height / 2.f,
        pstCmd->rectRotatedROI.size.width, pstCmd->rectRotatedROI.size.height );
    pstRpy->enStatus = _extractRotatedROI ( pstCmd->matInputImg, pstCmd->rectRotatedROI, matROI );
    if ( pstRpy->enStatus != VisionStatus::OK )
        return pstRpy->enStatus;

    cv::Mat matGray, matThreshold, matROIMask;
    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();

    if ( ! pstCmd->matMask.empty() )
         _extractRotatedROI ( pstCmd->matMask, pstCmd->rectRotatedROI, matROIMask );

    VectorOfPoint vecFitPoint;
    if ( pstCmd->enAlgorithm == PR_CALIPER_ALGORITHM::PROJECTION )
        _caliperByProjection ( matGray, matROIMask, rectNewROI, pstCmd->enDetectDir, vecFitPoint, pstRpy );
    else if ( pstCmd->enAlgorithm == PR_CALIPER_ALGORITHM::SECTION_AVG_GUASSIAN_DIFF )
        _caliperBySectionAvgGussianDiff ( matGray, matROIMask, rectNewROI, pstCmd->enDetectDir, vecFitPoint, pstRpy );
    else {
        WriteLog("The caliper algorithm is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( pstRpy->enStatus != VisionStatus::OK ) {
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    const int MIN_DIST_FROM_LINE_TO_ROI_EDGE = 5;
    const float HORIZONTAL_SLOPE_TOL = 0.1f;
    if ( pstRpy->bReversedFit ) {
        if ( pstRpy->fSlope < HORIZONTAL_SLOPE_TOL && 
            (fabs ( pstRpy->stLine.pt1.x - rectNewROI.x ) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
             fabs ( pstRpy->stLine.pt2.x - rectNewROI.x ) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
             fabs ( pstRpy->stLine.pt1.x - rectNewROI.x - rectNewROI.width ) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
             fabs ( pstRpy->stLine.pt2.x - rectNewROI.x - rectNewROI.width ) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ) ) {
            pstRpy->enStatus = VisionStatus::CALIPER_CAN_NOT_FIND_LINE;
            FINISH_LOGCASE;
            return pstRpy->enStatus;
        }
    }else {
        if ( pstRpy->fSlope < HORIZONTAL_SLOPE_TOL &&
            (fabs ( pstRpy->stLine.pt1.y - rectNewROI.y ) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
             fabs ( pstRpy->stLine.pt2.y - rectNewROI.y ) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
             fabs ( pstRpy->stLine.pt1.y - rectNewROI.y - rectNewROI.height ) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
             fabs ( pstRpy->stLine.pt2.y - rectNewROI.y - rectNewROI.height ) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ) ) {
            pstRpy->enStatus = VisionStatus::CALIPER_CAN_NOT_FIND_LINE;
            FINISH_LOGCASE;
            return pstRpy->enStatus;
        }
    }

    pstRpy->bLinerityCheckPass = false;
    if ( pstCmd->bCheckLinerity ) {
        int nInTolerancePointCount = 0;
        for (const auto &point : vecFitPoint) {
            if ( CalcUtils::ptDisToLine ( point, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept ) < pstCmd->fPointMaxOffset )
                ++ nInTolerancePointCount;
        }
        pstRpy->fLinerity = nInTolerancePointCount * 100.f / vecFitPoint.size();
        if ( pstRpy->fLinerity >= pstCmd->fMinLinerity )
            pstRpy->bLinerityCheckPass = true;
    }else {
        pstRpy->fLinerity = 0.f;
        pstRpy->bLinerityCheckPass = false;
    }

    if ( fabs ( pstCmd->rectRotatedROI.angle ) > 0.1 ) {
        cv::Mat matWarpBack = cv::getRotationMatrix2D ( pstCmd->rectRotatedROI.center, -pstCmd->rectRotatedROI.angle, 1. );
        pstRpy->stLine.pt1 = CalcUtils::warpPoint<double> ( matWarpBack, pstRpy->stLine.pt1 );
        pstRpy->stLine.pt2 = CalcUtils::warpPoint<double> ( matWarpBack, pstRpy->stLine.pt2 );
        pstRpy->bReversedFit = false;
        CalcUtils::lineSlopeIntercept ( pstRpy->stLine, pstRpy->fSlope, pstRpy->fIntercept );
    }  

    pstRpy->bAngleCheckPass = true;
    pstRpy->fAngle = ToFloat ( CalcUtils::calcSlopeDegree<float> ( pstRpy->stLine.pt1, pstRpy->stLine.pt2 ) );
    if ( pstCmd->bCheckAngle && fabs ( pstRpy->fAngle - pstCmd->fExpectedAngle ) < pstCmd->fAngleDiffTolerance )
        pstRpy->bAngleCheckPass = true;
    else
        pstRpy->bAngleCheckPass = false;

    if ( pstCmd->matInputImg.channels() > 1 )
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor ( pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR );

    cv::line( pstRpy->matResultImg, pstRpy->stLine.pt1, pstRpy->stLine.pt2, _constBlueScalar, 2 );
    VectorOfVectorOfPoint vevVecPoints;
    vevVecPoints.push_back ( CalcUtils::getCornerOfRotatedRect ( pstCmd->rectRotatedROI ) );
    cv::polylines ( pstRpy->matResultImg, vevVecPoints, true, _constGreenScalar, 2 );
    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_extractRotatedROI(const cv::Mat &matInputImg, const cv::RotatedRect &rectRotatedROI, cv::Mat &matROI ) {
    if ( fabs ( rectRotatedROI.angle ) < 0.1 ) {
        cv::Rect rectBounding ( ToInt32 ( rectRotatedROI.center.x - rectRotatedROI.size.width / 2.f ), ToInt32 ( rectRotatedROI.center.y - rectRotatedROI.size.height / 2.f ),
            ToInt32 ( rectRotatedROI.size.width ), ToInt32 ( rectRotatedROI.size.height ) );
        if ( rectBounding.x < 0 || rectBounding.y < 0 ||
            rectBounding.width <= 0 || rectBounding.height <= 0 ||
            (rectBounding.x + rectBounding.width) > matInputImg.cols ||
            (rectBounding.y + rectBounding.height) > matInputImg.rows ) {
            char charrMsg[1000];
            _snprintf ( charrMsg, sizeof(charrMsg), "The ROI rect (%d, %d, %d, %d) is invalid",
                rectBounding.x,rectBounding.y, rectBounding.width, rectBounding.height );
            WriteLog ( charrMsg );
            return VisionStatus::INVALID_PARAM;
        }
        matROI = cv::Mat ( matInputImg, rectBounding );
        return VisionStatus::OK;
    }
    auto rectBounding = rectRotatedROI.boundingRect();
    if ( rectBounding.x < 0 ) rectBounding.x = 0;
    if ( rectBounding.y < 0 ) rectBounding.y = 0;
    if ( ( rectBounding.x + rectBounding.width ) > matInputImg.cols )
        rectBounding.width = matInputImg.cols - rectBounding.x;
    if ( ( rectBounding.y + rectBounding.height ) > matInputImg.rows )
        rectBounding.height = matInputImg.rows - rectBounding.y;

    cv::Mat matBoundingROI ( matInputImg, rectBounding );
    cv::Point2f ptNewCenter ( rectRotatedROI.center.x - rectBounding.x, rectRotatedROI.center.y - rectBounding.y ); 
    cv::Mat matWarp = cv::getRotationMatrix2D ( ptNewCenter, rectRotatedROI.angle, 1. );
    auto vecVec = CalcUtils::matToVector<double>(matWarp);
    auto rectWarppedPoly = CalcUtils::warpRect<double>( matWarp, rectBounding );
    auto rectWarppedPolyBounding = cv::boundingRect ( rectWarppedPoly );
    cv::Mat matWarpResult;
	cv::warpAffine( matBoundingROI, matWarpResult, matWarp, rectWarppedPolyBounding.size() );
    cv::Rect2f rectSubROI( ptNewCenter.x - rectRotatedROI.size.width / 2.f, ptNewCenter.y - rectRotatedROI.size.height / 2.f,
        rectRotatedROI.size.width, rectRotatedROI.size.height );
    if ( rectSubROI.x < 0 ) rectSubROI.x = 0;
    if ( rectSubROI.y < 0 ) rectSubROI.y = 0;
    if ( ( rectSubROI.x + rectSubROI.width ) > matWarpResult.cols )
        rectSubROI.width = matWarpResult.cols - rectSubROI.x;
    if ( ( rectSubROI.y + rectSubROI.height ) > matWarpResult.rows )
        rectSubROI.height = matWarpResult.rows - rectSubROI.y;
    matROI = cv::Mat( matWarpResult, rectSubROI );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
        showImage("Bouding ROI", matBoundingROI );
        showImage("Warpped ROI", matWarpResult );
        showImage("Result ROI", matROI );
    }    
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_calcCaliperDirection(const cv::Mat &matRoiImg, bool bReverseFit, PR_CALIPER_DIR &enDirection ) {
    cv::Mat matLeftTop, matRightBottom;
    if ( bReverseFit ) {
        matLeftTop =     cv::Mat ( matRoiImg, cv::Rect ( 0, 0, matRoiImg.cols / 2, matRoiImg.rows ) );
        matRightBottom = cv::Mat ( matRoiImg, cv::Rect ( matRoiImg.cols / 2, 0, matRoiImg.cols / 2, matRoiImg.rows ) );
    }else {
        matLeftTop =     cv::Mat ( matRoiImg, cv::Rect ( 0, 0, matRoiImg.cols, matRoiImg.rows / 2 ) );
        matRightBottom = cv::Mat ( matRoiImg, cv::Rect ( 0, matRoiImg.rows / 2, matRoiImg.cols, matRoiImg.rows / 2 ) );
    }
    auto sumOfLeftTop = cv::sum ( matLeftTop )[0];
    auto sumOfRightBottom = cv::sum ( matRightBottom )[0];
    if ( sumOfLeftTop < sumOfRightBottom )
        enDirection = PR_CALIPER_DIR::MIN_TO_MAX;
    else
        enDirection = PR_CALIPER_DIR::MAX_TO_MIN;
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_caliperByProjection(const cv::Mat &matGray, const cv::Mat &matROIMask, const cv::Rect &rectROI, PR_CALIPER_DIR enDetectDir, VectorOfPoint &vecFitPoint, PR_CALIPER_RPY *const pstRpy) {
    VectorOfPoint vecPoints;
    std::vector<int> vecX, vecY, vecIndexCanFormLine, vecProjection;
    VectorOfVectorOfPoint vecVecPoint;

    int nThreshold = _autoThreshold ( matGray, matROIMask );
    cv::Mat matThreshold;
    cv::threshold ( matGray, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, cv::THRESH_BINARY );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Calibper threshold image", matThreshold );

    if ( ! matROIMask.empty() ) {   
        cv::Mat matROIMaskReverse = cv::Scalar(PR_MAX_GRAY_LEVEL) - matROIMask;
        matThreshold.setTo ( cv::Scalar(0), matROIMaskReverse );
    }

    cv::findNonZero ( matThreshold, vecPoints );

    if ( vecPoints.size() < 2 ) {
        WriteLog("Not enough points to detect line");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        return pstRpy->enStatus;
    }

    for ( auto &point : vecPoints ) {
        vecX.push_back ( point.x );
        vecY.push_back ( point.y );
    }

    pstRpy->bReversedFit = false;
    if ( CalcUtils::calcStdDeviation ( vecY ) > CalcUtils::calcStdDeviation ( vecX ) )
        pstRpy->bReversedFit = true;

    if ( PR_CALIPER_DIR::AUTO == enDetectDir )
        _calcCaliperDirection ( matGray, pstRpy->bReversedFit, enDetectDir );

    auto projectSize = pstRpy->bReversedFit ? matGray.cols : matGray.rows;
    vecProjection.resize ( projectSize, 0 );
    for ( const auto &point : vecPoints ) {
        if ( pstRpy->bReversedFit )
            ++ vecProjection[point.x];
        else
            ++ vecProjection[point.y];
    }

    int maxValue = *std::max_element ( vecProjection.begin(), vecProjection.end() );
    
    //Find the possible coordinate can form lines.
    for ( size_t index = 0; index < vecProjection.size(); ++ index ) {
        if ( vecProjection[ index ] > 0.8 * maxValue )
            vecIndexCanFormLine.push_back ( ToInt32 ( index ) );
    }

    auto initialLinePos = 0;
    if ( PR_CALIPER_DIR::MIN_TO_MAX == enDetectDir )
        initialLinePos = vecIndexCanFormLine[0];
    else
        initialLinePos = vecIndexCanFormLine.back();

    int rs = std::max ( int ( 0.1 *(vecIndexCanFormLine.back() - vecIndexCanFormLine[0] ) ), 20 );
    
    if ( pstRpy->bReversedFit ) vecVecPoint.resize( matGray.rows );
    else                        vecVecPoint.resize( matGray.cols );

    //Find all the points around the initial line pos.
    for ( const auto &point : vecPoints ) {
        if (pstRpy->bReversedFit) {
            if ( point.x > initialLinePos - rs && point.x < initialLinePos + rs )
                vecVecPoint[point.y].push_back ( point );
        }else {
            if ( point.y > initialLinePos - rs && point.y < initialLinePos + rs )
                vecVecPoint[point.x].push_back ( point );
        }
    }

    //Find the extreme point on each row or column.
    for ( const auto &vecPointTmp : vecVecPoint ) {
        if ( ! vecPointTmp.empty() ) {
            auto point = PR_CALIPER_DIR::MIN_TO_MAX == enDetectDir ? vecPointTmp.front() : vecPointTmp.back();
            if ( pstRpy->bReversedFit ) {
                if (point.x != initialLinePos - rs && point.x != initialLinePos - rs)
                    vecFitPoint.push_back ( point + cv::Point ( rectROI.x, rectROI.y ) );
            }else {
                if (point.y != initialLinePos - rs && point.y != initialLinePos - rs)
                    vecFitPoint.push_back ( point + cv::Point ( rectROI.x, rectROI.y ) );
            }
        }
    }

    Fitting::fitLine ( vecFitPoint, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->bReversedFit );
    pstRpy->stLine = CalcUtils::calcEndPointOfLine ( vecFitPoint, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept );
    
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ int VisionAlgorithm::_findMaxDiffPosInX( const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection ) {
    cv::Mat matAverage;
    cv::reduce ( matInput, matAverage, 0, cv::ReduceTypes::REDUCE_AVG );

    cv::Mat matGuassianDiffResult;   
    CalcUtils::filter2D_Conv ( matAverage, matGuassianDiffResult, CV_32F, matGuassianDiffKernel );
    double minValue = 0., maxValue = 0.;
    cv::Point ptMin, ptMax;
    cv::minMaxLoc ( matGuassianDiffResult, &minValue, &maxValue, &ptMin, &ptMax );
    if ( PR_CALIPER_DIR::MIN_TO_MAX == enDirection )
        return ptMin.x;
    return ptMax.x;
}

/*static*/ int VisionAlgorithm::_findMaxDiffPosInY( const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection ) {
    cv::Mat matAverage;
    cv::reduce ( matInput, matAverage, 1, cv::ReduceTypes::REDUCE_AVG );

    cv::Mat matGuassianDiffResult;   
    CalcUtils::filter2D_Conv ( matAverage, matGuassianDiffResult, CV_32F, matGuassianDiffKernel );
    double minValue = 0., maxValue = 0.;
    cv::Point ptMin, ptMax;
    cv::minMaxLoc ( matGuassianDiffResult, &minValue, &maxValue, &ptMin, &ptMax );
    if ( PR_CALIPER_DIR::MIN_TO_MAX == enDirection )
        return ptMin.y;
    return ptMax.y;
}

VisionStatus VisionAlgorithm::_caliperBySectionAvgGussianDiff(const cv::Mat &matInputImg, const cv::Mat &matROIMask, const cv::Rect &rectROI, PR_CALIPER_DIR enDirection, VectorOfPoint &vecFitPoint, PR_CALIPER_RPY *const pstRpy) {
    const int DIVIDE_SECTION = 20;
    const int GUASSIAN_DIFF_WIDTH = 2;
    const float GUASSIAN_KERNEL_SSQ = 1.f;
    if ( matInputImg.rows > matInputImg.cols )
        pstRpy->bReversedFit = true;
    else
        pstRpy->bReversedFit = false;
    if ( PR_CALIPER_DIR::AUTO == enDirection )
        _calcCaliperDirection ( matInputImg, pstRpy->bReversedFit, enDirection );

    int COLS = matInputImg.cols;
    int ROWS = matInputImg.rows;
    
    cv::Mat matGuassinKernel = CalcUtils::generateGuassinDiffKernel ( GUASSIAN_DIFF_WIDTH, GUASSIAN_KERNEL_SSQ );

    if ( pstRpy->bReversedFit ) {
        auto nInterval = ToInt32 (ToFloat ( matInputImg.rows ) / ToFloat ( DIVIDE_SECTION ) + 0.5f );
        int nCurrentProcessedPos = 0;
        while ( nCurrentProcessedPos < matInputImg.rows ) {
            int nROISize = ( nCurrentProcessedPos + nInterval ) < matInputImg.rows ? nInterval : matInputImg.rows - nCurrentProcessedPos;
            cv::Mat matSubROI ( matInputImg, cv::Rect ( 0, nCurrentProcessedPos, COLS, nROISize ) );
            int nJumpPos = _findMaxDiffPosInX ( matSubROI, matGuassinKernel, enDirection );
            if ( nJumpPos > 0 ) {
                if ( nCurrentProcessedPos == 0 )
                    vecFitPoint.emplace_back ( nJumpPos, nCurrentProcessedPos );
                else if ( ( nCurrentProcessedPos + nInterval ) >= matInputImg.rows )
                    vecFitPoint.emplace_back ( nJumpPos, matInputImg.rows - 1 );
                else
                    vecFitPoint.push_back ( cv::Point ( nJumpPos, ToInt32 ( nCurrentProcessedPos + nInterval / 2.f + 0.5f ) ) );

                if ( ! matROIMask.empty() && matROIMask.at<uchar>(vecFitPoint.back()) == 0 )
                    vecFitPoint.pop_back();
            }
            nCurrentProcessedPos += nInterval;
        }
    }else {
        cv::transpose ( matGuassinKernel, matGuassinKernel );
        int nInterval = ToInt32 ( ToFloat ( matInputImg.cols ) / ToFloat ( DIVIDE_SECTION ) + 0.5f );
        int nCurrentProcessedPos = 0;
        while ( nCurrentProcessedPos < matInputImg.cols ) {
            int nROISize = ( nCurrentProcessedPos + nInterval ) < matInputImg.cols ? nInterval : matInputImg.cols - nCurrentProcessedPos;
            cv::Mat matSubROI ( matInputImg, cv::Rect ( nCurrentProcessedPos, 0, nROISize , ROWS ) );
            int nJumpPos = _findMaxDiffPosInY ( matSubROI, matGuassinKernel, enDirection );
            if ( nJumpPos > 0 ) {
                if ( nCurrentProcessedPos == 0 )
                    vecFitPoint.emplace_back ( nCurrentProcessedPos, nJumpPos );
                else if ( ( nCurrentProcessedPos + nInterval ) >= matInputImg.cols )
                    vecFitPoint.emplace_back (  matInputImg.cols - 1, nJumpPos );
                else
                    vecFitPoint.emplace_back ( ToInt32 ( nCurrentProcessedPos + nInterval / 2.f + 0.5f ), nJumpPos );
                if ( ! matROIMask.empty() && matROIMask.at<uchar>(vecFitPoint.back()) == 0 )
                    vecFitPoint.pop_back();
            }
            nCurrentProcessedPos += nInterval;
        }
    }

    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
        cv::Mat matDisplay;
        cv::cvtColor ( matInputImg, matDisplay, CV_GRAY2BGR );
        for ( const auto &point : vecFitPoint ) {
            cv::circle ( matDisplay, point, 4, cv::Scalar(0, 0, 255 ), 2 );
        }
        showImage ( "Find out point", matDisplay );
        cv::waitKey(0);
    }   

    if ( vecFitPoint.size() < 2 ) {
        WriteLog("_caliperBySectionAvgGussianDiff can not find enough points");
        pstRpy->enStatus = VisionStatus::CALIPER_CAN_NOT_FIND_LINE;
        return pstRpy->enStatus;
    }

    for ( auto &point : vecFitPoint ) {
        point.x += rectROI.x;
        point.y += rectROI.y;
    }

    Fitting::fitLine ( vecFitPoint, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->bReversedFit );
    pstRpy->stLine = CalcUtils::calcEndPointOfLine ( vecFitPoint, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept );
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

VisionStatus VisionAlgorithm::fitParallelLine(PR_FIT_PARALLEL_LINE_CMD *pstCmd, PR_FIT_PARALLEL_LINE_RPY *pstRpy, bool bReplay)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char charrMsg[1000];

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFitParallelLine);

    cv::Mat matGray, matThreshold;
    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor ( pstCmd->matInputImg, matGray, CV_BGR2GRAY );
    else
        matGray = pstCmd->matInputImg.clone();

    ListOfPoint listPoint1 = _findPointsInRegionByThreshold ( matGray, pstCmd->rectArrROI[0], pstCmd->nThreshold, pstCmd->enAttribute );
    ListOfPoint listPoint2 = _findPointsInRegionByThreshold ( matGray, pstCmd->rectArrROI[1], pstCmd->nThreshold, pstCmd->enAttribute );
    if ( listPoint1.size() < 2 || listPoint2.size() < 2 )  {
        WriteLog("Not enough points to fit Parallel line");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        return pstRpy->enStatus;
    }else if ( listPoint1.size() > PR_FIT_LINE_MAX_POINT_COUNT || listPoint2.size() > PR_FIT_LINE_MAX_POINT_COUNT )  {
        _snprintf ( charrMsg, sizeof(charrMsg), "Too much noise to fit parallel line, points number %d, %d", listPoint1.size(), listPoint2.size() );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        return pstRpy->enStatus;
    }

    std::vector<int> vecX, vecY;
    for ( const auto &point : listPoint1 )
    {
        vecX.push_back ( point.x );
        vecY.push_back ( point.y );
    }

    //For y = kx + b, the k is the slope, when it is very large, the error is quite big, so change
    //to get the x = k1 + b1, and get the k = 1 / k1, b = -b1 / k1. Then the result is more accurate.
    pstRpy->bReversedFit = false;
    if ( CalcUtils::calcStdDeviation ( vecY ) > CalcUtils::calcStdDeviation ( vecX ) )
        pstRpy->bReversedFit = true;

    std::vector<ListOfPoint::const_iterator> overTolPoints1, overTolPoints2;
    int nIteratorNum = 0;
    do
    {
        for (const auto &it : overTolPoints1 )
            listPoint1.erase ( it );
        for (const auto &it : overTolPoints2)
            listPoint2.erase ( it );

        Fitting::fitParallelLine(listPoint1, listPoint2, pstRpy->fSlope, pstRpy->fIntercept1, pstRpy->fIntercept2, pstRpy->bReversedFit);
        overTolPoints1 = _findPointOverLineTol(listPoint1, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept1, pstCmd->enRmNoiseMethod, pstCmd->fErrTol);
        overTolPoints2 = _findPointOverLineTol(listPoint2, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept2, pstCmd->enRmNoiseMethod, pstCmd->fErrTol);
        ++ nIteratorNum;
    }while ( ( ! overTolPoints1.empty() || ! overTolPoints2.empty() ) &&  nIteratorNum < 20 );

    if ( listPoint1.size() < 2 || listPoint2.size() < 2 )  {
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        goto EXIT;
    }

    pstRpy->stLine1 = CalcUtils::calcEndPointOfLine ( listPoint1, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept1 );
    pstRpy->stLine2 = CalcUtils::calcEndPointOfLine ( listPoint2, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept2 );

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::line ( pstRpy->matResultImg, pstRpy->stLine1.pt1, pstRpy->stLine1.pt2, cv::Scalar(255,0,0), 2 );
    cv::line ( pstRpy->matResultImg, pstRpy->stLine2.pt1, pstRpy->stLine2.pt2, cv::Scalar(255,0,0), 2 );
    pstRpy->enStatus = VisionStatus::OK;
   
EXIT:
    FINISH_LOGCASE;    
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::fitRect(const PR_FIT_RECT_CMD *const pstCmd, PR_FIT_RECT_RPY *const pstRpy, bool bReplay /*= false*/ )
{
    char charrMsg [ 1000 ];
    if (NULL == pstCmd || NULL == pstRpy) {
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstCmd = %d, pstRpy = %d", (int)pstCmd, (int)pstRpy);
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFitRect);

    cv::Mat matGray, matThreshold;
    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor ( pstCmd->matInputImg, matGray, CV_BGR2GRAY );
    else
        matGray = pstCmd->matInputImg.clone();

    VectorOfListOfPoint vecListPoint;
    for ( int i = 0; i < PR_RECT_EDGE_COUNT; ++ i ) {
        vecListPoint.push_back ( _findPointsInRegionByThreshold ( matGray, pstCmd->rectArrROI[i], pstCmd->nThreshold, pstCmd->enAttribute ) );
        if ( vecListPoint [ i ].size() < 2 )  {
            _snprintf(charrMsg, sizeof(charrMsg), "Edge %d not enough points to fit rect", i);
            WriteLog(charrMsg);
            pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
            return pstRpy->enStatus;
        }else if ( vecListPoint [ i ].size() > PR_FIT_LINE_MAX_POINT_COUNT )  {
            _snprintf ( charrMsg, sizeof(charrMsg), "Too much noise to fit rect, line %d points number %d", i + 1, vecListPoint [ i ].size() );
            WriteLog(charrMsg);
            pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
            return pstRpy->enStatus;
        }
    }
    std::vector<int> vecX, vecY;
    for ( const auto &point : vecListPoint[0] )
    {
        vecX.push_back ( point.x );
        vecY.push_back ( point.y );
    }

    //For y = kx + b, the k is the slope, when it is very large, the error is quite big, so change
    //to get the x = k1 + b1, and get the k = 1 / k1, b = -b1 / k1. Then the result is more accurate.
    pstRpy->bLineOneReversedFit = false;
    pstRpy->bLineTwoReversedFit = false;
    if ( CalcUtils::calcStdDeviation ( vecY ) > CalcUtils::calcStdDeviation ( vecX ) )  {
        pstRpy->bLineOneReversedFit = true;
        pstRpy->bLineTwoReversedFit = false;
    }else {
        pstRpy->bLineOneReversedFit = false;
        pstRpy->bLineTwoReversedFit = true;
    }

    std::vector<std::vector<ListOfPoint::const_iterator>> vecVecOutTolIndex;
    vecVecOutTolIndex.reserve(vecListPoint.size());
    std::vector<float>    vecIntercept;
    auto hasPointOutTol = false;
    auto iterateCount = 0;
    do 
    {
        auto index = 0;
        for ( const auto &vecOutTolIndex : vecVecOutTolIndex ){
            for (const auto &it : vecOutTolIndex )
                vecListPoint[index].erase ( it );
            ++ index;
        }
        Fitting::fitRect ( vecListPoint, pstRpy->fSlope1, pstRpy->fSlope2, vecIntercept, pstRpy->bLineOneReversedFit, pstRpy->bLineTwoReversedFit );
        vecVecOutTolIndex.clear();
        hasPointOutTol = false;
        for( int i = 0; i < PR_RECT_EDGE_COUNT; ++ i )
        {
            std::vector<ListOfPoint::const_iterator> vecOutTolIndex;
            if ( i < 2 )
                vecOutTolIndex = _findPointOverLineTol( vecListPoint[i], pstRpy->bLineOneReversedFit, pstRpy->fSlope1, vecIntercept[i], pstCmd->enRmNoiseMethod, pstCmd->fErrTol );
            else
                vecOutTolIndex = _findPointOverLineTol( vecListPoint[i], pstRpy->bLineTwoReversedFit, pstRpy->fSlope2, vecIntercept[i], pstCmd->enRmNoiseMethod, pstCmd->fErrTol );
            vecVecOutTolIndex.push_back( vecOutTolIndex );
        }

        for ( const auto &vecOutTolIndex : vecVecOutTolIndex ) {
            if ( ! vecOutTolIndex.empty() ) {
                hasPointOutTol = true;
                break;
            }
        }
        ++ iterateCount;
    }while ( hasPointOutTol && iterateCount < 20 );

    int nLineNumber = 1;
    for ( const auto &listPoint : vecListPoint ) {
        if (listPoint.size() < 2)  {
            pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
            _snprintf(charrMsg, sizeof(charrMsg), "Rect line %d has too much noise to fit", nLineNumber);
            WriteLog(charrMsg);
            goto EXIT;
        }
        ++ nLineNumber;
    }

    for ( int i = 0; i < PR_RECT_EDGE_COUNT; ++ i ) {
        if ( i < 2 )
            pstRpy->arrLines[i] = CalcUtils::calcEndPointOfLine( vecListPoint[i], pstRpy->bLineOneReversedFit, pstRpy->fSlope1, vecIntercept[i]);
        else
            pstRpy->arrLines[i] = CalcUtils::calcEndPointOfLine( vecListPoint[i], pstRpy->bLineTwoReversedFit, pstRpy->fSlope2, vecIntercept[i]);
        pstRpy->fArrIntercept[i] = vecIntercept[i];
    }
    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    for ( int i = 0; i < PR_RECT_EDGE_COUNT; ++ i ) {
        cv::line ( pstRpy->matResultImg, pstRpy->arrLines[i].pt1, pstRpy->arrLines[i].pt2, cv::Scalar(255,0,0), 2 );
    }
    pstRpy->enStatus = VisionStatus::OK;

EXIT:
    FINISH_LOGCASE;    
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::findEdge(const PR_FIND_EDGE_CMD *const pstCmd, PR_FIND_EDGE_RPY *const pstRpy, bool bReplay /*= false*/)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char charrMsg [ 1000 ];

    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    if ( matROI.empty() ){
        WriteLog("ROI image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if ( ! pstCmd->bAutoThreshold && ( pstCmd->nThreshold <= 0 || pstCmd->nThreshold >= PR_MAX_GRAY_LEVEL ) )  {
         _snprintf(charrMsg, sizeof(charrMsg), "The threshold = %d is invalid", pstCmd->nThreshold);
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFindEdge);

    cv::Mat matROIGray, matThreshold, matBlur, matCannyResult;
    cv::cvtColor ( matROI, matROIGray, CV_BGR2GRAY );

    auto threshold = pstCmd->nThreshold;
    if ( pstCmd->bAutoThreshold )
        threshold = _autoThreshold ( matROIGray );

    cv::threshold( matROIGray, matThreshold, pstCmd->nThreshold, 255, cv::THRESH_BINARY );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Threshold Result", matThreshold );

    cv::blur ( matThreshold, matBlur, cv::Size(2, 2) );

    // Detect edges using canny
    cv::Canny( matBlur, matCannyResult, pstCmd->nThreshold, 255);
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Canny Result", matCannyResult );

    std::vector<cv::Vec4i> lines;
    auto voteThreshold = ToInt32 ( pstCmd->fMinLength / 2.f );
    cv::HoughLinesP ( matCannyResult, lines, 1, CV_PI / 180, voteThreshold, pstCmd->fMinLength, 10 );
    std::vector<PR_Line2f> vecLines, vecLinesAfterMerge, vecLineDisplay;
    for ( size_t i = 0; i < lines.size(); ++ i ) {
        cv::Point2f pt1((float)lines[i][0], (float)lines[i][1]);
        cv::Point2f pt2((float)lines[i][2], (float)lines[i][3]);
        PR_Line2f line(pt1, pt2);
        vecLines.push_back(line);
    }
    _mergeLines ( vecLines, vecLinesAfterMerge );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
        cv::Mat matResultImage = matROI.clone();
        for ( const auto &line : vecLinesAfterMerge )
            cv::line ( matResultImage, line.pt1, line.pt2, cv::Scalar(255, 0, 0), 2);
        showImage("Find edge Result", matResultImage );
    }
    Int32 nLineCount = 0;
    //if ( PR_EDGE_DIRECTION::BOTH == pstCmd->enDirection ) {
    //    pstRpy->nEdgeCount = ToInt32 ( vecLinesAfterMerge.size() );
    //    vecLineDisplay = vecLinesAfterMerge;
    //}else {
    for (const auto &line : vecLinesAfterMerge) {
        float fSlope = CalcUtils::lineSlope ( line );
        if ( fabs ( fSlope ) < 0.1 ) {
            if (PR_EDGE_DIRECTION::HORIZONTAL == pstCmd->enDirection || PR_EDGE_DIRECTION::BOTH == pstCmd->enDirection) {
                ++ nLineCount;
                vecLineDisplay.push_back ( line );
            }
        }
        else if ( ( line.pt1.x == line.pt2.x ) || fabs ( fSlope ) > 10 ) {
            if ( PR_EDGE_DIRECTION::VERTIAL == pstCmd->enDirection || PR_EDGE_DIRECTION::BOTH == pstCmd->enDirection) {
                ++ nLineCount;
                vecLineDisplay.push_back ( line );
            }
        }
    }
    pstRpy->nEdgeCount = nLineCount;
    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    for ( const auto &line : vecLineDisplay ) {
        cv::Point2f pt1( line.pt1.x + pstCmd->rectROI.x, line.pt1.y + pstCmd->rectROI.y );
        cv::Point2f pt2( line.pt2.x + pstCmd->rectROI.x, line.pt2.y + pstCmd->rectROI.y );
        cv::line ( pstRpy->matResultImg, pt1, pt2, _constBlueScalar, 2 );
    }
    pstRpy->enStatus = VisionStatus::OK;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::fitCircle(PR_FIT_CIRCLE_CMD *pstCmd, PR_FIT_CIRCLE_RPY *pstRpy, bool bReplay)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char charrMsg [ 1000 ];

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    
    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 || 
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows ) {
        WriteLog("The input ROI is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( ! pstCmd->matMask.empty() ) {
        if ( pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols ) {
            _snprintf(charrMsg, sizeof(charrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d )",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(charrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if ( pstCmd->matMask.channels() != 1 )  {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (PR_FIT_METHOD::RANSAC == pstCmd->enMethod && (1000 <= pstCmd->nMaxRansacTime || pstCmd->nMaxRansacTime <= 0)) {
        _snprintf(charrMsg, sizeof(charrMsg), "Max Rransac time %d is invalid", pstCmd->nMaxRansacTime);
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFitCircle);

    cv::Mat matROI ( pstCmd->matInputImg, pstCmd->rectROI);    
    cv::Mat matGray, matThreshold;

    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();

    if ( pstCmd->bPreprocessed )
        matThreshold = matGray;    
    else {
        cv::ThresholdTypes enThresType = cv::THRESH_BINARY;
        if ( PR_OBJECT_ATTRIBUTE::DARK == pstCmd->enAttribute )
            enThresType = cv::THRESH_BINARY_INV;

        cv::threshold(matGray, matThreshold, pstCmd->nThreshold, PR_MAX_GRAY_LEVEL, enThresType);
        CV_Assert(matThreshold.channels() == 1);

        if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
            showImage("Threshold image", matThreshold);
    }

    if ( ! pstCmd->matMask.empty() ) {
        cv::Mat matROIMask(pstCmd->matMask, pstCmd->rectROI);
        matROIMask = cv::Scalar(255) - matROIMask;
        matThreshold.setTo(cv::Scalar(0), matROIMask);
    }

    VectorOfPoint vecPoints;
    cv::findNonZero( matThreshold, vecPoints);
    cv::RotatedRect fitResult;

    if ( vecPoints.size() < 3 ) {
        WriteLog("Not enough points to fit circle");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    if ( vecPoints.size() > PR_FIT_CIRCLE_MAX_POINT ) {
        WriteLog("Too much points to fit circle");
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    
	if ( PR_FIT_METHOD::RANSAC == pstCmd->enMethod )
        fitResult = _fitCircleRansac ( vecPoints, pstCmd->fErrTol, pstCmd->nMaxRansacTime, vecPoints.size() / 2 );
	else if (PR_FIT_METHOD::LEAST_SQUARE == pstCmd->enMethod)
        fitResult = Fitting::fitCircle ( vecPoints );
    else if ( PR_FIT_METHOD::LEAST_SQUARE_REFINE == pstCmd->enMethod  )
        fitResult = _fitCircleIterate ( vecPoints, pstCmd->enRmNoiseMethod, pstCmd->fErrTol );

    if ( fitResult.size.width <= 0 ) {
        WriteLog("Failed to fit circle");
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
	pstRpy->ptCircleCtr = fitResult.center + cv::Point2f ( ToFloat ( pstCmd->rectROI.x ), ToFloat ( pstCmd->rectROI.y ) );
    pstRpy->fRadius = fitResult.size.width / 2;
    pstRpy->enStatus = VisionStatus::OK;

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
	cv::circle ( pstRpy->matResultImg, pstRpy->ptCircleCtr, (int)pstRpy->fRadius, _constBlueScalar, 2);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/* The ransac algorithm is from paper "Random Sample Consensus: A Paradigm for Model Fitting with Apphcatlons to Image Analysis and Automated Cartography".
*  Given a model that requires a minimum of n data points to instantiate
* its free parameters, and a set of data points P such that the number of
* points in P is greater than n, randomly select a subset S1 of
* n data points from P and instantiate the model. Use the instantiated
* model M1 to determine the subset S1* of points in P that are within
* some error tolerance of M1. The set S1* is called the consensus set of
* S1.
* If g (S1*) is greater than some threshold t, which is a function of the
* estimate of the number of gross errors in P, use S1* to compute
* (possibly using least squares) a new model M1* and return.
* If g (S1*) is less than t, randomly select a new subset S2 and repeat the
* above process. If, after some predetermined number of trials, no
* consensus set with t or more members has been found, either solve the
* model with the largest consensus set found, or terminate in failure. */
/*static*/ cv::RotatedRect VisionAlgorithm::_fitCircleRansac(const VectorOfPoint &vecPoints, float tolerance, int maxRansacTime, size_t numOfPtToFinish)
{   
    cv::RotatedRect fitResult;
    if (vecPoints.size() < 3)
        return fitResult;

    int nRansacTime = 0;
    const int RANSAC_CIRCLE_POINT = 3;
    size_t nMaxConsentNum = 0;

    while ( nRansacTime < maxRansacTime ) {
        VectorOfPoint vecSelectedPoints = _randomSelectPoints ( vecPoints, RANSAC_CIRCLE_POINT );
        cv::RotatedRect rectReult = Fitting::fitCircle ( vecSelectedPoints );
        vecSelectedPoints = _findPointsInCircleTol ( vecPoints, rectReult, tolerance );

        if ( vecSelectedPoints.size() >= numOfPtToFinish )
           return Fitting::fitCircle ( vecSelectedPoints );
        
        if ( vecSelectedPoints.size() > nMaxConsentNum )
        {
            fitResult = Fitting::fitCircle ( vecSelectedPoints );
            nMaxConsentNum = vecSelectedPoints.size();
        }
        ++ nRansacTime;
    }

    return fitResult;
}

/*static*/ VectorOfPoint VisionAlgorithm::_randomSelectPoints(const VectorOfPoint &vecPoints, int numOfPtToSelect)
{    
    std::set<int> setResult;
    VectorOfPoint vecResultPoints;

    if ( (int)vecPoints.size() < numOfPtToSelect )
        return vecResultPoints;

    while ((int)setResult.size() < numOfPtToSelect)   {
        int nValue = std::rand() % vecPoints.size();
        setResult.insert(nValue);
    }
    for ( auto index : setResult )
        vecResultPoints.push_back ( vecPoints[index] );
    return vecResultPoints;
}

/*static*/ VectorOfPoint VisionAlgorithm::_findPointsInCircleTol( const VectorOfPoint &vecPoints, const cv::RotatedRect &rotatedRect, float tolerance )
{
    VectorOfPoint vecResult;
    for ( const auto &point : vecPoints )  {
        auto disToCtr = sqrt( ( point.x - rotatedRect.center.x) * (point.x - rotatedRect.center.x) + ( point.y - rotatedRect.center.y) * ( point.y - rotatedRect.center.y ) );
        auto err = disToCtr - rotatedRect.size.width / 2;
        if ( fabs ( err ) < tolerance )  {
            vecResult.push_back(point);
        }
    }
    return vecResult;
}

/*static*/ VectorOfPoint VisionAlgorithm::_findPointsOverCircleTol( const VectorOfPoint &vecPoints, const cv::RotatedRect &rotatedRect, float tolerance )
{
    VectorOfPoint vecResult;
    for ( const auto &point : vecPoints )  {
        auto disToCtr = sqrt( ( point.x - rotatedRect.center.x) * (point.x - rotatedRect.center.x) + ( point.y - rotatedRect.center.y) * ( point.y - rotatedRect.center.y ) );
        auto err = disToCtr - rotatedRect.size.width / 2;
        if ( fabs ( err ) > tolerance )  {
            vecResult.push_back(point);
        }
    }
    return vecResult;
}

/*static*/ std::vector<ListOfPoint::const_iterator> VisionAlgorithm::_findPointsOverCircleTol( const ListOfPoint &listPoint, const cv::RotatedRect &rotatedRect, PR_RM_FIT_NOISE_METHOD enMethod, float tolerance )
{
    std::vector<ListOfPoint::const_iterator> vecResult;
    for ( ListOfPoint::const_iterator it = listPoint.begin(); it != listPoint.end(); ++ it )  {
        cv::Point2f point = *it;
        auto disToCtr = sqrt( ( point.x - rotatedRect.center.x) * (point.x - rotatedRect.center.x)  + ( point.y - rotatedRect.center.y) * ( point.y - rotatedRect.center.y ) );
        auto err = disToCtr - rotatedRect.size.width / 2;
        if ( PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR == enMethod && fabs ( err ) > tolerance )  {
            vecResult.push_back(it);
        }else if ( PR_RM_FIT_NOISE_METHOD::POSITIVE_ERR == enMethod && err > tolerance ) {
            vecResult.push_back(it);
        }else if ( PR_RM_FIT_NOISE_METHOD::NEGATIVE_ERR == enMethod && err < -tolerance ) {
            vecResult.push_back(it);
        }
    }
    return vecResult;
}

//method 1 : Exclude all the points out of positive error tolerance and inside the negative error tolerance.
//method 2 : Exclude all the points out of positive error tolerance.
//method 3 : Exclude all the points inside the negative error tolerance.
/*static*/ cv::RotatedRect VisionAlgorithm::_fitCircleIterate(const VectorOfPoint &vecPoints, PR_RM_FIT_NOISE_METHOD method, float tolerance)
{
    cv::RotatedRect rotatedRect;
    if (vecPoints.size() < 3)
        return rotatedRect;

    ListOfPoint listPoint;
    for ( const auto &point : vecPoints )
        listPoint.push_back ( point );
    
    rotatedRect = Fitting::fitCircle ( listPoint );
    
    std::vector<ListOfPoint::const_iterator> overTolPoints = _findPointsOverCircleTol ( listPoint, rotatedRect, method, tolerance );
    int nIteratorNum = 1;
    while ( ! overTolPoints.empty() && nIteratorNum < 20 ) {
        for (const auto &it : overTolPoints )
            listPoint.erase( it );
        rotatedRect = Fitting::fitCircle ( listPoint );
        overTolPoints = _findPointsOverCircleTol ( listPoint, rotatedRect, method, tolerance );
        ++ nIteratorNum;
    }
    return rotatedRect;
}

/*static*/ VisionStatus VisionAlgorithm::ocr(PR_OCR_CMD *pstCmd, PR_OCR_RPY *pstRpy, bool bReplay)
{    
    assert ( pstCmd != nullptr && pstRpy != nullptr );

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }
    
    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows )    {
        WriteLog("The OCR search range is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseOcr);

    VisionStatus enStatus = VisionStatus::OK;
    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    if ( pstCmd->enDirection != PR_DIRECTION::UP )  {
        float fAngle = 0.f;
        cv::Size sizeDest(pstCmd->rectROI.width, pstCmd->rectROI.height);
        switch (pstCmd->enDirection)
        {
            case PR_DIRECTION::LEFT:
                cv::transpose( matROI, matROI );  
                cv::flip(matROI, matROI,1); //transpose+flip(1)=CW
                break;
            case PR_DIRECTION::DOWN:
                cv::flip(matROI, matROI, -1 );    //flip(-1)=180
                break;
            case PR_DIRECTION::RIGHT:                
                cv::transpose(matROI, matROI);  
                cv::flip(matROI, matROI,0); //transpose+flip(0)=CCW
                break;
            default: break;        
        }        
    }

    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("OCR ROI Image", matROI );

    char *dataPath = "./tessdata";
    String strOcrCharList = Config::GetInstance()->getOcrCharList();
    if ( nullptr == _ptrOcrTesseract )
        _ptrOcrTesseract = cv::text::OCRTesseract::create(dataPath, "eng+eng1", strOcrCharList.c_str() );
    _ptrOcrTesseract->run ( matROI, pstRpy->strResult );
    if ( pstRpy->strResult.empty() )    {
        enStatus = VisionStatus::OCR_FAIL;
        WriteLog("OCR result is empty");
    }
    else
        enStatus = VisionStatus::OK;

    pstRpy->enStatus = enStatus;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME; 
    return enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::colorToGray(PR_COLOR_TO_GRAY_CMD *pstCmd, PR_COLOR_TO_GRAY_RPY *pstRpy)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    char charrMsg [ 1000 ];
    if ( pstCmd->matInputImg.channels() != 3 ) {
        _snprintf(charrMsg, sizeof(charrMsg), "Convert from color to gray, the input image channel number %d not equal to 3", pstCmd->matInputImg.channels());
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    std::vector<float> coefficients{ pstCmd->stRatio.fRatioB, pstCmd->stRatio.fRatioG, pstCmd->stRatio.fRatioR };
    cv::Mat matCoefficients = cv::Mat(coefficients).reshape(1, 1);
    cv::transform(pstCmd->matInputImg, pstRpy->matResultImg, matCoefficients);

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::filter(PR_FILTER_CMD *pstCmd, PR_FILTER_RPY *pstRpy)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char charrMsg [ 1000 ];

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows )    {
        WriteLog("The input ROI is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if ( PR_FILTER_TYPE::GAUSSIAN_FILTER == pstCmd->enType &&
        ( ( pstCmd->szKernel.width % 2 != 1 ) || ( pstCmd->szKernel.height % 2 != 1 ) ) )   {
        _snprintf(charrMsg, sizeof (charrMsg), "Guassian filter kernel size ( %d, %d ) is invalid", pstCmd->szKernel.width, pstCmd->szKernel.height );
        WriteLog ( charrMsg );
        pstRpy->enStatus = VisionStatus::GUASSIAN_FILTER_KERNEL_INVALID;
        return VisionStatus::GUASSIAN_FILTER_KERNEL_INVALID;
    }

    if ( PR_FILTER_TYPE::MEDIAN_FILTER == pstCmd->enType &&
        ( ( pstCmd->szKernel.width % 2 != 1 ) || ( pstCmd->szKernel.height % 2 != 1 ) ) )   {
        _snprintf(charrMsg, sizeof (charrMsg), "Median filter kernel size ( %d, %d ) is invalid", pstCmd->szKernel.width, pstCmd->szKernel.height );
        WriteLog ( charrMsg );
        pstRpy->enStatus = VisionStatus::MEDIAN_FILTER_KERNEL_INVALID;
        return VisionStatus::MEDIAN_FILTER_KERNEL_INVALID;
    }

    MARK_FUNCTION_START_TIME;

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matROI ( pstRpy->matResultImg, pstCmd->rectROI );

    VisionStatus enStatus = VisionStatus::OK;
    if ( PR_FILTER_TYPE::NORMALIZED_BOX_FILTER == pstCmd->enType )  {
        cv::blur( matROI, matROI, pstCmd->szKernel );
    }else if ( PR_FILTER_TYPE::GAUSSIAN_FILTER == pstCmd->enType )  {
        cv::GaussianBlur( matROI, matROI, pstCmd->szKernel, pstCmd->dSigmaX, pstCmd->dSigmaY);
    }else if ( PR_FILTER_TYPE::MEDIAN_FILTER == pstCmd->enType ) {
        cv::medianBlur( matROI, matROI, pstCmd->szKernel.width);
    }else if ( PR_FILTER_TYPE::BILATERIAL_FILTER == pstCmd->enType ) {
        cv::Mat matResultImg;
        cv::bilateralFilter ( matROI, matResultImg, pstCmd->nDiameter, pstCmd->dSigmaColor, pstCmd->dSigmaSpace );
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )   {
            showImage("Bilateral filter result", matResultImg );
        }
        matResultImg.copyTo( matROI );
    }else {
        enStatus = VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_END_TIME;

    pstRpy->enStatus = enStatus;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::removeCC(PR_REMOVE_CC_CMD *pstCmd, PR_REMOVE_CC_RPY *pstRpy, bool bReplay) {
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char chArrMsg [ 1000 ];

    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows )    {
        WriteLog("The input ROI is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if ( pstCmd->nConnectivity != 4 && pstCmd->nConnectivity != 8 ) {
        _snprintf(chArrMsg, sizeof (chArrMsg), "The connectivity %d is not 4 or 8", pstCmd->nConnectivity );
        WriteLog ( chArrMsg );
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseRemoveCC);    

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matROI ( pstRpy->matResultImg, pstCmd->rectROI );
    cv::Mat matGray = matROI;
    if ( matROI.channels() == 3 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );

    cv::Mat matLabels;
    cv::connectedComponents( matGray, matLabels, pstCmd->nConnectivity, CV_32S);
    TimeLog::GetInstance()->addTimeLog("cv::connectedComponents", stopWatch.AbsNow() - functionStart);

    double minValue = 0., maxValue = 0.;
    cv::minMaxIdx ( matLabels, &minValue, &maxValue );

    cv::Mat matRemoveMask = cv::Mat::zeros(matLabels.size(), CV_8UC1);

    if ( maxValue > Config::GetInstance()->getRemoveCCMaxComponents() ) {
        pstRpy->enStatus = VisionStatus::TOO_MUCH_CC_TO_REMOVE;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    int nCurrentLabel = 1;
    int nCurrentLabelCount = 0;
    int nTotalPixels = matLabels.rows * matLabels.cols;
    pstRpy->nRemovedCC = 0;

    do {
        cv::Mat matTemp = matLabels - nCurrentLabel;
        nCurrentLabelCount = nTotalPixels - cv::countNonZero ( matTemp );
        if ( 0 < nCurrentLabelCount && nCurrentLabelCount < pstCmd->fAreaThreshold ) {
            matRemoveMask += CalcUtils::genMaskByValue<Int32>( matLabels, nCurrentLabel );
            ++ pstRpy->nRemovedCC;
        }
        ++ nCurrentLabel;
    } while ( nCurrentLabelCount > 0 );

    pstRpy->nTotalCC = nCurrentLabel - 1;

    matGray.setTo(0, matRemoveMask);

    if ( matROI.channels() == 3 )
        cv::cvtColor ( matGray, matROI, CV_GRAY2BGR );

    pstRpy->enStatus = VisionStatus::OK;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::detectEdge(PR_DETECT_EDGE_CMD *pstCmd, PR_DETECT_EDGE_RPY *pstRpy, bool bReplay)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );

    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows )    {
        WriteLog("The input ROI is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseDetectEdge);

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matROI ( pstRpy->matResultImg, pstCmd->rectROI );

    cv::Mat matGray = matROI;
    if ( pstCmd->matInputImg.channels() == 3 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );

    cv::Mat matResultImg;
    cv::Canny ( matGray, matResultImg, pstCmd->nThreshold1, pstCmd->nThreshold2, pstCmd->nApertureSize );

    if ( pstCmd->matInputImg.channels() == 3 )
        cv::cvtColor ( matResultImg, matROI, CV_GRAY2BGR );
    else
        matResultImg.copyTo ( matROI );

    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::inspCircle(const PR_INSP_CIRCLE_CMD *const pstCmd, PR_INSP_CIRCLE_RPY *const pstRpy, bool bReplay)
{
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char chArrMsg[1000];

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows )    {
        WriteLog("The input ROI is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if ( ! pstCmd->matMask.empty() ) {
        if ( pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols ) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d )",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if ( pstCmd->matMask.channels() != 1 )  {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspCircle);

    cv::Mat matROI ( pstCmd->matInputImg, pstCmd->rectROI ); 
    cv::Mat matGray;

    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();

    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
        showImage("Before mask out ROI image", matGray );

    if ( ! pstCmd->matMask.empty() )    {
        cv::Mat matROIMask(pstCmd->matMask, pstCmd->rectROI);
        matROIMask = cv::Scalar(255) - matROIMask;
        matGray.setTo(cv::Scalar(0), matROIMask);
    }

    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
        showImage("mask out ROI image", matGray );

    VectorOfPoint vecPoints;
    cv::findNonZero( matGray, vecPoints);
    cv::RotatedRect fitResult;
    std::vector<float> vecDistance;

    if ( vecPoints.size() < 3 ) {
        WriteLog("Not enough points to fit circle");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    
    fitResult = Fitting::fitCircle ( vecPoints );

    if ( fitResult.center.x <= 0 || fitResult.center.y <= 0 || fitResult.size.width <= 0 ) {
        WriteLog("Failed to fit circle");
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
	pstRpy->ptCircleCtr = fitResult.center + cv::Point2f ( ToFloat ( pstCmd->rectROI.x ), ToFloat ( pstCmd->rectROI.y ) );
    pstRpy->fDiameter = ( fitResult.size.width + fitResult.size.height ) / 2;
    
    for ( const auto &point : vecPoints )   {
        cv::Point2f pt2f(point);
        vecDistance.push_back ( CalcUtils::distanceOf2Point<float> ( pt2f, fitResult.center ) - pstRpy->fDiameter / 2 );
    }
    pstRpy->fRoundness = ToFloat ( CalcUtils::calcStdDeviation<float>( vecDistance ) );
    pstRpy->matResultImg = pstCmd->matInputImg.clone();
	cv::circle ( pstRpy->matResultImg, pstRpy->ptCircleCtr, (int)pstRpy->fDiameter / 2, _constBlueScalar, 2);

    pstRpy->enStatus = VisionStatus::OK;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_fillHoleByContour(const cv::Mat &matInputImg, cv::Mat &matOutput, PR_OBJECT_ATTRIBUTE enAttribute)
{
    if ( matInputImg.channels() > 1 )
        cv::cvtColor ( matInputImg, matOutput, CV_BGR2GRAY );
    else
        matOutput = matInputImg.clone();

    if ( PR_OBJECT_ATTRIBUTE::DARK == enAttribute )
        cv::bitwise_not( matOutput, matOutput );

    VectorOfVectorOfPoint contours;
    cv::findContours( matOutput, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    for ( size_t i = 0; i < contours.size(); ++ i )    {
        cv::drawContours ( matOutput, contours, ToInt32(i), cv::Scalar::all(PR_MAX_GRAY_LEVEL), CV_FILLED );
    }

    if ( PR_OBJECT_ATTRIBUTE::DARK == enAttribute )
        cv::bitwise_not( matOutput, matOutput );

    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_fillHoleByMorph(const cv::Mat      &matInputImg,
                                                          cv::Mat            &matOutput,
                                                          PR_OBJECT_ATTRIBUTE enAttribute,
                                                          cv::MorphShapes     enMorphShape,
                                                          cv::Size            szMorphKernel,
                                                          Int16               nMorphIteration)
{
    char chArrMsg[1000];
    if ( szMorphKernel.width <= 0 ||  szMorphKernel.height <= 0 ) {
        _snprintf( chArrMsg, sizeof ( chArrMsg ), "The size of morph kernel ( %d, %d) is invalid.", szMorphKernel.width, szMorphKernel.height );
        WriteLog (chArrMsg );
        return VisionStatus::INVALID_PARAM;
    }

    if ( nMorphIteration <= 0 ) {
        _snprintf( chArrMsg, sizeof ( chArrMsg ), "The morph iteration number %d is invalid.", nMorphIteration );
        WriteLog (chArrMsg );
        return VisionStatus::INVALID_PARAM;
    }
    cv::Mat matKernal = cv::getStructuringElement ( enMorphShape, szMorphKernel );
    cv::MorphTypes morphType = PR_OBJECT_ATTRIBUTE::BRIGHT == enAttribute ? cv::MORPH_CLOSE : cv::MORPH_OPEN;
    cv::morphologyEx ( matInputImg, matOutput, morphType, matKernal, cv::Point(-1, -1), nMorphIteration );

    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::fillHole(PR_FILL_HOLE_CMD *pstCmd, PR_FILL_HOLE_RPY *pstRpy, bool bReplay)
{
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows )    {
        WriteLog("The input ROI is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFillHole);

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matROI ( pstRpy->matResultImg, pstCmd->rectROI);
    cv::Mat matGray;

    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();

    cv::Mat matFillHoleResult;
    if ( PR_FILL_HOLE_METHOD::CONTOUR == pstCmd->enMethod )
        pstRpy->enStatus = _fillHoleByContour ( matGray, matFillHoleResult, pstCmd->enAttribute );
    else
        pstRpy->enStatus = _fillHoleByMorph (matGray, matFillHoleResult, pstCmd->enAttribute, pstCmd->enMorphShape, pstCmd->szMorphKernel, pstCmd->nMorphIteration );

    if ( pstCmd->matInputImg.channels() == 3 )
        cv::cvtColor ( matFillHoleResult, matROI, CV_GRAY2BGR );
    else
        matFillHoleResult.copyTo ( matROI );

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;

    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::pickColor(PR_PICK_COLOR_CMD *pstCmd, PR_PICK_COLOR_RPY *pstRpy, bool bReplay)
{
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( pstCmd->matInputImg.channels() <= 1 ) {
        WriteLog("Input image must be color image.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows ) {
        WriteLog("The input ROI is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if ( ! pstCmd->rectROI.contains(pstCmd->ptPick) ) {
        WriteLog("The pick point must in selected ROI");
        pstRpy->enStatus = VisionStatus::PICK_PT_NOT_IN_ROI;
        return VisionStatus::PICK_PT_NOT_IN_ROI;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCasePickColor);

    cv::Mat matROI ( pstCmd->matInputImg, pstCmd->rectROI );

    std::vector<cv::Mat> vecMat;
    cv::split(matROI, vecMat);

    cv::Point ptPick(pstCmd->ptPick.x - pstCmd->rectROI.x, pstCmd->ptPick.y - pstCmd->rectROI.y );

    std::vector<int> vecValue;
    for ( const auto &mat : vecMat )
        vecValue.push_back( mat.at<uchar>( ptPick ) );

    assert( vecValue.size() == 3 );

    cv::Mat matGray;
    cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );

    int Tt = matGray.at<uchar>( ptPick );

    auto maxElement = std::max_element ( vecValue.begin(), vecValue.end() );
    auto maxIndex = std::distance ( vecValue.begin(), maxElement );
    std::vector<size_t> vecExcludedIndex{0, 1, 2};
    vecExcludedIndex.erase( vecExcludedIndex.begin() + maxIndex);
    cv::Mat matResultMask = cv::Mat::zeros(matROI.size(), CV_8UC1);
    int diffLowerLimit1 = vecValue[maxIndex] - vecValue[vecExcludedIndex[0]] - pstCmd->nColorDiff;
    int diffLowerLimit2 = vecValue[maxIndex] - vecValue[vecExcludedIndex[1]] - pstCmd->nColorDiff;
    int diffUpLimit1    = vecValue[maxIndex] - vecValue[vecExcludedIndex[0]] + pstCmd->nColorDiff;
    int diffUpLimit2    = vecValue[maxIndex] - vecValue[vecExcludedIndex[1]] + pstCmd->nColorDiff;

    int nPointCount = 0;
    for ( int row = 0; row < matROI.rows; ++ row )
    for ( int col = 0; col < matROI.cols; ++ col )
    {
        if ((vecMat[maxIndex].at<uchar>(row, col) - vecMat[vecExcludedIndex[0]].at<uchar>(row, col)) > diffLowerLimit1 &&
            (vecMat[maxIndex].at<uchar>(row, col) - vecMat[vecExcludedIndex[1]].at<uchar>(row, col)) > diffLowerLimit2 &&
            (vecMat[maxIndex].at<uchar>(row, col) - vecMat[vecExcludedIndex[0]].at<uchar>(row, col)) < diffUpLimit1    &&
            (vecMat[maxIndex].at<uchar>(row, col) - vecMat[vecExcludedIndex[1]].at<uchar>(row, col)) < diffUpLimit2    &&
            abs((int)matGray.at<uchar>(row, col) - Tt) < pstCmd->nGrayDiff)
        {
            matResultMask.at<uchar>(row, col) = 1;
            ++ nPointCount;
        }
    }

    pstRpy->enStatus = VisionStatus::OK;
    pstRpy->nPickPointCount = nPointCount;
    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matResultROI ( pstRpy->matResultImg, pstCmd->rectROI );
    matResultROI.setTo(cv::Scalar::all(0));
    matResultROI.setTo ( cv::Scalar::all(PR_MAX_GRAY_LEVEL), matResultMask );

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_findChessBoardCorners(const cv::Mat   &mat,
                                                                const cv::Mat   &matTmpl,
                                                                cv::Mat         &matCornerPointsImg,
                                                                const cv::Point &startPoint,
                                                                float            fStepSize,
                                                                int              nSrchSize,
                                                                float            fMinTmplMatchScore,
                                                                const cv::Size  &szBoardPattern,
                                                                VectorOfPoint2f &vecCorners,
                                                                VectorOfPoint   &vecFailedRowCol)
{
    VisionStatus enStatus;
    cv::cvtColor ( mat, matCornerPointsImg, CV_GRAY2BGR );
    float fStepOffset = 0.f;
    float fStepSizeY = fStepSize;
    float fMinCorrelation = fMinTmplMatchScore / ConstToPercentage;
    cv::Scalar colorOfResultPoint;
    bool bStepUpdatedX = false, bStepUpdatedY = false;
    for ( int row = 0; row < szBoardPattern.height; ++ row )
    for ( int col = 0; col < szBoardPattern.width;  ++ col )
    {        
        cv::Rect rectSrchROI ( ToInt32 ( startPoint.x + col * fStepSize - row * fStepOffset ), ToInt32 ( startPoint.y + row * fStepSizeY + col * fStepOffset ), nSrchSize, nSrchSize );
        if ( rectSrchROI.x < 0 ) rectSrchROI.x = 0;
        if ( rectSrchROI.y < 0 ) rectSrchROI.y = 0;
        if ( ( rectSrchROI.x + matTmpl.cols ) > mat.cols || ( rectSrchROI.y  + matTmpl.rows ) > mat.rows ) {
            WriteLog("The chessboard pattern is not correct.");
            return VisionStatus::CHESSBOARD_PATTERN_NOT_CORRECT;
        }
        if ( ( rectSrchROI.x + rectSrchROI.width )  > mat.cols ) rectSrchROI.width  = mat.cols - rectSrchROI.x;
        if ( ( rectSrchROI.y + rectSrchROI.height ) > mat.rows ) rectSrchROI.height = mat.rows - rectSrchROI.y;
//#ifdef _DEBUG
//        std::cout << "Srch ROI " << rectSrchROI.x << ", " << rectSrchROI.y << ", " << rectSrchROI.width << ", " << rectSrchROI.height << std::endl;
//#endif
        cv::Mat matSrchROI ( mat, rectSrchROI );
        cv::Point2f ptResult;
        float fRotation, fCorrelation = 0.f;
        enStatus = _matchTemplate ( matSrchROI, matTmpl, PR_OBJECT_MOTION::TRANSLATION, ptResult, fRotation, fCorrelation );
        if ( VisionStatus::OK != enStatus )
            fCorrelation = 0;
        
        ptResult.x += rectSrchROI.x;
        ptResult.y += rectSrchROI.y;

        if ( fCorrelation > fMinCorrelation ) {
            vecCorners.push_back ( ptResult );
            colorOfResultPoint = _constGreenScalar;
        }else {
            vecFailedRowCol.emplace_back(cv::Point(col, row));
            colorOfResultPoint = _constRedScalar;
        }
        
        if ( vecCorners.size() == 2 ) {
            fStepSize = CalcUtils::distanceOf2Point ( vecCorners[1], vecCorners[0] );
            fStepOffset = vecCorners[1].y - vecCorners[0].y;
        }else if ( vecCorners.size() == ( szBoardPattern.width + 1 ) ) {
            fStepSizeY = vecCorners[szBoardPattern.width].y - vecCorners[0].y;
        }

        cv::rectangle ( matCornerPointsImg, rectSrchROI, cv::Scalar(255,0,0), 2 );
        cv::circle ( matCornerPointsImg, ptResult, 10, colorOfResultPoint, 2);
        cv::line ( matCornerPointsImg, cv::Point2f(ptResult.x - 4, ptResult.y), cv::Point2f(ptResult.x + 4, ptResult.y), colorOfResultPoint, 1);
        cv::line ( matCornerPointsImg, cv::Point2f(ptResult.x, ptResult.y - 4), cv::Point2f(ptResult.x, ptResult.y + 4), colorOfResultPoint, 1);
    }

    if ( ToInt32( vecCorners.size() ) < Config::GetInstance()->getMinPointsToCalibCamera() ) {
        return VisionStatus::FAILED_TO_FIND_ENOUGH_CB_CORNERS;
    }
    return VisionStatus::OK;
}

/*static*/ void VisionAlgorithm::_calcBoardCornerPositions(const cv::Size &szBoardSize, float squareSize, const VectorOfPoint &vecFailedColRow, std::vector<cv::Point3f> &corners, CalibPattern patternType /*= CHESSBOARD*/)
{
    auto isNeedToSkip = [&vecFailedColRow](int col, int row) -> bool {
        for ( const auto &point : vecFailedColRow ) {
            if ( point.x == col && point.y == row )
                return true;
        }
        return false;
    };
    corners.clear();

    switch(patternType)
    {
    case CHESSBOARD:
    case CIRCLES_GRID:
        for( int row = 0; row < szBoardSize.height; ++ row )
        for( int col = 0; col < szBoardSize.width;  ++ col ) {
            if ( ! isNeedToSkip ( col, row ) )
                corners.push_back ( cv::Point3f ( col * squareSize, row * squareSize, 0 ) );
        }
        break;

    case ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < szBoardSize.height; ++ i )
            for( int j = 0; j < szBoardSize.width; ++ j )
                corners.push_back(cv::Point3f((2*j + i % 2)*squareSize, i*squareSize, 0));
        break;
    default:
        break;
    }
}

/*static*/ float VisionAlgorithm::_findChessBoardBlockSize ( const cv::Mat &matInputImg, const cv::Size szBoardPattern  ) {
    cv::Size2f size;
    int nRoughBlockSize = matInputImg.rows / szBoardPattern.width / 2;
    cv::Mat matROI ( matInputImg, cv::Rect ( matInputImg.cols / 2 - nRoughBlockSize * 2,  matInputImg.rows / 2 - nRoughBlockSize * 2, nRoughBlockSize * 4, nRoughBlockSize * 4 ) );  //Use the ROI 200x200 in the center of the image.
    cv::Mat matFilter;
    cv::GaussianBlur ( matROI, matFilter, cv::Size(3, 3), 1, 1 );
    cv::Mat matThreshold;
    int nThreshold = _autoThreshold ( matFilter );
    cv::threshold ( matFilter, matThreshold, nThreshold + 20, 255, cv::THRESH_BINARY_INV );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("findChessBoardBlockSize Threshold image", matThreshold);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours ( matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );
    
    float fMaxBlockArea = 0;
    for ( const auto &contour : contours ) {
        auto area = cv::contourArea ( contour );
        if ( area > 1000 )  {
            cv::RotatedRect rotatedRect = cv::minAreaRect ( contour );
            if ( ( fabs ( rotatedRect.size.width - rotatedRect.size.height ) / ( rotatedRect.size.width + rotatedRect.size.height ) < 0.05 ) &&
                ( rotatedRect.size.width * 2 * ( szBoardPattern.width - 1 ) < matInputImg.cols ) ) {
                float fArea = rotatedRect.size.width * rotatedRect.size.height;
                if ( fArea > fMaxBlockArea ) {
                    size = rotatedRect.size;
                    fMaxBlockArea = fArea;
                }
            }
        }
    }

    if ( fMaxBlockArea <= 0 ) {
        cv::threshold ( matFilter, matThreshold, nThreshold, 255, cv::THRESH_BINARY_INV );
        cv::Mat matKernal = cv::getStructuringElement ( cv::MorphShapes::MORPH_ELLIPSE, {6, 6} );
        cv::morphologyEx ( matThreshold, matThreshold, cv::MorphTypes::MORPH_CLOSE, matKernal, cv::Point(-1, -1), 3 );
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
            showImage ( "After morphologyEx image", matThreshold );
        }
        cv::findContours(matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (const auto &contour : contours) {
            auto area = cv::contourArea(contour);
            if (area > 1000)  {
                cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
                if ((fabs(rotatedRect.size.width - rotatedRect.size.height) / (rotatedRect.size.width + rotatedRect.size.height) < 0.05) &&
                    (rotatedRect.size.width * 2 * (szBoardPattern.width - 1) < matInputImg.cols)) {
                    float fArea = rotatedRect.size.width * rotatedRect.size.height;
                    if (fArea > fMaxBlockArea) {
                        size = rotatedRect.size;
                        fMaxBlockArea = fArea;
                    }
                }
            }
        }
    }

    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
        cv::Mat matDisplay;
        cv::cvtColor ( matROI, matDisplay, CV_GRAY2BGR );
        for ( size_t i = 0; i < contours.size(); ++ i )    {
            cv::drawContours ( matDisplay, contours, ToInt32(i), cv::Scalar(0,0,255) );
        }
        showImage("findChessBoardBlockSize contour image", matDisplay);
    }
    return  ( size.width + size.height ) / 2;
}

/*static*/ cv::Point2f VisionAlgorithm::_findFirstChessBoardCorner(const cv::Mat &matInputImg, float fBlockSize) {
    int nCornerRoiSize = ToInt32 ( fBlockSize * 3 );
    if ( nCornerRoiSize < 300 ) nCornerRoiSize = 300;
    cv::Mat matROI ( matInputImg, cv::Rect ( 0, 0, nCornerRoiSize, nCornerRoiSize ) );
    cv::Mat matFilter, matThreshold;
    cv::GaussianBlur ( matROI, matFilter, cv::Size(3, 3), 0.5, 0.5 );
    int nThreshold = _autoThreshold ( matFilter );
    nThreshold += 20;
    VectorOfVectorOfPoint contours, vecDrawContours;
    VectorOfPoint2f vecBlockCenters;

    //If didn't find the block, it may because the two blocks connected together make findContour find two connected blocks.
    //Need to use Morphology method to remove the dark connecter between them.
    cv::threshold(matFilter, matThreshold, nThreshold, 255, cv::THRESH_BINARY_INV);
    cv::Mat matKernal = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, { 7, 7 });
    cv::morphologyEx(matThreshold, matThreshold, cv::MorphTypes::MORPH_CLOSE, matKernal, cv::Point(-1, -1), 3);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("After morphologyEx image", matThreshold);
    }
    cv::findContours(matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
        auto area = cv::contourArea(contour);
        if (area > 1000) {
            cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
            if (fabs(rotatedRect.size.width - rotatedRect.size.height) / (rotatedRect.size.width + rotatedRect.size.height) < 0.1
                && fabs(rotatedRect.size.width - fBlockSize) / fBlockSize < 0.2) {
                vecDrawContours.push_back(contour);
                vecBlockCenters.push_back(rotatedRect.center);
            }
        }
    }

    float minDistanceToZero = 10000000;
    cv::Point ptUpperLeftBlockCenter;
    for ( const auto point : vecBlockCenters ) {
        auto distanceToZero = point.x * point.x + point.y * point.y;
        if ( distanceToZero < minDistanceToZero ) {
            ptUpperLeftBlockCenter = point;
            minDistanceToZero = distanceToZero;
        }          
    }
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
        cv::Mat matDisplay;
        cv::cvtColor ( matROI, matDisplay, CV_GRAY2BGR );
        for ( size_t i = 0; i < contours.size(); ++ i ) {
            cv::drawContours ( matDisplay, contours, ToInt32(i), _constRedScalar );
        }
        showImage("findChessBoardBlockSize contour image", matDisplay);
    }
    return cv::Point2f ( ptUpperLeftBlockCenter.x - fBlockSize / 2, ptUpperLeftBlockCenter.y - fBlockSize / 2 );
}

/*static*/ VisionStatus VisionAlgorithm::calibrateCamera(const PR_CALIBRATE_CAMERA_CMD *const pstCmd, PR_CALIBRATE_CAMERA_RPY *const pstRpy, bool bReplay) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( pstCmd->fPatternDist <= 0 ) {
        _snprintf(charrMsg, sizeof(charrMsg), "Pattern distance %f is invalid.", pstCmd->fPatternDist );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( pstCmd->szBoardPattern.width <= 0 || pstCmd->szBoardPattern.height <= 0 ) {
        _snprintf(charrMsg, sizeof(charrMsg), "Board pattern size %d, %d is invalid.", pstCmd->szBoardPattern.width, pstCmd->szBoardPattern.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseCalibrateCamera);
    
    cv::Mat matGray = pstCmd->matInputImg;
    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor ( pstCmd->matInputImg, matGray, CV_BGR2GRAY );

    VectorOfVectorOfPoint2f vecVecImagePoints(1);
    std::vector<std::vector<cv::Point3f> > vevVecObjectPoints(1);
    cv::Size imageSize = matGray.size();
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat matRotation, matTransformation;

    //Create the template chessbord corner. The upper left and lower right is white.
    cv::Mat matTmpl = cv::Mat::zeros(40, 40, CV_8UC1);
    cv::Mat matTmplROI(matTmpl, cv::Rect(0, 0, 20, 20));
    matTmplROI.setTo(cv::Scalar::all(PR_MAX_GRAY_LEVEL));
    cv::Mat matTmplROI1(matTmpl, cv::Rect(20, 20, 20, 20));
    matTmplROI1.setTo(cv::Scalar::all(PR_MAX_GRAY_LEVEL));

    float fBlockSize = _findChessBoardBlockSize ( matGray, pstCmd->szBoardPattern );
    if ( fBlockSize <= 5 ) {
        WriteLog("Failed to find chess board block size");
        pstRpy->enStatus = VisionStatus::FAILED_TO_FIND_CHESS_BOARD_BLOCK_SIZE;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    float fStepSize = fBlockSize * 2;
    int nSrchSize = ToInt32( fStepSize - 20 );
    if ( nSrchSize > 200 ) nSrchSize = 200;
    cv::Point2f ptFirstCorer = _findFirstChessBoardCorner ( matGray, fBlockSize );
    if ( ptFirstCorer.x <= 0 || ptFirstCorer.y <= 0 ) {
        WriteLog("Failed to find chess board first corner.");
        pstRpy->enStatus = VisionStatus::FAILED_TO_FIND_FIRST_CB_CORNER;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    cv::Point ptChessBoardSrchStartPoint = cv::Point ( ToInt32 ( ptFirstCorer.x - nSrchSize / 2 ), ToInt32 ( ptFirstCorer.y - nSrchSize/ 2 ) );
    if ( ptChessBoardSrchStartPoint.x < 0 || ptChessBoardSrchStartPoint.y < 0 ) {
        ptChessBoardSrchStartPoint.x += ToInt32( fBlockSize );
        ptChessBoardSrchStartPoint.y += ToInt32( fBlockSize );
    }
    VectorOfPoint vecFailedColRow;
    pstRpy->enStatus = _findChessBoardCorners ( matGray, matTmpl, pstRpy->matCornerPointsImg, ptChessBoardSrchStartPoint, fStepSize, nSrchSize, pstCmd->fMinTmplMatchScore, pstCmd->szBoardPattern, vecVecImagePoints[0], vecFailedColRow );
    if ( VisionStatus::OK != pstRpy->enStatus ) {
        WriteLog("Failed to find chess board corners.");
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    pstRpy->vecImagePoints = vecVecImagePoints[0];
    
    _calcBoardCornerPositions ( pstCmd->szBoardPattern, pstCmd->fPatternDist, vecFailedColRow, vevVecObjectPoints[0], CHESSBOARD );
    vevVecObjectPoints.resize ( vecVecImagePoints.size(), vevVecObjectPoints[0] );
    pstRpy->vecObjectPoints = vevVecObjectPoints[0];
    
    double rms = cv::calibrateCamera ( vevVecObjectPoints, vecVecImagePoints, imageSize, pstRpy->matIntrinsicMatrix,
            pstRpy->matDistCoeffs, rvecs, tvecs );
    
    cv::Rodrigues ( rvecs[0], matRotation );
    pstRpy->matExtrinsicMatrix = matRotation.clone();
    cv::transpose( pstRpy->matExtrinsicMatrix, pstRpy->matExtrinsicMatrix );
    matTransformation = tvecs[0];
    cv::transpose ( matTransformation, matTransformation );
    pstRpy->matExtrinsicMatrix.push_back( matTransformation );
    cv::transpose( pstRpy->matExtrinsicMatrix, pstRpy->matExtrinsicMatrix );

    //To output the initial camera matrix to do comarison.
    {
        pstRpy->matInitialIntrinsicMatrix = cv::initCameraMatrix2D( vevVecObjectPoints, vecVecImagePoints, imageSize, 0.);
        cv::Mat rvec, tvec, matRotation, matTransformation;
        bool bResult = cv::solvePnP( vevVecObjectPoints[0], vecVecImagePoints[0], pstRpy->matInitialIntrinsicMatrix, cv::Mat(), rvec, tvec );
        cv::Rodrigues ( rvec, matRotation );
        pstRpy->matInitialExtrinsicMatrix = matRotation.clone();
        cv::transpose( pstRpy->matInitialExtrinsicMatrix, pstRpy->matInitialExtrinsicMatrix );
        matTransformation = tvec;
        cv::transpose ( matTransformation, matTransformation );
        pstRpy->matInitialExtrinsicMatrix.push_back( matTransformation );
        cv::transpose( pstRpy->matInitialExtrinsicMatrix, pstRpy->matInitialExtrinsicMatrix );
    }    

    double z = pstRpy->matExtrinsicMatrix.at<double>(2, 3); //extrinsic matrix is 3x4 matric, the lower right element is z.
    double fx = pstRpy->matIntrinsicMatrix.at<double>(0, 0);
    double fy = pstRpy->matIntrinsicMatrix.at<double>(1, 1);
    pstRpy->dResolutionX = PR_MM_TO_UM * z / fx;
    pstRpy->dResolutionY = PR_MM_TO_UM * z / fy;

    cv::Mat matMapX, matMapY;
    cv::Mat matNewCemraMatrix = cv::getOptimalNewCameraMatrix ( pstRpy->matIntrinsicMatrix, pstRpy->matDistCoeffs, imageSize, 1, imageSize, 0 );
    cv::initUndistortRectifyMap ( pstRpy->matIntrinsicMatrix, pstRpy->matDistCoeffs, cv::Mat(), matNewCemraMatrix,
        imageSize, CV_32FC1, matMapX, matMapY );
    pstRpy->vecMatRestoreImage.push_back ( matMapX );
    pstRpy->vecMatRestoreImage.push_back ( matMapY );

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::restoreImage(const PR_RESTORE_IMG_CMD *const pstCmd, PR_RESTORE_IMG_RPY *const pstRpy, bool bReplay/* = false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( pstCmd->vecMatRestoreImage.size() != 2 ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The remap mat vector size %d is invalid.", pstCmd->vecMatRestoreImage.size() );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( pstCmd->vecMatRestoreImage[0].size() != pstCmd->matInputImg.size() || pstCmd->vecMatRestoreImage[1].size() != pstCmd->matInputImg.size() ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The remap mat size(%d, %d) is not match with input image size (%d, %d).", 
            pstCmd->vecMatRestoreImage[0].size().width, pstCmd->vecMatRestoreImage[0].size().height,
            pstCmd->matInputImg.size().width, pstCmd->matInputImg.size().height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseRestoreImg);

    cv::Mat matInputFloat;
    pstCmd->matInputImg.convertTo ( matInputFloat, CV_32FC1 );
    cv::remap ( matInputFloat, pstRpy->matResultImg, pstCmd->vecMatRestoreImage[0], pstCmd->vecMatRestoreImage[1], cv::INTER_LINEAR);

    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calcUndistortRectifyMap(const PR_CALC_UNDISTORT_RECTIFY_MAP_CMD *const pstCmd, PR_CALC_UNDISTORT_RECTIFY_MAP_RPY *const pstRpy) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if ( pstCmd->matIntrinsicMatrix.empty() || pstCmd->matIntrinsicMatrix.cols != 3 || pstCmd->matIntrinsicMatrix.rows != 3 ) {
        WriteLog("The intrinsic matrix is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    MARK_FUNCTION_START_TIME;

    cv::Mat matMapX, matMapY;
    cv::Mat matNewCemraMatrix = cv::getOptimalNewCameraMatrix ( pstCmd->matIntrinsicMatrix, pstCmd->matDistCoeffs, pstCmd->szImage, 1, pstCmd->szImage, 0 );
    cv::initUndistortRectifyMap ( pstCmd->matIntrinsicMatrix, pstCmd->matDistCoeffs, cv::Mat(), matNewCemraMatrix,
        pstCmd->szImage, CV_32FC1, matMapX, matMapY );
    pstRpy->vecMatRestoreImage.push_back ( matMapX );
    pstRpy->vecMatRestoreImage.push_back ( matMapY );

    pstRpy->enStatus = VisionStatus::OK;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::autoLocateLead(const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy, bool bReplay/* = false*/ )
{
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    if ( ! pstCmd->rectSrchWindow.contains ( cv::Point ( pstCmd->rectChipBody.x, pstCmd->rectChipBody.y ) )
        || ! pstCmd->rectSrchWindow.contains ( cv::Point ( pstCmd->rectChipBody.x + pstCmd->rectChipBody.width, pstCmd->rectChipBody.y + pstCmd->rectChipBody.height ) ) ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The chip window(%d, %d, %d, %d) should inside search window (%d, %d, %d, %d).", 
            pstCmd->rectChipBody.x, pstCmd->rectChipBody.y, pstCmd->rectChipBody.width, pstCmd->rectChipBody.height,
            pstCmd->rectSrchWindow.x, pstCmd->rectSrchWindow.y, pstCmd->rectSrchWindow.width, pstCmd->rectSrchWindow.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    if (pstCmd->rectSrchWindow.x < 0 || pstCmd->rectSrchWindow.y < 0 ||
        pstCmd->rectSrchWindow.width <= 0 || pstCmd->rectSrchWindow.height <= 0 ||
        ( pstCmd->rectSrchWindow.x + pstCmd->rectSrchWindow.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectSrchWindow.y + pstCmd->rectSrchWindow.height ) > pstCmd->matInputImg.rows ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The input search window (%d, %d, %d, %d) is invalid",
            pstCmd->rectSrchWindow.x, pstCmd->rectSrchWindow.y, pstCmd->rectSrchWindow.width, pstCmd->rectSrchWindow.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseAutoLocateLead);

    cv::Mat matGray;
    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor ( pstCmd->matInputImg, matGray, CV_BGR2GRAY);
    else
        matGray = pstCmd->matInputImg;

    cv::Mat matROI ( matGray, pstCmd->rectSrchWindow);
    cv::Mat matFilter;
    cv::GaussianBlur ( matROI, matFilter, cv::Size(3, 3), 1, 1 );

    cv::Mat matMask = cv::Mat::ones ( matROI.size(), CV_8UC1 );
    cv::Rect rectIcBodyInROI(pstCmd->rectChipBody.x - pstCmd->rectSrchWindow.x, pstCmd->rectChipBody.y - pstCmd->rectSrchWindow.y,
        pstCmd->rectChipBody.width, pstCmd->rectChipBody.height );
    cv::Mat matMaskROI(matMask, rectIcBodyInROI);
    matMaskROI.setTo(cv::Scalar(0));
    int nThreshold = _autoThreshold ( matFilter, matMask );

    matFilter.setTo ( cv::Scalar(0), cv::Scalar(1) - matMask ); //Set the body of the IC to pure dark.
    cv::Mat matThreshold;
    cv::threshold ( matFilter, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, cv::THRESH_BINARY );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("autoLocateLead threshold image", matThreshold );

    PR_FILL_HOLE_CMD stFillHoleCmd;
    PR_FILL_HOLE_RPY stFillHoleRpy;
    stFillHoleCmd.matInputImg = matThreshold;
    stFillHoleCmd.enAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stFillHoleCmd.enMethod = PR_FILL_HOLE_METHOD::CONTOUR;
    stFillHoleCmd.rectROI = cv::Rect(0, 0, matThreshold.cols, matThreshold.rows);
    fillHole(&stFillHoleCmd, &stFillHoleRpy);
    if ( VisionStatus::OK != stFillHoleRpy.enStatus ) {
        pstRpy->enStatus = stFillHoleRpy.enStatus;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    matThreshold = stFillHoleRpy.matResultImg;
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("autoLocateLead fillHole image", matThreshold );

    PR_REMOVE_CC_CMD stRemoveCcCmd;
    PR_REMOVE_CC_RPY stRemoveCcRpy;
    stRemoveCcCmd.matInputImg = matThreshold;
    stRemoveCcCmd.enCompareType = PR_COMPARE_TYPE::SMALLER;
    stRemoveCcCmd.nConnectivity = 4;
    stRemoveCcCmd.rectROI = cv::Rect(0, 0, matThreshold.cols, matThreshold.rows);
    stRemoveCcCmd.fAreaThreshold = 200;
    removeCC(&stRemoveCcCmd, &stRemoveCcRpy);
    if ( VisionStatus::OK != stRemoveCcRpy.enStatus ) {
        pstRpy->enStatus = stRemoveCcRpy.enStatus;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    matThreshold = stRemoveCcRpy.matResultImg;
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("autoLocateLead removeCC image", matThreshold );

    VectorOfVectorOfPoint contours, vecDrawContours;
    cv::findContours ( matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );
    if ( contours.size() <= 0 ) {
        pstRpy->enStatus = VisionStatus::FAILED_TO_FIND_LEAD;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    double dTotalContourArea = 0.;
    double dTotalRectArea = 0.;
    for ( const auto &contour : contours )  {
        auto area = cv::contourArea ( contour );
        dTotalContourArea += area;
        cv::Rect rect = cv::boundingRect(contour);
        dTotalRectArea += rect.width * rect.height;
    }
    double dAverageContourArea = dTotalContourArea / contours.size();
    double dAverageRectArea = dTotalRectArea / contours.size();
    double dMinAcceptContourArea = 0.4 * dAverageContourArea;
    double dMaxAcceptContourArea = 1.5 * dAverageContourArea;
    double dMinAcceptRectArea = 0.4 * dAverageRectArea;
    double dMaxAcceptRectArea = 1.5 * dAverageRectArea;

    cv::cvtColor ( matROI, pstRpy->matResultImg, CV_GRAY2BGR );

    for ( const auto &contour : contours ) {
        cv::Rect rect = cv::boundingRect(contour);
        auto contourArea = cv::contourArea ( contour );
        auto rectArea = rect.width * rect.height;
        if ( dMinAcceptContourArea < contourArea && contourArea < dMaxAcceptContourArea &&
            dMinAcceptRectArea < rectArea && rectArea < dMaxAcceptRectArea ) {
            if ( ( rect.br().x < rectIcBodyInROI.x ) && 
               ( ( rect.y < rectIcBodyInROI.y ) || ( rect.br().y > rectIcBodyInROI.br().y ) ) ) {
                cv::rectangle ( pstRpy->matResultImg, rect, _constRedScalar);
                continue;
            }

            if ( ( rect.x > rectIcBodyInROI.br().x ) && 
               ( ( rect.y < rectIcBodyInROI.y ) || ( rect.br().y > rectIcBodyInROI.br().y ) ) ) {
                cv::rectangle ( pstRpy->matResultImg, rect, _constRedScalar);
                continue;
            }

            if ( ( rect.br().y < rectIcBodyInROI.y ) && 
               ( ( rect.x < rectIcBodyInROI.x ) || ( rect.br().x > rectIcBodyInROI.br().x ) ) ) {
                cv::rectangle ( pstRpy->matResultImg, rect, _constRedScalar);
                continue;
            }

            if ( ( rect.y > rectIcBodyInROI.br().y ) && 
               ( ( rect.x < rectIcBodyInROI.x ) || ( rect.br().x > rectIcBodyInROI.br().x ) ) ) {
                cv::rectangle ( pstRpy->matResultImg, rect, _constRedScalar);
                continue;
            }
            cv::rectangle ( pstRpy->matResultImg, rect, _constGreenScalar );

            rect.x += pstCmd->rectSrchWindow.x;
            rect.y += pstCmd->rectSrchWindow.y;
            pstRpy->vecLeadLocation.push_back ( rect );
        }
    }
    pstRpy->enStatus = VisionStatus::OK;
    FINISH_LOGCASE;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspBridgeItem(const cv::Mat &matGray, cv::Mat &matResultImg, const PR_INSP_BRIDGE_CMD::INSP_ITEM &inspItem, PR_INSP_BRIDGE_RPY::ITEM_RESULT &inspResult) {
    //Initialize the result.
    inspResult.bWithBridge = false;
    inspResult.vecBridgeWindow.clear();

    //Draw the innter window and outer window.
    cv::rectangle ( matResultImg, inspItem.rectInnerWindow, _constBlueScalar );
    cv::rectangle ( matResultImg, inspItem.rectOuterWindow, _constBlueScalar );

    cv::Mat matROI, matFilter, matMask, matThreshold;
    cv::Point ptRoiTL;    
    cv::Rect rectInnerWindowNew(inspItem.rectInnerWindow.x - inspItem.rectOuterWindow.x, inspItem.rectInnerWindow.y - inspItem.rectOuterWindow.y, inspItem.rectInnerWindow.width, inspItem.rectInnerWindow.height);
    if ( PR_INSP_BRIDGE_MODE::INNER == inspItem.enMode ) {
         matROI = cv::Mat( matGray, inspItem.rectInnerWindow ).clone();
         ptRoiTL = inspItem.rectInnerWindow.tl();
    }else {
        matROI = cv::Mat ( matGray, inspItem.rectOuterWindow ).clone();        
        matMask = cv::Mat::ones(matROI.size(), CV_8UC1);        
        cv::Mat matMaskInnerWindow ( matMask, rectInnerWindowNew);
        matMaskInnerWindow.setTo ( cv::Scalar ( 0 ) );
        ptRoiTL = inspItem.rectOuterWindow.tl();
    }
    if ( PR_INSP_BRIDGE_MODE::OUTER == inspItem.enMode )
        matROI.setTo ( cv::Scalar(0), cv::Scalar(1) - matMask ); //Set the body of the IC to pure dark.
    cv::GaussianBlur ( matROI, matFilter, cv::Size(3, 3), 1, 1 );
    int nThreshold = _autoThreshold ( matFilter, matMask );
    if ( nThreshold < Config::GetInstance()->getInspBridgeMinThreshold() )
        nThreshold = Config::GetInstance()->getInspBridgeMinThreshold();

    cv::threshold ( matFilter, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, cv::THRESH_BINARY );

    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("_inspBridgeItem threshold", matThreshold );

    VectorOfVectorOfPoint contours, vecDrawContours;
    cv::findContours ( matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );
    if ( contours.size() <= 0 )        
        return VisionStatus::OK;

    const int MARGIN = 1;
    for ( const auto &contour : contours ) {
        cv::Rect rect = cv::boundingRect ( contour );
        if (PR_INSP_BRIDGE_MODE::INNER == inspItem.enMode) {
            if (rect.width > inspItem.stInnerInspCriteria.fMaxLengthX || rect.height > inspItem.stInnerInspCriteria.fMaxLengthY) {
                inspResult.bWithBridge = true;
                cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                inspResult.vecBridgeWindow.push_back(rectResult);
                cv::rectangle(matResultImg, rectResult, _constRedScalar, 1);
            }
        }
        else {
            for (const auto enDirection : inspItem.vecOuterInspDirection)
            if (PR_INSP_BRIDGE_DIRECTION::UP == enDirection) {
                int nTolerance = inspItem.rectInnerWindow.y - inspItem.rectOuterWindow.y - MARGIN;
                if (rect.br().y <= rectInnerWindowNew.tl().y && rect.height >= nTolerance) {
                    inspResult.bWithBridge = true;
                    cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                    inspResult.vecBridgeWindow.push_back(rectResult);
                    cv::rectangle(matResultImg, rectResult, _constRedScalar, 1);
                }
            }
            else if (PR_INSP_BRIDGE_DIRECTION::DOWN == enDirection) {
                int nTolerance = (inspItem.rectOuterWindow.y + inspItem.rectOuterWindow.height) - (inspItem.rectInnerWindow.y + inspItem.rectInnerWindow.height) - MARGIN;
                if (rect.tl().y >= rectInnerWindowNew.br().y && rect.height >= nTolerance) {
                    inspResult.bWithBridge = true;
                    cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                    inspResult.vecBridgeWindow.push_back(rectResult);
                    cv::rectangle(matResultImg, rectResult, _constRedScalar, 1);
                }
            }
            else
            if (PR_INSP_BRIDGE_DIRECTION::LEFT == enDirection) {
                int nTolerance = inspItem.rectInnerWindow.x - inspItem.rectOuterWindow.x - MARGIN;
                if (rect.br().x <= rectInnerWindowNew.tl().x && rect.width >= nTolerance) {
                    inspResult.bWithBridge = true;
                    cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                    inspResult.vecBridgeWindow.push_back(rectResult);
                    cv::rectangle(matResultImg, rectResult, _constRedScalar, 1);
                }
            }
            else
            if (PR_INSP_BRIDGE_DIRECTION::RIGHT == enDirection) {
                int nTolerance = (inspItem.rectOuterWindow.x + inspItem.rectOuterWindow.width) - (inspItem.rectInnerWindow.x + inspItem.rectInnerWindow.width) - MARGIN;
                if (rect.tl().x >= rectInnerWindowNew.br().x && rect.width >= nTolerance) {
                    inspResult.bWithBridge = true;
                    cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                    inspResult.vecBridgeWindow.push_back(rectResult);
                    cv::rectangle(matResultImg, rectResult, _constRedScalar, 1);
                }
            }
        }
    }
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::inspBridge(const PR_INSP_BRIDGE_CMD *const pstCmd, PR_INSP_BRIDGE_RPY *const pstRpy, bool bReplay /*= false*/ ) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( pstCmd->vecInspItems.empty() ) {
        WriteLog("The inspect items is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspBridge);

    cv::Mat matGray;
    if ( pstCmd->matInputImg.channels() > 1 )
        cv::cvtColor(pstCmd->matInputImg, matGray, CV_BGR2GRAY );
    else
        matGray = pstCmd->matInputImg.clone();
    cv::cvtColor ( matGray, pstRpy->matResultImg, CV_GRAY2BGR );

    int nIndexOfInspItem = 0;
    for ( const auto &inspItem : pstCmd->vecInspItems ) {
        if ( PR_INSP_BRIDGE_MODE::OUTER == inspItem.enMode ) {
            auto rectROI = inspItem.rectOuterWindow;
            if (rectROI.x < 0 || rectROI.y < 0 || rectROI.width <= 0 || rectROI.height <= 0
                || (rectROI.x + rectROI.width) > pstCmd->matInputImg.cols || (rectROI.y + rectROI.height) > pstCmd->matInputImg.rows)
            {
                _snprintf(charrMsg, sizeof(charrMsg), "The insect bridge outer check window (%d, %d, %d, %d) is invalid.",
                    rectROI.x, rectROI.y, rectROI.width, rectROI.height);
                WriteLog(charrMsg);
                pstRpy->enStatus = VisionStatus::INVALID_PARAM;
                return VisionStatus::INVALID_PARAM;
            }
            
            auto rectInnterWindow = inspItem.rectInnerWindow;
            if (!rectROI.contains(cv::Point(rectInnterWindow.x, rectInnterWindow.y)) ||
                !rectROI.contains(cv::Point(rectInnterWindow.x + rectInnterWindow.width, rectInnterWindow.y + rectInnterWindow.height))) {
                _snprintf(charrMsg, sizeof(charrMsg), "The inner check window(%d, %d, %d, %d) should inside outer check window (%d, %d, %d, %d).",
                    rectInnterWindow.x, rectInnterWindow.y, rectInnterWindow.width, rectInnterWindow.height,
                    rectROI.x, rectROI.y, rectROI.width, rectROI.height);
                WriteLog(charrMsg);
                pstRpy->enStatus = VisionStatus::INVALID_PARAM;
                return pstRpy->enStatus;
            }

            if ( inspItem.vecOuterInspDirection.empty() ) {
                _snprintf(charrMsg, sizeof(charrMsg), "The outer inspection direction of inspect Item %d is empty.", nIndexOfInspItem);
                WriteLog(charrMsg);
                pstRpy->enStatus = VisionStatus::INVALID_PARAM;
                return VisionStatus::INVALID_PARAM;
            }
        }else if ( PR_INSP_BRIDGE_MODE::INNER == inspItem.enMode ) {
            auto rectROI = inspItem.rectInnerWindow;
            if (rectROI.x < 0 || rectROI.y < 0 || rectROI.width <= 0 || rectROI.height <= 0
                || (rectROI.x + rectROI.width) > pstCmd->matInputImg.cols || (rectROI.y + rectROI.height) > pstCmd->matInputImg.rows) {
                _snprintf(charrMsg, sizeof(charrMsg), "The insect bridge outer check window (%d, %d, %d, %d) is invalid.",
                    rectROI.x, rectROI.y, rectROI.width, rectROI.height);
                WriteLog(charrMsg);
                pstRpy->enStatus = VisionStatus::INVALID_PARAM;
                return VisionStatus::INVALID_PARAM;
            }

            if ( inspItem.stInnerInspCriteria.fMaxLengthX <= 0 || inspItem.stInnerInspCriteria.fMaxLengthY <= 0 ) {
                _snprintf(charrMsg, sizeof(charrMsg), "The inner inspection criteria (%.2f, %.2f) of inspect Item %d is empty.", 
                    inspItem.stInnerInspCriteria.fMaxLengthX, inspItem.stInnerInspCriteria.fMaxLengthY, nIndexOfInspItem );
                WriteLog(charrMsg);
                pstRpy->enStatus = VisionStatus::INVALID_PARAM;
                return VisionStatus::INVALID_PARAM;
            }
        }
        
        PR_INSP_BRIDGE_RPY::ITEM_RESULT itemResult;
        pstRpy->enStatus =  _inspBridgeItem ( matGray, pstRpy->matResultImg, inspItem, itemResult);
        if ( VisionStatus::OK != pstRpy->enStatus ) {
            FINISH_LOGCASE;
            return pstRpy->enStatus;
        }
        pstRpy->vecInspResults.push_back ( itemResult );
        ++ nIndexOfInspItem;
    }

    pstRpy->enStatus = VisionStatus::OK;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::lrnChip( const PR_LRN_CHIP_CMD *const pstCmd, PR_LRN_CHIP_RPY *const pstRpy, bool bReplay /*= false*/ ) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
   
    if (pstCmd->rectChip.x < 0 || pstCmd->rectChip.y < 0 ||
        pstCmd->rectChip.width <= 0 || pstCmd->rectChip.height <= 0 ||
        ( pstCmd->rectChip.x + pstCmd->rectChip.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectChip.y + pstCmd->rectChip.height ) > pstCmd->matInputImg.rows ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The input chip window (%d, %d, %d, %d) is invalid",
            pstCmd->rectChip.x, pstCmd->rectChip.y, pstCmd->rectChip.width, pstCmd->rectChip.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;

    cv::Rect rectChipROI = CalcUtils::resizeRect<float> ( pstCmd->rectChip, pstCmd->rectChip.size() + cv::Size2f(20, 20) );
    if ( rectChipROI.x < 0  ) rectChipROI.x = 0;
    if ( rectChipROI.y < 0  ) rectChipROI.y = 0;
    if ( ( rectChipROI.x + rectChipROI.width ) > pstCmd->matInputImg.cols ) rectChipROI.width  = pstCmd->matInputImg.cols - rectChipROI.x;
    if ( ( rectChipROI.y + rectChipROI.height) > pstCmd->matInputImg.rows ) rectChipROI.height = pstCmd->matInputImg.rows - rectChipROI.y;
    cv::Mat matROI ( pstCmd->matInputImg, rectChipROI );
    cv::Mat matGray, matBlur, matThreshold;
    if ( matROI.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();
    cv::GaussianBlur ( matGray, matBlur, cv::Size(5, 5), 2, 2 );
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Blur image", matBlur);

    pstRpy->nThreshold = pstCmd->nThreshold;
    if ( pstCmd->bAutoThreshold )
        pstRpy->nThreshold = _autoThreshold ( matBlur );

    cv::ThresholdTypes enThresholdType = cv::THRESH_BINARY;
    if ( PR_INSP_CHIP_MODE::SQUARE == pstCmd->enInspMode )
        enThresholdType = cv::THRESH_BINARY_INV;
    cv::threshold ( matBlur, matThreshold, pstRpy->nThreshold, 255, enThresholdType );
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Threshold image", matThreshold);

    if ( pstCmd->matInputImg.channels() > 1 )
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor ( pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR );

    if ( PR_INSP_CHIP_MODE::HEAD == pstCmd->enInspMode ) {
        _lrnChipHeadMode ( matThreshold, rectChipROI, pstRpy );
    }else if ( PR_INSP_CHIP_MODE::SQUARE == pstCmd->enInspMode ) {
        _lrnChipSquareMode ( matThreshold, pstCmd->rectChip, rectChipROI, pstRpy );
    }else if ( PR_INSP_CHIP_MODE::CAE == pstCmd->enInspMode ) {
        _lrnChipCAEMode ( matThreshold, rectChipROI, pstRpy );
    }

    if ( VisionStatus::OK != pstRpy->enStatus ) {
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_END_TIME;

    pstRpy->enStatus = VisionStatus::OK;
    pstRpy->sizeDevice = pstCmd->rectChip.size();
    _writeChipRecord ( pstCmd, pstRpy );
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_lrnChipHeadMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_LRN_CHIP_RPY *const pstRpy) {
    VectorOfPoint vecPtElectrode;
    VectorOfSize2f vecElectrodeSize;
    VectorOfVectorOfPoint vecContours;
    pstRpy->enStatus = _findDeviceElectrode ( matThreshold, vecPtElectrode, vecElectrodeSize, vecContours );
    if ( pstRpy->enStatus != VisionStatus::OK) {
        WriteLog("Failed to find device electrode");
        return pstRpy->enStatus;
    }
    //for ( size_t index = 0; index < PR_ELECTRODE_COUNT && index < vecElectrodeSize.size(); ++ index ) {
    //    pstRpy->arrSizeElectrode[index] = vecElectrodeSize[index];
    //    cv::Rect2f rectElectrode ( vecPtElectrode[index].x - vecElectrodeSize[index].width / 2 + rectROI.x, vecPtElectrode[index].y - vecElectrodeSize[index].height / 2 + rectROI.y,
    //        vecElectrodeSize[index].width, vecElectrodeSize[index].height );
    //    cv::rectangle ( pstRpy->matResultImg, rectElectrode, _constBlueScalar );
    //}
    for ( auto &contour : vecContours ) {
        for ( auto &point : contour ) {
            point.x += rectROI.x;
            point.y += rectROI.y;
        }
    }
    cv::polylines( pstRpy->matResultImg, vecContours, true, _constBlueScalar, 2 );

    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_lrnChipCAEMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_LRN_CHIP_RPY *const pstRpy) {
    cv::Mat matFillHole;
    _fillHoleByContour ( matThreshold, matFillHole, PR_OBJECT_ATTRIBUTE::BRIGHT );

    VectorOfVectorOfPoint contours;
    cv::findContours( matFillHole, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );
    VectorOfPoint maxContour;
    float fMaxArea = std::numeric_limits<float>::min();
    for ( const auto &contour : contours ) {
        auto rectRotated = cv::minAreaRect ( contour );
        if ( rectRotated.size.area() > fMaxArea ) {
            maxContour = contour;
            fMaxArea = rectRotated.size.area();
        }
    }    
   
    int nMaxRansacTime = 20;
    float fFitTolerance = 4;
    auto fitCircleResult = _fitCircleRansac ( maxContour, fFitTolerance, nMaxRansacTime, maxContour.size() / 2 );
    auto vecLeftOverPoints = _findPointsOverCircleTol ( maxContour, fitCircleResult, fFitTolerance );    
    float fSlope = 0.f, fIntercept = 0.f;
    PR_Line2f stLine;
    _fitLineRansac ( vecLeftOverPoints, fFitTolerance, nMaxRansacTime, vecLeftOverPoints.size() / 2, false, fSlope, fIntercept, stLine );
    float fLineLength = CalcUtils::distanceOf2Point ( stLine.pt1, stLine.pt2 );

    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode() ) {
        cv::Mat matDisplay;
        cv::cvtColor ( matThreshold, matDisplay, CV_GRAY2BGR );
        cv::circle ( matDisplay, fitCircleResult.center, ToInt32 ( fitCircleResult.size.width / 2 + 0.5 ), _constBlueScalar, 2 );
        cv::line ( matDisplay, stLine.pt1, stLine.pt2, _constGreenScalar, 2 );
        showImage ( "lrnChipCAEMode", matDisplay );
    }

    if ( fitCircleResult.size.width < rectROI.width / 2 ) {
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        return pstRpy->enStatus;
    }

    if ( fLineLength < fitCircleResult.size.width / 2 || fLineLength > fitCircleResult.size.width ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CAE_LINE;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = VisionStatus::OK;
    for ( auto &point : maxContour ) {
        point.x += rectROI.x;
        point.y += rectROI.y;       
    }
    VectorOfVectorOfPoint vevVecPoint;
    vevVecPoint.push_back ( maxContour );
    cv::polylines ( pstRpy->matResultImg, vevVecPoint, true, _constBlueScalar, 2 );
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_lrnChipSquareMode(const cv::Mat &matThreshold, const cv::Rect &rectChip, const cv::Rect &rectROI, PR_LRN_CHIP_RPY *const pstRpy ) {
    cv::Mat matOneRow, matOneCol;
    cv::reduce ( matThreshold, matOneRow, 0, cv::ReduceTypes::REDUCE_SUM, CV_32SC1 );
    matOneRow /= 255;

    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode() ) {
        int nHeight = matThreshold.rows;
        cv::Mat matDisplay = cv::Mat::ones ( matThreshold.size(), CV_8UC3 ) * 255;
        matDisplay.setTo( cv::Scalar::all ( 255 ) );
        for ( int i = 1; i < matOneRow.cols; ++ i ) {
            cv::Point pt1 ( i - 1, nHeight - matOneRow.at<int>(0, i - 1) );
            cv::Point pt2 ( i, nHeight - matOneRow.at<int>(0, i) );
            cv::line ( matDisplay, pt1, pt2, _constBlueScalar, 2 );
        }
        showImage("Project Curve Row", matDisplay );
    }
    cv::reduce ( matThreshold, matOneCol, 1, cv::ReduceTypes::REDUCE_SUM, CV_32SC1 );
    matOneCol /= 255;
    cv::transpose ( matOneCol, matOneCol );
    int nLeftX = 0, nRightX = 0, nTopY = 0, nBottomY = 0;
    bool bSuccess = true;
    bSuccess = ( bSuccess && _findDataUpEdge   ( matOneRow, 0,                  matOneRow.cols / 2, nLeftX ) );
    bSuccess = ( bSuccess && _findDataDownEdge ( matOneRow, matOneRow.cols / 2, matOneRow.cols,     nRightX ) );

    bSuccess = ( bSuccess && _findDataUpEdge   ( matOneCol, 0,                  matOneCol.cols / 2, nTopY ) );
    bSuccess = ( bSuccess && _findDataDownEdge ( matOneCol, matOneCol.cols / 2, matOneCol.cols,     nBottomY ) );
    cv::rectangle ( pstRpy->matResultImg, cv::Point ( rectROI.x + nLeftX, rectROI.y + nTopY),
        cv::Point ( rectROI.x + nRightX, rectROI.y + nBottomY ), _constGreenScalar, 2 );
    
    if( ! bSuccess ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_SQUARE_EDGE;
        return pstRpy->enStatus;
    }

    const float INPUT_ERROR_TOL = 0.1f; //10% input error tolerance
    if ( abs ( nRightX - nLeftX - rectChip.width ) / ToFloat ( rectChip.width )  > INPUT_ERROR_TOL ||
        abs ( nBottomY - nTopY - rectChip.height ) / ToFloat ( rectChip.height ) > INPUT_ERROR_TOL ) {
        WriteLog("The found chip edges are not match with input chip window");
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_SQUARE_EDGE;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_writeChipRecord(const PR_LRN_CHIP_CMD *const pstCmd, PR_LRN_CHIP_RPY *const pstRpy) {
    ChipRecordPtr ptrRecord = std::make_shared<ChipRecord>( PR_RECORD_TYPE::CHIP );
    ptrRecord->setThreshold ( pstRpy->nThreshold );
    ptrRecord->setSize ( pstRpy->sizeDevice );
    ptrRecord->setInspMode ( pstCmd->enInspMode );
    RecordManager::getInstance()->add( ptrRecord, pstRpy->nRecordId );
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::inspChip( const PR_INSP_CHIP_CMD *const pstCmd, PR_INSP_CHIP_RPY *const pstRpy, bool bReplay /*= false*/ ) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
   
    if (pstCmd->rectSrchWindow.x < 0 || pstCmd->rectSrchWindow.y < 0 ||
        pstCmd->rectSrchWindow.width <= 0 || pstCmd->rectSrchWindow.height <= 0 ||
        ( pstCmd->rectSrchWindow.x + pstCmd->rectSrchWindow.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectSrchWindow.y + pstCmd->rectSrchWindow.height ) > pstCmd->matInputImg.rows ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The input search window (%d, %d, %d, %d) is invalid",
            pstCmd->rectSrchWindow.x, pstCmd->rectSrchWindow.y, pstCmd->rectSrchWindow.width, pstCmd->rectSrchWindow.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    ChipRecordPtr ptrRecord = nullptr;
    if ( pstCmd->nRecordId > 0 ) {
        ptrRecord = std::static_pointer_cast<ChipRecord> (RecordManager::getInstance ()->get ( pstCmd->nRecordId ));
        if (nullptr == ptrRecord) {
            _snprintf ( charrMsg, sizeof ( charrMsg ), "Failed to get record ID %d in system.", pstCmd->nRecordId );
            WriteLog ( charrMsg );
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (ptrRecord->getInspMode () != pstCmd->enInspMode) {
            _snprintf ( charrMsg, sizeof ( charrMsg ), "The record insp mode %d if record %d is not match with input insp mode %d",
                ToInt32 ( ptrRecord->getInspMode () ), pstCmd->nRecordId, ToInt32 ( pstCmd->enInspMode ) );
            WriteLog ( charrMsg );
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspChip);

    cv::Mat matROI ( pstCmd->matInputImg, pstCmd->rectSrchWindow );
    cv::Mat matGray, matBlur, matThreshold;
    if ( matROI.channels() > 1 ) {
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    }else {
        matGray = matROI.clone();
        cv::cvtColor ( pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR );
    }

    cv::GaussianBlur ( matGray, matBlur, cv::Size(5, 5), 2, 2 );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Blur image", matBlur);

    int nThreshold = 100;
    if ( ptrRecord != nullptr )
        nThreshold = ptrRecord->getThreshold();
    else
        nThreshold = _autoThreshold ( matBlur );

    cv::ThresholdTypes enThresholdType = cv::THRESH_BINARY;
    if ( PR_INSP_CHIP_MODE::SQUARE == pstCmd->enInspMode )
        enThresholdType = cv::THRESH_BINARY_INV;
    cv::threshold ( matBlur, matThreshold, nThreshold, 255, enThresholdType );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Threshold image", matThreshold);

    if ( PR_INSP_CHIP_MODE::HEAD == pstCmd->enInspMode ) {
        _inspChipHeadMode ( matThreshold, pstCmd->rectSrchWindow, pstRpy );
    }else if ( PR_INSP_CHIP_MODE::BODY == pstCmd->enInspMode ) {
        _inspChipBodyMode ( matThreshold, pstCmd->rectSrchWindow, pstRpy );
    }else if ( PR_INSP_CHIP_MODE::SQUARE == pstCmd->enInspMode ) {
        _inspChipSquareMode ( matThreshold, pstCmd->rectSrchWindow, pstRpy );
    }else if ( PR_INSP_CHIP_MODE::CAE == pstCmd->enInspMode ) {
        _inspChipCAEMode ( matThreshold, pstCmd->rectSrchWindow, pstRpy );
    }else if ( PR_INSP_CHIP_MODE::CIRCULAR == pstCmd->enInspMode )
        _inspChipCircularMode ( matThreshold, pstCmd->rectSrchWindow, pstRpy );

    pstRpy->rotatedRectResult.angle = fabs ( - pstRpy->rotatedRectResult.angle );

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipHeadMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy) {
    VectorOfPoint vecPtElectrode;
    VectorOfSize2f vecElectrodeSize;
    VectorOfVectorOfPoint vecContours;
    pstRpy->enStatus = _findDeviceElectrode ( matThreshold, vecPtElectrode, vecElectrodeSize, vecContours );
    if ( pstRpy->enStatus != VisionStatus::OK ) {
        WriteLog("Failed to find device electrode");
        return pstRpy->enStatus;
    }

    //Draw the electrodes.
    for ( auto &contour : vecContours ) {
        for ( auto &point : contour ) {
            point.x += rectROI.x;
            point.y += rectROI.y;
        }
    }
    cv::polylines( pstRpy->matResultImg, vecContours, true, _constBlueScalar, 2 );

    VectorOfPoint vecTwoContourTogether = vecContours[0];
    vecTwoContourTogether.insert ( vecTwoContourTogether.end(), vecContours[1].begin(), vecContours[1].end() );
    pstRpy->rotatedRectResult = cv::minAreaRect ( vecTwoContourTogether );
    VectorOfVectorOfPoint vevVecPoints;
    vevVecPoints.push_back ( CalcUtils::getCornerOfRotatedRect ( pstRpy->rotatedRectResult ) );
    cv::polylines( pstRpy->matResultImg, vevVecPoints, true, _constGreenScalar, 2 );
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipBodyMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy) {
    cv::Mat matLabels;
    cv::connectedComponents( matThreshold, matLabels, 4, CV_32S );

    double minValue = 0., maxValue = 0.;
    cv::minMaxIdx ( matLabels, &minValue, &maxValue );
    float fMaxArea = std::numeric_limits<float>::min();
    cv::Mat matMaxComponent;
    //0 represents the background label, so need to ignore.
    for ( int nLabel = 1; nLabel <= maxValue; ++ nLabel ) {
        cv::Mat matCC = CalcUtils::genMaskByValue<Int32>( matLabels, nLabel );
        float fArea = ToFloat ( cv::countNonZero ( matCC ) );
        if ( fArea > fMaxArea ) {
            fMaxArea = fArea;
            matMaxComponent = matCC;
        }
    }

    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Max Component", matMaxComponent );
    
    VectorOfVectorOfPoint contours;
    cv::findContours( matMaxComponent, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );
    if ( contours.size() <= 0 ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CIRCULAR_CHIP;
        return pstRpy->enStatus;
    }
    
    VectorOfPoint maxContour;
    fMaxArea = std::numeric_limits<float>::min();
    for ( const auto &contour : contours ) {
        auto rectRotated = cv::minAreaRect ( contour );
        if ( rectRotated.size.area() > fMaxArea ) {
            maxContour = contour;
            fMaxArea = rectRotated.size.area();
        }
    }

    auto rectBody = cv::minAreaRect ( maxContour );

    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode() ) {
        cv::Mat matDisplay;
        cv::cvtColor ( matThreshold, matDisplay, CV_GRAY2BGR );
        VectorOfVectorOfPoint vevVecPoints;
        vevVecPoints.push_back ( CalcUtils::getCornerOfRotatedRect ( rectBody ) );
        cv::polylines( matDisplay, vevVecPoints, true, _constGreenScalar, 2 );
        showImage ( "inspChipCircularMode", matDisplay );
    }

    float fTolerance = 4.f;
    if ( rectBody.size.width <= 2 * fTolerance || rectBody.size.height <= 2 * fTolerance ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CHIP_BODY;
        return pstRpy->enStatus;
    }

    float fMaxDim = std::max ( rectBody.size.width, rectBody.size.height );
    float fMinDim = std::min ( rectBody.size.width, rectBody.size.height );
    if ( fMaxDim / fMinDim  > 2.5 ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CHIP_BODY;
        return pstRpy->enStatus;
    }

    cv::Mat matVerify = cv::Mat::zeros ( rectROI.size(), CV_8UC1 );
    auto rectEnlarge = rectBody;
    rectEnlarge.size.width  += 2 * fTolerance;
    rectEnlarge.size.height += 2 * fTolerance;
    VectorOfVectorOfPoint vevVecPoint;
    vevVecPoint.push_back ( CalcUtils::getCornerOfRotatedRect( rectEnlarge ) );
    cv::fillPoly ( matVerify, vevVecPoint, cv::Scalar::all ( PR_MAX_GRAY_LEVEL ) );

    auto rectDeflate = rectBody;
    rectDeflate.size.width  -= 2 * fTolerance;
    rectDeflate.size.height -= 2 * fTolerance;
    vevVecPoint.clear();
    vevVecPoint.push_back ( CalcUtils::getCornerOfRotatedRect( rectDeflate ) );
    cv::fillPoly ( matVerify, vevVecPoint, cv::Scalar::all(0) );
    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode() )
        showImage ( "Verify Mat", matVerify );

    int nInTolPointCount = 0;
    for ( const auto &point : maxContour ) {
        if ( matVerify.at<uchar>(point) > 0 )
            ++ nInTolPointCount;
    }

    if ( nInTolPointCount < ToInt32 ( maxContour.size() ) / 2 ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CHIP_BODY;
        return pstRpy->enStatus;
    }

    rectBody.center.x += rectROI.x;
    rectBody.center.y += rectROI.y;
    pstRpy->rotatedRectResult = rectBody;

    VectorOfVectorOfPoint vevVecPoints;
    vevVecPoints.push_back ( CalcUtils::getCornerOfRotatedRect ( pstRpy->rotatedRectResult ) );
    cv::polylines( pstRpy->matResultImg, vevVecPoints, true, _constGreenScalar, 2 );
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipSquareMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy ) {
    cv::Mat matOneRow, matOneCol;
    cv::reduce ( matThreshold, matOneRow, 0, cv::ReduceTypes::REDUCE_SUM, CV_32SC1 );
    matOneRow /= 255;

    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode() ) {
        int nHeight = matThreshold.rows;
        cv::Mat matDisplay = cv::Mat::ones ( matThreshold.size(), CV_8UC3 ) * 255;
        matDisplay.setTo( cv::Scalar::all ( 255 ) );
        for ( int i = 1; i < matOneRow.cols; ++ i ) {
            cv::Point pt1 ( i - 1, nHeight - matOneRow.at<int>(0, i - 1) );
            cv::Point pt2 ( i, nHeight - matOneRow.at<int>(0, i) );
            cv::line ( matDisplay, pt1, pt2, _constBlueScalar, 2 );
        }
        showImage("Project Curve Row", matDisplay );
    }
    cv::reduce ( matThreshold, matOneCol, 1, cv::ReduceTypes::REDUCE_SUM, CV_32SC1 );
    matOneCol /= 255;
    cv::transpose ( matOneCol, matOneCol );

    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode() ) {
        int nHeight = matThreshold.cols;
        cv::Mat matDisplay = cv::Mat::ones ( cv::Size ( matThreshold.rows, matThreshold.cols ), CV_8UC3 ) * 255;
        matDisplay.setTo( cv::Scalar::all ( 255 ) );
        for ( int i = 1; i < matOneCol.cols; ++ i ) {
            cv::Point pt1 ( i - 1, nHeight - matOneCol.at<int>(0, i - 1) );
            cv::Point pt2 ( i, nHeight - matOneCol.at<int>(0, i) );
            cv::line ( matDisplay, pt1, pt2, _constBlueScalar, 2 );
        }
        showImage("Project Curve Col", matDisplay );
    }
    
    int nLeftX = 0, nRightX = 0, nTopY = 0, nBottomY = 0;
    bool bSuccess = true;
    bSuccess = ( bSuccess && _findDataUpEdge   ( matOneRow, 0,                  matOneRow.cols / 2, nLeftX ) );
    bSuccess = ( bSuccess && _findDataDownEdge ( matOneRow, matOneRow.cols / 2, matOneRow.cols,     nRightX ) );

    bSuccess = ( bSuccess && _findDataUpEdge   ( matOneCol, 0,                  matOneCol.cols / 2, nTopY ) );
    bSuccess = ( bSuccess && _findDataDownEdge ( matOneCol, matOneCol.cols / 2, matOneCol.cols,     nBottomY ) );
    cv::rectangle ( pstRpy->matResultImg, cv::Point ( rectROI.x + nLeftX, rectROI.y + nTopY),
        cv::Point ( rectROI.x + nRightX, rectROI.y + nBottomY ), _constGreenScalar, 2 );
    
    if( ! bSuccess ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_SQUARE_EDGE;
        return pstRpy->enStatus;
    }

    pstRpy->rotatedRectResult.angle = 0;
    pstRpy->rotatedRectResult.center.x = (nLeftX + nRightX  ) / 2.f + rectROI.x;
    pstRpy->rotatedRectResult.center.y = (nTopY  + nBottomY ) / 2.f + rectROI.y;
    pstRpy->rotatedRectResult.size.width  = ToFloat ( nRightX - nLeftX + 1 );
    pstRpy->rotatedRectResult.size.height = ToFloat ( nBottomY - nTopY + 1 );
    VectorOfVectorOfPoint vevVecPoints;
    vevVecPoints.push_back ( CalcUtils::getCornerOfRotatedRect ( pstRpy->rotatedRectResult ) );
    cv::polylines( pstRpy->matResultImg, vevVecPoints, true, _constGreenScalar, 2 );
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ bool VisionAlgorithm::_findDataUpEdge( const cv::Mat &matRow, int nStart, int nEnd, int &nEdgePos ) {
    const int EDGE_SIZE = 30;
    int EDGE_MIN_DIFF = matRow.cols / 3;
    nEdgePos = -1;
    for (int index = nStart; index < nEnd - EDGE_SIZE; ++index) {
        int nDiff = matRow.at<int> ( 0, index + EDGE_SIZE ) - matRow.at<int> ( 0, index );       
        if (nDiff > EDGE_MIN_DIFF)
            nEdgePos = index + EDGE_SIZE / 2;
    }
    if ( nEdgePos < 0 )
        return false;
    return true;
}

/*static*/ bool VisionAlgorithm::_findDataDownEdge ( const cv::Mat &matRow, int nStart, int nEnd, int &nEdgePos ) {
    const int EDGE_SIZE = 30;
    int EDGE_MIN_DIFF = matRow.cols / 4;
    nEdgePos = -1;
    for (int index = nEnd - EDGE_SIZE - 1; index >= nStart; -- index) {
        int nDiff = matRow.at<int>( 0, index ) - matRow.at<int> ( 0, index + EDGE_SIZE );     
        if ( nDiff > EDGE_MIN_DIFF )
            nEdgePos = index + EDGE_SIZE / 2;
    }
    if ( nEdgePos < 0 )
        return false;
    return true;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipCAEMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy) {
    cv::Mat matFillHole;
    _fillHoleByContour ( matThreshold, matFillHole, PR_OBJECT_ATTRIBUTE::BRIGHT );

    VectorOfVectorOfPoint contours;
    cv::findContours( matFillHole, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );
    VectorOfPoint maxContour;
    float fMaxArea = std::numeric_limits<float>::min();
    for ( const auto &contour : contours ) {
        auto rectRotated = cv::minAreaRect ( contour );
        if ( rectRotated.size.area() > fMaxArea ) {
            maxContour = contour;
            fMaxArea = rectRotated.size.area();
        }
    }
    
    int nMaxRansacTime = 20;
    float fFitTolerance = 4;
    auto fitCircleResult = _fitCircleRansac ( maxContour, fFitTolerance, nMaxRansacTime, maxContour.size() / 2 );
    auto vecLeftOverPoints = _findPointsOverCircleTol ( maxContour, fitCircleResult, fFitTolerance );    
    float fSlope = 0.f, fIntercept = 0.f;
    PR_Line2f stLine;
    _fitLineRansac ( vecLeftOverPoints, fFitTolerance, nMaxRansacTime, vecLeftOverPoints.size() / 2, false, fSlope, fIntercept, stLine );
    float fLineLength = CalcUtils::distanceOf2Point ( stLine.pt1, stLine.pt2 );

    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode() ) {
        cv::Mat matDisplay;
        cv::cvtColor ( matThreshold, matDisplay, CV_GRAY2BGR );
        cv::circle ( matDisplay, fitCircleResult.center, ToInt32 ( fitCircleResult.size.width / 2 + 0.5 ), _constBlueScalar, 2 );
        cv::line ( matDisplay, stLine.pt1, stLine.pt2, _constGreenScalar, 2 ); 
        showImage ( "lrnChipCAEMode", matDisplay );
    }

    if ( fitCircleResult.size.width < rectROI.width / 2 ) {
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        return pstRpy->enStatus;
    }

    if ( fLineLength < fitCircleResult.size.width / 2 || fLineLength > fitCircleResult.size.width ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CAE_LINE;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = VisionStatus::OK;
    for ( auto &point : maxContour ) {
        point.x += rectROI.x;
        point.y += rectROI.y;       
    }
    VectorOfVectorOfPoint vevVecPoint;
    vevVecPoint.push_back ( maxContour );
    cv::polylines ( pstRpy->matResultImg, vevVecPoint, true, _constBlueScalar, 2 );

    fitCircleResult.center.x += rectROI.x;
    fitCircleResult.center.y += rectROI.y;
    pstRpy->rotatedRectResult = fitCircleResult;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipCircularMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy) {
    cv::Mat matLabels;
    cv::connectedComponents( matThreshold, matLabels, 4, CV_32S );

    double minValue = 0., maxValue = 0.;
    cv::minMaxIdx ( matLabels, &minValue, &maxValue );
    float fMaxArea = std::numeric_limits<float>::min();
    cv::Mat matMaxComponent;
    //0 represents the background label, so need to ignore.
    for ( int nLabel = 1; nLabel <= maxValue; ++ nLabel ) {
        cv::Mat matCC = CalcUtils::genMaskByValue<Int32>( matLabels, nLabel );
        float fArea = ToFloat ( cv::countNonZero ( matCC ) );
        if ( fArea > fMaxArea ) {
            fMaxArea = fArea;
            matMaxComponent = matCC;
        }
    }

    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Max Component", matMaxComponent );
    
    VectorOfVectorOfPoint contours;
    cv::findContours( matMaxComponent, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );
    if ( contours.size() <= 0 ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CIRCULAR_CHIP;
        return pstRpy->enStatus;
    }
    
    VectorOfPoint maxContour;
    fMaxArea = std::numeric_limits<float>::min();
    for ( const auto &contour : contours ) {
        auto rectRotated = cv::minAreaRect ( contour );
        if ( rectRotated.size.area() > fMaxArea ) {
            maxContour = contour;
            fMaxArea = rectRotated.size.area();
        }
    }

    int nMaxRansacTime = 20;
    float fFitTolerance = 4;
    auto fitCircleResult = _fitCircleRansac ( maxContour, fFitTolerance, nMaxRansacTime, maxContour.size() / 2 );

    if ( PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode() ) {
        cv::Mat matDisplay;
        cv::cvtColor ( matThreshold, matDisplay, CV_GRAY2BGR );
        cv::circle ( matDisplay, fitCircleResult.center, ToInt32 ( fitCircleResult.size.width / 2 + 0.5 ), _constBlueScalar, 2 );
        showImage ( "inspChipCircularMode", matDisplay );
    }

    auto vecInTolPoints = _findPointsInCircleTol ( maxContour, fitCircleResult, fFitTolerance );
    if ( vecInTolPoints.size() < maxContour.size() / 2 ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CIRCULAR_CHIP;
        return pstRpy->enStatus;
    }

    fitCircleResult.center.x += rectROI.x;
    fitCircleResult.center.y += rectROI.y;
    pstRpy->rotatedRectResult = fitCircleResult;

    cv::circle ( pstRpy->matResultImg, fitCircleResult.center, ToInt32 ( fitCircleResult.size.width / 2 + 0.5 ), _constBlueScalar, 2 );
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::lrnContour(const PR_LRN_CONTOUR_CMD *const pstCmd, PR_LRN_CONTOUR_RPY *const pstRpy, bool bReplay /*= false*/ ) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
   
    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The input ROI rect (%d, %d, %d, %d) is invalid",
            pstCmd->rectROI.x, pstCmd->rectROI.y, pstCmd->rectROI.width, pstCmd->rectROI.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseLrnContour);

    cv::Mat matROI ( pstCmd->matInputImg, pstCmd->rectROI );
    cv::Mat matGray, matBlur, matThreshold;
    if ( matROI.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();
    cv::GaussianBlur ( matGray, matBlur, cv::Size(5, 5), 2, 2 );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Blur image", matBlur);

    pstRpy->nThreshold = pstCmd->nThreshold;
    if ( pstCmd->bAutoThreshold )
        pstRpy->nThreshold = _autoThreshold ( matBlur );
    cv::threshold ( matBlur, matThreshold, pstRpy->nThreshold, PR_MAX_GRAY_LEVEL, cv::ThresholdTypes::THRESH_BINARY );
    cv::Mat matContour = matThreshold.clone();

    VectorOfVectorOfPoint vecContours;
    cv::findContours ( matThreshold, vecContours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );

    cv::Mat matDraw = matROI.clone();
    // Filter contours
    int index = 0;
    for ( auto contour : vecContours ) {
        auto area = cv::contourArea ( contour );
        if ( area > 1000 ) {
            pstRpy->vecContours.push_back ( contour );
            if (Config::GetInstance ()->getDebugMode () == PR_DEBUG_MODE::SHOW_IMAGE) {
                cv::RNG rng ( 12345 );
                cv::Scalar color = cv::Scalar ( rng.uniform ( 0, 255 ), rng.uniform ( 0, 255 ), rng.uniform ( 0, 255 ) );
                cv::drawContours ( matDraw, vecContours, index, color );
            }
        }
        ++ index;
    }
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Contours", matDraw );

    if ( pstRpy->vecContours.empty() ) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CONTOUR;
        return pstRpy->enStatus;
    }

    if ( pstCmd->matInputImg.channels() > 1)
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor ( pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR );

    _writeContourRecord ( pstRpy, matBlur, matContour, pstRpy->vecContours );

    for ( size_t index = 0; index < pstRpy->vecContours.size(); ++ index ) {
        for ( auto &point : pstRpy->vecContours[index] ) {
            point.x += pstCmd->rectROI.x;
            point.y += pstCmd->rectROI.y;
        }
    }
    cv::polylines( pstRpy->matResultImg, pstRpy->vecContours, true, _constBlueScalar, 2 );

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_writeContourRecord(PR_LRN_CONTOUR_RPY *const pstRpy, const cv::Mat &matTmpl, const cv::Mat &matContour, const VectorOfVectorOfPoint &vecContours) {
    ContourRecordPtr ptrRecord = std::make_shared<ContourRecord>( PR_RECORD_TYPE::CONTOUR );
    ptrRecord->setThreshold ( pstRpy->nThreshold );
    ptrRecord->setTmpl ( matTmpl );
    ptrRecord->setContourMat ( matContour );
    ptrRecord->setContour ( vecContours );
    return RecordManager::getInstance()->add( ptrRecord, pstRpy->nRecordId );
}

/*static*/ VisionStatus VisionAlgorithm::inspContour ( const PR_INSP_CONTOUR_CMD *const pstCmd, PR_INSP_CONTOUR_RPY *const pstRpy, bool bReplay /*= false*/ ) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
   
    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The input ROI rect (%d, %d, %d, %d) is invalid.",
            pstCmd->rectROI.x, pstCmd->rectROI.y, pstCmd->rectROI.width, pstCmd->rectROI.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if ( pstCmd->nRecordId <= 0 ) {
        WriteLog("The input record id is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }
    
    ContourRecordPtr ptrRecord = std::static_pointer_cast<ContourRecord> ( RecordManager::getInstance()->get ( pstCmd->nRecordId ) );
    if ( nullptr == ptrRecord ) {
        _snprintf ( charrMsg, sizeof (charrMsg), "Failed to get record ID %d in system.", pstCmd->nRecordId );
        WriteLog ( charrMsg );
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspContour);

    cv::Mat matROI ( pstCmd->matInputImg, pstCmd->rectROI );
    cv::Mat matGray, matBlur, matThreshold;
    if ( matROI.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();
    cv::GaussianBlur ( matGray, matBlur, cv::Size(5, 5), 2, 2 );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Blur image", matBlur);    

    cv::Point2f ptResult;
    float fRotation = 0.f, fCorrelation = 0.f;
    cv::Mat matTmpl = ptrRecord->getTmpl();
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage ( "Template image", matTmpl );
    pstRpy->enStatus = _matchTemplate ( matBlur, matTmpl, PR_OBJECT_MOTION::TRANSLATION, ptResult, fRotation, fCorrelation );
    if ( pstRpy->enStatus != VisionStatus::OK ) {
        WriteLog ( "Template match fail." );
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = VisionStatus::OK;

    cv::Mat matCompareROI = cv::Mat ( matBlur, cv::Rect ( ToInt32 ( ptResult.x - matTmpl.cols / 2.f ), ToInt32 ( ptResult.y - matTmpl.rows / 2.f ), matTmpl.cols, matTmpl.rows ) );
    cv::Mat matDiff_1 = matTmpl - matCompareROI;
    cv::Mat matDiff_2 = matCompareROI - matTmpl;
    cv::Mat matDiff = matDiff_1 + matDiff_2;
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage ( "Original Diff Image", matDiff );

    cv::Mat matContourMask = _generateContourMask ( matTmpl.size(), ptrRecord->getContour(), pstCmd->fInnerMaskDepth, pstCmd->fOuterMaskDepth );    
    matDiff = matDiff > pstCmd->nDefectThreshold;
    matDiff.setTo ( cv::Scalar(0), matContourMask );
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage ( "Defect image", matDiff );
    // Filter contours
    VectorOfVectorOfPoint vecContours, vecDefectContour;
    cv::findContours ( matDiff, vecContours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );
    const float DEFECT_ON_CONTOUR_RATIO = 1.2f;
    int index = 0;
    for ( auto contour : vecContours ) {
        auto area = cv::contourArea ( contour );
        if ( area > pstCmd->fMinDefectArea ) {
            cv::Moments moment = cv::moments( contour );
            cv::Point2f ptCenter;
            ptCenter.x = static_cast< float >(moment.m10 / moment.m00);
            ptCenter.y = static_cast< float >(moment.m01 / moment.m00);

            auto contourTarget = _findNearestContour ( ptCenter, ptrRecord->getContour() );

            float fInnerFarthestDist, fInnerNearestDist, fOuterFarthestDist, fOuterNearestDist;
            _getContourToContourDistance ( contour, contourTarget, fInnerFarthestDist, fInnerNearestDist, fOuterFarthestDist, fOuterNearestDist );

            auto minRect = cv::minAreaRect ( contour );
            auto length = minRect.size.width > minRect.size.height ?  minRect.size.width : minRect.size.height;

            //Defect inside contour.
            if ( fInnerFarthestDist > 0 && fabs ( fInnerNearestDist ) < pstCmd->fInnerMaskDepth * DEFECT_ON_CONTOUR_RATIO && length > pstCmd->fDefectInnerLengthTol ) {
                vecDefectContour.push_back ( contour );
            }else 
            if ( fOuterFarthestDist < 0 && fabs ( fOuterNearestDist ) < pstCmd->fOuterMaskDepth * DEFECT_ON_CONTOUR_RATIO && length > pstCmd->fDefectOuterLengthTol ) {
                vecDefectContour.push_back ( contour );
            }
        }
        ++ index;
    }

    if ( pstCmd->matInputImg.channels() > 1 )
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor ( pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR );

    if ( ! vecDefectContour.empty() ) {
        for ( auto &contour : vecDefectContour ) {
            for ( auto &point : contour ) {
                point.x += pstCmd->rectROI.x;
                point.y += pstCmd->rectROI.y;
            }
        }
        cv::polylines ( pstRpy->matResultImg, vecDefectContour, true, _constRedScalar, 2 );
        pstRpy->vecDefectContour = vecDefectContour;
        pstRpy->enStatus = VisionStatus::CONTOUR_DEFECT_REJECT;
    }

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VectorOfPoint VisionAlgorithm::_findNearestContour(const cv::Point2f &ptInput, const VectorOfVectorOfPoint &vecContours) {
    float fNearestDist = std::numeric_limits<float>::max();
    VectorOfPoint contourResult;
    cv::Point ptNearestPoint;
    for ( const auto &contour : vecContours ) {
        float fDist = CalcUtils::calcPointToContourDist ( ptInput, contour, ptNearestPoint );
        if ( fDist < fNearestDist ) {
            contourResult = contour;
            fNearestDist = fDist;
        }
    }
    return contourResult;
}

/*static*/ void VisionAlgorithm::_getContourToContourDistance(const VectorOfPoint &contourInput,
                                                              const VectorOfPoint &contourTarget,
                                                              float               &fInnerFarthestDist,
                                                              float               &fInnerNearestDist,
                                                              float               &fOuterFarthestDist,
                                                              float               &fOuterNearestDist) {
    fOuterNearestDist = fInnerFarthestDist = - std::numeric_limits<float>::max();
    fInnerNearestDist = fOuterFarthestDist =   std::numeric_limits<float>::max();
    for ( const auto &point : contourInput ) {
        auto distance = ToFloat ( cv::pointPolygonTest ( contourTarget, point, true ) );
        // distance < 0 means the point is outside of the target contour.
        if ( distance < 0 ) {
            if ( distance < fOuterFarthestDist )
                fOuterFarthestDist = distance;
            if ( distance > fOuterNearestDist )
                fOuterNearestDist = distance;
        }else if ( distance > 0 ) { // distance > 0 means the point is inside of the target contour.
            if ( distance > fInnerFarthestDist )
                fInnerFarthestDist = distance;
            if ( distance < fInnerNearestDist )
                fInnerNearestDist = distance;
        }
    }
}

/*static*/ cv::Mat VisionAlgorithm::_generateContourMask ( const cv::Size &size, const VectorOfVectorOfPoint &vecContours, float fInnerDepth, float fOuterDepth ) {
    cv::Mat matMask = cv::Mat::zeros ( size, CV_8UC1 );
    if ( fabs ( fInnerDepth - fOuterDepth ) < 1 ) {
        for ( int index = 0; index < (int) ( vecContours.size() ); ++ index )
            cv::drawContours ( matMask, vecContours, index, cv::Scalar(PR_MAX_GRAY_LEVEL), ToInt32 ( 2.f * fOuterDepth ) );
        return matMask;
    }

    cv::Mat matOuterMask = matMask.clone();
    cv::Mat matInnerMask = matMask.clone();

    {// Generate outer mask
        for ( int index = 0; index < (int) ( vecContours.size() ); ++ index )
            cv::drawContours ( matOuterMask, vecContours, index, cv::Scalar(PR_MAX_GRAY_LEVEL), ToInt32 ( 2.f * fOuterDepth ) );
        cv::fillPoly ( matOuterMask, vecContours, cv::Scalar::all(0) );
    }

    {// Generate inner mask
        cv::Mat matTmpOuterMask = matMask.clone();
        for ( int index = 0; index < (int) ( vecContours.size() ); ++ index ) {
            cv::drawContours ( matTmpOuterMask, vecContours, index, cv::Scalar(PR_MAX_GRAY_LEVEL), ToInt32 ( 2.f * fInnerDepth ) );
            cv::drawContours ( matInnerMask, vecContours, index, cv::Scalar(PR_MAX_GRAY_LEVEL), ToInt32 ( 2.f * fInnerDepth ) );
        }
        cv::fillPoly ( matTmpOuterMask, vecContours, cv::Scalar::all(0) );
        matInnerMask -= matTmpOuterMask;
    }
    matMask = matInnerMask + matOuterMask;

    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
        cv::Mat matDisplay = cv::Mat::zeros ( size, CV_8UC3 );
        matDisplay.setTo ( _constGreenScalar, matOuterMask );
        matDisplay.setTo ( _constBlueScalar,  matInnerMask );
        for ( int index = 0; index < (int)(vecContours.size ()); ++index )
            cv::drawContours ( matDisplay, vecContours, index, _constRedScalar, 1 );
        showImage ( "Result Mask", matDisplay );
    }
    return matMask;
}

/*static*/ VisionStatus VisionAlgorithm::inspHole(const PR_INSP_HOLE_CMD *const pstCmd, PR_INSP_HOLE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char charrMsg[1000];
    if ( pstCmd->matInputImg.empty() ) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
   
    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInputImg.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInputImg.rows ) {
        _snprintf( charrMsg, sizeof( charrMsg ), "The input ROI rect (%d, %d, %d, %d) is invalid.",
            pstCmd->rectROI.x, pstCmd->rectROI.y, pstCmd->rectROI.width, pstCmd->rectROI.height );
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if ( ! pstCmd->matMask.empty() ) {
        if ( pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols ) {
            _snprintf(charrMsg, sizeof(charrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d ).",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(charrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if ( pstCmd->matMask.channels() != 1 )  {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI), matBlur, matSegmentResult;
    cv::GaussianBlur ( matROI, matBlur, cv::Size(5, 5), 2, 2 );

    if ( pstCmd->enSegmentMethod == PR_IMG_SEGMENT_METHOD::GRAY_SCALE_RANGE ) {
        pstRpy->enStatus = _segmentImgByGrayScaleRange ( matBlur, pstCmd->stGrayScaleRange, matSegmentResult );        
    }else if ( pstCmd->enSegmentMethod == PR_IMG_SEGMENT_METHOD::COLOR_RANGE ) {
        pstRpy->enStatus = _setmentImgByColorRange ( matBlur, pstCmd->stColorRange, matSegmentResult );
    }else {
        WriteLog ( "Unsupported image segment method." );
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( pstRpy->enStatus != VisionStatus::OK )
        return pstRpy->enStatus;

    cv::Mat matMaskROI;
    if ( ! pstCmd->matMask.empty() )
        matMaskROI = cv::Mat( pstCmd->matMask, pstCmd->rectROI );

    //Prepared the result image, should be skiped if in auto mode.
    cv::Mat matSegmentInWholeImg = cv::Mat::zeros ( pstCmd->matInputImg.size(), CV_8UC1);
    cv::Mat matCopyROI(matSegmentInWholeImg, pstCmd->rectROI);
    matSegmentResult.copyTo ( matCopyROI );
    if ( pstCmd->matInputImg.channels() > 1)
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor ( pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR );
    pstRpy->matResultImg.setTo ( _constYellowScalar, matSegmentInWholeImg );

    if ( pstCmd->enInspMode == PR_INSP_HOLE_MODE::RATIO )
        _inspHoleByRatioMode ( matSegmentResult, matMaskROI, pstCmd->stRatioModeCriteria, pstRpy );
    else {
        _inspHoleByBlobMode ( matSegmentResult, matMaskROI, pstCmd->stBlobModeCriteria, pstRpy );
        for ( auto &keypoint : pstRpy->stBlobModeResult.vecBlobs ) {
            keypoint.pt.x += pstCmd->rectROI.x;
            keypoint.pt.y += pstCmd->rectROI.y;
        }
        cv::drawKeypoints( pstRpy->matResultImg,  pstRpy->stBlobModeResult.vecBlobs, pstRpy->matResultImg, _constRedScalar, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    }
    
    

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_segmentImgByGrayScaleRange(const cv::Mat &matInput, const PR_INSP_HOLE_CMD::GRAY_SCALE_RANGE &stGrayScaleRange, cv::Mat &matResult) {
    if (stGrayScaleRange.nStart < 0 ||
        stGrayScaleRange.nStart >= stGrayScaleRange.nEnd ||
        stGrayScaleRange.nEnd > PR_MAX_GRAY_LEVEL ) {
        char charrMsg[1000];
        _snprintf ( charrMsg, sizeof(charrMsg), "The gray scale range (%d, %d) is invalid.",
            stGrayScaleRange.nStart, stGrayScaleRange.nEnd );
        WriteLog ( charrMsg );
        return VisionStatus::INVALID_PARAM;
    }
    cv::Mat matGray;
    if ( matInput.channels() > 1 )
        cv::cvtColor ( matInput, matGray, CV_BGR2GRAY );
    else
        matGray = matInput;
    cv::Mat matRange1, matRange2;
    matRange1 = matGray >= stGrayScaleRange.nStart;
    matRange2 = matGray <= stGrayScaleRange.nEnd;
    matResult = matRange1 & matRange2;
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_setmentImgByColorRange(const cv::Mat &matInput, const PR_INSP_HOLE_CMD::COLOR_RANGE &stColorRange, cv::Mat &matResult) {
    if (stColorRange.nStartB < 0 || stColorRange.nStartB >= stColorRange.nEndB || stColorRange.nEndB > PR_MAX_GRAY_LEVEL ||
        stColorRange.nStartG < 0 || stColorRange.nStartG >= stColorRange.nEndG || stColorRange.nEndG > PR_MAX_GRAY_LEVEL ||
        stColorRange.nStartR < 0 || stColorRange.nStartR >= stColorRange.nEndR || stColorRange.nEndR > PR_MAX_GRAY_LEVEL ) {
        char charrMsg[1000];
        _snprintf ( charrMsg, sizeof(charrMsg), "The color scale range (%d, %d, %d, %d, %d, %d) is invalid.",
            stColorRange.nStartB, stColorRange.nEndB, stColorRange.nStartG, stColorRange.nEndG, stColorRange.nStartR, stColorRange.nEndR );
        WriteLog ( charrMsg );
        return VisionStatus::INVALID_PARAM;
    }
    if ( matInput.channels() < 3 ) {
        WriteLog ( "Input image must be color for color range segment." );
        return VisionStatus::INVALID_PARAM;
    }
    std::vector<cv::Mat> vecChannels;
    cv::split( matInput, vecChannels );
    cv::Mat matRange1, matRange2, matRange3, matRange4, matRange5, matRange6;
    matRange1 = vecChannels[BGR_CHANNEL::BLUE] >= stColorRange.nStartB;
    matRange2 = vecChannels[BGR_CHANNEL::BLUE] <= stColorRange.nEndB;
    matRange3 = vecChannels[BGR_CHANNEL::GREEN] >= stColorRange.nStartG;
    matRange4 = vecChannels[BGR_CHANNEL::GREEN] <= stColorRange.nEndG;
    matRange5 = vecChannels[BGR_CHANNEL::RED] >= stColorRange.nStartR;
    matRange6 = vecChannels[BGR_CHANNEL::RED] <= stColorRange.nEndR;
    matResult = matRange1 & matRange2 & matRange3 & matRange4 & matRange5 & matRange6;
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_inspHoleByRatioMode(const cv::Mat &matInput, const cv::Mat &matMask, const PR_INSP_HOLE_CMD::RATIO_MODE_CRITERIA &stCriteria, PR_INSP_HOLE_RPY *const pstRpy) {
    cv::Mat matInputLocal = matInput.clone();
    if ( matMask.empty() )
        matInputLocal.setTo ( cv::Scalar(0), cv::Scalar(PR_MAX_GRAY_LEVEL) - matMask );
    
    auto fSelectedCount = ToFloat ( cv::countNonZero ( matInputLocal ) );
    pstRpy->stRatioModeResult.fRatio = fSelectedCount / ( matInput.rows * matInput.cols );
    if ( pstRpy->stRatioModeResult.fRatio < stCriteria.fMinRatio ) {
        pstRpy->enStatus = VisionStatus::RATIO_UNDER_LIMIT;
        return pstRpy->enStatus;
    }else if ( pstRpy->stRatioModeResult.fRatio > stCriteria.fMaxRatio ) {
        pstRpy->enStatus = VisionStatus::RATIO_OVER_LIMIT;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspHoleByBlobMode(const cv::Mat &matInput, const cv::Mat &matMask, const PR_INSP_HOLE_CMD::BLOB_MODE_CRITERIA &stCriteria, PR_INSP_HOLE_RPY *const pstRpy) {
    cv::SimpleBlobDetector::Params params;

    // Change thresholds
    params.blobColor = 0;

    params.minThreshold = 100;
    params.maxThreshold = PR_MAX_GRAY_LEVEL;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = stCriteria.fMinArea;
    params.maxArea = stCriteria.fMaxArea;

    params.filterByColor = false;
    params.filterByConvexity = false;

    if ( stCriteria.bEnableAdvancedCriteria ) {
        // Filter by Circularity
        params.filterByCircularity = true;
        params.minCircularity = stCriteria.stAdvancedCriteria.fMinCircularity;
        params.maxCircularity = stCriteria.stAdvancedCriteria.fMaxCircularity;

        // Filter by Inertia
        params.filterByInertia = true;
        params.maxInertiaRatio = stCriteria.stAdvancedCriteria.fMaxLengthWidthRatio;
        params.minInertiaRatio = stCriteria.stAdvancedCriteria.fMinLengthWidthRatio;
    }else {
        params.filterByCircularity = false;
        params.filterByInertia = false;
    }

    // Set up detector with params
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create ( params );

    std::vector<cv::Mat> vecInputMat;
    vecInputMat.push_back ( matInput );

    VectorOfVectorKeyPoint keyPoints;
    detector->detect ( vecInputMat, keyPoints, matMask );
    pstRpy->stBlobModeResult.vecBlobs = keyPoints[0];
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) {
        cv::Mat matDisplay;
        cv::cvtColor ( matInput, matDisplay, CV_GRAY2BGR );
        cv::drawKeypoints( matDisplay, keyPoints[0], matDisplay, _constRedScalar, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        showImage("Find blob result", matDisplay );
    }

    if ( keyPoints[0].size() < stCriteria.nMinBlobCount || keyPoints[0].size() > stCriteria.nMaxBlobCount ) {
        pstRpy->enStatus = VisionStatus::BLOB_COUNT_OUT_OF_RANGE;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
    
}

}
}