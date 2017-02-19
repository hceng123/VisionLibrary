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

using namespace cv::xfeatures2d;
using namespace std;
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

VisionAlgorithm::VisionAlgorithm()
{
}

/*static*/ unique_ptr<VisionAlgorithm> VisionAlgorithm::create()
{
    return make_unique<VisionAlgorithm>();
}

VisionStatus VisionAlgorithm::lrnTmpl(PR_LRN_TMPL_CMD * const pLrnTmplCmd, PR_LRN_TMPL_RPY *pLrnTmplRpy, bool bReplay)
{
    if ( NULL == pLrnTmplCmd || NULL == pLrnTmplRpy )
        return VisionStatus::INVALID_PARAM;

    if ( pLrnTmplCmd->mat.empty() )
        return VisionStatus::INVALID_PARAM;

    VisionStatus enReturnStatus = VisionStatus::OK;
    std::unique_ptr<LogCaseLrnTmpl> pLogCase;
    if ( ! bReplay )    {
        pLogCase = std::make_unique<LogCaseLrnTmpl>( Config::GetInstance()->getLogCaseDir() );
        pLogCase->WriteCmd ( pLrnTmplCmd );
    }

    cv::Ptr<cv::Feature2D> detector;
    if ( PR_ALIGN_ALGORITHM::SIFT == pLrnTmplCmd->enAlgorithm)
        detector = SIFT::create (_constMinHessian, _constOctave, _constOctaveLayer );
    else
        detector = SURF::create (_constMinHessian, _constOctave, _constOctaveLayer );
    cv::Mat matROI( pLrnTmplCmd->mat, pLrnTmplCmd->rectLrn );
    detector->detectAndCompute ( matROI, pLrnTmplCmd->mask, pLrnTmplRpy->vecKeyPoint, pLrnTmplRpy->matDescritor );

    if (pLrnTmplRpy->vecKeyPoint.size() > _constLeastFeatures)    {
        enReturnStatus = VisionStatus::OK;
        pLrnTmplRpy->ptCenter.x = pLrnTmplCmd->rectLrn.x + pLrnTmplCmd->rectLrn.width / 2.0f;
        pLrnTmplRpy->ptCenter.y = pLrnTmplCmd->rectLrn.y + pLrnTmplCmd->rectLrn.height / 2.0f;
        matROI.copyTo(pLrnTmplRpy->matTmpl);
        _writeLrnTmplRecord(pLrnTmplRpy);
    }        
    else
    {
        enReturnStatus = VisionStatus::LEARN_FAIL;
    }    

    if ( ! bReplay )
        pLogCase->WriteRpy(pLrnTmplRpy);

    pLrnTmplRpy->nStatus = ToInt32( enReturnStatus );
    return enReturnStatus;
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
    
    detector->detectAndCompute ( matSrchROI, mask, vecKeypointScene, matDescriptorScene );
    TimeLog::GetInstance()->addTimeLog("detectAndCompute");

    VectorOfDMatch good_matches;

    cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::KDTreeIndexParams;
    cv::Ptr<cv::flann::SearchParams> searchParams = new cv::flann::SearchParams;

    cv::FlannBasedMatcher matcher;

    VectorOfVectorOfDMatch vecVecMatches;
    vecVecMatches.reserve(2000);
	
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

    if ( good_matches.size() <= 0 ) {
        pFindObjRpy->nStatus = ToInt32 ( VisionStatus::SRCH_TMPL_FAIL );
        return VisionStatus::SRCH_TMPL_FAIL;
    }
    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); ++ i )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( pRecord->getModelKeyPoint()[good_matches[i].queryIdx].pt);
        scene.push_back( vecKeypointScene[good_matches[i].trainIdx].pt);
    }    
    
    pFindObjRpy->matHomography = cv::findHomography ( obj, scene, CV_RANSAC );
    TimeLog::GetInstance()->addTimeLog("cv::findHomography");

	pFindObjRpy->matHomography.at<double>(0, 2) += pFindObjCmd->rectSrchWindow.x;
	pFindObjRpy->matHomography.at<double>(1, 2) += pFindObjCmd->rectSrchWindow.y;

    cv::Mat matOriginalPos ( 3, 1, CV_64F );
    matOriginalPos.at<double>(0, 0) = pFindObjCmd->rectLrn.width  / 2.f;
    matOriginalPos.at<double>(1, 0) = pFindObjCmd->rectLrn.height / 2.f;
    matOriginalPos.at<double>(2, 0) = 1.0;

    cv::Mat destPos ( 3, 1, CV_64F );

    destPos = pFindObjRpy->matHomography * matOriginalPos;
    pFindObjRpy->ptObjPos.x = (float) destPos.at<double>(0, 0);
    pFindObjRpy->ptObjPos.y = (float) destPos.at<double>(1, 0);
    pFindObjRpy->szOffset.width = pFindObjRpy->ptObjPos.x - pFindObjCmd->ptExpectedPos.x;
    pFindObjRpy->szOffset.height = pFindObjRpy->ptObjPos.y - pFindObjCmd->ptExpectedPos.y;
	float fRotationValue = (float)(pFindObjRpy->matHomography.at<double>(1, 0) - pFindObjRpy->matHomography.at<double>(0, 1)) / 2.0f;
	pFindObjRpy->fRotation = (float) CalcUtils::radian2Degree ( asin(fRotationValue) );
    
    MARK_FUNCTION_END_TIME;
    pFindObjRpy->nStatus = ToInt32 ( VisionStatus::OK );
    return VisionStatus::OK;
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
    void _expSmooth(const cv::Mat &matInput, cv::Mat &matOutput, float alpha)
    {
        matInput.copyTo(matOutput);
        for (int col = 1; col < matInput.cols; ++col)
        {
            matOutput.at<T>(0, col) = alpha * matInput.at<T>(0, col) + (1.f - alpha) * matOutput.at<T>(0, col - 1);
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
        float       fArea;
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
std::vector<Int16> VisionAlgorithm::_autoMultiLevelThreshold(const cv::Mat &mat, const int N)
{
    assert(N >= 1 && N <= 4);

    const int sbins = 128;
    const float MIN_P = 0.001f;
    std::vector<Int16> vecThreshold;
    int channels[] = { 0 };
    int histSize[] = { sbins };

    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 255 };
    const float* ranges[] = { sranges };

    cv::MatND hist;
    cv::calcHist(&mat, 1, channels, cv::Mat(), // do not use mask
        hist, 1, histSize, ranges,
        true, // the histogram is uniform
        false);

    std::vector<float> P, S;
    P.push_back(0.f);
    S.push_back(0.f);
    float fTotalPixel = (float)mat.rows * mat.cols;
    for (int i = 0; i < sbins; ++i)   {
        float fPi = hist.at<float>(i, 0) / fTotalPixel;
        P.push_back(fPi + P[i]);
        S.push_back(i * fPi + S[i]);
    }

    std::vector<std::vector<float>> vecVecOmega(sbins + 1, vector<float>(sbins + 1, 0));
    std::vector<std::vector<float>> vecVecMu(sbins + 1, vector<float>(sbins + 1, 0));
    std::vector<std::vector<float>> vecVecEta(sbins + 1, vector<float>(sbins + 1, -1.f));
    for (int i = 0; i <= sbins; ++i)
    for (int j = i; j <= sbins; ++j)
    {
        vecVecOmega[i][j] = P[j] - P[i];
        vecVecMu[i][j]    = S[j] - S[i];
    }

    const int START = 1;
    if (N == 1)
    {
        float fMaxSigma = 0;
        int nResultT1 = 0;
        for (int T = START; T < sbins; ++ T)
        {
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
    }
    else if (N == 2)
    {
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
    }
    else if (N == 3)
    {
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

            if (fMaxSigma < fSigma)    {
                nResultT1 = T1;
                nResultT2 = T2;
                nResultT3 = T3;
                fMaxSigma = fSigma;
            }
        }
        vecThreshold.push_back(nResultT1 * PR_MAX_GRAY_LEVEL / sbins);
        vecThreshold.push_back(nResultT2 * PR_MAX_GRAY_LEVEL / sbins);
        vecThreshold.push_back(nResultT3 * PR_MAX_GRAY_LEVEL / sbins);
    }
    else if (N == 4)
    {
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

VisionStatus VisionAlgorithm::autoThreshold(PR_AUTO_THRESHOLD_CMD *pstCmd, PR_AUTO_THRESHOLD_RPY *pstRpy, bool bReplay)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInput.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInput.rows )    {
        WriteLog("The Fitting range is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseAutoThreshold);

    cv::Mat matROI(pstCmd->matInput, pstCmd->rectROI);
    pstRpy->vecThreshold = _autoMultiLevelThreshold ( matROI, pstCmd->nThresholdNum);

    pstRpy->enStatus = VisionStatus::OK;
    
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

int VisionAlgorithm::_autoThreshold(const cv::Mat &mat)
{
    auto vecThreshold = _autoMultiLevelThreshold(mat, 1);
    return vecThreshold[0];
}

VisionStatus VisionAlgorithm::_findDeviceElectrode(const cv::Mat &matDeviceROI, VectorOfPoint &vecElectrodePos)
{
    cv::Mat matCannyOutput;
    vector<vector<cv::Point> > vecContours, vecContourAfterFilter;
    vector<cv::Vec4i> hierarchy;
    vecElectrodePos.clear();

    // Detect edges using canny
    cv::Canny(matDeviceROI, matCannyOutput, 100, 255, 3);
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Canny Result", matCannyOutput );

    // Find contours
    cv::findContours(matCannyOutput, vecContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // Filter contours
    cv::Mat drawing = cv::Mat::zeros(matCannyOutput.size(), CV_8UC3);
    std::vector<AOI_COUNTOUR> vecAoiContour;
    int index = 0;
    for ( auto contour : vecContours )
    {
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )   {
            cv::RNG rng(12345);
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            drawContours ( drawing, vecContours, index ++ , color, 2, 8, hierarchy, 0, cv::Point() );
        }
        vector<cv::Point> approx;
        approxPolyDP(contour, approx, 3, true);
        auto area = cv::contourArea ( approx );
        if (area > 100){
            AOI_COUNTOUR stAoiContour;
            cv::Moments moment = cv::moments( contour );
            stAoiContour.ptCtr.x = static_cast< float >(moment.m10 / moment.m00);
            stAoiContour.ptCtr.y = static_cast< float >(moment.m01 / moment.m00);
            if (!_isSimilarContourExist(vecAoiContour, stAoiContour))   {
                vecAoiContour.push_back(stAoiContour);
                vecElectrodePos.push_back(stAoiContour.ptCtr);
            }
        }
    }
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE ) 
        showImage("Contours", drawing);

    if ( vecElectrodePos.size() != DEVICE_ELECTRODE_COUNT )   return VisionStatus::FIND_ELECTRODE_FAIL;
    
    return VisionStatus::OK;
}

//The alogorithm come from paper "A Visual Inspection System for Surface Mounted Components Based on Color Features"
VisionStatus VisionAlgorithm::_findDeviceEdge(const cv::Mat &matDeviceROI, const cv::Size2f &size, cv::Rect &rectDeviceResult)
{
    cv::Mat matGray, matThreshold, matHorizontalProjection, matVerticalProjection;
    Int32 nLeftEdge = 0, nRightEdge = 0, nTopEdge = 0, nBottomEdge = 0;

    _calculateHorizontalProjection(matDeviceROI, matHorizontalProjection);
    _expSmooth<float>(matHorizontalProjection, matHorizontalProjection, _constExpSmoothRatio);
    _findEdge(matHorizontalProjection, static_cast< Int32 >(size.width), nLeftEdge);
    _findEdgeReverse(matHorizontalProjection, static_cast< Int32 >(size.width), nRightEdge);
    if ( ( nRightEdge - nLeftEdge ) < size.width / 2)
        return VisionStatus::FIND_EDGE_FAIL;

    _calculateVerticalProjection(matThreshold, matVerticalProjection);
    _expSmooth<float>(matVerticalProjection, matVerticalProjection, _constExpSmoothRatio);
    _findEdge(matHorizontalProjection, static_cast< Int32 >(size.height), nTopEdge);
    _findEdgeReverse(matHorizontalProjection, static_cast< Int32 >(size.height), nBottomEdge);
    if ( (nBottomEdge - nTopEdge ) < size.height / 2 )
        return VisionStatus::FIND_EDGE_FAIL;

    rectDeviceResult = cv::Rect ( cv::Point ( nLeftEdge, nTopEdge ), cv::Point ( nRightEdge, nBottomEdge ) );

    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::lrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy)
{
    char charrMsg[1000];
    if ( nullptr == pstLrnDeviceCmd || nullptr == pstLrnDeivceRpy )   {
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstLrnDeviceCmd = %d, pstLrnDeivceRpy = %d", (int)pstLrnDeviceCmd, (int)pstLrnDeivceRpy);
        WriteLog(charrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    if ( pstLrnDeviceCmd->matInput.empty() )   {
        WriteLog("Input image is empty");
        return VisionStatus::INVALID_PARAM;
    }

    cv::Rect rectDeviceROI = CalcUtils::scaleRect<float> ( pstLrnDeviceCmd->rectDevice, 1.5 );
    if ( rectDeviceROI.x < 0  ) rectDeviceROI.x = 0;
    if ( rectDeviceROI.y < 0  ) rectDeviceROI.y = 0;
    if ( ( rectDeviceROI.x + rectDeviceROI.width ) > pstLrnDeviceCmd->matInput.cols ) rectDeviceROI.width  = pstLrnDeviceCmd->matInput.cols - rectDeviceROI.x;
    if ( ( rectDeviceROI.y + rectDeviceROI.height) > pstLrnDeviceCmd->matInput.rows ) rectDeviceROI.height = pstLrnDeviceCmd->matInput.rows - rectDeviceROI.y;

    cv::Mat matDevice ( pstLrnDeviceCmd->matInput, rectDeviceROI );
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
    {
        pstLrnDeviceCmd->nElectrodeThreshold = _autoThreshold ( matBlur );
    }
    cv::threshold ( matBlur, matThreshold, pstLrnDeviceCmd->nElectrodeThreshold, 255, cv::THRESH_BINARY );
    

    VectorOfPoint vecPtElectrode;
    VisionStatus enStatus = _findDeviceElectrode ( matThreshold, vecPtElectrode );
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
    if ( pstInspDeviceCmd == nullptr || pstInspDeivceRpy == nullptr )   {
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstInspDeviceCmd = %d, pstInspDeivceRpy = %d", (int)pstInspDeviceCmd, (int)pstInspDeivceRpy);
        WriteLog(charrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    if ( pstInspDeviceCmd->matInput.empty() )   {
        WriteLog("Input image is empty");
        return VisionStatus::INVALID_PARAM;
    }

    pstInspDeivceRpy->nDeviceCount = pstInspDeviceCmd->nDeviceCount;
    for (Int32 i = 0; i < pstInspDeviceCmd->nDeviceCount; ++ i)   {
        if ( pstInspDeviceCmd->astDeviceInfo[i].stSize.area() < 1)   {
            _snprintf(charrMsg, sizeof(charrMsg), "Device[ %d ] size is invalid", i );
            WriteLog(charrMsg);
            return VisionStatus::INVALID_PARAM;
        }

        cv::Rect rectDeviceROI = pstInspDeviceCmd->astDeviceInfo[i].rectSrchWindow;
        if (rectDeviceROI.x < 0) rectDeviceROI.x = 0;
        if (rectDeviceROI.y < 0) rectDeviceROI.y = 0;
        if ((rectDeviceROI.x + rectDeviceROI.width)  > pstInspDeviceCmd->matInput.cols) rectDeviceROI.width  = pstInspDeviceCmd->matInput.cols - rectDeviceROI.x;
        if ((rectDeviceROI.y + rectDeviceROI.height) > pstInspDeviceCmd->matInput.rows) rectDeviceROI.height = pstInspDeviceCmd->matInput.rows - rectDeviceROI.y;

        cv::Mat matROI(pstInspDeviceCmd->matInput, rectDeviceROI );
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
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )   {
            cv::imshow("Threshold image", matThreshold );
            cv::waitKey(0);
        }

        VectorOfPoint vecPtElectrode;
        VisionStatus enStatus = _findDeviceElectrode ( matThreshold, vecPtElectrode );
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
	pAlignRpy->fRotation = (float) CalcUtils::radian2Degree ( asin ( fRotationValue ) );

    return 0;
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

	cv::Mat matRotation = getRotationMatrix2D ( pInspCmd->ptObjPos, -pInspCmd->fRotation, 1 );
	cv::Mat matRotatedTmpl;
	warpAffine( matTmplFullRange, matRotatedTmpl, matRotation, matTmplFullRange.size() );

	cv::Mat matRotatedMask;
	warpAffine( matMask, matRotatedMask, matRotation, matMask.size() );
	matRotatedMask =  cv::Scalar::all(255) - matRotatedMask;	

	pInspCmd->matInsp.setTo(cv::Scalar(0), matRotatedMask );	

	cv::Mat matCmpResult = pInspCmd->matInsp - matRotatedTmpl;
	cv::Mat matRevsCmdResult = matRotatedTmpl - pInspCmd->matInsp;

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)   {
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
				pInspRpy->astDefect[pInspRpy->n16NDefect].fLength = CalcUtils::distanceOf2Point<float>(it.pt1, it.pt2);
				++pInspRpy->n16NDefect;
			}
		}
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )   {
            cv::imshow("Detected Lines", color_dst);
            cv::waitKey(0);
        }
	}
    MARK_FUNCTION_END_TIME;
	return 0;
}

int VisionAlgorithm::_merge2Line(const PR_Line2f &line1, const PR_Line2f &line2, PR_Line2f &lineResult)
{
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
    if (fabs(fPerpendicularLineSlope1) < 0.01 || fabs(fPerpendicularLineSlope2) < 0.01 )
        fDistanceOfLine = fabs ( ( line1.pt1.x + line1.pt2.x ) / 2.f - ( line2.pt1.x + line2.pt2.x ) / 2.f );
    else if (fabs(fPerpendicularLineSlope1) > 100 || fabs(fPerpendicularLineSlope2) > 100 )
        fDistanceOfLine = fabs ( ( line1.pt1.y + line1.pt2.y ) / 2.f - ( line2.pt1.y + line2.pt2.y ) / 2.f );
    else{
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
    lineResult.pt1 = vecPoint[nMinIndex];
    lineResult.pt2 = vecPoint[nMaxIndex];
	//cv::Point2f ptMidOfResult;
	//ptMidOfResult.x = ( vecPoint[nMinIndex].x + vecPoint[nMaxIndex].x ) / 2.f;
	//ptMidOfResult.y = ( vecPoint[nMinIndex].y + vecPoint[nMaxIndex].y ) / 2.f;
	//float fLineLength = CalcUtils::distanceOf2Point<float>( vecPoint[nMinIndex], vecPoint[nMaxIndex] );
	//float fAverageSlope = ( fLineSlope1 + fLineSlope2 ) / 2;
	//float fHypotenuse = sqrt( 1 + fAverageSlope * fAverageSlope );
	//float fSin = fAverageSlope / fHypotenuse;
	//float fCos = 1.f / fHypotenuse;

	//lineResult.pt1.x = ptMidOfResult.x + fLineLength * fCos / 2.f;
	//lineResult.pt1.y = ptMidOfResult.y + fLineLength * fSin / 2.f;
	//lineResult.pt2.x = ptMidOfResult.x - fLineLength * fCos / 2.f;
	//lineResult.pt2.y = ptMidOfResult.y - fLineLength * fSin / 2.f;
	return static_cast<int>(VisionStatus::OK);
}

int VisionAlgorithm::_mergeLines(const vector<PR_Line2f> &vecLines, vector<PR_Line2f> &vecResultLines)
{
	if ( vecLines.size() < 2 )  {
        vecResultLines = vecLines;
        return 0;
    }		

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

VisionStatus VisionAlgorithm::_writeLrnTmplRecord(PR_LRN_TMPL_RPY *pLrnTmplRpy)
{
    TmplRecord *pRecord = new TmplRecord(PR_RECORD_TYPE::ALIGNMENT);
    pRecord->setModelKeyPoint(pLrnTmplRpy->vecKeyPoint);
    pRecord->setModelDescriptor(pLrnTmplRpy->matDescritor);
    RecordManager::getInstance()->add(pRecord, pLrnTmplRpy->nRecordID);
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::_writeDeviceRecord(PR_LRN_DEVICE_RPY *pLrnDeviceRpy)
{
    DeviceRecord *pRecord = new DeviceRecord ( PR_RECORD_TYPE::DEVICE );
    pRecord->setElectrodeThreshold ( pLrnDeviceRpy->nElectrodeThreshold );
    pRecord->setSize ( pLrnDeviceRpy->sizeDevice );
    RecordManager::getInstance()->add(pRecord, pLrnDeviceRpy->nRecordID);
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

/*static*/ VisionStatus VisionAlgorithm::runLogCase(const std::string &strPath)
{
    if ( strPath.length() < 2 )
        return VisionStatus::INVALID_PARAM;

    if (  ! bfs::exists ( strPath ) )
        return VisionStatus::PATH_NOT_EXIST;

    VisionStatus enStatus = VisionStatus::OK;
    auto strLocalPath = strPath;
    char chFolderToken = '\\';
    if ( strPath.find('/') != string::npos )
        chFolderToken = '/';

    if ( strPath[ strPath.length() - 1 ] != chFolderToken )
        strLocalPath.append ( std::string { chFolderToken } );

    auto pos = strLocalPath.rfind ( chFolderToken, strLocalPath.length() - 2 );
    if ( String::npos == pos )
        return VisionStatus::INVALID_PARAM;

    auto folderName = strLocalPath.substr ( pos + 1 );
    pos = folderName.find('_');
    auto folderPrefix = folderName.substr(0, pos);

    std::unique_ptr<LogCase> pLogCase = nullptr;
    if ( LogCaseLrnTmpl::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique <LogCaseLrnTmpl>( strLocalPath, true );
    else if (LogCaseFitCircle::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique <LogCaseFitCircle>( strLocalPath, true );
    else if (LogCaseFitLine::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseFitLine>( strLocalPath, true );
    else if (LogCaseFitParallelLine::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseFitParallelLine>( strLocalPath, true );
    else if (LogCaseFitRect::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseFitRect>( strLocalPath, true );
    else if (LogCaseSrchFiducial::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseSrchFiducial>( strLocalPath, true );
    else if (LogCaseOcr::StaticGetFolderPrefix() == folderPrefix)
        pLogCase = std::make_unique<LogCaseOcr>( strLocalPath, true );
    else if (LogCaseRemoveCC::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseRemoveCC>( strLocalPath, true );
    else if (LogCaseDetectEdge::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseDetectEdge>( strLocalPath, true );
    else if (LogCaseCircleRoundness::StaticGetFolderPrefix() == folderPrefix )
        pLogCase = std::make_unique<LogCaseCircleRoundness>( strLocalPath, true );

    if ( nullptr != pLogCase )
        enStatus = pLogCase->RunLogCase();
    else
        enStatus = VisionStatus::INVALID_LOGCASE;
    return enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_refineSrchTemplate(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult)
{
    cv::Mat matWarp = cv::Mat::eye(2, 3, CV_32FC1);
    matWarp.at<float>(0,2) = ptResult.x;
    matWarp.at<float>(1,2) = ptResult.y;
    int number_of_iterations = 30;
    double termination_eps = 0.001;

    cv::findTransformECC ( matTmpl, mat, matWarp, cv::MOTION_TRANSLATION, cv::TermCriteria ( cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
        number_of_iterations, termination_eps) );
    ptResult.x = matWarp.at<float>(0, 2);
    ptResult.y = matWarp.at<float>(1, 2);

    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::matchTemplate(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult)
{
    cv::Mat img_display, matResult;
    const int match_method = CV_TM_SQDIFF;

    mat.copyTo(img_display);

    /// Create the result matrix
    int result_cols = mat.cols - matTmpl.cols + 1;
    int result_rows = mat.rows - matTmpl.rows + 1;

    matResult.create(result_rows, result_cols, CV_32FC1);

    /// Do the Matching and Normalize
    cv::matchTemplate(mat, matTmpl, matResult, match_method);
    cv::normalize ( matResult, matResult, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal;
    cv::Point minLoc, maxLoc, matchLoc;

    cv::minMaxLoc(matResult, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
        matchLoc = minLoc;
    else
        matchLoc = maxLoc;

    ptResult.x = (float)matchLoc.x;
    ptResult.y = (float)matchLoc.y;
    _refineSrchTemplate(mat, matTmpl, ptResult);

    ptResult.x += (float)( matTmpl.cols / 2 + 0.5 );
    ptResult.y += (float)( matTmpl.rows / 2 + 0.5 );
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::srchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy, bool bReplay)
{
    char charrMsg[1000];
    VisionStatus enStatus;
    cv::Point2f ptResult;

    if ( NULL == pstCmd || NULL == pstRpy ) {
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstCmd = %d, pstRpy = %d", (int)pstCmd, (int)pstRpy);
        WriteLog(charrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    if ( pstCmd->matInput.empty() ) {
        WriteLog("Input image is empty");
        return VisionStatus::INVALID_PARAM;
    }

    if ( pstCmd->rectSrchRange.x < 0 ||
         pstCmd->rectSrchRange.y < 0 ||
        (pstCmd->rectSrchRange.x + pstCmd->rectSrchRange.width) > pstCmd->matInput.cols || 
        (pstCmd->rectSrchRange.y + pstCmd->rectSrchRange.height) > pstCmd->matInput.rows )
    {
        _snprintf(charrMsg, sizeof(charrMsg), "Search range is invalid, rect x = %d, y = %d, width = %d, height = %d", 
                  pstCmd->rectSrchRange.x, pstCmd->rectSrchRange.y, pstCmd->rectSrchRange.width, pstCmd->rectSrchRange.height );
        WriteLog(charrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;

    std::unique_ptr<LogCaseSrchFiducial> pLogCase;
    if (!bReplay)    {
        pLogCase = std::make_unique<LogCaseSrchFiducial>(Config::GetInstance()->getLogCaseDir());
        if (PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode())
            pLogCase->WriteCmd(pstCmd);
    }

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

    cv::Mat matSrchROI( pstCmd->matInput, pstCmd->rectSrchRange );
    if ( matSrchROI.channels() > 1 )
        cv::cvtColor(matSrchROI, matSrchROI, cv::COLOR_BGR2GRAY);

    enStatus = matchTemplate ( matSrchROI, matTmpl, ptResult );

    pstRpy->ptPos.x = ptResult.x + pstCmd->rectSrchRange.x;
    pstRpy->ptPos.y = ptResult.y + pstCmd->rectSrchRange.y;

    pstRpy->nStatus = ToInt32 ( enStatus );

    pstRpy->matResult = pstCmd->matInput.clone();
    cv::circle(pstRpy->matResult, pstRpy->ptPos, 2, cv::Scalar(255, 0, 0), 2);
    cv::rectangle(pstRpy->matResult, cv::Rect(ToInt32(pstRpy->ptPos.x) - nTmplSize / 2, ToInt32(pstRpy->ptPos.y) - nTmplSize / 2, nTmplSize, nTmplSize), cv::Scalar(255, 0, 0), 1);

    if (!bReplay)    {
        if (PR_DEBUG_MODE::LOG_FAIL_CASE == Config::GetInstance()->getDebugMode() && enStatus != VisionStatus::OK)    {
            pLogCase->WriteCmd(pstCmd);
            pLogCase->WriteRpy(pstRpy);
        }

        if (PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode())
            pLogCase->WriteRpy(pstRpy);
    }

    MARK_FUNCTION_END_TIME;
    return enStatus;
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

/*static*/ std::vector<ListOfPoint::const_iterator> VisionAlgorithm::_findPointOverLineTol(const ListOfPoint   &listPoint,
                                                     bool                   bReversedFit,
                                                     const float            fSlope,
                                                     const float            fIntercept,
                                                     PR_RM_FIT_NOISE_METHOD method,
                                                     float                  tolerance)
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

VisionStatus VisionAlgorithm::fitLine(PR_FIT_LINE_CMD *pstCmd, PR_FIT_LINE_RPY *pstRpy, bool bReplay)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    
    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInput.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInput.rows )    {
        WriteLog("The OCR search range is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    std::unique_ptr<LogCaseFitLine> pLogCase;
    if ( ! bReplay )    {
        pLogCase = std::make_unique<LogCaseFitLine>( Config::GetInstance()->getLogCaseDir() );
        if ( PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode() )
            pLogCase->WriteCmd ( pstCmd );
    }

    VisionStatus enStatus = VisionStatus::OK;    
    std::vector<int> vecX, vecY;
    std::vector<ListOfPoint::const_iterator> vecOverTolIter;

    cv::Mat matGray, matThreshold;
    if ( pstCmd->matInput.channels() > 1 )
        cv::cvtColor ( pstCmd->matInput, matGray, CV_BGR2GRAY );
    else
        matGray = pstCmd->matInput.clone();

    ListOfPoint listPoint = _findPointsInRegionByThreshold( matGray, pstCmd->rectROI, pstCmd->nThreshold, pstCmd->enAttribute );
    if ( listPoint.size() < 2 )  {
        WriteLog("Input image is empty");
        enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        goto EXIT;
    }
    
    for ( const auto &point : listPoint )
    {
        vecX.push_back ( point.x );
        vecY.push_back ( point.y );
    }

    //For y = kx + b, the k is the slope, when it is very large, the error is quite big, so change
    //to get the x = k1 + b1, and get the k = 1 / k1, b = -b1 / k1. Then the result is more accurate.
    pstRpy->bReversedFit = false;
    if ( CalcUtils::calcStdDeviation ( vecY ) > CalcUtils::calcStdDeviation ( vecX ) )
        pstRpy->bReversedFit = true;

    int nIteratorNum = 0;
    do
    {
        for (const auto &it : vecOverTolIter )
            listPoint.erase( it );

        Fitting::fitLine ( listPoint, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->bReversedFit );
        vecOverTolIter = _findPointOverLineTol(listPoint, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept, pstCmd->enRmNoiseMethod, pstCmd->fErrTol);
    }while ( ! vecOverTolIter.empty() && nIteratorNum < 20 );

    if ( listPoint.size() < 2 )  {
        enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        goto EXIT;
    }

    pstRpy->stLine = CalcUtils::calcEndPointOfLine ( listPoint, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept );
    pstRpy->matResult = pstCmd->matInput.clone();
    cv::line ( pstRpy->matResult, pstRpy->stLine.pt1, pstRpy->stLine.pt2, cv::Scalar(255,0,0), 2 );
    enStatus = VisionStatus::OK;

EXIT:
    pstRpy->enStatus = enStatus;
    if ( ! bReplay )    {
        if ( PR_DEBUG_MODE::LOG_FAIL_CASE == Config::GetInstance()->getDebugMode() && enStatus != VisionStatus::OK )    {
            pLogCase->WriteCmd ( pstCmd );
            pLogCase->WriteRpy ( pstRpy );
        }

        if ( PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode() )
            pLogCase->WriteRpy ( pstRpy );
    }
    
    MARK_FUNCTION_END_TIME;
    return enStatus;
}

VisionStatus VisionAlgorithm::fitParallelLine(PR_FIT_PARALLEL_LINE_CMD *pstCmd, PR_FIT_PARALLEL_LINE_RPY *pstRpy, bool bReplay)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->nStatus = ToInt32 ( VisionStatus::INVALID_PARAM );
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;

    std::unique_ptr<LogCaseFitParallelLine> pLogCase;
    if ( ! bReplay )    {
        pLogCase = std::make_unique<LogCaseFitParallelLine>( Config::GetInstance()->getLogCaseDir() );
        if ( PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode() )
            pLogCase->WriteCmd ( pstCmd );
    }

    VisionStatus enStatus = VisionStatus::OK;
    cv::Mat matGray, matThreshold;
    if ( pstCmd->matInput.channels() > 1 )
        cv::cvtColor ( pstCmd->matInput, matGray, CV_BGR2GRAY );
    else
        matGray = pstCmd->matInput.clone();

    ListOfPoint listPoint1 = _findPointsInRegionByThreshold ( matGray, pstCmd->rectArrROI[0], pstCmd->nThreshold, pstCmd->enAttribute );
    ListOfPoint listPoint2 = _findPointsInRegionByThreshold ( matGray, pstCmd->rectArrROI[1], pstCmd->nThreshold, pstCmd->enAttribute );
    if ( listPoint1.size() < 2 || listPoint2.size() < 2 )  {
        WriteLog("Not enough points to fit Parallel line");
        pstRpy->nStatus = ToInt32(VisionStatus::NOT_ENOUGH_POINTS_TO_FIT);
        return VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
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
        enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        goto EXIT;
    }

    pstRpy->stLine1 = CalcUtils::calcEndPointOfLine ( listPoint1, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept1 );
    pstRpy->stLine2 = CalcUtils::calcEndPointOfLine ( listPoint2, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept2 );

    pstRpy->matResult = pstCmd->matInput.clone();
    cv::line ( pstRpy->matResult, pstRpy->stLine1.pt1, pstRpy->stLine1.pt2, cv::Scalar(255,0,0), 2 );
    cv::line ( pstRpy->matResult, pstRpy->stLine2.pt1, pstRpy->stLine2.pt2, cv::Scalar(255,0,0), 2 );
    enStatus = VisionStatus::OK;
   
EXIT:
    pstRpy->nStatus = ToInt32(enStatus);
    if ( ! bReplay )    {
        if ( PR_DEBUG_MODE::LOG_FAIL_CASE == Config::GetInstance()->getDebugMode() && enStatus != VisionStatus::OK )    {
            pLogCase->WriteCmd ( pstCmd );
            pLogCase->WriteRpy ( pstRpy );
        }

        if ( PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode() )
            pLogCase->WriteRpy ( pstRpy );
    }
    
    MARK_FUNCTION_END_TIME;
    return enStatus;
}

VisionStatus VisionAlgorithm::fitRect(PR_FIT_RECT_CMD *pstCmd, PR_FIT_RECT_RPY *pstRpy, bool bReplay)
{
    char charrMsg [ 1000 ];
    if (NULL == pstCmd || NULL == pstRpy) {
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstCmd = %d, pstRpy = %d", (int)pstCmd, (int)pstRpy);
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    std::unique_ptr<LogCaseFitRect> pLogCase;
    if ( ! bReplay )    {
        pLogCase = std::make_unique<LogCaseFitRect>( Config::GetInstance()->getLogCaseDir() );
        if ( PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode() )
            pLogCase->WriteCmd ( pstCmd );
    }

    VisionStatus enStatus = VisionStatus::OK;
    cv::Mat matGray, matThreshold;
    if ( pstCmd->matInput.channels() > 1 )
        cv::cvtColor ( pstCmd->matInput, matGray, CV_BGR2GRAY );
    else
        matGray = pstCmd->matInput.clone();

    VectorOfListOfPoint vecListPoint;
    for ( int i = 0; i < PR_RECT_EDGE_COUNT; ++ i ) {
        vecListPoint.push_back ( _findPointsInRegionByThreshold ( matGray, pstCmd->rectArrROI[i], pstCmd->nThreshold, pstCmd->enAttribute ) );
        if ( vecListPoint [ i ].size() < 2 )  {
            _snprintf(charrMsg, sizeof(charrMsg), "Edge %d not enough points to fit rect", i);
            WriteLog(charrMsg);
            pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
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
    for ( const auto &listPoint : vecListPoint )  {
        if (listPoint.size() < 2)  {
            enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
            _snprintf(charrMsg, sizeof(charrMsg), "Rect line %d has too much noise to fit", nLineNumber);
            WriteLog(charrMsg);
            goto EXIT;
        }
        ++ nLineNumber;
    }

    for ( int i = 0; i < PR_RECT_EDGE_COUNT; ++ i ) {
        if ( i < 2 )
            pstRpy->fArrLine[i] = CalcUtils::calcEndPointOfLine( vecListPoint[i], pstRpy->bLineOneReversedFit, pstRpy->fSlope1, vecIntercept[i]);
        else
            pstRpy->fArrLine[i] = CalcUtils::calcEndPointOfLine( vecListPoint[i], pstRpy->bLineTwoReversedFit, pstRpy->fSlope2, vecIntercept[i]);
        pstRpy->fArrIntercept[i] = vecIntercept[i];
    }
    pstRpy->matResult = pstCmd->matInput.clone();
    for ( int i = 0; i < PR_RECT_EDGE_COUNT; ++ i ) {
        cv::line ( pstRpy->matResult, pstRpy->fArrLine[i].pt1, pstRpy->fArrLine[i].pt2, cv::Scalar(255,0,0), 2 );
    }
    enStatus = VisionStatus::OK;

EXIT:
    pstRpy->enStatus = enStatus;
    if ( ! bReplay )    {
        if ( PR_DEBUG_MODE::LOG_FAIL_CASE == Config::GetInstance()->getDebugMode() && enStatus != VisionStatus::OK )    {
            pLogCase->WriteCmd ( pstCmd );
            pLogCase->WriteRpy ( pstRpy );
        }

        if ( PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode() )
            pLogCase->WriteRpy ( pstRpy );
    }
    
    MARK_FUNCTION_END_TIME;
    return enStatus;
}

VisionStatus VisionAlgorithm::findEdge(PR_FIND_EDGE_CMD *pstCmd, PR_FIND_EDGE_RPY *pstRpy)
{
    char charrMsg [ 1000 ];
    if (NULL == pstCmd || NULL == pstRpy) {
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstCmd = %d, pstRpy = %d", (int)pstCmd, (int)pstRpy);
        WriteLog(charrMsg);
        pstRpy->nStatus = ToInt32 ( VisionStatus::INVALID_PARAM );
        return VisionStatus::INVALID_PARAM;
    }

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->nStatus = ToInt32 ( VisionStatus::INVALID_PARAM );
        return VisionStatus::INVALID_PARAM;
    }

    cv::Mat matROI(pstCmd->matInput, pstCmd->rectROI);
    if ( matROI.empty() ){
        WriteLog("ROI image is empty");
        pstRpy->nStatus = ToInt32 ( VisionStatus::INVALID_PARAM );
        return VisionStatus::INVALID_PARAM;
    }

    if ( ! pstCmd->bAutothreshold && ( pstCmd->nThreshold <= 0 || pstCmd->nThreshold >= PR_MAX_GRAY_LEVEL ) )  {
         _snprintf(charrMsg, sizeof(charrMsg), "The threshold = %d is invalid", pstCmd->nThreshold);
        WriteLog(charrMsg);
        pstRpy->nStatus = ToInt32 ( VisionStatus::INVALID_PARAM );
        return VisionStatus::INVALID_PARAM;
    }    

    cv::Mat matROIGray, matThreshold, matBlur, matCannyResult;
    cv::cvtColor(matROI, matROIGray, CV_BGR2GRAY);

    auto threshold = pstCmd->nThreshold;
    if ( pstCmd->bAutothreshold )
        threshold = _autoThreshold(matROIGray);

    cv::threshold( matROIGray, matThreshold, pstCmd->nThreshold, 255, cv::THRESH_BINARY);
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Threshold Result", matThreshold );

    cv::blur ( matThreshold, matBlur, cv::Size(2, 2) );

    /// Detect edges using canny
    cv::Canny( matBlur, matCannyResult, pstCmd->nThreshold, 255);
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )
        showImage("Canny Result", matCannyResult );

    vector<cv::Vec4i> lines;
    auto voteThreshold = ToInt32 ( pstCmd->fMinLength );
    HoughLinesP(matCannyResult, lines, 1, CV_PI / 180, voteThreshold, pstCmd->fMinLength, 10);
    vector<PR_Line2f> vecLines, vecLinesAfterMerge;
    for ( size_t i = 0; i < lines.size(); ++ i )
    {
        cv::Point2f pt1((float)lines[i][0], (float)lines[i][1]);
        cv::Point2f pt2((float)lines[i][2], (float)lines[i][3]);
        PR_Line2f line(pt1, pt2);
        vecLines.push_back(line);
    }
    _mergeLines(vecLines, vecLinesAfterMerge);
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )   {
        cv::Mat matResultImage = matROI.clone();
        for ( const auto &line : vecLinesAfterMerge )
            cv::line ( matResultImage, line.pt1, line.pt2, cv::Scalar(255, 0, 0), 2);
        showImage("Find edge Result", matResultImage );
    }
    Int32 nLineCount = 0;
    if ( PR_EDGE_DIRECTION::ALL == pstCmd->enDirection )
        pstRpy->nEdgeCount = ToInt32(vecLinesAfterMerge.size());
    else
    {
        for ( const auto &line : vecLinesAfterMerge )
        {
            float fSlope = CalcUtils::lineSlope(line);
            if ( PR_EDGE_DIRECTION::HORIZONTAL == pstCmd->enDirection && fabs ( fSlope ) < 0.1 )
                ++ nLineCount;
            else if ( PR_EDGE_DIRECTION::VERTIAL == pstCmd->enDirection && fabs ( fSlope ) > 10 )
                ++ nLineCount;
        }
        pstRpy->nEdgeCount = nLineCount;
    }
    pstRpy->nStatus = ToInt32 ( VisionStatus::OK );
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::fitCircle(PR_FIT_CIRCLE_CMD *pstCmd, PR_FIT_CIRCLE_RPY *pstRpy, bool bReplay)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char charrMsg [ 1000 ];    

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    
    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 || 
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInput.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInput.rows ) {
        WriteLog("The input ROI is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if ( ! pstCmd->matMask.empty() ) {
        if ( pstCmd->matMask.rows != pstCmd->matInput.rows || pstCmd->matMask.cols != pstCmd->matInput.cols ) {
            _snprintf(charrMsg, sizeof(charrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d )",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInput.cols, pstCmd->matInput.rows);
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

    if (PR_FIT_CIRCLE_METHOD::RANSAC == pstCmd->enMethod && (1000 <= pstCmd->nMaxRansacTime || pstCmd->nMaxRansacTime <= 0)) {
        _snprintf(charrMsg, sizeof(charrMsg), "Max Rransac time %d is invalid", pstCmd->nMaxRansacTime);
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFitCircle);

    cv::Mat matROI ( pstCmd->matInput, pstCmd->rectROI);    
    cv::Mat matGray, matThreshold;

    if ( pstCmd->matInput.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();

    if ( pstCmd->bPreprocessed )    {
        matThreshold = matGray;
    }
    else
    {
        cv::ThresholdTypes enThresType = cv::THRESH_BINARY;
        if ( PR_OBJECT_ATTRIBUTE::DARK == pstCmd->enAttribute )
            enThresType = cv::THRESH_BINARY_INV;

        cv::threshold(matGray, matThreshold, pstCmd->nThreshold, PR_MAX_GRAY_LEVEL, enThresType);
        CV_Assert(matThreshold.channels() == 1);

        if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
            showImage("Threshold image", matThreshold);
    }

    if ( ! pstCmd->matMask.empty() )    {
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
        goto EXIT;
    }    
    
	if ( PR_FIT_CIRCLE_METHOD::RANSAC == pstCmd->enMethod )
        fitResult = _fitCircleRansac ( vecPoints, pstCmd->fErrTol, pstCmd->nMaxRansacTime, vecPoints.size() / 2 );
	else if (PR_FIT_CIRCLE_METHOD::LEAST_SQUARE == pstCmd->enMethod)
        fitResult = Fitting::fitCircle ( vecPoints );
    else if ( PR_FIT_CIRCLE_METHOD::LEAST_SQUARE_REFINE == pstCmd->enMethod  )
        fitResult = _fitCircleIterate ( vecPoints, pstCmd->enRmNoiseMethod, pstCmd->fErrTol );

    if ( fitResult.center.x <= 0 || fitResult.center.y <= 0 || fitResult.size.width <= 0 )   {
        WriteLog("Failed to fit circle");
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        goto EXIT;
    }
	pstRpy->ptCircleCtr = fitResult.center + cv::Point2f ( ToFloat ( pstCmd->rectROI.x ), ToFloat ( pstCmd->rectROI.y ) );
    pstRpy->fRadius = fitResult.size.width / 2;
    pstRpy->enStatus = VisionStatus::OK;

    pstRpy->matResult = pstCmd->matInput.clone();
	cv::circle ( pstRpy->matResult, pstRpy->ptCircleCtr, (int)pstRpy->fRadius, cv::Scalar(255, 0, 0), 2);

EXIT:
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

    while ( nRansacTime < maxRansacTime )   {
        VectorOfPoint vecSelectedPoints = _randomSelectPoints ( vecPoints, RANSAC_CIRCLE_POINT );
        cv::RotatedRect rectReult = Fitting::fitCircle ( vecSelectedPoints );
        vecSelectedPoints = _findPointsInCircleTol ( vecPoints, rectReult, tolerance );

        if ( vecSelectedPoints.size() >= numOfPtToFinish ) {
           return Fitting::fitCircle ( vecSelectedPoints );
        }
        else if ( vecSelectedPoints.size() > nMaxConsentNum )
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
        auto disToCtr = sqrt( ( point.x - rotatedRect.center.x) * (point.x - rotatedRect.center.x)  + ( point.y - rotatedRect.center.y) * ( point.y - rotatedRect.center.y ) );
        auto err = disToCtr - rotatedRect.size.width / 2;
        if ( fabs ( err ) < tolerance )  {
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
    for ( const auto &point : vecPoints ) {
        listPoint.push_back ( point );
    }
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

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }
    
    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInput.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInput.rows )    {
        WriteLog("The OCR search range is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseOcr);

    VisionStatus enStatus = VisionStatus::OK;
    cv::Mat matROI(pstCmd->matInput, pstCmd->rectROI);
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

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    char charrMsg [ 1000 ];
    if ( pstCmd->matInput.channels() != 3 ) {
        _snprintf(charrMsg, sizeof(charrMsg), "Convert from color to gray, the input image channel number %d not equal to 3", pstCmd->matInput.channels());
        WriteLog(charrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    std::vector<float> coefficients{ pstCmd->stRatio.fRatioB, pstCmd->stRatio.fRatioG, pstCmd->stRatio.fRatioR };
    cv::Mat matCoefficients = cv::Mat(coefficients).reshape(1, 1);
    cv::transform(pstCmd->matInput, pstRpy->matResult, matCoefficients);

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::filter(PR_FILTER_CMD *pstCmd, PR_FILTER_RPY *pstRpy)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char charrMsg [ 1000 ];

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInput.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInput.rows )    {
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

    pstRpy->matResult = pstCmd->matInput.clone();
    cv::Mat matROI ( pstRpy->matResult, pstCmd->rectROI );

    VisionStatus enStatus = VisionStatus::OK;
    if ( PR_FILTER_TYPE::NORMALIZED_BOX_FILTER == pstCmd->enType )  {
        cv::blur( matROI, matROI, pstCmd->szKernel );
    }else if ( PR_FILTER_TYPE::GAUSSIAN_FILTER == pstCmd->enType )  {
        cv::GaussianBlur( matROI, matROI, pstCmd->szKernel, pstCmd->dSigmaX, pstCmd->dSigmaY);
    }else if ( PR_FILTER_TYPE::MEDIAN_FILTER == pstCmd->enType ) {
        cv::medianBlur( matROI, matROI, pstCmd->szKernel.width);
    }else if ( PR_FILTER_TYPE::BILATERIAL_FILTER == pstCmd->enType ) {
        cv::Mat matResult;
        cv::bilateralFilter ( matROI, matResult, pstCmd->nDiameter, pstCmd->dSigmaColor, pstCmd->dSigmaSpace );
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )   {
            showImage("Bilateral filter result", matResult );
        }
        matResult.copyTo( matROI );
    }else {
        enStatus = VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_END_TIME;

    pstRpy->enStatus = enStatus;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::removeCC(PR_REMOVE_CC_CMD *pstCmd, PR_REMOVE_CC_RPY *pstRpy, bool bReplay)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );
    char chArrMsg [ 1000 ];

    if ( pstCmd->matInput.empty() ) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInput.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInput.rows )    {
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

    pstRpy->matResult = pstCmd->matInput.clone();
    cv::Mat matROI ( pstRpy->matResult, pstCmd->rectROI );
    cv::Mat matGray = matROI;
    if ( matROI.channels() == 3 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );

    cv::Mat matLabels;
    cv::connectedComponents( matGray, matLabels, pstCmd->nConnectivity, CV_32S);

    double minValue, maxValue;
    cv::minMaxIdx( matLabels, &minValue, &maxValue );

    VisionStatus enStatus = VisionStatus::OK;
    cv::Mat matRemoveMask = cv::Mat::zeros(matLabels.size(), CV_8UC1);

    if ( maxValue > Config::GetInstance()->getRemoveCCMaxComponents() ) {
        enStatus = VisionStatus::TOO_MUCH_CC_TO_REMOVE;
        goto EXIT;
    }

    int nCurrentLabel = 1;
    int nCurrentLabelCount = 0;
    int nTotalPixels = matLabels.rows * matLabels.cols;
    pstRpy->nRemovedCC = 0;

    do
    {
        cv::Mat matTemp = matLabels - nCurrentLabel;
        nCurrentLabelCount = nTotalPixels - cv::countNonZero ( matTemp );
        if ( 0 < nCurrentLabelCount && nCurrentLabelCount < pstCmd->fAreaThreshold )  {
            matRemoveMask += CalcUtils::genMaskByValue<Int32>( matLabels, nCurrentLabel );
            ++ pstRpy->nRemovedCC;
        }
        ++ nCurrentLabel;
    } while ( nCurrentLabelCount > 0 );

    pstRpy->nTotalCC = nCurrentLabel - 1;

    matGray.setTo(0, matRemoveMask);

    if ( matROI.channels() == 3 )
        cv::cvtColor ( matGray, matROI, CV_GRAY2BGR );

EXIT:
    pstRpy->enStatus = enStatus;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;

    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::detectEdge(PR_DETECT_EDGE_CMD *pstCmd, PR_DETECT_EDGE_RPY *pstRpy, bool bReplay)
{
    assert ( pstCmd != nullptr && pstRpy != nullptr );

    if ( pstCmd->matInput.empty() ) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInput.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInput.rows )    {
        WriteLog("The input ROI is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseDetectEdge);

    pstRpy->matResult = pstCmd->matInput.clone();
    cv::Mat matROI ( pstRpy->matResult, pstCmd->rectROI );

    cv::Mat matGray = matROI;
    if ( pstCmd->matInput.channels() == 3 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );

    cv::Mat matResult;
    cv::Canny ( matGray, matResult, pstCmd->nThreshold1, pstCmd->nThreshold2, pstCmd->nApertureSize );

    if ( pstCmd->matInput.channels() == 3 )
        cv::cvtColor ( matResult, matROI, CV_GRAY2BGR );

    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::circleRoundness(PR_CIRCLE_ROUNDNESS_CMD *pstCmd, PR_CIRCLE_ROUNDNESS_RPY *pstRpy, bool bReplay)
{
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char chArrMsg[1000];

    if (pstCmd->matInput.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        ( pstCmd->rectROI.x + pstCmd->rectROI.width ) > pstCmd->matInput.cols ||
        ( pstCmd->rectROI.y + pstCmd->rectROI.height ) > pstCmd->matInput.rows )    {
        WriteLog("The input ROI is invalid");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if ( ! pstCmd->matMask.empty() ) {
        if ( pstCmd->matMask.rows != pstCmd->matInput.rows || pstCmd->matMask.cols != pstCmd->matInput.cols ) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d )",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInput.cols, pstCmd->matInput.rows);
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
    SETUP_LOGCASE(LogCaseCircleRoundness);

    cv::Mat matROI ( pstCmd->matInput, pstCmd->rectROI);    
    cv::Mat matGray;

    if ( pstCmd->matInput.channels() > 1 )
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
        goto EXIT;
    }    
    
    fitResult = Fitting::fitCircle ( vecPoints );

    if ( fitResult.center.x <= 0 || fitResult.center.y <= 0 || fitResult.size.width <= 0 )   {
        WriteLog("Failed to fit circle");
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        goto EXIT;
    }
	pstRpy->ptCircleCtr = fitResult.center + cv::Point2f ( ToFloat ( pstCmd->rectROI.x ), ToFloat ( pstCmd->rectROI.y ) );
    pstRpy->fRadius = fitResult.size.width / 2;
    pstRpy->enStatus = VisionStatus::OK;
    
    for ( const auto &point : vecPoints )   {
        cv::Point2f pt2f(point);
        vecDistance.push_back ( CalcUtils::distanceOf2Point<float> ( pt2f, fitResult.center ) - pstRpy->fRadius );
    }
    pstRpy->fRoundness = ToFloat ( CalcUtils::calcStdDeviation<float>( vecDistance ) );
    pstRpy->matResult = pstCmd->matInput.clone();
	cv::circle ( pstRpy->matResult, pstRpy->ptCircleCtr, (int)pstRpy->fRadius, cv::Scalar(255, 0, 0), 2);

EXIT:
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}
}
}