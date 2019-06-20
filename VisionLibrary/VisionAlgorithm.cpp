#include <iostream>
#include <limits>
#include <algorithm>
#include <numeric>

#include "VisionAlgorithm.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "TimeLog.h"
#include "boost/filesystem.hpp"
#include "RecordManager.h"
#include "Config.h"
#include "CalcUtils.hpp"
#include "Log.h"
#include "FileUtils.h"
#include "Fitting.hpp"
#include "MatchTmpl.h"
#include "Unwrap.h"
#include "SubFunctions.h"
#include "Auxiliary.hpp"
#include "DataMatrix.h"
#include "TableMapping.h"

using namespace cv::xfeatures2d;
namespace bfs = boost::filesystem;

namespace AOI
{
namespace Vision
{

#define MARK_FUNCTION_START_TIME    CStopWatch      stopWatch; __int64 functionStart = stopWatch.AbsNow()
#define MARK_FUNCTION_END_TIME      TimeLog::GetInstance()->addTimeLog(__FUNCTION__, stopWatch.AbsNow() - functionStart)
#define AT __FILE__, __LINE__

#define SETUP_LOGCASE(classname) \
std::unique_ptr<classname> pLogCase; \
if (! bReplay) {   \
    pLogCase = std::make_unique<classname>(Config::GetInstance()->getLogCaseDir()); \
    if (PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode()) \
        pLogCase->WriteCmd (pstCmd); \
}

#define FINISH_LOGCASE \
if (! bReplay) {   \
    if (PR_DEBUG_MODE::LOG_FAIL_CASE == Config::GetInstance()->getDebugMode() && pstRpy->enStatus != VisionStatus::OK) { \
        pLogCase->WriteCmd(pstCmd); \
        pLogCase->WriteRpy(pstRpy); \
    } \
    if (PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode()) \
        pLogCase->WriteRpy(pstRpy); \
}

#define FINISH_LOGCASE_EX \
if (! bReplay) {   \
    if (PR_DEBUG_MODE::LOG_FAIL_CASE == Config::GetInstance()->getDebugMode() && pstRpy->enStatus != VisionStatus::OK) { \
        pLogCase->WriteCmd(pstCmd); \
        pLogCase->WriteRpy(pstCmd, pstRpy); \
    } \
    if (PR_DEBUG_MODE::LOG_ALL_CASE == Config::GetInstance()->getDebugMode()) \
        pLogCase->WriteRpy(pstCmd, pstRpy); \
}

/*static*/ OcrTesseractPtr VisionAlgorithm::_ptrOcrTesseract;
/*static*/ const cv::Scalar VisionAlgorithm::RED_SCALAR    (0,   0,   255 );
/*static*/ const cv::Scalar VisionAlgorithm::BLUE_SCALAR   (255, 0,   0   );
/*static*/ const cv::Scalar VisionAlgorithm::CYAN_SCALAR   (255, 255, 0   );
/*static*/ const cv::Scalar VisionAlgorithm::GREEN_SCALAR  (0,   255, 0   );
/*static*/ const cv::Scalar VisionAlgorithm::YELLOW_SCALAR (0,   255, 255 );
/*static*/ const String VisionAlgorithm::_strRecordLogPrefix   = "tmplDir.";
/*static*/ const float VisionAlgorithm::_constExpSmoothRatio   = 0.3f;
/*static*/ bool VisionAlgorithm::_bAutoMode = false;

VisionAlgorithm::VisionAlgorithm() {
}

/*static*/ std::unique_ptr<VisionAlgorithm> VisionAlgorithm::create() {
    return std::make_unique<VisionAlgorithm>();
}

/*static*/ bool VisionAlgorithm::isAutoMode() {
    return _bAutoMode;
}

/*static*/ void VisionAlgorithm::EnableAutoMode(bool bEnableAutoMode) {
    _bAutoMode = bEnableAutoMode;
}

/*static*/ VisionStatus VisionAlgorithm::lrnObj(const PR_LRN_OBJ_CMD *const pstCmd, PR_LRN_OBJ_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(NULL != pstCmd || NULL != pstRpy);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectLrn.x < 0 || pstCmd->rectLrn.y < 0 ||
        pstCmd->rectLrn.width <= 0 || pstCmd->rectLrn.height <= 0 ||
        (pstCmd->rectLrn.x + pstCmd->rectLrn.width) > pstCmd->matInputImg.cols ||
        (pstCmd->rectLrn.y + pstCmd->rectLrn.height) > pstCmd->matInputImg.rows) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The learn rect (%d, %d, %d, %d) is invalid.",
            pstCmd->rectLrn.x, pstCmd->rectLrn.y, pstCmd->rectLrn.width, pstCmd->rectLrn.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseLrnObj);

    cv::Ptr<cv::Feature2D> detector;
    if (PR_SRCH_OBJ_ALGORITHM::SIFT == pstCmd->enAlgorithm)
        detector = SIFT::create(_constMinHessian, _constOctave);
    else
        detector = SURF::create(_constMinHessian, _constOctave, _constOctaveLayer);
    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectLrn);
    cv::Mat matBlur;
    cv::blur(matROI, matBlur, cv::Size(3, 3));
    detector->detectAndCompute(matBlur, pstCmd->matMask, pstRpy->vecKeyPoint, pstRpy->matDescritor);

    if (pstRpy->vecKeyPoint.size() < _constLeastFeatures) {
        WriteLog("detectAndCompute can not find feature.");
        pstRpy->enStatus = VisionStatus::LEARN_OBJECT_FAIL;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    pstRpy->matResultImg = matROI.clone();
    cv::drawKeypoints(pstRpy->matResultImg, pstRpy->vecKeyPoint, pstRpy->matResultImg, RED_SCALAR, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    pstRpy->enStatus = VisionStatus::OK;
    pstRpy->ptCenter.x = pstCmd->rectLrn.x + pstCmd->rectLrn.width / 2.0f;
    pstRpy->ptCenter.y = pstCmd->rectLrn.y + pstCmd->rectLrn.height / 2.0f;
    matROI.copyTo(pstRpy->matTmpl);
    _writeLrnObjRecord(pstRpy);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::srchObj(const PR_SRCH_OBJ_CMD *const pstCmd, PR_SRCH_OBJ_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(NULL != pstCmd || NULL != pstRpy);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    char chArrMsg[1000];
    if (pstCmd->nRecordId <= 0) {
        _snprintf(chArrMsg, sizeof (chArrMsg), "The input record ID %d is invalid.", pstCmd->nRecordId);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    ObjRecordPtr ptrRecord = std::static_pointer_cast<ObjRecord> (RecordManager::getInstance()->get(pstCmd->nRecordId));
    if (nullptr == ptrRecord) {
        _snprintf(chArrMsg, sizeof (chArrMsg), "Failed to get record ID %d in system.", pstCmd->nRecordId);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (ptrRecord->getType() != PR_RECORD_TYPE::OBJECT) {
        _snprintf(chArrMsg, sizeof (chArrMsg), "The type of record ID %d is not OBJECT.", pstCmd->nRecordId);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseSrchObj);

    cv::Mat matSrchROI(pstCmd->matInputImg, pstCmd->rectSrchWindow);
    cv::Mat matBlur;
    cv::blur(matSrchROI, matBlur, cv::Size(3, 3));

    cv::Ptr<cv::Feature2D> detector;
    if (PR_SRCH_OBJ_ALGORITHM::SIFT == pstCmd->enAlgorithm)
        detector = SIFT::create(_constMinHessian, _constOctave);
    else
        detector = SURF::create(_constMinHessian, _constOctave, _constOctaveLayer);

    std::vector<cv::KeyPoint> vecKeypointScene;
    cv::Mat matDescriptorScene;
    cv::Mat mask;

    detector->detectAndCompute(matBlur, mask, vecKeypointScene, matDescriptorScene);
    TimeLog::GetInstance()->addTimeLog("detectAndCompute");

    if (vecKeypointScene.size() < _constLeastFeatures) {
        WriteLog("detectAndCompute can not find feature.");
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::SRCH_OBJ_FAIL;
        return pstRpy->enStatus;
    }

    VectorOfDMatch good_matches;

    cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::KDTreeIndexParams;
    cv::Ptr<cv::flann::SearchParams> searchParams = new cv::flann::SearchParams;

    cv::FlannBasedMatcher matcher;

    VectorOfVectorOfDMatch vecVecMatches;
    vecVecMatches.reserve(2000);

    matcher.knnMatch(ptrRecord->getModelDescriptor(), matDescriptorScene, vecVecMatches, 2);
    TimeLog::GetInstance()->addTimeLog("knnMatch");

    double max_dist = 0; double min_dist = 100.;
    int nScoreCount = 0;
    for (VectorOfVectorOfDMatch::const_iterator it = vecVecMatches.begin(); it != vecVecMatches.end(); ++ it) {
        VectorOfDMatch vecDMatch = *it;
        if ((vecDMatch[0].distance / vecDMatch[1].distance) < 0.8)
            good_matches.push_back(vecDMatch[0]);

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

    if (good_matches.size() <= 0) {
        pstRpy->enStatus = VisionStatus::SRCH_OBJ_FAIL;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); ++ i) {
        //-- Get the keypoints from the good matches
        obj.push_back(ptrRecord->getModelKeyPoint()[good_matches[i].queryIdx].pt);
        scene.push_back(vecKeypointScene[good_matches[i].trainIdx].pt);
    }

    pstRpy->matHomography = cv::findHomography(obj, scene, CV_RANSAC);
    TimeLog::GetInstance()->addTimeLog("cv::findHomography");

    pstRpy->matHomography.at<double>(0, 2) += pstCmd->rectSrchWindow.x;
    pstRpy->matHomography.at<double>(1, 2) += pstCmd->rectSrchWindow.y;

    auto vevVecHomography = CalcUtils::matToVector<double>(pstRpy->matHomography);

    cv::Point2f ptLrnObjCenter = ptrRecord->getObjCenter();
    cv::Rect2f rectLrn(0, 0, ptLrnObjCenter.x * 2.f, ptLrnObjCenter.y * 2.f);

    VectorOfPoint2f vecPoint2f = CalcUtils::warpRect<double>(pstRpy->matHomography, rectLrn);
    VectorOfVectorOfPoint vecVecPoint(1);
    for (const auto &point : vecPoint2f)
        vecVecPoint[0].push_back(point);
    cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);
    cv::polylines(pstRpy->matResultImg, vecVecPoint, true, RED_SCALAR);
    cv::drawKeypoints(pstRpy->matResultImg, vecKeypointScene, pstRpy->matResultImg, RED_SCALAR, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    pstRpy->ptObjPos = CalcUtils::warpPoint<double>(pstRpy->matHomography, ptLrnObjCenter);
    pstRpy->szOffset.width = pstRpy->ptObjPos.x - pstCmd->ptExpectedPos.x;
    pstRpy->szOffset.height = pstRpy->ptObjPos.y - pstCmd->ptExpectedPos.y;
    float fRotationValue = (float)(pstRpy->matHomography.at<double>(1, 0) - pstRpy->matHomography.at<double>(0, 1)) / 2.0f;
    pstRpy->fRotation = (float)CalcUtils::radian2Degree(asin(fRotationValue));
    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

namespace
{
    int _calculateVerticalProjection(const cv::Mat &mat, cv::Mat &output) {
        if (mat.empty())
            return 0;

        output = cv::Mat::zeros(1, mat.rows, CV_32FC1);
        VectorOfPoint vecPoints;
        cv::findNonZero(mat, vecPoints);
        for (const auto &point : vecPoints) {
            ++ output.at<float>(0, point.y);
        }
        return 0;
    }

    int _calculateHorizontalProjection(const cv::Mat &mat, cv::Mat &output) {
        if (mat.empty())
            return 0;
        cv::Mat matTranspose;
        cv::transpose(mat, matTranspose);
        _calculateVerticalProjection(matTranspose, output);
        return 0;
    }

    //The exponent smooth algorithm come from pater "A Visual Inspection System for Surface Mounted Components Based on Color Features".
    template<typename T>
    void _expSmooth(const cv::Mat &matInputImg, cv::Mat &matOutput, float alpha) {
        matInputImg.copyTo(matOutput);
        for (int col = 1; col < matInputImg.cols; ++col) {
            matOutput.at<T>(0, col) = alpha * matInputImg.at<T>(0, col) + (1.f - alpha) * matOutput.at<T>(0, col - 1);
        }
    }

    //To find edge from left to right, or from top to bottom.
    VisionStatus _findEdge(const cv::Mat &matProjection, int nObjectSize, int &nObjectEdgeResult) {
        if (matProjection.cols < nObjectSize)
            return VisionStatus::INVALID_PARAM;

        auto fMaxSumOfProjection = 0.f;
        nObjectEdgeResult = 0;
        for (int nObjectStart = 0; nObjectStart < matProjection.cols - nObjectSize; ++ nObjectStart) {
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
    VisionStatus _findEdgeReverse(const cv::Mat &matProjection, int nObjectSize, int &nObjectEdgeResult) {
        if (matProjection.cols < nObjectSize)
            return VisionStatus::INVALID_PARAM;

        auto fMaxSumOfProjection = 0.f;
        nObjectEdgeResult = 0;
        for (int nObjectStart = matProjection.cols - 1; nObjectStart > nObjectSize; --nObjectStart) {
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

    struct AOI_COUNTOUR
    {
        cv::Point2f     ptCtr;
        float           fArea;
        VectorOfPoint   contour;
    };

    bool _isSimilarContourExist(const std::vector<AOI_COUNTOUR> &vecAoiCountour, const AOI_COUNTOUR &stCountourInput) {
        for (auto stContour : vecAoiCountour) {
            if (CalcUtils::distanceOf2Point<float>(stContour.ptCtr, stCountourInput.ptCtr) < 3.f
                && fabs(stContour.fArea - stCountourInput.fArea) < 10)
                return true;
        }
        return false;
    }
}

namespace
{
    cv::Mat getpdf(const cv::Mat &matInput, double &dMin, double &dMax, int sbins) {
        assert(! matInput.empty());
        cv::Mat matMask = matInput == matInput;
        cv::minMaxIdx(matInput, &dMin, &dMax, 0, 0, matMask);
        cv::Mat matFloat;
        matInput.convertTo(matFloat, CV_32FC1);
        cv::Mat matNormlized = (matFloat - dMin) / (dMax - dMin);
        cv::MatND hist;
        int channels[] = {0};
        int histSize[] = {sbins};
        float sranges[] = {0, 256};
        matNormlized = matNormlized * 255;
        cv::Mat matConvertBack;
        matNormlized.convertTo(matConvertBack, CV_8UC1);
        const float* ranges[] = {sranges};
        cv::calcHist(&matConvertBack, 1, channels, matMask, // do not use mask
            hist, 1, histSize, ranges,
            true, // the histogram is uniform
            false);
        hist = hist / ToFloat(matConvertBack.total());
        return hist;
    }

}
/*******************************************************************************
* Purpose:   Auto find the threshold to distinguish the feature need to extract.
* Algorithm: The method is come from paper: "A Fast Algorithm for Multilevel
*            Thresholding".
* Input:     mat - The input image.
N - Number of therehold to calculate. 1 <= N <= 10
* Output:    The calculated threshold.
* Note:      Mu is name of greek symbol μ.
*            Omega and Eta are also greek symbol.
*******************************************************************************/
std::vector<Int16> VisionAlgorithm::autoMultiLevelThreshold(const cv::Mat &matInputImg, const cv::Mat &matMask, int N) {
    assert(N >= 1 && N <= PR_MAX_AUTO_THRESHOLD_COUNT);

    const int sbins = 255;
    const float MIN_P = 0.001f;
    std::vector<Int16> vecThreshold;
    int channels[] = {0};
    int histSize[] = {sbins};

    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = {0, 256};
    const float* ranges[] = {sranges};

    cv::MatND hist;
    cv::calcHist(&matInputImg, 1, channels, matMask,
        hist, 1, histSize, ranges,
        true, // the histogram is uniform
        false);

    std::vector<float> P, S;
    P.push_back(0.f);
    S.push_back(0.f);
    float fTotalPixel = (float)matInputImg.rows * matInputImg.cols;
    for (int i = 0; i < sbins; ++i) {
        float fPi = hist.at<float>(i, 0) / fTotalPixel;
        P.push_back(fPi + P[i]);
        S.push_back(i * fPi + S[i]);
    }

    std::vector<std::vector<float>> vecVecOmega(sbins + 1, std::vector<float>(sbins + 1, 0));
    std::vector<std::vector<float>> vecVecMu(sbins + 1, std::vector<float>(sbins + 1, 0));
    std::vector<std::vector<float>> vecVecEta(sbins + 1, std::vector<float>(sbins + 1, -1.f));
    for (int i = 0; i <= sbins; ++ i)
    for (int j = i; j <= sbins; ++ j) {
        vecVecOmega[i][j] = P[j] - P[i];
        vecVecMu[i][j] = S[j] - S[i];
    }

    const int START = 1;
    if (1 == N) {
        float fMaxSigma = 0;
        int nResultT1 = 0;
        for (int T = START; T < sbins; ++ T) {
            float fSigma = 0.f;
            if (vecVecOmega[START][T] > MIN_P)
                fSigma += vecVecMu[START][T] * vecVecMu[1][T] / vecVecOmega[START][T];
            if (vecVecOmega[T][sbins] > MIN_P)
                fSigma += vecVecMu[T][sbins] * vecVecMu[T][sbins] / vecVecOmega[T][sbins];
            if (fMaxSigma < fSigma) {
                nResultT1 = T;
                fMaxSigma = fSigma;
            }
        }
        vecThreshold.push_back(nResultT1 * PR_MAX_GRAY_LEVEL / sbins);
    }
    else if (2 == N) {
        float fMaxSigma = 0;
        int nResultT1 = 0, nResultT2 = 0;
        for (int T1 = START; T1 < sbins - 1; ++ T1)
        for (int T2 = T1 + 1; T2 < sbins; ++ T2) {
            float fSigma = 0.f;

            if (vecVecEta[START][T1] >= 0.f)
                fSigma += vecVecEta[START][T1];
            else {
                if (vecVecOmega[START][T1] > MIN_P)
                    vecVecEta[START][T1] = vecVecMu[START][T1] * vecVecMu[START][T1] / vecVecOmega[START][T1];
                else
                    vecVecEta[START][T1] = 0.f;
                fSigma += vecVecEta[START][T1];
            }

            if (vecVecEta[T1][T2] >= 0.f)
                fSigma += vecVecEta[T1][T2];
            else {
                if (vecVecOmega[T1][T2] > MIN_P)
                    vecVecEta[T1][T2] = vecVecMu[T1][T2] * vecVecMu[T1][T2] / vecVecOmega[T1][T2];
                else
                    vecVecEta[T1][T2] = 0.f;
                fSigma += vecVecEta[T1][T2];
            }

            if (vecVecEta[T2][sbins] >= 0.f)
                fSigma += vecVecEta[T2][sbins];
            else {
                if (vecVecOmega[T2][sbins] > MIN_P)
                    vecVecEta[T2][sbins] = vecVecMu[T2][sbins] * vecVecMu[T2][sbins] / vecVecOmega[T2][sbins];
                else
                    vecVecEta[T2][sbins] = 0.f;
                fSigma += vecVecEta[T2][sbins];
            }

            if (fMaxSigma < fSigma) {
                nResultT1 = T1;
                nResultT2 = T2;
                fMaxSigma = fSigma;
            }
        }
        vecThreshold.push_back(nResultT1 * PR_MAX_GRAY_LEVEL / sbins);
        vecThreshold.push_back(nResultT2 * PR_MAX_GRAY_LEVEL / sbins);
    }
    else if (N >= 3) {
        double dMin = 0.f, dMax = 0.f;
        cv::Mat matP = getpdf(matInputImg, dMin, dMax, sbins);

        cv::Mat omega = CalcUtils::cumsum<float>(matP, 1);
        cv::Mat matGo;
        matGo = CalcUtils::intervals<float>(1.f, 1.f, ToFloat(sbins));
        matGo = matGo.mul(matP);

        cv::Mat mu = CalcUtils::cumsum<float>(matGo, 1);
        float mu_t = mu.at<float>(ToInt32(mu.total() - 1));

        float fInterval = float(sbins - 1) / float(N + 1);
        float fThreshold = fInterval;
        std::vector<float> vecX;
        for (int i = 0; i < N; ++ i) {
            vecX.push_back(fThreshold);
            fThreshold += fInterval;
        }

        fminsearch(matlabObjFunction, vecX, sbins, omega, mu, mu_t);
        for (auto x : vecX) {
            auto normFactor = 255;
            auto fThreshold = dMin + round(x) / normFactor* (dMax - dMin);
            vecThreshold.push_back(ToInt16(round(fThreshold)));
        }
    }

    return vecThreshold;
}

VisionStatus VisionAlgorithm::autoThreshold(PR_AUTO_THRESHOLD_CMD *pstCmd, PR_AUTO_THRESHOLD_RPY *pstRpy, bool bReplay) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (! pstCmd->matMask.empty()) {
        if (pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d ).",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matMask.channels() != 1) {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseAutoThreshold);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    cv::Mat matMaskROI;
    if (! pstCmd->matMask.empty())
        matMaskROI = cv::Mat(pstCmd->matMask, pstCmd->rectROI).clone();
    pstRpy->vecThreshold = autoMultiLevelThreshold(matROI, matMaskROI, pstCmd->nThresholdNum);

    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

int VisionAlgorithm::_autoThreshold(const cv::Mat &mat, const cv::Mat &matMask/* = cv::Mat()*/) {
    auto vecThreshold = autoMultiLevelThreshold(mat, matMask, 1);
    return vecThreshold[0];
}

//Need to use the in tolerance method to check.
/*static*/ bool VisionAlgorithm::_isContourRectShape(const VectorOfPoint &vecPoint) {
    cv::Rect rect = cv::boundingRect(vecPoint);
    if (rect.width <= 0 || rect.height <= 0)
        return false;

    auto rectHead = cv::minAreaRect(vecPoint);

    float fTolerance = 4.f;
    if (rectHead.size.width <= 2 * fTolerance || rectHead.size.height <= 2 * fTolerance) {
        return false;
    }

    if (fabs(rectHead.angle) < 5) {
        if (! _isContourRectShapeProjection(vecPoint))
            return false;
    }

    cv::Mat matVerify = cv::Mat::zeros(rect.size() + cv::Size(rect.x, rect.y), CV_8UC1);
    auto rectEnlarge = rectHead;
    rectEnlarge.size.width += 2 * fTolerance;
    rectEnlarge.size.height += 2 * fTolerance;
    VectorOfVectorOfPoint vevVecPoint;
    vevVecPoint.push_back(CalcUtils::getCornerOfRotatedRect(rectEnlarge));
    cv::fillPoly(matVerify, vevVecPoint, cv::Scalar::all(PR_MAX_GRAY_LEVEL));

    auto rectDeflate = rectHead;
    rectDeflate.size.width -= 2 * fTolerance;
    rectDeflate.size.height -= 2 * fTolerance;
    vevVecPoint.clear();
    vevVecPoint.push_back(CalcUtils::getCornerOfRotatedRect(rectDeflate));
    cv::fillPoly(matVerify, vevVecPoint, cv::Scalar::all(0));
    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
        showImage("Verify Mat", matVerify);

    int nInTolPointCount = 0;
    for (const auto &point : vecPoint) {
        if (matVerify.at<uchar>(point) > 0)
            ++ nInTolPointCount;
    }

    if (nInTolPointCount < ToInt32(vecPoint.size() * 0.5))
        return false;
    return true;
}

/*static*/ bool VisionAlgorithm::_isContourRectShapeProjection(const VectorOfPoint &vecPoint) {
    cv::Rect rect = cv::boundingRect(vecPoint);
    if (rect.width <= 0 || rect.height <= 0)
        return false;

    VectorOfVectorOfPoint vevVecPoint;
    cv::Mat matBlack = cv::Mat::zeros(rect.size() + cv::Size(rect.x, rect.y), CV_8UC1);
    vevVecPoint.push_back(vecPoint);
    cv::fillPoly(matBlack, vevVecPoint, cv::Scalar::all(PR_MAX_GRAY_LEVEL));

    cv::Mat matROI(matBlack, rect);
    cv::Mat matOneRow, matOneCol;
    cv::reduce(matROI, matOneRow, 0, cv::ReduceTypes::REDUCE_SUM, CV_32SC1);
    cv::reduce(matROI, matOneCol, 1, cv::ReduceTypes::REDUCE_SUM, CV_32SC1);

    float fAverageHeight = ToFloat(cv::sum(matOneRow)[0] / matOneRow.cols);
    int nConsecutiveOverTolerance = 0;
    const float FLUCTUATE_TOL = 0.3f;
    const int FLUCTUATE_WIDTH_TOL = 10;
    for (int index = 0; index < matOneRow.cols; ++ index) {
        if (fabs(matOneRow.at<int>(0, index) - fAverageHeight) / fAverageHeight > FLUCTUATE_TOL)
            ++ nConsecutiveOverTolerance;
        else
            nConsecutiveOverTolerance = 0;

        if (nConsecutiveOverTolerance > FLUCTUATE_WIDTH_TOL)
            return false;
    }
    return true;
}

/*static*/ inline float VisionAlgorithm::getContourRectRatio(const VectorOfPoint &vecPoint) {
    auto rotatedRect = cv::minAreaRect(vecPoint);
    float fMaxDim = std::max(rotatedRect.size.width, rotatedRect.size.height);
    float fMinDim = std::min(rotatedRect.size.width, rotatedRect.size.height);
    return fMaxDim / fMinDim;
}

/*static*/ VisionStatus VisionAlgorithm::_findDeviceElectrode(const cv::Mat &matDeviceROI, VectorOfPoint &vecElectrodePos, VectorOfSize2f &vecElectrodeSize, VectorOfVectorOfPoint &vevVecContour) {
    VectorOfVectorOfPoint vecContours;
    std::vector<cv::Vec4i> hierarchy;
    vecElectrodePos.clear();
    const float MAX_TWO_DIM_RATIO = 4.f;

    cv::Mat matDraw = cv::Mat::zeros(matDeviceROI.size(), CV_8UC3);
    // Find contours
    cv::findContours(matDeviceROI, vecContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // Filter contours
    std::vector<AOI_COUNTOUR> vecAoiContour;
    int index = 0;
    for (auto contour : vecContours) {
        auto area = cv::contourArea(contour);
        if (area > 180 && getContourRectRatio(contour) < MAX_TWO_DIM_RATIO && _isContourRectShape(contour)) {
            AOI_COUNTOUR stAoiContour;
            cv::Moments moment = cv::moments(contour);
            stAoiContour.ptCtr.x = static_cast<float>(moment.m10 / moment.m00);
            stAoiContour.ptCtr.y = static_cast<float>(moment.m01 / moment.m00);
            stAoiContour.fArea = ToFloat(area);
            if (! _isSimilarContourExist(vecAoiContour, stAoiContour)) {
                stAoiContour.contour = contour;
                vecAoiContour.push_back(stAoiContour);

                if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
                    cv::RNG rng(12345);
                    cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                    cv::drawContours(matDraw, vecContours, index, color);
                }
            }
        }
        ++ index;
    }
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Contours", matDraw);

    if (vecAoiContour.size() < PR_ELECTRODE_COUNT)
        return VisionStatus::FIND_ELECTRODE_FAIL;

    if (vecAoiContour.size() > PR_ELECTRODE_COUNT) {
        std::sort(vecAoiContour.begin(), vecAoiContour.end(), [](const AOI_COUNTOUR &stA, const AOI_COUNTOUR &stB) {
            return stB.fArea < stA.fArea;
        });
        vecAoiContour = std::vector<AOI_COUNTOUR>(vecAoiContour.begin(), vecAoiContour.begin() + PR_ELECTRODE_COUNT);
    }

    float fArea1 = vecAoiContour[0].fArea;
    float fArea2 = vecAoiContour[1].fArea;
    if (fabs(fArea1 - fArea2) / fArea1 > 0.2)
        return VisionStatus::FIND_ELECTRODE_FAIL;

    for (const auto &stAOIContour : vecAoiContour) {
        vecElectrodePos.push_back(stAOIContour.ptCtr);
        auto rotatedRect = cv::minAreaRect(stAOIContour.contour);
        vecElectrodeSize.push_back(rotatedRect.size);
        vevVecContour.push_back(stAOIContour.contour);
    }

    return VisionStatus::OK;
}

//The alogorithm come from paper "A Visual Inspection System for Surface Mounted Components Based on Color Features"
/*static*/ VisionStatus VisionAlgorithm::_findDeviceEdge(const cv::Mat &matDeviceROI, const cv::Size2f &size, cv::Rect &rectDeviceResult) {
    cv::Mat matGray, matThreshold, matHorizontalProjection, matVerticalProjection;
    Int32 nLeftEdge = 0, nRightEdge = 0, nTopEdge = 0, nBottomEdge = 0;

    _calculateHorizontalProjection(matDeviceROI, matHorizontalProjection);
    _expSmooth<float>(matHorizontalProjection, matHorizontalProjection, _constExpSmoothRatio);
    _findEdge(matHorizontalProjection, static_cast<Int32>(size.width), nLeftEdge);
    _findEdgeReverse(matHorizontalProjection, static_cast<Int32>(size.width), nRightEdge);
    if ((nRightEdge - nLeftEdge) < (size.width / 2))
        return VisionStatus::FIND_DEVICE_EDGE_FAIL;

    _calculateVerticalProjection(matThreshold, matVerticalProjection);
    _expSmooth<float>(matVerticalProjection, matVerticalProjection, _constExpSmoothRatio);
    _findEdge(matHorizontalProjection, static_cast<Int32>(size.height), nTopEdge);
    _findEdgeReverse(matHorizontalProjection, static_cast<Int32>(size.height), nBottomEdge);
    if ((nBottomEdge - nTopEdge) < size.height / 2)
        return VisionStatus::FIND_DEVICE_EDGE_FAIL;

    rectDeviceResult = cv::Rect(cv::Point(nLeftEdge, nTopEdge), cv::Point(nRightEdge, nBottomEdge));
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::lrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy) {
    char chArrMsg[1000];
    if (nullptr == pstLrnDeviceCmd || nullptr == pstLrnDeivceRpy) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "Input is invalid, pstLrnDeviceCmd = %d, pstLrnDeivceRpy = %d", (int)pstLrnDeviceCmd, (int)pstLrnDeivceRpy);
        WriteLog(chArrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    if (pstLrnDeviceCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        return VisionStatus::INVALID_PARAM;
    }

    cv::Rect rectDeviceROI = CalcUtils::resizeRect<float>(pstLrnDeviceCmd->rectDevice, pstLrnDeviceCmd->rectDevice.size() + cv::Size2f(20, 20));
    CalcUtils::adjustRectROI(rectDeviceROI, pstLrnDeviceCmd->matInputImg);

    cv::Mat matDevice(pstLrnDeviceCmd->matInputImg, rectDeviceROI);
    cv::Mat matGray, matThreshold, matHorizontalProjection, matVerticalProjection;

    std::vector<cv::Mat> vecChannels;
    cv::split(matDevice, vecChannels);

    matGray = vecChannels[Config::GetInstance()->getDeviceInspChannel()];

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Learn channel image", matGray);

    cv::Mat matBlur;
    cv::blur(matGray, matBlur, cv::Size(2, 2));
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Blur image", matBlur);

    if (pstLrnDeviceCmd->bAutoThreshold)
        pstLrnDeviceCmd->nElectrodeThreshold = _autoThreshold(matBlur);

    cv::threshold(matBlur, matThreshold, pstLrnDeviceCmd->nElectrodeThreshold, 255, cv::THRESH_BINARY);

    VectorOfPoint vecPtElectrode;
    VectorOfSize2f vecElectrodeSize;
    VectorOfVectorOfPoint vecContours;
    VisionStatus enStatus = _findDeviceElectrode(matThreshold, vecPtElectrode, vecElectrodeSize, vecContours);
    if (enStatus != VisionStatus::OK) {
        pstLrnDeivceRpy->nStatus = ToInt32(enStatus);
        return enStatus;
    }

    cv::Rect rectFindDeviceResult;
    enStatus = _findDeviceEdge(matThreshold, cv::Size2f(pstLrnDeviceCmd->rectDevice.width, pstLrnDeviceCmd->rectDevice.height), rectFindDeviceResult);
    if (enStatus != VisionStatus::OK) {
        pstLrnDeivceRpy->nStatus = ToInt32(enStatus);
        return enStatus;
    }

    float fScaleX = ToFloat(rectFindDeviceResult.width) / pstLrnDeviceCmd->rectDevice.width;
    float fScaleY = ToFloat(rectFindDeviceResult.height) / pstLrnDeviceCmd->rectDevice.height;
    float fScaleMax = std::max<float>(fScaleX, fScaleY);
    float fScaleMin = std::min<float>(fScaleX, fScaleY);
    if (fScaleMax > 1.2f || fScaleMin < 0.8f) {
        pstLrnDeivceRpy->nStatus = ToInt32(VisionStatus::OBJECT_SCALE_FAIL);
        return VisionStatus::OBJECT_SCALE_FAIL;
    }

    pstLrnDeivceRpy->nStatus = ToInt32(VisionStatus::OK);
    pstLrnDeivceRpy->sizeDevice = cv::Size2f(ToFloat(rectFindDeviceResult.width), ToFloat(rectFindDeviceResult.height));
    pstLrnDeivceRpy->nElectrodeThreshold = pstLrnDeviceCmd->nElectrodeThreshold;
    _writeDeviceRecord(pstLrnDeivceRpy);
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::inspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy) {
    MARK_FUNCTION_START_TIME;
    char chArrMsg[1000];
    if (pstInspDeviceCmd == nullptr || pstInspDeivceRpy == nullptr) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "Input is invalid, pstInspDeviceCmd = %d, pstInspDeivceRpy = %d", (int)pstInspDeviceCmd, (int)pstInspDeivceRpy);
        WriteLog(chArrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    if (pstInspDeviceCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        return VisionStatus::INVALID_PARAM;
    }

    pstInspDeivceRpy->nDeviceCount = pstInspDeviceCmd->nDeviceCount;
    for (Int32 i = 0; i < pstInspDeviceCmd->nDeviceCount; ++ i) {
        if (pstInspDeviceCmd->astDeviceInfo[i].stSize.area() < 1) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "Device[ %d ] size is invalid", i);
            WriteLog(chArrMsg);
            return VisionStatus::INVALID_PARAM;
        }

        cv::Rect rectDeviceROI = pstInspDeviceCmd->astDeviceInfo[i].rectSrchWindow;
        if (rectDeviceROI.x < 0) rectDeviceROI.x = 0;
        if (rectDeviceROI.y < 0) rectDeviceROI.y = 0;
        if ((rectDeviceROI.x + rectDeviceROI.width)  > pstInspDeviceCmd->matInputImg.cols) rectDeviceROI.width = pstInspDeviceCmd->matInputImg.cols - rectDeviceROI.x;
        if ((rectDeviceROI.y + rectDeviceROI.height) > pstInspDeviceCmd->matInputImg.rows) rectDeviceROI.height = pstInspDeviceCmd->matInputImg.rows - rectDeviceROI.y;

        cv::Mat matROI(pstInspDeviceCmd->matInputImg, rectDeviceROI);
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
        cv::split(matROI, vecChannels);

        matGray = vecChannels[Config::GetInstance()->getDeviceInspChannel()];

        cv::Mat matBlur;
        cv::blur(matGray, matBlur, cv::Size(3, 3));
        if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
            showImage("Blur image", matBlur);

        //cv::cvtColor( matROI, matGray, CV_BGR2GRAY);
        cv::threshold(matBlur, matThreshold, pstInspDeviceCmd->nElectrodeThreshold, 255, cv::THRESH_BINARY);
        if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
            cv::imshow("Threshold image", matThreshold);
            cv::waitKey(0);
        }

        VectorOfPoint vecPtElectrode;
        VectorOfSize2f vecElectrodeSize;
        VectorOfVectorOfPoint vecContours;
        VisionStatus enStatus = _findDeviceElectrode(matThreshold, vecPtElectrode, vecElectrodeSize, vecContours);
        if (enStatus != VisionStatus::OK) {
            pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(enStatus);
            continue;
        }

        if (pstInspDeviceCmd->astDeviceInfo[i].stInspItem.bCheckRotation
            || pstInspDeviceCmd->astDeviceInfo[i].stInspItem.bCheckShift
            || pstInspDeviceCmd->astDeviceInfo[i].stInspItem.bCheckScale) {
            if (0 > pstInspDeviceCmd->astDeviceInfo[i].nCriteriaNo || pstInspDeviceCmd->astDeviceInfo[i].nCriteriaNo >= PR_MAX_CRITERIA_COUNT) {
                _snprintf(chArrMsg, sizeof(chArrMsg), "Device[ %d ] Input nCriteriaNo %d is invalid", i, pstInspDeviceCmd->astDeviceInfo[i].nCriteriaNo);
                WriteLog(chArrMsg);
                return VisionStatus::INVALID_PARAM;
            }
        }

        double dRotate = CalcUtils::calcSlopeDegree(vecPtElectrode[0], vecPtElectrode[1]);
        pstInspDeivceRpy->astDeviceResult[i].fRotation = ToFloat(dRotate);
        if (pstInspDeviceCmd->astDeviceInfo[i].stInspItem.bCheckRotation) {
            if (fabs(dRotate) > pstInspDeviceCmd->astCriteria[pstInspDeviceCmd->astDeviceInfo[i].nCriteriaNo].fMaxRotate) {
                pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(VisionStatus::OBJECT_ROTATE);
                continue;
            }
        }

        cv::Point2f ptDeviceCtr = CalcUtils::midOf2Points<float>(vecPtElectrode[0], vecPtElectrode[1]);
        //Map the find result from ROI to the whole image.
        ptDeviceCtr.x += rectDeviceROI.x;
        ptDeviceCtr.y += rectDeviceROI.y;

        pstInspDeivceRpy->astDeviceResult[i].ptPos = ptDeviceCtr;
        pstInspDeivceRpy->astDeviceResult[i].fOffsetX = pstInspDeivceRpy->astDeviceResult[i].ptPos.x - pstInspDeviceCmd->astDeviceInfo[i].stCtrPos.x;
        pstInspDeivceRpy->astDeviceResult[i].fOffsetY = pstInspDeivceRpy->astDeviceResult[i].ptPos.y - pstInspDeviceCmd->astDeviceInfo[i].stCtrPos.y;
        if (pstInspDeviceCmd->astDeviceInfo[i].stInspItem.bCheckShift) {
            if (fabs(pstInspDeivceRpy->astDeviceResult[i].fOffsetX) > pstInspDeviceCmd->astCriteria[pstInspDeviceCmd->astDeviceInfo[i].nCriteriaNo].fMaxOffsetX) {
                pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(VisionStatus::OBJECT_SHIFT);
                continue;
            }

            if (fabs(pstInspDeivceRpy->astDeviceResult[i].fOffsetY) > pstInspDeviceCmd->astCriteria[pstInspDeviceCmd->astDeviceInfo[i].nCriteriaNo].fMaxOffsetY) {
                pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(VisionStatus::OBJECT_SHIFT);
                continue;
            }
        }

        cv::Rect rectDevice;
        enStatus = _findDeviceEdge(matThreshold, pstInspDeviceCmd->astDeviceInfo[i].stSize, rectDevice);
        if (enStatus != VisionStatus::OK) {
            pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(enStatus);
            continue;
        }

        float fScaleX = ToFloat(rectDevice.width) / pstInspDeviceCmd->astDeviceInfo[i].stSize.width;
        float fScaleY = ToFloat(rectDevice.height) / pstInspDeviceCmd->astDeviceInfo[i].stSize.height;
        pstInspDeivceRpy->astDeviceResult[i].fScale = (fScaleX + fScaleY) / 2.f;
        if (pstInspDeviceCmd->astDeviceInfo[i].stInspItem.bCheckScale) {
            //float fScaleMax = std::max<float> ( fScaleX, fScaleY );
            //float fScaleMin = std::min<float> ( fScaleX, fScaleY );
            if (pstInspDeivceRpy->astDeviceResult[i].fScale > pstInspDeviceCmd->astCriteria[pstInspDeviceCmd->astDeviceInfo[i].nCriteriaNo].fMaxScale
                || pstInspDeivceRpy->astDeviceResult[i].fScale < pstInspDeviceCmd->astCriteria[pstInspDeviceCmd->astDeviceInfo[i].nCriteriaNo].fMinScale) {
                pstInspDeivceRpy->astDeviceResult[i].nStatus = ToInt32(VisionStatus::OBJECT_SCALE_FAIL);
                continue;
            }
        }
    }
    MARK_FUNCTION_END_TIME;
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::lrnTemplate(const PR_LRN_TEMPLATE_CMD *const pstCmd, PR_LRN_TEMPLATE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (!pstCmd->matMask.empty()) {
        if (pstCmd->matMask.cols != pstCmd->rectROI.width || pstCmd->matMask.rows != pstCmd->rectROI.height) {
            std::stringstream ss;
            ss << "The size of mask " << pstCmd->matMask.size() << " doesn't match with ROI size " << pstCmd->rectROI.size();
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseLrnTmpl);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    if (matROI.channels() > 1)
        cv::cvtColor(matROI, pstRpy->matTmpl, CV_BGR2GRAY);
    else
        pstRpy->matTmpl = matROI.clone();
    cv::Mat matEdgeMask;
    cv::Mat matMask = pstCmd->matMask.empty() ? pstCmd->matMask : pstCmd->matMask > 0;
    if (PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE == pstCmd->enAlgorithm) {
        cv::Mat matCanny;
        cv::Canny(pstRpy->matTmpl, matCanny, 50, 200, 3);
        cv::dilate(matCanny, matEdgeMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
        if (!pstCmd->matMask.empty()) {
            cv::Mat matReverseMask = PR_MAX_GRAY_LEVEL - matMask;
            matEdgeMask.setTo(0, matReverseMask);
        }
    }

    TmplRecordPtr ptrRecord = std::make_shared<TmplRecord>(PR_RECORD_TYPE::TEMPLATE, pstCmd->enAlgorithm, pstRpy->matTmpl, matMask, matEdgeMask);
    RecordManager::getInstance()->add(ptrRecord, pstRpy->nRecordId);
    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::matchTemplate(const PR_MATCH_TEMPLATE_CMD *const pstCmd, PR_MATCH_TEMPLATE_RPY * const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectSrchWindow.x < 0 || pstCmd->rectSrchWindow.y < 0 ||
        pstCmd->rectSrchWindow.width <= 0 || pstCmd->rectSrchWindow.height <= 0 ||
        (pstCmd->rectSrchWindow.x + pstCmd->rectSrchWindow.width) > pstCmd->matInputImg.cols ||
        (pstCmd->rectSrchWindow.y + pstCmd->rectSrchWindow.height) > pstCmd->matInputImg.rows) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The search window (%d, %d, %d, %d) is invalid.",
            pstCmd->rectSrchWindow.x, pstCmd->rectSrchWindow.y, pstCmd->rectSrchWindow.width, pstCmd->rectSrchWindow.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    if (pstCmd->nRecordId <= 0) {
        char chArrMsg[100];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The input record id %d is invalid.", pstCmd->nRecordId);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    TmplRecordPtr ptrRecord = std::static_pointer_cast<TmplRecord> (RecordManager::getInstance()->get(pstCmd->nRecordId));
    if (nullptr == ptrRecord) {
        char chArrMsg[100];
        _snprintf(chArrMsg, sizeof (chArrMsg), "Failed to get record ID %d in system.", pstCmd->nRecordId);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (ptrRecord->getTmpl().rows > pstCmd->rectSrchWindow.height || ptrRecord->getTmpl().cols > pstCmd->rectSrchWindow.width) {
        WriteLog("The template is bigger than search window.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (ptrRecord->getAlgorithm() != pstCmd->enAlgorithm) {
        char chArrMsg[100];
        _snprintf(chArrMsg, sizeof (chArrMsg), "The input algorithm %d is not match with record algorithm %d in system.", ToInt32(pstCmd->enAlgorithm), ToInt32(ptrRecord->getAlgorithm()));
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseMatchTmpl);

    cv::Mat matSrchROI(pstCmd->matInputImg, pstCmd->rectSrchWindow);
    if (matSrchROI.channels() > 1)
        cv::cvtColor(matSrchROI, matSrchROI, cv::COLOR_BGR2GRAY);

    cv::Mat matTmpl = ptrRecord->getTmpl();
    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Template Image", matTmpl);

    if (PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF == pstCmd->enAlgorithm) {
        float fCorrelation;
        pstRpy->enStatus = MatchTmpl::matchTemplate(matSrchROI, matTmpl, pstCmd->bSubPixelRefine, pstCmd->enMotion, pstRpy->ptObjPos, pstRpy->fRotation, fCorrelation, ptrRecord->getMask());
        pstRpy->ptObjPos.x += pstCmd->rectSrchWindow.x;
        pstRpy->ptObjPos.y += pstCmd->rectSrchWindow.y;
        pstRpy->fMatchScore = fCorrelation * ConstToPercentage;
    }
    else if (PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE == pstCmd->enAlgorithm) {
        cv::Point ptResult = MatchTmpl::matchByRecursionEdge(matSrchROI, matTmpl, ptrRecord->getEdgeMask());
        pstRpy->ptObjPos.x = ptResult.x + ptrRecord->getTmpl().cols / 2 + 0.5f + pstCmd->rectSrchWindow.x;
        pstRpy->ptObjPos.y = ptResult.y + ptrRecord->getTmpl().rows / 2 + 0.5f + pstCmd->rectSrchWindow.y;
        pstRpy->fRotation = 0.f;
        cv::Mat matImgROI(matSrchROI, cv::Rect(ptResult, matTmpl.size()));
        pstRpy->fMatchScore = MatchTmpl::calcCorrelation(matTmpl, matImgROI) * ConstToPercentage;
        pstRpy->enStatus = VisionStatus::OK;
    }
    else if (PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_AREA == pstCmd->enAlgorithm) {
        cv::Point ptResult = MatchTmpl::matchTemplateRecursive(matSrchROI, matTmpl);
        pstRpy->ptObjPos.x = ptResult.x + ptrRecord->getTmpl().cols / 2 + 0.5f + pstCmd->rectSrchWindow.x;
        pstRpy->ptObjPos.y = ptResult.y + ptrRecord->getTmpl().rows / 2 + 0.5f + pstCmd->rectSrchWindow.y;
        pstRpy->fRotation = 0.f;
        cv::Mat matImgROI(matSrchROI, cv::Rect(ptResult, matTmpl.size()));
        pstRpy->fMatchScore = MatchTmpl::calcCorrelation(matTmpl, matImgROI) * ConstToPercentage;
        pstRpy->enStatus = VisionStatus::OK;
    }

    if (pstRpy->fMatchScore <= pstCmd->fMinMatchScore)
        pstRpy->enStatus = VisionStatus::MATCH_SCORE_REJECT;

    if (!isAutoMode()) {
        if (pstCmd->matInputImg.channels() > 1)
            pstRpy->matResultImg = pstCmd->matInputImg.clone();
        else
            cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);

        cv::Mat matWarp = cv::getRotationMatrix2D(cv::Point(0, 0), -pstRpy->fRotation, 1.);
        float fCrossSize = 10.f;
        cv::Point2f crossLineOnePtOne(-fCrossSize, 0), crossLineOnePtTwo(fCrossSize, 0), crossLineTwoPtOne(0, -fCrossSize), crossLineTwoPtTwo(0, fCrossSize);
        crossLineOnePtOne = pstRpy->ptObjPos + CalcUtils::warpPoint<double>(matWarp, crossLineOnePtOne);
        crossLineOnePtTwo = pstRpy->ptObjPos + CalcUtils::warpPoint<double>(matWarp, crossLineOnePtTwo);
        crossLineTwoPtOne = pstRpy->ptObjPos + CalcUtils::warpPoint<double>(matWarp, crossLineTwoPtOne);
        crossLineTwoPtTwo = pstRpy->ptObjPos + CalcUtils::warpPoint<double>(matWarp, crossLineTwoPtTwo);

        int nLineWidth = 1;
        if (ptrRecord->getTmpl().total() > 1000)
            nLineWidth = 2;

        cv::line(pstRpy->matResultImg, crossLineOnePtOne, crossLineOnePtTwo, BLUE_SCALAR, nLineWidth);
        cv::line(pstRpy->matResultImg, crossLineTwoPtOne, crossLineTwoPtTwo, BLUE_SCALAR, nLineWidth);

        matWarp = cv::getRotationMatrix2D(pstRpy->ptObjPos, -pstRpy->fRotation, 1.);
        auto vecPoint2f = CalcUtils::warpRect<double>(matWarp, cv::Rect2f(pstRpy->ptObjPos.x - ToFloat(ptrRecord->getTmpl().cols / 2),
            pstRpy->ptObjPos.y - ToFloat(ptrRecord->getTmpl().rows / 2), ToFloat(ptrRecord->getTmpl().cols), ToFloat(ptrRecord->getTmpl().rows)));
        VectorOfVectorOfPoint vecVecPoint(1);
        for (const auto &point : vecPoint2f)
            vecVecPoint[0].push_back(point);
        cv::polylines(pstRpy->matResultImg, vecVecPoint, true, BLUE_SCALAR, nLineWidth);
        cv::rectangle(pstRpy->matResultImg, pstCmd->rectSrchWindow, GREEN_SCALAR, 1);
    }
    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

VisionStatus VisionAlgorithm::inspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy) {
    MARK_FUNCTION_START_TIME;
    if (NULL == pInspCmd || NULL == pInspRpy)
        return VisionStatus::INVALID_PARAM;

    cv::Mat matTmplROI(pInspCmd->matTmpl, pInspCmd->rectLrn);

    cv::Mat matTmplFullRange(pInspCmd->matInsp.size(), pInspCmd->matInsp.type());
    matTmplFullRange.setTo(cv::Scalar(0));

    cv::Mat matMask(pInspCmd->matInsp.size(), pInspCmd->matInsp.type());
    matMask.setTo(cv::Scalar(0));

    cv::Rect2f rect2fInspROI(pInspCmd->ptObjPos.x - pInspCmd->rectLrn.width / 2.f, pInspCmd->ptObjPos.y - pInspCmd->rectLrn.height / 2.f,
        pInspCmd->rectLrn.width, pInspCmd->rectLrn.height);
    cv::Mat matInspRoi(matTmplFullRange, rect2fInspROI);
    matTmplROI.copyTo(matInspRoi);

    cv::Mat matMaskRoi(matMask, rect2fInspROI);
    matMaskRoi.setTo(cv::Scalar(255, 255, 255));

    cv::Mat matRotation = cv::getRotationMatrix2D(pInspCmd->ptObjPos, -pInspCmd->fRotation, 1);
    cv::Mat matRotatedTmpl;
    cv::warpAffine(matTmplFullRange, matRotatedTmpl, matRotation, matTmplFullRange.size());

    cv::Mat matRotatedMask;
    cv::warpAffine(matMask, matRotatedMask, matRotation, matMask.size());
    matRotatedMask = cv::Scalar::all(255) - matRotatedMask;

    pInspCmd->matInsp.setTo(cv::Scalar(0), matRotatedMask);

    cv::Mat matCmpResult = pInspCmd->matInsp - matRotatedTmpl;
    cv::Mat matRevsCmdResult = matRotatedTmpl - pInspCmd->matInsp;

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        cv::imshow("Mask", matRotatedMask);
        cv::imshow("Masked Inspection", pInspCmd->matInsp);
        cv::imshow("Rotated template", matRotatedTmpl);
        cv::imshow("Compare result", matCmpResult);
        cv::imshow("Compare result Reverse", matRevsCmdResult);
        cv::waitKey(0);
    }

    pInspRpy->n16NDefect = 0;
    _findBlob(matCmpResult, matRevsCmdResult, pInspCmd, pInspRpy);
    _findLine(matCmpResult, pInspCmd, pInspRpy);
    MARK_FUNCTION_END_TIME;
    return VisionStatus::OK;
}

int VisionAlgorithm::_findBlob(const cv::Mat &mat, const cv::Mat &matRevs, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy) {
    cv::Mat im_with_keypoints(mat);
    mat.copyTo(im_with_keypoints);

    for (short nIndex = 0; nIndex < pInspCmd->u16NumOfDefectCriteria; ++nIndex) {
        PR_DefectCriteria stDefectCriteria = pInspCmd->astDefectCriteria[nIndex];
        if (stDefectCriteria.fArea <= 0.f)
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
        if (PR_DEFECT_ATTRIBUTE::BRIGHT == stDefectCriteria.enAttribute) {
            vecInput.push_back(mat);
        }
        else if (PR_DEFECT_ATTRIBUTE::DARK == stDefectCriteria.enAttribute) {
            vecInput.push_back(matRevs);
        }
        else if (PR_DEFECT_ATTRIBUTE::BOTH == stDefectCriteria.enAttribute) {
            vecInput.push_back(mat);
            vecInput.push_back(matRevs);
        }

        VectorOfVectorKeyPoint keyPoints;
        detector->detect(vecInput, keyPoints, pInspCmd->matMask);

        //VectorOfKeyPoint vecKeyPoint = keyPoints[0];
        for (const auto &vecKeyPoint : keyPoints) {
            if (vecKeyPoint.size() > 0 && pInspRpy->n16NDefect < MAX_NUM_OF_DEFECT_RESULT) {
                for (const auto &keyPoint : vecKeyPoint) {
                    pInspRpy->astDefect[pInspRpy->n16NDefect].enType = PR_DEFECT_TYPE::BLOB;
                    float fRadius = keyPoint.size / 2;
                    pInspRpy->astDefect[pInspRpy->n16NDefect].fRadius = fRadius;
                    pInspRpy->astDefect[pInspRpy->n16NDefect].fArea = (float)CV_PI * fRadius * fRadius; //Area = PI * R * R
                    ++ pInspRpy->n16NDefect;
                }
                drawKeypoints(im_with_keypoints, vecKeyPoint, im_with_keypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            }
        }
    }

    cv::imshow("Find blob result", im_with_keypoints);
    cv::waitKey(0);
    return 0;
}

int VisionAlgorithm::_findLine(const cv::Mat &mat, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy) {
    MARK_FUNCTION_START_TIME;
    for (short nIndex = 0; nIndex < pInspCmd->u16NumOfDefectCriteria; ++nIndex) {
        PR_DefectCriteria stDefectCriteria = pInspCmd->astDefectCriteria[nIndex];
        if (stDefectCriteria.fLength <= 0.f)
            continue;

        cv::Mat matThreshold;
        cv::threshold(mat, matThreshold, stDefectCriteria.ubContrast, 255, cv::THRESH_BINARY);

        cv::Mat matSobelX, matSobelY;
        Sobel(matThreshold, matSobelX, mat.depth(), 1, 0);
        Sobel(matThreshold, matSobelY, mat.depth(), 0, 1);

        short depth = matSobelX.depth();

        cv::Mat matSobelResult(mat.size(), CV_8U);
        for (short row = 0; row < mat.rows; ++row) {
            for (short col = 0; col < mat.cols; ++col) {
                unsigned char x = (matSobelX.at<unsigned char>(row, col));
                unsigned char y = (matSobelY.at<unsigned char>(row, col));
                matSobelResult.at<unsigned char>(row, col) = x + y;
            }
        }

        if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
            cv::imshow("Sobel Result", matSobelResult);
            cv::waitKey(0);
        }

        cv::Mat color_dst;
        cv::cvtColor(matSobelResult, color_dst, CV_GRAY2BGR);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(matSobelResult, lines, 1, CV_PI / 180, 30, stDefectCriteria.fLength, 10);

        //_mergeLines ( )
        std::vector<PR_Line2f> vecLines, vecLinesAfterMerge;
        for (size_t i = 0; i < lines.size(); i++) {
            cv::Point2f pt1((float)lines[i][0], (float)lines[i][1]);
            cv::Point2f pt2((float)lines[i][2], (float)lines[i][3]);
            PR_Line2f line(pt1, pt2);
            vecLines.push_back(line);
        }
        _mergeLines(vecLines, vecLinesAfterMerge);
        for (const auto it : vecLinesAfterMerge) {
            line(color_dst, it.pt1, it.pt2, cv::Scalar(0, 0, 255), 3, 8);
            if (pInspRpy->n16NDefect < MAX_NUM_OF_DEFECT_RESULT) {
                pInspRpy->astDefect[pInspRpy->n16NDefect].enType = PR_DEFECT_TYPE::LINE;
                pInspRpy->astDefect[pInspRpy->n16NDefect].fLength = CalcUtils::distanceOf2Point<float>(it.pt1, it.pt2);
                ++pInspRpy->n16NDefect;
            }
        }
        if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
            showImage("Detected Lines", color_dst);
    }
    MARK_FUNCTION_END_TIME;
    return 0;
}

/*static*/ int VisionAlgorithm::_merge2Line(const PR_Line2f &line1, const PR_Line2f &line2, PR_Line2f &lineResult) {
    float fLineSlope1 = (line1.pt2.y - line1.pt1.y) / (line1.pt2.x - line1.pt1.x);
    float fLineSlope2 = (line2.pt2.y - line2.pt1.y) / (line2.pt2.x - line2.pt1.x);

    if ((fabs(fLineSlope1) < 10.f || fabs(fLineSlope2) < 10.f)
        && fabs(fLineSlope1 - fLineSlope2) > PARALLEL_LINE_SLOPE_DIFF_LMT)
        return -1;
    else if ((fabs(fLineSlope1) > 10.f || fabs(fLineSlope2) > 10.f)
        && fabs(1 / fLineSlope1 - 1 / fLineSlope2) > PARALLEL_LINE_SLOPE_DIFF_LMT)
        return -1;

    //Find the intersection point with Y axis.
    float fLineCrossWithY1 = fLineSlope1 * (- line1.pt1.x) + line1.pt1.y;
    float fLineCrossWithY2 = fLineSlope2 * (- line2.pt1.x) + line2.pt1.y;

    float fPerpendicularLineSlope1 = - 1.f / fLineSlope1;
    float fPerpendicularLineSlope2 = - 1.f / fLineSlope2;
    float fDistanceOfLine = 0.f;
    const float HORIZONTAL_LINE_SLOPE = 0.1f;
    if (fabs(fPerpendicularLineSlope1) < HORIZONTAL_LINE_SLOPE || fabs(fPerpendicularLineSlope2) < HORIZONTAL_LINE_SLOPE)
        fDistanceOfLine = fabs((line1.pt1.x + line1.pt2.x) / 2.f - (line2.pt1.x + line2.pt2.x) / 2.f);
    else if (fabs(fPerpendicularLineSlope1) > 100 || fabs(fPerpendicularLineSlope2) > 100)
        fDistanceOfLine = fabs((line1.pt1.y + line1.pt2.y) / 2.f - (line2.pt1.y + line2.pt2.y) / 2.f);
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

    if (fDistanceOfLine > PARALLEL_LINE_MERGE_DIST_LMT)
        return -1;

    std::vector<cv::Point> vecPoint;
    vecPoint.push_back(line1.pt1);
    vecPoint.push_back(line1.pt2);
    vecPoint.push_back(line2.pt1);
    vecPoint.push_back(line2.pt2);
    //{line1.pt1, line1.pt2, line2.pt1, line2.pt2 }
    if (fabs(fLineSlope1) < HORIZONTAL_LINE_SLOPE && fabs(fLineSlope2) < HORIZONTAL_LINE_SLOPE) {
        lineResult.pt1 = *std::min_element(vecPoint.begin(), vecPoint.end(), [](const cv::Point &pt1, const cv::Point &pt2) { return pt1.x < pt2.x; });
        lineResult.pt2 = *std::max_element(vecPoint.begin(), vecPoint.end(), [](const cv::Point &pt1, const cv::Point &pt2) { return pt1.x < pt2.x; });
        float fAverageY = (line1.pt1.y + line1.pt2.y + line2.pt1.y + line2.pt2.y) / 4.f;
        lineResult.pt1.y = fAverageY;
        lineResult.pt2.y = fAverageY;
        return 0;
    }

    std::vector<float> vecPerpendicularLineB;
    //Find the expression of perpendicular lines y = a * x + b
    float fPerpendicularLineB1 = line1.pt1.y - fPerpendicularLineSlope1 * line1.pt1.x;
    vecPerpendicularLineB.push_back(fPerpendicularLineB1);
    float fPerpendicularLineB2 = line1.pt2.y - fPerpendicularLineSlope1 * line1.pt2.x;
    vecPerpendicularLineB.push_back(fPerpendicularLineB2);
    float fPerpendicularLineB3 = line2.pt1.y - fPerpendicularLineSlope2 * line2.pt1.x;
    vecPerpendicularLineB.push_back(fPerpendicularLineB3);
    float fPerpendicularLineB4 = line2.pt2.y - fPerpendicularLineSlope2 * line2.pt2.x;
    vecPerpendicularLineB.push_back(fPerpendicularLineB4);

    size_t nMaxIndex = 0, nMinIndex = 0;
    float fMaxB = fPerpendicularLineB1;
    float fMinB = fPerpendicularLineB1;
    for (size_t i = 1; i < vecPerpendicularLineB.size(); ++ i) {
        if (vecPerpendicularLineB[i] > fMaxB) {
            fMaxB = vecPerpendicularLineB[i];
            nMaxIndex = i;
        }

        if (vecPerpendicularLineB[i] < fMinB) {
            fMinB = vecPerpendicularLineB[i];
            nMinIndex = i;
        }
    }
    lineResult.pt1 = vecPoint[nMinIndex];
    lineResult.pt2 = vecPoint[nMaxIndex];
    return static_cast<int>(VisionStatus::OK);
}

/*static*/ int VisionAlgorithm::_mergeLines(const std::vector<PR_Line2f> &vecLines, std::vector<PR_Line2f> &vecResultLines) {
    if (vecLines.size() < 2) {
        vecResultLines = vecLines;
        return 0;
    }

    std::vector<int> vecMerged(vecLines.size(), false);

    for (size_t i = 0; i < vecLines.size(); ++ i) {
        if (vecMerged[i])
            continue;

        PR_Line2f stLineResult = vecLines[i];
        for (size_t j = i + 1; j < vecLines.size(); ++ j) {
            if (! vecMerged[j]) {
                if (0 == _merge2Line(stLineResult, vecLines[j], stLineResult)) {
                    vecMerged[i] = vecMerged[j] = true;
                }
            }
        }
        vecResultLines.push_back(stLineResult);
    }
    return 0;
}

/*static*/ VisionStatus VisionAlgorithm::_writeLrnObjRecord(PR_LRN_OBJ_RPY *const pstRpy) {
    ObjRecordPtr ptrRecord = std::make_shared<ObjRecord>(PR_RECORD_TYPE::OBJECT);
    ptrRecord->setModelKeyPoint(pstRpy->vecKeyPoint);
    ptrRecord->setModelDescriptor(pstRpy->matDescritor);
    ptrRecord->setObjCenter(cv::Point2f(ToFloat(pstRpy->matTmpl.cols) / 2.f, ToFloat(pstRpy->matTmpl.rows) / 2.f));
    RecordManager::getInstance()->add(ptrRecord, pstRpy->nRecordId);
    return VisionStatus::OK;
}

VisionStatus VisionAlgorithm::_writeDeviceRecord(PR_LRN_DEVICE_RPY *pLrnDeviceRpy) {
    DeviceRecordPtr ptrRecord = std::make_shared<DeviceRecord>(PR_RECORD_TYPE::DEVICE);
    ptrRecord->setElectrodeThreshold(pLrnDeviceRpy->nElectrodeThreshold);
    ptrRecord->setSize(pLrnDeviceRpy->sizeDevice);
    RecordManager::getInstance()->add(ptrRecord, pLrnDeviceRpy->nRecordId);
    return VisionStatus::OK;
}

/*static*/ void VisionAlgorithm::showImage(String windowName, const cv::Mat &mat) {
    static Int32 nImageCount = 0;
    String strWindowName = windowName + "_" + std::to_string(nImageCount ++);
    cv::namedWindow(strWindowName);
    cv::imshow(strWindowName, mat);
    cv::waitKey(0);
    cv::destroyWindow(strWindowName);
}

/*static*/ VisionStatus VisionAlgorithm::_checkInputROI(const cv::Rect &rect, const cv::Mat &matInputImg, const char *filename, int line) {
    if (rect.x < 0 || rect.y < 0 ||
        rect.width <= 0 || rect.height <= 0 ||
        (rect.x + rect.width)  > matInputImg.cols ||
        (rect.y + rect.height) > matInputImg.rows) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The input ROI rect (%d, %d, %d, %d) is invalid.",
            rect.x, rect.y, rect.width, rect.height);
        Log::GetInstance()->Write(chArrMsg, filename, line);
        return VisionStatus::INVALID_PARAM;
    }
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_checkMask(const cv::Mat &matInput, const cv::Mat &matMask, const char *filename, int line) {
    if (matMask.cols != matInput.cols || matMask.rows != matInput.rows) {
        std::stringstream ss;
        ss << "The size of mask " << matMask.size() << " doesn't match with ROI size " << matInput.size();
        Log::GetInstance()->Write(ss.str(), filename, line);
        return VisionStatus::INVALID_PARAM;
    }

    if (matMask.type() != CV_8UC1) {
        WriteLog("The mask must be gray image!");
        return VisionStatus::INVALID_PARAM;
    }
    return VisionStatus::OK;
}

/*static*/ LogCasePtr VisionAlgorithm::_createLogCaseInstance(const String &strFolderPrefix, const String &strLocalPath) {
    if (LogCaseLrnObj::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseLrnObj>(strLocalPath, true);

    if (LogCaseSrchObj::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique <LogCaseSrchObj>(strLocalPath, true);

    if (LogCaseFitCircle::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique <LogCaseFitCircle>(strLocalPath, true);

    if (LogCaseFindCircle::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique <LogCaseFindCircle>(strLocalPath, true);

    if (LogCaseFitLine::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseFitLine>(strLocalPath, true);

    if (LogCaseFindLine::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseFindLine>(strLocalPath, true);

    if (LogCaseFitParallelLine::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseFitParallelLine>(strLocalPath, true);

    if (LogCaseFitRect::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseFitRect>(strLocalPath, true);

    if (LogCaseFindEdge::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseFindEdge>(strLocalPath, true);

    if (LogCaseSrchFiducial::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseSrchFiducial>(strLocalPath, true);

    if (LogCaseOcr::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseOcr>(strLocalPath, true);

    if (LogCaseRemoveCC::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseRemoveCC>(strLocalPath, true);

    if (LogCaseDetectEdge::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseDetectEdge>(strLocalPath, true);

    if (LogCaseInspCircle::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseInspCircle>(strLocalPath, true);

    if (LogCaseAutoThreshold::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseAutoThreshold>(strLocalPath, true);

    if (LogCaseFillHole::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseFillHole>(strLocalPath, true);

    if (LogCaseLrnTmpl::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseLrnTmpl>(strLocalPath, true);

    if (LogCaseMatchTmpl::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseMatchTmpl>(strLocalPath, true);

    if (LogCasePickColor::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCasePickColor>(strLocalPath, true);

    if (LogCaseCalibrateCamera::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseCalibrateCamera>(strLocalPath, true);

    if (LogCaseRestoreImg::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseRestoreImg>(strLocalPath, true);

    if (LogCaseAutoLocateLead::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseAutoLocateLead>(strLocalPath, true);

    if (LogCaseInspBridge::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseInspBridge>(strLocalPath, true);

    if (LogCaseInspChip::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseInspChip>(strLocalPath, true);

    if (LogCaseLrnContour::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseLrnContour>(strLocalPath, true);

    if (LogCaseInspContour::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseInspContour>(strLocalPath, true);

    if (LogCaseInspHole::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseInspHole>(strLocalPath, true);

    if (LogCaseInspLead::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseInspLead>(strLocalPath, true);

    if (LogCaseInspLeadTmpl::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseInspLeadTmpl>(strLocalPath, true);

    if (LogCaseInspPolarity::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseInspPolarity>(strLocalPath, true);

    if (LogCaseCalib3DBase::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseCalib3DBase>(strLocalPath, true);

    if (LogCaseCalib3DHeight::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseCalib3DHeight>(strLocalPath, true);

    if (LogCaseCalc3DHeight::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseCalc3DHeight>(strLocalPath, true);

    if (LogCaseCalcCameraMTF::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseCalcCameraMTF>(strLocalPath, true);

    if (LogCaseLrnOcv::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseLrnOcv>(strLocalPath, true);

    if (LogCaseOcv::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseOcv>(strLocalPath, true);

    if (LogCase2DCode::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCase2DCode>(strLocalPath, true);

    if (LogCaseInsp3DSolder::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseInsp3DSolder>(strLocalPath, true);

    if (LogCaseCalc3DHeightDiff::StaticGetFolderPrefix() == strFolderPrefix)
        return std::make_unique<LogCaseCalc3DHeightDiff>(strLocalPath, true);

    static String msg = strFolderPrefix + " is not handled in " + __FUNCTION__;
    throw std::exception(msg.c_str());
}

/*static*/ VisionStatus VisionAlgorithm::runLogCase(const String &strFilePath) {
    if (strFilePath.length() < 2)
        return VisionStatus::INVALID_PARAM;

    if (! bfs::exists(strFilePath))
        return VisionStatus::PATH_NOT_EXIST;

    VisionStatus enStatus = VisionStatus::OK;
    auto strLocalPath = LogCase::unzip(strFilePath);

    char chFolderToken = '\\';
    if (strLocalPath.find('/') != std::string::npos)
        chFolderToken = '/';

    if (strLocalPath[strLocalPath.length() - 1] != chFolderToken)
        strLocalPath.append(std::string{chFolderToken});

    auto pos = strLocalPath.rfind(chFolderToken, strLocalPath.length() - 2);
    if (String::npos == pos)
        return VisionStatus::INVALID_PARAM;

    auto folderName = strLocalPath.substr(pos + 1);
    pos = folderName.find('_');
    auto strFolderPrefix = folderName.substr(0, pos);

    auto ptrLogCase = _createLogCaseInstance(strFolderPrefix, strLocalPath);
    assert(ptrLogCase->GetFolderPrefix() == strFolderPrefix);

    if (nullptr != ptrLogCase)
        enStatus = ptrLogCase->RunLogCase();
    else
        enStatus = VisionStatus::INVALID_LOGCASE;
    return enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::srchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    cv::Point2f ptResult;

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectSrchWindow.x < 0 || pstCmd->rectSrchWindow.y < 0 ||
        pstCmd->rectSrchWindow.width <= 0 || pstCmd->rectSrchWindow.height <= 0 ||
        (pstCmd->rectSrchWindow.x + pstCmd->rectSrchWindow.width) > pstCmd->matInputImg.cols ||
        (pstCmd->rectSrchWindow.y + pstCmd->rectSrchWindow.height) > pstCmd->matInputImg.rows) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "Search window is invalid, rect x = %d, y = %d, width = %d, height = %d.",
            pstCmd->rectSrchWindow.x, pstCmd->rectSrchWindow.y, pstCmd->rectSrchWindow.width, pstCmd->rectSrchWindow.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    auto nTmplSize = ToInt32(pstCmd->fSize + pstCmd->fMargin * 2);
    if (nTmplSize >= pstCmd->rectSrchWindow.width || nTmplSize >= pstCmd->rectSrchWindow.height) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "Generated template size %d is over search window size [%d, %d].",
            nTmplSize, pstCmd->rectSrchWindow.width, pstCmd->rectSrchWindow.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseSrchFiducial);

    cv::Mat matTmpl = cv::Mat::zeros(nTmplSize, nTmplSize, CV_8UC1);
    if (PR_FIDUCIAL_MARK_TYPE::SQUARE == pstCmd->enType) {
        cv::rectangle(matTmpl,
            cv::Rect2f(pstCmd->fMargin, pstCmd->fMargin, pstCmd->fSize, pstCmd->fSize),
            cv::Scalar::all(256),
            CV_FILLED);
    }
    else if (PR_FIDUCIAL_MARK_TYPE::CIRCLE == pstCmd->enType) {
        auto radius = ToInt32(pstCmd->fSize / 2.f);
        cv::circle(matTmpl, cv::Point(ToInt32(pstCmd->fMargin) + radius, ToInt32(pstCmd->fMargin) + radius), radius, cv::Scalar::all(256), CV_FILLED);
    }
    else {
        WriteLog("Fiducial mark type is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    cv::Mat matSrchROI(pstCmd->matInputImg, pstCmd->rectSrchWindow);
    if (matSrchROI.channels() > 1)
        cv::cvtColor(matSrchROI, matSrchROI, cv::COLOR_BGR2GRAY);

    float fRotation = 0.f, fCorrelation = 0.f;
    pstRpy->enStatus = MatchTmpl::matchTemplate(matSrchROI, matTmpl, true, PR_OBJECT_MOTION::TRANSLATION, ptResult, fRotation, fCorrelation);

    pstRpy->ptPos.x = ptResult.x + pstCmd->rectSrchWindow.x;
    pstRpy->ptPos.y = ptResult.y + pstCmd->rectSrchWindow.y;
    pstRpy->fMatchScore = fCorrelation * ConstToPercentage;

    if (pstRpy->fMatchScore <= pstCmd->fMinMatchScore)
        pstRpy->enStatus = VisionStatus::MATCH_SCORE_REJECT;

    if (! isAutoMode()) {
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
        cv::circle(pstRpy->matResultImg, pstRpy->ptPos, 2, BLUE_SCALAR, 2);
        cv::rectangle(pstRpy->matResultImg, cv::Rect(ToInt32(pstRpy->ptPos.x) - nTmplSize / 2, ToInt32(pstRpy->ptPos.y) - nTmplSize / 2, nTmplSize, nTmplSize), BLUE_SCALAR, 2);
        cv::rectangle(pstRpy->matResultImg, pstCmd->rectSrchWindow, YELLOW_SCALAR, 2);
    }

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VectorOfPoint VisionAlgorithm::_findPointInRegionOverThreshold(const cv::Mat &mat, const cv::Rect &rect, int nThreshold) {
    VectorOfPoint vecPoint;
    if (mat.empty())
        return vecPoint;

    cv::Mat matROI(mat, rect);
    cv::Mat matThreshold;

    cv::threshold(matROI, matThreshold, nThreshold, 256, CV_THRESH_BINARY);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("threshold result", matThreshold);
    VectorOfPoint vecPoints;
    vecPoints.reserve(100000);
    cv::findNonZero(matThreshold, vecPoints);
    for (const auto &point : vecPoints) {
        cv::Point point2f(point.x + rect.x, point.y + rect.y);
        vecPoint.push_back(point2f);
    }
    return vecPoint;
}

/*static*/ ListOfPoint VisionAlgorithm::_findPointsInRegionByThreshold(const cv::Mat &mat, const cv::Rect &rect, int nThreshold, PR_OBJECT_ATTRIBUTE enAttribute) {
    ListOfPoint listPoint;
    if (mat.empty())
        return listPoint;

    cv::Mat matROI(mat, rect);
    cv::Mat matThreshold;

    cv::ThresholdTypes enThresType = cv::THRESH_BINARY;
    if (PR_OBJECT_ATTRIBUTE::DARK == enAttribute)
        enThresType = cv::THRESH_BINARY_INV;
    cv::threshold(matROI, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, enThresType);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("threshold result", matThreshold);
    std::vector<cv::Point> vecPoints;
    vecPoints.reserve(100000);
    cv::findNonZero(matThreshold, vecPoints);
    for (const auto &point : vecPoints) {
        cv::Point point2f(point.x + rect.x, point.y + rect.y);
        listPoint.push_back(point2f);
    }
    return listPoint;
}

/*static*/ ListOfPoint VisionAlgorithm::_findPointsInRegion(const cv::Mat &mat, const cv::Rect &rect) {
    ListOfPoint listPoint;
    if (mat.empty())
        return listPoint;

    cv::Mat matROI(mat, rect);
    cv::Mat matThreshold;

    std::vector<cv::Point> vecPoints;
    vecPoints.reserve(100000);
    cv::findNonZero(matThreshold, vecPoints);
    for (const auto &point : vecPoints) {
        cv::Point point2f(point.x + rect.x, point.y + rect.y);
        listPoint.push_back(point2f);
    }
    return listPoint;
}

/*static*/ std::vector<size_t> VisionAlgorithm::_findPointOverLineTol(const VectorOfPoint   &vecPoint,
    bool                   bReversedFit,
    const float            fSlope,
    const float            fIntercept,
    PR_RM_FIT_NOISE_METHOD method,
    float                  tolerance) {
    std::vector<size_t> vecResult;
    for (size_t i = 0; i < vecPoint.size(); ++ i) {
        cv::Point2f point = vecPoint[i];
        auto disToLine = CalcUtils::ptDisToLine(point, bReversedFit, fSlope, fIntercept);
        if (PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR == method && fabs(disToLine) > tolerance) {
            vecResult.push_back(i);
        }
        else if (PR_RM_FIT_NOISE_METHOD::POSITIVE_ERR == method && disToLine > tolerance) {
            vecResult.push_back(i);
        }
        else if (PR_RM_FIT_NOISE_METHOD::NEGATIVE_ERR == method && disToLine < -tolerance) {
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
    float                    tolerance) {
    std::vector<ListOfPoint::const_iterator> vecResult;
    for (ListOfPoint::const_iterator it = listPoint.begin(); it != listPoint.end(); ++ it) {
        cv::Point2f point = *it;
        auto disToLine = CalcUtils::ptDisToLine(point, bReversedFit, fSlope, fIntercept);
        if (PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR == method && fabs(disToLine) > tolerance) {
            vecResult.push_back(it);
        }
        else if (PR_RM_FIT_NOISE_METHOD::POSITIVE_ERR == method && disToLine > tolerance) {
            vecResult.push_back(it);
        }
        else if (PR_RM_FIT_NOISE_METHOD::NEGATIVE_ERR == method && disToLine < -tolerance) {
            vecResult.push_back(it);
        }
    }
    return vecResult;
}

/*static*/ VisionStatus VisionAlgorithm::fitLine(const PR_FIT_LINE_CMD *const pstCmd, PR_FIT_LINE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char chArrMsg[1000];

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (! pstCmd->matMask.empty()) {
        if (pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d )",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matMask.channels() != 1) {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFitLine);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    cv::Mat matGray, matThreshold;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI.clone();

    if (pstCmd->bPreprocessed) {
        matThreshold = matGray;
    }
    else {
        cv::ThresholdTypes enThresType = cv::THRESH_BINARY;
        if (PR_OBJECT_ATTRIBUTE::DARK == pstCmd->enAttribute)
            enThresType = cv::THRESH_BINARY_INV;

        cv::threshold(matGray, matThreshold, pstCmd->nThreshold, PR_MAX_GRAY_LEVEL, enThresType);
        CV_Assert(matThreshold.channels() == 1);

        if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
            showImage("Threshold image", matThreshold);
    }

    if (! pstCmd->matMask.empty()) {
        cv::Mat matROIMask(pstCmd->matMask, pstCmd->rectROI);
        matROIMask = cv::Scalar(255) - matROIMask;
        matThreshold.setTo(cv::Scalar(0), matROIMask);
    }

    VectorOfPoint vecPoints;
    cv::findNonZero(matThreshold, vecPoints);
    if (vecPoints.size() < 2) {
        WriteLog("Not enough points to fit line");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    if (vecPoints.size() > PR_FIT_LINE_MAX_POINT_COUNT) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "Too much noise to fit line, points number %d", vecPoints.size());
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    std::vector<int> vecX, vecY;
    for (auto &point : vecPoints) {
        vecX.push_back(point.x);
        vecY.push_back(point.y);
        point = cv::Point(point.x + pstCmd->rectROI.x, point.y + pstCmd->rectROI.y);
    }

    //For y = kx + b, the k is the slope, when it is very large, the error is quite big, so change
    //to get the x = k1 + b1, and get the k = 1 / k1, b = -b1 / k1. Then the result is more accurate.
    pstRpy->bReversedFit = false;
    if (CalcUtils::calcStdDeviation(vecY) > CalcUtils::calcStdDeviation(vecX))
        pstRpy->bReversedFit = true;

    if (PR_FIT_METHOD::RANSAC == pstCmd->enMethod) {
        Fitting::fitLineRansac(vecPoints, pstCmd->fErrTol, pstCmd->nMaxRansacTime, pstCmd->nNumOfPtToFinishRansac, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->stLine);
    }
    else if (PR_FIT_METHOD::LEAST_SQUARE_REFINE == pstCmd->enMethod) {
        pstRpy->enStatus = Fitting::fitLineRefine(vecPoints, pstCmd->enRmNoiseMethod, pstCmd->fErrTol, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->stLine);
        if (VisionStatus::OK != pstRpy->enStatus) {
            FINISH_LOGCASE;
            return pstRpy->enStatus;
        }
    }
    else if (PR_FIT_METHOD::LEAST_SQUARE == pstCmd->enMethod) {
        Fitting::fitLine(vecPoints, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->bReversedFit);
        pstRpy->stLine = CalcUtils::calcEndPointOfLine(vecPoints, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept);
    }

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::line(pstRpy->matResultImg, pstRpy->stLine.pt1, pstRpy->stLine.pt2, cv::Scalar(255, 0, 0), 2);
    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::fitLineByPoint(const PR_FIT_LINE_BY_POINT_CMD *const pstCmd, PR_FIT_LINE_BY_POINT_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecPoints.size() < 2) {
        WriteLog("The count of input points is less than 2.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->vecPoints.size() > PR_FIT_LINE_MAX_POINT_COUNT) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The count input points %d is more than limit 10000.", pstCmd->vecPoints.size());
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    VectorOfFloat vecX, vecY;
    for (auto &point : pstCmd->vecPoints) {
        vecX.push_back(point.x);
        vecY.push_back(point.y);
    }

    //For y = kx + b, the k is the slope, when it is very large, the error is quite big, so change
    //to get the x = k1 + b1, and get the k = 1 / k1, b = -b1 / k1. Then the result is more accurate.
    pstRpy->bReversedFit = false;
    if (CalcUtils::calcStdDeviation(vecY) > CalcUtils::calcStdDeviation(vecX))
        pstRpy->bReversedFit = true;

    if (PR_FIT_METHOD::RANSAC == pstCmd->enMethod) {
        Fitting::fitLineRansac(pstCmd->vecPoints, pstCmd->fErrTol, pstCmd->nMaxRansacTime, pstCmd->nNumOfPtToFinishRansac, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->stLine);
    }
    else if (PR_FIT_METHOD::LEAST_SQUARE_REFINE == pstCmd->enMethod) {
        pstRpy->enStatus = Fitting::fitLineRefine(pstCmd->vecPoints, pstCmd->enRmNoiseMethod, pstCmd->fErrTol, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->stLine);
    }
    else if (PR_FIT_METHOD::LEAST_SQUARE == pstCmd->enMethod) {
        Fitting::fitLine(pstCmd->vecPoints, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->bReversedFit);
        pstRpy->stLine = CalcUtils::calcEndPointOfLine(pstCmd->vecPoints, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept);
    }

    pstRpy->enStatus = VisionStatus::OK;

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VectorOfPoint VisionAlgorithm::_findPointInLineTol(const VectorOfPoint   &vecPoint,
    bool                   bReversedFit,
    const float            fSlope,
    const float            fIntercept,
    float                  fTolerance) {
    VectorOfPoint vecPointInTol;
    for (const auto &point : vecPoint) {
        auto disToLine = CalcUtils::ptDisToLine(point, bReversedFit, fSlope, fIntercept);
        if (fabs(disToLine) <= fTolerance)
            vecPointInTol.push_back(point);
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

    while (nRansacTime < nMaxRansacTime) {
        VectorOfPoint vecSelectedPoints = Fitting::randomSelectPoints(vecPoints, LINE_RANSAC_POINT);
        Fitting::fitLine(vecSelectedPoints, fSlope, fIntercept, bReversedFit);
        vecSelectedPoints = _findPointInLineTol(vecPoints, bReversedFit, fSlope, fIntercept, fTolerance);

        if (vecSelectedPoints.size() >= nFinishThreshold) {
            Fitting::fitLine(vecSelectedPoints, fSlope, fIntercept, bReversedFit);
            stLine = CalcUtils::calcEndPointOfLine(vecSelectedPoints, bReversedFit, fSlope, fIntercept);
            return;
        }

        if (vecSelectedPoints.size() > nMaxConsentNum) {
            Fitting::fitLine(vecSelectedPoints, fSlope, fIntercept, bReversedFit);
            stLine = CalcUtils::calcEndPointOfLine(vecSelectedPoints, bReversedFit, fSlope, fIntercept);
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
    PR_Line2f              &stLine) {
    ListOfPoint listPoint;
    for (const auto &point : vecPoints)
        listPoint.push_back(point);

    int nIteratorNum = 0;
    std::vector<ListOfPoint::const_iterator> vecOverTolIter;
    do {
        for (const auto &it : vecOverTolIter)
            listPoint.erase(it);

        Fitting::fitLine(listPoint, fSlope, fIntercept, bReversedFit);
        vecOverTolIter = _findPointOverLineTol(listPoint, bReversedFit, fSlope, fIntercept, enRmNoiseMethod, fTolerance);
    } while (! vecOverTolIter.empty() && nIteratorNum < 20);

    if (listPoint.size() < 2)
        return VisionStatus::TOO_MUCH_NOISE_TO_FIT;

    stLine = CalcUtils::calcEndPointOfLine(listPoint, bReversedFit, fSlope, fIntercept);
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::findLine(const PR_FIND_LINE_CMD *const pstCmd, PR_FIND_LINE_RPY *const pstRpy, bool bReplay /* = false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectRotatedROI.size.width <= 0 || pstCmd->rectRotatedROI.size.height <= 0) {
        WriteLog("The findLine ROI size is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (! pstCmd->matMask.empty()) {
        if (pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d ).",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matMask.channels() != 1) {
            WriteLog("The mask must be single channel image.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (PR_FIND_LINE_ALGORITHM::CALIPER == pstCmd->enAlgorithm) {
        if (pstCmd->nCaliperCount < 2) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof (chArrMsg), "The caliper count %d is less than 2.", pstCmd->nCaliperCount);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->fCaliperWidth < 3) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof (chArrMsg), "The caliper width %f is less than 3.", pstCmd->fCaliperWidth);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstCmd->bFindPair && PR_FIND_LINE_ALGORITHM::CALIPER != pstCmd->enAlgorithm) {
        WriteLog("Find line pair can only use caliper algorithm.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->bCheckLinearity && (pstCmd->fPointMaxOffset <= 0.f || pstCmd->fMinLinearity <= 0.f)) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "Linerity check parameter is invalid. PointMaxOffset = %.2f, MinLinerity = %.2f",
            pstCmd->fPointMaxOffset, pstCmd->fMinLinearity);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFindLine);

    cv::Mat matROI, matROIGray, matROIMask;
    cv::Rect2f rectNewROI(pstCmd->rectRotatedROI.center.x - pstCmd->rectRotatedROI.size.width / 2.f, pstCmd->rectRotatedROI.center.y - pstCmd->rectRotatedROI.size.height / 2.f,
        pstCmd->rectRotatedROI.size.width, pstCmd->rectRotatedROI.size.height);
    pstRpy->enStatus = _extractRotatedROI(pstCmd->matInputImg, pstCmd->rectRotatedROI, matROI);
    if (pstRpy->enStatus != VisionStatus::OK)
        return pstRpy->enStatus;

    if (! pstCmd->matMask.empty())
        _extractRotatedROI(pstCmd->matMask, pstCmd->rectRotatedROI, matROIMask);

    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matROIGray, CV_BGR2GRAY);
    else
        matROIGray = matROI.clone();

    VectorOfPoint2f vecFitPoint1, vecFitPoint2;
    VectorOfRect vecSubRects;
    if (pstCmd->enAlgorithm == PR_FIND_LINE_ALGORITHM::PROJECTION)
        _findLineByProjection(matROIGray, matROIMask, rectNewROI, pstCmd->enDetectDir, vecFitPoint1, pstRpy);
    else if (pstCmd->enAlgorithm == PR_FIND_LINE_ALGORITHM::CALIPER)
        _findLineByCaliper(matROIGray, matROIMask, rectNewROI, pstCmd, vecFitPoint1, vecFitPoint2, vecSubRects, pstRpy);
    else {
        WriteLog("The findLine algorithm is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    VectorOfVectorOfPoint vecRotoatedSubRects;
    if (fabs(pstCmd->rectRotatedROI.angle) > 0.1) {
        cv::Mat matWarpBack = cv::getRotationMatrix2D(pstCmd->rectRotatedROI.center, -pstCmd->rectRotatedROI.angle, 1.);
        for (auto &point : vecFitPoint1)
            point = CalcUtils::warpPoint<double>(matWarpBack, point);

        for (auto &point : vecFitPoint2)
            point = CalcUtils::warpPoint<double>(matWarpBack, point);

        for (const auto &rect : vecSubRects)
            vecRotoatedSubRects.push_back(CalcUtils::warpRect<double>(matWarpBack, rect));
    }

    //Draw the result image.
    if (! isAutoMode()) {
        if (pstCmd->matInputImg.channels() > 1)
            pstRpy->matResultImg = pstCmd->matInputImg.clone();
        else
            cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);
        cv::line(pstRpy->matResultImg, pstRpy->stLine.pt1, pstRpy->stLine.pt2, GREEN_SCALAR, 2);
        if (pstCmd->bFindPair)
            cv::line(pstRpy->matResultImg, pstRpy->stLine2.pt1, pstRpy->stLine2.pt2, GREEN_SCALAR, 2);

        if (fabs(pstCmd->rectRotatedROI.angle) > 0.1)
            cv::polylines(pstRpy->matResultImg, vecRotoatedSubRects, true, CYAN_SCALAR, 1);
        else {
            for (const auto &rect : vecSubRects)
                cv::rectangle(pstRpy->matResultImg, rect, CYAN_SCALAR, 1);
        }
        for (const auto &point : vecFitPoint1)
            cv::circle(pstRpy->matResultImg, point, 3, BLUE_SCALAR, 1);

        for (const auto &point : vecFitPoint2)
            cv::circle(pstRpy->matResultImg, point, 3, BLUE_SCALAR, 1);
    }

    if (pstRpy->enStatus != VisionStatus::OK) {
        FINISH_LOGCASE_EX;
        return pstRpy->enStatus;
    }

    std::vector<float> vecX, vecY;
    for (const auto &point : vecFitPoint1) {
        vecX.push_back(point.x);
        vecY.push_back(point.y);
    }

    //For y = kx + b, the k is the slope, when it is very large, the error is quite big, so change
    //to get the x = k1 + b1, and get the k = 1 / k1, b = -b1 / k1. Then the result is more accurate.
    pstRpy->bReversedFit = false;
    if (CalcUtils::calcStdDeviation(vecY) > CalcUtils::calcStdDeviation(vecX))
        pstRpy->bReversedFit = true;

    if (pstCmd->bFindPair) {
        Fitting::fitParallelLine(vecFitPoint1, vecFitPoint2, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->fIntercept2, pstRpy->bReversedFit);
        pstRpy->stLine = CalcUtils::calcEndPointOfLine(vecFitPoint1, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept);
        pstRpy->stLine2 = CalcUtils::calcEndPointOfLine(vecFitPoint2, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept2);
        pstRpy->fDistance = CalcUtils::ptDisToLine(pstRpy->stLine2.pt1, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept);
    }
    else {
        auto vecPointsToKeep = Fitting::fitLineRemoveStray(vecFitPoint1, pstRpy->bReversedFit, pstCmd->fRmStrayPointRatio, pstRpy->fSlope, pstRpy->fIntercept);
        pstRpy->stLine = CalcUtils::calcEndPointOfLine(vecFitPoint1, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept);
        if (! isAutoMode()) {
            for (const auto &point : vecPointsToKeep)
                cv::circle(pstRpy->matResultImg, point, 3, GREEN_SCALAR, 1);
        }
    }

    //Draw the result lines.
    if (! isAutoMode()) {
        cv::line(pstRpy->matResultImg, pstRpy->stLine.pt1, pstRpy->stLine.pt2, GREEN_SCALAR, 2);
        if (pstCmd->bFindPair)
            cv::line(pstRpy->matResultImg, pstRpy->stLine2.pt1, pstRpy->stLine2.pt2, GREEN_SCALAR, 2);
    }

    pstRpy->enStatus = VisionStatus::OK;

    if (pstCmd->bCheckLinearity) {
        int nInTolerancePointCount = 0;
        for (const auto &point : vecFitPoint1) {
            if (CalcUtils::ptDisToLine(point, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept) < pstCmd->fPointMaxOffset)
                ++ nInTolerancePointCount;
        }
        pstRpy->fLinearity = ToFloat(nInTolerancePointCount) / ToFloat(vecFitPoint1.size());
        if (pstRpy->fLinearity < pstCmd->fMinLinearity)
            pstRpy->enStatus = VisionStatus::LINE_LINEARITY_REJECT;
    }

    pstRpy->fAngle = ToFloat(CalcUtils::calcSlopeDegree<float>(pstRpy->stLine.pt1, pstRpy->stLine.pt2));
    if (VisionStatus::OK == pstRpy->enStatus && pstCmd->bCheckAngle) {
        if (fabs(pstRpy->fAngle - pstCmd->fExpectedAngle) > pstCmd->fAngleDiffTolerance)
            pstRpy->enStatus = VisionStatus::LINE_ANGLE_OUT_OF_TOL;
    }
    
    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_extractRotatedROI(const cv::Mat &matInputImg, const cv::RotatedRect &rectRotatedROI, cv::Mat &matROI) {
    if (fabs(rectRotatedROI.angle) < 0.1) {
        cv::Rect rectBounding(ToInt32(rectRotatedROI.center.x - rectRotatedROI.size.width / 2.f), ToInt32(rectRotatedROI.center.y - rectRotatedROI.size.height / 2.f),
            ToInt32(rectRotatedROI.size.width), ToInt32(rectRotatedROI.size.height));
        if (rectBounding.x < 0 || rectBounding.y < 0 ||
            rectBounding.width <= 0 || rectBounding.height <= 0 ||
            (rectBounding.x + rectBounding.width) > matInputImg.cols ||
            (rectBounding.y + rectBounding.height) > matInputImg.rows) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The ROI rect (%d, %d, %d, %d) is invalid.",
                rectBounding.x, rectBounding.y, rectBounding.width, rectBounding.height);
            WriteLog(chArrMsg);
            return VisionStatus::INVALID_PARAM;
        }
        matROI = cv::Mat(matInputImg, rectBounding);
        return VisionStatus::OK;
    }
    //In below warpAffine function, dst(x,y) = src(M11 * x + M12 * y + M13, M21 * x + M22 * y + M23);
    //If the rotation is big, if not enlarge the fetch ROI, than some of the position in dst will not exit in src ROI
    //So the image will be black.
    auto newSize = std::max(rectRotatedROI.size.width, rectRotatedROI.size.height);
    auto rectRotatedROINew(rectRotatedROI);
    rectRotatedROINew.size = cv::Size2f(newSize, newSize);
    auto rectBounding = rectRotatedROINew.boundingRect();
    if (rectBounding.x < 0) rectBounding.x = 0;
    if (rectBounding.y < 0) rectBounding.y = 0;
    if ((rectBounding.x + rectBounding.width) > matInputImg.cols)
        rectBounding.width = matInputImg.cols - rectBounding.x;
    if ((rectBounding.y + rectBounding.height) > matInputImg.rows)
        rectBounding.height = matInputImg.rows - rectBounding.y;

    cv::Mat matBoundingROI(matInputImg, rectBounding);
    cv::Point2f ptNewCenter(rectRotatedROINew.center.x - rectBounding.x, rectRotatedROINew.center.y - rectBounding.y);
    cv::Mat matWarp = cv::getRotationMatrix2D(ptNewCenter, rectRotatedROINew.angle, 1.);
    auto vecVec = CalcUtils::matToVector<double>(matWarp);
    auto rectWarppedPoly = CalcUtils::warpRect<double>(matWarp, rectBounding);
    auto rectWarppedPolyBounding = cv::boundingRect(rectWarppedPoly);
    cv::Mat matWarpResult;
    cv::warpAffine(matBoundingROI, matWarpResult, matWarp, rectWarppedPolyBounding.size());
    cv::Rect2f rectSubROI(ptNewCenter.x - rectRotatedROI.size.width / 2.f, ptNewCenter.y - rectRotatedROI.size.height / 2.f,
        rectRotatedROI.size.width, rectRotatedROI.size.height);
    if (rectSubROI.x < 0) rectSubROI.x = 0;
    if (rectSubROI.y < 0) rectSubROI.y = 0;
    if ((rectSubROI.x + rectSubROI.width) > matWarpResult.cols)
        rectSubROI.width = matWarpResult.cols - rectSubROI.x;
    if ((rectSubROI.y + rectSubROI.height) > matWarpResult.rows)
        rectSubROI.height = matWarpResult.rows - rectSubROI.y;
    matROI = cv::Mat(matWarpResult, rectSubROI);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("Bouding ROI", matBoundingROI);
        showImage("Warpped ROI", matWarpResult);
        showImage("Result ROI", matROI);
    }

    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_calcCaliperDirection(const cv::Mat &matRoiImg, bool bReverseFit, PR_CALIPER_DIR &enDirection) {
    cv::Mat matLeftTop, matRightBottom;
    if (bReverseFit) {
        matLeftTop = cv::Mat(matRoiImg, cv::Rect(0, 0, matRoiImg.cols / 2, matRoiImg.rows));
        matRightBottom = cv::Mat(matRoiImg, cv::Rect(matRoiImg.cols / 2, 0, matRoiImg.cols / 2, matRoiImg.rows));
    }
    else {
        matLeftTop = cv::Mat(matRoiImg, cv::Rect(0, 0, matRoiImg.cols, matRoiImg.rows / 2));
        matRightBottom = cv::Mat(matRoiImg, cv::Rect(0, matRoiImg.rows / 2, matRoiImg.cols, matRoiImg.rows / 2));
    }
    auto sumOfLeftTop = cv::sum(matLeftTop)[0];
    auto sumOfRightBottom = cv::sum(matRightBottom)[0];
    if (sumOfLeftTop < sumOfRightBottom)
        enDirection = PR_CALIPER_DIR::DARK_TO_BRIGHT;
    else
        enDirection = PR_CALIPER_DIR::BRIGHT_TO_DARK;
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_findLineByProjection(const cv::Mat &matGray, const cv::Mat &matROIMask, const cv::Rect &rectROI, PR_CALIPER_DIR enDetectDir, VectorOfPoint2f &vecFitPoint, PR_FIND_LINE_RPY *const pstRpy) {
    VectorOfPoint vecPoints;
    std::vector<int> vecX, vecY, vecIndexCanFormLine, vecProjection;
    VectorOfVectorOfPoint vecVecPoint;

    int nThreshold = _autoThreshold(matGray, matROIMask);
    cv::Mat matThreshold;
    cv::threshold(matGray, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, cv::THRESH_BINARY);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Find line threshold image", matThreshold);

    if (! matROIMask.empty()) {
        cv::Mat matROIMaskReverse = cv::Scalar(PR_MAX_GRAY_LEVEL) - matROIMask;
        matThreshold.setTo(cv::Scalar(0), matROIMaskReverse);
    }

    cv::findNonZero(matThreshold, vecPoints);

    if (vecPoints.size() < 2) {
        WriteLog("Not enough points to find line.");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        return pstRpy->enStatus;
    }

    for (auto &point : vecPoints) {
        vecX.push_back(point.x);
        vecY.push_back(point.y);
    }

    pstRpy->bReversedFit = false;
    if (CalcUtils::calcStdDeviation(vecY) > CalcUtils::calcStdDeviation(vecX))
        pstRpy->bReversedFit = true;

    if (PR_CALIPER_DIR::AUTO == enDetectDir)
        _calcCaliperDirection(matGray, pstRpy->bReversedFit, enDetectDir);

    auto projectSize = pstRpy->bReversedFit ? matGray.cols : matGray.rows;
    vecProjection.resize(projectSize, 0);
    for (const auto &point : vecPoints) {
        if (pstRpy->bReversedFit)
            ++ vecProjection[point.x];
        else
            ++ vecProjection[point.y];
    }

    int maxValue = *std::max_element(vecProjection.begin(), vecProjection.end());

    //Find the possible coordinate can form lines.
    for (size_t index = 0; index < vecProjection.size(); ++ index) {
        if (vecProjection[index] > 0.8 * maxValue)
            vecIndexCanFormLine.push_back(ToInt32(index));
    }

    auto initialLinePos = 0;
    if (PR_CALIPER_DIR::DARK_TO_BRIGHT == enDetectDir)
        initialLinePos = vecIndexCanFormLine[0];
    else
        initialLinePos = vecIndexCanFormLine.back();

    int rs = std::max(int(0.1 *(vecIndexCanFormLine.back() - vecIndexCanFormLine[0])), 20);

    if (pstRpy->bReversedFit) vecVecPoint.resize(matGray.rows);
    else                        vecVecPoint.resize(matGray.cols);

    //Find all the points around the initial line pos.
    for (const auto &point : vecPoints) {
        if (pstRpy->bReversedFit) {
            if (point.x > initialLinePos - rs && point.x < initialLinePos + rs)
                vecVecPoint[point.y].push_back(point);
        }
        else {
            if (point.y > initialLinePos - rs && point.y < initialLinePos + rs)
                vecVecPoint[point.x].push_back(point);
        }
    }

    //Find the extreme point on each row or column.
    for (const auto &vecPointTmp : vecVecPoint) {
        if (! vecPointTmp.empty()) {
            auto point = PR_CALIPER_DIR::DARK_TO_BRIGHT == enDetectDir ? vecPointTmp.front() : vecPointTmp.back();
            if (pstRpy->bReversedFit) {
                if (point.x != initialLinePos - rs && point.x != initialLinePos - rs)
                    vecFitPoint.push_back(point + cv::Point(rectROI.x, rectROI.y));
            }
            else {
                if (point.y != initialLinePos - rs && point.y != initialLinePos - rs)
                    vecFitPoint.push_back(point + cv::Point(rectROI.x, rectROI.y));
            }
        }
    }

    Fitting::fitLine(vecFitPoint, pstRpy->fSlope, pstRpy->fIntercept, pstRpy->bReversedFit);
    pstRpy->stLine = CalcUtils::calcEndPointOfLine(vecFitPoint, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept);

    const int MIN_DIST_FROM_LINE_TO_ROI_EDGE = 5;
    const float HORIZONTAL_SLOPE_TOL = 0.1f;
    if (pstRpy->bReversedFit) {
        if (pstRpy->fSlope < HORIZONTAL_SLOPE_TOL &&
            (fabs(pstRpy->stLine.pt1.x - rectROI.x) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
            fabs(pstRpy->stLine.pt2.x - rectROI.x) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
            fabs(pstRpy->stLine.pt1.x - rectROI.x - rectROI.width) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
            fabs(pstRpy->stLine.pt2.x - rectROI.x - rectROI.width) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE)) {
            pstRpy->enStatus = VisionStatus::PROJECTION_CANNOT_FIND_LINE;
            return pstRpy->enStatus;
        }
    }
    else {
        if (pstRpy->fSlope < HORIZONTAL_SLOPE_TOL &&
            (fabs(pstRpy->stLine.pt1.y - rectROI.y) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
            fabs(pstRpy->stLine.pt2.y - rectROI.y) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
            fabs(pstRpy->stLine.pt1.y - rectROI.y - rectROI.height) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE ||
            fabs(pstRpy->stLine.pt2.y - rectROI.y - rectROI.height) <= MIN_DIST_FROM_LINE_TO_ROI_EDGE)) {
            pstRpy->enStatus = VisionStatus::PROJECTION_CANNOT_FIND_LINE;
            return pstRpy->enStatus;
        }
    }

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ bool VisionAlgorithm::_findEdgePosInX(const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection, PR_CALIPER_SELECT_EDGE enSelectEdge, int nEdgeThreshold, bool bFindPair, int &nPos1, int &nPos2) {
    cv::Mat matAverage;
    cv::reduce(matInput, matAverage, 0, cv::ReduceTypes::REDUCE_AVG);
    bool bFound = false;
    cv::Mat matGuassianDiffResult;
    CalcUtils::filter2D_Conv(matAverage, matGuassianDiffResult, CV_32F, matGuassianDiffKernel);
    if (PR_CALIPER_SELECT_EDGE::MAX_EDGE == enSelectEdge) {
        double minValue = 0., maxValue = 0.;
        cv::Point ptMin, ptMax;
        cv::minMaxLoc(matGuassianDiffResult, &minValue, &maxValue, &ptMin, &ptMax);
        if (bFindPair) {
            if (fabs(minValue) > nEdgeThreshold && fabs(maxValue) > nEdgeThreshold) {
                if (PR_CALIPER_DIR::DARK_TO_BRIGHT == enDirection) {
                    nPos1 = ptMin.x;
                    nPos2 = ptMax.x;
                }
                else {
                    nPos1 = ptMax.x;
                    nPos2 = ptMin.x;
                }
                bFound = true;
            }
        }
        else {
            if (PR_CALIPER_DIR::DARK_TO_BRIGHT == enDirection && -minValue > nEdgeThreshold) {
                nPos1 = ptMin.x;
                bFound = true;
            }
            else if (PR_CALIPER_DIR::BRIGHT_TO_DARK == enDirection && maxValue > nEdgeThreshold) {
                nPos1 = ptMax.x;
                bFound = true;
            }
        }
    }
    else {
        for (int col = 0; col < matGuassianDiffResult.cols; ++ col) {
            auto value = matGuassianDiffResult.at<float>(col);
            if (PR_CALIPER_DIR::DARK_TO_BRIGHT == enDirection && -value > nEdgeThreshold) {
                nPos1 = col;
                bFound = true;
            }
            else if (PR_CALIPER_DIR::BRIGHT_TO_DARK == enDirection && value > nEdgeThreshold) {
                nPos1 = col;
                bFound = true;
            }
        }
    }
    return bFound;
}

/*static*/ bool VisionAlgorithm::_findEdgePosInY(const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection, PR_CALIPER_SELECT_EDGE enSelectEdge, int nEdgeThreshold, bool bFindPair, int &nPos1, int &nPos2) {
    cv::Mat matAverage;
    cv::reduce(matInput, matAverage, 1, cv::ReduceTypes::REDUCE_AVG);
    bool bFound = false;
    cv::Mat matGuassianDiffResult;
    CalcUtils::filter2D_Conv(matAverage, matGuassianDiffResult, CV_32F, matGuassianDiffKernel);
    if (PR_CALIPER_SELECT_EDGE::MAX_EDGE == enSelectEdge) {
        double minValue = 0., maxValue = 0.;
        cv::Point ptMin, ptMax;
        cv::minMaxLoc(matGuassianDiffResult, &minValue, &maxValue, &ptMin, &ptMax);
        if (bFindPair) {
            if (fabs(minValue) > nEdgeThreshold && fabs(maxValue) > nEdgeThreshold) {
                if (PR_CALIPER_DIR::DARK_TO_BRIGHT == enDirection) {
                    nPos1 = ptMin.y;
                    nPos2 = ptMax.y;
                }
                else {
                    nPos1 = ptMax.y;
                    nPos2 = ptMin.y;
                }
                bFound = true;
            }
        }
        else {
            if (PR_CALIPER_DIR::DARK_TO_BRIGHT == enDirection && -minValue > nEdgeThreshold) {
                nPos1 = ptMin.y;
                bFound = true;
            }
            else if (PR_CALIPER_DIR::BRIGHT_TO_DARK == enDirection && maxValue > nEdgeThreshold) {
                nPos1 = ptMax.y;
                bFound = true;
            }
        }
    }
    else {
        for (int row = 0; row < matGuassianDiffResult.rows; ++ row) {
            auto value = matGuassianDiffResult.at<float>(row);
            if (PR_CALIPER_DIR::DARK_TO_BRIGHT == enDirection && -value > nEdgeThreshold) {
                nPos1 = row;
                bFound = true;
            }
            else if (PR_CALIPER_DIR::BRIGHT_TO_DARK == enDirection && value > nEdgeThreshold) {
                nPos1 = row;
                bFound = true;
            }
        }
    }
    return bFound;
}

/*static*/ int VisionAlgorithm::_findMaxDiffPosInX(const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection) {
    cv::Mat matAverage;
    cv::reduce(matInput, matAverage, 0, cv::ReduceTypes::REDUCE_AVG);
    cv::Mat matGuassianDiffResult;
    CalcUtils::filter2D_Conv(matAverage, matGuassianDiffResult, CV_32F, matGuassianDiffKernel);

    double minValue = 0., maxValue = 0.;
    cv::Point ptMin, ptMax;
    cv::minMaxLoc(matGuassianDiffResult, &minValue, &maxValue, &ptMin, &ptMax);
    if (PR_CALIPER_DIR::DARK_TO_BRIGHT == enDirection)
        return ptMin.x;
    return ptMax.x;
}

/*static*/ int VisionAlgorithm::_findMaxDiffPosInY(const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection) {
    cv::Mat matAverage;
    cv::reduce(matInput, matAverage, 1, cv::ReduceTypes::REDUCE_AVG);

    cv::Mat matGuassianDiffResult;
    CalcUtils::filter2D_Conv(matAverage, matGuassianDiffResult, CV_32F, matGuassianDiffKernel);
    double minValue = 0., maxValue = 0.;
    cv::Point ptMin, ptMax;
    cv::minMaxLoc(matGuassianDiffResult, &minValue, &maxValue, &ptMin, &ptMax);
    if (PR_CALIPER_DIR::DARK_TO_BRIGHT == enDirection)
        return ptMin.y;
    return ptMax.y;
}

//The cognex method of find line.
VisionStatus VisionAlgorithm::_findLineByCaliper(const cv::Mat &matInputImg, const cv::Mat &matROIMask, const cv::Rect &rectROI, const PR_FIND_LINE_CMD *const pstCmd,
    VectorOfPoint2f &vecFitPoint1, VectorOfPoint2f &vecFitPoint2, VectorOfRect &vecSubRects, PR_FIND_LINE_RPY *const pstRpy) {
    if (matInputImg.rows > matInputImg.cols)
        pstRpy->bReversedFit = true;
    else
        pstRpy->bReversedFit = false;
    auto enDirection = pstCmd->enDetectDir;
    if (PR_CALIPER_DIR::AUTO == enDirection)
        _calcCaliperDirection(matInputImg, pstRpy->bReversedFit, enDirection);

    int COLS = matInputImg.cols;
    int ROWS = matInputImg.rows;

    cv::Mat matGuassinKernel = CalcUtils::generateGuassinDiffKernel(pstCmd->nDiffFilterHalfW, pstCmd->fDiffFilterSigma);

    if (pstRpy->bReversedFit) {
        auto fInterval = ToFloat(ROWS) / ToFloat(pstCmd->nCaliperCount);
        for (int nCaliperNo = 0; nCaliperNo < pstCmd->nCaliperCount; ++ nCaliperNo) {
            int nCurrentProcessedPos = ToInt32(fInterval * nCaliperNo + pstCmd->fCaliperWidth / 2.f);
            int nROISizeBegin = nCurrentProcessedPos - ToInt32(pstCmd->fCaliperWidth / 2.f);
            if (nROISizeBegin < 0) nROISizeBegin = 0;
            int nROISizeEnd = nCurrentProcessedPos + ToInt32(pstCmd->fCaliperWidth / 2.f);
            if (nROISizeEnd > matInputImg.rows) nROISizeEnd = ROWS;
            cv::Rect rectSubROI(0, nROISizeBegin, COLS, nROISizeEnd - nROISizeBegin);
            vecSubRects.push_back(rectSubROI);
            cv::Mat matSubROI(matInputImg, rectSubROI);
            int nEdgePos1, nEdgePos2;
            bool bResult = _findEdgePosInX(matSubROI, matGuassinKernel, enDirection, pstCmd->enSelectEdge, pstCmd->nEdgeThreshold, pstCmd->bFindPair, nEdgePos1, nEdgePos2);
            if (bResult) {
                vecFitPoint1.push_back(cv::Point(nEdgePos1, nCurrentProcessedPos));
                if (! matROIMask.empty() && matROIMask.at<uchar>(vecFitPoint1.back()) == 0)
                    vecFitPoint1.pop_back();

                if (pstCmd->bFindPair) {
                    vecFitPoint2.push_back(cv::Point(nEdgePos2, nCurrentProcessedPos));
                    if (! matROIMask.empty() && matROIMask.at<uchar>(vecFitPoint2.back()) == 0)
                        vecFitPoint2.pop_back();
                }
            }
        }
    }
    else {
        cv::transpose(matGuassinKernel, matGuassinKernel);
        auto fInterval = ToFloat(COLS) / ToFloat(pstCmd->nCaliperCount);
        int nCurrentProcessedPos = 0;
        for (int nCaliperNo = 0; nCaliperNo < pstCmd->nCaliperCount; ++ nCaliperNo) {
            int nCurrentProcessedPos = ToInt32(fInterval * nCaliperNo + pstCmd->fCaliperWidth / 2.f);
            if (nCurrentProcessedPos > COLS) break;
            int nROISizeBegin = nCurrentProcessedPos - ToInt32(pstCmd->fCaliperWidth / 2.f);
            if (nROISizeBegin < 0) nROISizeBegin = 0;
            int nROISizeEnd = nCurrentProcessedPos + ToInt32(pstCmd->fCaliperWidth / 2.f);
            if (nROISizeEnd > matInputImg.cols) nROISizeEnd = COLS;
            cv::Rect rectSubROI(nROISizeBegin, 0, nROISizeEnd - nROISizeBegin, ROWS);
            cv::Mat matSubROI(matInputImg, rectSubROI);
            vecSubRects.push_back(rectSubROI);
            int nEdgePos1, nEdgePos2;
            bool bResult = _findEdgePosInY(matSubROI, matGuassinKernel, enDirection, pstCmd->enSelectEdge, pstCmd->nEdgeThreshold, pstCmd->bFindPair, nEdgePos1, nEdgePos2);
            if (bResult) {
                vecFitPoint1.push_back(cv::Point(nCurrentProcessedPos, nEdgePos1));
                if (! matROIMask.empty() && matROIMask.at<uchar>(vecFitPoint1.back()) == 0)
                    vecFitPoint1.pop_back();

                if (pstCmd->bFindPair) {
                    vecFitPoint2.push_back(cv::Point(nCurrentProcessedPos, nEdgePos2));
                    if (! matROIMask.empty() && matROIMask.at<uchar>(vecFitPoint2.back()) == 0)
                        vecFitPoint2.pop_back();
                }
            }
        }
    }

    for (auto &point : vecFitPoint1) {
        point.x += rectROI.x;
        point.y += rectROI.y;
    }

    for (auto &point : vecFitPoint2) {
        point.x += rectROI.x;
        point.y += rectROI.y;
    }

    for (auto &rect : vecSubRects) {
        rect.x += rectROI.x;
        rect.y += rectROI.y;
    }

    if (vecFitPoint1.size() < 2) {
        WriteLog("_findLineByCaliper can not find enough points.");
        pstRpy->enStatus = VisionStatus::CALIPER_CAN_NOT_FIND_LINE;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::fitParallelLine(const PR_FIT_PARALLEL_LINE_CMD *const pstCmd, PR_FIT_PARALLEL_LINE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFitParallelLine);

    cv::Mat matGray, matThreshold;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(pstCmd->matInputImg, matGray, CV_BGR2GRAY);
    else
        matGray = pstCmd->matInputImg.clone();

    ListOfPoint listPoint1 = _findPointsInRegionByThreshold(matGray, pstCmd->rectArrROI[0], pstCmd->nThreshold, pstCmd->enAttribute);
    ListOfPoint listPoint2 = _findPointsInRegionByThreshold(matGray, pstCmd->rectArrROI[1], pstCmd->nThreshold, pstCmd->enAttribute);
    if (listPoint1.size() < 2 || listPoint2.size() < 2) {
        WriteLog("Not enough points to fit Parallel line.");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        return pstRpy->enStatus;
    }
    else if (listPoint1.size() > PR_FIT_LINE_MAX_POINT_COUNT || listPoint2.size() > PR_FIT_LINE_MAX_POINT_COUNT) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "Too much noise to fit parallel line, points number %d, %d.", listPoint1.size(), listPoint2.size());
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        return pstRpy->enStatus;
    }

    std::vector<int> vecX, vecY;
    for (const auto &point : listPoint1) {
        vecX.push_back(point.x);
        vecY.push_back(point.y);
    }

    //For y = kx + b, the k is the slope, when it is very large, the error is quite big, so change
    //to get the x = k1 + b1, and get the k = 1 / k1, b = -b1 / k1. Then the result is more accurate.
    pstRpy->bReversedFit = false;
    if (CalcUtils::calcStdDeviation(vecY) > CalcUtils::calcStdDeviation(vecX))
        pstRpy->bReversedFit = true;

    std::vector<ListOfPoint::const_iterator> overTolPoints1, overTolPoints2;
    int nIteratorNum = 0;
    do {
        for (const auto &it : overTolPoints1)
            listPoint1.erase(it);
        for (const auto &it : overTolPoints2)
            listPoint2.erase(it);

        Fitting::fitParallelLine(listPoint1, listPoint2, pstRpy->fSlope, pstRpy->fIntercept1, pstRpy->fIntercept2, pstRpy->bReversedFit);
        overTolPoints1 = _findPointOverLineTol(listPoint1, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept1, pstCmd->enRmNoiseMethod, pstCmd->fErrTol);
        overTolPoints2 = _findPointOverLineTol(listPoint2, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept2, pstCmd->enRmNoiseMethod, pstCmd->fErrTol);
        ++ nIteratorNum;
    } while ((! overTolPoints1.empty() || ! overTolPoints2.empty()) && nIteratorNum < 20);

    if (listPoint1.size() < 2 || listPoint2.size() < 2) {
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    pstRpy->stLine1 = CalcUtils::calcEndPointOfLine(listPoint1, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept1);
    pstRpy->stLine2 = CalcUtils::calcEndPointOfLine(listPoint2, pstRpy->bReversedFit, pstRpy->fSlope, pstRpy->fIntercept2);

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::line(pstRpy->matResultImg, pstRpy->stLine1.pt1, pstRpy->stLine1.pt2, cv::Scalar(255, 0, 0), 2);
    cv::line(pstRpy->matResultImg, pstRpy->stLine2.pt1, pstRpy->stLine2.pt2, cv::Scalar(255, 0, 0), 2);
    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::fitRect(const PR_FIT_RECT_CMD *const pstCmd, PR_FIT_RECT_RPY *const pstRpy, bool bReplay /*= false*/) {
    char chArrMsg[1000];
    if (NULL == pstCmd || NULL == pstRpy) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "Input is invalid, pstCmd = %d, pstRpy = %d", (int)pstCmd, (int)pstRpy);
        WriteLog(chArrMsg);
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
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(pstCmd->matInputImg, matGray, CV_BGR2GRAY);
    else
        matGray = pstCmd->matInputImg.clone();

    VectorOfListOfPoint vecListPoint;
    for (int i = 0; i < PR_RECT_EDGE_COUNT; ++ i) {
        vecListPoint.push_back(_findPointsInRegionByThreshold(matGray, pstCmd->rectArrROI[i], pstCmd->nThreshold, pstCmd->enAttribute));
        if (vecListPoint[i].size() < 2) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "Edge %d not enough points to fit rect", i);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
            return pstRpy->enStatus;
        }
        else if (vecListPoint[i].size() > PR_FIT_LINE_MAX_POINT_COUNT) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "Too much noise to fit rect, line %d points number %d", i + 1, vecListPoint[i].size());
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
            return pstRpy->enStatus;
        }
    }
    std::vector<int> vecX, vecY;
    for (const auto &point : vecListPoint[0]) {
        vecX.push_back(point.x);
        vecY.push_back(point.y);
    }

    //For y = kx + b, the k is the slope, when it is very large, the error is quite big, so change
    //to get the x = k1 + b1, and get the k = 1 / k1, b = -b1 / k1. Then the result is more accurate.
    pstRpy->bLineOneReversedFit = false;
    pstRpy->bLineTwoReversedFit = false;
    if (CalcUtils::calcStdDeviation(vecY) > CalcUtils::calcStdDeviation(vecX)) {
        pstRpy->bLineOneReversedFit = true;
        pstRpy->bLineTwoReversedFit = false;
    }
    else {
        pstRpy->bLineOneReversedFit = false;
        pstRpy->bLineTwoReversedFit = true;
    }

    std::vector<std::vector<ListOfPoint::const_iterator>> vecVecOutTolIndex;
    vecVecOutTolIndex.reserve(vecListPoint.size());
    std::vector<float>    vecIntercept;
    auto hasPointOutTol = false;
    auto iterateCount = 0;
    do {
        auto index = 0;
        for (const auto &vecOutTolIndex : vecVecOutTolIndex) {
            for (const auto &it : vecOutTolIndex)
                vecListPoint[index].erase(it);
            ++ index;
        }
        Fitting::fitRect(vecListPoint, pstRpy->fSlope1, pstRpy->fSlope2, vecIntercept, pstRpy->bLineOneReversedFit, pstRpy->bLineTwoReversedFit);
        vecVecOutTolIndex.clear();
        hasPointOutTol = false;
        for (int i = 0; i < PR_RECT_EDGE_COUNT; ++ i) {
            std::vector<ListOfPoint::const_iterator> vecOutTolIndex;
            if (i < 2)
                vecOutTolIndex = _findPointOverLineTol(vecListPoint[i], pstRpy->bLineOneReversedFit, pstRpy->fSlope1, vecIntercept[i], pstCmd->enRmNoiseMethod, pstCmd->fErrTol);
            else
                vecOutTolIndex = _findPointOverLineTol(vecListPoint[i], pstRpy->bLineTwoReversedFit, pstRpy->fSlope2, vecIntercept[i], pstCmd->enRmNoiseMethod, pstCmd->fErrTol);
            vecVecOutTolIndex.push_back(vecOutTolIndex);
        }

        for (const auto &vecOutTolIndex : vecVecOutTolIndex) {
            if (! vecOutTolIndex.empty()) {
                hasPointOutTol = true;
                break;
            }
        }
        ++ iterateCount;
    } while (hasPointOutTol && iterateCount < 20);

    int nLineNumber = 1;
    for (const auto &listPoint : vecListPoint) {
        if (listPoint.size() < 2) {
            pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
            _snprintf(chArrMsg, sizeof(chArrMsg), "Rect line %d has too much noise to fit.", nLineNumber);
            WriteLog(chArrMsg);
            FINISH_LOGCASE;
            return pstRpy->enStatus;
        }
        ++ nLineNumber;
    }

    for (int i = 0; i < PR_RECT_EDGE_COUNT; ++ i) {
        if (i < 2)
            pstRpy->arrLines[i] = CalcUtils::calcEndPointOfLine(vecListPoint[i], pstRpy->bLineOneReversedFit, pstRpy->fSlope1, vecIntercept[i]);
        else
            pstRpy->arrLines[i] = CalcUtils::calcEndPointOfLine(vecListPoint[i], pstRpy->bLineTwoReversedFit, pstRpy->fSlope2, vecIntercept[i]);
        pstRpy->fArrIntercept[i] = vecIntercept[i];
    }
    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    for (int i = 0; i < PR_RECT_EDGE_COUNT; ++ i)
        cv::line(pstRpy->matResultImg, pstRpy->arrLines[i].pt1, pstRpy->arrLines[i].pt2, cv::Scalar(255, 0, 0), 2);

    pstRpy->enStatus = VisionStatus::OK;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::findEdge(const PR_FIND_EDGE_CMD *const pstCmd, PR_FIND_EDGE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char chArrMsg[1000];

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    if (matROI.empty()) {
        WriteLog("ROI image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (! pstCmd->bAutoThreshold && (pstCmd->nThreshold <= 0 || pstCmd->nThreshold >= PR_MAX_GRAY_LEVEL)) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "The threshold = %d is invalid", pstCmd->nThreshold);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFindEdge);

    cv::Mat matROIGray, matThreshold, matBlur, matCannyResult;
    cv::cvtColor(matROI, matROIGray, CV_BGR2GRAY);

    auto threshold = pstCmd->nThreshold;
    if (pstCmd->bAutoThreshold)
        threshold = _autoThreshold(matROIGray);

    cv::threshold(matROIGray, matThreshold, pstCmd->nThreshold, 255, cv::THRESH_BINARY);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Threshold Result", matThreshold);

    cv::blur(matThreshold, matBlur, cv::Size(2, 2));

    // Detect edges using canny
    cv::Canny(matBlur, matCannyResult, pstCmd->nThreshold, 255);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Canny Result", matCannyResult);

    std::vector<cv::Vec4i> lines;
    auto voteThreshold = ToInt32(pstCmd->fMinLength / 2.f);
    cv::HoughLinesP(matCannyResult, lines, 1, CV_PI / 180, voteThreshold, pstCmd->fMinLength, 10);
    std::vector<PR_Line2f> vecLines, vecLinesAfterMerge, vecLineDisplay;
    for (size_t i = 0; i < lines.size(); ++ i) {
        cv::Point2f pt1((float)lines[i][0], (float)lines[i][1]);
        cv::Point2f pt2((float)lines[i][2], (float)lines[i][3]);
        PR_Line2f line(pt1, pt2);
        vecLines.push_back(line);
    }
    _mergeLines(vecLines, vecLinesAfterMerge);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        cv::Mat matResultImage = matROI.clone();
        for (const auto &line : vecLinesAfterMerge)
            cv::line(matResultImage, line.pt1, line.pt2, cv::Scalar(255, 0, 0), 2);
        showImage("Find edge Result", matResultImage);
    }
    Int32 nLineCount = 0;
    for (const auto &line : vecLinesAfterMerge) {
        float fSlope = CalcUtils::lineSlope(line);
        if (fabs(fSlope) < 0.1) {
            if (PR_EDGE_DIRECTION::HORIZONTAL == pstCmd->enDirection || PR_EDGE_DIRECTION::BOTH == pstCmd->enDirection) {
                ++ nLineCount;
                vecLineDisplay.push_back(line);
            }
        }
        else if ((line.pt1.x == line.pt2.x) || fabs(fSlope) > 10) {
            if (PR_EDGE_DIRECTION::VERTIAL == pstCmd->enDirection || PR_EDGE_DIRECTION::BOTH == pstCmd->enDirection) {
                ++ nLineCount;
                vecLineDisplay.push_back(line);
            }
        }
    }
    pstRpy->nEdgeCount = nLineCount;
    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    for (const auto &line : vecLineDisplay) {
        cv::Point2f pt1(line.pt1.x + pstCmd->rectROI.x, line.pt1.y + pstCmd->rectROI.y);
        cv::Point2f pt2(line.pt2.x + pstCmd->rectROI.x, line.pt2.y + pstCmd->rectROI.y);
        cv::line(pstRpy->matResultImg, pt1, pt2, BLUE_SCALAR, 2);
    }
    pstRpy->enStatus = VisionStatus::OK;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::fitCircle(const PR_FIT_CIRCLE_CMD *const pstCmd, PR_FIT_CIRCLE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (! pstCmd->matMask.empty()) {
        if (pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d ).",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matMask.channels() != 1) {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (PR_FIT_METHOD::RANSAC == pstCmd->enMethod && (1000 <= pstCmd->nMaxRansacTime || pstCmd->nMaxRansacTime <= 0)) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "Max Ransac time %d is invalid.", pstCmd->nMaxRansacTime);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFitCircle);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    cv::Mat matGray, matThreshold;

    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI.clone();

    if (pstCmd->bPreprocessed)
        matThreshold = matGray;
    else {
        cv::ThresholdTypes enThresType = cv::THRESH_BINARY;
        if (PR_OBJECT_ATTRIBUTE::DARK == pstCmd->enAttribute)
            enThresType = cv::THRESH_BINARY_INV;

        cv::threshold(matGray, matThreshold, pstCmd->nThreshold, PR_MAX_GRAY_LEVEL, enThresType);
        CV_Assert(matThreshold.channels() == 1);

        if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
            showImage("Threshold image", matThreshold);
    }

    if (! pstCmd->matMask.empty()) {
        cv::Mat matROIMask(pstCmd->matMask, pstCmd->rectROI);
        matROIMask = cv::Scalar(255) - matROIMask;
        matThreshold.setTo(cv::Scalar(0), matROIMask);
    }

    VectorOfPoint vecPoints;
    cv::findNonZero(matThreshold, vecPoints);
    cv::RotatedRect fitResult;

    if (vecPoints.size() < 3) {
        WriteLog("Not enough points to fit circle");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    if (vecPoints.size() > PR_FIT_CIRCLE_MAX_POINT) {
        WriteLog("Too much points to fit circle");
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    if (PR_FIT_METHOD::RANSAC == pstCmd->enMethod)
        fitResult = Fitting::fitCircleRansac(vecPoints, pstCmd->fErrTol, pstCmd->nMaxRansacTime, vecPoints.size() / 2);
    else if (PR_FIT_METHOD::LEAST_SQUARE == pstCmd->enMethod)
        fitResult = Fitting::fitCircle(vecPoints);
    else if (PR_FIT_METHOD::LEAST_SQUARE_REFINE == pstCmd->enMethod)
        fitResult = Fitting::fitCircleIterate(vecPoints, pstCmd->enRmNoiseMethod, pstCmd->fErrTol);

    if (fitResult.size.width <= 0) {
        WriteLog("Failed to fit circle");
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    pstRpy->ptCircleCtr = fitResult.center + cv::Point2f(ToFloat(pstCmd->rectROI.x), ToFloat(pstCmd->rectROI.y));
    pstRpy->fRadius = fitResult.size.width / 2;
    pstRpy->enStatus = VisionStatus::OK;

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::circle(pstRpy->matResultImg, pstRpy->ptCircleCtr, (int)pstRpy->fRadius, BLUE_SCALAR, 2);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::fitCircleByPoint(const PR_FIT_CIRCLE_BY_POINT_CMD *const pstCmd, PR_FIT_CIRCLE_BY_POINT_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (PR_FIT_METHOD::RANSAC == pstCmd->enMethod && (1000 <= pstCmd->nMaxRansacTime || pstCmd->nMaxRansacTime <= 0)) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "Max Ransac time %d is invalid.", pstCmd->nMaxRansacTime);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->vecPoints.size() < 3) {
        WriteLog("Not enough points to fit circle");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        return pstRpy->enStatus;
    }

    if (pstCmd->vecPoints.size() > PR_FIT_CIRCLE_MAX_POINT) {
        WriteLog("Too much points to fit circle");
        pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    cv::RotatedRect fitResult;
    if (PR_FIT_METHOD::RANSAC == pstCmd->enMethod)
        fitResult = Fitting::fitCircleRansac(pstCmd->vecPoints, pstCmd->fErrTol, pstCmd->nMaxRansacTime, pstCmd->vecPoints.size() / 2);
    else if (PR_FIT_METHOD::LEAST_SQUARE == pstCmd->enMethod)
        fitResult = Fitting::fitCircle(pstCmd->vecPoints);
    else if (PR_FIT_METHOD::LEAST_SQUARE_REFINE == pstCmd->enMethod)
        fitResult = Fitting::fitCircleIterate(pstCmd->vecPoints, pstCmd->enRmNoiseMethod, pstCmd->fErrTol);

    if (fitResult.size.width <= 0) {
        WriteLog("Failed to fit circle");
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        return pstRpy->enStatus;
    }
    pstRpy->ptCircleCtr = fitResult.center;
    pstRpy->fRadius = fitResult.size.width / 2;
    pstRpy->enStatus = VisionStatus::OK;

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VectorOfPoint VisionAlgorithm::_findPointsOverCircleTol(const VectorOfPoint &vecPoints, const cv::RotatedRect &rotatedRect, float tolerance) {
    VectorOfPoint vecResult;
    for (const auto &point : vecPoints) {
        auto disToCtr = sqrt((point.x - rotatedRect.center.x) * (point.x - rotatedRect.center.x) + (point.y - rotatedRect.center.y) * (point.y - rotatedRect.center.y));
        auto err = disToCtr - rotatedRect.size.width / 2;
        if (fabs(err) > tolerance)
            vecResult.push_back(point);
    }
    return vecResult;
}

/*static*/ VisionStatus VisionAlgorithm::findCircle(const PR_FIND_CIRCLE_CMD *const pstCmd, PR_FIND_CIRCLE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    //Initialize reply first.
    pstRpy->fRadius = 0.f;
    pstRpy->fRadius2 = 0.f;

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->fMaxSrchRadius < pstCmd->fMinSrchRadius) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The max search radius %f is smaller than the min search radius %f.",
            pstCmd->fMaxSrchRadius, pstCmd->fMinSrchRadius);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->fMinSrchRadius <= 0.f) {
        WriteLog("The search radius should be positive value.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (fabs(pstCmd->fEndSrchAngle - pstCmd->fStartSrchAngle) < 10) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The search angle range [%f, %f] is too small.",
            pstCmd->fStartSrchAngle, pstCmd->fEndSrchAngle);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->nCaliperCount < 3) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The circle caliper count %d is less than 3.", pstCmd->nCaliperCount);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->fCaliperWidth < 3) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The circle caliper width %f is less than 3.", pstCmd->fCaliperWidth);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->fRmStrayPointRatio < 0.f || pstCmd->fRmStrayPointRatio > 0.9) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The remove stray point ratio %f is invalid.", pstCmd->fRmStrayPointRatio);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->bFindCirclePair && PR_CALIPER_SELECT_EDGE::FIRST_EDGE == pstCmd->enSelectEdge) {
        WriteLog("Find circle pair cannot use select first edge method.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFindCircle);

    cv::Mat matGray = pstCmd->matInputImg;
    if (pstCmd->matInputImg.channels() == 3)
        cv::cvtColor(pstCmd->matInputImg, matGray, CV_BGR2GRAY);

    const float CHECK_ROI_HEIGHT = pstCmd->fCaliperWidth;
    const float CHECK_ROI_WIDTH = pstCmd->fMaxSrchRadius - pstCmd->fMinSrchRadius;
    cv::Mat matGuassinKernel = CalcUtils::generateGuassinDiffKernel(pstCmd->nDiffFilterHalfW, pstCmd->fDiffFilterSigma);
    float fAngleInterval = (pstCmd->fEndSrchAngle - pstCmd->fStartSrchAngle) / pstCmd->nCaliperCount;
    float fMiddleOfSrchRadius = (pstCmd->fMaxSrchRadius + pstCmd->fMinSrchRadius) / 2.f;

    if (! isAutoMode())
        cv::cvtColor(matGray, pstRpy->matResultImg, CV_GRAY2BGR);

    VectorOfPoint2f vecPoints, vecPoints2, vecKeepPoints1, vecKeepPoints2;
    for (int i = 0; i < pstCmd->nCaliperCount; ++ i) {
        float fAngle = pstCmd->fStartSrchAngle + i * fAngleInterval + fAngleInterval / 2.f;
        float fCos = ToFloat(cos(CalcUtils::degree2Radian(fAngle)));
        float fSin = ToFloat(sin(CalcUtils::degree2Radian(fAngle)));

        cv::Point2f ptROICtr;
        ptROICtr.x = pstCmd->ptExpectedCircleCtr.x + fMiddleOfSrchRadius * fCos;
        ptROICtr.y = pstCmd->ptExpectedCircleCtr.y + fMiddleOfSrchRadius * fSin;
        cv::RotatedRect rotatedRectROI;
        rotatedRectROI.angle = fAngle;
        rotatedRectROI.center = ptROICtr;
        rotatedRectROI.size.width  = CHECK_ROI_WIDTH;
        rotatedRectROI.size.height = CHECK_ROI_HEIGHT;

        if (! isAutoMode()) {
            VectorOfVectorOfPoint vevVecPoints;
            vevVecPoints.push_back(CalcUtils::getCornerOfRotatedRect(rotatedRectROI));
            cv::polylines(pstRpy->matResultImg, vevVecPoints, true, CYAN_SCALAR, 1);
        }

        cv::Mat matROI;
        _extractRotatedROI(matGray, rotatedRectROI, matROI);
        PR_CALIPER_DIR enDir = PR_CALIPER_DIR::DARK_TO_BRIGHT;
        if (PR_OBJECT_ATTRIBUTE::BRIGHT == pstCmd->enInnerAttribute)
            enDir = PR_CALIPER_DIR::BRIGHT_TO_DARK;
        else if (PR_OBJECT_ATTRIBUTE::DARK == pstCmd->enInnerAttribute)
            enDir = PR_CALIPER_DIR::DARK_TO_BRIGHT;

        int nEdgePos1, nEdgePos2;
        bool bResult = _findEdgePosInX(matROI, matGuassinKernel, enDir, pstCmd->enSelectEdge, pstCmd->nEdgeThreshold, pstCmd->bFindCirclePair, nEdgePos1, nEdgePos2);

        if (bResult) {
            cv::Point2f ptFindPos;
            ptFindPos.x = pstCmd->ptExpectedCircleCtr.x + (nEdgePos1 + pstCmd->fMinSrchRadius) * fCos;
            ptFindPos.y = pstCmd->ptExpectedCircleCtr.y + (nEdgePos1 + pstCmd->fMinSrchRadius) * fSin;
            bool bMaskedOut = false;
            if (! pstCmd->matMask.empty() && pstCmd->matMask.at<uchar>(ptFindPos) == 0)
                bMaskedOut = true;
            if (! bMaskedOut)
                vecPoints.push_back(ptFindPos);
            if (pstCmd->bFindCirclePair) {
                ptFindPos.x = pstCmd->ptExpectedCircleCtr.x + (nEdgePos2 + pstCmd->fMinSrchRadius) * fCos;
                ptFindPos.y = pstCmd->ptExpectedCircleCtr.y + (nEdgePos2 + pstCmd->fMinSrchRadius) * fSin;

                bool bMaskedOut = false;
                if (! pstCmd->matMask.empty() && pstCmd->matMask.at<uchar>(ptFindPos) == 0)
                    bMaskedOut = true;
                if (! bMaskedOut)
                    vecPoints2.push_back(ptFindPos);
            }
        }
    }

    if (vecPoints.size() > 3) {
        if (pstCmd->bFindCirclePair) {
            if (vecPoints2.size() > 3) {
                Fitting::fitParallelCircleRemoveStray(vecPoints, vecPoints2, vecKeepPoints1, vecKeepPoints2, pstRpy->ptCircleCtr, pstRpy->fRadius, pstRpy->fRadius2, pstCmd->fRmStrayPointRatio);
                pstRpy->enStatus = VisionStatus::OK;
            }
            else {
                WriteLog("Find circle caliper cannot find enough edge points, please check the input parameter and image.");
                pstRpy->enStatus = VisionStatus::CALIPER_NOT_ENOUGH_EDGE_POINTS;
            }
        }
        else {
            auto fitResult = Fitting::fitCircleRemoveStray(vecPoints, pstCmd->fRmStrayPointRatio);
            pstRpy->ptCircleCtr = fitResult.center;
            pstRpy->fRadius = fitResult.size.width / 2.f;
            if (pstRpy->fRadius > 0.f)
                pstRpy->enStatus = VisionStatus::OK;
            else {
                WriteLog("Too much noise to fit the circle in findCircle.");
                pstRpy->enStatus = VisionStatus::TOO_MUCH_NOISE_TO_FIT;
            }
        }
    }
    else {
        WriteLog("Find circle caliper cannot find enough edge points, please check the input parameter and image.");
        pstRpy->enStatus = VisionStatus::CALIPER_NOT_ENOUGH_EDGE_POINTS;
    }

    if (! isAutoMode()) {
        for (const auto &point : vecPoints)
            cv::circle(pstRpy->matResultImg, point, 3, BLUE_SCALAR, 2);

        for (const auto &point : vecKeepPoints1)
            cv::circle(pstRpy->matResultImg, point, 3, GREEN_SCALAR, 2);

        if (pstCmd->bFindCirclePair) {
            for (const auto &point : vecPoints2)
                cv::circle(pstRpy->matResultImg, point, 3, BLUE_SCALAR, 2);

            for (const auto &point : vecKeepPoints2)
                cv::circle(pstRpy->matResultImg, point, 3, GREEN_SCALAR, 2);
        }

        cv::circle(pstRpy->matResultImg, pstCmd->ptExpectedCircleCtr, 3, RED_SCALAR, 1);
        cv::circle(pstRpy->matResultImg, pstRpy->ptCircleCtr, 3, GREEN_SCALAR, 1);

        if (VisionStatus::OK == pstRpy->enStatus) {
            cv::circle(pstRpy->matResultImg, pstRpy->ptCircleCtr, ToInt32(pstRpy->fRadius), GREEN_SCALAR, 1);
            if (pstCmd->bFindCirclePair)
                cv::circle(pstRpy->matResultImg, pstRpy->ptCircleCtr, ToInt32(pstRpy->fRadius2), GREEN_SCALAR, 1);
        }
    }
    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::inspCircle(const PR_INSP_CIRCLE_CMD *const pstCmd, PR_INSP_CIRCLE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char chArrMsg[1000];

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (! pstCmd->matMask.empty()) {
        if (pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d )",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matMask.channels() != 1) {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspCircle);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    cv::Mat matGray;

    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI.clone();

    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
        showImage("Before mask out ROI image", matGray);

    if (! pstCmd->matMask.empty()) {
        cv::Mat matROIMask(pstCmd->matMask, pstCmd->rectROI);
        matROIMask = cv::Scalar(255) - matROIMask;
        matGray.setTo(cv::Scalar(0), matROIMask);
    }

    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
        showImage("mask out ROI image", matGray);

    VectorOfPoint vecPoints;
    cv::findNonZero(matGray, vecPoints);

    if (vecPoints.size() < 3) {
        WriteLog("Not enough points to fit circle");
        pstRpy->enStatus = VisionStatus::NOT_ENOUGH_POINTS_TO_FIT;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    cv::RotatedRect fitResult = Fitting::fitCircle(vecPoints);

    if (fitResult.center.x <= 0 || fitResult.center.y <= 0 || fitResult.size.width <= 0) {
        WriteLog("Failed to fit circle");
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    pstRpy->ptCircleCtr = fitResult.center + cv::Point2f(ToFloat(pstCmd->rectROI.x), ToFloat(pstCmd->rectROI.y));
    pstRpy->fDiameter = (fitResult.size.width + fitResult.size.height) / 2;

    std::vector<float> vecDistance;
    for (const auto &point : vecPoints) {
        cv::Point2f pt2f(point);
        vecDistance.push_back(CalcUtils::distanceOf2Point<float>(pt2f, fitResult.center) - pstRpy->fDiameter / 2);
    }
    pstRpy->fRoundness = ToFloat(CalcUtils::calcStdDeviation<float>(vecDistance));
    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::circle(pstRpy->matResultImg, pstRpy->ptCircleCtr, (int)pstRpy->fDiameter / 2, BLUE_SCALAR, 2);

    pstRpy->enStatus = VisionStatus::OK;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::ocr(PR_OCR_CMD *pstCmd, PR_OCR_RPY *pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseOcr);

    VisionStatus enStatus = VisionStatus::OK;
    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    if (pstCmd->enDirection != PR_DIRECTION::UP) {
        float fAngle = 0.f;
        cv::Size sizeDest(pstCmd->rectROI.width, pstCmd->rectROI.height);
        switch (pstCmd->enDirection) {
        case PR_DIRECTION::LEFT:
            cv::transpose(matROI, matROI);
            cv::flip(matROI, matROI, 1); //transpose+flip(1)=CW
            break;
        case PR_DIRECTION::DOWN:
            cv::flip(matROI, matROI, -1);    //flip(-1)=180
            break;
        case PR_DIRECTION::RIGHT:
            cv::transpose(matROI, matROI);
            cv::flip(matROI, matROI, 0); //transpose+flip(0)=CCW
            break;
        default: break;
        }
    }

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("OCR ROI Image", matROI);

    char *dataPath = "./tessdata";
    String strOcrCharList = Config::GetInstance()->getOcrCharList();
    if (nullptr == _ptrOcrTesseract)
        _ptrOcrTesseract = cv::text::OCRTesseract::create(dataPath, "eng+eng1", strOcrCharList.c_str());
    _ptrOcrTesseract->run(matROI, pstRpy->strResult);
    if (pstRpy->strResult.empty()) {
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

/*static*/ VisionStatus VisionAlgorithm::colorToGray(PR_COLOR_TO_GRAY_CMD *pstCmd, PR_COLOR_TO_GRAY_RPY *pstRpy) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matInputImg.channels() != 3) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "Convert from color to gray, the input image channel number %d not equal to 3", pstCmd->matInputImg.channels());
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    std::vector<cv::Mat> vecChannels;
    cv::split(pstCmd->matInputImg, vecChannels);

    vecChannels[0] = vecChannels[0] * pstCmd->stRatio.fRatioB;
    vecChannels[1] = vecChannels[1] * pstCmd->stRatio.fRatioG;
    vecChannels[2] = vecChannels[2] * pstCmd->stRatio.fRatioR;

    cv::merge(vecChannels, pstRpy->matResultImg);
    cv::cvtColor(pstRpy->matResultImg, pstRpy->matResultImg, CV_BGR2GRAY);

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::filter(PR_FILTER_CMD *pstCmd, PR_FILTER_RPY *pstRpy) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (PR_FILTER_TYPE::GAUSSIAN_FILTER == pstCmd->enType &&
        ((pstCmd->szKernel.width % 2 != 1) || (pstCmd->szKernel.height % 2 != 1))) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof (chArrMsg), "Guassian filter kernel size ( %d, %d ) is invalid", pstCmd->szKernel.width, pstCmd->szKernel.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::GUASSIAN_FILTER_KERNEL_INVALID;
        return VisionStatus::GUASSIAN_FILTER_KERNEL_INVALID;
    }

    if (PR_FILTER_TYPE::MEDIAN_FILTER == pstCmd->enType &&
        ((pstCmd->szKernel.width % 2 != 1) || (pstCmd->szKernel.height % 2 != 1))) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof (chArrMsg), "Median filter kernel size ( %d, %d ) is invalid", pstCmd->szKernel.width, pstCmd->szKernel.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::MEDIAN_FILTER_KERNEL_INVALID;
        return VisionStatus::MEDIAN_FILTER_KERNEL_INVALID;
    }

    MARK_FUNCTION_START_TIME;

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matROI(pstRpy->matResultImg, pstCmd->rectROI);

    VisionStatus enStatus = VisionStatus::OK;
    if (PR_FILTER_TYPE::NORMALIZED_BOX_FILTER == pstCmd->enType) {
        cv::blur(matROI, matROI, pstCmd->szKernel);
    }
    else if (PR_FILTER_TYPE::GAUSSIAN_FILTER == pstCmd->enType) {
        cv::GaussianBlur(matROI, matROI, pstCmd->szKernel, pstCmd->dSigmaX, pstCmd->dSigmaY);
    }
    else if (PR_FILTER_TYPE::MEDIAN_FILTER == pstCmd->enType) {
        cv::medianBlur(matROI, matROI, pstCmd->szKernel.width);
    }
    else if (PR_FILTER_TYPE::BILATERIAL_FILTER == pstCmd->enType) {
        cv::Mat matResultImg;
        cv::bilateralFilter(matROI, matResultImg, pstCmd->nDiameter, pstCmd->dSigmaColor, pstCmd->dSigmaSpace);
        if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
            showImage("Bilateral filter result", matResultImg);
        }
        matResultImg.copyTo(matROI);
    }
    else {
        enStatus = VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = enStatus;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::removeCC(PR_REMOVE_CC_CMD *pstCmd, PR_REMOVE_CC_RPY *pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (pstCmd->nConnectivity != 4 && pstCmd->nConnectivity != 8) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof (chArrMsg), "The connectivity %d is not 4 or 8", pstCmd->nConnectivity);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseRemoveCC);

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matROI(pstRpy->matResultImg, pstCmd->rectROI);
    cv::Mat matGray = matROI;
    if (matROI.channels() == 3)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);

    cv::Mat matLabels;
    cv::connectedComponents(matGray, matLabels, pstCmd->nConnectivity, CV_32S);
    TimeLog::GetInstance()->addTimeLog("cv::connectedComponents", stopWatch.AbsNow() - functionStart);

    double minValue = 0., maxValue = 0.;
    cv::minMaxIdx(matLabels, &minValue, &maxValue);

    cv::Mat matRemoveMask = cv::Mat::zeros(matLabels.size(), CV_8UC1);

    if (maxValue > Config::GetInstance()->getRemoveCCMaxComponents()) {
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
        nCurrentLabelCount = nTotalPixels - cv::countNonZero(matTemp);
        if (0 < nCurrentLabelCount && nCurrentLabelCount < pstCmd->fAreaThreshold) {
            matRemoveMask += CalcUtils::genMaskByValue<Int32>(matLabels, nCurrentLabel);
            ++ pstRpy->nRemovedCC;
        }
        ++ nCurrentLabel;
    } while (nCurrentLabelCount > 0);

    pstRpy->nTotalCC = nCurrentLabel - 1;

    matGray.setTo(0, matRemoveMask);

    if (matROI.channels() == 3)
        cv::cvtColor(matGray, matROI, CV_GRAY2BGR);

    pstRpy->enStatus = VisionStatus::OK;
    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::detectEdge(PR_DETECT_EDGE_CMD *pstCmd, PR_DETECT_EDGE_RPY *pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseDetectEdge);

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matROI(pstRpy->matResultImg, pstCmd->rectROI);

    cv::Mat matGray = matROI;
    if (pstCmd->matInputImg.channels() == 3)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);

    cv::Mat matResultImg;
    cv::Canny(matGray, matResultImg, pstCmd->nThreshold1, pstCmd->nThreshold2, pstCmd->nApertureSize);

    if (pstCmd->matInputImg.channels() == 3)
        cv::cvtColor(matResultImg, matROI, CV_GRAY2BGR);
    else
        matResultImg.copyTo(matROI);

    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_fillHoleByContour(const cv::Mat &matInputImg, cv::Mat &matOutput, PR_OBJECT_ATTRIBUTE enAttribute) {
    if (matInputImg.channels() > 1)
        cv::cvtColor(matInputImg, matOutput, CV_BGR2GRAY);
    else
        matOutput = matInputImg.clone();

    if (PR_OBJECT_ATTRIBUTE::DARK == enAttribute)
        cv::bitwise_not(matOutput, matOutput);

    VectorOfVectorOfPoint contours;
    cv::findContours(matOutput, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); ++ i)
        cv::drawContours(matOutput, contours, ToInt32(i), cv::Scalar::all(PR_MAX_GRAY_LEVEL), CV_FILLED);

    if (PR_OBJECT_ATTRIBUTE::DARK == enAttribute)
        cv::bitwise_not(matOutput, matOutput);

    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_fillHoleByMorph(const cv::Mat      &matInputImg,
    cv::Mat            &matOutput,
    PR_OBJECT_ATTRIBUTE enAttribute,
    cv::MorphShapes     enMorphShape,
    cv::Size            szMorphKernel,
    Int16               nMorphIteration) {
    char chArrMsg[1000];
    if (szMorphKernel.width <= 0 || szMorphKernel.height <= 0) {
        _snprintf(chArrMsg, sizeof (chArrMsg), "The size of morph kernel ( %d, %d) is invalid.", szMorphKernel.width, szMorphKernel.height);
        WriteLog(chArrMsg);
        return VisionStatus::INVALID_PARAM;
    }

    if (nMorphIteration <= 0) {
        _snprintf(chArrMsg, sizeof (chArrMsg), "The morph iteration number %d is invalid.", nMorphIteration);
        WriteLog(chArrMsg);
        return VisionStatus::INVALID_PARAM;
    }
    cv::Mat matKernal = cv::getStructuringElement(enMorphShape, szMorphKernel);
    cv::MorphTypes morphType = PR_OBJECT_ATTRIBUTE::BRIGHT == enAttribute ? cv::MORPH_CLOSE : cv::MORPH_OPEN;
    cv::morphologyEx(matInputImg, matOutput, morphType, matKernal, cv::Point(-1, -1), nMorphIteration);

    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::fillHole(PR_FILL_HOLE_CMD *pstCmd, PR_FILL_HOLE_RPY *pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseFillHole);

    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matROI(pstRpy->matResultImg, pstCmd->rectROI);
    cv::Mat matGray;

    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI.clone();

    cv::Mat matFillHoleResult;
    if (PR_FILL_HOLE_METHOD::CONTOUR == pstCmd->enMethod)
        pstRpy->enStatus = _fillHoleByContour(matGray, matFillHoleResult, pstCmd->enAttribute);
    else
        pstRpy->enStatus = _fillHoleByMorph(matGray, matFillHoleResult, pstCmd->enAttribute, pstCmd->enMorphShape, pstCmd->szMorphKernel, pstCmd->nMorphIteration);

    if (pstCmd->matInputImg.channels() == 3)
        cv::cvtColor(matFillHoleResult, matROI, CV_GRAY2BGR);
    else
        matFillHoleResult.copyTo(matROI);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::pickColor(const PR_PICK_COLOR_CMD *const pstCmd, PR_PICK_COLOR_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matInputImg.channels() <= 1) {
        WriteLog("Input image must be color image.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (PR_PICK_COLOR_METHOD::SELECT_POINT == pstCmd->enMethod && !pstCmd->rectROI.contains(pstCmd->ptPick)) {
        WriteLog("The pick point must in selected ROI");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCasePickColor);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);

    std::vector<int> vecValue(3);
    if (PR_PICK_COLOR_METHOD::SELECT_POINT == pstCmd->enMethod) {
        auto vecColor = pstCmd->matInputImg.at<cv::Vec3b>(pstCmd->ptPick);
        vecValue[0] = vecColor[0]; vecValue[1] = vecColor[1]; vecValue[2] = vecColor[2];
    }else {
        vecValue[0] = ToInt32(pstCmd->scalarSelect[0]); vecValue[1] = ToInt32(pstCmd->scalarSelect[1]); vecValue[2] = ToInt32(pstCmd->scalarSelect[2]);
    }

    auto matResultMask = _pickColor(vecValue, matROI, pstCmd->nColorDiff, pstCmd->nGrayDiff, pstRpy->nPickPointCount);

    pstRpy->enStatus = VisionStatus::OK;
    pstRpy->matResultImg = pstCmd->matInputImg.clone();
    cv::Mat matResultROI(pstRpy->matResultImg, pstCmd->rectROI);
    matResultROI.setTo(cv::Scalar::all(0));
    matResultROI.setTo(cv::Scalar::all(PR_MAX_GRAY_LEVEL), matResultMask);

    pstRpy->matResultMask = cv::Mat::zeros(pstCmd->matInputImg.size(), CV_8UC1);
    cv::Mat matMaskROI(pstRpy->matResultMask, pstCmd->rectROI);
    matMaskROI.setTo(cv::Scalar::all(PR_MAX_GRAY_LEVEL), matResultMask);

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ cv::Mat VisionAlgorithm::_pickColor(const std::vector<int> &vecValue, const cv::Mat &matColor, int nColorDiff, int nGrayDiff, UInt32 &nPointCountOut) {
    assert(vecValue.size() == 3);

    std::vector<cv::Mat> vecMat;
    cv::split(matColor, vecMat);

    cv::Mat matGray;
    cv::cvtColor(matColor, matGray, CV_BGR2GRAY);

    int Tt = ToInt32(0.114f * vecValue[0] + 0.587f * vecValue[1] + 0.299f * vecValue[2]);

    auto maxElement = std::max_element(vecValue.begin(), vecValue.end());
    auto maxIndex = std::distance(vecValue.begin(), maxElement);
    std::vector<size_t> vecExcludedIndex{0, 1, 2};
    vecExcludedIndex.erase(vecExcludedIndex.begin() + maxIndex);
    cv::Mat matResultMask = cv::Mat::zeros(matColor.size(), CV_8UC1);
    int diffLowerLimit1 = std::max(1, vecValue[maxIndex] - vecValue[vecExcludedIndex[0]] - nColorDiff);
    int diffUpLimit1    = vecValue[maxIndex] - vecValue[vecExcludedIndex[0]] + nColorDiff;

    int diffLowerLimit2 = std::max(1, vecValue[maxIndex] - vecValue[vecExcludedIndex[1]] - nColorDiff);
    int diffUpLimit2    = vecValue[maxIndex] - vecValue[vecExcludedIndex[1]] + nColorDiff;

    nPointCountOut = 0;
    for (int row = 0; row < matColor.rows; ++ row)
    for (int col = 0; col < matColor.cols; ++ col) {
        auto targetColorValue = vecMat[maxIndex].at<uchar>(row, col);
        auto compareValue1 = vecMat[vecExcludedIndex[0]].at<uchar>(row, col);
        auto compareValue2 = vecMat[vecExcludedIndex[1]].at<uchar>(row, col);
        if ((targetColorValue - compareValue1) > diffLowerLimit1 &&
            (targetColorValue - compareValue2) > diffLowerLimit2 &&
            (targetColorValue - compareValue1) < diffUpLimit1 &&
            (targetColorValue - compareValue2) < diffUpLimit2 &&
            abs((int)matGray.at<uchar>(row, col) - Tt) < nGrayDiff) {
            matResultMask.at<uchar>(row, col) = PR_MAX_GRAY_LEVEL;
            ++ nPointCountOut;
        }
    }

    return matResultMask;
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
    VectorOfPoint   &vecFailedRowCol) {
    VisionStatus enStatus;
    cv::cvtColor(mat, matCornerPointsImg, CV_GRAY2BGR);
    float fStepOffset = 0.f;
    float fStepSizeY = fStepSize;
    float fMinCorrelation = fMinTmplMatchScore / ConstToPercentage;
    cv::Scalar colorOfResultPoint;
    bool bStepUpdatedX = false, bStepUpdatedY = false;
    for (int row = 0; row < szBoardPattern.height; ++ row)
    for (int col = 0; col < szBoardPattern.width; ++ col) {
        cv::Rect rectSrchROI(ToInt32(startPoint.x + col * fStepSize - row * fStepOffset), ToInt32(startPoint.y + row * fStepSizeY + col * fStepOffset), nSrchSize, nSrchSize);
        if (rectSrchROI.x < 0) rectSrchROI.x = 0;
        if (rectSrchROI.y < 0) rectSrchROI.y = 0;
        if ((rectSrchROI.x + matTmpl.cols) > mat.cols || (rectSrchROI.y + matTmpl.rows) > mat.rows) {
            WriteLog("The chessboard pattern is not correct.");
            return VisionStatus::CHESSBOARD_PATTERN_NOT_CORRECT;
        }
        if ((rectSrchROI.x + rectSrchROI.width)  > mat.cols) rectSrchROI.width = mat.cols - rectSrchROI.x;
        if ((rectSrchROI.y + rectSrchROI.height) > mat.rows) rectSrchROI.height = mat.rows - rectSrchROI.y;
        //#ifdef _DEBUG
        //        std::cout << "Srch ROI " << rectSrchROI.x << ", " << rectSrchROI.y << ", " << rectSrchROI.width << ", " << rectSrchROI.height << std::endl;
        //#endif
        cv::Mat matSrchROI(mat, rectSrchROI);
        cv::Point2f ptResult;
        float fRotation, fCorrelation = 0.f;
        enStatus = MatchTmpl::matchTemplate(matSrchROI, matTmpl, true, PR_OBJECT_MOTION::TRANSLATION, ptResult, fRotation, fCorrelation);
        if (VisionStatus::OK != enStatus)
            fCorrelation = 0;

        ptResult.x += rectSrchROI.x;
        ptResult.y += rectSrchROI.y;

        if (fCorrelation > fMinCorrelation) {
            vecCorners.push_back(ptResult);
            colorOfResultPoint = GREEN_SCALAR;
        }
        else {
            vecFailedRowCol.emplace_back(cv::Point(col, row));
            colorOfResultPoint = RED_SCALAR;
        }

        if (vecCorners.size() == 2) {
            fStepSize = CalcUtils::distanceOf2Point(vecCorners[1], vecCorners[0]);
            fStepOffset = vecCorners[1].y - vecCorners[0].y;
        }
        else if (vecCorners.size() == (szBoardPattern.width + 1)) {
            fStepSizeY = vecCorners[szBoardPattern.width].y - vecCorners[0].y;
        }

        cv::rectangle(matCornerPointsImg, rectSrchROI, cv::Scalar(255, 0, 0), 2);
        cv::circle(matCornerPointsImg, ptResult, 10, colorOfResultPoint, 2);
        cv::line(matCornerPointsImg, cv::Point2f(ptResult.x - 4, ptResult.y), cv::Point2f(ptResult.x + 4, ptResult.y), colorOfResultPoint, 1);
        cv::line(matCornerPointsImg, cv::Point2f(ptResult.x, ptResult.y - 4), cv::Point2f(ptResult.x, ptResult.y + 4), colorOfResultPoint, 1);
    }

    if (ToInt32(vecCorners.size()) < Config::GetInstance()->getMinPointsToCalibCamera()) {
        return VisionStatus::NOT_FIND_ENOUGH_CB_CORNERS;
    }
    return VisionStatus::OK;
}

/*static*/ void VisionAlgorithm::_calcBoardCornerPositions(const cv::Size &szBoardSize, float squareSize, const VectorOfPoint &vecFailedColRow, std::vector<cv::Point3f> &corners, CalibPattern patternType /*= CHESSBOARD*/) {
    auto isNeedToSkip = [&vecFailedColRow](int col, int row) -> bool {
        for (const auto &point : vecFailedColRow) {
            if (point.x == col && point.y == row)
                return true;
        }
        return false;
    };
    corners.clear();

    switch (patternType) {
    case CHESSBOARD:
    case CIRCLES_GRID:
        for (int row = 0; row < szBoardSize.height; ++ row)
        for (int col = 0; col < szBoardSize.width; ++ col) {
            if (! isNeedToSkip(col, row))
                corners.push_back(cv::Point3f(col * squareSize, row * squareSize, 0));
        }
        break;

    case ASYMMETRIC_CIRCLES_GRID:
        for (int i = 0; i < szBoardSize.height; ++ i)
        for (int j = 0; j < szBoardSize.width; ++ j)
            corners.push_back(cv::Point3f((2 * j + i % 2)*squareSize, i*squareSize, 0));
        break;
    default:
        break;
    }
}

/*static*/ float VisionAlgorithm::_findChessBoardBlockSize(const cv::Mat &matInputImg, const cv::Size szBoardPattern) {
    cv::Size2f size;
    int nRoughBlockSize = matInputImg.rows / szBoardPattern.width / 2;
    cv::Mat matROI(matInputImg, cv::Rect(matInputImg.cols / 2 - nRoughBlockSize * 2, matInputImg.rows / 2 - nRoughBlockSize * 2, nRoughBlockSize * 4, nRoughBlockSize * 4));  //Use the ROI 200x200 in the center of the image.
    cv::Mat matFilter;
    cv::GaussianBlur(matROI, matFilter, cv::Size(3, 3), 1, 1);
    cv::Mat matThreshold;
    int nThreshold = _autoThreshold(matFilter);
    cv::threshold(matFilter, matThreshold, nThreshold + 20, 255, cv::THRESH_BINARY_INV);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("findChessBoardBlockSize Threshold image", matThreshold);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    float fMaxBlockArea = 0;
    for (const auto &contour : contours) {
        auto area = cv::contourArea(contour);
        if (area > (nRoughBlockSize - 1) * (nRoughBlockSize - 1)) {
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

    if (fMaxBlockArea <= 0) {
        cv::threshold(matFilter, matThreshold, nThreshold, 255, cv::THRESH_BINARY_INV);
        cv::Mat matKernal = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, {6, 6});
        cv::morphologyEx(matThreshold, matThreshold, cv::MorphTypes::MORPH_CLOSE, matKernal, cv::Point(-1, -1), 3);
        if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
            showImage("After morphologyEx image", matThreshold);
        }
        cv::findContours(matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (const auto &contour : contours) {
            auto area = cv::contourArea(contour);
            if (area > 1000) {
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

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        cv::Mat matDisplay;
        cv::cvtColor(matROI, matDisplay, CV_GRAY2BGR);
        for (size_t i = 0; i < contours.size(); ++ i) {
            cv::drawContours(matDisplay, contours, ToInt32(i), cv::Scalar(0, 0, 255));
        }
        showImage("findChessBoardBlockSize contour image", matDisplay);
    }
    return  (size.width + size.height) / 2;
}

/*static*/ cv::Point2f VisionAlgorithm::_findFirstChessBoardCorner(const cv::Mat &matInputImg, float fBlockSize) {
    int nCornerRoiSize = ToInt32(fBlockSize * 3);
    if (nCornerRoiSize < 300) nCornerRoiSize = 300;
    cv::Mat matROI(matInputImg, cv::Rect(0, 0, nCornerRoiSize, nCornerRoiSize));
    cv::Mat matFilter, matThreshold;
    cv::GaussianBlur(matROI, matFilter, cv::Size(3, 3), 0.5, 0.5);
    int nThreshold = _autoThreshold(matFilter);
    nThreshold += 20;
    VectorOfVectorOfPoint contours, vecDrawContours;
    VectorOfPoint2f vecBlockCenters;

    //If didn't find the block, it may because the two blocks connected together make findContour find two connected blocks.
    //Need to use Morphology method to remove the dark connecter between them.
    cv::threshold(matFilter, matThreshold, nThreshold, 255, cv::THRESH_BINARY_INV);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("Before morphologyEx image", matThreshold);
    }

    float fBlockAreaMin = (fBlockSize - 5) * (fBlockSize - 5);

    if (fBlockSize > 60) {
        cv::Mat matKernal = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, {7, 7});
        cv::morphologyEx(matThreshold, matThreshold, cv::MorphTypes::MORPH_CLOSE, matKernal, cv::Point(-1, -1), 3);
        if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
            showImage("After morphologyEx image", matThreshold);
        }
        fBlockAreaMin = (fBlockSize - 10) * (fBlockSize - 10);
    }
    cv::findContours(matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
        auto area = cv::contourArea(contour);
        if (area > fBlockAreaMin) {
            cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
            if (fabs(rotatedRect.size.width - rotatedRect.size.height) / (rotatedRect.size.width + rotatedRect.size.height) < 0.1
                && fabs(rotatedRect.size.width - fBlockSize) / fBlockSize < 0.2) {
                vecDrawContours.push_back(contour);
                vecBlockCenters.push_back(rotatedRect.center);
            }
        }
    }

    float minDistanceToZero = std::numeric_limits<float>::max();
    cv::Point ptUpperLeftBlockCenter;
    for (const auto point : vecBlockCenters) {
        auto distanceToZero = point.x * point.x + point.y * point.y;
        if (distanceToZero < minDistanceToZero) {
            ptUpperLeftBlockCenter = point;
            minDistanceToZero = distanceToZero;
        }
    }
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        cv::Mat matDisplay;
        cv::cvtColor(matROI, matDisplay, CV_GRAY2BGR);
        for (size_t i = 0; i < contours.size(); ++ i) {
            cv::drawContours(matDisplay, contours, ToInt32(i), RED_SCALAR);
        }
        showImage("findChessBoardBlockSize contour image", matDisplay);
    }
    return cv::Point2f(ptUpperLeftBlockCenter.x - fBlockSize / 2, ptUpperLeftBlockCenter.y - fBlockSize / 2);
}

/*static*/ VisionStatus VisionAlgorithm::calibrateCamera(const PR_CALIBRATE_CAMERA_CMD *const pstCmd, PR_CALIBRATE_CAMERA_RPY *const pstRpy, bool bReplay) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char chArrMsg[1000];
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->fPatternDist <= 0) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "Pattern distance %f is invalid.", pstCmd->fPatternDist);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->szBoardPattern.width <= 0 || pstCmd->szBoardPattern.height <= 0) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "Board pattern size %d, %d is invalid.", pstCmd->szBoardPattern.width, pstCmd->szBoardPattern.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseCalibrateCamera);

    cv::Mat matGray = pstCmd->matInputImg;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(pstCmd->matInputImg, matGray, CV_BGR2GRAY);

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

    float fBlockSize = _findChessBoardBlockSize(matGray, pstCmd->szBoardPattern);
    if (fBlockSize <= 5) {
        WriteLog("Failed to find chess board block size");
        pstRpy->enStatus = VisionStatus::FIND_CHESS_BOARD_BLOCK_SIZE_FAIL;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    float fStepSize = fBlockSize * 2;
    int nSrchSize = ToInt32(fStepSize - 20);
    if (nSrchSize > 200) nSrchSize = 200;
    if (nSrchSize < (matTmpl.rows + 10)) nSrchSize = matTmpl.rows + 10;
    cv::Point2f ptFirstCorer = _findFirstChessBoardCorner(matGray, fBlockSize);
    if (ptFirstCorer.x <= 0 || ptFirstCorer.y <= 0) {
        WriteLog("Failed to find chess board first corner.");
        pstRpy->enStatus = VisionStatus::FIND_FIRST_CB_CORNER_FAIL;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    cv::Point ptChessBoardSrchStartPoint = cv::Point(ToInt32(ptFirstCorer.x - nSrchSize / 2), ToInt32(ptFirstCorer.y - nSrchSize / 2));
    if (ptChessBoardSrchStartPoint.x < 0 || ptChessBoardSrchStartPoint.y < 0) {
        ptChessBoardSrchStartPoint.x += ToInt32(fBlockSize);
        ptChessBoardSrchStartPoint.y += ToInt32(fBlockSize);
    }
    VectorOfPoint vecFailedColRow;
    pstRpy->enStatus = _findChessBoardCorners(matGray, matTmpl, pstRpy->matCornerPointsImg, ptChessBoardSrchStartPoint, fStepSize, nSrchSize, pstCmd->fMinTmplMatchScore, pstCmd->szBoardPattern, vecVecImagePoints[0], vecFailedColRow);
    if (VisionStatus::OK != pstRpy->enStatus) {
        WriteLog("Failed to find chess board corners.");
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    pstRpy->vecImagePoints = vecVecImagePoints[0];

    _calcBoardCornerPositions(pstCmd->szBoardPattern, pstCmd->fPatternDist, vecFailedColRow, vevVecObjectPoints[0], CHESSBOARD);
    vevVecObjectPoints.resize(vecVecImagePoints.size(), vevVecObjectPoints[0]);
    pstRpy->vecObjectPoints = vevVecObjectPoints[0];

    double rms = cv::calibrateCamera(vevVecObjectPoints, vecVecImagePoints, imageSize, pstRpy->matIntrinsicMatrix,
        pstRpy->matDistCoeffs, rvecs, tvecs);

    cv::Rodrigues(rvecs[0], matRotation);
    pstRpy->matExtrinsicMatrix = matRotation.clone();
    cv::transpose(pstRpy->matExtrinsicMatrix, pstRpy->matExtrinsicMatrix);
    matTransformation = tvecs[0];
    cv::transpose(matTransformation, matTransformation);
    pstRpy->matExtrinsicMatrix.push_back(matTransformation);
    cv::transpose(pstRpy->matExtrinsicMatrix, pstRpy->matExtrinsicMatrix);

    //To output the initial camera matrix to do comarison.
    {
        pstRpy->matInitialIntrinsicMatrix = cv::initCameraMatrix2D(vevVecObjectPoints, vecVecImagePoints, imageSize, 0.);
        cv::Mat rvec, tvec, matRotation, matTransformation;
        bool bResult = cv::solvePnP(vevVecObjectPoints[0], vecVecImagePoints[0], pstRpy->matInitialIntrinsicMatrix, cv::Mat(), rvec, tvec);
        cv::Rodrigues(rvec, matRotation);
        pstRpy->matInitialExtrinsicMatrix = matRotation.clone();
        cv::transpose(pstRpy->matInitialExtrinsicMatrix, pstRpy->matInitialExtrinsicMatrix);
        matTransformation = tvec;
        cv::transpose(matTransformation, matTransformation);
        pstRpy->matInitialExtrinsicMatrix.push_back(matTransformation);
        cv::transpose(pstRpy->matInitialExtrinsicMatrix, pstRpy->matInitialExtrinsicMatrix);
    }

    double z = pstRpy->matExtrinsicMatrix.at<double>(2, 3); //extrinsic matrix is 3x4 matric, the lower right element is z.
    double fx = pstRpy->matIntrinsicMatrix.at<double>(0, 0);
    double fy = pstRpy->matIntrinsicMatrix.at<double>(1, 1);
    pstRpy->dResolutionX = PR_MM_TO_UM * z / fx;
    pstRpy->dResolutionY = PR_MM_TO_UM * z / fy;

    cv::Mat matMapX, matMapY;
    cv::Mat matNewCemraMatrix = cv::getOptimalNewCameraMatrix(pstRpy->matIntrinsicMatrix, pstRpy->matDistCoeffs, imageSize, 1, imageSize, 0);
    cv::initUndistortRectifyMap(pstRpy->matIntrinsicMatrix, pstRpy->matDistCoeffs, cv::Mat(), matNewCemraMatrix,
        imageSize, CV_32FC1, matMapX, matMapY);

    cv::convertMaps(matMapX, matMapY, matMapX, matMapY, CV_16SC2, true);

    pstRpy->vecMatRestoreMap.push_back(matMapX);
    pstRpy->vecMatRestoreMap.push_back(matMapY);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::restoreImage(const PR_RESTORE_IMG_CMD *const pstCmd, PR_RESTORE_IMG_RPY *const pstRpy, bool bReplay/* = false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    char chArrMsg[1000];
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->vecMatRestoreMap.size() != 2) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "The remap mat vector size %d is invalid.", pstCmd->vecMatRestoreMap.size());
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->vecMatRestoreMap[0].size() != pstCmd->matInputImg.size()) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "The remap mat size(%d, %d) is not match with input image size (%d, %d).",
            pstCmd->vecMatRestoreMap[0].size().width, pstCmd->vecMatRestoreMap[0].size().height,
            pstCmd->matInputImg.size().width, pstCmd->matInputImg.size().height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (!pstCmd->vecMatRestoreMap[1].empty() && pstCmd->vecMatRestoreMap[1].size() != pstCmd->matInputImg.size()) {
        _snprintf(chArrMsg, sizeof(chArrMsg), "The remap mat size(%d, %d) is not match with input image size (%d, %d).",
            pstCmd->vecMatRestoreMap[1].size().width, pstCmd->vecMatRestoreMap[1].size().height,
            pstCmd->matInputImg.size().width, pstCmd->matInputImg.size().height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseRestoreImg);

    cv::Mat matInputFloat = pstCmd->matInputImg;
    cv::remap(matInputFloat, pstRpy->matResultImg, pstCmd->vecMatRestoreMap[0], pstCmd->vecMatRestoreMap[1], cv::InterpolationFlags::INTER_NEAREST);

    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calcUndistortRectifyMap(const PR_CALC_UNDISTORT_RECTIFY_MAP_CMD *const pstCmd, PR_CALC_UNDISTORT_RECTIFY_MAP_RPY *const pstRpy) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matIntrinsicMatrix.empty() || pstCmd->matIntrinsicMatrix.cols != 3 || pstCmd->matIntrinsicMatrix.rows != 3) {
        WriteLog("The intrinsic matrix is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    MARK_FUNCTION_START_TIME;

    cv::Mat matMapX, matMapY;
    cv::Mat matNewCemraMatrix = cv::getOptimalNewCameraMatrix(pstCmd->matIntrinsicMatrix, pstCmd->matDistCoeffs, pstCmd->szImage, 1, pstCmd->szImage, 0);
    cv::initUndistortRectifyMap(pstCmd->matIntrinsicMatrix, pstCmd->matDistCoeffs, cv::Mat(), matNewCemraMatrix,
        pstCmd->szImage, CV_32FC1, matMapX, matMapY);
    pstRpy->vecMatRestoreMap.push_back(matMapX);
    pstRpy->vecMatRestoreMap.push_back(matMapY);

    pstRpy->enStatus = VisionStatus::OK;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::autoLocateLead(const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy, bool bReplay/* = false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    pstRpy->vecLeadLocation.clear();
    pstRpy->vecLeadInfo.clear();
    pstRpy->vecLeadRecordId.clear();
    pstRpy->vecPadRecordId.clear();
    
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectChipBody, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    pstRpy->enStatus = _checkInputROI(pstCmd->rectSrchWindow, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (! CalcUtils::isRectInRect(pstCmd->rectChipBody, pstCmd->rectSrchWindow)) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The chip window(%d, %d, %d, %d) should inside search window (%d, %d, %d, %d).",
            pstCmd->rectChipBody.x, pstCmd->rectChipBody.y, pstCmd->rectChipBody.width, pstCmd->rectChipBody.height,
            pstCmd->rectSrchWindow.x, pstCmd->rectSrchWindow.y, pstCmd->rectSrchWindow.width, pstCmd->rectSrchWindow.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseAutoLocateLead);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectSrchWindow), matGray;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI;

    if (pstCmd->matInputImg.channels() > 1)
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);

    if (PR_AUTO_LOCATE_LEAD_METHOD::TEMPLATE_MATCH == pstCmd->enMethod)
        _autoLocateLeadByTmpl(matGray, pstCmd, pstRpy);
    else
        _autoLocateLeadByContour(matGray, pstCmd, pstRpy);

    FINISH_LOGCASE_EX;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_autoLocateLeadByTmpl(const cv::Mat &matGray, const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy) {
    pstRpy->vecLeadInfo.clear();
    pstRpy->vecPadRecordId.clear();
    pstRpy->vecLeadRecordId.clear();
    pstRpy->enStatus = _checkInputROI(pstCmd->rectLeadWindow, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (! CalcUtils::isRectInRect(pstCmd->rectLeadWindow, pstCmd->rectSrchWindow)) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The lead window (%d, %d, %d, %d) should inside search window (%d, %d, %d, %d).",
            pstCmd->rectLeadWindow.x, pstCmd->rectLeadWindow.y, pstCmd->rectLeadWindow.width, pstCmd->rectLeadWindow.height,
            pstCmd->rectSrchWindow.x, pstCmd->rectSrchWindow.y, pstCmd->rectSrchWindow.width, pstCmd->rectSrchWindow.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    cv::Mat matLead(matGray, cv::Rect(pstCmd->rectLeadWindow.x - pstCmd->rectSrchWindow.x, pstCmd->rectLeadWindow.y - pstCmd->rectSrchWindow.y, pstCmd->rectLeadWindow.width, pstCmd->rectLeadWindow.height));

    cv::Mat matPad;
    if (pstCmd->rectPadWindow.area() > 0) {
        pstRpy->enStatus = _checkInputROI(pstCmd->rectPadWindow, pstCmd->matInputImg, AT);
        if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

        if (! CalcUtils::isRectInRect(pstCmd->rectPadWindow, pstCmd->rectSrchWindow)) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The pad window (%d, %d, %d, %d) should inside search window (%d, %d, %d, %d).",
                pstCmd->rectLeadWindow.x, pstCmd->rectLeadWindow.y, pstCmd->rectLeadWindow.width, pstCmd->rectLeadWindow.height,
                pstCmd->rectSrchWindow.x, pstCmd->rectSrchWindow.y, pstCmd->rectSrchWindow.width, pstCmd->rectSrchWindow.height);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
        matPad = cv::Mat(matGray, cv::Rect(pstCmd->rectPadWindow.x - pstCmd->rectSrchWindow.x, pstCmd->rectPadWindow.y - pstCmd->rectSrchWindow.y, pstCmd->rectPadWindow.width, pstCmd->rectPadWindow.height));
    }

    if (pstCmd->vecSrchLeadDirections.empty()) {
        WriteLog("Search for lead direction is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    PR_DIRECTION enLeadDir = PR_DIRECTION::UP;
    // Auto determine the direction of the input template lead.
    if (pstCmd->rectPadWindow.area() > 0) {
        cv::Point ptLeadCtr = cv::Point(pstCmd->rectLeadWindow.x + pstCmd->rectLeadWindow.width / 2, pstCmd->rectLeadWindow.y + pstCmd->rectLeadWindow.height / 2);
        cv::Point ptPadCtr  = cv::Point(pstCmd->rectPadWindow.x  + pstCmd->rectPadWindow.width  / 2, pstCmd->rectPadWindow.y  + pstCmd->rectPadWindow.height  / 2);
        cv::Mat matLeadTmp, matPadTmp;
        // Lead is vertical
        if (abs(ptPadCtr.y - ptLeadCtr.y) >abs(ptPadCtr.x - ptLeadCtr.x)) {
            if (ptPadCtr.y < ptLeadCtr.y) {
                enLeadDir = PR_DIRECTION::UP;
                matLeadTmp = matLead;
                matPadTmp = matPad;
            }else {
                enLeadDir = PR_DIRECTION::DOWN;
                cv::flip(matPad, matPadTmp, -1);    //flip(-1)=180
                cv::flip(matLead, matLeadTmp, -1);    //flip(-1)=180
            }
        }else {
            if (ptPadCtr.x < ptLeadCtr.x) {
                enLeadDir = PR_DIRECTION::LEFT;

                cv::transpose(matPad, matPadTmp);
                cv::flip(matPadTmp, matPadTmp, 1); //transpose+flip(1)=CW
            
                cv::transpose(matLead, matLeadTmp);
                cv::flip(matLeadTmp, matLeadTmp, 1); //transpose+flip(1)=CW
            }else {
                enLeadDir = PR_DIRECTION::RIGHT;

                cv::transpose(matPad, matPadTmp);
                cv::flip(matPadTmp, matPadTmp, 0); //transpose+flip(0)=CCW
                cv::transpose(matLead, matLeadTmp);
                cv::flip(matLeadTmp, matLeadTmp, 0); //transpose+flip(0)=CCW
            }
        }
        matLead = matLeadTmp;
        matPad = matPadTmp;
    }

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("Pad template", matPad);
        showImage("Lead template", matLead);
    }
    
    pstRpy->enStatus = VisionStatus::OK;

    const int nMarginToChip = 40;
    const int nEnlargeChipMargin = 10;
    for (auto enDir : pstCmd->vecSrchLeadDirections) {
        cv::Rect rectSubRegion;
        cv::Mat matLeadTmp, matPadTmp;
        if (PR_DIRECTION::UP == enDir) {
            rectSubRegion = cv::Rect(pstCmd->rectChipBody.x - pstCmd->rectSrchWindow.x - nEnlargeChipMargin, 0, pstCmd->rectChipBody.width + nEnlargeChipMargin * 2, 
                pstCmd->rectChipBody.y - pstCmd->rectSrchWindow.y + nMarginToChip);
            matLeadTmp = matLead;
            matPadTmp = matPad;
        }else if (PR_DIRECTION::LEFT == enDir) {
            cv::transpose(matPad, matPadTmp);
            cv::flip(matPadTmp, matPadTmp, 0); //transpose+flip(0)=CCW
            cv::transpose(matLead, matLeadTmp);
            cv::flip(matLeadTmp, matLeadTmp, 0); //transpose+flip(0)=CCW
            rectSubRegion = cv::Rect(0, pstCmd->rectChipBody.y - pstCmd->rectSrchWindow.y - nEnlargeChipMargin, 
                pstCmd->rectChipBody.x - pstCmd->rectSrchWindow.x + nMarginToChip, pstCmd->rectChipBody.height + nMarginToChip * 2);
        }else if (PR_DIRECTION::DOWN == enDir) {
            cv::flip(matPad, matPadTmp, -1);    //flip(-1)=180
            cv::flip(matLead, matLeadTmp, -1);    //flip(-1)=180
            rectSubRegion = cv::Rect(pstCmd->rectChipBody.x - pstCmd->rectSrchWindow.x - nEnlargeChipMargin, pstCmd->rectChipBody.br().y - pstCmd->rectSrchWindow.y - nMarginToChip,
                pstCmd->rectChipBody.width + nEnlargeChipMargin * 2, pstCmd->rectSrchWindow.br().y - pstCmd->rectChipBody.br().y + nMarginToChip);
        }else if (PR_DIRECTION::RIGHT == enDir) {
            cv::transpose(matPad, matPadTmp);
            cv::flip(matPadTmp, matPadTmp, 1); //transpose+flip(1)=CW
            
            cv::transpose(matLead, matLeadTmp);
            cv::flip(matLeadTmp, matLeadTmp, 1); //transpose+flip(1)=CW
            rectSubRegion = cv::Rect(pstCmd->rectChipBody.br().x - pstCmd->rectSrchWindow.x - nMarginToChip, pstCmd->rectChipBody.y - pstCmd->rectSrchWindow.y - nMarginToChip,
                pstCmd->rectSrchWindow.br().x - pstCmd->rectChipBody.br().x + nMarginToChip, pstCmd->rectChipBody.height + nMarginToChip * 2);
        }

        CalcUtils::adjustRectROI(rectSubRegion, matGray);

        Int32 nPadRecordId = 0, nLeadRecordId = 0;
        auto ptrPadRecord = std::make_shared<TmplRecord>(PR_RECORD_TYPE::TEMPLATE);
        ptrPadRecord->setTmpl(matPadTmp);
        ptrPadRecord->setAlgorithm(PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF);
        RecordManagerInstance->add(ptrPadRecord, nPadRecordId);
        pstRpy->vecPadRecordId.push_back(nPadRecordId);

        auto ptrLeadRecord = std::make_shared<TmplRecord>(PR_RECORD_TYPE::TEMPLATE);
        ptrLeadRecord->setAlgorithm(PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF);
        ptrLeadRecord->setTmpl(matLeadTmp);
        RecordManagerInstance->add(ptrLeadRecord, nLeadRecordId);
        pstRpy->vecLeadRecordId.push_back(nLeadRecordId);

        _autoLocateLeadOneSide(matGray, rectSubRegion, enDir, matPadTmp, matLeadTmp, nPadRecordId, nLeadRecordId, pstCmd, pstRpy);
        if (VisionStatus::OK != pstRpy->enStatus)
            break;
    }

    // If some of the process fails, then free all the records.
    if (VisionStatus::OK != pstRpy->enStatus) {
        for (auto recordId : pstRpy->vecPadRecordId)
            RecordManagerInstance->free(recordId);
        pstRpy->vecPadRecordId.clear();

        for (auto recordId : pstRpy->vecLeadRecordId)
            RecordManagerInstance->free(recordId);
        pstRpy->vecLeadRecordId.clear();
    }

    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_autoLocateLeadOneSide(const cv::Mat                       &matGray,
                                                                const cv::Rect                      &rectSubRegion,
                                                                const PR_DIRECTION                   enDir,
                                                                const cv::Mat                       &matPad,
                                                                const cv::Mat                       &matLead,
                                                                Int32                                nPadRecordId,
                                                                Int32                                nLeadRecordId,
                                                                const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd,
                                                                PR_AUTO_LOCATE_LEAD_RPY       *const pstRpy)
{
    cv::Mat matSubRegion = cv::Mat(matGray, rectSubRegion);
    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Auto locate lead sub region", matSubRegion);

    pstRpy->enStatus = _fillHoleByMorph(
        matSubRegion,
        matSubRegion,
        PR_OBJECT_ATTRIBUTE::BRIGHT,
        cv::MorphShapes::MORPH_ELLIPSE,
        cv::Size(10, 10),
        2);
    if (VisionStatus::OK != pstRpy->enStatus) {
        WriteLog("Failed to fill hole in _autoLocateLeadOneSide");
        return pstRpy->enStatus;
    }

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("Fill hole", matSubRegion);
        cv::imwrite("Fill Hole result.png", matSubRegion);
    }

    cv::Mat matRow;
    if (PR_DIRECTION::UP == enDir || PR_DIRECTION::DOWN == enDir)
        cv::reduce(matSubRegion, matRow, 0, cv::ReduceTypes::REDUCE_AVG);
    else {
        cv::Mat matCol;
        cv::reduce(matSubRegion, matCol, 1, cv::ReduceTypes::REDUCE_AVG);
        cv::transpose(matCol, matRow);
    }

    auto vecThreshold = autoMultiLevelThreshold(matRow, cv::Mat(), 2);
    assert(vecThreshold.size() == 2);

    cv::Mat matBW = matRow > (vecThreshold[1]);
    cv::medianBlur(matBW, matBW, 9);
    matBW.convertTo(matBW, CV_32FC1);
    cv::Mat matBWDiff = CalcUtils::diff(matBW, 1, 2);
#ifdef _DEBUG
    auto vecRow = CalcUtils::matToVector<uchar>(matRow);
    auto vecBW = CalcUtils::matToVector<float>(matBW);
    auto vecBWDiff = CalcUtils::matToVector<float>(matBWDiff);
#endif
    std::vector<int> vecIndex1, vecIndex2;
    for (int i = 0; i < matBWDiff.cols; ++i) {
        if (matBWDiff.at<float>(i) > 0)
            vecIndex1.push_back(i);
        else if (matBWDiff.at<float>(i) < 0)
            vecIndex2.push_back(i);
    }

    if (vecIndex1.empty() || vecIndex2.empty()) {
        WriteLog("Failed split image to auto locate lead");
        pstRpy->enStatus = VisionStatus::AUTO_LOCATE_LEAD_FAIL;
        return pstRpy->enStatus;
    }

    auto n = vecIndex1.size();
    std::vector<int> vecSegPoint;
    vecSegPoint.push_back(0);
    for (size_t i = 0; i < n - 1; ++ i)
        vecSegPoint.push_back((vecIndex1[i + 1] + vecIndex2[i]) / 2);

    if (PR_DIRECTION::UP == static_cast<PR_DIRECTION>(enDir) || PR_DIRECTION::DOWN == static_cast<PR_DIRECTION>(enDir))
        vecSegPoint.push_back(matSubRegion.cols);
    else
        vecSegPoint.push_back(matSubRegion.rows);

    //Adjust segment position for the first and last lead.
    if (n > 2) {
        if (vecSegPoint[1] - vecSegPoint[0] > vecSegPoint[2] - vecSegPoint[1])
            vecSegPoint[0] = vecSegPoint[1] - (vecSegPoint[2] - vecSegPoint[1]);
        if (vecSegPoint[n] - vecSegPoint[n-1] > vecSegPoint[n-1] - vecSegPoint[n-2])
            vecSegPoint[n] = vecSegPoint[n-1] + vecSegPoint[n-1] - vecSegPoint[n-2];
    }

    VectorOfPoint2f vecLeadPositions, vecPadPositions;
    for (int i = 0; i < n; ++ i) {
        cv::Rect rectLeadSrchWindow;
        PR_AUTO_LOCATE_LEAD_RPY::PR_LEAD_INFO stLeadInfo;
        stLeadInfo.enDir = enDir;
        if (PR_DIRECTION::UP == enDir || PR_DIRECTION::DOWN == enDir)
            rectLeadSrchWindow = cv::Rect(vecSegPoint[i], 0, vecSegPoint[i + 1] - vecSegPoint[i], matSubRegion.rows);
        else
            rectLeadSrchWindow = cv::Rect(0, vecSegPoint[i], matSubRegion.cols, vecSegPoint[i + 1] - vecSegPoint[i]);

        if (PR_DIRECTION::UP == static_cast<PR_DIRECTION>(enDir) || PR_DIRECTION::DOWN == static_cast<PR_DIRECTION>(enDir))
            rectLeadSrchWindow = CalcUtils::resizeRect(rectLeadSrchWindow, cv::Size(rectLeadSrchWindow.width + 10, rectLeadSrchWindow.height));
        else
            rectLeadSrchWindow = CalcUtils::resizeRect(rectLeadSrchWindow, cv::Size(rectLeadSrchWindow.width, rectLeadSrchWindow.height + 10));
        CalcUtils::adjustRectROI(rectLeadSrchWindow, matSubRegion);
        cv::Mat matLeadSrchROI(matSubRegion, rectLeadSrchWindow);
        cv::Point2f ptPadPosition, ptLeadPosition;
        float fPadRotation, fLeadRotation, fPadScore, fLeadScore;
        if (! matPad.empty()) {
            if (matPad.cols >= matLeadSrchROI.cols || matPad.rows >= matLeadSrchROI.rows) {
                char chArrMsg[1000];
                _snprintf(chArrMsg, sizeof(chArrMsg), "The pad template size (%d, %d) is bigger than search area size (%d, %d).",
                    matPad.cols, matPad.rows, matLeadSrchROI.cols, matLeadSrchROI.rows);
                WriteLog(chArrMsg);
                pstRpy->enStatus = VisionStatus::AUTO_LOCATE_LEAD_FAIL;
                return pstRpy->enStatus;
            }

            MatchTmpl::matchTemplate(matLeadSrchROI, matPad, false, PR_OBJECT_MOTION::TRANSLATION, ptPadPosition, fPadRotation, fPadScore);
            if (fPadScore < 0.6f) {
                char msg[1000];
                _snprintf(msg, sizeof(msg), "Auto locate lead at direction %d failed to search pad at %d lead.", ToInt32(enDir), i);
                WriteLog(msg);
                pstRpy->enStatus = VisionStatus::AUTO_LOCATE_LEAD_FAIL;
                continue;
            }
            vecPadPositions.push_back(ptPadPosition);
            cv::Point ptInOriginImg(ToInt32(ptPadPosition.x + rectLeadSrchWindow.x + rectSubRegion.x + pstCmd->rectSrchWindow.x),
                                    ToInt32(ptPadPosition.y + rectLeadSrchWindow.y + rectSubRegion.y + pstCmd->rectSrchWindow.y));
            cv::circle(pstRpy->matResultImg, ptInOriginImg, 3, GREEN_SCALAR);
            stLeadInfo.rectPadWindow = cv::Rect(ptInOriginImg.x - matPad.cols / 2, ptInOriginImg.y - matPad.rows / 2, matPad.cols, matPad.rows);
            cv::rectangle(pstRpy->matResultImg, stLeadInfo.rectPadWindow, GREEN_SCALAR, 1);
            stLeadInfo.nPadRecordId = nPadRecordId;
        }

        if (matLead.cols >= matLeadSrchROI.cols || matLead.rows >= matLeadSrchROI.rows) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The lead template size (%d, %d) is bigger than search area size (%d, %d).",
                matLead.cols, matLead.rows, matLeadSrchROI.cols, matLeadSrchROI.rows);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::AUTO_LOCATE_LEAD_FAIL;
            return pstRpy->enStatus;
        }

        MatchTmpl::matchTemplate(matLeadSrchROI, matLead, false, PR_OBJECT_MOTION::TRANSLATION, ptLeadPosition, fLeadRotation, fLeadScore);
        if (fPadScore < 0.6f) {
            char msg[1000];
            _snprintf(msg, sizeof(msg), "Auto locate lead at direction %d failed to search lead at lead index %d.", ToInt32(enDir), i);
            WriteLog(msg);
            pstRpy->enStatus = VisionStatus::AUTO_LOCATE_LEAD_FAIL;
            continue;
        }
        vecLeadPositions.push_back(ptLeadPosition);
        cv::Point ptInOriginImg(ToInt32(ptLeadPosition.x + rectLeadSrchWindow.x + rectSubRegion.x + pstCmd->rectSrchWindow.x),
                                ToInt32(ptLeadPosition.y + rectLeadSrchWindow.y + rectSubRegion.y + pstCmd->rectSrchWindow.y));
        stLeadInfo.rectLeadWindow = cv::Rect(ptInOriginImg.x - matLead.cols / 2, ptInOriginImg.y - matLead.rows / 2, matLead.cols, matLead.rows);
        cv::circle(pstRpy->matResultImg, ptInOriginImg, 3, CYAN_SCALAR);
        cv::rectangle(pstRpy->matResultImg, stLeadInfo.rectLeadWindow, CYAN_SCALAR, 1);
        stLeadInfo.nLeadRecordId = nLeadRecordId;

        // Adjust the search window for lead according to the find out lead position.
        auto leadOffsetInSrchWindowX = ToInt32(ptLeadPosition.x - rectLeadSrchWindow.width  / 2.f);
        auto leadOffsetInSrchWindowY = ToInt32(ptLeadPosition.y - rectLeadSrchWindow.height / 2.f);
        cv::Point ptLeadSrchWindowCtr(rectLeadSrchWindow.x + rectLeadSrchWindow.width / 2, rectLeadSrchWindow.y + rectLeadSrchWindow.height / 2);
        if (PR_DIRECTION::UP == enDir || PR_DIRECTION::DOWN == enDir) {
            // Make lead search window use lead center as center
            ptLeadSrchWindowCtr.x += leadOffsetInSrchWindowX;
            rectLeadSrchWindow.width  +=  abs(leadOffsetInSrchWindowX) * 2; // Enlarge the window to make the segment line don't change.
        } else {
            // Make lead search window use lead center as center
            ptLeadSrchWindowCtr.y += leadOffsetInSrchWindowY;
            rectLeadSrchWindow.height +=  abs(leadOffsetInSrchWindowY) * 2; // Enlarge the window to make the segment line don't change.
        }

        rectLeadSrchWindow = cv::Rect(ptLeadSrchWindowCtr.x - rectLeadSrchWindow.width / 2, ptLeadSrchWindowCtr.y - rectLeadSrchWindow.height / 2,
                rectLeadSrchWindow.width, rectLeadSrchWindow.height);

        rectLeadSrchWindow.x += rectSubRegion.x + pstCmd->rectSrchWindow.x;
        rectLeadSrchWindow.y += rectSubRegion.y + pstCmd->rectSrchWindow.y;
        stLeadInfo.rectSrchWindow = rectLeadSrchWindow;
        cv::rectangle(pstRpy->matResultImg, stLeadInfo.rectSrchWindow, BLUE_SCALAR, 1);
        pstRpy->vecLeadInfo.push_back(stLeadInfo);
    }

    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_autoLocateLeadByContour(const cv::Mat &matGray, const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy) {
    cv::Mat matFilter;
    cv::GaussianBlur(matGray, matFilter, cv::Size(3, 3), 1, 1);

    cv::Mat matMask = cv::Mat::ones(matGray.size(), CV_8UC1);
    cv::Rect rectIcBodyInROI(pstCmd->rectChipBody.x - pstCmd->rectSrchWindow.x, pstCmd->rectChipBody.y - pstCmd->rectSrchWindow.y,
        pstCmd->rectChipBody.width, pstCmd->rectChipBody.height);
    cv::Mat matMaskROI(matMask, rectIcBodyInROI);
    matMaskROI.setTo(cv::Scalar(0));
    int nThreshold = _autoThreshold(matFilter, matMask);

    matFilter.setTo(cv::Scalar(0), cv::Scalar(1) - matMask); //Set the body of the IC to pure dark.
    cv::Mat matThreshold;
    cv::threshold(matFilter, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, cv::THRESH_BINARY);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("AutoLocateLead threshold image", matThreshold);

    PR_FILL_HOLE_CMD stFillHoleCmd;
    PR_FILL_HOLE_RPY stFillHoleRpy;
    stFillHoleCmd.matInputImg = matThreshold;
    stFillHoleCmd.enAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stFillHoleCmd.enMethod = PR_FILL_HOLE_METHOD::CONTOUR;
    stFillHoleCmd.rectROI = cv::Rect(0, 0, matThreshold.cols, matThreshold.rows);
    fillHole(&stFillHoleCmd, &stFillHoleRpy);
    if (VisionStatus::OK != stFillHoleRpy.enStatus) {
        pstRpy->enStatus = stFillHoleRpy.enStatus;
        return pstRpy->enStatus;
    }
    matThreshold = stFillHoleRpy.matResultImg;
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("autoLocateLead fillHole image", matThreshold);

    PR_REMOVE_CC_CMD stRemoveCcCmd;
    PR_REMOVE_CC_RPY stRemoveCcRpy;
    stRemoveCcCmd.matInputImg = matThreshold;
    stRemoveCcCmd.enCompareType = PR_COMPARE_TYPE::SMALLER;
    stRemoveCcCmd.nConnectivity = 4;
    stRemoveCcCmd.rectROI = cv::Rect(0, 0, matThreshold.cols, matThreshold.rows);
    stRemoveCcCmd.fAreaThreshold = 200;
    removeCC(&stRemoveCcCmd, &stRemoveCcRpy);
    if (VisionStatus::OK != stRemoveCcRpy.enStatus) {
        pstRpy->enStatus = stRemoveCcRpy.enStatus;
        return pstRpy->enStatus;
    }
    matThreshold = stRemoveCcRpy.matResultImg.clone();
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("autoLocateLead removeCC image", matThreshold);

    VectorOfVectorOfPoint contours, effectiveContours, vecDrawContours;
    cv::findContours(matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    if (contours.size() <= 0) {
        pstRpy->enStatus = VisionStatus::AUTO_LOCATE_LEAD_FAIL;
        return pstRpy->enStatus;
    }

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        cv::Mat matDisplay;
        cv::cvtColor(stRemoveCcRpy.matResultImg, matDisplay, CV_GRAY2BGR);
        for (int index = 0; index < ToInt32(contours.size()); ++ index)
            cv::drawContours(matDisplay, contours, index, GREEN_SCALAR);
        showImage("Find contour result", matDisplay);
    }

    double dTotalContourArea = 0.;
    double dTotalRectArea = 0.;
    for (const auto &contour : contours) {
        cv::Point2f ptCenter = CalcUtils::getContourCtr(contour);
        if (rectIcBodyInROI.contains(ptCenter))
            continue;

        auto area = cv::contourArea(contour);
        dTotalContourArea += area;
        cv::Rect rect = cv::boundingRect(contour);
        dTotalRectArea += rect.width * rect.height;
        effectiveContours.push_back(contour);
    }
    double dAverageContourArea = dTotalContourArea / effectiveContours.size();
    double dAverageRectArea = dTotalRectArea / effectiveContours.size();
    double dMinAcceptContourArea = 0.4 * dAverageContourArea;
    double dMaxAcceptContourArea = 1.5 * dAverageContourArea;
    double dMinAcceptRectArea = 0.4 * dAverageRectArea;
    double dMaxAcceptRectArea = 1.5 * dAverageRectArea;

    for (const auto &contour : effectiveContours) {
        cv::Rect rect = cv::boundingRect(contour);
        auto contourArea = cv::contourArea(contour);
        auto rectArea = rect.width * rect.height;
        if (dMinAcceptContourArea < contourArea && contourArea < dMaxAcceptContourArea &&
            dMinAcceptRectArea < rectArea && rectArea < dMaxAcceptRectArea) {
            if ((rect.br().x < rectIcBodyInROI.x) &&
                ((rect.y < rectIcBodyInROI.y) || (rect.br().y > rectIcBodyInROI.br().y))) {
                rect.x += pstCmd->rectSrchWindow.x;
                rect.y += pstCmd->rectSrchWindow.y;
                cv::rectangle(pstRpy->matResultImg, rect, RED_SCALAR, 2);
                continue;
            }

            if ((rect.x > rectIcBodyInROI.br().x) &&
                ((rect.y < rectIcBodyInROI.y) || (rect.br().y > rectIcBodyInROI.br().y))) {
                rect.x += pstCmd->rectSrchWindow.x;
                rect.y += pstCmd->rectSrchWindow.y;
                cv::rectangle(pstRpy->matResultImg, rect, RED_SCALAR, 2);
                continue;
            }

            if ((rect.br().y < rectIcBodyInROI.y) &&
                ((rect.x < rectIcBodyInROI.x) || (rect.br().x > rectIcBodyInROI.br().x))) {
                rect.x += pstCmd->rectSrchWindow.x;
                rect.y += pstCmd->rectSrchWindow.y;
                cv::rectangle(pstRpy->matResultImg, rect, RED_SCALAR, 2);
                continue;
            }

            if ((rect.y > rectIcBodyInROI.br().y) &&
                ((rect.x < rectIcBodyInROI.x) || (rect.br().x > rectIcBodyInROI.br().x))) {
                rect.x += pstCmd->rectSrchWindow.x;
                rect.y += pstCmd->rectSrchWindow.y;
                cv::rectangle(pstRpy->matResultImg, rect, RED_SCALAR, 2);
                continue;
            }

            rect.x += pstCmd->rectSrchWindow.x;
            rect.y += pstCmd->rectSrchWindow.y;
            pstRpy->vecLeadLocation.push_back(rect);
            cv::rectangle(pstRpy->matResultImg, rect, GREEN_SCALAR, 2);
        }
    }
    pstRpy->enStatus = VisionStatus::OK;
    if (pstRpy->vecLeadLocation.empty())
        pstRpy->enStatus = VisionStatus::AUTO_LOCATE_LEAD_FAIL;

    return pstRpy->enStatus;
}

/*static*/ void VisionAlgorithm::_addBridgeIfNotExist(VectorOfRect& vecBridgeWindow, const cv::Rect& rectBridge) {
    auto iterBridge = std::find(vecBridgeWindow.begin(), vecBridgeWindow.end(), rectBridge);
    if (iterBridge == vecBridgeWindow.end())
        vecBridgeWindow.push_back(rectBridge);
}

/*static*/ void VisionAlgorithm::_inspBridgeItem(const PR_INSP_BRIDGE_CMD * const pstCmd, PR_INSP_BRIDGE_RPY * const pstRpy) {
    cv::Mat matROI, matFilter, matMask, matThreshold;
    cv::Point ptRoiTL;
    cv::Rect rectOuterWindowNew(pstCmd->rectROI.x - pstCmd->rectOuterSrchWindow.x, pstCmd->rectROI.y - pstCmd->rectOuterSrchWindow.y, pstCmd->rectROI.width, pstCmd->rectROI.height);
    if (PR_INSP_BRIDGE_MODE::INNER == pstCmd->enInspMode) {
        matROI = cv::Mat(pstCmd->matInputImg, pstCmd->rectROI).clone();
        ptRoiTL = pstCmd->rectROI.tl();
    }
    else {
        matROI = cv::Mat(pstCmd->matInputImg, pstCmd->rectOuterSrchWindow).clone();
        matMask = cv::Mat::ones(matROI.size(), CV_8UC1);
        cv::Mat matMaskOuterWindow(matMask, rectOuterWindowNew);
        matMaskOuterWindow.setTo(cv::Scalar(0));
        matROI.setTo(cv::Scalar(0), cv::Scalar(1) - matMask); //Set the body of the IC to pure dark.
        ptRoiTL = pstCmd->rectOuterSrchWindow.tl();
    }

    cv::Mat matGray;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI.clone();

    cv::GaussianBlur(matGray, matFilter, cv::Size(3, 3), 1, 1);
    int nThreshold = _autoThreshold(matFilter, matMask);
    if (nThreshold < Config::GetInstance()->getInspBridgeMinThreshold())
        nThreshold = Config::GetInstance()->getInspBridgeMinThreshold();

    cv::threshold(matFilter, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, cv::THRESH_BINARY);

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("_inspBridgeItem threshold", matThreshold);

    VectorOfVectorOfPoint contours;
    cv::findContours(matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    if (contours.size() <= 0)
        return;

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        cv::Mat matDraw = matROI.clone();
        for (int index = 0; index < contours.size(); ++ index) {
            cv::RNG rng(12345);
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            cv::drawContours(matDraw, contours, index, color, 2);
        }
        showImage("Contours", matDraw);
    }

    const int MARGIN = 3;
    for (const auto &contour : contours) {
        auto area = cv::contourArea(contour);
        if (area < 50.f)
            continue;

        cv::Rect rect = cv::boundingRect(contour);
        cv::Point ptCenter = CalcUtils::getContourCtr(contour);

        if (PR_INSP_BRIDGE_MODE::INNER == pstCmd->enInspMode) {
            if (rect.width > pstCmd->stInnerInspCriteria.fMaxLengthX || rect.height > pstCmd->stInnerInspCriteria.fMaxLengthY) {
                cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                pstRpy->vecBridgeWindow.push_back(rectResult);
            }
        }
        else {
            cv::Point ptWindowCtr(rectOuterWindowNew.x + rectOuterWindowNew.width / 2, rectOuterWindowNew.y + rectOuterWindowNew.height / 2);
            for (const auto enDirection : pstCmd->vecOuterInspDirection)
            if (PR_DIRECTION::UP == enDirection) {
                int nTolerance = pstCmd->rectROI.y - pstCmd->rectOuterSrchWindow.y - MARGIN;
                if (ptCenter.y <= (ptWindowCtr.y - rectOuterWindowNew.height / 4) && rect.height >= nTolerance) {
                    cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                    _addBridgeIfNotExist(pstRpy->vecBridgeWindow, rectResult);
                }
            }
            else if (PR_DIRECTION::DOWN == enDirection) {
                int nTolerance = (pstCmd->rectOuterSrchWindow.y + pstCmd->rectOuterSrchWindow.height) - (pstCmd->rectROI.y + pstCmd->rectROI.height) - MARGIN;
                if (ptCenter.y >= (ptWindowCtr.y + rectOuterWindowNew.height / 4) && rect.height >= nTolerance) {
                    cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                    _addBridgeIfNotExist(pstRpy->vecBridgeWindow, rectResult);
                }
            }
            else
            if (PR_DIRECTION::LEFT == enDirection) {
                int nTolerance = pstCmd->rectROI.x - pstCmd->rectOuterSrchWindow.x - MARGIN;
                if (ptCenter.x <= (ptWindowCtr.x - rectOuterWindowNew.width / 4) && rect.width >= nTolerance) {
                    cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                    _addBridgeIfNotExist(pstRpy->vecBridgeWindow, rectResult);
                }
            }
            else
            if (PR_DIRECTION::RIGHT == enDirection) {
                int nTolerance = (pstCmd->rectOuterSrchWindow.x + pstCmd->rectOuterSrchWindow.width) - (pstCmd->rectROI.x + pstCmd->rectROI.width) - MARGIN;
                if (ptCenter.x >= (ptWindowCtr.x + rectOuterWindowNew.width / 4) && rect.width >= nTolerance) {
                    cv::Rect rectResult(rect.x + ptRoiTL.x, rect.y + ptRoiTL.y, rect.width, rect.height);
                    _addBridgeIfNotExist(pstRpy->vecBridgeWindow, rectResult);
                }
            }
        }
    }

    if (!pstRpy->vecBridgeWindow.empty())
        pstRpy->enStatus = VisionStatus::BRIDGE_DEFECT;

    //Draw the innter window and outer window.
    if (!isAutoMode()) {
        if (pstCmd->matInputImg.channels() > 1)
            pstRpy->matResultImg = pstCmd->matInputImg.clone();
        else
            cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);

        cv::rectangle(pstRpy->matResultImg, pstCmd->rectROI, BLUE_SCALAR);
        if (PR_INSP_BRIDGE_MODE::OUTER == pstCmd->enInspMode)
            cv::rectangle(pstRpy->matResultImg, pstCmd->rectOuterSrchWindow, BLUE_SCALAR);
        for (const auto &rectBridge : pstRpy->vecBridgeWindow)
            cv::rectangle(pstRpy->matResultImg, rectBridge, RED_SCALAR, 2);
    }
}

/*static*/ VisionStatus VisionAlgorithm::inspBridge(const PR_INSP_BRIDGE_CMD *const pstCmd, PR_INSP_BRIDGE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    //Initialize the result.
    pstRpy->enStatus = VisionStatus::OK;
    pstRpy->vecBridgeWindow.clear();

    char chArrMsg[1000];
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspBridge);

    if (PR_INSP_BRIDGE_MODE::OUTER == pstCmd->enInspMode) {
        pstRpy->enStatus = _checkInputROI(pstCmd->rectOuterSrchWindow, pstCmd->matInputImg, AT);
        if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

        if (! CalcUtils::isRectInRect(pstCmd->rectROI, pstCmd->rectOuterSrchWindow)) {
             _snprintf(chArrMsg, sizeof(chArrMsg), "The inspect bridge window of outter mode should inside the search window.");
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->vecOuterInspDirection.empty()) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "The outer inspection direction is empty.");
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }
    else if (PR_INSP_BRIDGE_MODE::INNER == pstCmd->enInspMode) {
        if (pstCmd->stInnerInspCriteria.fMaxLengthX <= 0 || pstCmd->stInnerInspCriteria.fMaxLengthY <= 0) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "The inner inspection criteria (%.2f, %.2f) is empty.",
                pstCmd->stInnerInspCriteria.fMaxLengthX, pstCmd->stInnerInspCriteria.fMaxLengthY);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    _inspBridgeItem(pstCmd, pstRpy);

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::lrnChip(const PR_LRN_CHIP_CMD *const pstCmd, PR_LRN_CHIP_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectChip.x < 0 || pstCmd->rectChip.y < 0 ||
        pstCmd->rectChip.width <= 0 || pstCmd->rectChip.height <= 0 ||
        (pstCmd->rectChip.x + pstCmd->rectChip.width) > pstCmd->matInputImg.cols ||
        (pstCmd->rectChip.y + pstCmd->rectChip.height) > pstCmd->matInputImg.rows) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The input chip window (%d, %d, %d, %d) is invalid.",
            pstCmd->rectChip.x, pstCmd->rectChip.y, pstCmd->rectChip.width, pstCmd->rectChip.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    cv::Rect rectChipROI = CalcUtils::resizeRect<float>(pstCmd->rectChip, pstCmd->rectChip.size() + cv::Size2f(20, 20));
    if (rectChipROI.x < 0) rectChipROI.x = 0;
    if (rectChipROI.y < 0) rectChipROI.y = 0;
    if ((rectChipROI.x + rectChipROI.width)  > pstCmd->matInputImg.cols) rectChipROI.width  = pstCmd->matInputImg.cols - rectChipROI.x;
    if ((rectChipROI.y + rectChipROI.height) > pstCmd->matInputImg.rows) rectChipROI.height = pstCmd->matInputImg.rows - rectChipROI.y;
    cv::Mat matROI(pstCmd->matInputImg, rectChipROI);
    cv::Mat matGray, matBlur, matThreshold;
    if (matROI.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI.clone();
    cv::GaussianBlur(matGray, matBlur, cv::Size(5, 5), 2, 2);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Blur image", matBlur);

    pstRpy->nThreshold = pstCmd->nThreshold;
    if (pstCmd->bAutoThreshold)
        pstRpy->nThreshold = _autoThreshold(matBlur);

    cv::ThresholdTypes enThresholdType = cv::THRESH_BINARY;
    if (PR_INSP_CHIP_MODE::SQUARE == pstCmd->enInspMode)
        enThresholdType = cv::THRESH_BINARY_INV;
    cv::threshold(matBlur, matThreshold, pstRpy->nThreshold, 255, enThresholdType);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Threshold image", matThreshold);

    if (pstCmd->matInputImg.channels() > 1)
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);

    if (PR_INSP_CHIP_MODE::HEAD == pstCmd->enInspMode) {
        _lrnChipHeadMode(matThreshold, rectChipROI, pstRpy);
    }
    else if (PR_INSP_CHIP_MODE::SQUARE == pstCmd->enInspMode) {
        _lrnChipSquareMode(matThreshold, pstCmd->rectChip, rectChipROI, pstRpy);
    }
    else if (PR_INSP_CHIP_MODE::CAE == pstCmd->enInspMode) {
        _lrnChipCAEMode(matThreshold, rectChipROI, pstRpy);
    }

    if (VisionStatus::OK != pstRpy->enStatus) {
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_END_TIME;

    pstRpy->enStatus = VisionStatus::OK;
    pstRpy->sizeDevice = pstCmd->rectChip.size();
    _writeChipRecord(pstCmd, pstRpy, matGray);
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_lrnChipHeadMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_LRN_CHIP_RPY *const pstRpy) {
    VectorOfPoint vecPtElectrode;
    VectorOfSize2f vecElectrodeSize;
    VectorOfVectorOfPoint vecContours;
    pstRpy->enStatus = _findDeviceElectrode(matThreshold, vecPtElectrode, vecElectrodeSize, vecContours);
    if (pstRpy->enStatus != VisionStatus::OK) {
        WriteLog("Failed to find device electrode.");
        return pstRpy->enStatus;
    }
    //for ( size_t index = 0; index < PR_ELECTRODE_COUNT && index < vecElectrodeSize.size(); ++ index ) {
    //    pstRpy->arrSizeElectrode[index] = vecElectrodeSize[index];
    //    cv::Rect2f rectElectrode ( vecPtElectrode[index].x - vecElectrodeSize[index].width / 2 + rectROI.x, vecPtElectrode[index].y - vecElectrodeSize[index].height / 2 + rectROI.y,
    //        vecElectrodeSize[index].width, vecElectrodeSize[index].height );
    //    cv::rectangle ( pstRpy->matResultImg, rectElectrode, BLUE_SCALAR );
    //}
    for (auto &contour : vecContours) {
        for (auto &point : contour) {
            point.x += rectROI.x;
            point.y += rectROI.y;
        }
    }
    cv::polylines(pstRpy->matResultImg, vecContours, true, BLUE_SCALAR, 2);

    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_lrnChipCAEMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_LRN_CHIP_RPY *const pstRpy) {
    cv::Mat matFillHole;
    _fillHoleByContour(matThreshold, matFillHole, PR_OBJECT_ATTRIBUTE::BRIGHT);

    VectorOfVectorOfPoint contours;
    cv::findContours(matFillHole, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    VectorOfPoint maxContour;
    float fMaxArea = std::numeric_limits<float>::min();
    for (const auto &contour : contours) {
        auto rectRotated = cv::minAreaRect(contour);
        if (rectRotated.size.area() > fMaxArea) {
            maxContour = contour;
            fMaxArea = rectRotated.size.area();
        }
    }

    int nMaxRansacTime = 20;
    float fFitTolerance = 4;
    auto fitCircleResult = Fitting::fitCircleRansac(maxContour, fFitTolerance, nMaxRansacTime, maxContour.size() / 2);
    auto vecLeftOverPoints = _findPointsOverCircleTol(maxContour, fitCircleResult, fFitTolerance);
    float fSlope = 0.f, fIntercept = 0.f;
    PR_Line2f stLine;
    _fitLineRansac(vecLeftOverPoints, fFitTolerance, nMaxRansacTime, vecLeftOverPoints.size() / 2, false, fSlope, fIntercept, stLine);
    float fLineLength = CalcUtils::distanceOf2Point(stLine.pt1, stLine.pt2);

    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode()) {
        cv::Mat matDisplay;
        cv::cvtColor(matThreshold, matDisplay, CV_GRAY2BGR);
        cv::circle(matDisplay, fitCircleResult.center, ToInt32(fitCircleResult.size.width / 2 + 0.5), BLUE_SCALAR, 2);
        cv::line(matDisplay, stLine.pt1, stLine.pt2, GREEN_SCALAR, 2);
        showImage("lrnChipCAEMode", matDisplay);
    }

    if (fitCircleResult.size.width < rectROI.width / 2) {
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        return pstRpy->enStatus;
    }

    if (fLineLength < fitCircleResult.size.width / 2 || fLineLength > fitCircleResult.size.width) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CAE_LINE;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = VisionStatus::OK;
    for (auto &point : maxContour) {
        point.x += rectROI.x;
        point.y += rectROI.y;
    }
    VectorOfVectorOfPoint vevVecPoint;
    vevVecPoint.push_back(maxContour);
    cv::polylines(pstRpy->matResultImg, vevVecPoint, true, BLUE_SCALAR, 2);
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_lrnChipSquareMode(const cv::Mat &matThreshold, const cv::Rect &rectChip, const cv::Rect &rectROI, PR_LRN_CHIP_RPY *const pstRpy) {
    cv::Mat matOneRow, matOneCol;
    cv::reduce(matThreshold, matOneRow, 0, cv::ReduceTypes::REDUCE_SUM, CV_32SC1);
    matOneRow /= 255;

    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode()) {
        int nHeight = matThreshold.rows;
        cv::Mat matDisplay = cv::Mat::ones(matThreshold.size(), CV_8UC3) * 255;
        matDisplay.setTo(cv::Scalar::all(255));
        for (int i = 1; i < matOneRow.cols; ++ i) {
            cv::Point pt1(i - 1, nHeight - matOneRow.at<int>(0, i - 1));
            cv::Point pt2(i, nHeight - matOneRow.at<int>(0, i));
            cv::line(matDisplay, pt1, pt2, BLUE_SCALAR, 2);
        }
        showImage("Project Curve Row", matDisplay);
    }
    cv::reduce(matThreshold, matOneCol, 1, cv::ReduceTypes::REDUCE_SUM, CV_32SC1);
    matOneCol /= 255;
    cv::transpose(matOneCol, matOneCol);
    int nLeftX = 0, nRightX = 0, nTopY = 0, nBottomY = 0;
    bool bSuccess = true;
    bSuccess = (bSuccess && _findDataUpEdge(matOneRow, 0, matOneRow.cols / 2, nLeftX));
    bSuccess = (bSuccess && _findDataDownEdge(matOneRow, matOneRow.cols / 2, matOneRow.cols, nRightX));

    bSuccess = (bSuccess && _findDataUpEdge(matOneCol, 0, matOneCol.cols / 2, nTopY));
    bSuccess = (bSuccess && _findDataDownEdge(matOneCol, matOneCol.cols / 2, matOneCol.cols, nBottomY));
    cv::rectangle(pstRpy->matResultImg, cv::Point(rectROI.x + nLeftX, rectROI.y + nTopY),
        cv::Point(rectROI.x + nRightX, rectROI.y + nBottomY), GREEN_SCALAR, 2);

    if (! bSuccess) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_SQUARE_EDGE;
        return pstRpy->enStatus;
    }

    const float INPUT_ERROR_TOL = 0.1f; //10% input error tolerance
    if (abs(nRightX - nLeftX - rectChip.width) / ToFloat(rectChip.width) > INPUT_ERROR_TOL ||
        abs(nBottomY - nTopY - rectChip.height) / ToFloat(rectChip.height) > INPUT_ERROR_TOL) {
        WriteLog("The found chip edges are not match with input chip window");
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_SQUARE_EDGE;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_writeChipRecord(const PR_LRN_CHIP_CMD *const pstCmd, PR_LRN_CHIP_RPY *const pstRpy, const cv::Mat &matTmpl) {
    ChipRecordPtr ptrRecord = std::make_shared<ChipRecord>(PR_RECORD_TYPE::CHIP);
    ptrRecord->setThreshold(pstRpy->nThreshold);
    ptrRecord->setSize(pstRpy->sizeDevice);
    ptrRecord->setInspMode(pstCmd->enInspMode);
    ptrRecord->setTmpl(matTmpl);
    RecordManager::getInstance()->add(ptrRecord, pstRpy->nRecordId);
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::inspChip(const PR_INSP_CHIP_CMD *const pstCmd, PR_INSP_CHIP_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectSrchWindow, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    ChipRecordPtr ptrRecord = nullptr;
    if (pstCmd->nRecordId > 0) {
        ptrRecord = std::static_pointer_cast<ChipRecord> (RecordManager::getInstance()->get(pstCmd->nRecordId));
        if (nullptr == ptrRecord) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof (chArrMsg), "Failed to get record ID %d in system.", pstCmd->nRecordId);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (ptrRecord->getInspMode() != pstCmd->enInspMode) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof (chArrMsg), "The record insp mode %d if record %d is not match with input insp mode %d",
                ToInt32(ptrRecord->getInspMode()), pstCmd->nRecordId, ToInt32(pstCmd->enInspMode));
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspChip);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectSrchWindow);
    cv::Mat matGray, matBlur, matThreshold;
    if (matROI.channels() > 1) {
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    }
    else {
        matGray = matROI.clone();
        cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);
    }

    cv::GaussianBlur(matGray, matBlur, cv::Size(5, 5), 2, 2);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Blur image", matBlur);

    int nThreshold = 100;
    if (ptrRecord != nullptr)
        nThreshold = ptrRecord->getThreshold();
    else
        nThreshold = _autoThreshold(matBlur);

    cv::ThresholdTypes enThresholdType = cv::THRESH_BINARY;
    if (PR_INSP_CHIP_MODE::SQUARE == pstCmd->enInspMode)
        enThresholdType = cv::THRESH_BINARY_INV;
    cv::threshold(matBlur, matThreshold, nThreshold, 255, enThresholdType);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Threshold image", matThreshold);

    if (PR_INSP_CHIP_MODE::HEAD == pstCmd->enInspMode)
        _inspChipHeadMode(matThreshold, pstCmd->rectSrchWindow, pstRpy);
    else if (PR_INSP_CHIP_MODE::BODY == pstCmd->enInspMode)
        _inspChipBodyMode(matThreshold, pstCmd->rectSrchWindow, pstRpy);
    else if (PR_INSP_CHIP_MODE::SQUARE == pstCmd->enInspMode)
        _inspChipSquareMode(matThreshold, pstCmd->rectSrchWindow, pstRpy);
    else if (PR_INSP_CHIP_MODE::CAE == pstCmd->enInspMode)
        _inspChipCAEMode(matThreshold, pstCmd->rectSrchWindow, pstRpy);
    else if (PR_INSP_CHIP_MODE::CIRCULAR == pstCmd->enInspMode)
        _inspChipCircularMode(matThreshold, pstCmd->rectSrchWindow, pstRpy);

    pstRpy->rotatedRectResult.angle = fabs(- pstRpy->rotatedRectResult.angle);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipHeadMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy) {
    VectorOfPoint vecPtElectrode;
    VectorOfSize2f vecElectrodeSize;
    VectorOfVectorOfPoint vecContours;
    pstRpy->enStatus = _findDeviceElectrode(matThreshold, vecPtElectrode, vecElectrodeSize, vecContours);
    if (pstRpy->enStatus != VisionStatus::OK) {
        WriteLog("Failed to find device electrode.");
        return pstRpy->enStatus;
    }

    //Draw the electrodes.
    for (auto &contour : vecContours) {
        for (auto &point : contour) {
            point.x += rectROI.x;
            point.y += rectROI.y;
        }
    }
    cv::polylines(pstRpy->matResultImg, vecContours, true, BLUE_SCALAR, 2);

    VectorOfPoint vecTwoContourTogether = vecContours[0];
    vecTwoContourTogether.insert(vecTwoContourTogether.end(), vecContours[1].begin(), vecContours[1].end());
    pstRpy->rotatedRectResult = cv::minAreaRect(vecTwoContourTogether);
    VectorOfVectorOfPoint vevVecPoints;
    vevVecPoints.push_back(CalcUtils::getCornerOfRotatedRect(pstRpy->rotatedRectResult));
    cv::polylines(pstRpy->matResultImg, vevVecPoints, true, GREEN_SCALAR, 2);
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipBodyMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy) {
    cv::Mat matLabels;
    cv::connectedComponents(matThreshold, matLabels, 4, CV_32S);

    double minValue = 0., maxValue = 0.;
    cv::minMaxIdx(matLabels, &minValue, &maxValue);
    float fMaxArea = std::numeric_limits<float>::min();
    cv::Mat matMaxComponent;
    //0 represents the background label, so need to ignore.
    for (int nLabel = 1; nLabel <= maxValue; ++ nLabel) {
        cv::Mat matCC = CalcUtils::genMaskByValue<Int32>(matLabels, nLabel);
        float fArea = ToFloat(cv::countNonZero(matCC));
        if (fArea > fMaxArea) {
            fMaxArea = fArea;
            matMaxComponent = matCC;
        }
    }

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Max Component", matMaxComponent);

    VectorOfVectorOfPoint contours;
    cv::findContours(matMaxComponent, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    if (contours.size() <= 0) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CIRCULAR_CHIP;
        return pstRpy->enStatus;
    }

    VectorOfPoint maxContour;
    fMaxArea = std::numeric_limits<float>::min();
    for (const auto &contour : contours) {
        auto rectRotated = cv::minAreaRect(contour);
        if (rectRotated.size.area() > fMaxArea) {
            maxContour = contour;
            fMaxArea = rectRotated.size.area();
        }
    }

    auto rectBody = cv::minAreaRect(maxContour);

    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode()) {
        cv::Mat matDisplay;
        cv::cvtColor(matThreshold, matDisplay, CV_GRAY2BGR);
        VectorOfVectorOfPoint vevVecPoints;
        vevVecPoints.push_back(CalcUtils::getCornerOfRotatedRect(rectBody));
        cv::polylines(matDisplay, vevVecPoints, true, GREEN_SCALAR, 2);
        showImage("inspChipCircularMode", matDisplay);
    }

    float fTolerance = 4.f;
    if (rectBody.size.width <= 2 * fTolerance || rectBody.size.height <= 2 * fTolerance) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CHIP_BODY;
        return pstRpy->enStatus;
    }

    float fMaxDim = std::max(rectBody.size.width, rectBody.size.height);
    float fMinDim = std::min(rectBody.size.width, rectBody.size.height);
    if (fMaxDim / fMinDim > 2.5) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CHIP_BODY;
        return pstRpy->enStatus;
    }

    cv::Mat matVerify = cv::Mat::zeros(rectROI.size(), CV_8UC1);
    auto rectEnlarge = rectBody;
    rectEnlarge.size.width += 2 * fTolerance;
    rectEnlarge.size.height += 2 * fTolerance;
    VectorOfVectorOfPoint vevVecPoint;
    vevVecPoint.push_back(CalcUtils::getCornerOfRotatedRect(rectEnlarge));
    cv::fillPoly(matVerify, vevVecPoint, cv::Scalar::all(PR_MAX_GRAY_LEVEL));

    auto rectDeflate = rectBody;
    rectDeflate.size.width -= 2 * fTolerance;
    rectDeflate.size.height -= 2 * fTolerance;
    vevVecPoint.clear();
    vevVecPoint.push_back(CalcUtils::getCornerOfRotatedRect(rectDeflate));
    cv::fillPoly(matVerify, vevVecPoint, cv::Scalar::all(0));
    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode())
        showImage("Verify Mat", matVerify);

    int nInTolPointCount = 0;
    for (const auto &point : maxContour) {
        if (matVerify.at<uchar>(point) > 0)
            ++ nInTolPointCount;
    }

    if (nInTolPointCount < ToInt32(maxContour.size()) / 2) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CHIP_BODY;
        return pstRpy->enStatus;
    }

    rectBody.center.x += rectROI.x;
    rectBody.center.y += rectROI.y;
    pstRpy->rotatedRectResult = rectBody;

    VectorOfVectorOfPoint vevVecPoints;
    vevVecPoints.push_back(CalcUtils::getCornerOfRotatedRect(pstRpy->rotatedRectResult));
    cv::polylines(pstRpy->matResultImg, vevVecPoints, true, GREEN_SCALAR, 2);
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipSquareMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy) {
    cv::Mat matOneRow, matOneCol;
    cv::reduce(matThreshold, matOneRow, 0, cv::ReduceTypes::REDUCE_SUM, CV_32SC1);
    matOneRow /= 255;

    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode()) {
        int nHeight = matThreshold.rows;
        cv::Mat matDisplay = cv::Mat::ones(matThreshold.size(), CV_8UC3) * 255;
        matDisplay.setTo(cv::Scalar::all(255));
        for (int i = 1; i < matOneRow.cols; ++ i) {
            cv::Point pt1(i - 1, nHeight - matOneRow.at<int>(0, i - 1));
            cv::Point pt2(i, nHeight - matOneRow.at<int>(0, i));
            cv::line(matDisplay, pt1, pt2, BLUE_SCALAR, 2);
        }
        showImage("Project Curve Row", matDisplay);
    }
    cv::reduce(matThreshold, matOneCol, 1, cv::ReduceTypes::REDUCE_SUM, CV_32SC1);
    matOneCol /= 255;
    cv::transpose(matOneCol, matOneCol);

    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode()) {
        int nHeight = matThreshold.cols;
        cv::Mat matDisplay = cv::Mat::ones(cv::Size(matThreshold.rows, matThreshold.cols), CV_8UC3) * 255;
        matDisplay.setTo(cv::Scalar::all(255));
        for (int i = 1; i < matOneCol.cols; ++ i) {
            cv::Point pt1(i - 1, nHeight - matOneCol.at<int>(0, i - 1));
            cv::Point pt2(i, nHeight - matOneCol.at<int>(0, i));
            cv::line(matDisplay, pt1, pt2, BLUE_SCALAR, 2);
        }
        showImage("Project Curve Col", matDisplay);
    }

    int nLeftX = 0, nRightX = 0, nTopY = 0, nBottomY = 0;
    bool bSuccess = true;
    bSuccess = (bSuccess && _findDataUpEdge  (matOneRow, 0,                  matOneRow.cols / 2, nLeftX));
    bSuccess = (bSuccess && _findDataDownEdge(matOneRow, matOneRow.cols / 2, matOneRow.cols,     nRightX));

    bSuccess = (bSuccess && _findDataUpEdge  (matOneCol, 0,                  matOneCol.cols / 2, nTopY));
    bSuccess = (bSuccess && _findDataDownEdge(matOneCol, matOneCol.cols / 2, matOneCol.cols,     nBottomY));
    cv::rectangle(pstRpy->matResultImg, cv::Point(rectROI.x + nLeftX, rectROI.y + nTopY),
        cv::Point(rectROI.x + nRightX, rectROI.y + nBottomY), GREEN_SCALAR, 2);

    if (! bSuccess) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_SQUARE_EDGE;
        return pstRpy->enStatus;
    }

    pstRpy->rotatedRectResult.angle = 0;
    pstRpy->rotatedRectResult.center.x = (nLeftX + nRightX) / 2.f + rectROI.x;
    pstRpy->rotatedRectResult.center.y = (nTopY + nBottomY) / 2.f + rectROI.y;
    pstRpy->rotatedRectResult.size.width = ToFloat(nRightX - nLeftX + 1);
    pstRpy->rotatedRectResult.size.height = ToFloat(nBottomY - nTopY + 1);
    VectorOfVectorOfPoint vevVecPoints;
    vevVecPoints.push_back(CalcUtils::getCornerOfRotatedRect(pstRpy->rotatedRectResult));
    cv::polylines(pstRpy->matResultImg, vevVecPoints, true, GREEN_SCALAR, 2);
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

//Serch from up to down.
/*static*/ bool VisionAlgorithm::_findDataUpEdge(const cv::Mat &matRow, int nStart, int nEnd, int &nEdgePos) {
    const int EDGE_SIZE = 30;
    int nEdgeDiffThreshold = 50;
    nEdgePos = -1;
    for (int index = nEnd - EDGE_SIZE - 1; index > nStart; -- index) {
        int nDiff = matRow.at<int>(0, index + EDGE_SIZE) - matRow.at<int>(0, index);
        if (nDiff > nEdgeDiffThreshold) {
            nEdgePos = index + EDGE_SIZE / 2;
            nEdgeDiffThreshold = nDiff;
        }
    }
    if (nEdgePos < 0)
        return false;
    return true;
}

/*static*/ bool VisionAlgorithm::_findDataDownEdge(const cv::Mat &matRow, int nStart, int nEnd, int &nEdgePos) {
    const int EDGE_SIZE = 30;
    int nEdgeDiffThreshold = 50;
    nEdgePos = -1;
    for (int index = nStart; index <= nEnd - EDGE_SIZE - 1; ++ index) {
        int nDiff = matRow.at<int>(0, index) - matRow.at<int>(0, index + EDGE_SIZE);
        if (nDiff > nEdgeDiffThreshold) {
            nEdgePos = index + EDGE_SIZE / 2;
            nEdgeDiffThreshold = nDiff;
        }
    }
    if (nEdgePos < 0)
        return false;
    return true;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipCAEMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy) {
    cv::Mat matFillHole;
    _fillHoleByContour(matThreshold, matFillHole, PR_OBJECT_ATTRIBUTE::BRIGHT);

    VectorOfVectorOfPoint contours;
    cv::findContours(matFillHole, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    VectorOfPoint maxContour;
    float fMaxArea = std::numeric_limits<float>::min();
    for (const auto &contour : contours) {
        auto rectRotated = cv::minAreaRect(contour);
        if (rectRotated.size.area() > fMaxArea) {
            maxContour = contour;
            fMaxArea = rectRotated.size.area();
        }
    }

    int nMaxRansacTime = 20;
    float fFitTolerance = 4;
    auto fitCircleResult = Fitting::fitCircleRansac(maxContour, fFitTolerance, nMaxRansacTime, maxContour.size() / 2);
    auto vecLeftOverPoints = _findPointsOverCircleTol(maxContour, fitCircleResult, fFitTolerance);
    float fSlope = 0.f, fIntercept = 0.f;
    PR_Line2f stLine;
    _fitLineRansac(vecLeftOverPoints, fFitTolerance, nMaxRansacTime, vecLeftOverPoints.size() / 2, false, fSlope, fIntercept, stLine);
    float fLineLength = CalcUtils::distanceOf2Point(stLine.pt1, stLine.pt2);

    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode()) {
        cv::Mat matDisplay;
        cv::cvtColor(matThreshold, matDisplay, CV_GRAY2BGR);
        cv::circle(matDisplay, fitCircleResult.center, ToInt32(fitCircleResult.size.width / 2 + 0.5), BLUE_SCALAR, 2);
        cv::line(matDisplay, stLine.pt1, stLine.pt2, GREEN_SCALAR, 2);
        showImage("lrnChipCAEMode", matDisplay);
    }

    if (fitCircleResult.size.width < rectROI.width / 2) {
        pstRpy->enStatus = VisionStatus::FAIL_TO_FIT_CIRCLE;
        return pstRpy->enStatus;
    }

    if (fLineLength < fitCircleResult.size.width / 2 || fLineLength > fitCircleResult.size.width) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CAE_LINE;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = VisionStatus::OK;
    for (auto &point : maxContour) {
        point.x += rectROI.x;
        point.y += rectROI.y;
    }
    VectorOfVectorOfPoint vevVecPoint;
    vevVecPoint.push_back(maxContour);
    cv::polylines(pstRpy->matResultImg, vevVecPoint, true, BLUE_SCALAR, 2);

    fitCircleResult.center.x += rectROI.x;
    fitCircleResult.center.y += rectROI.y;
    pstRpy->rotatedRectResult = fitCircleResult;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspChipCircularMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy) {
    cv::Mat matLabels;
    cv::connectedComponents(matThreshold, matLabels, 4, CV_32S);

    double minValue = 0., maxValue = 0.;
    cv::minMaxIdx(matLabels, &minValue, &maxValue);
    float fMaxArea = std::numeric_limits<float>::min();
    cv::Mat matMaxComponent;
    //0 represents the background label, so need to ignore.
    for (int nLabel = 1; nLabel <= maxValue; ++ nLabel) {
        cv::Mat matCC = CalcUtils::genMaskByValue<Int32>(matLabels, nLabel);
        float fArea = ToFloat(cv::countNonZero(matCC));
        if (fArea > fMaxArea) {
            fMaxArea = fArea;
            matMaxComponent = matCC;
        }
    }

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Max Component", matMaxComponent);

    VectorOfVectorOfPoint contours;
    cv::findContours(matMaxComponent, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    if (contours.size() <= 0) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CIRCULAR_CHIP;
        return pstRpy->enStatus;
    }

    VectorOfPoint maxContour;
    fMaxArea = std::numeric_limits<float>::min();
    for (const auto &contour : contours) {
        auto rectRotated = cv::minAreaRect(contour);
        if (rectRotated.size.area() > fMaxArea) {
            maxContour = contour;
            fMaxArea = rectRotated.size.area();
        }
    }

    int nMaxRansacTime = 20;
    float fFitTolerance = 4;
    auto fitCircleResult = Fitting::fitCircleRansac(maxContour, fFitTolerance, nMaxRansacTime, maxContour.size() / 2);

    if (PR_DEBUG_MODE::SHOW_IMAGE == Config::GetInstance()->getDebugMode()) {
        cv::Mat matDisplay;
        cv::cvtColor(matThreshold, matDisplay, CV_GRAY2BGR);
        cv::circle(matDisplay, fitCircleResult.center, ToInt32(fitCircleResult.size.width / 2 + 0.5), BLUE_SCALAR, 2);
        showImage("inspChipCircularMode", matDisplay);
    }

    auto vecInTolPoints = Fitting::findPointsInCircleTol(maxContour, fitCircleResult, fFitTolerance);
    if (vecInTolPoints.size() < maxContour.size() / 2) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CIRCULAR_CHIP;
        return pstRpy->enStatus;
    }

    fitCircleResult.center.x += rectROI.x;
    fitCircleResult.center.y += rectROI.y;
    pstRpy->rotatedRectResult = fitCircleResult;

    cv::circle(pstRpy->matResultImg, fitCircleResult.center, ToInt32(fitCircleResult.size.width / 2 + 0.5), BLUE_SCALAR, 2);
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::lrnContour(const PR_LRN_CONTOUR_CMD *const pstCmd, PR_LRN_CONTOUR_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        (pstCmd->rectROI.x + pstCmd->rectROI.width) > pstCmd->matInputImg.cols ||
        (pstCmd->rectROI.y + pstCmd->rectROI.height) > pstCmd->matInputImg.rows) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The input ROI rect (%d, %d, %d, %d) is invalid.",
            pstCmd->rectROI.x, pstCmd->rectROI.y, pstCmd->rectROI.width, pstCmd->rectROI.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseLrnContour);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    cv::Mat matGray, matBlur, matThreshold;
    if (matROI.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI.clone();
    cv::GaussianBlur(matGray, matBlur, cv::Size(5, 5), 2, 2);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Blur image", matBlur);

    pstRpy->nThreshold = pstCmd->nThreshold;
    if (pstCmd->bAutoThreshold)
        pstRpy->nThreshold = _autoThreshold(matBlur);
    cv::threshold(matBlur, matThreshold, pstRpy->nThreshold, PR_MAX_GRAY_LEVEL, cv::ThresholdTypes::THRESH_BINARY);
    cv::Mat matContour = matThreshold.clone();

    VectorOfVectorOfPoint vecContours;
    cv::findContours(matThreshold, vecContours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat matDraw = matROI.clone();
    // Filter contours
    int index = 0;
    for (auto contour : vecContours) {
        auto area = cv::contourArea(contour);
        if (area > 1000) {
            pstRpy->vecContours.push_back(contour);
            if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
                cv::RNG rng(12345);
                cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                cv::drawContours(matDraw, vecContours, index, color);
            }
        }
        ++ index;
    }
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Contours", matDraw);

    if (pstRpy->vecContours.empty()) {
        pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_CONTOUR;
        return pstRpy->enStatus;
    }

    if (pstCmd->matInputImg.channels() > 1)
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);

    _writeContourRecord(pstRpy, matBlur, matContour, pstRpy->vecContours);

    for (size_t index = 0; index < pstRpy->vecContours.size(); ++ index) {
        for (auto &point : pstRpy->vecContours[index]) {
            point.x += pstCmd->rectROI.x;
            point.y += pstCmd->rectROI.y;
        }
    }
    cv::polylines(pstRpy->matResultImg, pstRpy->vecContours, true, BLUE_SCALAR, 2);

    pstRpy->enStatus = VisionStatus::OK;

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_writeContourRecord(PR_LRN_CONTOUR_RPY *const pstRpy, const cv::Mat &matTmpl, const cv::Mat &matContour, const VectorOfVectorOfPoint &vecContours) {
    ContourRecordPtr ptrRecord = std::make_shared<ContourRecord>(PR_RECORD_TYPE::CONTOUR);
    ptrRecord->setThreshold(pstRpy->nThreshold);
    ptrRecord->setTmpl(matTmpl);
    ptrRecord->setContourMat(matContour);
    ptrRecord->setContour(vecContours);
    return RecordManager::getInstance()->add(ptrRecord, pstRpy->nRecordId);
}

/*static*/ VisionStatus VisionAlgorithm::inspContour(const PR_INSP_CONTOUR_CMD *const pstCmd, PR_INSP_CONTOUR_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->rectROI.x < 0 || pstCmd->rectROI.y < 0 ||
        pstCmd->rectROI.width <= 0 || pstCmd->rectROI.height <= 0 ||
        (pstCmd->rectROI.x + pstCmd->rectROI.width) > pstCmd->matInputImg.cols ||
        (pstCmd->rectROI.y + pstCmd->rectROI.height) > pstCmd->matInputImg.rows) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The input ROI rect (%d, %d, %d, %d) is invalid.",
            pstCmd->rectROI.x, pstCmd->rectROI.y, pstCmd->rectROI.width, pstCmd->rectROI.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->nRecordId <= 0) {
        char chArrMsg[100];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The input record id %d is invalid.", pstCmd->nRecordId);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return VisionStatus::INVALID_PARAM;
    }

    ContourRecordPtr ptrRecord = std::static_pointer_cast<ContourRecord> (RecordManager::getInstance()->get(pstCmd->nRecordId));
    if (nullptr == ptrRecord) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof (chArrMsg), "Failed to get record ID %d in system.", pstCmd->nRecordId);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspContour);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    cv::Mat matGray, matBlur, matThreshold;
    if (matROI.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI.clone();
    cv::GaussianBlur(matGray, matBlur, cv::Size(5, 5), 2, 2);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Blur image", matBlur);

    cv::Point2f ptResult;
    float fRotation = 0.f, fCorrelation = 0.f;
    cv::Mat matTmpl = ptrRecord->getTmpl();
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Template image", matTmpl);
    pstRpy->enStatus = MatchTmpl::matchTemplate(matBlur, matTmpl, true, PR_OBJECT_MOTION::TRANSLATION, ptResult, fRotation, fCorrelation);
    if (pstRpy->enStatus != VisionStatus::OK) {
        WriteLog("Template match fail.");
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = VisionStatus::OK;

    cv::Mat matCompareROI = cv::Mat(matBlur, cv::Rect(ToInt32(ptResult.x - matTmpl.cols / 2.f), ToInt32(ptResult.y - matTmpl.rows / 2.f), matTmpl.cols, matTmpl.rows));
    cv::Mat matDiff_1 = matTmpl - matCompareROI;
    cv::Mat matDiff_2 = matCompareROI - matTmpl;
    cv::Mat matDiff = matDiff_1 + matDiff_2;
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Original Diff Image", matDiff);

    cv::Mat matContourMask = _generateContourMask(matTmpl.size(), ptrRecord->getContour(), pstCmd->fInnerMaskDepth, pstCmd->fOuterMaskDepth);
    matDiff = matDiff > pstCmd->nDefectThreshold;
    matDiff.setTo(cv::Scalar(0), matContourMask);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Defect image", matDiff);
    // Filter contours
    VectorOfVectorOfPoint vecContours, vecDefectContour;
    cv::findContours(matDiff, vecContours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    const float DEFECT_ON_CONTOUR_RATIO = 1.2f;
    int index = 0;
    for (auto contour : vecContours) {
        auto area = cv::contourArea(contour);
        if (area > pstCmd->fMinDefectArea) {
            cv::Point2f ptCenter = CalcUtils::getContourCtr(contour);
            auto contourTarget = _findNearestContour(ptCenter, ptrRecord->getContour());

            float fInnerFarthestDist, fInnerNearestDist, fOuterFarthestDist, fOuterNearestDist;
            _getContourToContourDistance(contour, contourTarget, fInnerFarthestDist, fInnerNearestDist, fOuterFarthestDist, fOuterNearestDist);

            auto minRect = cv::minAreaRect(contour);
            auto length = minRect.size.width > minRect.size.height ? minRect.size.width : minRect.size.height;

            //Defect inside contour.
            if (fInnerFarthestDist > 0 && fabs(fInnerNearestDist) < pstCmd->fInnerMaskDepth * DEFECT_ON_CONTOUR_RATIO && length > pstCmd->fDefectInnerLengthTol) {
                vecDefectContour.push_back(contour);
            }
            else
            if (fOuterFarthestDist < 0 && fabs(fOuterNearestDist) < pstCmd->fOuterMaskDepth * DEFECT_ON_CONTOUR_RATIO && length > pstCmd->fDefectOuterLengthTol) {
                vecDefectContour.push_back(contour);
            }
        }
        ++ index;
    }

    if (pstCmd->matInputImg.channels() > 1)
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
    else
        cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);

    if (! vecDefectContour.empty()) {
        for (auto &contour : vecDefectContour) {
            for (auto &point : contour) {
                point.x += pstCmd->rectROI.x;
                point.y += pstCmd->rectROI.y;
            }
        }
        cv::polylines(pstRpy->matResultImg, vecDefectContour, true, RED_SCALAR, 2);
        pstRpy->vecDefectContour = vecDefectContour;
        pstRpy->enStatus = VisionStatus::CONTOUR_DEFECT_REJECT;
    }

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VectorOfPoint VisionAlgorithm::_findNearestContour(const cv::Point2f &ptInput, const VectorOfVectorOfPoint &vecContours) {
    float fNearestDist = std::numeric_limits<float>::max();
    VectorOfPoint contourResult;
    cv::Point ptNearestPoint;
    for (const auto &contour : vecContours) {
        float fDist = CalcUtils::calcPointToContourDist(ptInput, contour, ptNearestPoint);
        if (fDist < fNearestDist) {
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
    fInnerNearestDist = fOuterFarthestDist = std::numeric_limits<float>::max();
    for (const auto &point : contourInput) {
        auto distance = ToFloat(cv::pointPolygonTest(contourTarget, point, true));
        // distance < 0 means the point is outside of the target contour.
        if (distance < 0) {
            if (distance < fOuterFarthestDist)
                fOuterFarthestDist = distance;
            if (distance > fOuterNearestDist)
                fOuterNearestDist = distance;
        }
        else if (distance > 0) { // distance > 0 means the point is inside of the target contour.
            if (distance > fInnerFarthestDist)
                fInnerFarthestDist = distance;
            if (distance < fInnerNearestDist)
                fInnerNearestDist = distance;
        }
    }
}

/*static*/ cv::Mat VisionAlgorithm::_generateContourMask(const cv::Size &size, const VectorOfVectorOfPoint &vecContours, float fInnerDepth, float fOuterDepth) {
    cv::Mat matMask = cv::Mat::zeros(size, CV_8UC1);
    if (fabs(fInnerDepth - fOuterDepth) < 1) {
        for (int index = 0; index < (int)(vecContours.size()); ++ index)
            cv::drawContours(matMask, vecContours, index, cv::Scalar(PR_MAX_GRAY_LEVEL), ToInt32(2.f * fOuterDepth));
        return matMask;
    }

    cv::Mat matOuterMask = matMask.clone();
    cv::Mat matInnerMask = matMask.clone();

    {// Generate outer mask
        for (int index = 0; index < (int)(vecContours.size()); ++ index)
            cv::drawContours(matOuterMask, vecContours, index, cv::Scalar(PR_MAX_GRAY_LEVEL), ToInt32(2.f * fOuterDepth));
        cv::fillPoly(matOuterMask, vecContours, cv::Scalar::all(0));
    }

    {// Generate inner mask
        cv::Mat matTmpOuterMask = matMask.clone();
        for (int index = 0; index < (int)(vecContours.size()); ++ index) {
            cv::drawContours(matTmpOuterMask, vecContours, index, cv::Scalar(PR_MAX_GRAY_LEVEL), ToInt32(2.f * fInnerDepth));
            cv::drawContours(matInnerMask, vecContours, index, cv::Scalar(PR_MAX_GRAY_LEVEL), ToInt32(2.f * fInnerDepth));
        }
        cv::fillPoly(matTmpOuterMask, vecContours, cv::Scalar::all(0));
        matInnerMask -= matTmpOuterMask;
    }
    matMask = matInnerMask + matOuterMask;

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        cv::Mat matDisplay = cv::Mat::zeros(size, CV_8UC3);
        matDisplay.setTo(GREEN_SCALAR, matOuterMask);
        matDisplay.setTo(BLUE_SCALAR, matInnerMask);
        for (int index = 0; index < (int)(vecContours.size()); ++index)
            cv::drawContours(matDisplay, vecContours, index, RED_SCALAR, 1);
        showImage("Result Mask", matDisplay);
    }
    return matMask;
}

/*static*/ VisionStatus VisionAlgorithm::inspHole(const PR_INSP_HOLE_CMD *const pstCmd, PR_INSP_HOLE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (! pstCmd->matMask.empty()) {
        if (pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d ).",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matMask.channels() != 1) {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (PR_INSP_HOLE_MODE::RATIO == pstCmd->enInspMode) {
        if (pstCmd->stRatioModeCriteria.fMaxRatio <= pstCmd->stRatioModeCriteria.fMinRatio) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The inspect hole area max ratio %.2f is smaller than the min ratio %.2f.",
                pstCmd->stRatioModeCriteria.fMaxRatio, pstCmd->stRatioModeCriteria.fMinRatio);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->stRatioModeCriteria.fMaxRatio > 1.f) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The inspect hole area max ratio %.2f is larger than 1.", pstCmd->stRatioModeCriteria.fMaxRatio);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->stRatioModeCriteria.fMinRatio < 0.f) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The inspect hole area min ratio %.2f is smallar than 0.", pstCmd->stRatioModeCriteria.fMaxRatio);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }
    else if (PR_INSP_HOLE_MODE::BLOB == pstCmd->enInspMode) {
        if (pstCmd->stBlobModeCriteria.nMaxBlobCount <= pstCmd->stBlobModeCriteria.nMinBlobCount) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The inspect hole max blob count %d is smaller than the min blob count %d.",
                pstCmd->stBlobModeCriteria.nMaxBlobCount, pstCmd->stBlobModeCriteria.nMinBlobCount);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->stBlobModeCriteria.nMinBlobCount < 0) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The inspect hole min blob count %d is smallar than 0.", pstCmd->stBlobModeCriteria.nMinBlobCount);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspHole);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI), matSegmentResult;

    if (pstCmd->bPreprocessedImg) {
        cv::Mat matGray;
        if (pstCmd->matInputImg.channels() > 1)
            cv::cvtColor(matROI, matSegmentResult, CV_BGR2GRAY);
        else
            matSegmentResult = matROI;
    }else {
        cv::Mat matBlur;
        cv::GaussianBlur(matROI, matBlur, cv::Size(5, 5), 2, 2);

        if (pstCmd->enSegmentMethod == PR_IMG_SEGMENT_METHOD::GRAY_SCALE_RANGE) {
            pstRpy->enStatus = _segmentImgByGrayScaleRange(matBlur, pstCmd->stGrayScaleRange, matSegmentResult);
        }
        else if (pstCmd->enSegmentMethod == PR_IMG_SEGMENT_METHOD::COLOR_RANGE) {
            pstRpy->enStatus = _setmentImgByColorRange(matBlur, pstCmd->stColorRange, matSegmentResult);
        }
        else {
            WriteLog("Unsupported image segment method.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstRpy->enStatus != VisionStatus::OK)
        return pstRpy->enStatus;

    cv::Mat matMaskROI;
    if (! pstCmd->matMask.empty())
        matMaskROI = cv::Mat(pstCmd->matMask, pstCmd->rectROI);

    if (! isAutoMode()) {
        //Prepared the result image, should be skiped if in auto mode.
        cv::Mat matSegmentInWholeImg = cv::Mat::zeros(pstCmd->matInputImg.size(), CV_8UC1);
        cv::Mat matCopyROI(matSegmentInWholeImg, pstCmd->rectROI);
        matSegmentResult.copyTo(matCopyROI);
        if (pstCmd->matInputImg.channels() > 1)
            pstRpy->matResultImg = pstCmd->matInputImg.clone();
        else
            cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);
        pstRpy->matResultImg.setTo(YELLOW_SCALAR, matSegmentInWholeImg);
    }

    pstRpy->stRatioModeResult.fRatio = 0.f;
    if (pstCmd->enInspMode == PR_INSP_HOLE_MODE::RATIO)
        _inspHoleByRatioMode(matSegmentResult, matMaskROI, pstCmd->stRatioModeCriteria, pstRpy);
    else {
        _inspHoleByBlobMode(matSegmentResult, matMaskROI, pstCmd->stBlobModeCriteria, pstRpy);

        for (auto &keypoint : pstRpy->stBlobModeResult.vecBlobs) {
            keypoint.pt.x += pstCmd->rectROI.x;
            keypoint.pt.y += pstCmd->rectROI.y;
        }

        if (! isAutoMode())
            cv::drawKeypoints(pstRpy->matResultImg, pstRpy->stBlobModeResult.vecBlobs, pstRpy->matResultImg, RED_SCALAR, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_segmentImgByGrayScaleRange(const cv::Mat &matInput, const PR_INSP_HOLE_CMD::GRAY_SCALE_RANGE &stGrayScaleRange, cv::Mat &matResult) {
    if (stGrayScaleRange.nStart < 0 ||
        stGrayScaleRange.nStart >= stGrayScaleRange.nEnd ||
        stGrayScaleRange.nEnd > PR_MAX_GRAY_LEVEL) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The gray scale range (%d, %d) is invalid.",
            stGrayScaleRange.nStart, stGrayScaleRange.nEnd);
        WriteLog(chArrMsg);
        return VisionStatus::INVALID_PARAM;
    }
    cv::Mat matGray;
    if (matInput.channels() > 1)
        cv::cvtColor(matInput, matGray, CV_BGR2GRAY);
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
        stColorRange.nStartR < 0 || stColorRange.nStartR >= stColorRange.nEndR || stColorRange.nEndR > PR_MAX_GRAY_LEVEL) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The color scale range (%d, %d, %d, %d, %d, %d) is invalid.",
            stColorRange.nStartB, stColorRange.nEndB, stColorRange.nStartG, stColorRange.nEndG, stColorRange.nStartR, stColorRange.nEndR);
        WriteLog(chArrMsg);
        return VisionStatus::INVALID_PARAM;
    }
    if (matInput.channels() < 3) {
        WriteLog("Input image must be color for color range segment.");
        return VisionStatus::INVALID_PARAM;
    }
    std::vector<cv::Mat> vecChannels;
    cv::split(matInput, vecChannels);
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
    if (! matMask.empty())
        matInputLocal.setTo(cv::Scalar(0), cv::Scalar(PR_MAX_GRAY_LEVEL) - matMask);

    auto fSelectedCount = ToFloat(cv::countNonZero(matInputLocal));
    pstRpy->stRatioModeResult.fRatio = fSelectedCount / (matInput.rows * matInput.cols);
    if (pstRpy->stRatioModeResult.fRatio < stCriteria.fMinRatio) {
        pstRpy->enStatus = VisionStatus::RATIO_UNDER_LIMIT;
        return pstRpy->enStatus;
    }
    else if (pstRpy->stRatioModeResult.fRatio > stCriteria.fMaxRatio) {
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

    if (stCriteria.bEnableAdvancedCriteria) {
        // Filter by Circularity
        params.filterByCircularity = true;
        params.minCircularity = stCriteria.stAdvancedCriteria.fMinCircularity;
        params.maxCircularity = stCriteria.stAdvancedCriteria.fMaxCircularity;

        // Filter by Inertia
        params.filterByInertia = true;
        params.maxInertiaRatio = stCriteria.stAdvancedCriteria.fMaxLengthWidthRatio;
        params.minInertiaRatio = stCriteria.stAdvancedCriteria.fMinLengthWidthRatio;
    }
    else {
        params.filterByCircularity = false;
        params.filterByInertia = false;
    }

    // Set up detector with params
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    std::vector<cv::Mat> vecInputMat;
    vecInputMat.push_back(matInput);

    VectorOfVectorKeyPoint keyPoints;
    detector->detect(vecInputMat, keyPoints, matMask);
    pstRpy->stBlobModeResult.vecBlobs = keyPoints[0];
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        cv::Mat matDisplay;
        cv::cvtColor(matInput, matDisplay, CV_GRAY2BGR);
        cv::drawKeypoints(matDisplay, keyPoints[0], matDisplay, RED_SCALAR, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        showImage("Find blob result", matDisplay);
    }

    if (ToInt16(keyPoints[0].size()) < stCriteria.nMinBlobCount || ToInt16(keyPoints[0].size()) > stCriteria.nMaxBlobCount) {
        pstRpy->enStatus = VisionStatus::BLOB_COUNT_OUT_OF_RANGE;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::inspLead(const PR_INSP_LEAD_CMD *const pstCmd, PR_INSP_LEAD_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspLead);

    pstRpy->enStatus = VisionStatus::OK;
    for (const auto &stLeadInfo : pstCmd->vecLeads) {
        PR_INSP_LEAD_RPY::LEAD_RESULT stLeadResult;
        VisionStatus enStatus = _inspSingleLead(pstCmd->matInputImg, stLeadInfo, pstCmd, stLeadResult);
        pstRpy->vecLeadResult.push_back(stLeadResult);
        if (VisionStatus::OK == pstRpy->enStatus)
            pstRpy->enStatus = enStatus;
    }

    if (! isAutoMode()) {
        if (pstCmd->matInputImg.channels() > 1)
            pstRpy->matResultImg = pstCmd->matInputImg.clone();
        else
            cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);
        for (size_t i = 0; i < pstCmd->vecLeads.size(); ++ i) {
            auto vecPoint2f = CalcUtils::getCornerOfRotatedRect(pstCmd->vecLeads[i].rectSrchWindow);
            VectorOfVectorOfPoint vecVecSrchWindowPoint(1);
            for (const auto &point : vecPoint2f)
                vecVecSrchWindowPoint[0].push_back(point);

            if (pstRpy->vecLeadResult[i].bFound) {
                cv::polylines(pstRpy->matResultImg, vecVecSrchWindowPoint, true, GREEN_SCALAR);

                auto vecPoint2f = CalcUtils::getCornerOfRotatedRect(pstRpy->vecLeadResult[i].rectLead);
                VectorOfVectorOfPoint vecVecLeadPoint(1);
                for (const auto &point : vecPoint2f)
                    vecVecLeadPoint[0].push_back(point);
                cv::polylines(pstRpy->matResultImg, vecVecLeadPoint, true, BLUE_SCALAR);
            }
            else {
                cv::polylines(pstRpy->matResultImg, vecVecSrchWindowPoint, true, RED_SCALAR);
            }
        }
    }

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::inspLeadTmpl(const PR_INSP_LEAD_TMPL_CMD *const pstCmd, PR_INSP_LEAD_TMPL_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (pstCmd->nLeadRecordId <= 0) {
        char msg[100];
        _snprintf(msg, sizeof(msg), "The input lead record id %d is invalid.", pstCmd->nLeadRecordId);
        WriteLog(msg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspLeadTmpl);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI), matGrayROI;
    if (matROI.type() == CV_8UC1)
        matGrayROI = matROI;
    else if (matROI.type() == CV_8UC3)
        cv::cvtColor(matROI, matGrayROI, CV_BGR2GRAY);
    else {
        WriteLog("Input image format is invalid.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (ConfigInstance->getDebugMode() == Vision::PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("ROI Image", matGrayROI);

    TmplRecordPtr ptrLeadRecord = std::static_pointer_cast<TmplRecord>(RecordManagerInstance->get(pstCmd->nLeadRecordId));
    if (nullptr == ptrLeadRecord) {
        char chArrMsg[100];
        _snprintf(chArrMsg, sizeof (chArrMsg), "Failed to get lead record ID %d in system.", pstCmd->nLeadRecordId);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (ConfigInstance->getDebugMode() == Vision::PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Lead template", ptrLeadRecord->getTmpl());

    cv::Point2f ptLeadPos, ptPadPos;
    float fLeadRotation, fPadRotation, fLeadScore, fPadScore;
    MatchTmpl::matchTemplate(matGrayROI, ptrLeadRecord->getTmpl(), false, PR_OBJECT_MOTION::TRANSLATION, ptLeadPos, fLeadRotation, fLeadScore);
    if (fLeadScore * ConstToPercentage < pstCmd->fMinMatchScore) {
        pstRpy->enStatus = VisionStatus::NOT_FIND_LEAD;
        FINISH_LOGCASE_EX;
        return pstRpy->enStatus;
    }

    TmplRecordPtr ptrPadRecord;
    if (pstCmd->nPadRecordId > 0) {
        ptrPadRecord = std::static_pointer_cast<TmplRecord>(RecordManagerInstance->get(pstCmd->nPadRecordId));
        if (nullptr == ptrPadRecord) {
            char chArrMsg[100];
            _snprintf(chArrMsg, sizeof(chArrMsg), "Failed to get pad record ID %d in system.", pstCmd->nPadRecordId);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (ConfigInstance->getDebugMode() == Vision::PR_DEBUG_MODE::SHOW_IMAGE)
            showImage("Pad template", ptrPadRecord->getTmpl());

        MatchTmpl::matchTemplate(matGrayROI, ptrPadRecord->getTmpl(), false, PR_OBJECT_MOTION::TRANSLATION, ptPadPos, fPadRotation, fPadScore);
        if (fPadScore * ConstToPercentage < pstCmd->fMinMatchScore) {
            pstRpy->enStatus = VisionStatus::NOT_FIND_LEAD;
            FINISH_LOGCASE_EX;
            return pstRpy->enStatus;
        }

        pstRpy->enStatus = VisionStatus::OK;
        if (pstCmd->rectROI.width > pstCmd->rectROI.height) // Lead is horizontal
        {
            pstRpy->fLeadOffsetX = fabs(ptLeadPos.x - ptPadPos.x) - pstCmd->fLrnedPadLeadDist;
            pstRpy->fLeadOffsetY = fabs(ptPadPos.y - ptLeadPos.y);
        }else { // Lead is vertical
            pstRpy->fLeadOffsetX = fabs(ptPadPos.x - ptLeadPos.x);
            pstRpy->fLeadOffsetY = fabs(ptLeadPos.y - ptPadPos.y) - pstCmd->fLrnedPadLeadDist;
        }
    }else {
        throw std::exception("Software not implemented.");
    }

    if (pstRpy->fLeadOffsetX > pstCmd->fMaxLeadOffsetX || pstRpy->fLeadOffsetY > pstCmd->fMaxLeadOffsetY) {
        pstRpy->enStatus = VisionStatus::LEAD_OFFSET_OVER_LIMIT;
    }

    pstRpy->ptLeadPos = cv::Point2f(pstCmd->rectROI.x + ptLeadPos.x, pstCmd->rectROI.y + ptLeadPos.y);
    if (ptrPadRecord != nullptr)
        pstRpy->ptPadPos = cv::Point2f(pstCmd->rectROI.x + ptPadPos.x, pstCmd->rectROI.y + ptPadPos.y);

    if (!isAutoMode()) {
        if (pstCmd->matInputImg.channels() > 1)
            pstRpy->matResultImg = pstCmd->matInputImg.clone();
        else
            cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);
        if (ptrPadRecord != nullptr) {
            auto matPadTmpl = ptrPadRecord->getTmpl();
            cv::Rect rectPad(ToInt32(pstRpy->ptPadPos.x - matPadTmpl.cols / 2), ToInt32(pstRpy->ptPadPos.y - matPadTmpl.rows / 2), matPadTmpl.cols, matPadTmpl.rows);
            cv::rectangle(pstRpy->matResultImg, rectPad, GREEN_SCALAR, 2, cv::LineTypes::LINE_8);
        }

        auto matLeadTmpl = ptrLeadRecord->getTmpl();
        cv::Rect rectLead(ToInt32(pstRpy->ptLeadPos.x - matLeadTmpl.cols / 2), ToInt32(pstRpy->ptLeadPos.y - matLeadTmpl.rows / 2), matLeadTmpl.cols, matLeadTmpl.rows);
        cv::rectangle(pstRpy->matResultImg, rectLead, GREEN_SCALAR, 2, cv::LineTypes::LINE_8);
    }

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_inspSingleLead(
    const cv::Mat                              &matInput,
    const PR_INSP_LEAD_CMD::LEAD_INPUT_INFO    &stLeadInput,
    const PR_INSP_LEAD_CMD                     *pstCmd,
    PR_INSP_LEAD_RPY::LEAD_RESULT              &stLeadResult) {
    cv::Mat matROI;
    VisionStatus enStatus = _extractRotatedROI(matInput, stLeadInput.rectSrchWindow, matROI);
    PR_DIRECTION enSrchDirection;
    if (stLeadInput.rectSrchWindow.angle <= 0.1) {
        if (stLeadInput.rectSrchWindow.center.x < pstCmd->rectChipWindow.center.x && stLeadInput.rectSrchWindow.center.y >(pstCmd->rectChipWindow.center.y - pstCmd->rectChipWindow.size.height / 2))
            enSrchDirection = PR_DIRECTION::LEFT;
        else if (stLeadInput.rectSrchWindow.center.x > pstCmd->rectChipWindow.center.x && stLeadInput.rectSrchWindow.center.y > (pstCmd->rectChipWindow.center.y - pstCmd->rectChipWindow.size.height / 2))
            enSrchDirection = PR_DIRECTION::RIGHT;
        else if (stLeadInput.rectSrchWindow.center.y < pstCmd->rectChipWindow.center.y && stLeadInput.rectSrchWindow.center.x >(pstCmd->rectChipWindow.center.x - pstCmd->rectChipWindow.size.width / 2))
            enSrchDirection = PR_DIRECTION::UP;
        else
            enSrchDirection = PR_DIRECTION::DOWN;
    }
    else
        enSrchDirection = PR_DIRECTION::RIGHT;
    cv::Mat matGray, matThreshold;
    if (matROI.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI.clone();
    auto nThreshold = _autoThreshold(matGray);
    cv::threshold(matGray, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, cv::ThresholdTypes::THRESH_BINARY);
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Lead Threshold", matThreshold);
    matThreshold /= PR_MAX_GRAY_LEVEL;
    cv::Mat matOneRow, matOneCol;
    int nSrchWindowWidth = 0;
    if (PR_DIRECTION::LEFT == enSrchDirection || PR_DIRECTION::RIGHT == enSrchDirection) {
        cv::reduce(matThreshold, matOneRow, 0, cv::ReduceTypes::REDUCE_SUM, CV_32SC1);
        nSrchWindowWidth = ToInt32(stLeadInput.rectSrchWindow.size.height);
    }
    else {
        cv::reduce(matThreshold, matOneCol, 1, cv::ReduceTypes::REDUCE_SUM, CV_32SC1);
        cv::transpose(matOneCol, matOneRow);
        nSrchWindowWidth = ToInt32(stLeadInput.rectSrchWindow.size.width);
    }

    auto nStartWidthThreshold = ToInt32(nSrchWindowWidth * pstCmd->fLeadStartWidthRatio);
    auto nEndWidthThreshold = ToInt32(nSrchWindowWidth * pstCmd->fLeadEndWidthRatio);
    int nConsecutiveLength = 0;
    int nLeadStart = 0, nLeadEnd = 0;
    for (int i = 0; i < matOneRow.cols; ++ i) {
        if (matOneRow.at<Int32>(0, i) > nStartWidthThreshold) {
            ++ nConsecutiveLength;
        }
        else
            nConsecutiveLength = 0;
        if (nConsecutiveLength >= pstCmd->nLeadStartConsecutiveLength) {
            nLeadStart = i - nConsecutiveLength;
            break;
        }
    }

    nConsecutiveLength = 0;
    for (int i = matOneRow.cols - 1; i >= 0; -- i) {
        if (matOneRow.at<Int32>(0, i) > nStartWidthThreshold) {
            ++ nConsecutiveLength;
        }
        else
            nConsecutiveLength = 0;
        if (nConsecutiveLength >= pstCmd->nLeadStartConsecutiveLength) {
            nLeadEnd = i + nConsecutiveLength;
            break;
        }
    }
    if (nLeadStart == 0 || nLeadEnd == 0 || nLeadStart >= nLeadEnd) {
        stLeadResult.bFound = false;
        return VisionStatus::NOT_FIND_LEAD;
    }

    if (abs(nLeadEnd - nLeadStart) < 0.1 * matOneRow.cols) {
        WriteLog("Found lead length is too short and has been rejected.");
        stLeadResult.bFound = false;
        return VisionStatus::NOT_FIND_LEAD;
    }

    const int GUASSIAN_DIFF_WIDTH = 2;
    const float GUASSIAN_KERNEL_SSQ = 1.f;
    cv::Mat matGuassinKernel = CalcUtils::generateGuassinDiffKernel(GUASSIAN_DIFF_WIDTH, GUASSIAN_KERNEL_SSQ);
    if (PR_DIRECTION::LEFT == enSrchDirection || PR_DIRECTION::RIGHT == enSrchDirection) {
        cv::Mat matAverage;
        cv::reduce(matGray, matAverage, 1, cv::ReduceTypes::REDUCE_AVG);

        cv::Mat matGuassianDiffResult;
        cv::transpose(matGuassinKernel, matGuassinKernel);
        CalcUtils::filter2D_Conv(matAverage, matGuassianDiffResult, CV_32F, matGuassinKernel);
        double minValue = 0., maxValue = 0.;
        cv::Point ptMin, ptMax;
        cv::minMaxLoc(matGuassianDiffResult, &minValue, &maxValue, &ptMin, &ptMax);
        int nLeadTop = ptMin.y;
        int nLeadBottom = ptMax.y;
        stLeadResult.rectLead.size.height = ToFloat(nLeadBottom - nLeadTop);
        stLeadResult.rectLead.center.y = stLeadInput.rectSrchWindow.center.y - matROI.rows / 2 + (nLeadBottom + nLeadTop) / 2;
    }
    else {
        cv::Mat matAverage;
        cv::reduce(matGray, matAverage, 0, cv::ReduceTypes::REDUCE_AVG);

        cv::Mat matGuassianDiffResult;
        CalcUtils::filter2D_Conv(matAverage, matGuassianDiffResult, CV_32F, matGuassinKernel);
        double minValue = 0., maxValue = 0.;
        cv::Point ptMin, ptMax;
        cv::minMaxLoc(matGuassianDiffResult, &minValue, &maxValue, &ptMin, &ptMax);
        int nLeadLeft = ptMin.x;
        int nLeadRight = ptMax.y;
        stLeadResult.rectLead.size.height = ToFloat(nLeadRight - nLeadLeft);
        stLeadResult.rectLead.center.x = stLeadInput.rectSrchWindow.center.x - matROI.cols / 2 + (nLeadRight + nLeadLeft) / 2;
    }
    stLeadResult.rectLead.size.width = ToFloat(nLeadEnd - nLeadStart);
    if (stLeadResult.rectLead.size.height <= 0) {
        stLeadResult.bFound = false;
        return VisionStatus::NOT_FIND_LEAD;
    }
    stLeadResult.bFound = true;
    stLeadResult.rectLead.angle = stLeadInput.rectSrchWindow.angle;
    if (PR_DIRECTION::LEFT == enSrchDirection || PR_DIRECTION::RIGHT == enSrchDirection)
        stLeadResult.rectLead.center.x = stLeadInput.rectSrchWindow.center.x - matROI.cols / 2 + (nLeadEnd + nLeadStart) / 2;
    else
        stLeadResult.rectLead.center.y = stLeadInput.rectSrchWindow.center.y - matROI.rows / 2 + (nLeadEnd + nLeadStart) / 2;
    return enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::inspPolarity(const PR_INSP_POLARITY_CMD *const pstCmd, PR_INSP_POLARITY_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectInspROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    pstRpy->enStatus = _checkInputROI(pstCmd->rectCompareROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInspPolarity);

    cv::Mat matInspROI(pstCmd->matInputImg, pstCmd->rectInspROI);
    cv::Mat matCompareROI(pstCmd->matInputImg, pstCmd->rectCompareROI);
    cv::Mat matInspROIGray, matCompareROIGray;
    if (matInspROI.channels() > 1) {
        cv::cvtColor(matInspROI, matInspROIGray, CV_BGR2GRAY);
        cv::cvtColor(matCompareROI, matCompareROIGray, CV_BGR2GRAY);
    }
    else {
        matInspROIGray = matInspROI.clone();
        matCompareROIGray = matCompareROI.clone();
    }

    auto meanGrayScaleInspROI = ToInt32(cv::mean(matInspROIGray)[0]);
    auto meanGrayScaleCompareROI = ToInt32(cv::mean(matCompareROIGray)[0]);
    pstRpy->nGrayScaleDiff = meanGrayScaleInspROI - meanGrayScaleCompareROI;
    if (pstCmd->enInspROIAttribute == PR_OBJECT_ATTRIBUTE::BRIGHT) {
        if (pstRpy->nGrayScaleDiff >= pstCmd->nGrayScaleDiffTol)
            pstRpy->enStatus = VisionStatus::OK;
        else
            pstRpy->enStatus = VisionStatus::POLARITY_FAIL;
    }
    else if (pstCmd->enInspROIAttribute == PR_OBJECT_ATTRIBUTE::DARK) {
        if ((-pstRpy->nGrayScaleDiff) >= pstCmd->nGrayScaleDiffTol)
            pstRpy->enStatus = VisionStatus::OK;
        else
            pstRpy->enStatus = VisionStatus::POLARITY_FAIL;
    }

    if (! isAutoMode()) {
        if (pstCmd->matInputImg.channels() > 1)
            pstRpy->matResultImg = pstCmd->matInputImg.clone();
        else
            cv::cvtColor(pstCmd->matInputImg, pstRpy->matResultImg, CV_GRAY2BGR);
        cv::rectangle(pstRpy->matResultImg, pstCmd->rectInspROI, BLUE_SCALAR, 2, cv::LineTypes::LINE_8);
        cv::rectangle(pstRpy->matResultImg, pstCmd->rectCompareROI, BLUE_SCALAR, 2, cv::LineTypes::LINE_4);

        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 2;
        int baseline = 0;

        {
            char strGrayScale[100];
            _snprintf(strGrayScale, sizeof(strGrayScale), "%d", meanGrayScaleInspROI);
            cv::Size textSize = cv::getTextSize(strGrayScale, fontFace, fontScale, thickness, &baseline);
            //The height use '+' because text origin start from left-bottom.
            cv::Point ptTextOrg(pstCmd->rectInspROI.x + (pstCmd->rectInspROI.width - textSize.width) / 2, pstCmd->rectInspROI.y + (pstCmd->rectInspROI.height + textSize.height) / 2);
            cv::putText(pstRpy->matResultImg, strGrayScale, ptTextOrg, fontFace, fontScale, CYAN_SCALAR, thickness);
        }

        {
            char strGrayScale[100];
            _snprintf(strGrayScale, sizeof(strGrayScale), "%d", meanGrayScaleCompareROI);
            cv::Size textSize = cv::getTextSize(strGrayScale, fontFace, fontScale, thickness, &baseline);
            //The height use '+' because text origin start from left-bottom.
            cv::Point ptTextOrg(pstCmd->rectCompareROI.x + (pstCmd->rectCompareROI.width - textSize.width) / 2, pstCmd->rectCompareROI.y + (pstCmd->rectCompareROI.height + textSize.height) / 2);
            cv::putText(pstRpy->matResultImg, strGrayScale, ptTextOrg, fontFace, fontScale, CYAN_SCALAR, thickness);
        }
    }

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::gridAvgGrayScale(const PR_GRID_AVG_GRAY_SCALE_CMD *const pstCmd, PR_GRID_AVG_GRAY_SCALE_RPY *const pstRpy) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecInputImgs.empty()) {
        WriteLog("There is no input image.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &matInput : pstCmd->vecInputImgs) {
        if (matInput.empty()) {
            WriteLog("Input image is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }
    cv::Mat matAverage, matGray, matFloat;
    if (pstCmd->vecInputImgs[0].channels() > 1)
        cv::cvtColor(pstCmd->vecInputImgs[0], matGray, CV_BGR2GRAY);
    else
        matGray = pstCmd->vecInputImgs[0].clone();
    matGray.convertTo(matAverage, CV_32FC1);

    if (pstCmd->vecInputImgs.size() > 1) {
        for (size_t i = 1; i < pstCmd->vecInputImgs.size(); ++ i) {
            if (pstCmd->vecInputImgs[i].size() != matAverage.size()) {
                char chArrMsg[1000];
                _snprintf(chArrMsg, sizeof (chArrMsg), "The input image [%d] size is not consistent.", i);
                WriteLog(chArrMsg);
                pstRpy->enStatus = VisionStatus::INVALID_PARAM;
                return pstRpy->enStatus;
            }

            if (pstCmd->vecInputImgs[i].channels() > 1)
                cv::cvtColor(pstCmd->vecInputImgs[i], matGray, CV_BGR2GRAY);
            else
                matGray = pstCmd->vecInputImgs[i].clone();
            matGray.convertTo(matFloat, CV_32FC1);
            matAverage += matFloat;
        }
    }

    matAverage /= ToFloat(pstCmd->vecInputImgs.size());
    matAverage.convertTo(pstRpy->matResultImg, CV_8UC1);
    cv::cvtColor(pstRpy->matResultImg, pstRpy->matResultImg, CV_GRAY2BGR);

    int ROWS = matAverage.rows;
    int COLS = matAverage.cols;
    int nIntervalX = matAverage.cols / pstCmd->nGridCol;
    int nIntervalY = matAverage.rows / pstCmd->nGridRow;

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 3;
    pstRpy->vecVecGrayScale.clear();
    for (int j = 0; j < pstCmd->nGridRow; ++ j) {
        std::vector<float> vecFloat;
        for (int i = 0; i < pstCmd->nGridCol; ++ i) {
            cv::Rect rectROI(i * nIntervalX, j * nIntervalY, nIntervalX, nIntervalY);
            cv::Mat matROI(matAverage, rectROI);
            float fAverage = ToFloat(cv::sum(matROI)[0] / rectROI.area());
            vecFloat.push_back(fAverage);

            char strAverage[100];
            _snprintf(strAverage, sizeof(strAverage), "%.2f", fAverage);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(strAverage, fontFace, fontScale, thickness, &baseline);
            //The height use '+' because text origin start from left-bottom.
            cv::Point ptTextOrg(rectROI.x + (rectROI.width - textSize.width) / 2, rectROI.y + (rectROI.height + textSize.height) / 2);
            cv::putText(pstRpy->matResultImg, strAverage, ptTextOrg, fontFace, fontScale, BLUE_SCALAR, thickness);
        }
        pstRpy->vecVecGrayScale.push_back(vecFloat);
    }

    int nGridLineSize = 3;
    cv::Scalar scalarCyan(255, 255, 0);
    for (int i = 1; i < pstCmd->nGridCol; ++ i)
        cv::line(pstRpy->matResultImg, cv::Point(i * nIntervalX, 0), cv::Point(i * nIntervalX, ROWS), scalarCyan, nGridLineSize);
    for (int i = 1; i < pstCmd->nGridRow; ++ i)
        cv::line(pstRpy->matResultImg, cv::Point(0, i * nIntervalY), cv::Point(COLS, i * nIntervalY), scalarCyan, nGridLineSize);

    MARK_FUNCTION_START_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calib3DBase(const PR_CALIB_3D_BASE_CMD *const pstCmd, PR_CALIB_3D_BASE_RPY *const pstRpy, bool bReplay/* = false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecInputImgs.size() != 12) {
        WriteLog("The input image count is not 12.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &mat : pstCmd->vecInputImgs) {
        if (mat.empty()) {
            WriteLog("The input image is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }
    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseCalib3DBase);

    Unwrap::calib3DBase(pstCmd, pstRpy);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calc3DBase(const PR_CALC_3D_BASE_CMD *const pstCmd, PR_CALC_3D_BASE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matBaseSurfaceParam.empty()) {
        WriteLog("matBaseSurfaceParam is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    MARK_FUNCTION_START_TIME;

    pstRpy->matBaseSurface = Unwrap::calculateBaseSurface(pstCmd->nImageRows, pstCmd->nImageCols, pstCmd->matBaseSurfaceParam);

    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calib3DHeight(const PR_CALIB_3D_HEIGHT_CMD *const pstCmd, PR_CALIB_3D_HEIGHT_RPY *const pstRpy, bool bReplay/* = false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecInputImgs.size() < 8) {
        WriteLog("The input image count is less than 8.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &mat : pstCmd->vecInputImgs) {
        if (mat.empty()) {
            WriteLog("The input image is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstCmd->matThickToThinK.empty()) {
        WriteLog("matThickToThinStripeK is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matBaseWrappedAlpha.empty()) {
        WriteLog("matBaseWrappedAlpha is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matBaseWrappedAlpha.size() != pstCmd->vecInputImgs[0].size()) {
        WriteLog("The size of matBaseWrappedAlpha is not match with the input image size.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matBaseWrappedBeta.empty()) {
        WriteLog("matBaseWrappedBeta is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->bUseThinnestPattern) {
        if (pstCmd->matBaseWrappedGamma.empty()) {
            WriteLog("matBaseWrappedGamma is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matThickToThinnestK.empty()) {
            WriteLog("matThickToThinnestK is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->vecInputImgs.size() != 3 * PR_GROUP_TEXTURE_IMG_COUNT) {
            WriteLog("Please input 12 images if using the thinnest pattern.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstCmd->nBlockStepCount <= 0 || pstCmd->nBlockStepCount > PR_MAX_AUTO_THRESHOLD_COUNT) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The BlockStepCount %d is not in range [1, 5].", pstCmd->nBlockStepCount);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->nResultImgGridRow <= 0 || pstCmd->nResultImgGridRow > 100 ||
        pstCmd->nResultImgGridCol <= 0 || pstCmd->nResultImgGridCol > 100) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The Result image grid row %d or column %d is not in range [1, 100].", pstCmd->nResultImgGridRow, pstCmd->nResultImgGridCol);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->szMeasureWinSize.width <= 0 || pstCmd->szMeasureWinSize.width > pstCmd->vecInputImgs[0].cols ||
        pstCmd->szMeasureWinSize.height <= 0 || pstCmd->szMeasureWinSize.height > pstCmd->vecInputImgs[0].rows) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The measure window size (%d, %d) is invalid.", pstCmd->szMeasureWinSize.width, pstCmd->szMeasureWinSize.height);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseCalib3DHeight);

    Unwrap::calib3DHeight(pstCmd, pstRpy);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::integrate3DCalib(const PR_INTEGRATE_3D_CALIB_CMD *const pstCmd, PR_INTEGRATE_3D_CALIB_RPY *const pstRpy, bool bReplay/* = false*/) {
    if (pstCmd->vecCalibData.empty()) {
        WriteLog("The input calibration data vector is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &stCalibData : pstCmd->vecCalibData) {
        if (stCalibData.matPhase.empty()) {
            WriteLog("The input phase data is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (stCalibData.matDivideStepIndex.empty()) {
            WriteLog("The input phase step divide data is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;

    Unwrap::integrate3DCalib(pstCmd, pstRpy);

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::motorCalib3D(const PR_MOTOR_CALIB_3D_CMD *const pstCmd, PR_MOTOR_CALIB_3D_RPY *const pstRpy, bool bReplay /*= false*/) {
    if (pstCmd->vecPairHeightPhase.size() < 2) {
        WriteLog("The input height count is less than 2.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    Unwrap::motorCalib3D(pstCmd, pstRpy);

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::motorCalib3DNew(const PR_MOTOR_CALIB_3D_CMD *const pstCmd, PR_MOTOR_CALIB_3D_NEW_RPY *const pstRpy, bool bReplay /*= false*/) {
    if (pstCmd->vecPairHeightPhase.size() < 2) {
        WriteLog("The input height count is less than 2.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    Unwrap::motorCalib3DNew(pstCmd, pstRpy);

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calibDlpOffset(const PR_CALIB_DLP_OFFSET_CMD *const pstCmd, PR_CALIB_DLP_OFFSET_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->nCalibPosRows <= 0 || pstCmd->nCalibPosCols <= 0) {
        std::stringstream ss;
        ss << "The nCalibPosRows " << pstCmd->nCalibPosRows << " and nCalibPosCols " << pstCmd->nCalibPosCols << " is invalid";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->fResolution < 0.01f || pstCmd->fResolution > 0.02f) {
        std::stringstream ss;
        ss << "The fResolution " << pstCmd->fResolution << " is invalid";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (int i = 0; i < NUM_OF_DLP; ++ i) {
        if (pstCmd->arrVecDlpH[i].empty() || pstCmd->arrVecDlpH[i].size() != pstCmd->nCalibPosRows * pstCmd->nCalibPosCols) {
            std::stringstream ss;
            ss << "DLP " << i + 1 << " input number of frame heights " << pstCmd->arrVecDlpH[i].size() << " is invalid";
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;

    Unwrap::calibDlpOffset(pstCmd, pstRpy);

    MARK_FUNCTION_END_TIME;

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calc3DHeight(const PR_CALC_3D_HEIGHT_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecInputImgs.size() < 8) {
        WriteLog("The input image count is less than 8.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &mat : pstCmd->vecInputImgs) {
        if (mat.empty()) {
            WriteLog("The input image is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstCmd->matThickToThinK.empty()) {
        WriteLog("The matThickToThinK is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matBaseWrappedAlpha.empty()) {
        WriteLog("matBaseWrappedAlpha is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matBaseWrappedAlpha.size() != pstCmd->vecInputImgs[0].size()) {
        WriteLog("The size of matBaseWrappedAlpha is not match with the input image size..");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matBaseWrappedBeta.empty()) {
        WriteLog("matBaseWrappedBeta is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->nRemoveBetaJumpMaxSpan < 0 || pstCmd->nRemoveBetaJumpMinSpan < 0 || pstCmd->nRemoveBetaJumpMinSpan > pstCmd->nRemoveBetaJumpMaxSpan) {
        std::stringstream ss;
        ss << "The RemoveBetaJumpMaxSpan " << pstCmd->nRemoveBetaJumpMaxSpan << " and RemoveBetaJumpMinSpan " << pstCmd->nRemoveBetaJumpMinSpan << " is invalid.";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->bUseThinnestPattern) {
        if (pstCmd->matBaseWrappedGamma.empty()) {
            WriteLog("matBaseWrappedGamma is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matThickToThinnestK.empty()) {
            WriteLog("matThickToThinnestK is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->vecInputImgs.size() != 3 * PR_GROUP_TEXTURE_IMG_COUNT) {
            WriteLog("Please input 12 images if using the thinnest pattern.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    //The Calc3DHeight logcase need to log 12 images, which is too slow when log always, and not useful right now, so disable it first.
    //SETUP_LOGCASE(LogCaseCalc3DHeight);

    Unwrap::calc3DHeight(pstCmd, pstRpy);

    //FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calc3DHeightNew(const PR_CALC_3D_HEIGHT_NEW_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecInputImgs.size() < 8) {
        WriteLog("The input image count is less than 8.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &mat : pstCmd->vecInputImgs) {
        if (mat.empty()) {
            WriteLog("The input image is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstCmd->matThickToThinK.empty()) {
        WriteLog("The matThickToThinK is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matBaseWrappedAlpha.empty()) {
        WriteLog("matBaseWrappedAlpha is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matBaseWrappedAlpha.size() != pstCmd->vecInputImgs[0].size()) {
        WriteLog("The size of matBaseWrappedAlpha is not match with the input image size..");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matBaseWrappedBeta.empty()) {
        WriteLog("matBaseWrappedBeta is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->nRemoveJumpSpan < 0) {
        std::stringstream ss;
        ss << "The nRemoveJumpSpan " << pstCmd->nRemoveJumpSpan << " is invalid.";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->nCompareRemoveJumpSpan < 0) {
        std::stringstream ss;
        ss << "The nCompareRemoveJumpSpan " << pstCmd->nCompareRemoveJumpSpan << " is invalid.";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->bUseThinnestPattern) {
        if (pstCmd->matBaseWrappedGamma.empty()) {
            WriteLog("matBaseWrappedGamma is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matThickToThinnestK.empty()) {
            WriteLog("matThickToThinnestK is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->vecInputImgs.size() != 3 * PR_GROUP_TEXTURE_IMG_COUNT) {
            WriteLog("Please input 12 images if using the thinnest pattern.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    //The Calc3DHeight logcase need to log 12 images, which is too slow when log always, and not useful right now, so disable it first.
    //SETUP_LOGCASE(LogCaseCalc3DHeight);

    Unwrap::calc3DHeightNew(pstCmd, pstRpy);

    //FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calc3DHeightGpu(const PR_CALC_3D_HEIGHT_GPU_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecInputImgs.size() < 8) {
        WriteLog("The input image count is less than 8.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &mat : pstCmd->vecInputImgs) {
        if (mat.empty()) {
            WriteLog("The input image is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstCmd->matThickToThinK.empty()) {
        WriteLog("The matThickToThinK is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->nRemoveJumpSpan < 0) {
        std::stringstream ss;
        ss << "The nRemoveJumpSpan " << pstCmd->nRemoveJumpSpan << " is invalid.";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->nCompareRemoveJumpSpan < 0) {
        std::stringstream ss;
        ss << "The nCompareRemoveJumpSpan " << pstCmd->nCompareRemoveJumpSpan << " is invalid.";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->bUseThinnestPattern) {
        if (pstCmd->matThickToThinnestK.empty()) {
            WriteLog("matThickToThinnestK is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->vecInputImgs.size() != 3 * PR_GROUP_TEXTURE_IMG_COUNT) {
            WriteLog("Please input 12 images if using the thinnest pattern.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    //The Calc3DHeight logcase need to log 12 images, which is too slow when log always, and not useful right now, so disable it first.
    //SETUP_LOGCASE(LogCaseCalc3DHeight);

    Unwrap::calc3DHeightGpu(pstCmd, pstRpy);

    //FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::merge3DHeight(const PR_MERGE_3D_HEIGHT_CMD *const pstCmd, PR_MERGE_3D_HEIGHT_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->vecMatHeight.size() < 2) {
        WriteLog("The vector of height size is less than 2.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->vecMatNanMask.size() < 2) {
        WriteLog("The vector of nan mask size is less than 2.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &mat : pstCmd->vecMatHeight) {
        if (mat.empty()) {
            WriteLog("The input height matrix is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    for (const auto &mat : pstCmd->vecMatNanMask) {
        if (mat.empty()) {
            WriteLog("The input nan mask is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }
    MARK_FUNCTION_START_TIME;

    Unwrap::merge3DHeight(pstCmd, pstRpy);

    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calcMerge4DlpHeight(const PR_CALC_MERGE_4_DLP_HEIGHT_CMD *const pstCmd, PR_CALC_MERGE_4_DLP_HEIGHT_RPY *const pstRpy) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    for (int dlp = 0; dlp < NUM_OF_DLP; ++dlp) {
        const auto& stCmd = pstCmd->arrCalcHeightCmd[dlp];
        if (stCmd.vecInputImgs.size() < 8) {
            WriteLog("The input image count is less than 8.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        for (const auto &mat : stCmd.vecInputImgs) {
            if (mat.empty()) {
                WriteLog("The input image is empty.");
                pstRpy->enStatus = VisionStatus::INVALID_PARAM;
                return pstRpy->enStatus;
            }
        }

        if (stCmd.matThickToThinK.empty()) {
            WriteLog("The matThickToThinK is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (stCmd.nRemoveJumpSpan < 0) {
            std::stringstream ss;
            ss << "The nRemoveJumpSpan " << stCmd.nRemoveJumpSpan << " is invalid.";
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (stCmd.nCompareRemoveJumpSpan < 0) {
            std::stringstream ss;
            ss << "The nCompareRemoveJumpSpan " << stCmd.nCompareRemoveJumpSpan << " is invalid.";
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (stCmd.bUseThinnestPattern) {
            if (stCmd.matThickToThinnestK.empty()) {
                WriteLog("matThickToThinnestK is empty.");
                pstRpy->enStatus = VisionStatus::INVALID_PARAM;
                return pstRpy->enStatus;
            }

            if (stCmd.vecInputImgs.size() != 3 * PR_GROUP_TEXTURE_IMG_COUNT) {
                WriteLog("Please input 12 images if using the thinnest pattern.");
                pstRpy->enStatus = VisionStatus::INVALID_PARAM;
                return pstRpy->enStatus;
            }
        }

        if (stCmd.fPhaseShift > 1.f || stCmd.fPhaseShift < -1.f) {
            std::stringstream ss;
            ss << "The fPhaseShift " << stCmd.fPhaseShift << " is invalid";
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (stCmd.fBaseRangeMin < 0.f || stCmd.fBaseRangeMax > 1.f || stCmd.fBaseRangeMin > stCmd.fBaseRangeMax) {
            std::stringstream ss;
            ss << "The fBaseRangeMin " << stCmd.fBaseRangeMin << " and fBaseRangeMax " << stCmd.fBaseRangeMax << " is invalid";
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;

    Unwrap::calcMerge4DlpHeight(pstCmd, pstRpy);

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calc3DHeightDiff(const PR_CALC_3D_HEIGHT_DIFF_CMD *const pstCmd, PR_CALC_3D_HEIGHT_DIFF_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matHeight.empty()) {
        WriteLog("The input height matrix is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matHeight, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (!pstCmd->matMask.empty()) {
        pstRpy->enStatus = _checkMask(pstCmd->matHeight, pstCmd->matMask, AT);
        if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;
    }

    if (pstCmd->vecRectBases.empty()) {
        WriteLog("The base vector is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (auto const &rect : pstCmd->vecRectBases) {
        pstRpy->enStatus = _checkInputROI(rect, pstCmd->matHeight, AT);
        if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;
    }

    if (1.f <= pstCmd->fEffectHRatioStart || pstCmd->fEffectHRatioStart < 0.f) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The EffectHRatioStart %.2f is invalid, it should between 0 ~ 1.", pstCmd->fEffectHRatioStart);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (1.f <= pstCmd->fEffectHRatioEnd || pstCmd->fEffectHRatioEnd < 0.f) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The EffectHRatioEnd %.2f is invalid, it should between 0 ~ 1.", pstCmd->fEffectHRatioEnd);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->fEffectHRatioEnd <= pstCmd->fEffectHRatioStart) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The EffectHRatioEnd %.2f should larger than EffectHRatioStart %.2f.",
            pstCmd->fEffectHRatioEnd, pstCmd->fEffectHRatioStart);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseCalc3DHeightDiff);

    Unwrap::calc3DHeightDiff(pstCmd, pstRpy);

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::insp3DSolder(const PR_INSP_3D_SOLDER_CMD *pstCmd, PR_INSP_3D_SOLDER_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    pstRpy->vecResults.clear();
    if (pstCmd->matHeight.empty()) {
        WriteLog("The input height matrix is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matColorImg.empty()) {
        WriteLog("The input color image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matColorImg.size() != pstCmd->matHeight.size()) {
        std::stringstream ss;
        ss << "The input color image size " << pstCmd->matColorImg.size() << " doesn't match with height matrix size " << pstCmd->matHeight.size();
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matColorImg.channels() < 3) {
        WriteLog("The input color image is not color image.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->fHeightVariationCoverage > 0.95f || pstCmd->fHeightVariationCoverage < 0.1f) {
        std::stringstream ss;
        ss << "The fHeightVariationCoverage " << pstCmd->fHeightVariationCoverage << " is invalid, it should between [0.1, 0.95]";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectDeviceROI, pstCmd->matHeight, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (pstCmd->vecRectCheckROIs.empty()) {
        WriteLog("The vector of check ROI is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &roiCheck : pstCmd->vecRectCheckROIs) {
        if (roiCheck.area() <= 10.f) {
            std::stringstream ss;
            ss << "Check ROI " << roiCheck << " is invalid";
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus; 
        }

        if (!CalcUtils::isRectInRect(roiCheck, pstCmd->rectDeviceROI)) {
            std::stringstream ss;
            ss << "Check ROI " << roiCheck << " is not inside device ROI " << pstCmd->rectDeviceROI << std::endl;
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseInsp3DSolder);

    std::vector<int> vecValue(3);
    vecValue[0] = ToInt32(pstCmd->scalarBaseColor[0]); vecValue[1] = ToInt32(pstCmd->scalarBaseColor[1]); vecValue[2] = ToInt32(pstCmd->scalarBaseColor[2]);
    cv::Mat matColorROI(pstCmd->matColorImg, pstCmd->rectDeviceROI);
    UInt32 nBasePointCount = 0;
    auto matBaseMask = _pickColor(vecValue, matColorROI, pstCmd->nBaseColorDiff, pstCmd->nBaseGrayDiff, nBasePointCount);

    // Remove the areas in between the two check ROIs to avoid the device is same color as backgroud
    if (pstCmd->vecRectCheckROIs.size() >= 2) {
        cv::Point topLft = pstCmd->vecRectCheckROIs[0].tl();
        cv::Point btmRgt = pstCmd->vecRectCheckROIs[1].br();

        // If the first ROI is on right or bottom
        if (topLft.x < btmRgt.x) {
            topLft = pstCmd->vecRectCheckROIs[1].tl();
            btmRgt = pstCmd->vecRectCheckROIs[0].br();
        }

        topLft.x -= pstCmd->rectDeviceROI.x;
        topLft.y -= pstCmd->rectDeviceROI.y;

        btmRgt.x -= pstCmd->rectDeviceROI.x;
        btmRgt.y -= pstCmd->rectDeviceROI.y;

        cv::Rect rectMidle(topLft, btmRgt);
        cv::Mat matOnDeviceROI(matBaseMask, rectMidle);
        matOnDeviceROI.setTo(0);
    }

    TimeLog::GetInstance()->addTimeLog("Pick color.", stopWatch.Span());

    if (!isAutoMode()) {
        pstRpy->matResultImg = pstCmd->matColorImg.clone();
        cv::rectangle(pstRpy->matResultImg, pstCmd->rectDeviceROI, CYAN_SCALAR, 1);
        for (const auto &roiCheck : pstCmd->vecRectCheckROIs)
            cv::rectangle(pstRpy->matResultImg, roiCheck, GREEN_SCALAR, 1);
    }

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Base Mask", matBaseMask);

    Unwrap::insp3DSolder(pstCmd, matBaseMask, pstRpy);

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calcDlpOffset(const PR_CALC_DLP_OFFSET_CMD *const pstCmd, PR_CALC_DLP_OFFSET_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matHeight1.empty() || pstCmd->matHeight2.empty() || pstCmd->matHeight3.empty() || pstCmd->matHeight4.empty()) {
        WriteLog("The input height matrix is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    cv::Mat matNan1 = CalcUtils::getNanMask(pstCmd->matHeight1);
    cv::Mat matNan2 = CalcUtils::getNanMask(pstCmd->matHeight2);
    cv::Mat matNan3 = CalcUtils::getNanMask(pstCmd->matHeight3);
    cv::Mat matNan4 = CalcUtils::getNanMask(pstCmd->matHeight4);
    cv::Mat matNanTotal = matNan1 | matNan2 | matNan3 | matNan4;
    cv::Mat matNonNan = 255 - matNanTotal;

    auto meanHeight1 = cv::mean(pstCmd->matHeight1, matNonNan)[0];
    auto meanHeight2 = cv::mean(pstCmd->matHeight2, matNonNan)[0];
    auto meanHeight3 = cv::mean(pstCmd->matHeight3, matNonNan)[0];
    auto meanHeight4 = cv::mean(pstCmd->matHeight4, matNonNan)[0];

    pstRpy->fOffset1 = ToFloat(meanHeight1 - meanHeight4);
    pstRpy->fOffset2 = ToFloat(meanHeight2 - meanHeight4);
    pstRpy->fOffset3 = ToFloat(meanHeight3 - meanHeight4);
    pstRpy->fOffset4 = ToFloat(meanHeight4 - meanHeight4);

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calcFrameValue(const PR_CALC_FRAME_VALUE_CMD *const pstCmd, PR_CALC_FRAME_VALUE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->vecVecRefFrameCenters.empty() || pstCmd->vecVecRefFrameCenters[0].empty()) {
        WriteLog("The input reference frame center vector is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->vecVecRefFrameValues.empty() || pstCmd->vecVecRefFrameValues[0].empty()) {
        WriteLog("The input reference frame value vector is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->vecVecRefFrameCenters.size() != pstCmd->vecVecRefFrameValues.size() || pstCmd->vecVecRefFrameCenters[0].size() != pstCmd->vecVecRefFrameValues[0].size()) {
        WriteLog("The size of input reference frame center and value vector is not match.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    const auto ROWS = pstCmd->vecVecRefFrameCenters.size();
    const auto COLS = pstCmd->vecVecRefFrameCenters[0].size();
    int row = 0, col = 0;
    if (1 == ROWS)
        row = 0;
    else {
        for (; row < ROWS; ++ row)
            if (pstCmd->ptTargetFrameCenter.y >= pstCmd->vecVecRefFrameCenters[row][0].y)
                break;
    }

    if (1 == COLS)
        col = 0;
    else {
        for (; col < COLS; ++ col)
            if (pstCmd->ptTargetFrameCenter.x <= pstCmd->vecVecRefFrameCenters[0][col].x)
                break;
    }

    if (0 == row) {
        if (0 == col)
            pstRpy->fResult = pstCmd->vecVecRefFrameValues[row][col];
        else if (COLS == col)
            pstRpy->fResult = pstCmd->vecVecRefFrameValues[row][col - 1];
        else {
            pstRpy->fResult = CalcUtils::linearInterpolate(
                pstCmd->vecVecRefFrameCenters[row][col - 1],
                pstCmd->vecVecRefFrameValues[row][col - 1],
                pstCmd->vecVecRefFrameCenters[row][col],
                pstCmd->vecVecRefFrameValues[row][col],
                pstCmd->ptTargetFrameCenter);
        }
    }else if (ROWS == row) {
        if (COLS == col)
            pstRpy->fResult = pstCmd->vecVecRefFrameValues[row - 1][col - 1];
        else if (0 == col)
            pstRpy->fResult = pstCmd->vecVecRefFrameValues[row - 1][col];
        else {
            pstRpy->fResult = CalcUtils::linearInterpolate(
                pstCmd->vecVecRefFrameCenters[row - 1][col - 1],
                pstCmd->vecVecRefFrameValues [row - 1][col -1],
                pstCmd->vecVecRefFrameCenters[row - 1][col],
                pstCmd->vecVecRefFrameValues [row - 1][col],
                pstCmd->ptTargetFrameCenter);
        }
    }else if (0 == col) {
        if (ROWS == row)
            pstRpy->fResult = pstCmd->vecVecRefFrameValues[row - 1][col];
        else {
            pstRpy->fResult = CalcUtils::linearInterpolate(
                pstCmd->vecVecRefFrameCenters[row - 1][col],
                pstCmd->vecVecRefFrameValues [row - 1][col],
                pstCmd->vecVecRefFrameCenters[row][col],
                pstCmd->vecVecRefFrameValues [row][col],
                pstCmd->ptTargetFrameCenter);
        }
    }else if (COLS == col) {
        if (0 == row)
            pstRpy->fResult = pstCmd->vecVecRefFrameValues[0][col - 1];
        else if (ROWS == row)
            pstRpy->fResult = pstCmd->vecVecRefFrameValues[row - 1][col - 1];
        else {
            pstRpy->fResult = CalcUtils::linearInterpolate(
                pstCmd->vecVecRefFrameCenters[row - 1][col - 1],
                pstCmd->vecVecRefFrameValues [row - 1][col - 1],
                pstCmd->vecVecRefFrameCenters[row][col - 1],
                pstCmd->vecVecRefFrameValues [row][col - 1],
                pstCmd->ptTargetFrameCenter);
        }
    }else {
        VectorOfPoint2f vecPoints;
        VectorOfFloat vecValues;
        vecPoints.push_back(pstCmd->vecVecRefFrameCenters[row - 1][col - 1]);
        vecValues.push_back(pstCmd->vecVecRefFrameValues[row - 1][col - 1]);
        vecPoints.push_back(pstCmd->vecVecRefFrameCenters[row][col - 1]);
        vecValues.push_back(pstCmd->vecVecRefFrameValues[row][col - 1]);
        vecPoints.push_back(pstCmd->vecVecRefFrameCenters[row][col]);
        vecValues.push_back(pstCmd->vecVecRefFrameValues[row][col]);
        vecPoints.push_back(pstCmd->vecVecRefFrameCenters[row - 1][col]);
        vecValues.push_back(pstCmd->vecVecRefFrameValues[row - 1][col]);
        pstRpy->fResult = CalcUtils::bilinearInterpolate(vecPoints, vecValues, pstCmd->ptTargetFrameCenter);
    }

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calcCameraMTF(const PR_CALC_CAMERA_MTF_CMD *const pstCmd, PR_CALC_CAMERA_MTF_RPY *const pstRpy, bool bReplay/* = false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectVBigPatternROI, pstCmd->matInputImg, AT);
    if (pstRpy->enStatus != VisionStatus::OK) return pstRpy->enStatus;
    pstRpy->enStatus = _checkInputROI(pstCmd->rectHBigPatternROI, pstCmd->matInputImg, AT);
    if (pstRpy->enStatus != VisionStatus::OK) return pstRpy->enStatus;
    pstRpy->enStatus = _checkInputROI(pstCmd->rectVSmallPatternROI, pstCmd->matInputImg, AT);
    if (pstRpy->enStatus != VisionStatus::OK) return pstRpy->enStatus;
    pstRpy->enStatus = _checkInputROI(pstCmd->rectHSmallPatternROI, pstCmd->matInputImg, AT);
    if (pstRpy->enStatus != VisionStatus::OK) return pstRpy->enStatus;

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseCalcCameraMTF);

    cv::Mat matGray;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(pstCmd->matInputImg, matGray, CV_BGR2GRAY);
    else
        matGray = pstCmd->matInputImg;

    VectorOfMat vecMat;
    vecMat.push_back(cv::Mat(matGray, pstCmd->rectVBigPatternROI));
    vecMat.push_back(cv::Mat(matGray, pstCmd->rectHBigPatternROI));
    vecMat.push_back(cv::Mat(matGray, pstCmd->rectVSmallPatternROI));
    vecMat.push_back(cv::Mat(matGray, pstCmd->rectHSmallPatternROI));
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("T1", vecMat[0]); showImage("T2", vecMat[1]); showImage("T3", vecMat[2]); showImage("T4", vecMat[3]);
    }

    const int MIN_PATTERN_SIZE = 5;
    for (auto &mat : vecMat) {
        cv::Mat matCanny;
        cv::Canny(mat, matCanny, 150, 50);
        if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
            showImage("Canny Result", matCanny);
        VectorOfPoint vecPoints;
        cv::findNonZero(matCanny, vecPoints);
        if (vecPoints.empty()) {
            WriteLog("Can not find MTF pattern in the given ROI.");
            pstRpy->enStatus = VisionStatus::CAN_NOT_FIND_MTF_PATTERN;
            return pstRpy->enStatus;
        }

        int xMin, xMax, yMin, yMax;
        CalcUtils::findMinMaxCoord(vecPoints, xMin, xMax, yMin, yMax);
        int margin = ToInt32(0.05f * mat.rows);
        xMin += margin; xMax -= margin;
        yMin += margin; yMax -= margin;
        if ((xMax - xMin) < MIN_PATTERN_SIZE || (yMax - yMin) < MIN_PATTERN_SIZE) {
            WriteLog("The MTF pattern is too small.");
            pstRpy->enStatus = VisionStatus::MTF_PATTERN_TOO_SMALL;
            return pstRpy->enStatus;
        }
        mat = cv::Mat(mat, cv::Range(yMin, yMax), cv::Range(xMin, xMax));
    }
    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("Trimmed T1", vecMat[0]); showImage("Trimmed T2", vecMat[1]);
        showImage("Trimmed T3", vecMat[2]); showImage("Trimmed T4", vecMat[3]);
    }

    const int DATA_LEN = 5;
    pstRpy->vecBigPatternAbsMtfV.clear();
    pstRpy->vecBigPatternAbsMtfH.clear();
    pstRpy->vecSmallPatternAbsMtfV.clear();
    pstRpy->vecSmallPatternAbsMtfH.clear();
    for (int row = 0; row < vecMat[0].rows - 1; ++ row) {
        cv::Mat matCol(vecMat[0], cv::Rect(0, row, vecMat[0].cols, 1));
        cv::Mat matSorted;
        cv::sort(matCol, matSorted, cv::SortFlags::SORT_EVERY_ROW + cv::SortFlags::SORT_ASCENDING);
        auto vecVecCol = CalcUtils::matToVector<uchar>(matCol);
        auto vecVecSorted = CalcUtils::matToVector<uchar>(matSorted);
        double dMaxAvg = cv::mean(cv::Mat(matSorted, cv::Rect(matSorted.cols - DATA_LEN, 0, DATA_LEN, 1)))[0];
        double dMinAvg = cv::mean(cv::Mat(matSorted, cv::Rect(0, 0, DATA_LEN, 1)))[0];
        pstRpy->vecBigPatternAbsMtfV.push_back(ToFloat(dMaxAvg - dMinAvg) / PR_MAX_GRAY_LEVEL);
    }

    for (int col = 0; col < vecMat[1].cols - 1; ++ col) {
        cv::Mat matRow(vecMat[1], cv::Rect(col, 0, 1, vecMat[1].rows));
        cv::Mat matSorted;
        cv::sort(matRow, matSorted, cv::SortFlags::SORT_EVERY_COLUMN + cv::SortFlags::SORT_ASCENDING);

        double dMaxAvg = cv::mean(cv::Mat(matSorted, cv::Rect(0, matSorted.rows - DATA_LEN, 1, DATA_LEN)))[0];
        double dMinAvg = cv::mean(cv::Mat(matSorted, cv::Rect(0, 0, 1, DATA_LEN)))[0];
        pstRpy->vecBigPatternAbsMtfH.push_back(ToFloat(dMaxAvg - dMinAvg) / PR_MAX_GRAY_LEVEL);
    }

    _calcMtfByFFT(vecMat[2], false, pstRpy->vecSmallPatternAbsMtfV);
    _calcMtfByFFT(vecMat[3], true, pstRpy->vecSmallPatternAbsMtfH);

    pstRpy->fBigPatternAbsMtfV = CalcUtils::mean(pstRpy->vecBigPatternAbsMtfV);
    pstRpy->fBigPatternAbsMtfH = CalcUtils::mean(pstRpy->vecBigPatternAbsMtfH);
    pstRpy->fSmallPatternAbsMtfV = CalcUtils::mean(pstRpy->vecSmallPatternAbsMtfV);
    pstRpy->fSmallPatternAbsMtfH = CalcUtils::mean(pstRpy->vecSmallPatternAbsMtfH);
    pstRpy->fSmallPatternRelMtfV = pstRpy->fSmallPatternAbsMtfV / pstRpy->fBigPatternAbsMtfV;
    pstRpy->fSmallPatternRelMtfH = pstRpy->fSmallPatternAbsMtfH / pstRpy->fBigPatternAbsMtfH;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_calcMtfByFFT(const cv::Mat &matInput, bool bHorizontalStrip, VectorOfFloat &vecMtf) {
    cv::Mat matDouble;
    matInput.convertTo(matDouble, CV_64FC1);
    if (bHorizontalStrip)
        cv::transpose(matDouble, matDouble);
    auto vecVecDouble = CalcUtils::matToVector<double>(matDouble);
    const int ROWS = matDouble.rows;
    const int COLS = matDouble.cols;
    const float Nf = 32;
    float fInterval = 1.f / (Nf / (COLS - 1));
    cv::Mat matRowDouble(matDouble, cv::Rect(0, ROWS / 2, COLS, 1));
    auto vecRowDouble = CalcUtils::matToVector<double>(matRowDouble)[0];
    VectorOfDouble vecX(vecRowDouble.size());
    std::iota(vecX.begin(), vecX.end(), 1);
    VectorOfDouble vecXq;
    vecXq.reserve(static_cast<size_t>(Nf));
    for (double xq = 1.; xq < COLS; xq += fInterval)
        vecXq.push_back(xq);
    auto vecVq = CalcUtils::interp1(vecX, vecRowDouble, vecXq, false);
    cv::Mat matVq(vecVq);

    cv::Mat matDft, matAbsDft;
    cv::dft(vecVq, matDft, cv::DftFlags::DCT_ROWS);
    matAbsDft = cv::abs(matDft);
    auto vecVecAbsDft = CalcUtils::matToVector<double>(matAbsDft);
    matAbsDft = cv::Mat(matAbsDft, cv::Rect(1, 0, matAbsDft.cols / 2, 1));
    cv::Point ptMin, ptMax;
    double dMin = 0., dMax = 0.;
    cv::minMaxLoc(matAbsDft, &dMin, &dMax, &ptMin, &ptMax);
    int nMaxIdx = ptMax.x + 1;
    float f00 = (nMaxIdx) / fInterval / Nf;
    float f01 = (nMaxIdx - 1) / fInterval / Nf;
    float f02 = (nMaxIdx + 1) / fInterval / Nf;
    float f11 = (f00 + f01) / 2.f;
    float f12 = (f00 + f02) / 2.f;
    cv::Mat matTt1 = CalcUtils::intervals<double>(1.f, fInterval, ToFloat(COLS));
    matTt1 = matTt1.reshape(1, 1);
    float nff = 20.f;
    VectorOfDouble vecA;
    vecA.reserve(ToInt32(nff));
    for (float f1 = f11; f1 <= f12; f1 += (f12 - f11) / 20) {
        cv::Mat matXX1 = CalcUtils::sin<double>(matTt1 * 2 * CV_PI * f1);
        matXX1.push_back(CalcUtils::cos<double>(matTt1 * 2 * CV_PI * f1));
        matXX1.push_back(cv::Mat(cv::Mat::ones(1, matXX1.cols, CV_64FC1)));
        cv::transpose(matXX1, matXX1);
        cv::Mat matK;
        cv::solve(matXX1, matVq, matK, cv::DecompTypes::DECOMP_SVD);
        double A = 2.0 * sqrt(matK.at<double>(0) * matK.at<double>(0) + matK.at<double>(1) * matK.at<double>(1));
        vecA.push_back(A);
    }
    auto maxElement = std::max_element(vecA.begin(), vecA.end());
    int nIndex = ToInt32(std::distance(vecA.begin(), maxElement)) + 1;

    float fFinal = f11 + (nIndex - 1.f) * (f12 - f11) / nff;

    cv::Mat matXX1 = CalcUtils::sin<double>(matTt1 * 2 * CV_PI * fFinal);
    matXX1.push_back(CalcUtils::cos<double>(matTt1 * 2 * CV_PI * fFinal));
    matXX1.push_back(cv::Mat(cv::Mat::ones(1, matXX1.cols, CV_64FC1)));
    cv::transpose(matXX1, matXX1);

    for (int row = 0; row < ROWS; ++ row) {
        cv::Mat matRow = cv::Mat(matDouble, cv::Rect(0, row, COLS, 1)).clone();
        auto vecRowLocal = CalcUtils::matToVector<double>(matRow)[0];
        auto vecVqLocal = CalcUtils::interp1(vecX, vecRowLocal, vecXq, true);
        cv::Mat matRowInterp(vecVqLocal);
        cv::Mat matK;
        cv::solve(matXX1, matRowInterp, matK, cv::DecompTypes::DECOMP_SVD);
        float A = ToFloat(2. * sqrt(matK.at<double>(0) * matK.at<double>(0) + matK.at<double>(1) * matK.at<double>(1)));
        float fMtf = ToFloat(A / matK.at<double>(2) / 2.);
        vecMtf.push_back(fMtf);
    }
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::calcMTF(const PR_CALC_MTF_CMD *const pstCmd, PR_CALC_MTF_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecInputImgs.empty() || pstCmd->vecInputImgs.size() % PR_GROUP_TEXTURE_IMG_COUNT != 0) {
        WriteLog("The input image count is multiple of 4.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &mat : pstCmd->vecInputImgs) {
        if (mat.empty()) {
            WriteLog("The input image is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstCmd->fMagnitudeOfDLP <= 0 || pstCmd->fMagnitudeOfDLP > PR_MAX_GRAY_LEVEL) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The MagnitudeOfDLP %f is not in range [1, 255].", pstCmd->fMagnitudeOfDLP);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    Unwrap::calcMTF(pstCmd, pstRpy);
    MARK_FUNCTION_END_TIME;

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calcPD(const PR_CALC_PD_CMD *const pstCmd, PR_CALC_PD_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecInputImgs.empty() || pstCmd->vecInputImgs.size() % PR_GROUP_TEXTURE_IMG_COUNT != 0) {
        WriteLog("The input image count is multiple of 4.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto &mat : pstCmd->vecInputImgs) {
        if (mat.empty()) {
            WriteLog("The input image is empty.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstCmd->fMagnitudeOfDLP <= 0 || pstCmd->fMagnitudeOfDLP > PR_MAX_GRAY_LEVEL) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The MagnitudeOfDLP %f is not in range [1, 255].", pstCmd->fMagnitudeOfDLP);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    Unwrap::calcPD(pstCmd, pstRpy);
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::parallelLineDist(const PR_PARALLEL_LINE_DIST_CMD *const pstCmd, PR_PARALLEL_LINE_DIST_RPY *const pstRpy) {
    bool bReverseFit = fabs(pstCmd->line1.pt2.y - pstCmd->line1.pt1.y) > (pstCmd->line1.pt2.x - pstCmd->line1.pt1.x);
    float fLineSlope1 = -10.f, fLineSlope2 = 10.f;
    float fIntercept1 = 0.f, fIntercept2 = 0.f;
    if (bReverseFit) {
        fLineSlope1 = (pstCmd->line1.pt2.x - pstCmd->line1.pt1.x) / (pstCmd->line1.pt2.y - pstCmd->line1.pt1.y);
        fIntercept1 = pstCmd->line1.pt1.x - fLineSlope1 * pstCmd->line1.pt1.y;
        fLineSlope2 = (pstCmd->line2.pt2.x - pstCmd->line2.pt1.x) / (pstCmd->line2.pt2.y - pstCmd->line2.pt1.y);
        fIntercept2 = pstCmd->line2.pt1.x - fLineSlope1 * pstCmd->line2.pt1.y;
    }
    else {
        fLineSlope1 = (pstCmd->line1.pt2.y - pstCmd->line1.pt1.y) / (pstCmd->line1.pt2.x - pstCmd->line1.pt1.x);
        fIntercept1 = pstCmd->line1.pt1.y - fLineSlope1 * pstCmd->line1.pt1.x;
        fLineSlope2 = (pstCmd->line2.pt2.y - pstCmd->line2.pt1.y) / (pstCmd->line2.pt2.x - pstCmd->line2.pt1.x);
        fIntercept2 = pstCmd->line2.pt1.y - fLineSlope1 * pstCmd->line2.pt1.x;
    }

    if (fabs(fLineSlope1 - fLineSlope2) > PARALLEL_LINE_SLOPE_DIFF_LMT) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The input two lines are not parallel. Line 1 slope %f, line 2 slope %f", fLineSlope1, fLineSlope2);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->fDistance = CalcUtils::ptDisToLine(pstCmd->line2.pt1, bReverseFit, fLineSlope1, fIntercept1);
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::measureDist(const PR_MEASURE_DIST_CMD* const pstCmd, PR_MEASURE_DIST_RPY* const pstRpy) {
    switch (pstCmd->enMeasureMode)
    {
    case PR_MEASURE_DIST_CMD::MODE::DIRECT_LINE:
        pstRpy->fDistance = CalcUtils::distanceOf2Point(pstCmd->ptStart, pstCmd->ptEnd);
        break;
    case PR_MEASURE_DIST_CMD::MODE::X_DIRECTION:
        if (fabs(pstCmd->fFiducialSlope) < 0.0001f) {
            pstRpy->fDistance = fabs(pstCmd->ptStart.x - pstCmd->ptEnd.x);
            pstRpy->ptCross = cv::Point2f(pstCmd->ptStart.x, pstCmd->ptEnd.y);
            pstRpy->bMeasureWithStart = false;
        }else {
            pstRpy->ptCross = CalcUtils::calcDistPoint(pstCmd->ptStart, pstCmd->ptEnd, pstCmd->fFiducialSlope);
            float tanWithStart = 0.f, tanWithEnd = 0.f;
            if (fabs(pstCmd->ptStart.x - pstRpy->ptCross.x) < 0.0001f)
                tanWithStart = 100000.f;
            else
                tanWithStart = (pstCmd->ptStart.y - pstRpy->ptCross.y) / (pstCmd->ptStart.x - pstRpy->ptCross.x);

            if (fabs(pstCmd->ptEnd.x - pstRpy->ptCross.x) < 0.0001f)
                tanWithEnd = 100000.f;
            else
                tanWithEnd = (pstCmd->ptEnd.y - pstRpy->ptCross.y) / (pstCmd->ptEnd.x - pstRpy->ptCross.x);

            // Measure with the one follow the slope direction
            if (fabs(fabs(tanWithStart) - fabs(pstCmd->fFiducialSlope)) <= fabs(fabs(tanWithEnd) - fabs(pstCmd->fFiducialSlope))) {
                pstRpy->fDistance = CalcUtils::distanceOf2Point(pstRpy->ptCross, pstCmd->ptStart);
                pstRpy->bMeasureWithStart = true;
            }else {
                pstRpy->fDistance = CalcUtils::distanceOf2Point(pstRpy->ptCross, pstCmd->ptEnd);
                pstRpy->bMeasureWithStart = false;
            }
        }
        break;
    case PR_MEASURE_DIST_CMD::MODE::Y_DIRECTION:
        if (fabs(pstCmd->fFiducialSlope) < 0.0001f) {
            pstRpy->fDistance = fabs(pstCmd->ptStart.y - pstCmd->ptEnd.y);
            pstRpy->ptCross = cv::Point2f(pstCmd->ptStart.x, pstCmd->ptEnd.y);
            pstRpy->bMeasureWithStart = true;
        }
        else {
            pstRpy->ptCross = CalcUtils::calcDistPoint(pstCmd->ptStart, pstCmd->ptEnd, pstCmd->fFiducialSlope);
            float tanWithStart = 0.f, tanWithEnd = 0.f;
            if (fabs(pstCmd->ptStart.x - pstRpy->ptCross.x) < 0.0001f)
                tanWithStart = 100000.f;
            else
                tanWithStart = (pstCmd->ptStart.y - pstRpy->ptCross.y) / (pstCmd->ptStart.x - pstRpy->ptCross.x);

            if (fabs(pstCmd->ptEnd.x - pstRpy->ptCross.x) < 0.0001f)
                tanWithEnd = 100000.f;
            else
                tanWithEnd = (pstCmd->ptEnd.y - pstRpy->ptCross.y) / (pstCmd->ptEnd.x - pstRpy->ptCross.x);

            // Measure with the perpendicular to the slope direction
            float fPerpendicularSlope = -1.f / pstCmd->fFiducialSlope;
            if (fabs(fabs(tanWithStart) - fabs(fPerpendicularSlope)) <= fabs(fabs(tanWithEnd) - fabs(fPerpendicularSlope))) {
                pstRpy->fDistance = CalcUtils::distanceOf2Point(pstRpy->ptCross, pstCmd->ptStart);
                pstRpy->bMeasureWithStart = true;
            }else {
                pstRpy->fDistance = CalcUtils::distanceOf2Point(pstRpy->ptCross, pstCmd->ptEnd);
                pstRpy->bMeasureWithStart = false;
            }

            pstRpy->fDistance = CalcUtils::distanceOf2Point(pstRpy->ptCross, pstCmd->ptEnd);
        }
        break;
    default:
        break;
    }
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::crossSectionArea(const PR_CROSS_SECTION_AREA_CMD *const pstCmd, PR_CROSS_SECTION_AREA_RPY *const pstRpy) {
    auto contour = pstCmd->vecContourPoints;
    if (! pstCmd->bClosed) {
        float nMinX = std::numeric_limits<float>::max();
        float nMaxX = std::numeric_limits<float>::min();
        for (const auto &point : contour) {
            if (point.x > nMaxX)
                nMaxX = point.x;
            if (point.x < nMinX)
                nMinX = point.x;
        }
        contour.emplace_back(nMaxX, 0.f);
        contour.emplace_back(nMinX, 0.f);
    }

    pstRpy->fArea = ToFloat(cv::contourArea(contour));
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::combineImg(const PR_COMBINE_IMG_CMD *const pstCmd, PR_COMBINE_IMG_RPY *const pstRpy, bool bReplay /*= false*/) {
    pstRpy->vecResultImages.clear();
    if (pstCmd->nCountOfFrameX <= 0 || pstCmd->nCountOfFrameY <= 0 || pstCmd->nCountOfImgPerFrame <= 0) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The CountOfFrameX %d or CountOfFrameY %d or CountOfImgPerFrame %d is invalid.",
            pstCmd->nCountOfFrameX, pstCmd->nCountOfFrameY, pstCmd->nCountOfImgPerFrame);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    int nCountOfImgPerRowMin = (pstCmd->nCountOfFrameX - 1) * pstCmd->nCountOfImgPerFrame;
    int nCountOfImgPerRowMax = (pstCmd->nCountOfFrameX) * pstCmd->nCountOfImgPerFrame;
    if (pstCmd->nCountOfImgPerRow <= nCountOfImgPerRowMin || pstCmd->nCountOfImgPerRow > nCountOfImgPerRowMax) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The nCountOfImgPerRow %d is invalid, it should in range (%d, %d].",
            pstCmd->nCountOfImgPerRow, nCountOfImgPerRowMin, nCountOfImgPerRowMax);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->vecInputImages.size() != pstCmd->nCountOfFrameY * pstCmd->nCountOfImgPerRow) {
        char chArrMsg[1000];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The input image count %d is invalid, it should be %d.",
            pstCmd->vecInputImages.size(), pstCmd->nCountOfFrameY * pstCmd->nCountOfImgPerRow);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    auto szImage = pstCmd->vecInputImages[0].size();
    int nResultPixCols = pstCmd->nCountOfFrameX * szImage.width  - pstCmd->nOverlapX * (pstCmd->nCountOfFrameX - 1);
    int nResultPixRows = pstCmd->nCountOfFrameY * szImage.height - pstCmd->nOverlapY * (pstCmd->nCountOfFrameY - 1);

    for (int imgNo = 0; imgNo < pstCmd->nCountOfImgPerFrame; ++ imgNo) {
        cv::Mat matResult = cv::Mat::zeros(nResultPixRows, nResultPixCols, pstCmd->vecInputImages[0].type());

        for (int nRow = 0; nRow < pstCmd->nCountOfFrameY; ++ nRow)
        for (int nCol = 0; nCol < pstCmd->nCountOfFrameX; ++ nCol) {
            if ((nCol * pstCmd->nCountOfImgPerFrame + imgNo) >= pstCmd->nCountOfImgPerRow)
                continue;

            int nImageIndex = nCol * pstCmd->nCountOfImgPerFrame + nRow * pstCmd->nCountOfImgPerRow + imgNo;
            cv::Mat mat = pstCmd->vecInputImages[nImageIndex];

            if (nCol > 0) {
                auto autoPreviousOverlapROIX = nCol * (szImage.width - pstCmd->nOverlapX);
                if (PR_SCAN_IMAGE_DIR::RIGHT_TO_LEFT == pstCmd->enScanDir)
                    autoPreviousOverlapROIX = nResultPixCols - (szImage.width * nCol - (nCol - 1) * pstCmd->nOverlapX);

                auto autoPreviousOverlapROIY = nRow * (szImage.height - pstCmd->nOverlapY);
                cv::Mat matPreviousFrameOverlap(matResult, cv::Rect(autoPreviousOverlapROIX, autoPreviousOverlapROIY, pstCmd->nOverlapX, szImage.height));
                cv::Mat matCurrentFrameOvelap(mat, cv::Rect(szImage.width - pstCmd->nOverlapX, 0, pstCmd->nOverlapX, szImage.height));
                matPreviousFrameOverlap = (matPreviousFrameOverlap + matCurrentFrameOvelap) / 2;
            }

            auto autoROIX = nCol * (szImage.width - pstCmd->nOverlapX);
            if (PR_SCAN_IMAGE_DIR::RIGHT_TO_LEFT == pstCmd->enScanDir)
                autoROIX = nResultPixCols - (szImage.width * (nCol + 1) - nCol * pstCmd->nOverlapX);
            auto autoROIY = nRow * (szImage.height - pstCmd->nOverlapY);
            cv::Mat matROI(matResult, cv::Rect(autoROIX, autoROIY, szImage.width, szImage.height));
            mat.copyTo(matROI);
        }
        pstRpy->vecResultImages.push_back(matResult);
    }

    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::combineImgNew(const PR_COMBINE_IMG_NEW_CMD *const pstCmd, PR_COMBINE_IMG_NEW_RPY *const pstRpy, bool bReplay/* = false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    if (pstCmd->vecInputImages.empty()) {
        WriteLog("Input image vector is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (const auto& matImage : pstCmd->vecInputImages) {
        if (matImage.empty()) {
            WriteLog("Input image matrix is empty");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    if (pstCmd->vecVecFrameCtr.empty()) {
        WriteLog("Input image frame center is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    const auto TOTAL_FRAME_NUMBER = pstCmd->vecVecFrameCtr.size() * pstCmd->vecVecFrameCtr[0].size();
    if (pstCmd->vecInputImages.size() < TOTAL_FRAME_NUMBER) {
        std::stringstream ss;
        ss << "Input image size " << pstCmd->vecInputImages.size() << " less than total frame number " << TOTAL_FRAME_NUMBER;
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    auto topLftFrameCtr = pstCmd->vecVecFrameCtr.front().front();
    auto btnRgtFrameCtr = pstCmd->vecVecFrameCtr.back().back();
    const int ORIGIN_IMAGE_WIDTH   = pstCmd->vecInputImages[0].cols;
    const int ORIGIN_IMAGE_HEIGHT  = pstCmd->vecInputImages[0].rows;
    const int CUTTED_IMAGE_WIDTH  = ORIGIN_IMAGE_WIDTH  - 2 * pstCmd->nCutBorderPixel;
    const int CUTTED_IMAGE_HEIGHT = ORIGIN_IMAGE_HEIGHT - 2 * pstCmd->nCutBorderPixel;

    if (pstCmd->nCutBorderPixel < 0 || pstCmd->nCutBorderPixel > ORIGIN_IMAGE_WIDTH / 4) {
        std::stringstream ss;
        ss << "Cut image border pixel " << pstCmd->nCutBorderPixel << " is invalid";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (btnRgtFrameCtr.x < topLftFrameCtr.x) {
        std::stringstream ss;
        ss << "Right side frame center " << btnRgtFrameCtr.x << " is smaller than left side frame center " << topLftFrameCtr.x;
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (btnRgtFrameCtr.y < topLftFrameCtr.y) {
        std::stringstream ss;
        ss << "Bottom side frame center " << btnRgtFrameCtr.y << " is smaller than top side frame center " << topLftFrameCtr.y;
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    int TOTAL_WIDTH  = btnRgtFrameCtr.x - topLftFrameCtr.x + ORIGIN_IMAGE_WIDTH;
    int TOTAL_HEIGHT = btnRgtFrameCtr.y - topLftFrameCtr.y + ORIGIN_IMAGE_HEIGHT;
    pstRpy->matResultImage = cv::Mat::zeros(TOTAL_HEIGHT, TOTAL_WIDTH, pstCmd->vecInputImages[0].type());
    int index = 0;
    for (int row = 0; row < pstCmd->vecVecFrameCtr.size(); ++ row)
    for (int col = 0; col < pstCmd->vecVecFrameCtr[0].size(); ++ col) {
        if (pstCmd->vecInputImages[index].empty()) {
            std::stringstream ss;
            ss << "Input image at index " << index << " is empty";
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        auto ptFrameCtr = pstCmd->vecVecFrameCtr[row][col];
        cv::Rect rectDstROI(ptFrameCtr.x - CUTTED_IMAGE_WIDTH / 2, ptFrameCtr.y - CUTTED_IMAGE_HEIGHT / 2, CUTTED_IMAGE_WIDTH, CUTTED_IMAGE_HEIGHT);

        if (rectDstROI.x < 0) {
            rectDstROI = CalcUtils::resizeRect(rectDstROI, cv::Size(rectDstROI.width + 2 * rectDstROI.x, rectDstROI.height));
        }

        if (rectDstROI.y < 0) {
            rectDstROI = CalcUtils::resizeRect(rectDstROI, cv::Size(rectDstROI.width, rectDstROI.height + 2 * rectDstROI.y));
        }

        if ((rectDstROI.x + rectDstROI.width) > TOTAL_WIDTH) {
            int reduceSizeOneSide = (rectDstROI.x + rectDstROI.width) - TOTAL_WIDTH;
            rectDstROI = CalcUtils::resizeRect(rectDstROI, cv::Size(rectDstROI.width - 2 * reduceSizeOneSide, rectDstROI.height));
        }

        if ((rectDstROI.y + rectDstROI.height) > TOTAL_HEIGHT) {
            int reduceSizeOneSide = (rectDstROI.y + rectDstROI.height) - TOTAL_HEIGHT;
            rectDstROI = CalcUtils::resizeRect(rectDstROI, cv::Size(rectDstROI.width, rectDstROI.height - 2 * reduceSizeOneSide));
        }

        //rectDstROI = cv::Rect(ptFrameCtr.x - rectDstROI.width / 2, ptFrameCtr.y - rectDstROI.height / 2, rectDstROI.width, rectDstROI.height);
        cv::Rect rectSrcROI(ORIGIN_IMAGE_WIDTH / 2 - rectDstROI.width / 2, ORIGIN_IMAGE_HEIGHT / 2 - rectDstROI.height / 2, rectDstROI.width, rectDstROI.height);
        cv::Mat matSrc = cv::Mat(pstCmd->vecInputImages[index], rectSrcROI);

        cv::Mat matDstROI(pstRpy->matResultImage, rectDstROI);
        matSrc.copyTo(matDstROI);
        ++index;
    }

    if (pstCmd->bDrawFrame && pstRpy->matResultImage.type() == CV_8UC3) {
        for (int row = 0; row < pstCmd->vecVecFrameCtr.size(); ++ row)
        for (int col = 0; col < pstCmd->vecVecFrameCtr[0].size(); ++ col) {
            auto ptFrameCtr = pstCmd->vecVecFrameCtr[row][col];
            cv::Rect rectDstROI(ptFrameCtr.x - CUTTED_IMAGE_WIDTH / 2, ptFrameCtr.y - CUTTED_IMAGE_HEIGHT / 2, CUTTED_IMAGE_WIDTH, CUTTED_IMAGE_HEIGHT);
            cv::rectangle(pstRpy->matResultImage, rectDstROI, CYAN_SCALAR, 1);
        }
    }

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::threshold(const PR_THRESHOLD_CMD *const pstCmd, PR_THRESHOLD_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    cv::Mat matGray;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(pstCmd->matInputImg, matGray, CV_BGR2GRAY);
    else
        matGray = pstCmd->matInputImg;

    if (pstCmd->bDoubleThreshold) {
        cv::Mat matRange1, matRange2;
        matRange1 = matGray >= pstCmd->nThreshold1;
        matRange2 = matGray <= pstCmd->nThreshold2;
        pstRpy->matResultImg = matRange1 & matRange2;
    }else {
        cv::threshold(matGray, pstRpy->matResultImg, pstCmd->nThreshold1, PR_MAX_GRAY_LEVEL, cv::ThresholdTypes::THRESH_BINARY);
    }

    if (pstCmd->bInverseResult)
        pstRpy->matResultImg = PR_MAX_GRAY_LEVEL - pstRpy->matResultImg;

    MARK_FUNCTION_END_TIME;

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::heightToGray(const PR_HEIGHT_TO_GRAY_CMD *const pstCmd, PR_HEIGHT_TO_GRAY_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matHeight.empty()) {
        WriteLog("Input height mat is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    double dMinValue = 0, dMaxValue = 0;
    cv::Mat matMask = (pstCmd->matHeight == pstCmd->matHeight);
    cv::minMaxIdx(pstCmd->matHeight, &dMinValue, &dMaxValue, 0, 0, matMask);
    if (dMaxValue - dMinValue < 0.0001f) {
        WriteLog("The input image is all zero.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    
    cv::Mat matNewHeight = pstCmd->matHeight - dMinValue;

    float dRatio = 255.f / ToFloat(dMaxValue - dMinValue);
    matNewHeight = matNewHeight * dRatio;

    matNewHeight.convertTo(pstRpy->matGray, CV_8UC1);

    MARK_FUNCTION_END_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::lrnOcv(const PR_LRN_OCV_CMD *const pstCmd, PR_LRN_OCV_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("The input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (pstCmd->nCharCount > 50) {
        char chArrMsg[100];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The input char count %d is bigger than limit 50.", pstCmd->nCharCount);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = VisionStatus::OK;

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseLrnOcv);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI), matGray;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI;

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Original learn ROI image", matGray);

    _rotateOcvImage(pstCmd->enDirection, matGray);

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Rotated learn ROI image", matGray);

    std::vector<int> vecIndex1, vecIndex2;
    pstRpy->enStatus = _imageSplitByMaxNew(matGray, pstCmd->nCharCount, vecIndex1, vecIndex2);
    if (VisionStatus::OK != pstRpy->enStatus) {
        FINISH_LOGCASE_EX;
        return pstRpy->enStatus;
    }

    const int nMargin = 2;
    auto len = std::min(vecIndex1.size(), vecIndex2.size());
    VectorOfRect vecCharRects;
    for (int i = 0; i < len; ++ i) {
        if (vecIndex2[i] - vecIndex1[i] <= 0) {
            pstRpy->enStatus = VisionStatus::FAILED_TO_SPLIT_IMAGE;
            FINISH_LOGCASE_EX;
            return pstRpy->enStatus;
        }
        cv::Rect rectSubRegion(vecIndex1[i], 0, vecIndex2[i] - vecIndex1[i], matGray.rows);
        cv::Mat matSubRegion(matGray, rectSubRegion), matSubRegionT;
        cv::transpose(matSubRegion, matSubRegionT);
        std::vector<int> vecIndex3, vecIndex4;
        pstRpy->enStatus = _imageSplitByMax(matSubRegionT, 1, vecIndex3, vecIndex4);
        if (VisionStatus::OK != pstRpy->enStatus) {
            FINISH_LOGCASE_EX;
            return pstRpy->enStatus;
        }
        if (vecIndex3.empty() || vecIndex4.empty() || vecIndex4[0] - vecIndex3[0] <= 0) {
            pstRpy->enStatus = VisionStatus::FAILED_TO_SPLIT_IMAGE;
            FINISH_LOGCASE_EX;
            return pstRpy->enStatus;
        }
        cv::Rect rectChar(vecIndex1[i], vecIndex3[0], vecIndex2[i] - vecIndex1[i], vecIndex4[0] - vecIndex3[0]);
        rectChar = CalcUtils::resizeRect(rectChar, cv::Size(rectChar.width+ nMargin * 2, rectChar.height + nMargin * 2));
        CalcUtils::adjustRectROI(rectChar, matGray);
        vecCharRects.push_back(rectChar);
    }

    auto ptrOcvRecord = std::make_shared<OcvRecord>(matGray, vecCharRects);
    RecordManagerInstance->add(ptrOcvRecord, pstRpy->nRecordId);

    if (!isAutoMode()) {
        cv::Mat matDrawResult;
        cv::cvtColor(matGray, matDrawResult, CV_GRAY2BGR);
        for (auto rectChar : vecCharRects)
            cv::rectangle(matDrawResult, rectChar, GREEN_SCALAR, 1);

        _rotateOcvImageBack(pstCmd->enDirection, matDrawResult);
        pstRpy->matResultImg = pstCmd->matInputImg.clone();
        cv::Mat matResultROI(pstRpy->matResultImg, pstCmd->rectROI);
        matDrawResult.copyTo(matResultROI);
    }

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_imageSplitByMax(const cv::Mat &matInput, UInt16 objCount, std::vector<int> &vecIndex1, std::vector<int> &vecIndex2) {
    cv::Mat matOneRow, matFilterResult;
    cv::reduce(matInput, matOneRow, 0, cv::ReduceTypes::REDUCE_MAX);
    cv::medianBlur(matOneRow, matFilterResult, 5);
#ifdef _DEBUG
    auto vecVecRow = CalcUtils::matToVector<uchar>(matOneRow);
    auto vecVecFilter = CalcUtils::matToVector<uchar>(matFilterResult);
#endif
    matOneRow = matFilterResult;
    auto threshold = _autoThreshold(matOneRow);
    for (int col = 0; col < matOneRow.cols - 1; ++ col) {
        auto value = matOneRow.at<uchar>(col);
        auto nextValue = matOneRow.at<uchar>(col + 1);
        if (value <= threshold && nextValue > threshold)
            vecIndex1.push_back(col);
        else if (value > threshold && nextValue <= threshold)
            vecIndex2.push_back(col);
    }

    double minValue = 0., maxValue = 0.;
    cv::minMaxIdx(matOneRow, &minValue, &maxValue);
    auto minGray = static_cast<uchar>(minValue);
    auto maxGray = static_cast<uchar>(maxValue);
    int nIteration = 0;
    while (vecIndex1.size() != objCount || vecIndex2.size() != objCount) {
        ++ nIteration;
        if (vecIndex1.size() > objCount || vecIndex2.size() > objCount) {
            maxGray = threshold;
            threshold = (threshold + minGray) / 2;
        }else {
            minGray = threshold;
            threshold = (threshold + maxGray) / 2;
        }

        vecIndex1.clear(); vecIndex2.clear();
        for (int col = 0; col < matOneRow.cols - 1; ++ col) {
            auto value = matOneRow.at<uchar>(col);
            auto nextValue = matOneRow.at<uchar>(col + 1);
            if (value <= threshold && nextValue > threshold)
                vecIndex1.push_back(col);
            else if (value > threshold && nextValue <= threshold)
                vecIndex2.push_back(col);
        }

        if (nIteration > 10) {
            if (objCount > 1)
                return VisionStatus::FAILED_TO_SPLIT_IMAGE;
            else {
                if (vecIndex1.empty())
                    vecIndex1.push_back(0);
                if (vecIndex2.empty())
                    vecIndex2.push_back(matOneRow.cols - 1);
                break;
            }
        }
    }
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_imageSplitByMaxNew(const cv::Mat &matInput, UInt16 objCount, std::vector<int> &vecIndex1, std::vector<int> &vecIndex2) {
    cv::Mat matOneRow, matFilterResult;
    cv::reduce(matInput, matOneRow, 0, cv::ReduceTypes::REDUCE_MAX);
    matOneRow.convertTo(matOneRow, CV_32FC1);
    cv::blur(matOneRow, matFilterResult, cv::Size(5, 1), cv::Point(-1,-1), cv::BorderTypes::BORDER_REPLICATE);
#ifdef _DEBUG
    auto vecVecRow = CalcUtils::matToVector<float>(matOneRow);
    auto vecVecFilter = CalcUtils::matToVector<float>(matFilterResult);
#endif
    matOneRow = matFilterResult;

    int index1 = 0, index2 = 0;
    // Find left side minimum value
    for (int i = 0; i < matOneRow.cols - 1; ++ i) {
        if (matOneRow.at<float>(i+1) > matOneRow.at<float>(i)) {
            index1 = i;
            break;
        }
    }

    // Find right side minimum value
    for(int i = matOneRow.cols - 1; i > 0; -- i) {
        if (matOneRow.at<float>(i-1) > matOneRow.at<float>(i)) {
            index2 = i;
            break;
        }
    }

    auto y1 = ToFloat(matOneRow.at<float>(index1));
    auto y2 = ToFloat(matOneRow.at<float>(index2));

    float k = (y1 - y2) / (index1 - index2);
    float b = y1 - k * index1;
    auto matXX = CalcUtils::intervals<float>(0.f, 1.f, ToFloat(matOneRow.cols - 1));
    cv::Mat matLine = matXX * k + b;
    cv::transpose(matLine, matLine);
    matOneRow = matOneRow - matLine;
#ifdef _DEBUG
    vecVecRow = CalcUtils::matToVector<float>(matOneRow);
#endif
    auto minGray = std::max(matOneRow.at<float>(index1), matOneRow.at<float>(index2));

    const int COMPARE_COUNT = 5;
    if ((index2 - index1) <= 2 * COMPARE_COUNT)
        return VisionStatus::FAILED_TO_SPLIT_IMAGE;

    auto isValueLargest = [](float value, const cv::Mat &matInput) {
        for (int i = 0; i < matInput.cols; ++ i)
            if (value < matInput.at<float>(i))
                return false;
        return true;
    };

    std::vector<int> vecPeakPoints;
    std::vector<float> vecPeakValue;
    for (int i = index1 + COMPARE_COUNT; i <= index2 - COMPARE_COUNT; ++ i) {
        auto value = matOneRow.at<float>(i);
        cv::Mat matRange(matOneRow, cv::Rect(i - COMPARE_COUNT, 0, 2 * COMPARE_COUNT, 1));
        if(isValueLargest(value, matRange)) {
            vecPeakPoints.push_back(i);
            vecPeakValue.push_back(value);
        }
    }
    if (vecPeakValue.empty())
        return VisionStatus::FAILED_TO_SPLIT_IMAGE;
    auto maxGray = std::min(vecPeakValue.front(), vecPeakValue.back());

    auto threshold = (minGray + maxGray) / 2;
    for (int col = 0; col < matOneRow.cols - 1; ++ col) {
        auto value = matOneRow.at<float>(col);
        auto nextValue = matOneRow.at<float>(col + 1);
        if (value <= threshold && nextValue > threshold)
            vecIndex1.push_back(col);
        else if (value > threshold && nextValue <= threshold)
            vecIndex2.push_back(col);
    }

    int nIteration = 0;
    while (vecIndex1.size() != objCount || vecIndex2.size() != objCount) {
        ++ nIteration;
        if (vecIndex1.size() > objCount || vecIndex2.size() > objCount) {
            maxGray = threshold;
            threshold = (threshold + minGray) / 2;
        }else {
            minGray = threshold;
            threshold = (threshold + maxGray) / 2;
        }

        vecIndex1.clear(); vecIndex2.clear();
        for (int col = 0; col < matOneRow.cols - 1; ++ col) {
            auto value = matOneRow.at<float>(col);
            auto nextValue = matOneRow.at<float>(col + 1);
            if (value <= threshold && nextValue > threshold)
                vecIndex1.push_back(col);
            else if (value > threshold && nextValue <= threshold)
                vecIndex2.push_back(col);
        }

        if (nIteration > 10) {
            if (objCount > 1)
                return VisionStatus::FAILED_TO_SPLIT_IMAGE;
            else {
                if (vecIndex1.empty())
                    vecIndex1.push_back(0);
                if (vecIndex2.empty())
                    vecIndex2.push_back(matOneRow.cols - 1);
                break;
            }
        }
    }

    if (vecIndex1.size() > 1) {
        for (int i = 0; i < vecIndex1.size() - 1; ++ i) {
            int indexTmp1 = vecIndex2[i];
            int indexTmp2 = vecIndex1[i+1];
            int indexMin = indexTmp1;
            auto minValue = matOneRow.at<float>(indexMin);
            for (int j = indexTmp1 + 1; j <= indexTmp2; ++ j) {
                if (matOneRow.at<float>(j) < minValue) {
                    minValue = matOneRow.at<float>(j);
                    indexMin = j;
                }
            }
            int d1 = abs(indexTmp1 - indexMin);
            int d2 = abs(indexTmp2 - indexMin);
            if (d1 > d2)
                vecIndex2[i] = indexMin - d2;
            else
                vecIndex1[i+1] = indexMin + d1;
        }
    }
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::ocv(const PR_OCV_CMD *const pstCmd, PR_OCV_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("The input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (pstCmd->vecRecordId.empty()) {
        WriteLog("The input record id list is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCaseOcv);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI), matGray;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI;

    std::vector<OcvRecordPtr> vecRecordPtr;
    for (const auto recordId : pstCmd->vecRecordId) {
        OcvRecordPtr ptrOcvRecord = std::static_pointer_cast<OcvRecord>(RecordManagerInstance->get(recordId));
        if (nullptr == ptrOcvRecord) {
            char chArrMsg[100];
            _snprintf(chArrMsg, sizeof(chArrMsg), "Failed to get record ID %d in system.", recordId);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            FINISH_LOGCASE_EX;
            return pstRpy->enStatus;
        }

        if (ConfigInstance->getDebugMode() == Vision::PR_DEBUG_MODE::SHOW_IMAGE)
            showImage("Ocv template", ptrOcvRecord->getBigTmpl());

        vecRecordPtr.push_back(ptrOcvRecord);
    }

    _ocvOneDirection(pstCmd->enDirection, matGray, vecRecordPtr, pstCmd, pstRpy);
    if (pstRpy->enStatus != VisionStatus::OK && pstCmd->bAcceptReverse) {
        PR_DIRECTION enReverseDir;
        switch (pstCmd->enDirection) {
        case PR_DIRECTION::UP:
            enReverseDir = PR_DIRECTION::DOWN;
            break;
        case PR_DIRECTION::DOWN:
            enReverseDir = PR_DIRECTION::UP;
            break;
        case PR_DIRECTION::LEFT:
            enReverseDir = PR_DIRECTION::RIGHT;
            break;
        case PR_DIRECTION::RIGHT:
            enReverseDir = PR_DIRECTION::LEFT;
            break;
        default:
            char chArrMsg[100];
            _snprintf(chArrMsg, sizeof(chArrMsg), "Input direction %d is invalid.", ToInt32(pstCmd->enDirection));
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            break;
        }

        _ocvOneDirection(enReverseDir, matGray, vecRecordPtr, pstCmd, pstRpy);
    }

    FINISH_LOGCASE_EX;
    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::_checkOcvTmplSize(const cv::Mat &matInput, const std::vector<OcvRecordPtr>& vecRecordPtr)
{
    for (const auto ptrOcvRecord : vecRecordPtr) {
        if (ConfigInstance->getDebugMode() == Vision::PR_DEBUG_MODE::SHOW_IMAGE)
            showImage("Ocv template", ptrOcvRecord->getBigTmpl());

        if (ptrOcvRecord->getBigTmpl().cols > matInput.cols || ptrOcvRecord->getBigTmpl().rows > matInput.rows) {
            char chArrMsg[100];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The record template size (%d, %d) is larger than search ROI size (%d, %d).",
                ptrOcvRecord->getBigTmpl().cols, ptrOcvRecord->getBigTmpl().rows, matInput.cols, matInput.rows);
            return VisionStatus::INVALID_PARAM;
        }
    }
    return VisionStatus::OK;
}

/*static*/ VisionStatus VisionAlgorithm::_ocvOneDirection(PR_DIRECTION enDirection, const cv::Mat &matInput, const std::vector<OcvRecordPtr>& vecRecordPtr, const PR_OCV_CMD *const pstCmd, PR_OCV_RPY *const pstRpy) {
    pstRpy->enStatus = VisionStatus::OK;

    cv::Mat matGray = matInput.clone();
    _rotateOcvImage(enDirection, matGray);
    if (ConfigInstance->getDebugMode() == Vision::PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Rotated search image", matGray);

    pstRpy->enStatus = _checkOcvTmplSize(matGray, vecRecordPtr);
    if (VisionStatus::OK != pstRpy->enStatus)
        return pstRpy->enStatus;

    VectorOfPoint2f vecResultPos;
    VectorOfFloat vecCorrelation;
    
    for (const auto& ptrOcvRecord : vecRecordPtr) {
        cv::Point2f ptPosition;
        float fRotation = 0.f, fCorrelation = 0.f;
        MatchTmpl::matchTemplate(matGray, ptrOcvRecord->getBigTmpl(), false, PR_OBJECT_MOTION::TRANSLATION, ptPosition, fRotation, fCorrelation);
        vecResultPos.push_back(ptPosition);
        vecCorrelation.push_back(fCorrelation);
    }

    cv::Mat matDrawResult;
    if (!isAutoMode())
        cv::cvtColor(matGray, matDrawResult, CV_GRAY2BGR);

    auto iterMaxScore = std::max_element(vecCorrelation.begin(), vecCorrelation.end());
    pstRpy->fOverallScore = *iterMaxScore * ConstToPercentage;
    if (pstRpy->fOverallScore < pstCmd->fMinMatchScore) {
        char chArrMsg[100];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The best match socre %f is lower than minimum required score %f.", pstRpy->fOverallScore, pstCmd->fMinMatchScore);
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::OCV_MATCH_SCORE_UNDER_LIMIT;
        return pstRpy->enStatus;
    }   

    auto maxScoreIndex = iterMaxScore - vecCorrelation.begin();
    cv::Point ptTarget = vecResultPos[maxScoreIndex];
    auto ptrRecordPtr = vecRecordPtr[maxScoreIndex];
    cv::Rect rectTarget(ptTarget.x - (ptrRecordPtr->getBigTmpl().cols + 1) / 2, ptTarget.y - (ptrRecordPtr->getBigTmpl().rows + 1) / 2, ptrRecordPtr->getBigTmpl().cols, ptrRecordPtr->getBigTmpl().rows);
    cv::Mat matBigTarget(matGray, rectTarget);
    auto vecCharRects = ptrRecordPtr->getCharRects();
    const int nSrchRegionMargin = 5;
    for (size_t i = 0; i < vecCharRects.size(); ++ i) {
        auto rectChar = vecCharRects[i];
        auto rectCharSrchWindow = CalcUtils::resizeRect(rectChar, cv::Size(rectChar.width + nSrchRegionMargin * 2, rectChar.height + nSrchRegionMargin * 2));
        CalcUtils::adjustRectROI(rectCharSrchWindow, matBigTarget);
        cv::Mat matCharSrch(matBigTarget, rectCharSrchWindow);
        if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
            showImage("Char Srch Image", matCharSrch);
        cv::Mat matCharTmpl(ptrRecordPtr->getBigTmpl(), rectChar);
        cv::Point2f ptPosition;
        float fRotation = 0.f, fCorrelation = 0.f;
        MatchTmpl::matchTemplate(matCharSrch, matCharTmpl, false, PR_OBJECT_MOTION::TRANSLATION, ptPosition, fRotation, fCorrelation);
        if (!isAutoMode()) {
            cv::rectangle(matDrawResult, rectTarget, BLUE_SCALAR, 1);

            cv::Rect rectChar(ToInt32(ptPosition.x) - matCharTmpl.cols / 2, ToInt32(ptPosition.y) - matCharTmpl.rows / 2, matCharTmpl.cols, matCharTmpl.rows);
            rectChar.x += rectCharSrchWindow.x + rectTarget.x;
            rectChar.y += rectCharSrchWindow.y + rectTarget.y;
            cv::rectangle(matDrawResult, rectChar, GREEN_SCALAR, 1);

            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.8;
            int thickness = 2;
            int baseline = 0;

            char strCharScore[100];
            _snprintf(strCharScore, sizeof(strCharScore), "%d", ToInt32(fCorrelation * ConstToPercentage));
            cv::Size textSize = cv::getTextSize(strCharScore, fontFace, fontScale, thickness, &baseline);
            //The height use '+' because text origin start from left-bottom.
            cv::Point ptTextOrg(rectChar.x + (rectChar.width - textSize.width) / 2, rectChar.y - 10);
            cv::putText(matDrawResult, strCharScore, ptTextOrg, fontFace, fontScale, CYAN_SCALAR, thickness);
        }
        pstRpy->vecCharScore.push_back(fCorrelation * ConstToPercentage);
        if (fCorrelation * ConstToPercentage < pstCmd->fMinMatchScore) {
            char chArrMsg[100];
            _snprintf(chArrMsg, sizeof(chArrMsg), "Char %d match socre %f is lower than minimum required score %f.", i, fCorrelation * ConstToPercentage, pstCmd->fMinMatchScore);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::OCV_MATCH_SCORE_UNDER_LIMIT;
        }
    }

    if (!isAutoMode()) {
        _rotateOcvImageBack(pstCmd->enDirection, matDrawResult);
        if (pstRpy->matResultImg.empty())
            pstRpy->matResultImg = pstCmd->matInputImg.clone();
        if (pstRpy->matResultImg.channels() == 1)
            cv::cvtColor(pstRpy->matResultImg, pstRpy->matResultImg, CV_GRAY2BGR);
        cv::rectangle(pstRpy->matResultImg, pstCmd->rectROI, BLUE_SCALAR, 2);
        cv::Mat matResultROI(pstRpy->matResultImg, pstCmd->rectROI);
        matDrawResult.copyTo(matResultROI);
    }
    return pstRpy->enStatus;
}

// If the input image direction is not from left to right, then rotate it to left to right.
/*static*/ void VisionAlgorithm::_rotateOcvImage(PR_DIRECTION enDirection, cv::Mat &matInOut) {
    switch (enDirection)
    {
    case PR_DIRECTION::LEFT:
        cv::flip(matInOut, matInOut, -1); //flip(-1)=180
        break;
    case PR_DIRECTION::UP:
        cv::transpose(matInOut, matInOut);
        cv::flip(matInOut, matInOut, 1); //transpose+flip(1)=CW
        break;
    case PR_DIRECTION::DOWN:
        cv::transpose(matInOut, matInOut);
        cv::flip(matInOut, matInOut, 0); //transpose+flip(0)=CCW
        break;
    default:
        break;
    }
}

// The input image is from left to right, rotate it to given direction
/*static*/ void VisionAlgorithm::_rotateOcvImageBack(PR_DIRECTION enDirection, cv::Mat &matInOut) {
    switch (enDirection)
    {
    case PR_DIRECTION::LEFT:
        cv::flip(matInOut, matInOut, -1);    //flip(-1)=180
        break;
    case PR_DIRECTION::DOWN:
        cv::transpose(matInOut, matInOut);
        cv::flip(matInOut, matInOut, 1); //transpose+flip(1)=CW
        break;
    case PR_DIRECTION::UP:
        cv::transpose(matInOut, matInOut);
        cv::flip(matInOut, matInOut, 0); //transpose+flip(0)=CCW
        break;
    default:
        break;
    }
}

/*static*/ VisionStatus VisionAlgorithm::read2DCode(const PR_READ_2DCODE_CMD *const pstCmd, PR_READ_2DCODE_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("The input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    MARK_FUNCTION_START_TIME;
    SETUP_LOGCASE(LogCase2DCode);

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI), matGray;
    if (pstCmd->matInputImg.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray = matROI;

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("ROI gray image", matGray);

    cv::erode(matGray, matGray, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(7, 7)));

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Erode result image", matGray);

    cv::Mat matCanny;
    cv::Canny(matGray, matCanny, pstCmd->nEdgeThreshold, pstCmd->nEdgeThreshold * 2);

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Canny result image", matCanny);

    cv::Mat matSumX, matSumY;
    cv::reduce(matCanny, matSumX, 0, cv::ReduceTypes::REDUCE_SUM, CV_32FC1);
    cv::reduce(matCanny, matSumY, 1, cv::ReduceTypes::REDUCE_SUM, CV_32FC1);
    cv::transpose(matSumY, matSumY);

    cv::blur(matSumX, matSumX, cv::Size(51, 1));
    cv::blur(matSumY, matSumY, cv::Size(51, 1));

    matSumX = matSumX / PR_MAX_GRAY_LEVEL;
    matSumY = matSumY / PR_MAX_GRAY_LEVEL;

#ifdef _DEBUG
    auto vecVecSumX = CalcUtils::matToVector<float>(matSumX);
    auto vecVecSumY = CalcUtils::matToVector<float>(matSumY);
#endif

    std::vector<int> vecIndexX, vecIndexY;
    for (int i = 0; i < matSumX.cols; ++ i)
        if (matSumX.at<float>(i) > 6)
            vecIndexX.push_back(i);

    if (vecIndexX.size() < 10) {
        pstRpy->enStatus = VisionStatus::NO_2DCODE_DETECTED;
        return pstRpy->enStatus;
    }

    for (int i = 0; i < matSumY.cols; ++ i)
        if (matSumY.at<float>(i) > 6)
            vecIndexY.push_back(i);

    if (vecIndexY.size() < 10) {
        pstRpy->enStatus = VisionStatus::NO_2DCODE_DETECTED;
        return pstRpy->enStatus;
    }

    const int margin = 5;
    int lft = (vecIndexX.front() - margin) > 0 ? (vecIndexX.front() - margin) : 0;
    int rgt = (vecIndexX.back()  + margin) > matGray.cols ? matGray.cols : (vecIndexX.back() + margin);
    int top = (vecIndexY.front() - margin) > 0 ? (vecIndexY.front() - margin) : 0;
    int btm = (vecIndexY.back()  + margin) > matGray.rows ? matGray.rows : (vecIndexY.back() + margin);
    cv::Rect rect2DCode(lft, top, rgt - lft, btm - top);

    cv::Mat matBarcode(matGray, rect2DCode);

    cv::Canny(matBarcode, matCanny, pstCmd->nEdgeThreshold, pstCmd->nEdgeThreshold * 2);

    cv::Mat matLabels;
    cv::connectedComponents(matCanny, matLabels, 8, CV_32S);

    double minValue = 0., maxValue = 0.;
    cv::minMaxIdx(matLabels, &minValue, &maxValue);

    for (int nLabel = 1; nLabel <= ToInt32(maxValue); ++ nLabel) {
        cv::Mat matCC = CalcUtils::genMaskByValue<Int32>(matLabels, nLabel);
        float fArea = ToFloat(cv::countNonZero(matCC));
        if (fArea < pstCmd->nRemoveNoiseArea)
            matLabels.setTo(0, matCC);
    }

    cv::Mat mat2DCodeEdge = matLabels > 0;

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("2D code edge", mat2DCodeEdge);

    cv::Mat matOuterEdge;
    cv::Mat matImageFill = _imageFilling(mat2DCodeEdge, matOuterEdge);
    matImageFill = _imageFilling(matImageFill, matOuterEdge);

    //cv::Canny(matImageFill, matOuterEdge, 100, 255);
    //cv::dilate(matOuterEdge, matOuterEdge, cv::getStructuringElement(cv::MorphShapes::MORPH_CROSS, cv::Size(3, 3)));
    cv::dilate(matOuterEdge, matOuterEdge, cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(1, 1)));

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("Image fill result", matImageFill);
        showImage("Outer edge", matOuterEdge);
    }
    std::vector<PR_Line2f> vecLines;
    cv::Mat matHoughResult;
    cv::cvtColor(matOuterEdge, matHoughResult, cv::COLOR_GRAY2BGR);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(matOuterEdge, lines, 5, CV_PI / 180, 30, 0, 0);

    std::vector<int> vecReverseFit;
    VectorOfFloat vecSlope, vecIntercept;
    std::vector<HOUGH_LINE> vecHoughLines;
    for (size_t i = 0; i < lines.size(); ++ i) {
        if (vecHoughLines.size() >= 4)
            break;

        float rho = lines[i][0], theta = lines[i][1];
        if (std::find(vecHoughLines.begin(), vecHoughLines.end(), HOUGH_LINE{rho, theta}) != vecHoughLines.end())
            continue;
        vecHoughLines.push_back(HOUGH_LINE{rho, theta});
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        PR_Line2f line(pt1, pt2);
        vecLines.push_back(line);
        cv::Scalar scalar(YELLOW_SCALAR);
        if (i == 0)
            scalar = BLUE_SCALAR;
        else if (i == 1)
            scalar = GREEN_SCALAR;
        else if (i == 2)
            scalar = RED_SCALAR;
        cv::line(matHoughResult, pt1, pt2, scalar, 1, CV_AA);

        VectorOfPoint vecPoints;
        VectorOfPoint vecPointToFit;
        cv::findNonZero(matOuterEdge, vecPoints);
        for (const auto &point : vecPoints) {
            auto dist = point.x * a + point.y * b - rho;
            if (fabs(dist) < 5)
                vecPointToFit.push_back(point);
        }

        bool bReversedFit = false;
        float fSlope = 0.f, fIntercept = 0.f;
        auto vecPointsToKeep = Fitting::fitLineRemoveStray(vecPointToFit, bReversedFit, 0.2f, fSlope, fIntercept);
        auto stLine = CalcUtils::calcEndPointOfLine(vecPointsToKeep, bReversedFit, fSlope, fIntercept);
        vecReverseFit.push_back(bReversedFit);
        vecSlope.push_back(fSlope);
        vecIntercept.push_back(fIntercept);

        cv::line(matHoughResult, stLine.pt1, stLine.pt2, CYAN_SCALAR, 1, CV_AA);
    }

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Hough result", matHoughResult);

    if (lines.size() < 4) {
        pstRpy->enStatus = VisionStatus::NO_2DCODE_DETECTED;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }

    auto vecCorners = _lineIntersect4P(vecReverseFit, vecSlope, vecIntercept);

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        cv::Mat matBarcodeCornerResult;
        cv::cvtColor(matBarcode, matBarcodeCornerResult, CV_GRAY2BGR);
        for (int i = 0; i < 3; ++i)
            cv::line(matBarcodeCornerResult, vecCorners[i], vecCorners[i + 1], GREEN_SCALAR, 1);
        cv::line(matBarcodeCornerResult, vecCorners[3], vecCorners[0], GREEN_SCALAR, 1);
        showImage("Find corner result", matBarcodeCornerResult);
    }

    int radius = 5;
    for (const auto &point : vecCorners)
        cv::circle(matHoughResult, point, radius++, GREEN_SCALAR, 2);

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Find corner result", matHoughResult);

    cv::Point2f ptCenter((vecCorners[0].x + vecCorners[2].x) / 2.f, (vecCorners[0].y + vecCorners[2].y) / 2.f);
    cv::Point2f ptCenter1((vecCorners[1].x + vecCorners[3].x) / 2.f, (vecCorners[1].y + vecCorners[3].y) / 2.f);
    float angle = atan2(vecCorners[1].y - vecCorners[0].y, vecCorners[1].x - vecCorners[0].x);  // OpenCV Y is reversed of traditional coordinate. So the atan2 angle is also reversed.
    float angle1 = atan2(vecCorners[2].y - vecCorners[3].y, vecCorners[2].x - vecCorners[3].x);  // OpenCV Y is reversed of traditional coordinate. So the atan2 angle is also reversed.
    angle = CalcUtils::radian2Degree(angle);
    cv::Mat matWarp = cv::getRotationMatrix2D(ptCenter, angle, 1.f);

    float width  = CalcUtils::distanceOf2Point(vecCorners[0], vecCorners[1]);
    float height = CalcUtils::distanceOf2Point(vecCorners[1], vecCorners[2]);

    cv::Mat matWarpResult;
    cv::warpAffine(matBarcode, matWarpResult, matWarp, matGray.size());
    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Warp result", matWarpResult);

    cv::Rect2f rectBarcode(ptCenter.x - width / 2.f , ptCenter.y - height / 2.f, width, height);
    cv::Mat matReshapedBarcode(matWarpResult, rectBarcode);
    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Warp ROI result", matReshapedBarcode);

    int m, n, startRow, startCol;
    if (!_lineScanMatrix(matReshapedBarcode, m, n, startRow, startCol)) {
        pstRpy->enStatus = VisionStatus::NO_2DCODE_DETECTED;
        FINISH_LOGCASE;
        return pstRpy->enStatus;
    }
    
    cv::Mat matSubBarcode(matReshapedBarcode, cv::Rect(0, startRow, matReshapedBarcode.cols - startCol, matReshapedBarcode.rows - startRow));
    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)
        showImage("Refined Barcode", matSubBarcode);

    cv::Mat matDataMatrix = cv::Mat::zeros(2*m, 2*n, CV_8UC1);
    cv::Mat matFilter;
    float fGridWidth  = ToFloat(matSubBarcode.cols) / ToFloat(2 * n);
    float fGridHeight = ToFloat(matSubBarcode.rows) / ToFloat(2 * m);
    const int KERNEL_SIZE = ToInt32(fGridWidth * 3.f) / 2 * 2 + 1;
    cv::blur(matSubBarcode, matFilter, cv::Size(KERNEL_SIZE, KERNEL_SIZE), cv::Point(-1, -1), cv::BorderTypes::BORDER_REPLICATE);
    cv::Mat matCompare = matFilter < matSubBarcode;    
    int GRID_SIZE = 10;
    int BORDER_SIZE = 10;
    cv::Mat matConvertedBarcode = cv::Mat::ones(2*m*GRID_SIZE + BORDER_SIZE * 2, 2*n*GRID_SIZE + BORDER_SIZE * 2, CV_8UC1) * PR_MAX_GRAY_LEVEL;
    for (int row = 0; row < 2*m; ++ row) {
        for (int col = 0; col < 2*n; ++ col) {
            cv::Point point(ToInt32(fGridWidth / 2 + fGridWidth * col), ToInt32(fGridHeight / 2 + fGridHeight * row));
            matDataMatrix.at<uchar>(row, col) = matCompare.at<uchar>(point);
            if (matDataMatrix.at<uchar>(row, col) == 0) {
                cv::Mat matGrid(matConvertedBarcode, cv::Rect(col * GRID_SIZE + BORDER_SIZE, row * GRID_SIZE + BORDER_SIZE, GRID_SIZE, GRID_SIZE));
                matGrid.setTo(0);
            }
        }
    }

    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("Compare result", matCompare);
        showImage("Result Barcode", matConvertedBarcode);
    }

    cv::imwrite("./data/Gen2DCode.png", matConvertedBarcode);

    if (! DataMatrix::read(matConvertedBarcode, pstRpy->strReadResult))
        pstRpy->enStatus = VisionStatus::FAILED_TO_READ_2DCODE;

    FINISH_LOGCASE;
    MARK_FUNCTION_END_TIME;

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ cv::Mat VisionAlgorithm::_imageFilling(const cv::Mat &matInput, cv::Mat &matOuterEdge) {
    cv::Mat matResult = matInput.clone();
    matOuterEdge = cv::Mat::zeros(matInput.size(), CV_8UC1);
    for (int col = 0; col < matInput.cols; ++ col) {
        cv::Mat matCol = matResult.col(col);
        VectorOfPoint vecPoints;
        vecPoints.reserve(matInput.rows);
        cv::findNonZero(matCol, vecPoints);
        if (vecPoints.size() > 2) {
            for (int y = vecPoints.front().y; y < vecPoints.back().y; ++ y)
                matResult.at<uchar>(y, col) = PR_MAX_GRAY_LEVEL;

            matOuterEdge.at<uchar>(vecPoints.front().y, col) = PR_MAX_GRAY_LEVEL;
            matOuterEdge.at<uchar>(vecPoints.back().y,  col) = PR_MAX_GRAY_LEVEL;
        }else if (vecPoints.size() == 1)
            matOuterEdge.at<uchar>(vecPoints.front().y, col) = PR_MAX_GRAY_LEVEL;
    }

    for (int row = 0; row < matInput.rows; ++ row) {
        cv::Mat matRow = matResult.row(row);
        VectorOfPoint vecPoints;
        vecPoints.reserve(matInput.cols);
        cv::findNonZero(matRow, vecPoints);
        if (vecPoints.size() > 2) {
            for (int x = vecPoints.front().x; x < vecPoints.back().x; ++ x)
                matResult.at<uchar>(row, x) = PR_MAX_GRAY_LEVEL;

            matOuterEdge.at<uchar>(row, vecPoints.front().x) = PR_MAX_GRAY_LEVEL;
            matOuterEdge.at<uchar>(row, vecPoints.back().x)  = PR_MAX_GRAY_LEVEL;
        }else if (vecPoints.size() == 1)
            matOuterEdge.at<uchar>(row, vecPoints.front().x) = PR_MAX_GRAY_LEVEL;
    }

    return matResult;
}

/*static*/ VectorOfPoint2f VisionAlgorithm::_lineIntersect4P(std::vector<int> &vecReverseFit, VectorOfFloat &vecSlope, VectorOfFloat &vecIntercept) {
    assert(vecReverseFit.size() == 4);
    assert(vecSlope.size() == 4);
    assert(vecIntercept.size() == 4);

    VectorOfFloat vecTheta;
    for (int i = 0; i < 4; ++ i) {
        if (vecReverseFit[i]) {
            if (vecSlope[i] < 0.0001)
                vecTheta.push_back(ToFloat(CV_PI) / 2.f);
            else
                vecTheta.push_back(atan(1.f / vecSlope[i]));
        }else
            vecTheta.push_back(atan(vecSlope[i]));
    }

    // If line 2 and line 3 are parallel, then change the sequence of line 3 and line 4.
    if (fabs(vecTheta[1] - vecTheta[2]) < 10.f / 180.f * CV_PI) {
        std::swap(vecTheta[2], vecTheta[3]);
        std::swap(vecReverseFit[2], vecReverseFit[3]);
        std::swap(vecSlope[2], vecSlope[3]);
        std::swap(vecIntercept[2], vecIntercept[3]);
    }

    vecReverseFit.push_back(vecReverseFit.front());
    vecSlope.push_back(vecSlope.front());
    vecIntercept.push_back(vecIntercept.front());

    VectorOfPoint2f vecPoints;
    for (int i = 0; i < 4; ++ i) {
        cv::Point2f point;
        CalcUtils::twoLineIntersect(vecReverseFit[i] != 0, vecSlope[i], vecIntercept[i], vecReverseFit[i + 1] != 0, vecSlope[i + 1], vecIntercept[i + 1], point);
        vecPoints.push_back(point);
    }

    auto angle1 = atan2(vecPoints[1].y - vecPoints[0].y, vecPoints[1].x - vecPoints[0].x);
    auto angle2 = atan2(vecPoints[2].y - vecPoints[1].y, vecPoints[2].x - vecPoints[1].x);
    auto th = angle2 - angle1;
    auto nn = ToFloat(std::floor(th / 2.f / CV_PI + 0.5));
    th = th - nn * 2 * ToFloat(CV_PI);
    if (th > 0) {
        std::reverse(vecPoints.begin() + 1, vecPoints.end());
    }
    return vecPoints;
}

/*static*/ bool VisionAlgorithm::_lineScanRow(const cv::Mat &matInput, int &n, int &startRow) {
    bool bFoundRow = false;
    for (int row = 0; row < matInput.rows; ++ row) {
        cv::Mat matCol = matInput.row(row);
        int th = _autoThreshold(matCol);
        cv::Mat matThreshold;
        cv::threshold(matCol, matThreshold, th, PR_MAX_GRAY_LEVEL, cv::ThresholdTypes::THRESH_BINARY);
        matThreshold.convertTo(matThreshold, CV_32FC1);
        cv::Mat matDiff = CalcUtils::diff(matThreshold, 1, CalcUtils::DIFF_ON_X_DIR);
#ifdef _DEBUG
        auto vecVecThreshold = CalcUtils::matToVector<float>(matThreshold);
        auto vecVecDiff = CalcUtils::matToVector<float>(matDiff);
#endif
        std::vector<int> vecIndex1, vecIndex2;
        for (int i = 0; i < matDiff.cols; ++ i) {
            auto value = matDiff.at<float>(i);
            if (value > 0)
                vecIndex1.push_back(i);
            else if (value < 0)
                vecIndex2.push_back(i);
        }

        if (vecIndex1.size() < 2 || vecIndex2.size() < 2)
            continue;

        std::vector<int> vecDD;
        for (size_t i = 0; i < vecIndex1.size() - 1; ++ i)
            vecDD.push_back(vecIndex1[i + 1] - vecIndex1[i]);
        for (size_t i = 0; i < vecIndex2.size() - 1; ++ i)
            vecDD.push_back(vecIndex2[i + 1] - vecIndex2[i]);
        int maxD = *std::max_element(vecDD.begin(), vecDD.end());
        int minD = *std::min_element(vecDD.begin(), vecDD.end());
        int D1 = maxD - minD;
        float meanD = CalcUtils::mean(vecDD);
        
        n = ToInt32(std::max(vecIndex1.size(), vecIndex2.size()));
        float Tc = floor(ToFloat(matCol.cols) / ToFloat(n) + 0.5f);
        if (fabs(meanD - Tc) < 1 && D1 < 5) {
            bFoundRow = true;
            startRow = row;
            break;
        }
    }
    return bFoundRow;
}

/*static*/ bool VisionAlgorithm::_lineScanMatrix(const cv::Mat &matInput, int &m, int &n, int &startRow, int &startCol) {
    if (!_lineScanRow(matInput, n, startRow))
        return false;
    cv::Mat matSubMat(matInput, cv::Rect(0, startRow, matInput.cols, matInput.rows - startRow));
    cv::Mat matSubMatT;
    cv::transpose(matSubMat, matSubMatT);
    cv::Mat matFlip;
    cv::flip(matSubMatT, matFlip, 0);
    if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
        showImage("Subtract", matSubMat);
        showImage("Subtract Tranpose 2D code", matSubMatT);
        showImage("Flip image", matFlip);
    }
    return _lineScanRow(matFlip, m, startCol);
}

/*static*/ VisionStatus VisionAlgorithm::lrnSimilarity(const PR_LRN_SIMILARITY_CMD *const pstCmd, PR_LRN_SIMILARITY_RPY *const pstRpy, bool bReplay/*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;
    if (! pstCmd->matMask.empty()) {
        if (pstCmd->matMask.rows != pstCmd->matInputImg.rows || pstCmd->matMask.cols != pstCmd->matInputImg.cols) {
            char chArrMsg[1000];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The mask size ( %d, %d ) not match with input image size ( %d, %d ).",
                pstCmd->matMask.cols, pstCmd->matMask.rows, pstCmd->matInputImg.cols, pstCmd->matInputImg.rows);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (pstCmd->matMask.channels() != 1) {
            WriteLog("The mask must be gray image!");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }    

    MARK_FUNCTION_START_TIME;

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI);
    if (matROI.channels() > 1)
        cv::cvtColor(matROI, pstRpy->matTmpl, CV_BGR2GRAY);
    else
        pstRpy->matTmpl = matROI.clone();

    cv::Mat matMaskROI;
    if (!pstCmd->matMask.empty())
        matMaskROI = cv::Mat(pstCmd->matMask, pstCmd->rectROI).clone();

    SimilarityRecordPtr ptrRecord = std::make_shared<SimilarityRecord>(PR_RECORD_TYPE::SIMILARITY, pstRpy->matTmpl, matMaskROI);
    RecordManager::getInstance()->add(ptrRecord, pstRpy->nRecordId);
    pstRpy->enStatus = VisionStatus::OK;

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::inspSimilarity(const PR_INSP_SIMILARITY_CMD *const pstCmd, PR_INSP_SIMILARITY_RPY *const pstRpy, bool bReplay/* = false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->matInputImg.empty()) {
        WriteLog("Input image is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    pstRpy->enStatus = _checkInputROI(pstCmd->rectROI, pstCmd->matInputImg, AT);
    if (VisionStatus::OK != pstRpy->enStatus) return pstRpy->enStatus;

    if (pstCmd->vecRecordId.empty()) {
        char chArrMsg[100];
        _snprintf(chArrMsg, sizeof(chArrMsg), "The vector of input record id invalid.");
        WriteLog(chArrMsg);
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;
    pstRpy->enStatus = VisionStatus::OK;
    pstRpy->vecSimilarity.clear();
    VectorOfRect vecResultRects;

    cv::Mat matROI(pstCmd->matInputImg, pstCmd->rectROI), matGray;
    if (matROI.channels() > 1)
        cv::cvtColor(matROI, matGray, CV_BGR2GRAY);
    else
        matGray= matROI.clone();

    for (const auto recordId : pstCmd->vecRecordId) {
        if (recordId <= 0) {
            char chArrMsg[100];
            _snprintf(chArrMsg, sizeof(chArrMsg), "The input record id %d is invalid.", recordId);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        SimilarityRecordPtr ptrRecord = std::static_pointer_cast<SimilarityRecord>(RecordManager::getInstance()->get(recordId));
        if (nullptr == ptrRecord) {
            char chArrMsg[100];
            _snprintf(chArrMsg, sizeof(chArrMsg), "Failed to get record ID %d in system.", recordId);
            WriteLog(chArrMsg);
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        if (ptrRecord->getTmpl().rows >= pstCmd->rectROI.height || ptrRecord->getTmpl().cols >= pstCmd->rectROI.width) {
            WriteLog("The inspect ROI size not match with template size.");
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }

        float fSimilarity = 0.f, fRotation = 0.f;
        cv::Point2f ptResult;

        pstRpy->enStatus = MatchTmpl::matchTemplate(matGray, ptrRecord->getTmpl(), false, PR_OBJECT_MOTION::TRANSLATION, ptResult, fRotation, fSimilarity, ptrRecord->getMask());
        if (pstRpy->enStatus != VisionStatus::OK) {
            WriteLog("Match template fail when inspect similarity.");
            pstRpy->enStatus = VisionStatus::INSP_SIMILARITY_FAIL;
            break;
        }

        pstRpy->vecSimilarity.push_back(fSimilarity * ConstToPercentage);
        cv::Point ptCtr(ToInt32(ptResult.x) + pstCmd->rectROI.x, ToInt32(ptResult.y) + pstCmd->rectROI.y);
        vecResultRects.emplace_back(ptCtr.x - ptrRecord->getTmpl().cols / 2, ptCtr.y - ptrRecord->getTmpl().rows / 2,
            ptrRecord->getTmpl().cols, ptrRecord->getTmpl().rows);
    } 

    if (VisionStatus::OK == pstRpy->enStatus) {
        auto iterMax = std::max_element(pstRpy->vecSimilarity.begin(), pstRpy->vecSimilarity.end());
        if (*iterMax < pstCmd->fMinSimilarity) {
            pstRpy->enStatus = VisionStatus::INSP_SIMILARITY_FAIL;
        }else {
            if (!isAutoMode()) {
                assert(pstRpy->vecSimilarity.size() == vecResultRects.size());
                pstRpy->matResultImg = pstCmd->matInputImg.clone();
                cv::rectangle(pstRpy->matResultImg, pstCmd->rectROI, BLUE_SCALAR, 1);
                auto index = iterMax - pstRpy->vecSimilarity.begin();
                cv::rectangle(pstRpy->matResultImg, vecResultRects[index], GREEN_SCALAR, 1);
            }
        }
    }

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::tableMapping(const PR_TABLE_MAPPING_CMD *const pstCmd, PR_TABLE_MAPPING_RPY *const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);

    if (pstCmd->vecFramePoints.empty()) {
        WriteLog("Input frame points is empty.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    for (size_t i = 0; i < pstCmd->vecFramePoints.size(); ++ i) {
        if (pstCmd->vecFramePoints[i].empty()) {
            std::stringstream ss;
            ss << "Frame " << i << " mapping points vector is empty.";
            WriteLog(ss.str());
            pstRpy->enStatus = VisionStatus::INVALID_PARAM;
            return pstRpy->enStatus;
        }
    }

    MARK_FUNCTION_START_TIME;

    TableMapping::run(pstCmd, pstRpy);

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus VisionAlgorithm::calcRestoreIdx(const PR_CALC_RESTORE_IDX_CMD* const pstCmd, PR_CALC_RESTORE_IDX_RPY* const pstRpy, bool bReplay /*= false*/) {
    assert(pstCmd != nullptr && pstRpy != nullptr);
    pstRpy->vecVecRestoreIndex.clear();

    if (pstCmd->matXOffsetParam.empty() || pstCmd->matYOffsetParam.empty()) {
        WriteLog("Input matXOffsetParam and matYOffsetParam is empty");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    const int bezierParamNum = pstCmd->nBezierRankX * pstCmd->nBezierRankY;
    if (pstCmd->matXOffsetParam.rows != bezierParamNum) {
        std::stringstream ss;
        ss << "The matXOffsetParam rows " << pstCmd->matXOffsetParam.rows << " not match with input bezier rank multiply " << bezierParamNum;
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    if (pstCmd->matYOffsetParam.rows != bezierParamNum) {
        std::stringstream ss;
        ss << "The matYOffsetParam rows " << pstCmd->matYOffsetParam.rows << " not match with input bezier rank multiply " << bezierParamNum;
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    MARK_FUNCTION_START_TIME;

    TableMapping::calcRestoreIdx(pstCmd, pstRpy);

    MARK_FUNCTION_END_TIME;
    return pstRpy->enStatus;
}

}
}
