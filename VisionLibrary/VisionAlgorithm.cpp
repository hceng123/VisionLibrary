#include "VisionAlgorithm.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"
#include "TimeLog.h"
#include "logcase.h"
#include "boost/filesystem.hpp"
#include "RecordManager.h"
#include "Config.h"
#include "CalcUtils.h"
#include "Log.h"
#include "FileUtils.h"

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

    VisionStatus _findEdge(const cv::Mat &matProjection, int nObjectSize, int &nObjectEdgeResult)
    {
        if (matProjection.cols < nObjectSize )
            return VisionStatus::INVALID_PARAM;

        auto fMaxSumOfProjection = 0.f;
        nObjectEdgeResult = 0;
        for (int nObjectStart = 0; nObjectStart < matProjection.cols - nObjectSize; ++nObjectStart)
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
        Point2f     ptCtr;
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

int VisionAlgorithm::_autoThreshold(const cv::Mat &mat)
{
    const int sbins = 32;
    int channels[] = { 0 };
    int histSize[] = { sbins };

    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, PR_MAX_GRAY_LEVEL };
    const float* ranges[] = { sranges };

    MatND hist;
    calcHist(&mat, 1, channels, Mat(), // do not use mask
        hist, 1, histSize, ranges,
        true, // the histogram is uniform
        false);  

    if (Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE)   {
        double maxVal = 0;
        minMaxLoc ( hist, 0, &maxVal, 0, 0 );
        const int scale = 10;
        const int nDisplayRatio = 3;
        Mat histImg = Mat::zeros ( ToInt32( maxVal / nDisplayRatio ), sbins * scale, CV_8UC3);

        for (int s = 0; s < sbins; s++)
        {
            float binVal = hist.at<float>(s, 0);
            int intensity = cvRound(binVal * 255 / maxVal);
            rectangle(histImg, Point(s*scale, 0),
                Point((s + 1)*scale - 1, ToInt32 ( binVal / nDisplayRatio ) ),
                Scalar::all(intensity),
                CV_FILLED);
        }
        showImage("Image Histogram", histImg );
    }

    vector<float> vecH;
    float fTotalPixel = (float)mat.rows * mat.cols;
    for (int i = 0; i < sbins; ++i)
        vecH.push_back(hist.at <float>(i, 0) / fTotalPixel);

    float fSigmaSquareMin = 100000.f;
    int nResultT = 0;
    for (int T = sbins - 1; T >= 0; --T)  {
        float fSigmaSquare = 0.f;
        if (vecH[T] <= 0.0001f)
            continue;
        float fMu1 = 0.f, fMu2 = 0.f;
        float fSumH_1 = 0.f, fSumH_2 = 0.f;
        float fSumHxL_1 = 0.f, fSumHxL_2 = 0.f;

        for (int i = 0; i < T; ++i)  {
            fSumH_1 += vecH[i];
            fSumHxL_1 += i * vecH[i];
        }

        for (int i = T; i < sbins; ++i)  {
            fSumH_2 += vecH[i];
            fSumHxL_2 += i * vecH[i];
        }

        if (fSumH_1 <= 0.00001f || fSumH_2 < 0.000001f)
            continue;

        fMu1 = fSumHxL_1 / fSumH_1;
        fMu2 = fSumHxL_2 / fSumH_2;

        for (int i = 0; i < sbins; ++i)  {
            if (i < T)    {
                fSigmaSquare += (i - fMu1) * (i - fMu1) * vecH[i];
            }
            else
            {
                fSigmaSquare += (i - fMu2) * (i - fMu2) * vecH[i];
            }
        }
        if (fSigmaSquare < fSigmaSquareMin)   {
            fSigmaSquareMin = fSigmaSquare;
            nResultT = T;
        }
    }

    int nThreshold = nResultT * PR_MAX_GRAY_LEVEL / sbins;
    return nThreshold;
}

VisionStatus VisionAlgorithm::_findDeviceElectrode(const cv::Mat &matDeviceROI, VectorOfPoint &vecElectrodePos)
{
    cv::Mat canny_output;
    vector<vector<cv::Point> > vecContours, vecContourAfterFilter;
    vector<cv::Vec4i> hierarchy;
    vecElectrodePos.clear();

    /// Detect edges using canny
    Canny(matDeviceROI, canny_output, 100, 255, 3);
    if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )   {
        imshow("Canny Result", canny_output );
        cv::waitKey(0);
    }

    /// Find contours
    findContours(canny_output, vecContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    /// Filter contours
    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
    std::vector<AOI_COUNTOUR> vecAoiContour;
    int index = 0;
    for ( auto contour : vecContours )
    {
        if ( Config::GetInstance()->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE )   {
            RNG rng(12345);
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            drawContours(drawing, vecContours, index ++ , color, 2, 8, hierarchy, 0, Point());
        }
        vector<Point> approx;
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

VisionStatus VisionAlgorithm::_findDeviceEdge(const cv::Mat &matDeviceROI, const cv::Size2f &size, Rect &rectDeviceResult)
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
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstLrnDeviceCmd = %d, pstLrnDeivceRpy = %d", pstLrnDeviceCmd, pstLrnDeivceRpy );
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
        _snprintf(charrMsg, sizeof(charrMsg), "Input is invalid, pstInspDeviceCmd = %d, pstInspDeivceRpy = %d", pstInspDeviceCmd, pstInspDeivceRpy );
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
        cv::blur ( matGray, matBlur, cv::Size(2, 2) );
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

	float fDistanceOfLine = CalcUtils::distanceOf2Point<float> ( ptCrossPointWithLine1, ptCrossPointWithLine2 );
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
	float fLineLength = CalcUtils::distanceOf2Point<float>( vecPoint[nMinIndex], vecPoint[nMaxIndex] );
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

void VisionAlgorithm::showImage(String windowName, const cv::Mat &mat)
{
    static Int32 nImageCount = 0;
    String strWindowName = windowName + "_" + std::to_string(nImageCount ++ );
    cv::namedWindow ( strWindowName );
    cv::imshow( strWindowName, mat );
    cv::waitKey(0);
    cv::destroyWindow ( strWindowName );
}

VisionStatus VisionAlgorithm::runLogCase(const std::string &strPath)
{
    if ( strPath.length() < 2 )
        return VisionStatus::INVALID_PARAM;

    if (  ! bfs::exists ( strPath ) )
        return VisionStatus::PATH_NOT_EXIST;

    VisionStatus enStatus = VisionStatus::OK;
    auto strLocalPath = strPath;
    if ( strPath[ strPath.length() - 1 ] != '\\')
        strLocalPath.append("\\");

    auto pos = strLocalPath.rfind ( '\\', strLocalPath.length() - 2 );
    if ( String::npos == pos )
        return VisionStatus::INVALID_PARAM;

    auto folderName = strLocalPath.substr ( pos + 1 );
    pos = folderName.find('_');
    auto folderPrefix = folderName.substr(0, pos);

    std::shared_ptr<LogCase> pLogCase = nullptr;
    if ( LogCaseLrnTmpl::FOLDER_PREFIX == folderPrefix )
        pLogCase = std::make_shared <LogCaseLrnTmpl>( strLocalPath, true );
    
    if ( nullptr != pLogCase )
        enStatus = pLogCase->RunLogCase();
    else
        enStatus = VisionStatus::INVALID_LOGCASE;
    return enStatus;
}

}
}