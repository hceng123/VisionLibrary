#include "MatchTmpl.h"
#include "opencv2/video.hpp"
#include "CalcUtils.h"

namespace AOI
{
namespace Vision
{

MatchTmpl::MatchTmpl ()
{
}

MatchTmpl::~MatchTmpl ()
{
}

/*static*/ cv::Point MatchTmpl::matchByRecursionEdge(const cv::Mat &matInput, const cv::Mat &matTmpl, const cv::Mat &matMaskOfTmpl ) {
    if ( matTmpl.cols * matTmpl.rows > 1e4 ) {
        cv::Mat matInputPyr, matTmplPyr, matMaskOfTmplPyr;

        auto interpFlag = cv::InterpolationFlags::INTER_AREA;
        cv::resize ( matInput, matInputPyr, cv::Size(), 0.5, 0.5, interpFlag );
        cv::resize ( matTmpl, matTmplPyr, cv::Size(), 0.5, 0.5, interpFlag );
        cv::resize ( matMaskOfTmpl, matMaskOfTmplPyr, cv::Size(), 0.5, 0.5, interpFlag );

        cv::Point ptResult = matchByRecursionEdge ( matInputPyr, matTmplPyr, matMaskOfTmplPyr );
        ptResult.x *= 2;
        ptResult.y *= 2;

        int nStartX = MAX ( -ptResult.x, -2 );
        int nStartY = MAX ( -ptResult.y, -2 );
        int nEndX = MIN ( matInput.cols - ptResult.x - matTmpl.cols, 2 );
        int nEndY = MIN ( matInput.rows - ptResult.y - matTmpl.rows, 2 );
        cv::Mat matResult = cv::Mat::zeros ( nEndY - nStartY + 1, nEndX - nStartX + 1, CV_32FC1 );
        int nRow = 0, nCol = 0;

        std::vector<int> vecMaskOfTmplIndex;
        int nTotalIndex = matMaskOfTmpl.rows * matMaskOfTmpl.cols;
        vecMaskOfTmplIndex.reserve( nTotalIndex / 5 );
        for ( int index = 0; index < nTotalIndex; index += 3 ) {
            if ( matMaskOfTmpl.at<uchar>( index ) > 0 )
                vecMaskOfTmplIndex.push_back ( index );
        }

        float fSum = 0.f, fDiff = 0.f;
        for ( int y = nStartY; y <= nEndY; ++ y ) {
            nCol = 0;
            for ( int x = nStartX; x <= nEndX; ++ x ) {
                cv::Mat matROI( matInput, cv::Rect ( ptResult.x + x, ptResult.y + y, matTmpl.cols, matTmpl.rows ) );
                float fTotalDiffSqr = 0.f;
                for ( const auto &index : vecMaskOfTmplIndex ) {
                     auto diff = static_cast<float>( matTmpl.at<uchar>(index) ) - static_cast<float>( matROI.at<uchar>(index) );
                     fTotalDiffSqr += diff * diff;
                }
                matResult.at<float>(nRow, nCol) = fTotalDiffSqr;
                ++ nCol;
            }
            ++ nRow;
        }
        cv::Point minLoc;
        double minVal = 0., maxVal = 0.f;
        cv::minMaxLoc(matResult, &minVal, nullptr, &minLoc, nullptr );
        ptResult.x += (minLoc.x + nStartX);
        ptResult.y += (minLoc.y + nStartY);
        return ptResult;
    }else {
        const int nn = 3;
        int nTotalCol = ToInt32 ( floor ( ( matInput.cols - matTmpl.cols ) / nn ) ) + 1;
        int nTotalRow = ToInt32 ( floor ( ( matInput.rows - matTmpl.rows ) / nn ) ) + 1;
        cv::Mat matResult = cv::Mat::zeros ( nTotalRow, nTotalCol, CV_32FC1 );
        int nRow = 0, nCol = 0;
        
        std::vector<uchar> vecMaskOfTmpl = matMaskOfTmpl.reshape(1, 1);
        std::vector<uchar> vecTmpl = matTmpl.reshape(1, 1);
        std::vector<int> vecMaskOfTmplIndex;
        vecMaskOfTmplIndex.reserve( vecMaskOfTmpl.size() / 5 );
        for ( int index = 0; index < ToInt32 ( vecMaskOfTmpl.size() ); index += 1 )
            if ( vecMaskOfTmpl[index] > 0 )
                vecMaskOfTmplIndex.push_back ( index );

        for ( int i = 0; i <= matInput.rows - matTmpl.rows; i += nn ) {
            nCol = 0;
            for ( int j = 0; j <= matInput.cols - matTmpl.cols; j += nn ) {
                cv::Mat matROI( matInput, cv::Rect ( j, i, matTmpl.cols, matTmpl.rows ) );
                float fTotalDiffSqr = 0.f;
                for ( const auto &index : vecMaskOfTmplIndex ) {
                     auto diff = static_cast<float>( vecTmpl[index] ) - static_cast<float>( matROI.at<uchar>(index) );
                     fTotalDiffSqr += diff * diff;
                }
                matResult.at<float>(nRow, nCol) = fTotalDiffSqr;
                ++ nCol;
            }
            ++ nRow;
        }
        cv::Point minLoc;
        double minVal = 0., maxVal = 0.f;
        cv::minMaxLoc ( matResult, &minVal, nullptr, &minLoc, nullptr );
        cv::Point ptResult = minLoc;
        ptResult.x *= nn;
        ptResult.y *= nn;

        //Fine search again
        int nStartX = MAX ( -ptResult.x, 1 - nn );
        int nStartY = MAX ( -ptResult.y, 1 - nn );
        int nEndX = MIN ( matInput.cols - ptResult.x - matTmpl.cols, nn - 1 );
        int nEndY = MIN ( matInput.rows - ptResult.y - matTmpl.rows, nn - 1 );
        matResult = cv::Mat::zeros ( nEndY - nStartY + 1, nEndX - nStartX + 1, CV_32FC1 );
        nRow = 0;
        for ( int y = nStartY; y <= nEndY; ++ y ) {
            nCol = 0;
            for ( int x = nStartX; x <= nEndX; ++ x ) {
                cv::Mat matROI( matInput, cv::Rect ( ptResult.x + x, ptResult.y + y, matTmpl.cols, matTmpl.rows ) );
                float fTotalDiffSqr = 0.f;
                for ( const auto & index : vecMaskOfTmplIndex ) {
                     auto diff = static_cast<float>( vecTmpl[index] ) - static_cast<float>( matROI.at<uchar>(index) );
                     fTotalDiffSqr += diff * diff;
                }
                matResult.at<float>(nRow, nCol) = fTotalDiffSqr;
                ++ nCol;
            }
            ++ nRow;
        }
        cv::minMaxLoc ( matResult, &minVal, nullptr, &minLoc, nullptr );
        ptResult.x += (minLoc.x + nStartX);
        ptResult.y += (minLoc.y + nStartY);
        return ptResult;
    }
}

/*static*/ VisionStatus MatchTmpl::refineSrchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation, float &fCorrelation)
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
    fRotation = ToFloat ( CalcUtils::radian2Degree ( asin ( matWarp.at<float>( 1, 0 ) ) ) );

    return VisionStatus::OK;
}

/*static*/ VisionStatus MatchTmpl::matchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, bool bSubPixelRefine, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation, float &fCorrelation)
{
    ptResult = myMatchTemplate ( mat, matTmpl );
    fRotation = 0.f;
    if ( bSubPixelRefine ) {
        try {
            refineSrchTemplate ( mat, matTmpl, enMotion, ptResult, fRotation, fCorrelation );
        }
        catch (std::exception) {
            return VisionStatus::OPENCV_EXCEPTION;
        }
    }

    ptResult.x += (float)( matTmpl.cols / 2 + 0.5f );
    ptResult.y += (float)( matTmpl.rows / 2 + 0.5f );
    return VisionStatus::OK;
}

/*static*/ cv::Point MatchTmpl::myMatchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl)
{
    cv::Mat matResult;
    const int match_method = CV_TM_SQDIFF;

    /// Create the result matrix
    int result_cols = mat.cols - matTmpl.cols + 1;
    int result_rows = mat.rows - matTmpl.rows + 1;

    matResult.create(result_rows, result_cols, CV_32FC1);

    /// Do the Matching and Normalize
    cv::matchTemplate ( mat, matTmpl, matResult, match_method );

    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal;
    cv::Point minLoc, maxLoc, matchLoc;

    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if ( match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED ) {
        cv::minMaxLoc(matResult, &minVal, nullptr, &minLoc, nullptr );
        matchLoc = minLoc;
    }
    else {
        cv::minMaxLoc(matResult, nullptr, &maxVal, nullptr, &maxLoc );
        matchLoc = maxLoc;
    }

    return matchLoc;
}

/*static*/ cv::Point MatchTmpl::matchTemplateRecursive(const cv::Mat &matInput, const cv::Mat &matTmpl ) {
    if ( matTmpl.cols * matTmpl.rows > 4e4 ) {
        cv::Mat matInputPyr, matTmplPyr;
        auto interpFlag = cv::InterpolationFlags::INTER_AREA;
        cv::resize( matInput, matInputPyr, cv::Size(), 0.5, 0.5, interpFlag );
        cv::resize( matTmpl,  matTmplPyr,  cv::Size(), 0.5, 0.5, interpFlag );
        //cv::pyrDown ( matInput, matInputPyr );
        //cv::pyrDown ( matTmpl,  matTmplPyr );

        cv::Point ptResult = matchTemplateRecursive ( matInputPyr, matTmplPyr );
        ptResult.x *= 2;
        ptResult.y *= 2;

        int nStartX = MAX ( -ptResult.x, -2 );
        int nStartY = MAX ( -ptResult.y, -2 );
        int nEndX = MIN ( matInput.cols - ptResult.x - matTmpl.cols, 2 );
        int nEndY = MIN ( matInput.rows - ptResult.y - matTmpl.rows, 2 );
        cv::Mat matROI ( matInput, cv::Rect ( ptResult.x + nStartX, ptResult.y + nStartY, matTmpl.cols + ( nEndX - nStartX ), matTmpl.rows + ( nEndY - nStartY ) ) );
        
        auto subResult = myMatchTemplate ( matROI, matTmpl );
        ptResult.x += (subResult.x + nStartX);
        ptResult.y += (subResult.y + nStartY);
        return ptResult;
    }else {
        return myMatchTemplate ( matInput, matTmpl );
    }
}

}
}