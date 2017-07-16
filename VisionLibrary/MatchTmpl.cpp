#include "MatchTmpl.h"

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

        std::vector<size_t> vecMaskOfTmplIndex;
        size_t nTotalIndex = matMaskOfTmpl.rows * matMaskOfTmpl.cols;
        vecMaskOfTmplIndex.reserve( nTotalIndex / 5 );
        for ( size_t index = 0; index < nTotalIndex; index += 3 ) {
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
        std::vector<size_t> vecMaskOfTmplIndex;
        vecMaskOfTmplIndex.reserve( vecMaskOfTmpl.size() / 5 );
        for ( size_t index = 0; index < vecMaskOfTmpl.size(); index += 1 )
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
        cv::minMaxLoc(matResult, &minVal, nullptr, &minLoc, nullptr );
        ptResult.x += (minLoc.x + nStartX);
        ptResult.y += (minLoc.y + nStartY);
        return ptResult;
    }
}

}
}