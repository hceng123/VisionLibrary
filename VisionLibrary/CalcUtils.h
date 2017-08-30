#ifndef _CALC_UTILS_H_
#define _CALC_UTILS_H_

#include "BaseType.h"
#include "VisionHeader.h"
#include <math.h>

namespace AOI
{
namespace Vision
{

class CalcUtils
{
private:
    CalcUtils();
    CalcUtils(CalcUtils const &);
    CalcUtils(CalcUtils const &&);
    CalcUtils &operator=(CalcUtils const &);        
    ~CalcUtils();
public:
    template<typename T>
    static inline float distanceOf2Point(const cv::Point_<T> &pt1, const cv::Point_<T> &pt2)
    {
        float fOffsetX = ( float ) pt1.x - pt2.x;
        float fOffsetY = ( float ) pt1.y - pt2.y;
        float fDistance = sqrt(fOffsetX * fOffsetX + fOffsetY * fOffsetY);
        return fDistance;
    }

    template<typename T>
    static inline double calcSlopeDegree(const cv::Point_<T> &pt1, const cv::Point_<T> &pt2)
    {
        return radian2Degree ( atan2f ( ToFloat( pt2.y - pt1.y ), ToFloat ( pt2.x - pt1.x ) ) );
    }

    template<typename T>
    static inline cv::Point_<T>midOf2Points(const cv::Point_<T> &pt1, const cv::Point_<T> &pt2)
    {
        return cv::Point_<T>( ( pt1.x + pt2.x ) / 2.f, ( pt1.y + pt2.y ) / 2.f );
    }

    template<typename T>
    static inline cv::Rect_<T> scaleRect(const cv::Rect_<T> &rectInput, float fScale)
    {
        return cv::Rect_<T> (cv::Point_<T>(rectInput.x + ( 1.f - fScale ) * rectInput.width, rectInput.y + ( 1.f - fScale ) * rectInput.height ),
            cv::Point_<T>(rectInput.x + ( 1.f + fScale ) * rectInput.width, rectInput.y + ( 1.f + fScale ) * rectInput.height ) );
    }

    template<typename T>
    static inline cv::Rect_<T> resizeRect(const cv::Rect_<T> &rectInput, cv::Size_<T> szNew)
    {
        return cv::Rect_<T> (rectInput.x + rectInput.width / 2.f - szNew.width / 2, rectInput.y + rectInput.height / 2.f  - szNew.height / 2,
            szNew.width, szNew.height );
    }

    template<typename T>
    static inline cv::Rect_<T> Rect(const cv::Rect_<T> &rectInput, float fScale)
    {
        return cv::Rect_<T> (cv::Point_<T>(rectInput.x + ( 1.f - fScale ) * rectInput.width, rectInput.y + ( 1.f - fScale ) * rectInput.height ),
            cv::Point_<T>(rectInput.x + ( 1.f + fScale ) * rectInput.width, rectInput.y + ( 1.f + fScale ) * rectInput.height ) );
    }

    //Calculate deviation in one pass. The algorithm is get from https://www.strchr.com/standard_deviation_in_one_pass
    template<typename T>
    static double calcStdDeviation(std::vector<T> vecValue)
    {
        if (vecValue.empty())
            return 0.0;
        double sum = 0;
        double sq_sum = 0;
        for (const auto value : vecValue) {
            sum += value;
            sq_sum += value * value;
        }
        auto n = vecValue.size();
        double mean = sum / n;
        double variance = sq_sum / n - mean * mean;
        return sqrt(variance);
    }

    template<typename _Tp>
    static inline cv::Mat genMaskByValue(const cv::Mat &matInputImg, const _Tp value) {
        cv::Mat matResultImg;
        cv::compare ( matInputImg, value, matResultImg, cv::CmpTypes::CMP_EQ );
        return matResultImg;
    }    

    template<typename _Tp>
    static inline cv::Point2f warpPoint(const cv::Mat &matWarp, const cv::Point2f &ptInput) {
        cv::Mat matPoint = cv::Mat::zeros(3, 1, matWarp.type());
        matPoint.at<_Tp>(0, 0) = ptInput.x;
        matPoint.at<_Tp>(1, 0) = ptInput.y;
        matPoint.at<_Tp>(2, 0) = 1.f;
        cv::Mat matResultImg = matWarp * matPoint;
        return cv::Point_<_Tp> ( ToFloat ( matResultImg.at<_Tp>(0, 0) ), ToFloat ( matResultImg.at<_Tp>(1, 0) ) );
    }

    template<typename _Tp>
    static inline VectorOfPoint2f warpRect(const cv::Mat &matWarp, const cv::Rect2f &rectInput) {
        VectorOfPoint2f vecPoints;
        vecPoints.push_back ( warpPoint<_Tp>( matWarp, rectInput.tl() ) );
        vecPoints.push_back ( warpPoint<_Tp>( matWarp, cv::Point2f(rectInput.x + rectInput.width, rectInput.y ) ) );
        vecPoints.push_back ( warpPoint<_Tp>( matWarp, rectInput.br() ) );
        vecPoints.push_back ( warpPoint<_Tp>( matWarp, cv::Point2f(rectInput.x, rectInput.y + rectInput.height ) ) );
        return vecPoints;
    }

    template<typename _Tp>
    static inline std::vector<std::vector<_Tp>> matToVector(const cv::Mat &matInputImg) {
        std::vector<std::vector<_Tp>> vecVecArray;
        if ( matInputImg.isContinuous () ) {
            for ( int row = 0; row < matInputImg.rows; ++row ) {
                std::vector<_Tp> vecRow;
                int nRowStart = row * matInputImg.cols;
                vecRow.assign ( (_Tp *)matInputImg.datastart + nRowStart, (_Tp *)matInputImg.datastart + nRowStart + matInputImg.cols );
                vecVecArray.push_back ( vecRow );
            }
        }else {
            for ( int row = 0; row < matInputImg.rows; ++row ) {
                std::vector<_Tp> vecRow;
                vecRow.assign ( (_Tp*)matInputImg.ptr<uchar> ( row ), (_Tp*)matInputImg.ptr<uchar> ( row ) +matInputImg.cols );
                vecVecArray.push_back ( vecRow );
            }
        }
        return vecVecArray;
    }

    template<typename _Tp>
    static inline cv::Mat cumsum ( const cv::Mat &matInput, int nDimension ) {
        cv::Mat matResult = matInput.clone ();
        if( nDimension == 1 )
            cv::transpose ( matResult, matResult );
        for( int row = 0; row < matResult.rows; ++row )
        for( int col = 0; col < matResult.cols; ++col ) {
            if( col != 0 )
                matResult.at<_Tp> ( row, col ) += matResult.at<_Tp> ( row, col - 1 );
        }
        if( nDimension == 1 )
            cv::transpose ( matResult, matResult );
        return matResult;
    }

    template<typename _tp>
    static void meshgrid ( float xStart, float xInterval, float xEnd, float yStart, float yInterval, float yEnd, cv::Mat &matX, cv::Mat &matY ) {
        std::vector<_tp> vectorX, vectorY;
        _tp xValue = xStart;
        while( xValue <= xEnd )    {
            vectorX.push_back ( xValue );
            xValue += xInterval;
        }

        _tp yValue = yStart;
        while( yValue <= yEnd )    {
            vectorY.push_back ( yValue );
            yValue += yInterval;
        }
        cv::Mat matCol ( vectorX );
        matCol = matCol.reshape ( 1, 1 );

        cv::Mat matRow ( vectorY );
        matRow = matRow.reshape ( 1, ToInt32 ( vectorY.size() ) );
        matX = cv::repeat ( matCol, ToInt32 ( vectorY.size() ), 1 );
        matY = cv::repeat ( matRow, 1, ToInt32 ( vectorX.size () ) );
    }

    template<typename T>
    static inline cv::Mat floor ( const cv::Mat &matInput ) {
        cv::Mat matResult = matInput.clone ();
        for( int row = 0; row < matInput.rows; ++row )
        for( int col = 0; col < matInput.cols; ++col )
            matResult.at<T> ( row, col ) = std::floor ( matResult.at<T> ( row, col ) );
        return matResult;
    }

    static double radian2Degree( double dRadian );
    static double degree2Radian( double dDegree );
    static float ptDisToLine(const cv::Point2f &ptInput, bool bReversedFit, float fSlope, float fIntercept );
    static PR_Line2f calcEndPointOfLine( const VectorOfPoint &vecPoint, bool bReversedFit, float fSlope, float fIntercept );
    static PR_Line2f calcEndPointOfLine( const ListOfPoint &listPoint, bool bReversedFit, float fSlope, float fIntercept );
    static cv::Point2f lineIntersect(float fSlope1, float fIntercept1, float fSlope2, float fIntercept2);
    static float lineSlope(const PR_Line2f &line);
    static void lineSlopeIntercept(const PR_Line2f &line, float &fSlope, float &fIntercept);
    static VectorOfPoint getCornerOfRotatedRect(const cv::RotatedRect &rotatedRect);
    static float guassianValue(float ssq, float x );
    static cv::Mat generateGuassinDiffKernel ( int nOneSideWidth, float ssq );
    static void filter2D_Conv(cv::InputArray src, cv::OutputArray dst, int ddepth,
                   cv::InputArray kernel, cv::Point anchor = cv::Point(-1,-1),
                   double delta = 0, int borderType = cv::BORDER_DEFAULT );
    static float calcPointToContourDist(const cv::Point &ptInput, const VectorOfPoint &contour, cv::Point &ptResult );
    static cv::Point2f getContourCtr(const VectorOfPoint &contour);
    static cv::Mat diff ( const cv::Mat &matInput, int nRecersiveTime, int nDimension );
};

}
}
#endif