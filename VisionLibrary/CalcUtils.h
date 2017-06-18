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
    static float distanceOf2Point(const cv::Point_<T> &pt1, const cv::Point_<T> &pt2)
    {
        float fOffsetX = ( float ) pt1.x - pt2.x;
        float fOffsetY = ( float ) pt1.y - pt2.y;
        float fDistance = sqrt(fOffsetX * fOffsetX + fOffsetY * fOffsetY);
        return fDistance;
    }

    template<typename T>
    static double calcSlopeDegree(const cv::Point_<T> &pt1, const cv::Point_<T> &pt2)
    {
        return radian2Degree ( atan2f ( ToFloat( pt2.y - pt1.y ), ToFloat ( pt2.x - pt1.x ) ) );
    }

    template<typename T>
    static cv::Point_<T>midOf2Points(const cv::Point_<T> &pt1, const cv::Point_<T> &pt2)
    {
        return cv::Point_<T>( ( pt1.x + pt2.x ) / 2.f, ( pt1.y + pt2.y ) / 2.f );
    }

    template<typename T>
    static cv::Rect_<T> scaleRect(const cv::Rect_<T> &rectInput, float fScale)
    {
        return cv::Rect_<T> (cv::Point_<T>(rectInput.x + ( 1.f - fScale ) * rectInput.width, rectInput.y + ( 1.f - fScale ) * rectInput.height ),
            cv::Point_<T>(rectInput.x + ( 1.f + fScale ) * rectInput.width, rectInput.y + ( 1.f + fScale ) * rectInput.height ) );
    }

    template<typename T>
    static cv::Rect_<T> resizeRect(const cv::Rect_<T> &rectInput, cv::Size2f szNew)
    {
        return cv::Rect_<T> (rectInput.x + rectInput.width / 2.f - szNew.width / 2, rectInput.y + rectInput.height / 2.f  - szNew.height / 2,
            szNew.width, szNew.height );
    }

    template<typename T>
    static cv::Rect_<T> Rect(const cv::Rect_<T> &rectInput, float fScale)
    {
        return cv::Rect_<T> (cv::Point_<T>(rectInput.x + ( 1.f - fScale ) * rectInput.width, rectInput.y + ( 1.f - fScale ) * rectInput.height ),
            cv::Point_<T>(rectInput.x + ( 1.f + fScale ) * rectInput.width, rectInput.y + ( 1.f + fScale ) * rectInput.height ) );
    }

    template<typename T>
    static double MatSquareSum(const cv::Mat &mat)
    {
        double dSum = 0;
        for (int row = 0; row < mat.rows; ++row)
            for (int col = 0; col < mat.cols; ++col)
            {
                float fValue = mat.at<T>(row, col);
                dSum += fValue * fValue;
            }
        return dSum;
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
    static cv::Mat genMaskByValue(const cv::Mat &matInputImg, const _Tp value) {
        cv::Mat matResultImg;
        cv::compare ( matInputImg, value, matResultImg, cv::CmpTypes::CMP_EQ );
        return matResultImg;
    }

    static double radian2Degree( double dRadian );
    static double degree2Radian( double dDegree );
    static float ptDisToLine(const cv::Point2f &ptInput, bool bReversedFit, float fSlope, float fIntercept );
    static PR_Line2f calcEndPointOfLine( const VectorOfPoint &vecPoint, bool bReversedFit, float fSlope, float fIntercept );
    static PR_Line2f calcEndPointOfLine( const ListOfPoint &listPoint, bool bReversedFit, float fSlope, float fIntercept );
    static cv::Point2f lineIntersect(float fSlope1, float fIntercept1, float fSlope2, float fIntercept2);
    static float lineSlope(const PR_Line2f &line);
    static VectorOfPoint getCornerOfRotatedRect(const cv::RotatedRect &rotatedRect);
    static float guassianValue(float ssq, float x );
    static cv::Mat generateGuassinDiffKernel ( int nOneSideWidth, float ssq );
    static void filter2D_Conv(cv::InputArray src, cv::OutputArray dst, int ddepth,
                   cv::InputArray kernel, cv::Point anchor = cv::Point(-1,-1),
                   double delta = 0, int borderType = cv::BORDER_DEFAULT );

    template<typename _Tp>
    static cv::Point2f warpPoint(const cv::Mat &matWarp, const cv::Point2f &ptInput)
    {
        cv::Mat matPoint = cv::Mat::zeros(3, 1, matWarp.type());
        matPoint.at<_Tp>(0, 0) = ptInput.x;
        matPoint.at<_Tp>(1, 0) = ptInput.y;
        matPoint.at<_Tp>(2, 0) = 1.f;
        cv::Mat matResultImg = matWarp * matPoint;
        return cv::Point_<_Tp> ( ToFloat ( matResultImg.at<_Tp>(0, 0) ),  ToFloat ( matResultImg.at<_Tp>(1, 0) ) );
    }

    template<typename _Tp>
    static VectorOfPoint2f warpRect(const cv::Mat &matWarp, const cv::Rect2f &rectInput) {
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
    if ( matInputImg.isContinuous() ) {
        for ( int row = 0; row < matInputImg.rows; ++ row ) {
            std::vector<_Tp> vecRow;
            int nRowStart = row * matInputImg.cols;
            vecRow.assign ( (_Tp *)matInputImg.datastart + nRowStart, (_Tp *)matInputImg.datastart + nRowStart + matInputImg.cols );
            vecVecArray.push_back(vecRow);
        }
    }else {
        for ( int row = 0; row < matInputImg.rows; ++ row ) {
            std::vector<_Tp> vecRow;
            vecRow.assign((_Tp*)matInputImg.ptr<uchar>(row), (_Tp*)matInputImg.ptr<uchar>(row) + matInputImg.cols);
            vecVecArray.push_back(vecRow);
        }
    }
    return vecVecArray;
}
};

}
}
#endif