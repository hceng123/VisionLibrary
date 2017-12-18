#ifndef _AOI_VISION_FITTING_H_
#define _AOI_VISION_FITTING_H_

#include <vector>
#include "opencv2/core.hpp"
#include "VisionStruct.h"

namespace AOI
{
namespace Vision
{

class Fitting
{
public:
    // This function is based on Modified Least Square Methods from Paper
    // "A Few Methods for Fitting Circles to Data".
    template<typename T>
    static cv::RotatedRect fitCircle ( const T &points ) {
        cv::RotatedRect rotatedRect;
        if( points.size () < 3 )
            return rotatedRect;

        float fAverageX = 0, fAverageY = 0;
        float n = ToFloat ( points.size () );
        for( const auto &point : points ) {
            fAverageX += point.x / n;
            fAverageY += point.y / n;
        }

        //Here offset the input points with its average value, to prevent the intermediate value becomes too big, even over
        //the range of std::numenic_limits<double>::max(). Later in the end, the average value will add back.
        VectorOfPoint2f vecPointsClone;
        vecPointsClone.reserve ( points.size() );
        for( auto &point : points ) {
            cv::Point2f ptNew ( point );
            ptNew.x -= fAverageX;
            ptNew.y -= fAverageY;
            vecPointsClone.push_back ( ptNew );
        }

        double Sx = 0., Sy = 0., Sxx = 0., Sx2 = 0., Sy2 = 0., Sxy = 0., Syy = 0., Sx3 = 0., Sy3 = 0., Sxy2 = 0., Syx2 = 0.;
        for( const auto &point : vecPointsClone ) {
            Sx += point.x;
            Sy += point.y;
            Sx2 += point.x * point.x;
            Sy2 += point.y * point.y;
            Sxy += point.x * point.y;
            Sx3 += point.x * point.x * point.x;
            Sy3 += point.y * point.y * point.y;
            Sxy2 += point.x * point.y * point.y;
            Syx2 += point.y * point.x * point.x;
        }

        double A, B, C, D, E;
        A = n * Sx2 - Sx * Sx;
        B = n * Sxy - Sx * Sy;
        C = n * Sy2 - Sy * Sy;
        D = 0.5 * (n * Sxy2 - Sx * Sy2 + n * Sx3 - Sx * Sx2);
        E = 0.5 * (n * Syx2 - Sy * Sx2 + n * Sy3 - Sy * Sy2);

        auto AC_B2 = (A * C - B * B);  // The variable name is from AC - B^2
        auto am = (D * C - B * E) / AC_B2;
        auto bm = (A * E - B * D) / AC_B2;

        double rSqureSum = 0.f;
        for( const auto &point : vecPointsClone )
            rSqureSum += sqrt ( (point.x - am) * (point.x - am) + (point.y - bm) * (point.y - bm) );

        float r = static_cast<float>(rSqureSum / n);
        rotatedRect.center.x = static_cast<float>(am) + fAverageX;
        rotatedRect.center.y = static_cast<float>(bm) + fAverageY;
        rotatedRect.size = cv::Size2f ( 2.f * r, 2.f * r );
        return rotatedRect;
    }

    static void fitLine(const VectorOfPoint &vecPoints, float &fSlope, float &fIntercept, bool reverseFit = false);
    static void fitLine(const ListOfPoint &listPoint, float &fSlope, float &fIntercept, bool reverseFit = false);
    static void fitParallelLine(const VectorOfPoint &vecPoints1,
                                const VectorOfPoint &vecPoints2,
                                float               &fSlope,
                                float               &fIntercept1,
                                float               &fIntercept2,
                                bool                 reverseFit = false);
    static void fitParallelLine(const ListOfPoint &listPoints1,
                                const ListOfPoint &listPoints2,
                                float             &fSlope,
                                float             &fIntercept1,
                                float             &fIntercept2,
                                bool               reverseFit = false);
    static void fitRect(VectorOfVectorOfPoint &vecVecPoint, 
                        float                 &fSlope1, 
                        float                 &fSlope2, 
                        std::vector<float>    &vecIntercept,
                        bool                   bLineOneReversedFit,
                        bool                   bLineTwoReversedFit);
    static void fitRect(VectorOfListOfPoint &vecListPoint, 
                        float               &fSlope1, 
                        float               &fSlope2, 
                        std::vector<float>  &vecIntercept,
                        bool                 bLineOneReversedFit,
                        bool                 bLineTwoReversedFit);
};

}
}

#endif