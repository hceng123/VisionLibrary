#include "CalcUtils.h"

namespace AOI
{
namespace Vision
{

/*static*/ double CalcUtils::radian2Degree( double fRadian )
{
    return ( fRadian * 180.f / CV_PI );
}

/*static*/ double CalcUtils::degree2Radian( double fDegree )
{
    return ( fDegree * CV_PI / 180.f );
}

/*static*/ double CalcUtils::ptDisToLine(const cv::Point2f ptInput, float fSlope, float fIntercept )
{
    double distance = 0.f;
    if ( fabs ( fSlope )  < 0.0001 )   {
        distance = ptInput.y - fIntercept;
    }
    else
    {
        float fSlopeOfPerpendicularLine = - 1.f / fSlope;
        float fInterceptOfPerpendicularLine = ptInput.y - fSlopeOfPerpendicularLine * ptInput.x;
        cv::Point2f ptCrossPoint;
        ptCrossPoint.x = ( fInterceptOfPerpendicularLine - fIntercept ) / ( fSlope - fSlopeOfPerpendicularLine );
        ptCrossPoint.y = fSlope * ptCrossPoint.x + fIntercept;
        distance = sqrt ( ( ptInput.x - ptCrossPoint.x ) * ( ptInput.x - ptCrossPoint.x ) + ( ptInput.y - ptCrossPoint.y ) * ( ptInput.y - ptCrossPoint.y ) );
        if ( ptInput.y < ptCrossPoint.y )
            distance = - distance;
    }
    return distance;
}

/*static*/ PR_Line2f CalcUtils::calcEndPointOfLine( const VectorOfPoint &vecPoint, float fSlope, float fIntercept )
{
    float fMinX = 10000.f, fMinY = 10000.f, fMaxX = -10000.f, fMaxY = -10000.f;
    for ( const auto &point : vecPoint )    {
        if ( point.x < fMinX ) fMinX = point.x;
        if ( point.x > fMaxX ) fMaxX = point.x;
        if ( point.y < fMinY ) fMinY = point.y;
        if ( point.y > fMaxY ) fMaxY = point.y;
    }
    PR_Line2f line;
    if (fabs(fSlope) < 1)   {
        line.pt1.x = fMinX;
        line.pt1.y = fSlope * line.pt1.x + fIntercept;
        line.pt2.x = fMaxX;
        line.pt2.y = fSlope * line.pt2.x + fIntercept;
    }
    else
    {
        line.pt1.y = fMinY;
        line.pt1.x = line.pt1.y / fSlope - fIntercept / fSlope;
        line.pt2.y = fMaxY;
        line.pt2.x = line.pt2.y / fSlope - fIntercept / fSlope;
    }
    return line;
}

/*static*/ cv::Point2f CalcUtils::lineIntersect(float fSlope1, float fIntercept1, float fSlope2, float fIntercept2)
{
    cv::Point2f ptResult;
    if ( fabs ( fSlope1 - fSlope2 ) < 0.0001 )
        return ptResult;
    cv::Mat A(2, 2, CV_32FC1);
    cv::Mat B(2, 1, CV_32FC1);
    A.at<float>(0, 0) = fSlope1; A.at<float>(0, 1) = -1.f;
    A.at<float>(1, 0) = fSlope2; A.at<float>(1, 1) = -1.f;

    B.at<float>(0, 0) = -fIntercept1;
    B.at<float>(1, 0) = -fIntercept2;

    cv::Mat matResult;
    if ( cv::solve(A, B, matResult ) )   {
        ptResult.x = matResult.at<float>(0, 0);
        ptResult.y = matResult.at<float>(1, 0);
    }
    return ptResult;
}

/*static*/ float CalcUtils::lineSlope(const PR_Line2f &line)
{
    return ( line.pt2.y - line.pt1.y ) / ( line.pt2.x - line.pt1.x );
}

}
}