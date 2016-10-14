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
        distance = fabs ( ptInput.y - fIntercept );
    }
    else
    {
        float fSlopeOfPerpendicularLine = - 1.f / fSlope;
        float fInterceptOfPerpendicularLine = ptInput.y - fSlopeOfPerpendicularLine * ptInput.x;
        cv::Point2f ptCrossPoint;
        ptCrossPoint.x = ( fInterceptOfPerpendicularLine - fIntercept ) / ( fSlope - fSlopeOfPerpendicularLine );
        ptCrossPoint.y = fSlope * ptCrossPoint.x + fIntercept;
        distance = sqrt ( ( ptInput.x - ptCrossPoint.x ) * ( ptInput.x - ptCrossPoint.x ) + ( ptInput.y - ptCrossPoint.y ) * ( ptInput.y - ptCrossPoint.y ) );
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
    line.pt1.x = fMinX;
    line.pt1.y = fSlope * line.pt1.x + fIntercept;
    line.pt2.x = fMaxX;
    line.pt2.y = fSlope * line.pt2.x + fIntercept;
    return line;
}

}
}