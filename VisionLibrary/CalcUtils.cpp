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

/*static*/ float CalcUtils::ptDisToLine(const cv::Point2f &ptInput, bool bReversedFit, float fSlope, float fIntercept )
{
    float distance = 0.f;
    if ( bReversedFit ) {
        if (fabs(fSlope) < 0.0001) {
            distance = ptInput.x - fIntercept;
        }else {
            float fSlopeOfPerpendicularLine = -1.f / fSlope;
            float fInterceptOfPerpendicularLine = ptInput.x - fSlopeOfPerpendicularLine * ptInput.y;
            cv::Point2f ptCrossPoint;
            ptCrossPoint.y = (fInterceptOfPerpendicularLine - fIntercept) / (fSlope - fSlopeOfPerpendicularLine);
            ptCrossPoint.x = fSlope * ptCrossPoint.y + fIntercept;
            distance = sqrt((ptInput.x - ptCrossPoint.x) * (ptInput.x - ptCrossPoint.x) + (ptInput.y - ptCrossPoint.y) * (ptInput.y - ptCrossPoint.y));
            if (ptInput.x < ptCrossPoint.x)
                distance = -distance;
        }
    }
    else
    {
        if (fabs(fSlope) < 0.0001) {
            distance = ptInput.y - fIntercept;
        }
        else
        {
            float fSlopeOfPerpendicularLine = -1.f / fSlope;
            float fInterceptOfPerpendicularLine = ptInput.y - fSlopeOfPerpendicularLine * ptInput.x;
            cv::Point2f ptCrossPoint;
            ptCrossPoint.x = (fInterceptOfPerpendicularLine - fIntercept) / (fSlope - fSlopeOfPerpendicularLine);
            ptCrossPoint.y = fSlope * ptCrossPoint.x + fIntercept;
            distance = sqrt((ptInput.x - ptCrossPoint.x) * (ptInput.x - ptCrossPoint.x) + (ptInput.y - ptCrossPoint.y) * (ptInput.y - ptCrossPoint.y));
            if (ptInput.y < ptCrossPoint.y)
                distance = -distance;
        }
    }
    
    return distance;
}

/*static*/ PR_Line2f CalcUtils::calcEndPointOfLine( const VectorOfPoint &vecPoint, bool bReversedFit, float fSlope, float fIntercept )
{
    float fMinX = 10000.f, fMinY = 10000.f, fMaxX = -10000.f, fMaxY = -10000.f;
    for ( const auto &point : vecPoint )    {
        cv::Point2f pt2f(point);
        if ( pt2f.x < fMinX ) fMinX = pt2f.x;
        if ( pt2f.x > fMaxX ) fMaxX = pt2f.x;
        if ( pt2f.y < fMinY ) fMinY = pt2f.y;
        if ( pt2f.y > fMaxY ) fMaxY = pt2f.y;
    }
    PR_Line2f line;
    if ( bReversedFit ) {
        line.pt1.y = fMinY;
        line.pt1.x = line.pt1.y * fSlope + fIntercept;
        line.pt2.y = fMaxY;
        line.pt2.x = line.pt2.y * fSlope + fIntercept;
    }else {
        line.pt1.x = fMinX;
        line.pt1.y = fSlope * line.pt1.x + fIntercept;
        line.pt2.x = fMaxX;
        line.pt2.y = fSlope * line.pt2.x + fIntercept;
    }
    
    return line;
}

/*static*/ PR_Line2f CalcUtils::calcEndPointOfLine( const ListOfPoint &listPoint, bool bReversedFit, float fSlope, float fIntercept )
{
    float fMinX = 10000.f, fMinY = 10000.f, fMaxX = -10000.f, fMaxY = -10000.f;
    for ( const auto &point : listPoint )    {
        cv::Point2f pt2f(point);
        if ( pt2f.x < fMinX ) fMinX = pt2f.x;
        if ( pt2f.x > fMaxX ) fMaxX = pt2f.x;
        if ( pt2f.y < fMinY ) fMinY = pt2f.y;
        if ( pt2f.y > fMaxY ) fMaxY = pt2f.y;
    }
    PR_Line2f line;
    if ( bReversedFit ) {
        line.pt1.y = fMinY;
        line.pt1.x = line.pt1.y * fSlope + fIntercept;
        line.pt2.y = fMaxY;
        line.pt2.x = line.pt2.y * fSlope + fIntercept;
    }else {
        line.pt1.x = fMinX;
        line.pt1.y = fSlope * line.pt1.x + fIntercept;
        line.pt2.x = fMaxX;
        line.pt2.y = fSlope * line.pt2.x + fIntercept;
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

    cv::Mat matResultImg;
    if ( cv::solve(A, B, matResultImg ) )   {
        ptResult.x = matResultImg.at<float>(0, 0);
        ptResult.y = matResultImg.at<float>(1, 0);
    }
    return ptResult;
}

/*static*/ float CalcUtils::lineSlope(const PR_Line2f &line) {
    return ( line.pt2.y - line.pt1.y ) / ( line.pt2.x - line.pt1.x );
}

/*static*/ VectorOfPoint CalcUtils::getCornerOfRotatedRect(const cv::RotatedRect &rotatedRect) {
    VectorOfPoint vecPoint;
    cv::Point2f arrPt[4];
    rotatedRect.points( arrPt );
    for ( int i = 0; i < 4; ++ i )
        vecPoint.push_back ( arrPt[i] );
    return vecPoint;
}

}
}