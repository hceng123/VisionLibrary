#include "CalcUtils.h"
#include "spline.h"
#include <fstream>

namespace AOI
{
namespace Vision
{

/*static*/ double CalcUtils::radian2Degree( double fRadian ) {
    return ( fRadian * 180.f / CV_PI );
}

/*static*/ double CalcUtils::degree2Radian( double fDegree ) {
    return ( fDegree * CV_PI / 180.f );
}

/*static*/ float CalcUtils::ptDisToLine(const cv::Point2f &ptInput, bool bReversedFit, float fSlope, float fIntercept ) {
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
    }else {
        if (fabs(fSlope) < 0.0001) {
            distance = ptInput.y - fIntercept;
        }else {
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

/*static*/ PR_Line2f CalcUtils::calcEndPointOfLine( const VectorOfPoint &vecPoint, bool bReversedFit, float fSlope, float fIntercept ) {
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

/*static*/ PR_Line2f CalcUtils::calcEndPointOfLine( const ListOfPoint &listPoint, bool bReversedFit, float fSlope, float fIntercept ) {
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

/*static*/ cv::Point2f CalcUtils::lineIntersect(float fSlope1, float fIntercept1, float fSlope2, float fIntercept2) {
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

/*static*/ void CalcUtils::lineSlopeIntercept(const PR_Line2f &line, float &fSlope, float &fIntercept) {
    fSlope = ( line.pt2.y - line.pt1.y ) / ( line.pt2.x - line.pt1.x );
    fIntercept = line.pt2.y  - fSlope * line.pt2.x;
}

/*static*/ VectorOfPoint CalcUtils::getCornerOfRotatedRect(const cv::RotatedRect &rotatedRect) {
    VectorOfPoint vecPoint;
    cv::Point2f arrPt[4];
    rotatedRect.points( arrPt );
    for ( int i = 0; i < 4; ++ i )
        vecPoint.push_back ( arrPt[i] );
    return vecPoint;
}

/*static*/ float CalcUtils::guassianValue(float ssq, float x ) {
    return ToFloat ( exp ( - ( x * x ) / ( 2.0 * ssq ) ) / ( CV_PI * ssq ) );
}

/*static*/ cv::Mat CalcUtils::generateGuassinDiffKernel(int nOneSideWidth, float ssq) {
    std::vector<float> vecGuassian;
    for ( int i = -nOneSideWidth; i <= nOneSideWidth; ++ i )
        vecGuassian.push_back ( ToFloat ( i ) * guassianValue ( ssq, ToFloat ( i ) ) );
    cv::Mat matGuassian ( vecGuassian );
    cv::transpose ( matGuassian, matGuassian );
    cv::Mat matTmp( matGuassian, cv::Range::all(), cv::Range ( 0, nOneSideWidth ) );
    float fSum = ToFloat ( fabs ( cv::sum(matTmp)[0] ) );
    matGuassian /= fSum;
    return matGuassian;
}

/*static*/ void CalcUtils::filter2D_Conv(cv::InputArray src, cv::OutputArray dst, int ddepth,
                   cv::InputArray kernel, cv::Point anchor,
                   double delta, int borderType ) {
    cv::Mat newKernel;
    const int FLIP_H_Z = -1;
    cv::flip ( kernel, newKernel, FLIP_H_Z );
    cv::Point newAnchor = anchor;
    if ( anchor.x > 0 && anchor.y >= 0 )
        newAnchor = cv::Point ( newKernel.cols - anchor.x - 1, newKernel.rows - anchor.y - 1 );
    cv::filter2D ( src, dst, ddepth, newKernel, newAnchor, delta, borderType );
}

float CalcUtils::calcPointToContourDist(const cv::Point &ptInput, const VectorOfPoint &contour, cv::Point &ptResult ) {
    cv::Point ptNearestPoint1, ptNearestPoint2;
    float fNearestDist1, fNearestDist2;
    fNearestDist1 = fNearestDist2 = std::numeric_limits<float>::max();
    for ( const auto &ptOfContour : contour ) {
        auto distance = distanceOf2Point<int> ( ptOfContour, ptInput );
        if ( distance < fNearestDist1 ) {
            fNearestDist2 = fNearestDist1;
            ptNearestPoint2 = ptNearestPoint1;

            fNearestDist1 = distance;
            ptNearestPoint1 = ptOfContour;            
        }else if ( distance < fNearestDist2 ) {
            fNearestDist2 = distance;
            ptNearestPoint2 = ptOfContour;
        }
    }
    float C = distanceOf2Point<int>( ptNearestPoint1, ptNearestPoint2 );
    if ( C < 5 ) {
        ptResult.x = ToInt32 ( ( ptNearestPoint1.x + ptNearestPoint2.x ) / 2.f + 0.5f );
        ptResult.y = ToInt32 ( ( ptNearestPoint1.y + ptNearestPoint2.y ) / 2.f + 0.5f );
        float fDistance = distanceOf2Point<int> ( ptInput, ptResult );
        return fDistance;
    }
    float A = distanceOf2Point<int> ( ptInput, ptNearestPoint1 );
    float B = distanceOf2Point<int> ( ptInput, ptNearestPoint2 );
    
    float cosOfAngle = ( A*A + C*C - B*B ) / ( 2.f * A * C );
    float angle = acos ( cosOfAngle );
    float fDistance;
    if ( angle <= CV_PI / 2 ) {
        fDistance = A * std::sin ( angle );
        float D = A * std::cos ( angle );
        float DCRatio = D / C;
        ptResult.x = ToInt32 ( ptNearestPoint1.x * (1.f - DCRatio) + ptNearestPoint2.x * DCRatio + 0.5f );
        ptResult.y = ToInt32 ( ptNearestPoint1.y * (1.f - DCRatio) + ptNearestPoint2.y * DCRatio + 0.5f );
    }else {
        fDistance = A * ToFloat ( std::sin ( CV_PI - angle ) );
        float D = A * ToFloat ( std::cos ( CV_PI - angle ) );
        float DCRatio = D / C;
        ptResult.x = ToInt32 ( ptNearestPoint1.x + D / C * ( ptNearestPoint1.x - ptNearestPoint2.x) );
        ptResult.y = ToInt32 ( ptNearestPoint1.y + D / C * ( ptNearestPoint1.y - ptNearestPoint2.y) );
    }    
    return fDistance;
}

/*static*/ cv::Point2f CalcUtils::getContourCtr(const VectorOfPoint &contour) {
    cv::Moments moment = cv::moments ( contour );
    cv::Point2f ptCenter;
    ptCenter.x = static_cast<float>(moment.m10 / moment.m00);
    ptCenter.y = static_cast<float>(moment.m01 / moment.m00);
    return ptCenter;
}

/*static*/ int CalcUtils::countOfNan(const cv::Mat &matInput) {
    cv::Mat matNan;
    cv::compare ( matInput, matInput, matNan, cv::CmpTypes::CMP_EQ );
    auto nCount = cv::countNonZero (  matNan );
    return ToInt32 ( matInput.total() ) - nCount;
}

/*static*/ void CalcUtils::findMinMaxCoord(const VectorOfPoint &vecPoints, int &xMin, int &xMax, int &yMin, int &yMax) {
    xMin = yMin = std::numeric_limits<int>::max();
    xMax = yMax = std::numeric_limits<int>::min();
    for ( const auto &point : vecPoints ) {
        if ( point.x > xMax )
            xMax = point.x;
        if ( point.x < xMin )
            xMin = point.x;
        if ( point.y > yMax )
            yMax = point.y;
        if ( point.y < yMin )
            yMin = point.y;
    }
}

/*static*/ float CalcUtils::calcFrequency(const cv::Mat &matInput) {
    assert ( matInput.rows == 1 );
    const float SAMPLE_FREQUENCY = 1.f;
    cv::Mat matRowFloat;
    matInput.convertTo ( matRowFloat, CV_32FC1 );
    int m = ToInt32 ( pow ( 2, std::floor ( log2 ( matRowFloat.cols ) ) ) );
    matRowFloat = cv::Mat ( matRowFloat, cv::Range::all (), cv::Range ( 0, m ) );
    cv::Mat matDft;
    cv::dft ( matRowFloat, matDft, cv::DftFlags::DCT_ROWS );
    auto vecVecRowFloat = CalcUtils::matToVector<float> ( matRowFloat );
    cv::Mat matAbsDft = cv::abs ( matDft );
    matAbsDft = cv::Mat ( matAbsDft, cv::Rect ( 1, 0, matAbsDft.cols - 1, 1 ) ).clone ();
    auto vecVecDft = CalcUtils::matToVector<float> ( matAbsDft );
    auto maxElement = std::max_element ( vecVecDft[0].begin (), vecVecDft[0].end () );
    int nIndex = ToInt32 ( std::distance ( vecVecDft[0].begin (), maxElement ) ) + 1;
    float fFrequency = (float)nIndex * SAMPLE_FREQUENCY / m;
    return fFrequency;
}

/*static*/ VectorOfDouble CalcUtils::interp1(const VectorOfDouble &vecX, const VectorOfDouble &vecV, const VectorOfDouble &vecXq, bool bSpine ) {
    tk::spline s;
    s.set_points ( vecX, vecV, bSpine );    // currently it is required that X is already sorted
    VectorOfDouble vecResult;
    vecResult.reserve ( vecXq.size() );
    for ( const auto xq : vecXq )
        vecResult.push_back ( s( xq ) );
    return vecResult;
}

/*static*/ void CalcUtils::saveMatToCsv(cv::Mat &matrix, std::string filename) {
    std::ofstream outputFile(filename);
    if ( ! outputFile.is_open() )
        return;
    outputFile << cv::format ( matrix, cv::Formatter::FMT_CSV ) << std::endl;
    outputFile.close();
}

/*static*/ int CalcUtils::findLineCrossPoint(const PR_Line2f &line1, const PR_Line2f &line2, cv::Point2f ptResult)
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

/*static*/ float CalcUtils::calc2LineAngle(const PR_Line2f &line1, const PR_Line2f &line2) {
    cv::Point2f ptCrossPoint;
    //If cannot find cross point of two line, which means they are parallel, so the angle is 0.
    if ( findLineCrossPoint ( line1, line2, ptCrossPoint ) != 0 ) {
        return 0.f;
    }

    cv::Point ptLine1ChoosePoint = line1.pt1;
    float A = distanceOf2Point ( ptCrossPoint, line1.pt1 );
    if ( A < 1 ) {
        A = distanceOf2Point ( ptCrossPoint, line1.pt2 );
        ptLine1ChoosePoint = line1.pt2;
    }

    cv::Point ptLine2ChoosePoint = line2.pt1;
    float B = distanceOf2Point ( ptCrossPoint, line2.pt1 );
    if ( B < 1 ) {
        B = distanceOf2Point ( ptCrossPoint, line2.pt2 );
        ptLine2ChoosePoint = line2.pt2;
    }

    float C = distanceOf2Point ( ptLine1ChoosePoint, ptLine2ChoosePoint );
    //Use the cos rules to calculate the cos of angle.
    float fCos = ( A*A + B*B - C*C ) / ( 2.f * A * B );
    float fAngleDegree = radian2Degree ( acos ( fCos ) );
    return fAngleDegree;
}

}
}