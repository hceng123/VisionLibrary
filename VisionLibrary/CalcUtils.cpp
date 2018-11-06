#include "CalcUtils.h"
#include "spline.h"
#include <fstream>

namespace AOI
{
namespace Vision
{

/*static*/ float CalcUtils::ptDisToLine(const cv::Point2f &ptInput, bool bReversedFit, float fSlope, float fIntercept) {
    float distance = 0.f;
    if (bReversedFit) {
        if (fabs(fSlope) < 0.0001) {
            distance = ptInput.x - fIntercept;
        }
        else {
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
    else {
        if (fabs(fSlope) < 0.0001) {
            distance = ptInput.y - fIntercept;
        }
        else {
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

/*static*/ float CalcUtils::lineSlope(const PR_Line2f &line) {
    return (line.pt2.y - line.pt1.y) / (line.pt2.x - line.pt1.x);
}

/*static*/ void CalcUtils::lineSlopeIntercept(const PR_Line2f &line, float &fSlope, float &fIntercept) {
    fSlope = (line.pt2.y - line.pt1.y) / (line.pt2.x - line.pt1.x);
    fIntercept = line.pt2.y - fSlope * line.pt2.x;
}

/*static*/ VectorOfPoint CalcUtils::getCornerOfRotatedRect(const cv::RotatedRect &rotatedRect) {
    VectorOfPoint vecPoint;
    cv::Point2f arrPt[4];
    rotatedRect.points(arrPt);
    for (int i = 0; i < 4; ++ i)
        vecPoint.push_back(arrPt[i]);
    return vecPoint;
}

/*static*/ float CalcUtils::guassianValue(float ssq, float x) {
    return ToFloat(exp(- (x * x) / (2.0 * ssq)) / (CV_PI * ssq));
}

/*static*/ cv::Mat CalcUtils::generateGuassinDiffKernel(int nOneSideWidth, float ssq) {
    std::vector<float> vecGuassian;
    for (int i = -nOneSideWidth; i <= nOneSideWidth; ++ i)
        vecGuassian.push_back(ToFloat(i) * guassianValue(ssq, ToFloat(i)));
    cv::Mat matGuassian(vecGuassian);
    cv::transpose(matGuassian, matGuassian);
    cv::Mat matTmp(matGuassian, cv::Range::all(), cv::Range(0, nOneSideWidth));
    float fSum = ToFloat(fabs(cv::sum(matTmp)[0]));
    matGuassian /= fSum;
    return matGuassian;
}

/*static*/ void CalcUtils::filter2D_Conv(cv::InputArray src, cv::OutputArray dst, int ddepth,
    cv::InputArray kernel, cv::Point anchor,
    double delta, int borderType) {
    cv::Mat newKernel;
    const int FLIP_H_Z = -1;
    cv::flip(kernel, newKernel, FLIP_H_Z);
    cv::Point newAnchor = anchor;
    if (anchor.x > 0 && anchor.y >= 0)
        newAnchor = cv::Point(newKernel.cols - anchor.x - 1, newKernel.rows - anchor.y - 1);
    cv::filter2D(src, dst, ddepth, newKernel, newAnchor, delta, borderType);
}

float CalcUtils::calcPointToContourDist(const cv::Point &ptInput, const VectorOfPoint &contour, cv::Point &ptResult) {
    cv::Point ptNearestPoint1, ptNearestPoint2;
    float fNearestDist1, fNearestDist2;
    fNearestDist1 = fNearestDist2 = std::numeric_limits<float>::max();
    for (const auto &ptOfContour : contour) {
        auto distance = distanceOf2Point<int>(ptOfContour, ptInput);
        if (distance < fNearestDist1) {
            fNearestDist2 = fNearestDist1;
            ptNearestPoint2 = ptNearestPoint1;

            fNearestDist1 = distance;
            ptNearestPoint1 = ptOfContour;
        }
        else if (distance < fNearestDist2) {
            fNearestDist2 = distance;
            ptNearestPoint2 = ptOfContour;
        }
    }
    float C = distanceOf2Point<int>(ptNearestPoint1, ptNearestPoint2);
    if (C < 5) {
        ptResult.x = ToInt32((ptNearestPoint1.x + ptNearestPoint2.x) / 2.f + 0.5f);
        ptResult.y = ToInt32((ptNearestPoint1.y + ptNearestPoint2.y) / 2.f + 0.5f);
        float fDistance = distanceOf2Point<int>(ptInput, ptResult);
        return fDistance;
    }
    float A = distanceOf2Point<int>(ptInput, ptNearestPoint1);
    float B = distanceOf2Point<int>(ptInput, ptNearestPoint2);

    float cosOfAngle = (A*A + C*C - B*B) / (2.f * A * C);
    float angle = acos(cosOfAngle);
    float fDistance;
    if (angle <= CV_PI / 2) {
        fDistance = A * std::sin(angle);
        float D = A * std::cos(angle);
        float DCRatio = D / C;
        ptResult.x = ToInt32(ptNearestPoint1.x * (1.f - DCRatio) + ptNearestPoint2.x * DCRatio + 0.5f);
        ptResult.y = ToInt32(ptNearestPoint1.y * (1.f - DCRatio) + ptNearestPoint2.y * DCRatio + 0.5f);
    }
    else {
        fDistance = A * ToFloat(std::sin(CV_PI - angle));
        float D = A * ToFloat(std::cos(CV_PI - angle));
        float DCRatio = D / C;
        ptResult.x = ToInt32(ptNearestPoint1.x + D / C * (ptNearestPoint1.x - ptNearestPoint2.x));
        ptResult.y = ToInt32(ptNearestPoint1.y + D / C * (ptNearestPoint1.y - ptNearestPoint2.y));
    }
    return fDistance;
}

/*static*/ cv::Point2f CalcUtils::getContourCtr(const VectorOfPoint &contour) {
    cv::Moments moment = cv::moments(contour);
    cv::Point2f ptCenter;
    ptCenter.x = static_cast<float>(moment.m10 / moment.m00);
    ptCenter.y = static_cast<float>(moment.m01 / moment.m00);
    return ptCenter;
}

/*static*/ int CalcUtils::countOfNan(const cv::Mat &matInput) {
    cv::Mat matNan;
    cv::compare(matInput, matInput, matNan, cv::CmpTypes::CMP_EQ);
    auto nCount = cv::countNonZero(matNan);
    return ToInt32(matInput.total()) - nCount;
}

/*static*/ void CalcUtils::findMinMaxCoord(const VectorOfPoint &vecPoints, int &xMin, int &xMax, int &yMin, int &yMax) {
    xMin = yMin = std::numeric_limits<int>::max();
    xMax = yMax = std::numeric_limits<int>::min();
    for (const auto &point : vecPoints) {
        if (point.x > xMax)
            xMax = point.x;
        if (point.x < xMin)
            xMin = point.x;
        if (point.y > yMax)
            yMax = point.y;
        if (point.y < yMin)
            yMin = point.y;
    }
}

/*static*/ float CalcUtils::calcFrequency(const cv::Mat &matInput) {
    assert(matInput.rows == 1);
    const float SAMPLE_FREQUENCY = 1.f;
    cv::Mat matRowFloat;
    matInput.convertTo(matRowFloat, CV_32FC1);
    int m = ToInt32(pow(2, std::floor(log2(matRowFloat.cols))));
    matRowFloat = cv::Mat(matRowFloat, cv::Range::all(), cv::Range(0, m));
    cv::Mat matDft;
    cv::dft(matRowFloat, matDft, cv::DftFlags::DCT_ROWS);
    auto vecVecRowFloat = CalcUtils::matToVector<float>(matRowFloat);
    cv::Mat matAbsDft = cv::abs(matDft);
    matAbsDft = cv::Mat(matAbsDft, cv::Rect(1, 0, matAbsDft.cols - 1, 1)).clone();
    auto vecVecDft = CalcUtils::matToVector<float>(matAbsDft);
    auto maxElement = std::max_element(vecVecDft[0].begin(), vecVecDft[0].end());
    int nIndex = ToInt32(std::distance(vecVecDft[0].begin(), maxElement)) + 1;
    float fFrequency = (float)nIndex * SAMPLE_FREQUENCY / m;
    return fFrequency;
}

/*static*/ VectorOfDouble CalcUtils::interp1(const VectorOfDouble &vecX, const VectorOfDouble &vecV, const VectorOfDouble &vecXq, bool bSpine) {
    tk::spline s;
    s.set_points(vecX, vecV, bSpine);    // currently it is required that X is already sorted
    VectorOfDouble vecResult;
    vecResult.reserve(vecXq.size());
    for (const auto xq : vecXq)
        vecResult.push_back(s(xq));
    return vecResult;
}

/*static*/ void CalcUtils::saveMatToCsv(const cv::Mat &matrix, std::string filename) {
    std::ofstream outputFile(filename);
    if (! outputFile.is_open())
        return;
    outputFile << cv::format(matrix, cv::Formatter::FMT_CSV);
    outputFile.close();
}

/*static*/ int CalcUtils::twoLineIntersect(const PR_Line2f &line1, const PR_Line2f &line2, cv::Point2f &ptResult) {
    const auto Precision = std::numeric_limits<float>::epsilon();
    const auto VerticalLineSlopeThreshold = 100.f;  //If the slope the line large than this value, consider it is vertical.
    float fLineSlope1 = (line1.pt2.y - line1.pt1.y) / (line1.pt2.x - line1.pt1.x);
    float fLineSlope2 = (line2.pt2.y - line2.pt1.y) / (line2.pt2.x - line2.pt1.x);
    if (fabs(fLineSlope1 - fLineSlope2) < PARALLEL_LINE_SLOPE_DIFF_LMT) {
        printf("No cross point for parallized lines");
        return -1;
    }

    //Line1 can be expressed as y = fLineSlope1 * x + fLineCrossWithY1
    //Line2 can be expressed as y = fLineSlope2 * x + fLineCrossWithY2
    float fLineCrossWithY1 = fLineSlope1 * (- line1.pt1.x) + line1.pt1.y;
    float fLineCrossWithY2 = fLineSlope2 * (- line2.pt1.x) + line2.pt1.y;

    if (fLineSlope1 == fLineSlope1 && fabs(fLineSlope1) < VerticalLineSlopeThreshold &&
        fLineSlope2 == fLineSlope2 && fabs(fLineSlope2) < VerticalLineSlopeThreshold) {
        ptResult.x = (fLineCrossWithY2 - fLineCrossWithY1) / (fLineSlope1 - fLineSlope2);
        ptResult.y = fLineSlope1 * ptResult.x + fLineCrossWithY1;
    }
    else
    if ((fLineSlope1 != fLineSlope1 || fabs(fLineSlope1) > VerticalLineSlopeThreshold) &&
        (fLineSlope2 != fLineSlope2 || fabs(fLineSlope2) > VerticalLineSlopeThreshold)) {
        //Both of the line is vertical, no cross point
        return -1;
    }
    //Line1 is vertical
    else if (fLineSlope1 != fLineSlope1 || fabs(fLineSlope1) > VerticalLineSlopeThreshold) {
        ptResult.x = line1.pt1.x;
        ptResult.y = fLineSlope2 * ptResult.x + fLineCrossWithY2;
    }
    else if (fLineSlope2 != fLineSlope2 || fabs(fLineSlope2) > VerticalLineSlopeThreshold) {
        ptResult.x = line2.pt1.x;
        ptResult.y = fLineSlope1 * ptResult.x + fLineCrossWithY1;
    }
    else {
        return -1;
    }
    return 0;
}

/*static*/ int CalcUtils::twoLineIntersect(bool bReverseFit1, float fSlope1, float fIntercept1, bool bReverseFit2, float fSlope2, float fIntercept2, cv::Point2f &ptResult) {
    if (bReverseFit1 == bReverseFit2) {
        if (fabs(fSlope1 - fSlope2) < 0.01)
            return -1;

        if (bReverseFit1) {
            ptResult.y = (fIntercept2 - fIntercept1) / (fSlope1 - fSlope2);
            ptResult.x = fSlope1 * ptResult.y + fIntercept1;
        }else {
            ptResult.x = (fIntercept2 - fIntercept1) / (fSlope1 - fSlope2);
            ptResult.y = fSlope1 * ptResult.y + fIntercept1;
        }
    }else {
        if (bReverseFit1) {
            ptResult.y = (fIntercept2 + fIntercept1 * fSlope2) / (1 - fSlope1 * fSlope2);
            ptResult.x = fSlope1 * ptResult.y + fIntercept1;
        }else {
            ptResult.x = (fIntercept2 + fIntercept1 * fSlope2) / (1 - fSlope1 * fSlope2);
            ptResult.y = fSlope1 * ptResult.x + fIntercept1;
        }
    }
    return 0;
}

/*static*/ float CalcUtils::calc2LineAngle(const PR_Line2f &line1, const PR_Line2f &line2) {
    cv::Point2f ptCrossPoint;
    //If cannot find cross point of two line, which means they are parallel, so the angle is 0.
    if (twoLineIntersect(line1, line2, ptCrossPoint) != 0) {
        return 0.f;
    }

    cv::Point ptLine1ChoosePoint = line1.pt1;
    float A = distanceOf2Point(ptCrossPoint, line1.pt1);
    if (A < 1) {
        A = distanceOf2Point(ptCrossPoint, line1.pt2);
        ptLine1ChoosePoint = line1.pt2;
    }

    cv::Point ptLine2ChoosePoint = line2.pt1;
    float B = distanceOf2Point(ptCrossPoint, line2.pt1);
    if (B < 1) {
        B = distanceOf2Point(ptCrossPoint, line2.pt2);
        ptLine2ChoosePoint = line2.pt2;
    }

    float C = distanceOf2Point(ptLine1ChoosePoint, ptLine2ChoosePoint);
    //Use the cos rules to calculate the cos of angle.
    float fCos = (A*A + B*B - C*C) / (2.f * A * B);
    float fAngleDegree = ToFloat(radian2Degree(acos(fCos)));
    return fAngleDegree;
}

/*static*/ float CalcUtils::linearInterpolate(const cv::Point2f &pt1, float fValue1, const cv::Point2f &pt2, float fValue2, const cv::Point2f &ptToGet) {
    float fResult = 0;
    if (fabs(pt1.x - pt2.x) < 0.1)
        fResult = (pt2.y - ptToGet.y) / (pt2.y - pt1.y) * fValue1 + (ptToGet.y - pt1.y) / (pt2.y - pt1.y) * fValue2;
    else if(fabs(pt1.y - pt2.y) < 0.1)
        fResult = (pt2.x - ptToGet.x) / (pt2.x - pt1.x) * fValue1 + (ptToGet.x - pt1.x) / (pt2.x - pt1.x) * fValue2;
    else
        assert(0);

    return fResult;
}

// Abount bilinear interpolation in Wikipedia:
// http://en.wikipedia.org/wiki/Bilinear_interpolation
/*static*/ float CalcUtils::bilinearInterpolate(const VectorOfPoint2f &vecPoint, const VectorOfFloat &vecValue, const cv::Point2f &ptToGet) {
    assert(vecPoint.size() == 4);
    assert(vecValue.size() == 4);

    for (size_t index = 0; index < vecPoint.size(); ++ index) {
        if (distanceOf2Point(vecPoint[index], ptToGet) < 1)
            return vecValue[index];
    }

    float x1 = vecPoint[0].x;
    float y1 = vecPoint[0].y;
    float x2 = vecPoint[2].x;
    float y2 = vecPoint[2].y;
    auto Q11 = vecValue[0];
    auto Q12 = vecValue[1];
    auto Q22 = vecValue[2];
    auto Q21 = vecValue[3];
    float fxy1 = 0, fxy2 = 0;
    if (fabs(x2 - x1) > 0.1) {
        fxy1 = (x2 - ptToGet.x) / (x2 - x1) * Q11 + (ptToGet.x - x1) / (x2 - x1) * Q21;
        fxy2 = (x2 - ptToGet.x) / (x2 - x1) * Q12 + (ptToGet.x - x1) / (x2 - x1) * Q22;
    }else {
        fxy1 = (Q11 + Q21) / 2.f;
        fxy2 = (Q12 + Q22) / 2.f;
    }

    if ( fabs(fxy1 - fxy2) < 0.001 || fabs(y2 - y1) < 0.1)
        return (fxy1 + fxy2) / 2.f;

    auto result = (y2 - ptToGet.y) / (y2 - y1) * fxy1 + (ptToGet.y - y1) / (y2- y1) * fxy2;
    return result;
}

/*static*/ bool CalcUtils::isRectInRect(const cv::Rect2f &rectIn, const cv::Rect2f &rectOut) {
    if (rectOut.contains(rectIn.tl()) && rectOut.contains(rectIn.br()))
        return true;
    return false;
}

/*static*/ void CalcUtils::adjustRectROI(cv::Rect &rect, const cv::Mat &matInput) {
    if (rect.x < 0) rect.x = 0;
    if (rect.y < 0) rect.y = 0;
    if ((rect.x + rect.width)  > matInput.cols) rect.width  = matInput.cols - rect.x;
    if ((rect.y + rect.height) > matInput.rows) rect.height = matInput.rows - rect.y;
}

/*static*/ cv::Rect CalcUtils::boundingRect(const VectorOfRect &vecRect) {
    VectorOfPoint vecPoints;
    vecPoints.reserve(vecRect.size() * 4);
    for (const auto &rect : vecRect) {
        vecPoints.emplace_back(rect.x, rect.y);
        vecPoints.emplace_back(rect.x + rect.width, rect.y);
        vecPoints.emplace_back(rect.x + rect.width, rect.y + rect.height);
        vecPoints.emplace_back(rect.x, rect.y + rect.height);
    }
    return cv::boundingRect(vecPoints);
}

/*static*/ void CalcUtils::setToExcept(cv::Mat &matInOut, int value, const cv::Rect &rectROI) {
    cv::Mat matMask = cv::Mat::ones(matInOut.size(), CV_8UC1);
    cv::Mat matMaskROI(matMask, rectROI);
    matMaskROI.setTo(0);
    matInOut.setTo(value, matMask);
}

/*static*/ bool CalcUtils::isVerticalROI(const VectorOfRect &vecRectROIs) {
    if (vecRectROIs.size() != 2)
        return false;
    cv::Point ptCenter1(vecRectROIs[0].x + vecRectROIs[0].width / 2, vecRectROIs[0].y + vecRectROIs[0].height / 2);
    cv::Point ptCenter2(vecRectROIs[1].x + vecRectROIs[1].width / 2, vecRectROIs[1].y + vecRectROIs[1].height / 2);
    if (abs(ptCenter1.y - ptCenter2.y) > abs(ptCenter1.x - ptCenter2.x))
        return true;
    return false;
}

}
}