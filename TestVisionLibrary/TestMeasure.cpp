#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestTwoLineAngle() {
    PR_TWO_LINE_ANGLE_CMD stCmd;
    PR_TWO_LINE_ANGLE_RPY stRpy;

    stCmd.line1.pt1 = cv::Point ( 0, 0 );
    stCmd.line1.pt2 = cv::Point ( 10, 10 );

    stCmd.line2.pt1 = cv::Point ( 0, 0 );
    stCmd.line2.pt2 = cv::Point ( 10, -10 );
    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;
}

void TestTwoLineAngle_1() {
    PR_TWO_LINE_ANGLE_CMD stCmd;
    PR_TWO_LINE_ANGLE_RPY stRpy;

    stCmd.line1.pt1 = cv::Point ( -10, 1 );
    stCmd.line1.pt2 = cv::Point (  10, 1 );

    stCmd.line2.pt1 = cv::Point ( 0, 0 );
    stCmd.line2.pt2 = cv::Point ( 10, -10 );
    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;
}

void TestTwoLineAngle_2() {
    PR_TWO_LINE_ANGLE_CMD stCmd;
    PR_TWO_LINE_ANGLE_RPY stRpy;

    stCmd.line1.pt1 = cv::Point ( 1, -10 );
    stCmd.line1.pt2 = cv::Point ( 1,  10 );

    stCmd.line2.pt1 = cv::Point ( 0, 0 );
    stCmd.line2.pt2 = cv::Point ( 10, -10 );
    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;
}

void TestTwoLineAngle_3() {
    PR_TWO_LINE_ANGLE_CMD stCmd;
    PR_TWO_LINE_ANGLE_RPY stRpy;    

    stCmd.line1.pt1 = cv::Point ( 0, 0 );
    stCmd.line1.pt2 = cv::Point ( 10, -10 );

    stCmd.line2.pt1 = cv::Point ( 1, -10 );
    stCmd.line2.pt2 = cv::Point ( 1,  10 );

    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;
}

void TestTwoLineAngle_4() {
    PR_TWO_LINE_ANGLE_CMD stCmd;
    PR_TWO_LINE_ANGLE_RPY stRpy;    

    stCmd.line1.pt1 = cv::Point ( 2, 0 );
    stCmd.line1.pt2 = cv::Point ( 2, -10 );

    stCmd.line2.pt1 = cv::Point ( 1, -10 );
    stCmd.line2.pt2 = cv::Point ( 1,  10 );

    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;
}

void TestTwoLineAngle_5() {
    PR_TWO_LINE_ANGLE_CMD stCmd;
    PR_TWO_LINE_ANGLE_RPY stRpy;    

    stCmd.line1.pt1 = cv::Point2f ( 2, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 2.02f, -100 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1.03f,  10 );

    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;
}

void TestPointLineDistance_1() {
    PR_POINT_LINE_DISTANCE_CMD stCmd;
    PR_POINT_LINE_DISTANCE_RPY stRpy;

    stCmd.ptInput = cv::Point2f ( 0, 0 );
    stCmd.bReversedFit = true;
    stCmd.fSlope = 0;
    stCmd.fIntercept = 100;
    PR_PointLineDistance ( &stCmd, &stRpy );
    std::cout << "Distance " << stRpy.fDistance << std::endl;
}

void TestParallelLineDistance_1() {
    PR_PARALLEL_LINE_DIST_CMD stCmd;
    PR_PARALLEL_LINE_DIST_RPY stRpy;

    stCmd.line1.pt1 = cv::Point2f ( 2, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 2.02f, -100 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1.03f,  10 );

    PR_ParallelLineDist ( &stCmd, &stRpy );
    std::cout << "PR_ParallelLineDist status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus )
        std::cout << "Distance " << stRpy.fDistance << std::endl;
}

void TestParallelLineDistance_2() {
    PR_PARALLEL_LINE_DIST_CMD stCmd;
    PR_PARALLEL_LINE_DIST_RPY stRpy;

    stCmd.line1.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 10, 10 );

    stCmd.line2.pt1 = cv::Point2f ( 10, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 20,  10 );

    PR_ParallelLineDist ( &stCmd, &stRpy );
    std::cout << "PR_ParallelLineDist status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus )
        std::cout << "Distance " << stRpy.fDistance << std::endl;
}

void TestParallelLineDistance_3() {
    PR_PARALLEL_LINE_DIST_CMD stCmd;
    PR_PARALLEL_LINE_DIST_RPY stRpy;

    stCmd.line1.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 0, 10 );

    stCmd.line2.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 10, 0 );

    PR_ParallelLineDist ( &stCmd, &stRpy );
    std::cout << "PR_ParallelLineDist status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus )
        std::cout << "Distance " << stRpy.fDistance << std::endl;
}

cv::Point2f lineIntersect(float fSlope1, float fIntercept1, float fSlope2, float fIntercept2) {
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
    if ( cv::solve(A, B, matResultImg ) ) {
        ptResult.x = matResultImg.at<float>(0, 0);
        ptResult.y = matResultImg.at<float>(1, 0);
    }
    return ptResult;
}

void lineSlopeIntercept(const PR_Line2f &line, float &fSlope, float &fIntercept) {
    fSlope = ( line.pt2.y - line.pt1.y ) / ( line.pt2.x - line.pt1.x );
    fIntercept = line.pt2.y  - fSlope * line.pt2.x;
}

cv::Point2f lineIntersect ( const PR_Line2f &line1, const PR_Line2f &line2 ) {
    float fSlope1, fIntercept1, fSlope2, fIntercept2;
    lineSlopeIntercept ( line1, fSlope1, fIntercept1 );
    lineSlopeIntercept ( line2, fSlope2, fIntercept2 );
    return lineIntersect ( fSlope1, fIntercept1, fSlope2, fIntercept2 );
}

void CalcCornerRadius(const cv::Mat &matInput, const cv::Rect &rectROI, const PR_Line2f &line1, const PR_Line2f &line2, float &fRadius, cv::Point &ptCenter ) {

}