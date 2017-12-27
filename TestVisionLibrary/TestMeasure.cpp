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

void TestLineIntersect() {
    PR_TWO_LINE_INTERSECT_CMD stCmd;
    PR_TWO_LINE_INTERSECT_RPY stRpy;

    stCmd.line1.pt1 = cv::Point2f ( 2, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 2.02f, -100 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1.03f,  10 );

    PR_TwoLineIntersect ( &stCmd, &stRpy );
}

void CalcCornerRadius(const cv::Mat &matInput, const cv::Rect &rectROI, const PR_Line2f &line1, const PR_Line2f &line2, float &fRadius, cv::Point &ptCenter ) {

}

void TestCrossSectionArea() {
    VectorOfPoint2f contour;
    contour.emplace_back ( 10.f, 10.f );
    contour.emplace_back ( 20.f, 20.f );
    contour.emplace_back ( 30.f, 10.f );

    float nMinX = std::numeric_limits<float>::max();
    float nMaxX = std::numeric_limits<float>::min();
    for ( const auto &point : contour ) {
        if ( point.x > nMaxX )
            nMaxX = point.x;
        if ( point.x < nMinX )
            nMinX = point.x;
    }
    contour.emplace_back ( nMaxX, 0.f );
    contour.emplace_back ( nMinX, 0.f );
    double dArea = cv::contourArea ( contour );
    std::cout << "Area " << dArea << std::endl;
}

void TestFindCircle() {
    PR_FIND_CIRCLE_CMD stCmd;
    PR_FIND_CIRCLE_RPY stRpy;   

    stCmd.matInputImg = cv::imread ("./data/CaliperImage.bmp", cv::IMREAD_GRAYSCALE );
    //cv::Mat matImage = 255 - stCmd.matInputImg;
    //cv::imwrite ("./data/CaliperImageReverse.png", matImage );
    stCmd.enInnerAttribute = PR_OBJECT_ATTRIBUTE::DARK;
    stCmd.fMinSrchRadius = 120;
    stCmd.fMaxSrchRadius = 280;
    stCmd.fStartSrchAngle = -165;
    stCmd.fEndSrchAngle = -15;
    stCmd.ptExpectedCircleCtr = cv::Point(620, 650);

    PR_FindCircle ( &stCmd, &stRpy );
    cv::imwrite ("./data/DetectCircleResult.png", stRpy.matResultImg );
}