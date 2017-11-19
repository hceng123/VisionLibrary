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
    stCmd.line1.pt2 = cv::Point2f ( 2.02, -100 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1.03,  10 );

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