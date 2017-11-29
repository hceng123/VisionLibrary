#include "stdafx.h"
#include "UtilityFunc.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>

namespace AOI
{
namespace Vision
{

void TestTwoLineAngle() {
    PR_TWO_LINE_ANGLE_CMD stCmd;
    PR_TWO_LINE_ANGLE_RPY stRpy;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "TWO LINE ANGLE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 10, 10 );

    stCmd.line2.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 10, -10 );
    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two Line input:" << std::endl;
    printPRLine ( stCmd.line1 );
    printPRLine ( stCmd.line2 );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "TWO LINE ANGLE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( -10, 1 );
    stCmd.line1.pt2 = cv::Point2f (  10, 1 );

    stCmd.line2.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 10, -10 );
    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two Line input:" << std::endl;
    printPRLine ( stCmd.line1 );
    printPRLine ( stCmd.line2 );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "TWO LINE ANGLE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line1.pt2 = cv::Point2f ( 1,  10 );

    stCmd.line2.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 10, -10 );
    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two Line input:" << std::endl;
    printPRLine ( stCmd.line1 );
    printPRLine ( stCmd.line2 );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "TWO LINE ANGLE REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 10, -10 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1,  10 );

    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two Line input:" << std::endl;
    printPRLine ( stCmd.line1 );
    printPRLine ( stCmd.line2 );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "TWO LINE ANGLE REGRESSION TEST #5 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 2, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 2, -10 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1,  10 );

    PR_TwoLineAngle ( &stCmd, &stRpy );

    std::cout << "Two Line input:" << std::endl;
    printPRLine ( stCmd.line1 );
    printPRLine ( stCmd.line2 );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "TWO LINE ANGLE REGRESSION TEST #6 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 2, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 2.02f, -100 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1.03f,  10 );

    PR_TwoLineAngle ( &stCmd, &stRpy );

    std::cout << "Two Line input:" << std::endl;
    printPRLine ( stCmd.line1 );
    printPRLine ( stCmd.line2 );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;
}

void TestPointLineDistance() {
    PR_POINT_LINE_DISTANCE_CMD stCmd;
    PR_POINT_LINE_DISTANCE_RPY stRpy;

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "POINT LINE DISTANCE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;

    stCmd.ptInput = cv::Point2f ( 0, 0 );
    stCmd.bReversedFit = true;
    stCmd.fSlope = 0;
    stCmd.fIntercept = 100;
    PR_PointLineDistance ( &stCmd, &stRpy );
    std::cout << "Point " << stCmd.ptInput << std::endl;
    std::cout << "ReverseFit " << stCmd.bReversedFit << " Slope " << stCmd.fSlope << " Intercept " << stCmd.fIntercept << std::endl;
    std::cout << "Distance " << stRpy.fDistance << std::endl;

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "POINT LINE DISTANCE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;

    stCmd.ptInput = cv::Point2f ( 0, 0 );
    stCmd.bReversedFit = false;
    stCmd.fSlope = 0;
    stCmd.fIntercept = 100;
    PR_PointLineDistance ( &stCmd, &stRpy );
    std::cout << "Point " << stCmd.ptInput << std::endl;
    std::cout << "ReverseFit " << stCmd.bReversedFit << " Slope " << stCmd.fSlope << " Intercept " << stCmd.fIntercept << std::endl;
    std::cout << "Distance " << stRpy.fDistance << std::endl;

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "POINT LINE DISTANCE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;

    stCmd.ptInput = cv::Point2f ( 0, 0 );
    stCmd.bReversedFit = false;
    stCmd.fSlope = 1;
    stCmd.fIntercept = 100;
    PR_PointLineDistance ( &stCmd, &stRpy );
    std::cout << "Point " << stCmd.ptInput << std::endl;
    std::cout << "ReverseFit " << stCmd.bReversedFit << " Slope " << stCmd.fSlope << " Intercept " << stCmd.fIntercept << std::endl;
    std::cout << "Distance " << stRpy.fDistance << std::endl;
}

}
}