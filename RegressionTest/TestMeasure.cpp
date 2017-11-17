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
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #1 STARTING";
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
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #2 STARTING";
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
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #3 STARTING";
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
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #4 STARTING";
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
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #5 STARTING";
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
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #6 STARTING";
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

}
}