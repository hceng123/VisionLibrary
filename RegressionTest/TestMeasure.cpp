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

static void TestTwoLineIntersectSub(const PR_TWO_LINE_INTERSECT_CMD &stCmd, PR_TWO_LINE_INTERSECT_RPY &stRpy) {
    std::cout << "Two Line input:" << std::endl;
    printPRLine ( stCmd.line1 );
    printPRLine ( stCmd.line2 );

    PR_TwoLineIntersect ( &stCmd, &stRpy );
    std::cout << "PR_TwoLineIntersect status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus )
        std::cout << "Intersect Point " << stRpy.ptIntersect << std::endl;
}

void TestTwoLineIntersect() {
    PR_TWO_LINE_INTERSECT_CMD stCmd;
    PR_TWO_LINE_INTERSECT_RPY stRpy;

    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "TWO LINE INTERSECT REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 10, 10 );

    stCmd.line2.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 10, -10 );
    TestTwoLineIntersectSub ( stCmd, stRpy );

    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "TWO LINE INTERSECT REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( -10, 1 );
    stCmd.line1.pt2 = cv::Point2f (  10, 1 );

    stCmd.line2.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 10, -10 );
    TestTwoLineIntersectSub ( stCmd, stRpy );

    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "TWO LINE INTERSECT REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line1.pt2 = cv::Point2f ( 1,  10 );

    stCmd.line2.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 10, -10 );
    TestTwoLineIntersectSub ( stCmd, stRpy );

    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "TWO LINE INTERSECT REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 10, -10 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1,  10 );

    TestTwoLineIntersectSub ( stCmd, stRpy );

    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "TWO LINE INTERSECT REGRESSION TEST #5 STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 2, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 2, -10 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1,  10 );

    TestTwoLineIntersectSub ( stCmd, stRpy );

    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "TWO LINE INTERSECT REGRESSION TEST #6 STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 2, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 2.02f, -100 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1.03f,  10 );

    TestTwoLineIntersectSub ( stCmd, stRpy );
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

void TestParallelLineDistanceSub(const PR_PARALLEL_LINE_DIST_CMD &stCmd, PR_PARALLEL_LINE_DIST_RPY &stRpy) {
    std::cout << "Two Line input:" << std::endl;
    printPRLine ( stCmd.line1 );
    printPRLine ( stCmd.line2 );

    PR_ParallelLineDist ( &stCmd, &stRpy );
    std::cout << "PR_ParallelLineDist status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus )
        std::cout << "Distance " << stRpy.fDistance << std::endl;
}

void TestParallelLineDistance() {
    PR_PARALLEL_LINE_DIST_CMD stCmd;
    PR_PARALLEL_LINE_DIST_RPY stRpy;

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "POINT LINE DISTANCE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;

    stCmd.line1.pt1 = cv::Point2f ( 2, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 2.02f, -100 );

    stCmd.line2.pt1 = cv::Point2f ( 1, -10 );
    stCmd.line2.pt2 = cv::Point2f ( 1.03f,  10 );

    TestParallelLineDistanceSub ( stCmd, stRpy );

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "POINT LINE DISTANCE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;
    stCmd.line1.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 10, 10 );

    stCmd.line2.pt1 = cv::Point2f ( 10, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 20,  10 );

    TestParallelLineDistanceSub ( stCmd, stRpy );

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "POINT LINE DISTANCE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;
    stCmd.line1.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line1.pt2 = cv::Point2f ( 0, 10 );

    stCmd.line2.pt1 = cv::Point2f ( 0, 0 );
    stCmd.line2.pt2 = cv::Point2f ( 10, 0 );

    TestParallelLineDistanceSub ( stCmd, stRpy );
}

void TestCrossSectionArea() {
    PR_CROSS_SECTION_AREA_CMD stCmd;
    PR_CROSS_SECTION_AREA_RPY stRpy;

    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "CROSS SECTION AREA REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    stCmd.vecContourPoints.emplace_back ( 10.f, 10.f );
    stCmd.vecContourPoints.emplace_back ( 20.f, 20.f );
    stCmd.vecContourPoints.emplace_back ( 30.f, 10.f );
    stCmd.bClosed = false;

    PR_CrossSectionArea ( &stCmd, &stRpy );
    std::cout << "PR_CrossSectionArea status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus )
        std::cout << "Distance " << stRpy.fArea << std::endl;
}

}
}