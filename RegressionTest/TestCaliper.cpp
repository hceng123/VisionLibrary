#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

static void PrintRpy ( const PR_CALIPER_RPY &stRpy ) {
    char chArrMsg[100];
    std::cout << "Caliper status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if (VisionStatus::OK == stRpy.enStatus) {
        std::cout << "ReversedFit = " << stRpy.bReversedFit << std::endl;
        std::cout << std::fixed << std::setprecision ( 2 ) << "Line slope = " << stRpy.fSlope << ", intercept = " << stRpy.fIntercept << std::endl;
        _snprintf ( chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine.pt1.x, stRpy.stLine.pt1.y, stRpy.stLine.pt2.x, stRpy.stLine.pt2.y );
        std::cout << "Line coordinate: " << chArrMsg << std::endl;
        std::cout << std::fixed << std::setprecision ( 2 ) << "Linerity  = " << stRpy.fLinerity << std::endl;
        std::cout << "Linerity check pass: " << stRpy.bLinerityCheckPass << std::endl;
        std::cout << std::fixed << std::setprecision ( 2 ) << "Angle  = " << stRpy.fAngle << std::endl;
        std::cout << "Angle check pass: " << stRpy.bAngleCheckPass << std::endl;
    }else {
        PR_GET_ERROR_INFO_RPY stErrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrRpy);
        std::cout << "Error Msg: " << stErrRpy.achErrorStr << std::endl;
    }
};

void TestCaliperSectionAvgGuassianDiff();
void TestCaliperRoatedROI();

void TestCaliper() {
    PR_CALIPER_CMD stCmd;
    PR_CALIPER_RPY stRpy;    

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIPER REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    cv::Rect rectROI = cv::Rect(1216, 802, 142, 332);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::MAX_TO_MIN;
    stCmd.bCheckLinerity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinerity = 60.;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIPER REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(928, 1276, 270, 120);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::MIN_TO_MAX;
    stCmd.fExpectedAngle = 0;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIPER REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(1496, 1576, 228, 88);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::MAX_TO_MIN;
    stCmd.fExpectedAngle = 0;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIPER REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(1591, 970, 51, 90);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::MAX_TO_MIN;
    stCmd.bCheckLinerity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinerity = 60.;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;

    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    TestCaliperSectionAvgGuassianDiff();
    TestCaliperRoatedROI();
}

void TestCaliperSectionAvgGuassianDiff() {
    PR_CALIPER_CMD stCmd;
    PR_CALIPER_RPY stRpy;    

    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl << "CALIPER GUASSIAN DIFF REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/CognexEdge.png");
    cv::Rect rectROI = cv::Rect(304, 732, 208, 532);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enAlgorithm = PR_CALIPER_ALGORITHM::SECTION_AVG_GAUSSIAN_DIFF;
    stCmd.enDetectDir = PR_CALIPER_DIR::MAX_TO_MIN;
    stCmd.bCheckLinerity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinerity = 60.;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl << "CALIPER GUASSIAN DIFF REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl;

    rectROI = cv::Rect(728, 744, 164, 632);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::MIN_TO_MAX;
    stCmd.fExpectedAngle = 90;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl << "CALIPER GUASSIAN DIFF REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(998, 1298, 226, 68);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::MAX_TO_MIN;
    stCmd.fExpectedAngle = 0;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl << "CALIPER GUASSIAN DIFF REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(952, 1820, 188, 70);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::MIN_TO_MAX;
    stCmd.fExpectedAngle = 0;

    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);
}

void TestCaliperRoatedROI() {
    PR_CALIPER_CMD stCmd;
    PR_CALIPER_RPY stRpy;    

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "CALIPER ROTATED ROI REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/TestRotatedCaliper.png");
    stCmd.rectRotatedROI.center = cv::Point (85, 121 );
    stCmd.rectRotatedROI.size = cv::Size(100, 30);
    stCmd.rectRotatedROI.angle = -70;
    stCmd.enAlgorithm = PR_CALIPER_ALGORITHM::SECTION_AVG_GAUSSIAN_DIFF;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinerity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinerity = 60.;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = -70;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "CALIPER ROTATED ROI REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;

    stCmd.rectRotatedROI.center = cv::Point (171, 34 );
    stCmd.rectRotatedROI.size = cv::Size(200, 30);
    stCmd.rectRotatedROI.angle = 20;
    stCmd.enAlgorithm = PR_CALIPER_ALGORITHM::PROJECTION;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinerity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinerity = 60.;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 20;
    stCmd.fAngleDiffTolerance = 5;
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "CALIPER ROTATED ROI REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;
    stCmd.rectRotatedROI.center = cv::Point (368, 54);
    stCmd.rectRotatedROI.size = cv::Size(200, 30);
    stCmd.rectRotatedROI.angle = 20;
    stCmd.enAlgorithm = PR_CALIPER_ALGORITHM::SECTION_AVG_GAUSSIAN_DIFF;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinerity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinerity = 60.;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "CALIPER ROTATED ROI REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;
    stCmd.rectRotatedROI.center = cv::Point (200, 167 );
    stCmd.rectRotatedROI.size = cv::Size(200, 30);
    stCmd.rectRotatedROI.angle = 20;
    stCmd.enAlgorithm = PR_CALIPER_ALGORITHM::PROJECTION;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinerity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinerity = 60.;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "CALIPER ROTATED ROI REGRESSION TEST #5 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;
    stCmd.matInputImg = cv::imread("./data/FindAngle_1.png");
    stCmd.matMask = cv::imread("./data/FindAngle_1_Mask.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectRotatedROI.center = cv::Point2f (420.80f, 268.00f );
    stCmd.rectRotatedROI.size = cv::Size(330, 185);
    stCmd.rectRotatedROI.angle = -12;
    stCmd.enAlgorithm = PR_CALIPER_ALGORITHM::SECTION_AVG_GAUSSIAN_DIFF;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinerity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinerity = 60.;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = -12;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);
}

}
}