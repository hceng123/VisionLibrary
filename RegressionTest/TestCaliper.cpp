#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

void TestCaliper() {
    PR_CALIPER_CMD stCmd;
    PR_CALIPER_RPY stRpy;

    auto PrintRpy = [](const PR_CALIPER_RPY &stRpy) {
        char chArrMsg[100];
        std::cout << "Caliper status " << ToInt32(stRpy.enStatus) << std::endl;
        if (VisionStatus::OK == stRpy.enStatus) {
            std::cout << "ReversedFit = " << stRpy.bReversedFit << std::endl;
            std::cout << std::fixed << std::setprecision(2) << "Line slope = " << stRpy.fSlope << ", intercept = " << stRpy.fIntercept << std::endl;
            _snprintf(chArrMsg, sizeof(chArrMsg), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine.pt1.x, stRpy.stLine.pt1.y, stRpy.stLine.pt2.x, stRpy.stLine.pt2.y);
            std::cout << "Line coordinate: " << chArrMsg << std::endl;
            std::cout << std::fixed << std::setprecision(2) << "Linerity  = " << stRpy.fLinerity << std::endl;
            std::cout << "Linerity check pass: " << stRpy.bLinerityCheckPass << std::endl;
            std::cout << std::fixed << std::setprecision(2) << "Angle  = " << stRpy.fAngle << std::endl;
            std::cout << "Angle check pass: " << stRpy.bAngleCheckPass << std::endl;
        }
    };

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIPER REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    stCmd.rectROI = cv::Rect(1216, 802, 142, 332);
    stCmd.enDetectDir = PR_DETECT_LINE_DIR::MAX_TO_MIN;
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
    stCmd.rectROI = cv::Rect(928, 1276, 270, 120);
    stCmd.enDetectDir = PR_DETECT_LINE_DIR::MIN_TO_MAX;
    stCmd.fExpectedAngle = 0;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIPER REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    stCmd.rectROI = cv::Rect(1496, 1576, 228, 88);
    stCmd.enDetectDir = PR_DETECT_LINE_DIR::MAX_TO_MIN;
    stCmd.fExpectedAngle = 0;
    
    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIPER REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    stCmd.rectROI = cv::Rect(1591, 970, 51, 90);
    stCmd.enDetectDir = PR_DETECT_LINE_DIR::MAX_TO_MIN;
    stCmd.bCheckLinerity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinerity = 60.;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;

    PR_Caliper ( &stCmd, &stRpy );
    PrintRpy(stRpy);
}

}
}