#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

void TestDetectLine()
{
    PR_DETECT_LINE_CMD stCmd;
    PR_DETECT_LINE_RPY stRpy;

    auto PrintRpy = [](const PR_DETECT_LINE_RPY &stRpy)    {
        char chArrMsg[100];
        std::cout << "Fit rect status " << ToInt32(stRpy.enStatus) << std::endl;
        if (VisionStatus::OK == stRpy.enStatus)   {
            std::cout << "ReversedFit = " << stRpy.bReversedFit << std::endl;
            std::cout << std::fixed << std::setprecision(2) << "Line slope = " << stRpy.fSlope << ", intercept = " << stRpy.fIntercept << std::endl;
            _snprintf(chArrMsg, sizeof(chArrMsg), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine.pt1.x, stRpy.stLine.pt1.y, stRpy.stLine.pt2.x, stRpy.stLine.pt2.y);
            std::cout << "Line coordinate: " << chArrMsg << std::endl;            
        }
    };

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "DETECT LINE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInput = cv::imread("./data/F1-5-1_Threshold.png");
    stCmd.rectROI = cv::Rect(1216, 802, 142, 332);
    stCmd.enDetectDir = PR_DETECT_LINE_DIR::MAX_TO_MIN;
    
    PR_DetectLine ( &stCmd, &stRpy );    
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "DETECT LINE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInput = cv::imread("./data/F1-5-1_Threshold.png");
    stCmd.rectROI = cv::Rect(928, 1276, 270, 120);
    stCmd.enDetectDir = PR_DETECT_LINE_DIR::MIN_TO_MAX;
    
    PR_DetectLine ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "DETECT LINE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInput = cv::imread("./data/F1-5-1_Threshold.png");
    stCmd.rectROI = cv::Rect(1496, 1576, 228, 88);
    stCmd.enDetectDir = PR_DETECT_LINE_DIR::MAX_TO_MIN;
    
    PR_DetectLine ( &stCmd, &stRpy );
    PrintRpy(stRpy);
}

}
}