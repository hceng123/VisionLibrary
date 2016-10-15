#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

using namespace AOI::Vision;

void TestFitLine()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "FIT LINE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_FIT_LINE_CMD stCmd;
    stCmd.matInput = cv::imread("./data/lowangle_250.png", cv::IMREAD_GRAYSCALE);
    stCmd.nThreshold = 200;
    stCmd.rectROI = cv::Rect(139,276,20,400);
    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;

    PR_FIT_LINE_RPY stRpy;
    PR_FitLine ( &stCmd, &stRpy );    
    std::cout << "Fit line status " << stRpy.nStatus << std::endl; 
    std::cout << std::fixed << std::setprecision(1) << "Line slope = " << stRpy.fSlope << ", intercept = " << stRpy.fIntercept << std::endl;
    char chArrMsg[100];
    _snprintf(chArrMsg, sizeof ( chArrMsg ), "(%.1f, %.1f), (%.1f, %.1f)", stRpy.stLine.pt1.x, stRpy.stLine.pt1.y, stRpy.stLine.pt2.x, stRpy.stLine.pt2.y);
    std::cout << "Line coordinate: " << chArrMsg << std::endl;
}