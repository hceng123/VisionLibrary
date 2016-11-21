#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
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
    std::cout << std::fixed << std::setprecision(2) << "Line slope = " << stRpy.fSlope << ", intercept = " << stRpy.fIntercept << std::endl;
    char chArrMsg[100];
    _snprintf(chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine.pt1.x, stRpy.stLine.pt1.y, stRpy.stLine.pt2.x, stRpy.stLine.pt2.y);
    std::cout << "Line coordinate: " << chArrMsg << std::endl;
}

void TestFitParellelLine()
{
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FIT PARALLEL LINE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_FIT_PARALLEL_LINE_CMD stCmd;
    PR_FIT_PARALLEL_LINE_RPY stRpy;
    char chArrMsg[100];

    stCmd.matInput = cv::imread("./data/lowangle_250.png", cv::IMREAD_GRAYSCALE);
    stCmd.nThreshold = 200;
    stCmd.rectArrROI[0] = cv::Rect(139,276,20,400);
    stCmd.rectArrROI[1] = cv::Rect(205,276,20,400);
    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    
    PR_FitParallelLine ( &stCmd, &stRpy );    
    std::cout << "Fit line status " << stRpy.nStatus << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Line slope = " << stRpy.fSlope << ", intercept1 = " << stRpy.fIntercept1 << ", intercept2 = " << stRpy.fIntercept2 << std::endl;
    _snprintf(chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine1.pt1.x, stRpy.stLine1.pt1.y, stRpy.stLine1.pt2.x, stRpy.stLine1.pt2.y);
    std::cout << "Line 1 coordinate: " << chArrMsg << std::endl;
    _snprintf(chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine2.pt1.x, stRpy.stLine2.pt1.y, stRpy.stLine2.pt2.x, stRpy.stLine2.pt2.y);
    std::cout << "Line 2 coordinate: " << chArrMsg << std::endl;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FIT PARALLEL LINE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    std::cout << std::endl;
    stCmd.matInput = cv::imread("./data/lowangle_250.png");
    stCmd.nThreshold = 200;
    stCmd.rectArrROI[0] = cv::Rect(444,291,149,25);
    stCmd.rectArrROI[1] = cv::Rect(333,709,369,30);
    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;

    PR_FitParallelLine ( &stCmd, &stRpy );    
    std::cout << "Fit line status " << stRpy.nStatus << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Line slope = " << stRpy.fSlope << ", intercept1 = " << stRpy.fIntercept1 << ", intercept2 = " << stRpy.fIntercept2 << std::endl;
    
    _snprintf(chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine1.pt1.x, stRpy.stLine1.pt1.y, stRpy.stLine1.pt2.x, stRpy.stLine1.pt2.y);
    std::cout << "Line 1 coordinate: " << chArrMsg << std::endl;
    _snprintf(chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine2.pt1.x, stRpy.stLine2.pt1.y, stRpy.stLine2.pt2.x, stRpy.stLine2.pt2.y);
    std::cout << "Line 2 coordinate: " << chArrMsg << std::endl;
}

void TestFitRect()
{
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FIT RECT REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_FIT_RECT_CMD stCmd;
    PR_FIT_RECT_RPY stRpy;
    char chArrMsg[100];

    stCmd.matInput = cv::imread("./data/lowangle_250.png");
    stCmd.nThreshold = 210;
    stCmd.rectArrROI[0] = cv::Rect(274,284,20, 395);
    stCmd.rectArrROI[1] = cv::Rect(740,297,28, 374);
    stCmd.rectArrROI[2] = cv::Rect(444,291,149,25);
    stCmd.rectArrROI[3] = cv::Rect(333,709,369,30);

    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    
    PR_FitRect ( &stCmd, &stRpy );    
    std::cout << "Fit rect status " << stRpy.nStatus << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Line slope1 = " << stRpy.fSlope1 << ", slope2 = " << stRpy.fSlope2 << std::endl;
    _snprintf(chArrMsg, sizeof ( chArrMsg ), "%.2f, %.2f, %.2f, %.2f", stRpy.fArrIntercept[0], stRpy.fArrIntercept[1], stRpy.fArrIntercept[2], stRpy.fArrIntercept[3]);
    std::cout << std::fixed << std::setprecision(2) << "Intercept = " << chArrMsg << std::endl;
    
    for (int i = 0; i < PR_RECT_EDGE_COUNT; ++i)    {
        _snprintf(chArrMsg, sizeof(chArrMsg), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.fArrLine [ i ].pt1.x, stRpy.fArrLine [ i ].pt1.y, stRpy.fArrLine [ i ].pt2.x, stRpy.fArrLine [ i ].pt2.y);
        std::cout << "Line " << i + 1 << " coordinate: " << chArrMsg << std::endl;
    }
}