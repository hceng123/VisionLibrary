#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

void TestFitCircle()
{
    PR_FIT_CIRCLE_CMD stCmd;
    PR_FIT_CIRCLE_RPY stRpy;
    auto PrintRpy = [](const PR_FIT_CIRCLE_RPY &stRpy)    {
        char chArrMsg[100];
        std::cout << "Fit circle status " << ToInt32( stRpy.enStatus ) << std::endl;
        std::cout << std::fixed << std::setprecision(2) << "Radius = " << stRpy.fRadius << std::endl;
        _snprintf(chArrMsg, sizeof(chArrMsg), "(%.2f, %.2f)", stRpy.ptCircleCtr.x, stRpy.ptCircleCtr.y);
        std::cout << "Circle center: " << chArrMsg << std::endl;
    };

    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "FIT CIRCLE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInput = cv::imread("./data/lowangle_250.png");
	stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
	stCmd.fErrTol = 5;
    stCmd.rectROI = cv::Rect ( 407, 373, 226, 226 );
    stCmd.matMask = cv::Mat::ones ( stCmd.matInput.size(), CV_8UC1 ) * PR_MAX_GRAY_LEVEL;
    cv::circle ( stCmd.matMask, cv::Point(520, 487), 90, cv::Scalar(0), CV_FILLED );
    cv::rectangle ( stCmd.matMask, cv::Rect(493, 590, 51, 21), CV_FILLED, CV_FILLED );
    stCmd.bPreprocessed = false;
	stCmd.bAutoThreshold = false;
	stCmd.nThreshold = 210;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
	stCmd.enMethod = PR_FIT_CIRCLE_METHOD::LEAST_SQUARE_REFINE;
    
	PR_FitCircle(&stCmd, &stRpy);
    PrintRpy(stRpy);

    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "FIT CIRCLE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    stCmd.matInput = cv::imread("./data/Fitting Test.png");
	stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
	stCmd.fErrTol = 5;
    stCmd.rectROI = cv::Rect(464, 106, 214, 214 );
    stCmd.matMask = cv::Mat::ones(stCmd.matInput.size(), CV_8UC1) * PR_MAX_GRAY_LEVEL;
    cv::circle ( stCmd.matMask, cv::Point(571, 213), 76, cv::Scalar(0), CV_FILLED );
    stCmd.bPreprocessed = false;
	stCmd.bAutoThreshold = false;
	stCmd.nThreshold = 200;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::DARK;
	stCmd.enMethod = PR_FIT_CIRCLE_METHOD::LEAST_SQUARE_REFINE;
    
	PR_FitCircle(&stCmd, &stRpy);    
    PrintRpy(stRpy);
}

void TestFitLine()
{
    PR_FIT_LINE_CMD stCmd;
    PR_FIT_LINE_RPY stRpy;
    auto PrintRpy = [](const PR_FIT_LINE_RPY &stRpy)    {
        char chArrMsg[100];
        std::cout << "Fit line status " << ToInt32(stRpy.enStatus) << std::endl;
        std::cout << "ReversedFit = "   << stRpy.bReversedFit      << std::endl;
        std::cout << std::fixed << std::setprecision(2) << "Line slope = " << stRpy.fSlope << ", intercept = " << stRpy.fIntercept << std::endl;
        _snprintf(chArrMsg, sizeof(chArrMsg), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine.pt1.x, stRpy.stLine.pt1.y, stRpy.stLine.pt2.x, stRpy.stLine.pt2.y);
        std::cout << "Line coordinate: " << chArrMsg << std::endl;
    };

    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "FIT LINE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInput = cv::imread("./data/lowangle_250.png", cv::IMREAD_GRAYSCALE);
    stCmd.bPreprocessed = false;
    stCmd.nThreshold = 200;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stCmd.rectROI = cv::Rect(139,276,20,400);
    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    
    PR_FitLine ( &stCmd, &stRpy );    
    PrintRpy(stRpy);

    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "FIT LINE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInput = cv::imread("./data/Fitting Test.png", cv::IMREAD_GRAYSCALE);
    stCmd.bPreprocessed = false;
    stCmd.nThreshold = 200;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::DARK;
    stCmd.rectROI = cv::Rect(233, 37, 166, 111);
    stCmd.fErrTol = 5;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    
    PR_FitLine ( &stCmd, &stRpy );    
    PrintRpy(stRpy);

    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "FIT LINE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInput = cv::imread("./data/Fitting Test.png", cv::IMREAD_GRAYSCALE);
    stCmd.nThreshold = 200;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::DARK;
    stCmd.rectROI = cv::Rect(97, 56, 96, 130);
    stCmd.fErrTol = 5;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    
    PR_FitLine ( &stCmd, &stRpy );    
    PrintRpy(stRpy);
}

void TestFitParellelLine()
{
    PR_FIT_PARALLEL_LINE_CMD stCmd;
    PR_FIT_PARALLEL_LINE_RPY stRpy;
    auto PrintRpy = [](const PR_FIT_PARALLEL_LINE_RPY &stRpy)    {
        char chArrMsg[100];
        std::cout << "Fit parallel line status " << ToInt32(stRpy.enStatus) << std::endl;
        std::cout << "ReversedFit = " << stRpy.bReversedFit << std::endl;
        std::cout << std::fixed << std::setprecision(2) << "Line slope = " << stRpy.fSlope << ", intercept1 = " << stRpy.fIntercept1 << ", intercept2 = " << stRpy.fIntercept2 << std::endl;
        _snprintf(chArrMsg, sizeof(chArrMsg), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine1.pt1.x, stRpy.stLine1.pt1.y, stRpy.stLine1.pt2.x, stRpy.stLine1.pt2.y);
        std::cout << "Line 1 coordinate: " << chArrMsg << std::endl;
        _snprintf(chArrMsg, sizeof(chArrMsg), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine2.pt1.x, stRpy.stLine2.pt1.y, stRpy.stLine2.pt2.x, stRpy.stLine2.pt2.y);
        std::cout << "Line 2 coordinate: " << chArrMsg << std::endl;
    };

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FIT PARALLEL LINE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;    

    stCmd.matInput = cv::imread("./data/lowangle_250.png", cv::IMREAD_GRAYSCALE);
    stCmd.nThreshold = 200;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stCmd.rectArrROI[0] = cv::Rect(139,276,20,400);
    stCmd.rectArrROI[1] = cv::Rect(205,276,20,400);
    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    
    PR_FitParallelLine ( &stCmd, &stRpy );    
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FIT PARALLEL LINE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInput = cv::imread("./data/lowangle_250.png");
    stCmd.nThreshold = 200;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stCmd.rectArrROI[0] = cv::Rect(444, 291, 149, 25);
    stCmd.rectArrROI[1] = cv::Rect(333, 709, 369, 30);
    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;

    PR_FitParallelLine ( &stCmd, &stRpy );    
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FIT PARALLEL LINE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInput = cv::imread("./data/Fitting Test.png");
    stCmd.nThreshold = 200;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::DARK;
    stCmd.rectArrROI[0] = cv::Rect(97, 56, 96, 130);
    stCmd.rectArrROI[1] = cv::Rect(334,205,103,127);
    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;

    PR_FitParallelLine ( &stCmd, &stRpy );    
    PrintRpy(stRpy);
}

void TestFitRect()
{
    PR_FIT_RECT_CMD stCmd;
    PR_FIT_RECT_RPY stRpy;    

    auto PrintRpy = [](const PR_FIT_RECT_RPY &stRpy)    {
        char chArrMsg[100];
        std::cout << "Fit rect status " << ToInt32(stRpy.enStatus) << std::endl;
        if (VisionStatus::OK == stRpy.enStatus)   {
            std::cout << "LineOneReversedFit = " << stRpy.bLineOneReversedFit << ", LineTwoReversedFit = " << stRpy.bLineTwoReversedFit << std::endl;
            std::cout << std::fixed << std::setprecision(2) << "Line slope1 = " << stRpy.fSlope1 << ", slope2 = " << stRpy.fSlope2 << std::endl;
            _snprintf(chArrMsg, sizeof(chArrMsg), "%.2f, %.2f, %.2f, %.2f", stRpy.fArrIntercept[0], stRpy.fArrIntercept[1], stRpy.fArrIntercept[2], stRpy.fArrIntercept[3]);
            std::cout << std::fixed << std::setprecision(2) << "Intercept = " << chArrMsg << std::endl;

            for (int i = 0; i < PR_RECT_EDGE_COUNT; ++i)    {
                _snprintf(chArrMsg, sizeof(chArrMsg), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.arrLines[i].pt1.x, stRpy.arrLines[i].pt1.y, stRpy.arrLines[i].pt2.x, stRpy.arrLines[i].pt2.y);
                std::cout << "Line " << i + 1 << " coordinate: " << chArrMsg << std::endl;
            }
        }
    };

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FIT RECT REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;    

    stCmd.matInput = cv::imread("./data/lowangle_250.png");
    stCmd.nThreshold = 210;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stCmd.rectArrROI[0] = cv::Rect(274,284,20, 395);
    stCmd.rectArrROI[1] = cv::Rect(740,297,28, 374);
    stCmd.rectArrROI[2] = cv::Rect(444,291,149,25);
    stCmd.rectArrROI[3] = cv::Rect(333,709,369,30);

    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    
    PR_FitRect ( &stCmd, &stRpy );    
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FIT RECT REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInput = cv::imread("./data/lowangle_250.png");
    stCmd.nThreshold = 210;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stCmd.rectArrROI[0] = cv::Rect(444,291,149,25);
    stCmd.rectArrROI[1] = cv::Rect(333,709,369,30);
    stCmd.rectArrROI[2] = cv::Rect(274,284,20, 395);
    stCmd.rectArrROI[3] = cv::Rect(740,297,28, 374);

    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    
    PR_FitRect ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FIT RECT REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInput = cv::imread("./data/Fitting Test.png");
    stCmd.nThreshold = 200;
    stCmd.enAttribute = PR_OBJECT_ATTRIBUTE::DARK;
    stCmd.rectArrROI[0] = cv::Rect(97, 56, 96, 130);
    stCmd.rectArrROI[1] = cv::Rect(334,205,103,127);
    stCmd.rectArrROI[2] = cv::Rect(100,218,208,106);
    stCmd.rectArrROI[3] = cv::Rect(240,40, 178,117);

    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    
    PR_FitRect ( &stCmd, &stRpy );
    PrintRpy(stRpy);
}

}
}