#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

static void PrintFindLineRpy(const PR_FIND_LINE_RPY &stRpy) {
    char chArrMsg[100];
    std::cout << "Find line status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus) {
        PR_GET_ERROR_INFO_RPY stErrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrRpy);
        std::cout << "Error Msg: " << stErrRpy.achErrorStr << std::endl;
        if (PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR == stErrRpy.enErrorLevel)
            return;
    }

    std::cout << "ReversedFit = " << stRpy.bReversedFit << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Line slope = " << stRpy.fSlope << ", intercept = " << stRpy.fIntercept << std::endl;
    _snprintf(chArrMsg, sizeof (chArrMsg), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine.pt1.x, stRpy.stLine.pt1.y, stRpy.stLine.pt2.x, stRpy.stLine.pt2.y);
    std::cout << "Line coordinate: " << chArrMsg << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Linerity  = " << stRpy.fLinearity << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Angle  = " << stRpy.fAngle << std::endl;
};


void TestFindLineByProjection() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::PROJECTION;

    std::cout << std::endl << "---------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY PROJECTION REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    cv::Rect rectROI = cv::Rect(1216, 802, 142, 332);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::BRIGHT_TO_DARK;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY PROJECTION REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(928, 1276, 270, 120);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::DARK_TO_BRIGHT;
    stCmd.fExpectedAngle = 0;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY PROJECTION REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(1496, 1576, 228, 88);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::BRIGHT_TO_DARK;
    stCmd.fExpectedAngle = 0;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY PROJECTION REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "---------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(1591, 970, 51, 90);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::BRIGHT_TO_DARK;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;

    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);
}

void TestFindLineByCaliper() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;    

    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY CALIPER REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/CognexEdge.png");
    cv::Rect rectROI = cv::Rect(304, 732, 208, 532);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::CALIPER;
    stCmd.enDetectDir = PR_CALIPER_DIR::BRIGHT_TO_DARK;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY CALIPER REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    rectROI = cv::Rect(728, 744, 164, 632);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::DARK_TO_BRIGHT;
    stCmd.fExpectedAngle = 90;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY CALIPER REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(998, 1298, 226, 68);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::DARK_TO_BRIGHT;
    stCmd.fExpectedAngle = 0;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY CALIPER REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    rectROI = cv::Rect(952, 1820, 188, 70);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::BRIGHT_TO_DARK;
    stCmd.fExpectedAngle = 0;

    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);
}

void TestFindLineRoatedROI() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;

    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY CALIPER ROTATED ROI REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/TestRotatedCaliper.png");
    stCmd.rectRotatedROI.center = cv::Point (85, 121 );
    stCmd.rectRotatedROI.size = cv::Size(100, 30);
    stCmd.rectRotatedROI.angle = -70;
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::CALIPER;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = -70;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY CALIPER ROTATED ROI REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl;

    stCmd.rectRotatedROI.center = cv::Point (171, 34 );
    stCmd.rectRotatedROI.size = cv::Size(200, 30);
    stCmd.rectRotatedROI.angle = 20;
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::PROJECTION;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 20;
    stCmd.fAngleDiffTolerance = 5;
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY CALIPER ROTATED ROI REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl;
    stCmd.rectRotatedROI.center = cv::Point (368, 54);
    stCmd.rectRotatedROI.size = cv::Size(200, 30);
    stCmd.rectRotatedROI.angle = 20;
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::CALIPER;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY CALIPER ROTATED ROI REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl;
    stCmd.rectRotatedROI.center = cv::Point (200, 167 );
    stCmd.rectRotatedROI.size = cv::Size(200, 30);
    stCmd.rectRotatedROI.angle = 20;
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::PROJECTION;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);

    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl << "FIND LINE BY CALIPER ROTATED ROI REGRESSION TEST #5 STARTING";
    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl;
    stCmd.matInputImg = cv::imread("./data/FindAngle_1.png");
    stCmd.matMask = cv::imread("./data/FindAngle_1_Mask.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectRotatedROI.center = cv::Point2f (420.80f, 268.00f );
    stCmd.rectRotatedROI.size = cv::Size(330, 185);
    stCmd.rectRotatedROI.angle = -12;
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::CALIPER;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = -12;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy(stRpy);
}

void TestFindLinePair() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;
    stCmd.bFindPair = true;

    std::cout << std::endl << "------------------------------------------------------";
    std::cout << std::endl << "FIND LINE PAIR  BY CALIPER REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    stCmd.matMask = cv::Mat::ones ( stCmd.matInputImg.size(), CV_8UC1 );
    cv::Mat matMaskedROI(stCmd.matMask, cv::Rect (1310, 1422, 180, 252 ) );
    matMaskedROI.setTo(0);  //Maked out the center area.
    stCmd.rectRotatedROI.center = cv::Point2f (1393.00f, 1555.00f );
    stCmd.rectRotatedROI.size = cv::Size(537, 209);
    stCmd.rectRotatedROI.angle = 0;
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::CALIPER;
    stCmd.nCaliperCount = 10;
    stCmd.fCaliperWidth = 20;

    PR_FindLine ( &stCmd, &stRpy );
    char chArrMsg[100];
    std::cout << "Find line status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if (VisionStatus::OK == stRpy.enStatus) {
        std::cout << "ReversedFit = " << stRpy.bReversedFit << std::endl;
        std::cout << std::fixed << std::setprecision ( 2 ) << "Line slope = " << stRpy.fSlope << ", intercept = " << stRpy.fIntercept << ", intercept2 = " << stRpy.fIntercept2 << std::endl;
        _snprintf ( chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine.pt1.x, stRpy.stLine.pt1.y, stRpy.stLine.pt2.x, stRpy.stLine.pt2.y );
        std::cout << "Line coordinate: " << chArrMsg << std::endl;
        _snprintf ( chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f), (%.2f, %.2f)", stRpy.stLine2.pt1.x, stRpy.stLine2.pt1.y, stRpy.stLine2.pt2.x, stRpy.stLine2.pt2.y );
        std::cout << "Line2 coordinate: " << chArrMsg << std::endl;
        std::cout << "Two line distance " << stRpy.fDistance << std::endl;
    }else {
        PR_GET_ERROR_INFO_RPY stErrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrRpy);
        std::cout << "Error Msg: " << stErrRpy.achErrorStr << std::endl;
    }
}

static void PrintFindCircleRpy(const PR_FIND_CIRCLE_RPY &stRpy)    {
    char chArrMsg[100];
    std::cout << "Find circle status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus ) {
        PR_GET_ERROR_INFO_RPY stErrorInfo;
        PR_GetErrorInfo ( stRpy.enStatus, &stErrorInfo );
        std::cout << "Error message " << stErrorInfo.achErrorStr << std::endl;
        return;
    }
        
    _snprintf ( chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f)", stRpy.ptCircleCtr.x, stRpy.ptCircleCtr.y );
    std::cout << "Circle center: " << chArrMsg << std::endl;
    std::cout << std::fixed << std::setprecision ( 2 ) << "Radius = " << stRpy.fRadius << std::endl;
    std::cout << std::fixed << std::setprecision ( 2 ) << "Radius2 = " << stRpy.fRadius2 << std::endl;
};

void TestFindCircle() {
    PR_FIND_CIRCLE_CMD stCmd;
    PR_FIND_CIRCLE_RPY stRpy;

    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl << "CALIPER FIND CIRCLE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread ("./data/RoundContourInsp.png", cv::IMREAD_GRAYSCALE );
    stCmd.ptExpectedCircleCtr = cv::Point(130, 123);
    stCmd.enInnerAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stCmd.bFindCirclePair = false;
    stCmd.fMinSrchRadius = 78.f;
    stCmd.fMaxSrchRadius = 110.f;
    stCmd.fStartSrchAngle = 50.f;
    stCmd.fEndSrchAngle = -230.f;    
    stCmd.nCaliperCount = 20;
    stCmd.fCaliperWidth = 20.f;

    PR_FindCircle ( &stCmd, &stRpy );
    PrintFindCircleRpy ( stRpy );

    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl << "CALIPER FIND CIRCLE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread ("./data/RoundContourInsp.png", cv::IMREAD_GRAYSCALE );
    stCmd.bFindCirclePair = true;
    stCmd.enInnerAttribute = PR_OBJECT_ATTRIBUTE::DARK;    
    stCmd.ptExpectedCircleCtr = cv::Point(129, 127);
    stCmd.fMinSrchRadius = 38;
    stCmd.fMaxSrchRadius = 114;
    stCmd.fStartSrchAngle = 50.f;
    stCmd.fEndSrchAngle = -230.f;    
    stCmd.nCaliperCount = 20;
    stCmd.fCaliperWidth = 20.f;
    stCmd.fRmStrayPointRatio = 0.2f;

    PR_FindCircle ( &stCmd, &stRpy );
    PrintFindCircleRpy ( stRpy );

    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl << "CALIPER FIND CIRCLE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "-------------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread ("./data/RoundContourInsp.png", cv::IMREAD_GRAYSCALE );
    //Test mask out all the points, the result should fail.
    stCmd.matMask = cv::Mat::zeros ( stCmd.matInputImg.size(), CV_8UC1 );
    stCmd.enInnerAttribute = PR_OBJECT_ATTRIBUTE::DARK;
    stCmd.ptExpectedCircleCtr = cv::Point(129, 127);
    stCmd.fMinSrchRadius = 38;
    stCmd.fMaxSrchRadius = 114;
    stCmd.fStartSrchAngle = 50.f;
    stCmd.fEndSrchAngle = -230.f;    
    stCmd.nCaliperCount = 20;
    stCmd.fCaliperWidth = 20.f;
    stCmd.fRmStrayPointRatio = 0.2f;

    PR_FindCircle ( &stCmd, &stRpy );
    PrintFindCircleRpy ( stRpy );
}

void TestCaliper() {
    TestFindLineByProjection();
    TestFindLineByCaliper();
    TestFindLineRoatedROI();
    TestFindLinePair();
    TestFindCircle();
}

}
}