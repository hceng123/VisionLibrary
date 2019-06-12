// TestVisionLibrary.cpp : Defines the entry point for the console application.
//

#include <ctime>
#include <iostream>
#include <iomanip>

#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../RegressionTest/UtilityFunc.h"
#include "TestSub.h"


using namespace AOI::Vision;

int TestVisionAlgorithm()
{
    float WZ = 0, TX = 0, TY = 0;
    printf("Please enter WZ, TX and TY, separated by space.\n");
    printf("Example: -0.01 5 -3\n");
    printf(">");

    // Here we will store our images.
    IplImage* pColorPhoto = 0; // Photo of a butterfly on a flower.
    IplImage* pGrayPhoto = 0;  // Grayscale copy of the photo.
    IplImage* pImgT = 0;       // Template T.
    IplImage* pImgI = 0;       // Image I.

    // Here we will store our warp matrix.
    CvMat* W = 0;  // Warp W(p)

    // Create two simple windows. 
    cvNamedWindow("template"); // Here we will display T(x).
    //cvNamedWindow("image"); // Here we will display I(x).

    // Load photo.
    pColorPhoto = cvLoadImage("./data/photo.jpg");
    if (NULL == pColorPhoto)
        return -1;

    // Create other images.
    CvSize photo_size = cvSize(pColorPhoto->width, pColorPhoto->height);
    pGrayPhoto = cvCreateImage(photo_size, IPL_DEPTH_8U, 1);
    pImgT = cvCreateImage(photo_size, IPL_DEPTH_8U, 1);
    pImgI = cvCreateImage(photo_size, IPL_DEPTH_8U, 1);

    // Convert photo to grayscale, because we need intensity values instead of RGB.    
    cvCvtColor(pColorPhoto, pGrayPhoto, CV_RGB2GRAY);

    // Create warp matrix, we will use it to create image I.
    //W = cvCreateMat(3, 3, CV_32F);
    cv::Mat matRotation = cv::getRotationMatrix2D(cv::Point(210, 175), 3, 1);
    printf("Rotation Mat \n");
    printfMat<double> ( matRotation );
    cv::Mat matRotated;
    cv::Mat src = cv::cvarrToMat ( pGrayPhoto );
    warpAffine(src, matRotated, matRotation, src.size());
    cv::line(matRotated, cv::Point(210, 175), cv::Point(260, 225), cv::Scalar(255,255, 0), 2 );        //Manually add defect.
    circle(matRotated, cv::Point(240, 190), 15, cv::Scalar(255,255, 0), CV_FILLED);

    circle(matRotated, cv::Point(200, 160), 15, cv::Scalar(0,0, 0), CV_FILLED);
    imshow("Rotated", matRotated);

    // Create template T
    // Set region of interest to butterfly's bounding rectangle.
    cvCopy(pGrayPhoto, pImgT);
    CvRect omega = cvRect(110, 100, 200, 150);

    // Create I by warping photo with sub-pixel accuracy. 
    //init_warp(W, WZ, TX, TY);
    //warp_image(pGrayPhoto, pImgI, W);

    // Draw the initial template position approximation rectangle.
    //cvSetIdentity(W);
    //draw_warped_rect(pImgI, omega, W);

    // Show T and I and wait for key press.
    cvSetImageROI(pImgT, omega);
    cvShowImage("template", pImgT);
    //cvShowImage("image", pImgI);
    // printf("Press any key to start Lucas-Kanade algorithm.\n");
    cvWaitKey(0);
    cvResetImageROI(pImgT);

    // Print what parameters were entered by user.
    printf("==========================================================\n");
    printf("Ground truth:  WZ=%f TX=%f TY=%f\n", WZ, TX, TY);
    printf("==========================================================\n");

    //init_warp(W, WZ, TX, TY);

    // Restore image I
    //warp_image(pGrayPhoto, pImgI, W);
    PR_SRCH_OBJ_ALGORITHM enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SURF;
    PR_LRN_OBJ_CMD stLrnTmplCmd;
    PR_LRN_OBJ_RPY stLrnTmplRpy;
    stLrnTmplCmd.matInputImg = cv::cvarrToMat(pImgT, true);
    stLrnTmplCmd.rectLrn = omega;
    stLrnTmplCmd.enAlgorithm = enAlgorithm;
    PR_LrnObj(&stLrnTmplCmd, &stLrnTmplRpy);

    PR_SRCH_OBJ_CMD stSrchTmplCmd;
    PR_SRCH_OBJ_RPY stSrchTmplRpy;
    stSrchTmplCmd.matInputImg = matRotated;
    //stSrchTmplCmd.rectLrn = omega;
    stSrchTmplCmd.nRecordId = stLrnTmplRpy.nRecordId;
    stSrchTmplCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
    cv::Rect rectSrchWindow ( cv::Point(100, 80), cv::Point ( stSrchTmplCmd.matInputImg.cols - 70, stSrchTmplCmd.matInputImg.rows - 60 ) );
    stSrchTmplCmd.rectSrchWindow = rectSrchWindow;
    stSrchTmplCmd.enAlgorithm = enAlgorithm;

    PR_SrchObj(&stSrchTmplCmd, &stSrchTmplRpy);

    printf("Match score %.2f \n", stSrchTmplRpy.fMatchScore );
    printfMat<double>(stSrchTmplRpy.matHomography);

    PR_INSP_SURFACE_CMD stInspCmd;
    PR_INSP_SURFACE_RPY stInspRpy;
    stInspCmd.matTmpl = cv::cvarrToMat(pImgT, true);
    stInspCmd.rectLrn = omega;
    stInspCmd.ptObjPos = stSrchTmplRpy.ptObjPos;
    stInspCmd.fRotation = stSrchTmplRpy.fRotation;
    stInspCmd.matInsp = matRotated;
    stInspCmd.u16NumOfDefectCriteria = 2;
    stInspCmd.astDefectCriteria[0].fArea = 50;
    stInspCmd.astDefectCriteria[0].fLength = 0.f;
    stInspCmd.astDefectCriteria[0].ubContrast = 20;
    stInspCmd.astDefectCriteria[0].enAttribute = PR_DEFECT_ATTRIBUTE::BOTH;

    //PR_InspSurface(&stInspCmd, &stInspRpy);

    stInspCmd.astDefectCriteria[1].fLength = 50.0;
    stInspCmd.astDefectCriteria[1].fArea = 0.f;
    stInspCmd.astDefectCriteria[1].ubContrast = 20;

    PR_SetDebugMode(PR_DEBUG_MODE::SHOW_IMAGE);
    PR_InspSurface ( &stInspCmd, &stInspRpy );

    //PR_AlignCmd stAlignCmd;
    //PR_AlignRpy stAlignRpy;
    //stAlignCmd.matInputImg = matRotated;
    //stAlignCmd.rectLrn = omega;
    //stAlignCmd.rectSrchWindow.x = stFindObjRpy.ptObjPos.x - ( omega.width + 30.f ) / 2.0f;
    //stAlignCmd.rectSrchWindow.y = stFindObjRpy.ptObjPos.y - ( omega.height + 30.f) / 2.0f;
    //stAlignCmd.rectSrchWindow.width = (omega.width + 30.f);
    //stAlignCmd.rectSrchWindow.height = (omega.height + 30.f);
    //stAlignCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
    //cv::Mat matT = cv::cvarrToMat(pImgT);
    //stAlignCmd.matTmpl = cv::Mat(matT, omega);
    //visionAlgorithm.align(&stAlignCmd, &stAlignRpy);

    //printf("SURF and findTransformECC result \n");
    //printfMat<float>(stAlignRpy.matAffine);

    /* Lucas-Kanade */
    //align_image_forwards_additive(pImgT, omega, pImgI);

    //printf("Press any key to start Baker-Dellaert-Matthews algorithm.\n");
    PR_DumpTimeLog("PRTime.log");
    cvWaitKey();
    return 0;
}

void TestSearchFiducialMark()
{
    PR_SRCH_OBJ_ALGORITHM enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SURF;
    cv::Rect2f omega = cv::Rect2f(311, 45, 300, 300 );
    PR_LRN_OBJ_CMD stLrnTmplCmd;
    PR_LRN_OBJ_RPY stLrnTmplRpy;
    stLrnTmplCmd.matInputImg = cv::imread(".\\data\\FiducialLrn.png");
    stLrnTmplCmd.rectLrn = omega;
    stLrnTmplCmd.enAlgorithm = enAlgorithm;
    VisionStatus enStatus = PR_LrnObj(&stLrnTmplCmd, &stLrnTmplRpy);
    if ( VisionStatus::OK != enStatus ) {
        std::cout << "Learn fail with status " << static_cast<int>(enStatus) << std::endl;
        return;
    }

    PR_SRCH_OBJ_CMD stSrchTmplCmd;
    PR_SRCH_OBJ_RPY stSrchTmplRpy;
    stSrchTmplCmd.matInputImg = cv::imread(".\\data\\FiducialSrch_BK.png");
    //stSrchTmplCmd.rectLrn = omega;
    stSrchTmplCmd.nRecordId = stLrnTmplRpy.nRecordId;
    stSrchTmplCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
    cv::Rect rectSrchWindow ( 40, 40, 390, 390 );
    stSrchTmplCmd.rectSrchWindow = rectSrchWindow;
    stSrchTmplCmd.enAlgorithm = enAlgorithm;

    PR_SrchObj(&stSrchTmplCmd, &stSrchTmplRpy);

    printf("Match score %.2f \n", stSrchTmplRpy.fMatchScore );
    printfMat<double>(stSrchTmplRpy.matHomography);

    PR_DumpTimeLog("PRTime.log");
}

void TestSearchFiducialMark_1()
{
    PR_SRCH_OBJ_ALGORITHM enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SURF;
    cv::Rect2f omega = cv::Rect2f(464, 591, 194, 194 );
    PR_LRN_OBJ_CMD stLrnTmplCmd;
    PR_LRN_OBJ_RPY stLrnTmplRpy;
    stLrnTmplCmd.matInputImg = cv::imread(".\\data\\PCB_FiducialMark1.jpg");
    stLrnTmplCmd.rectLrn = omega;
    stLrnTmplCmd.enAlgorithm = enAlgorithm;
    VisionStatus enStatus = PR_LrnObj(&stLrnTmplCmd, &stLrnTmplRpy);
    if ( VisionStatus::OK != enStatus ) {
        std::cout << "Learn fail with status " << static_cast<int>(enStatus) << std::endl;
        return;
    }

    PR_SRCH_OBJ_CMD stSrchTmplCmd;
    PR_SRCH_OBJ_RPY stSrchTmplRpy;
    stSrchTmplCmd.matInputImg = cv::imread(".\\data\\PCB_FiducialMark2.jpg");
    //stSrchTmplCmd.rectLrn = omega;
    stSrchTmplCmd.nRecordId = stLrnTmplRpy.nRecordId;
    stSrchTmplCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
    cv::Rect rectSrchWindow ( 300, 600, 270, 270 );
    stSrchTmplCmd.rectSrchWindow = rectSrchWindow;
    stSrchTmplCmd.enAlgorithm = enAlgorithm;

    PR_SrchObj(&stSrchTmplCmd, &stSrchTmplRpy);

    printf("Match score %.2f \n", stSrchTmplRpy.fMatchScore );
    printfMat<double>(stSrchTmplRpy.matHomography);

    PR_DumpTimeLog("PRTime.log");
}

void TestSearchFiducialMark_2()
{
    cv::Mat mat = cv::imread("./data/F6-313-1-Gray.bmp", cv::IMREAD_GRAYSCALE);
    if ( mat.empty() )  {
        std::cout << "Read input image fail" << std::endl;
        return;
    }

    PR_SRCH_FIDUCIAL_MARK_CMD stCmd;
    stCmd.enType = PR_FIDUCIAL_MARK_TYPE::SQUARE;
    stCmd.fSize = 54;
    stCmd.fMargin = 8;
    stCmd.matInputImg = mat;
    stCmd.rectSrchWindow = cv::Rect(1459,155, 500, 500 );

    PR_SRCH_FIDUCIAL_MARK_RPY stRpy;
    VisionStatus enStatus = PR_SrchFiducialMark(&stCmd, &stRpy);
    std::cout << "Srch fiducial result " << stRpy.ptPos.x << " " << stRpy.ptPos.y << std::endl;
}

void TestSearchFiducialMark_3()
{
    cv::Mat mat = cv::imread("./data/CircleFiducialMark.png", cv::IMREAD_GRAYSCALE);
    if ( mat.empty() )  {
        std::cout << "Read input image fail" << std::endl;
        return;
    }

    PR_SRCH_FIDUCIAL_MARK_CMD stCmd;
    stCmd.enType = PR_FIDUCIAL_MARK_TYPE::CIRCLE;
    stCmd.fSize = 64;
    stCmd.fMargin = 8;
    stCmd.matInputImg = mat;
    stCmd.rectSrchWindow = cv::Rect(0, 40, 250, 250 );

    PR_SRCH_FIDUCIAL_MARK_RPY stRpy;
    VisionStatus enStatus = PR_SrchFiducialMark(&stCmd, &stRpy);
    std::cout << "Srch fiducial result " << stRpy.ptPos.x << " " << stRpy.ptPos.y << std::endl; 
}

void TestInspDevice()
{
    PR_FreeAllRecord();

    VisionStatus enStatus;
    PR_LRN_DEVICE_CMD stLrnDeviceCmd;
    stLrnDeviceCmd.matInputImg = cv::imread("./data/TmplResistor.png");
    stLrnDeviceCmd.bAutoThreshold = true;
    stLrnDeviceCmd.nElectrodeThreshold = 180;
    stLrnDeviceCmd.rectDevice = cv::Rect2f( 39, 27, 104, 65 );
    PR_LRN_DEVICE_RPY stLrnDeviceRpy;
    
    PR_SetDebugMode(PR_DEBUG_MODE::SHOW_IMAGE);
    enStatus = PR_LrnDevice ( &stLrnDeviceCmd, &stLrnDeviceRpy );
    if ( enStatus != VisionStatus::OK )
    {
        std::cout << "Failed to learn device" << std::endl;
        return;
    }

    PR_INSP_DEVICE_CMD stInspDeviceCmd;
    stInspDeviceCmd.matInputImg = cv::imread("./data/RotatedDevice.png");
    stInspDeviceCmd.nElectrodeThreshold = stLrnDeviceRpy.nElectrodeThreshold;
    stInspDeviceCmd.nDeviceCount = 1;
    stInspDeviceCmd.astDeviceInfo[0].nCriteriaNo = 0;
    stInspDeviceCmd.astDeviceInfo[0].rectSrchWindow = cv::Rect ( 33, 18, 96, 76 );
    stInspDeviceCmd.astDeviceInfo[0].stCtrPos = cv::Point(72,44);
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckMissed = true;
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckRotation = true;
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckScale = true;
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckShift = true;
    stInspDeviceCmd.astDeviceInfo[0].stSize = stLrnDeviceRpy.sizeDevice;
    stInspDeviceCmd.astCriteria[0].fMaxOffsetX = 15;
    stInspDeviceCmd.astCriteria[0].fMaxOffsetY = 15;
    stInspDeviceCmd.astCriteria[0].fMaxRotate = 5;
    stInspDeviceCmd.astCriteria[0].fMaxScale = 1.3f;
    stInspDeviceCmd.astCriteria[0].fMinScale = 0.7f;

    PR_INSP_DEVICE_RPY stInspDeviceRpy;
    
    PR_InspDevice( &stInspDeviceCmd, &stInspDeviceRpy );
    std::cout << "Device inspection status " << stInspDeviceRpy.astDeviceResult[0].nStatus << std::endl;

    stInspDeviceCmd.matInputImg = cv::imread("./data/ShiftedDevice.png");
    stInspDeviceCmd.astDeviceInfo[0].stCtrPos = cv::Point(85, 48);
    stInspDeviceCmd.astDeviceInfo[0].stSize = cv::Size2f(99, 52);
    stInspDeviceCmd.astDeviceInfo[0].rectSrchWindow = cv::Rect ( 42, 16, 130, 100 );
    
    PR_InspDevice( &stInspDeviceCmd, &stInspDeviceRpy );
    std::cout << "Device inspection status " << stInspDeviceRpy.astDeviceResult[0].nStatus << std::endl;
}

void TestRunLogcase()
{
    VisionStatus enStatus = PR_RunLogCase(".\\Vision\\Logcase\\LrnTmpl_2016_07_30_14_06_16_125\\");
    std::cout << "Run logcase result " << static_cast<int> ( enStatus ) << std::endl;
}

void TestAutoThreshold() {
    PR_AUTO_THRESHOLD_CMD stCmd;
    PR_AUTO_THRESHOLD_RPY stRpy;
    stCmd.matInputImg = cv::imread ( "./data/Compare.png", cv::IMREAD_GRAYSCALE );
    stCmd.rectROI = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.nThresholdNum = 1;
    PR_AutoThreshold ( &stCmd, &stRpy);
    if ( stRpy.enStatus == VisionStatus::OK )
        std::cout << "Auto theshold result: " << stRpy.vecThreshold[0] << std::endl;
}

std::string GetTime()
{
    std::time_t t; std::time(&t);
    std::tm *p = std::localtime(&t);
    std::string   s = std::asctime(p);
    size_t   k = s.find_first_of("\r\n");
    return (k == std::string::npos) ? s : s.substr(0, k);
}

void TestFitLine()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "FIT LINE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_FIT_LINE_CMD stCmd;
    stCmd.matInputImg = cv::imread("./data/lowangle_250.png", cv::IMREAD_COLOR);
    stCmd.nThreshold = 200;
    stCmd.rectROI = cv::Rect(139,276,20,400);
    stCmd.fErrTol = 8;
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;

    PR_FIT_LINE_RPY stRpy;
    PR_FitLine ( &stCmd, &stRpy );
    std::cout << "Fit line status " << ToInt32( stRpy.enStatus ) << std::endl; 
    std::cout << "Line slope = " << stRpy.fSlope << ", intercept = " << stRpy.fIntercept << std::endl;
    char chArrMsg[100];
    _snprintf(chArrMsg, sizeof ( chArrMsg ), "(%f, %f), (%f, %f)", stRpy.stLine.pt1.x, stRpy.stLine.pt1.y, stRpy.stLine.pt2.x, stRpy.stLine.pt2.y);
    std::cout << "Line coordinate: " << chArrMsg << std::endl;
    cv::line (stCmd.matInputImg, stRpy.stLine.pt1, stRpy.stLine.pt2, cv::Scalar(255, 0, 0 ), 2 );
    cv::imshow("Result image", stCmd.matInputImg);
    cv::waitKey(0);
}

void TestFindEdge()
{
    PR_FIND_EDGE_CMD stCmd;

    stCmd.matInputImg = cv::imread("./data/Resisters1.png");
    stCmd.enDirection = PR_EDGE_DIRECTION::BOTH;
    stCmd.bAutoThreshold = true;
    stCmd.nThreshold = 50;
    stCmd.fMinLength = 20;
    stCmd.rectROI = cv::Rect(163, 189, 46, 97);

    PR_FIND_EDGE_RPY stRpy;
    PR_FindEdge(&stCmd, &stRpy);
    
    std::cout << "Find edge status " << ToInt32 ( stRpy.enStatus ) << std::endl; 
    std::cout << "Edge count = " << stRpy.nEdgeCount << std::endl;
}

void TestFindEdge1()
{
    PR_FIND_EDGE_CMD stCmd;

    stCmd.matInputImg = cv::imread("./data/HalfCircle.png");
    stCmd.enDirection = PR_EDGE_DIRECTION::BOTH;
    stCmd.bAutoThreshold = true;
    stCmd.nThreshold = 50;
    stCmd.fMinLength = 50;
    stCmd.rectROI = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows);

    PR_FIND_EDGE_RPY stRpy;
    PR_FindEdge(&stCmd, &stRpy);
    
    std::cout << "Find edge status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    std::cout << "Edge count = " << stRpy.nEdgeCount << std::endl;
}

void TestFitCircle()
{
    PR_SetDebugMode(PR_DEBUG_MODE::SHOW_IMAGE);

    PR_FIT_CIRCLE_CMD stCmd;
    stCmd.matInputImg = cv::imread("./data/001.bmp");
    stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
    stCmd.fErrTol = 5;
    //stCmd.ptRangeCtr = cv::Point2f(240, 235);
    //stCmd.fRangeInnterRadius = 180;
    //stCmd.fRangeOutterRadius = 210;
    stCmd.bPreprocessed = false;
    stCmd.bAutoThreshold = false;
    stCmd.nThreshold = 200;
    stCmd.enMethod = PR_FIT_METHOD::RANSAC;
    stCmd.nMaxRansacTime = 20;

    PR_FIT_CIRCLE_RPY stRpy;
    VisionStatus visionStatus = PR_FitCircle(&stCmd, &stRpy);
    if (VisionStatus::OK != visionStatus)    {
        std::cout << "Failed to fit circle, VisionStatus = " << ToInt32( stRpy.enStatus ) << std::endl;
        return;
    }
    cv::Mat matResultImg = stCmd.matInputImg.clone();
    //cv::circle(matResultImg, stCmd.ptRangeCtr, (int)stCmd.fRangeInnterRadius, cv::Scalar(0, 255, 0), 1);
    //cv::circle(matResultImg, stCmd.ptRangeCtr, (int)stCmd.fRangeOutterRadius, cv::Scalar(0, 255, 0), 1);
    cv::circle(matResultImg, stRpy.ptCircleCtr, (int)stRpy.fRadius, cv::Scalar(255, 0, 0), 2);
    cv::imshow("Fit result", matResultImg);
    cv::waitKey(0);
}

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

void TestCaliper() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/F1-5-1_Threshold.png");
    //cv::Rect rectROI(1591, 970, 51, 90);
    //cv::Rect rectROI(546, 320, 300, 100 );
    cv::Rect rectROI(998, 1298, 226, 68);
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = PR_CALIPER_DIR::DARK_TO_BRIGHT;
    stCmd.nCaliperCount = 5;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_FindLine ( &stCmd, &stRpy );
    PrintFindLineRpy ( stRpy );
}

void TestCaliper_1() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/TestRotatedCaliper.png");
    //cv::Rect rectROI(1591, 970, 51, 90);
    cv::Rect rectROI(546, 320, 300, 100 );
    stCmd.rectRotatedROI.center = cv::Point (85, 121 );
    stCmd.rectRotatedROI.size = cv::Size(100, 30);
    stCmd.rectRotatedROI.angle = 110;
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::CALIPER;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_FindLine ( &stCmd, &stRpy );
    std::cout << "PR_FindLine status: " << ToInt32 ( stRpy.enStatus ) << std::endl;
}

void TestCaliper_2() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/TestRotatedCaliper.png");
    //cv::Rect rectROI(1591, 970, 51, 90);
    cv::Rect rectROI(546, 320, 300, 100 );
    stCmd.rectRotatedROI.center = cv::Point (171, 34 );
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
    std::cout << "PR_FindLine status: " << ToInt32 ( stRpy.enStatus ) << std::endl;
}

void TestCaliper_NoLine() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/TestRotatedCaliper.png");
    stCmd.rectRotatedROI.center = cv::Point (368, 54 );
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
    std::cout << "PR_FindLine status: " << ToInt32 ( stRpy.enStatus ) << std::endl;
}

void TestCaliper_NoLine_1() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/TestRotatedCaliper.png");
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
    std::cout << "PR_FindLine status: " << ToInt32 ( stRpy.enStatus ) << std::endl;
}

void TestCaliper_3() {
    PR_FIND_LINE_CMD stCmd;
    PR_FIND_LINE_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/FindAngle_1.jpg");
    stCmd.rectRotatedROI.center = cv::Point (404, 270 );
    stCmd.rectRotatedROI.size = cv::Size(300, 50);
    stCmd.rectRotatedROI.angle = -15;
    stCmd.enAlgorithm = PR_FIND_LINE_ALGORITHM::PROJECTION;
    stCmd.enDetectDir = PR_CALIPER_DIR::AUTO;
    stCmd.bCheckLinearity = true;
    stCmd.fPointMaxOffset = 5;
    stCmd.fMinLinearity = 0.6f;
    stCmd.bCheckAngle = true;
    stCmd.fExpectedAngle = 90;
    stCmd.fAngleDiffTolerance = 5;
    
    PR_FindLine ( &stCmd, &stRpy );
    std::cout << "PR_FindLine status: " << ToInt32 ( stRpy.enStatus ) << std::endl;
}

template<typename _tp>
static cv::Mat intervals ( _tp start, _tp interval, _tp end ) {
    std::vector<_tp> vecValue;
    int nSize = ToInt32 ( (end - start) / interval );
    vecValue.reserve ( nSize );
    _tp value = start;
    //end += interval / 2.f;  //Add some margin to prevent 0.999 <= 1.000 problem.
    if (interval > 0) {
        while (value <= end) {
            vecValue.push_back ( value );
            value += interval;
        }
    }
    else {
        while (value >= end) {
            vecValue.push_back ( value );
            value += interval;
        }
    }
    //cv::Mat matResult ( vecValue ); //This have problem, because matResult share the memory with vecValue, after leave this function, the memory already released.
    return cv::Mat ( vecValue ).clone ();
}

void ChangeColor()
{
    cv::Mat matInput = cv::imread("C://Users//shenxiao//Pictures//Icons//CaliperCircle.png", cv::IMREAD_COLOR );
    if ( matInput.empty() )
        return;
    cv::Mat matGray;
    cv::cvtColor ( matInput, matGray, cv::COLOR_BGR2GRAY);
    cv::Mat matCompare = matGray < 100;
    matInput.setTo ( cv::Scalar (241, 152, 71), matCompare );
    cv::imwrite ( "C://Users//shenxiao//Pictures//Icons//CaliperCircle1.png", matInput );
}

void addBridgeIfNotExist(VectorOfRect& vecBridgeWindow, const cv::Rect& rectBridge) {
    auto iterBridge = std::find(vecBridgeWindow.begin(), vecBridgeWindow.end(), rectBridge);
    if (iterBridge == vecBridgeWindow.end())
        vecBridgeWindow.push_back(rectBridge);
}

void TestReshape() {
    std::vector<float> vecInput {
            1,  3,
            7,  11,
            17, 19
        };
    cv::Mat matInput(vecInput);
    cv::Mat matReshape1 = matInput.reshape(1, 3);
    std::cout << "Reshape 1 result: " << std::endl;
    printfMat<float>(matReshape1);

    matInput = matInput.reshape(1, 1);
    cv::Mat matReshape2 = matInput.reshape(1, 3);
    std::cout << "Reshape 2 result: " << std::endl;
    printfMat<float>(matReshape2);
}

int _tmain(int argc, _TCHAR* argv[])
{
    TestReshape();

    cv::Rect rect(1, 5, 10, 10);
    VectorOfRect vecBridgeWindow;
    vecBridgeWindow.push_back(rect);

    cv::Rect rect1(rect);
    addBridgeIfNotExist(vecBridgeWindow, rect1);

    auto matResult = intervals<float>(0, 0.1f, 1.f);

    cv::Mat matNan (3, 3, CV_32FC1, NAN );
    cv::Mat matCmpResult = cv::Mat ( matNan == matNan );
    int nCount = cv::countNonZero ( matNan );
    std::cout << "find NAN cout " << matCmpResult.total() - nCount << std::endl;

    PR_VERSION_INFO stVersionInfo;
    PR_GetVersion(&stVersionInfo);
    std::cout << "VisionLibrary Version: " << stVersionInfo.chArrVersion << std::endl;

    float fTest = -1.2f;
    int nTest = static_cast<int>(fTest);

    auto fFloorResult = std::floor(fTest);

    PR_Init();
    PR_SetDebugMode(PR_DEBUG_MODE::LOG_FAIL_CASE);

    //PR_RunLogCase("./Vision/LogCase/CalibrateCamera_2018_12_21_08_03_14_131.logcase");
    //TestTemplate();
    //TestInspDevice();
    //TestRunLogcase();
    //TestSearchFiducialMark();
    //TestSearchFiducialMark_1();
    //TestSearchFiducialMark_2();
    //TestSearchFiducialMark_3();
    //TestFitLine();

    //TestFindEdge();
    //TestFindEdge1();

    //TestInspDeviceAutoThreshold();
    //TestFitCircle();

    //TestCalibCamera();
    //TestCalibCamera_1();
    //TestCalibCamera_2();
    //TestCalibCamera_3();
    //TestRestoreImage();

    //TestCompareInputAndResult();
    //TestRunRestoreImgLogCase();

    //TestAutoLocateLead();
    //TestAutoLocateLeadTmpl_1();
    //TestAutoLocateLeadTmpl_2();
    //TestAutoLocateLeadTmpl_3();

    //TestInspBridge();
    //TestInspBridge_1();

    //TestAutoThreshold();

    //TestCaliper();
    //TestCaliper_1();
    //TestCaliper_2();
    //TestCaliper_NoLine();
    //TestCaliper_NoLine_1();
    //TestCaliper_3();

    //PR_FreeAllRecord();

    //TestLrnObj();
    //TestLrnObj_1();
    //TestSrchObj();

    //TestInspChipHead();

    //TestInspLead();

    //TestSrchDie();

    //DynamicCalculateDistortion();
    //TestCalcRestoreIdx();
    //ConvertRestoreCsvToYml();
    //TestTableMappingAndCombineImage();

    //TestIntegrate3DCalibHaoYu();

    //TestCalcMTF();

    //PR_RunLogCase("./Vision/Logcase/AutoLocateLead_2018_07_11_10_14_18_277.logcase");
    //PR_RunLogCase("D:/xsg/Logcase/Calib3DBase_2017_09_10_21_13_02_836.logcase");
    //TestCalcPD();

    //TestCalcCameraMTF();
    //TestTwoLineAngle();
    //TestTwoLineAngle_1();
    //TestTwoLineAngle_2();
    //TestTwoLineAngle_3();
    //TestTwoLineAngle_4();
    //TestTwoLineAngle_5();

    //TestPointLineDistance_1();
    //TestParallelLineDistance_1();
    //TestParallelLineDistance_2();
    //TestParallelLineDistance_3();

    //TestLineIntersect();

    //TestCrossSectionArea();

    //TestFindCircle();
    //TestFindCircle_1();

    //TestCombineImage();
    //TestCombineImage_1();
    //TestCombineImage_2();
    //TestCombineImage_HaoYu();

    //ChangeColor();

    //TestCalib3dBase();
    //TestCalib3dBase_1();

    //TestCalib3DHeight_01();
    //TestCalib3DHeight_02();
    //TestCalib3DHeight_03();
    //TestCalc3DHeight();
    //for (int i = 0; i < 10; ++ i)
    //    TestFastCalc3DHeight();
    //TestFastCalc3DHeight_1();

    //TestIntegrate3DCalib();
    //TestCalc3DHeight_With_NormalCalibParam();
    //TestMotor3DCalibNew();

    //TestCalc3DHeight_With_IntegrateCalibParam();
    //TestMerge3DHeight();
    //TestCalibDlpOffset_1();

    //TestMotor3DCalib();

    //ConvertBaseParamCsvToYml();

    //TestCalc4DLPHeight();
    //TestCalc4DLPHeightOnePass();
    //TestCalc4DLPHeight_SimulateMachine();
    //CompareHeightSame();
    //TestScanImage();
    //VeryHeightMergeResult();
    //ComparePatchPhaseResult();
    //CompareMask();
    //TestMergeHeightMax();
    //TestSortIndexValue();

    //TestInsp3DSolder();
    //TestInsp3DSolder_1();
    //TestInsp3DSolder_2();

    //TestCalcFrameValue_1();
    //TestCalcFrameValue_2();
    
    //TestQueryDlpOffset();

    //TestSolve();
    //_PR_InternalTest();

    //PR_FreeAllRecord();
    //TestPrThreshold();

    //TestCalcFrameValue();

    //TestInspPolarity_01();
    
    //TestOCV_1();
    //TestOCV_2();
    //TestOCV_3();

    //TestMatchTmpl_1();

    //TestRead2DCode_1();

    //TestRead2DCode_2();

    //TestDrawDashRect();
    
    //TestTableMapping();
    
    //TestTableMapping_1();
    //TestTableMapping_2();
    //TestTableMapping_3();
    
    //TestCombineImageNew_1();
    //TestCombineImageNew_2();
    //TestCombineMachineImage();
    //FindFramePositionFromBigImage();

    //TestGenerateSelectedImage();

    TestMeasureDist();

    PR_DumpTimeLog("./Vision/Time.log");
    return 0;
}

