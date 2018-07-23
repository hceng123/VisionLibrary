#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestOCV_1() {
    PR_LRN_OCV_CMD stLrnCmd;
    PR_LRN_OCV_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/TestOCV.bmp");
    stLrnCmd.rectROI = cv::Rect(2927, 1024, 180, 61);
    stLrnCmd.nCharCount = 4;
    
    PR_LrnOCV(&stLrnCmd, &stLrnRpy);
}