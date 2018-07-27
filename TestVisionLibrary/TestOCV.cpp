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
    
    PR_LrnOcv(&stLrnCmd, &stLrnRpy);

    if (stLrnRpy.enStatus != VisionStatus::OK)
        return;

    PR_OCV_CMD stOcvCmd;
    PR_OCV_RPY stOcvRpy;
    stOcvCmd.matInputImg = stLrnCmd.matInputImg;
    stOcvCmd.rectROI = cv::Rect(2909, 1321, 220, 102);
    //stOcvCmd.rectROI = cv::Rect(2862, 1609, 231, 84);
    stOcvCmd.vecRecordId.push_back(stLrnRpy.nRecordId);

    PR_Ocv(&stOcvCmd, &stOcvRpy);
}