#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestInspBridge() {
    PR_INSP_BRIDGE_CMD stCmd;
    PR_INSP_BRIDGE_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/TestInspBridge.png", cv::IMREAD_GRAYSCALE);
    stCmd.enInspMode = PR_INSP_BRIDGE_MODE::OUTER;
    //stCmd.rectROI = cv::Rect(221, 195, 28, 82);
    //stCmd.rectOuterSrchWindow = cv::Rect(212, 187, 43, 102);
    stCmd.rectROI = cv::Rect(169, 195, 28, 82);
    stCmd.rectOuterSrchWindow = cv::Rect(158, 187, 45, 108);
    for (int i = 0; i < 4; ++ i)
        stCmd.vecOuterInspDirection.push_back(PR_DIRECTION(i));
    
    PR_InspBridge(&stCmd, &stRpy);
    if (VisionStatus::OK != stRpy.enStatus) {
        std::cout << "Failed to inspect bridge. Error Status: " << ToInt32(stRpy.enStatus) << std::endl;
        return;
    }

    if (!stRpy.matResultImg.empty())
        cv::imwrite("./data/TestInspBridge_Result.png", stRpy.matResultImg);
}

void TestInspBridge_1() {
    PR_INSP_BRIDGE_CMD stCmd;
    PR_INSP_BRIDGE_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/F1-7-4.png", cv::IMREAD_COLOR);
    stCmd.enInspMode = PR_INSP_BRIDGE_MODE::INNER;
    stCmd.rectROI = cv::Rect(1539, 1858, 24, 108);
    stCmd.stInnerInspCriteria.fMaxLengthX = 15;
    stCmd.stInnerInspCriteria.fMaxLengthY = 50;

    //inspItem.rectInnerWindow = cv::Rect(1607, 1862, 24, 108);
    //stCmd.vecInspItems.push_back ( inspItem );

    //inspItem.rectInnerWindow = cv::Rect(1542, 2048, 24, 108);
    //stCmd.vecInspItems.push_back ( inspItem );
    
    PR_InspBridge(&stCmd, &stRpy);
    if ( VisionStatus::OK != stRpy.enStatus ) {
        std::cout << "Failed to inspect bridge. Error Status: " << ToInt32(stRpy.enStatus) << std::endl;
        return;
    }
    cv::imwrite("./data/TestInspBridge_Result.png", stRpy.matResultImg );
}