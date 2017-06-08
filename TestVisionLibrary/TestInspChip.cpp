#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

using namespace AOI::Vision;

void TestInspChipHead() {
    PR_INSP_CHIP_CMD stCmd;
    PR_INSP_CHIP_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/InspChip.png");
    stCmd.enInspMode = PR_INSP_CHIP_MODE::HEAD;
    stCmd.nRecordId = -1;
    stCmd.rectSrchWindow = cv::Rect( cv::Point(1452, 1590), cv::Point(1732, 2062));
    PR_InspChip(&stCmd, &stRpy);
}