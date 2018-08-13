#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestRead2DCode_1()
{
    PR_READ_2DCODE_CMD stCmd;
    PR_READ_2DCODE_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/BarcodeImage/image4.bmp");
    stCmd.rectROI = cv::Rect(686, 658, 240, 232);
    stCmd.nEdgeThreshold = 35;

    PR_Read2DCode(&stCmd, &stRpy);

    std::cout << "PR_Read2DCode status " << ToInt32(stRpy.enStatus) << std::endl;
    if (stRpy.enStatus != VisionStatus::OK)
        return;

    std::cout << "Read 2D code result " << stRpy.strReadResult << std::endl;
}