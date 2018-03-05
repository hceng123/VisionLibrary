#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

using namespace AOI::Vision;

void TestInspPolarity_01() {
    PR_INSP_POLARITY_CMD stCmd;
    PR_INSP_POLARITY_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/TestInspPolarity.png");
    stCmd.rectInspROI = cv::Rect ( 47, 144, 162, 28 );
    stCmd.rectCompareROI = cv::Rect ( 47, 339, 162, 28 );
    stCmd.enInspROIAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stCmd.nGrayScaleDiffTol = 80;

    PR_InspPolarity ( &stCmd, &stRpy );

    std::cout << "PR_InspChip Status: " << ToInt32(stRpy.enStatus) << std::endl;
    cv::imwrite("./data/TestInspPolarityResult.png", stRpy.matResultImg );
}