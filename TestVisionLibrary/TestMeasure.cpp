#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestTwoLineAngle() {
    PR_TWO_LINE_ANGLE_CMD stCmd;
    PR_TWO_LINE_ANGLE_RPY stRpy;

    stCmd.line1.pt1 = cv::Point ( 0, 0 );
    stCmd.line1.pt2 = cv::Point ( 10, 10 );

    stCmd.line2.pt1 = cv::Point ( 0, 0 );
    stCmd.line2.pt2 = cv::Point ( 10, -10 );
    PR_TwoLineAngle ( &stCmd, &stRpy );
    std::cout << "Two line angle " << stRpy.fAngle << std::endl;
}