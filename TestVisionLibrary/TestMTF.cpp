#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include "TestSub.h"

using namespace AOI::Vision;

void TestCalcCameraMTF() {
    PR_CALC_CAMERA_MTF_CMD stCmd;
    PR_CALC_CAMERA_MTF_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/MTF_Target.png", cv::IMREAD_GRAYSCALE );
    stCmd.rectVBigPatternROI = cv::Rect ( 508, 744, 206, 210 );
    stCmd.rectHBigPatternROI = cv::Rect ( 508, 954, 206, 210 );
    stCmd.rectVSmallPatternROI = cv::Rect ( 989, 983, 27, 27 );
    stCmd.rectHSmallPatternROI = cv::Rect ( 989, 1010, 27, 27 );
    PR_CalcCameraMTF ( &stCmd, &stRpy );
    std::cout << "PR_CalcCameraMTF status " << ToInt32( stRpy.enStatus ) << std::endl;
}