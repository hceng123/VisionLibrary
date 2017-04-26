#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"

using namespace AOI::Vision;

void TestCalibCamera()
{
    PR_CALIBRATE_CAMERA_CMD stCmd;
    PR_CALIBRATE_CAMERA_RPY stRpy;
    stCmd.matInput = cv::imread("./data/ChessBoard.png", cv::IMREAD_GRAYSCALE );
    stCmd.fPatternDist = 2.;
    stCmd.szBoardPattern = cv::Size(25, 23);

    PR_CalibrateCamera ( &stCmd, &stRpy );
}