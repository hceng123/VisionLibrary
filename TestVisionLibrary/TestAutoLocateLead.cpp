#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestAutoLocateLead()
{
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/F1-7-2.bmp", cv::IMREAD_COLOR );
    stCmd.rectSrchWindow = cv::Rect(2332, 1920, 710, 710);
    stCmd.rectChipBody = cv::Rect(2389, 1977, 596, 596 );
    PR_AutoLocateLead (&stCmd, &stRpy);
}

void TestAutoLocateLead_1()
{
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR );
    stCmd.rectSrchWindow = cv::Rect(3262, 301, 306, 314);
    stCmd.rectChipBody =   cv::Rect(3263, 361, 302, 198);
    PR_AutoLocateLead (&stCmd, &stRpy);
}

void TestAutoLocateLead_2()
{
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR );
    stCmd.rectSrchWindow = cv::Rect(1869, 477, 721, 505);
    stCmd.rectChipBody =   cv::Rect(1965, 505, 549, 461);
    PR_AutoLocateLead (&stCmd, &stRpy);
}

void TestAutoLocateLead_3()
{
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR );
    stCmd.rectSrchWindow = cv::Rect(1579, 2017, 617, 617);
    stCmd.rectChipBody =   cv::Rect(1639, 2077, 497, 497);
    PR_AutoLocateLead (&stCmd, &stRpy);
}