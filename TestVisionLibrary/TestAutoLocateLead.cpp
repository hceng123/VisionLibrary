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
    stCmd.matInputImg = cv::imread("./data/F1-7-2.bmp", cv::IMREAD_COLOR);
    stCmd.rectSrchWindow = cv::Rect(2332, 1920, 710, 710);
    stCmd.rectChipBody = cv::Rect(2389, 1977, 596, 596);
    PR_AutoLocateLead (&stCmd, &stRpy);
}

void TestAutoLocateLead_1()
{
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR);
    stCmd.rectSrchWindow = cv::Rect(3262, 301, 306, 314);
    stCmd.rectChipBody =   cv::Rect(3263, 361, 302, 198);
    PR_AutoLocateLead (&stCmd, &stRpy);
}

void TestAutoLocateLead_2()
{
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR);
    stCmd.rectSrchWindow = cv::Rect(1869, 477, 721, 505);
    stCmd.rectChipBody =   cv::Rect(1965, 505, 549, 461);
    PR_AutoLocateLead (&stCmd, &stRpy);
}

void TestAutoLocateLead_3()
{
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR);
    stCmd.rectSrchWindow = cv::Rect(1579, 2017, 617, 617);
    stCmd.rectChipBody =   cv::Rect(1639, 2077, 497, 497);
    PR_AutoLocateLead (&stCmd, &stRpy);
}

void TestAutoLocateLeadTmpl_1()
{
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/ScanBigImage.bmp", cv::IMREAD_COLOR);
    stCmd.enMethod = PR_AUTO_LOCATE_LEAD_METHOD::TEMPLATE_MATCH;
    stCmd.rectSrchWindow = cv::Rect(181, 475, 1060, 1060);
    stCmd.rectChipBody =   cv::Rect(400, 700, 620, 620);
    stCmd.rectPadWindow = cv::Rect(440, 497, 42, 92);
    stCmd.rectLeadWindow = cv::Rect(448, 596, 100, 89);
    stCmd.vecSrchLeadDirections.push_back(PR_DIRECTION::RIGHT);
    stCmd.vecSrchLeadDirections.push_back(PR_DIRECTION::UP);
    stCmd.vecSrchLeadDirections.push_back(PR_DIRECTION::DOWN);
    stCmd.vecSrchLeadDirections.push_back(PR_DIRECTION::LEFT);
    PR_AutoLocateLead(&stCmd, &stRpy);
    std::cout << "PR_AutoLocateLead status " << ToInt32(stRpy.enStatus) << std::endl;
    cv::imwrite("./data/AutoLocateLeadResult.bmp", stRpy.matResultImg);
    if (stRpy.enStatus != VisionStatus::OK)
        return;

    PR_INSP_LEAD_TMPL_CMD stInspLeadCmd;
    PR_INSP_LEAD_TMPL_RPY stInspLeadRpy;
    stInspLeadCmd.matInputImg = stCmd.matInputImg;
    stInspLeadCmd.rectROI = stRpy.vecLeadInfo[0].rectSrchWindow;
    stInspLeadCmd.nLeadRecordId = stRpy.vecLeadInfo[0].nLeadRecordId;
    stInspLeadCmd.nPadRecordId = stRpy.vecLeadInfo[0].nPadRecordId;
    stInspLeadCmd.fLrnedPadLeadDist = abs((stRpy.vecLeadInfo[0].rectPadWindow.y + stRpy.vecLeadInfo[0].rectPadWindow.height / 2) - (stRpy.vecLeadInfo[0].rectLeadWindow.y + stRpy.vecLeadInfo[0].rectLeadWindow.height / 2));
    stInspLeadCmd.fMaxLeadOffsetX = 5;
    stInspLeadCmd.fMaxLeadOffsetY = 5;
    PR_InspLeadTmpl(&stInspLeadCmd, &stInspLeadRpy);
    std::cout << "PR_InspLeadTmpl status " << ToInt32(stRpy.enStatus) << std::endl;
    cv::imwrite("./data/InspLeadResult.bmp", stInspLeadRpy.matResultImg);
}