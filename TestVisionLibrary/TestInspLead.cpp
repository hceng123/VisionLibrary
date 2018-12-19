#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

using namespace AOI::Vision;

void TestInspLead() {
    PR_INSP_LEAD_CMD stCmd;
    PR_INSP_LEAD_RPY stRpy;
    stCmd.matInputImg = cv::imread ( "./data/F4-26-2_InspLead.bmp");
    stCmd.rectChipWindow.angle = 0.f;
    cv::Rect rectChip(2830, 1611, 375, 492 );
    stCmd.rectChipWindow.center.x = ToFloat ( rectChip.x + rectChip.width  / 2 );
    stCmd.rectChipWindow.center.y = ToFloat ( rectChip.y + rectChip.height / 2 );
    stCmd.rectChipWindow.size = rectChip.size();

    PR_INSP_BRIDGE_CMD::INNER_INSP_CRITERIA stCriteria;

    PR_INSP_LEAD_CMD::LEAD_INPUT_INFO stLeadInput;
    cv::Rect rectLead_1(2674, 1637, 149, 62 );
    stLeadInput.rectSrchWindow = cv::RotatedRect( cv::Point ( rectLead_1.x + rectLead_1.width / 2, rectLead_1.y + rectLead_1.height / 2 ), rectLead_1.size(), 0.f );
    stCmd.vecLeads.push_back ( stLeadInput );

    cv::Rect rectLead_2(3202, 1766, 136, 54 );
    stLeadInput.rectSrchWindow = cv::RotatedRect( cv::Point ( rectLead_2.x + rectLead_2.width / 2, rectLead_2.y + rectLead_2.height / 2 ), rectLead_2.size(), 0.f );
    stCmd.vecLeads.push_back ( stLeadInput );

    cv::Rect rectLead_3(2894, 1545, 35, 60 );
    stLeadInput.rectSrchWindow = cv::RotatedRect( cv::Point ( rectLead_3.x + rectLead_3.width / 2, rectLead_3.y + rectLead_3.height / 2 ), rectLead_3.size(), 0.f );
    stCmd.vecLeads.push_back ( stLeadInput );

    PR_InspLead ( &stCmd, &stRpy );
    if (!stRpy.matResultImg.empty()) {
        cv::imwrite("./data/LeadInspResult.png", stRpy.matResultImg);
    }
}