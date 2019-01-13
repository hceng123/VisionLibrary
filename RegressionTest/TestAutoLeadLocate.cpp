#include <iostream>
#include <iomanip>
#include "UtilityFunc.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"

namespace AOI
{
namespace Vision
{

void TestAutoLocateLead()
{
    auto PrintRpy = [](const PR_AUTO_LOCATE_LEAD_RPY &stRpy) {
        char chArrMsg[100];
        std::cout << "Auto locate lead status " << ToInt32(stRpy.enStatus) << std::endl;
        if (VisionStatus::OK == stRpy.enStatus)   {
            if (stRpy.vecLeadLocation.size() > 0) {
                std::cout << "Lead number: " << stRpy.vecLeadLocation.size() << ", lead locations: " << std::endl;
                for ( const auto &rect : stRpy.vecLeadLocation ) {
                    _snprintf(chArrMsg, sizeof(chArrMsg), "%d, %d, %d, %d", rect.x, rect.y, rect.width, rect.height);
                    std::cout << chArrMsg << std::endl;
                }
            }

            if (stRpy.vecLeadInfo.size() > 0) {
                std::cout << "Lead number: " << stRpy.vecLeadInfo.size() << ", lead search locations: " << std::endl;
                 for ( const auto &leadInfo : stRpy.vecLeadInfo ) {
                    _snprintf(chArrMsg, sizeof(chArrMsg), "%d, %d, %d, %d",
                        leadInfo.rectSrchWindow.x, leadInfo.rectSrchWindow.y, leadInfo.rectSrchWindow.width, leadInfo.rectSrchWindow.height);
                    std::cout << chArrMsg << std::endl;
                }
            }
        }
    };

    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.enMethod = PR_AUTO_LOCATE_LEAD_METHOD::FIND_CONTOUR;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInputImg = cv::imread("./data/F1-7-2.bmp", cv::IMREAD_COLOR);
    stCmd.rectSrchWindow = cv::Rect(2332, 1920, 710, 710);
    stCmd.rectChipBody =   cv::Rect(2389, 1977, 596, 596);
    PR_AutoLocateLead (&stCmd, &stRpy);
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR);
    stCmd.rectSrchWindow = cv::Rect(3262, 301, 306, 314);
    stCmd.rectChipBody =   cv::Rect(3263, 361, 302, 198);
    PR_AutoLocateLead (&stCmd, &stRpy);
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR);
    stCmd.rectSrchWindow = cv::Rect(1579, 2017, 617, 617);
    stCmd.rectChipBody =   cv::Rect(1639, 2077, 497, 497);
    PR_AutoLocateLead (&stCmd, &stRpy);
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInputImg = cv::imread("./data/ChipWithLead.png", cv::IMREAD_COLOR);
    stCmd.rectSrchWindow = cv::Rect(12, 15, 457, 454);
    stCmd.rectChipBody =   cv::Rect(102, 100, 365, 366);
    PR_AutoLocateLead (&stCmd, &stRpy);
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #5 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInputImg = cv::imread("./data/TestAutoLocateLead.png", cv::IMREAD_COLOR);
    stCmd.enMethod = PR_AUTO_LOCATE_LEAD_METHOD::TEMPLATE_MATCH;
    stCmd.rectSrchWindow = cv::Rect(cv::Point(631, 399), cv::Point( 1719, 1491));
    stCmd.rectChipBody =   cv::Rect(cv::Point(860, 630), cv::Point(1491, 1261));
    stCmd.rectPadWindow = cv::Rect(cv::Point(898, 446), cv::Point(946, 544));
    stCmd.rectLeadWindow = cv::Rect(cv::Point(900, 544), cv::Point(944, 630));
    stCmd.vecSrchLeadDirections.push_back(PR_DIRECTION::RIGHT);
    stCmd.vecSrchLeadDirections.push_back(PR_DIRECTION::UP);
    stCmd.vecSrchLeadDirections.push_back(PR_DIRECTION::DOWN);
    stCmd.vecSrchLeadDirections.push_back(PR_DIRECTION::LEFT);
    PR_AutoLocateLead(&stCmd, &stRpy);
    PrintRpy(stRpy);
    if (VisionStatus::OK == stRpy.enStatus)
        cv::imwrite("./data/TestAutoLocateLead_Result.png", stRpy.matResultImg);
}

}
}