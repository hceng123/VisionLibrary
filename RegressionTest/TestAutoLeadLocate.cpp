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
    auto PrintRpy = [](const PR_AUTO_LOCATE_LEAD_RPY &stRpy)    {
        char chArrMsg[100];
        std::cout << "Auto locate lead status " << ToInt32(stRpy.enStatus) << std::endl;
        std::cout << "Lead number: " << stRpy.vecLeadLocation.size() << ", lead locations: " << std::endl;
        if (VisionStatus::OK == stRpy.enStatus)   {
            for ( const auto &rect : stRpy.vecLeadLocation ) {
                _snprintf(chArrMsg, sizeof(chArrMsg), "%d, %d, %d, %d", rect.x, rect.y, rect.width, rect.height);
                std::cout << chArrMsg << std::endl;
            }
        }
    };

    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInputImg = cv::imread("./data/F1-7-2.bmp", cv::IMREAD_COLOR );
    stCmd.rectSrchWindow = cv::Rect(2332, 1920, 710, 710);
    stCmd.rectChipBody =   cv::Rect(2389, 1977, 596, 596);
    PR_AutoLocateLead (&stCmd, &stRpy);
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR );
    stCmd.rectSrchWindow = cv::Rect(3262, 301, 306, 314);
    stCmd.rectChipBody =   cv::Rect(3263, 361, 302, 198);
    PR_AutoLocateLead (&stCmd, &stRpy);
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "AUTO LOCATE LEAD REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInputImg = cv::imread("./data/F8-58-2.bmp", cv::IMREAD_COLOR );
    stCmd.rectSrchWindow = cv::Rect(1579, 2017, 617, 617);
    stCmd.rectChipBody =   cv::Rect(1639, 2077, 497, 497);
    PR_AutoLocateLead (&stCmd, &stRpy);
    PrintRpy(stRpy);

}

}
}