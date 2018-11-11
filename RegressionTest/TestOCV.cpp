#include <iostream>
#include <iomanip>
#include "UtilityFunc.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"

namespace AOI
{
namespace Vision
{

void PrintOcvRpy(const PR_OCV_RPY &stRpy)
{
    std::cout << "PR_Ocv status " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Overall score: " << stRpy.fOverallScore << std::endl;
    std::cout << "Individual char score: ";
    for (auto score : stRpy.vecCharScore)
        std::cout << score << " ";
    std::cout << std::endl;
}

void TestOCV_1()
{
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "OCV REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_LRN_OCV_CMD stLrnCmd;
    PR_LRN_OCV_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/TestOCV.png");
    stLrnCmd.rectROI = cv::Rect(311, 818, 180, 61);
    stLrnCmd.nCharCount = 4;
    
    PR_LrnOcv(&stLrnCmd, &stLrnRpy);
    std::cout << "PR_LrnOcv status " << ToInt32(stLrnRpy.enStatus) << std::endl;
    if (stLrnRpy.enStatus != VisionStatus::OK)
        return;

    PR_OCV_CMD stOcvCmd;
    PR_OCV_RPY stOcvRpy;
    stOcvCmd.matInputImg = stLrnCmd.matInputImg;
    stOcvCmd.rectROI = cv::Rect(288, 1075, 217, 142);
    stOcvCmd.vecRecordId.push_back(stLrnRpy.nRecordId);

    PR_Ocv(&stOcvCmd, &stOcvRpy);
    PrintOcvRpy(stOcvRpy);
}

void TestOCV_2()
{
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "OCV REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_LRN_OCV_CMD stLrnCmd;
    PR_LRN_OCV_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/TestOCV_2.png");
    stLrnCmd.rectROI = cv::Rect(148, 206, 89, 152);
    stLrnCmd.enDirection = PR_DIRECTION::UP;
    stLrnCmd.nCharCount = 3;

    PR_LrnOcv(&stLrnCmd, &stLrnRpy);
    std::cout << "PR_LrnOcv status " << ToInt32(stLrnRpy.enStatus) << std::endl;
    if (stLrnRpy.enStatus != VisionStatus::OK)
        return;

    PR_OCV_CMD stOcvCmd;
    PR_OCV_RPY stOcvRpy;
    stOcvCmd.matInputImg = cv::imread("./data/TestOCV_2.png");
    stOcvCmd.rectROI = cv::Rect(141, 903, 120, 296);
    stOcvCmd.enDirection = PR_DIRECTION::UP;
    stOcvCmd.vecRecordId.push_back(stLrnRpy.nRecordId);
    PR_Ocv(&stOcvCmd, &stOcvRpy);
    PrintOcvRpy(stOcvRpy);
}

void TestOCV()
{
    TestOCV_1();
    TestOCV_2();
}

}
}