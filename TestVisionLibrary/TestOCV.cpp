#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestOCV_1() {
    PR_LRN_OCV_CMD stLrnCmd;
    PR_LRN_OCV_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/TestOCV.bmp");
    stLrnCmd.rectROI = cv::Rect(2927, 1024, 180, 61);
    stLrnCmd.nCharCount = 4;
    
    PR_LrnOcv(&stLrnCmd, &stLrnRpy);

    std::cout << "PR_LrnOcv status " << ToInt32(stLrnRpy.enStatus) << std::endl;
    if (stLrnRpy.enStatus != VisionStatus::OK)
        return;

    cv::imwrite("./data/LrnOcvResult.png", stLrnRpy.matResultImg);

    PR_OCV_CMD stOcvCmd;
    PR_OCV_RPY stOcvRpy;
    stOcvCmd.matInputImg = stLrnCmd.matInputImg;
    stOcvCmd.rectROI = cv::Rect(2909, 1321, 220, 102);
    //stOcvCmd.rectROI = cv::Rect(2862, 1609, 231, 84);
    stOcvCmd.vecRecordId.push_back(stLrnRpy.nRecordId);

    PR_Ocv(&stOcvCmd, &stOcvRpy);

    std::cout << "PR_Ocv status " << ToInt32(stOcvRpy.enStatus) << std::endl;
    if (stLrnRpy.enStatus != VisionStatus::OK)
        return;

    cv::imwrite("./data/InspOcvResult.png", stOcvRpy.matResultImg);
}

void PrintOcvRpy(const PR_OCV_RPY &stRpy)
{
    std::cout << "PR_Ocv status " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << "Overall score: " << stRpy.fOverallScore << std::endl;
    std::cout << "Individual char score: ";
    for (auto score : stRpy.vecCharScore)
        std::cout << score << " ";
    std::cout << std::endl;
}

void TestOCV_2()
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
    cv::imwrite("./data/LrnOcvResult.png", stLrnRpy.matResultImg);

    PR_GET_RECORD_INFO_RPY stRecordInfo;
    PR_GetRecordInfo(stLrnRpy.nRecordId, &stRecordInfo);
    if (VisionStatus::OK == stRecordInfo.enStatus)
        cv::imwrite("./data/LrnOcvRecordImage.png", stRecordInfo.matImage);

    PR_OCV_CMD stOcvCmd;
    PR_OCV_RPY stOcvRpy;
    stOcvCmd.matInputImg = stLrnCmd.matInputImg;
    stOcvCmd.rectROI = cv::Rect(288, 1075, 217, 142);
    stOcvCmd.vecRecordId.push_back(stLrnRpy.nRecordId);

    PR_Ocv(&stOcvCmd, &stOcvRpy);
    PrintOcvRpy(stOcvRpy);
    cv::imwrite("./data/OcvResultImage.png", stOcvRpy.matResultImg);
}