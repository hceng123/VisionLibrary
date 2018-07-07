#include <iostream>
#include <iomanip>
#include "UtilityFunc.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"

namespace AOI
{
namespace Vision
{

void TestInspBridge() {
    auto PrintRpy = [](const PR_INSP_BRIDGE_RPY &stRpy) {
        char chArrMsg[100];
        std::cout << "Inspect bridge status " << ToInt32(stRpy.enStatus) << std::endl;
        int nBridgeWindowIndex = 0;
        for (const auto &rectBridge : stRpy.vecBridgeWindow) {
            _snprintf(chArrMsg, sizeof(chArrMsg), "%d, %d, %d, %d", rectBridge.x, rectBridge.y, rectBridge.width, rectBridge.height);
            std::cout << "Bridge window " << nBridgeWindowIndex << " : " << chArrMsg << std::endl;
            ++ nBridgeWindowIndex;
        }
    };

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSPECT BRIDGE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    {
        PR_INSP_BRIDGE_CMD stCmd;
        PR_INSP_BRIDGE_RPY stRpy;
        stCmd.matInputImg = cv::imread("./data/TestInspBridge.png", cv::IMREAD_GRAYSCALE);

        stCmd.enInspMode = PR_INSP_BRIDGE_MODE::OUTER;
        stCmd.rectROI = cv::Rect(221, 195, 28, 82);
        stCmd.rectOuterSrchWindow = cv::Rect(212, 187, 43, 102);
        for (int i = 0; i < 4; ++i)
            stCmd.vecOuterInspDirection.push_back(PR_DIRECTION(i));

        PR_InspBridge(&stCmd, &stRpy);
        PrintRpy(stRpy);
    }

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSPECT BRIDGE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    {
        PR_INSP_BRIDGE_CMD stCmd;
        PR_INSP_BRIDGE_RPY stRpy;
        stCmd.matInputImg = cv::imread("./data/TestInspBridge.png", cv::IMREAD_GRAYSCALE);

        stCmd.enInspMode = PR_INSP_BRIDGE_MODE::OUTER;
        stCmd.rectROI = cv::Rect(169, 195, 26, 82);
        stCmd.rectOuterSrchWindow = cv::Rect(158, 187, 54, 108);
        for (int i = 0; i < 4; ++i)
            stCmd.vecOuterInspDirection.push_back(PR_DIRECTION(i));

        PR_InspBridge(&stCmd, &stRpy);
        PrintRpy(stRpy);
    }

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSPECT BRIDGE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    {
        PR_INSP_BRIDGE_CMD stCmd;
        PR_INSP_BRIDGE_RPY stRpy;
        stCmd.matInputImg = cv::imread("./data/F1-7-4.png", cv::IMREAD_GRAYSCALE);
        stCmd.enInspMode = PR_INSP_BRIDGE_MODE::INNER;
        stCmd.rectROI = cv::Rect(1539, 1858, 24, 108);
        stCmd.stInnerInspCriteria.fMaxLengthX = 15;
        stCmd.stInnerInspCriteria.fMaxLengthY = 50;

        PR_InspBridge(&stCmd, &stRpy);
        PrintRpy(stRpy);
    }

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSPECT BRIDGE REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    {
        PR_INSP_BRIDGE_CMD stCmd;
        PR_INSP_BRIDGE_RPY stRpy;
        stCmd.matInputImg = cv::imread("./data/F1-7-4.png", cv::IMREAD_GRAYSCALE);
        stCmd.enInspMode = PR_INSP_BRIDGE_MODE::INNER;
        stCmd.rectROI = cv::Rect(1607, 1862, 24, 108);
        stCmd.stInnerInspCriteria.fMaxLengthX = 15;
        stCmd.stInnerInspCriteria.fMaxLengthY = 50;

        PR_InspBridge(&stCmd, &stRpy);
        PrintRpy(stRpy);
    }

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSPECT BRIDGE REGRESSION TEST #5 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    {
        PR_INSP_BRIDGE_CMD stCmd;
        PR_INSP_BRIDGE_RPY stRpy;
        stCmd.matInputImg = cv::imread("./data/F1-7-4.png", cv::IMREAD_GRAYSCALE);
        stCmd.enInspMode = PR_INSP_BRIDGE_MODE::INNER;
        stCmd.rectROI = cv::Rect(1542, 2048, 24, 108);
        stCmd.stInnerInspCriteria.fMaxLengthX = 15;
        stCmd.stInnerInspCriteria.fMaxLengthY = 50;

        PR_InspBridge(&stCmd, &stRpy);
        PrintRpy(stRpy);
    }
}

}
}