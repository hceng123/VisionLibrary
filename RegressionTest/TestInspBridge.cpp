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
    auto PrintRpy = [](const PR_INSP_BRIDGE_RPY &stRpy)    {
        char chArrMsg[100];
        std::cout << "Inspect bridge status " << ToInt32(stRpy.enStatus) << std::endl;
        int nInspResultIndex = 0;
        for ( const auto &inspResult : stRpy.vecInspResults ) {
            std::cout << "Inspect Item " << nInspResultIndex << ". With Bridge: " << inspResult.bWithBridge << std::endl;
            int nBridgeWindowIndex = 0;
            for (const auto &rectBridge : inspResult.vecBridgeWindow) {
                _snprintf(chArrMsg, sizeof(chArrMsg), "%d, %d, %d, %d", rectBridge.x, rectBridge.y, rectBridge.width, rectBridge.height);
                std::cout << "Bridge winow " << nBridgeWindowIndex << " : " << chArrMsg << std::endl;
                ++ nBridgeWindowIndex;
            }
            ++ nInspResultIndex;
        }
    };

    PR_INSP_BRIDGE_CMD stCmd;
    PR_INSP_BRIDGE_RPY stRpy;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSPECT BRIDGE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/TestInspBridge.png", cv::IMREAD_GRAYSCALE);
    PR_INSP_BRIDGE_CMD::INSP_ITEM inspItem;
         
    inspItem.enMode = PR_INSP_BRIDGE_MODE::OUTER;
    inspItem.rectInnerWindow = cv::Rect(221, 195, 28, 82);
    inspItem.rectOuterWindow = cv::Rect(212, 187, 43, 102);
    for ( int i = 0; i < 4; ++ i ) {
        PR_INSP_BRIDGE_CMD::INSP_CRITERIA inspCriteria;
        inspCriteria.enDirection = PR_INSP_BRIDGE_DIRECTION(i);
        inspItem.vecInspCriteria.push_back ( inspCriteria );
    }
    stCmd.vecInspItems.push_back ( inspItem );

    inspItem.rectInnerWindow = cv::Rect(169, 195, 26, 82);
    inspItem.rectOuterWindow = cv::Rect(158, 187, 54, 108);
    stCmd.vecInspItems.push_back ( inspItem ); 
   
    PR_InspBridge(&stCmd, &stRpy);
    PrintRpy(stRpy);
}

}
}