#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>

namespace AOI
{
namespace Vision
{

void TestInspDevice()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "INSPECT DEVICE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_FreeAllRecord();

    VisionStatus enStatus;
    PR_LRN_DEVICE_CMD stLrnDeviceCmd;
    stLrnDeviceCmd.matInput = cv::imread(".\\data\\TmplResistor.png");
    stLrnDeviceCmd.bAutoThreshold = false;
    stLrnDeviceCmd.nElectrodeThreshold = 150;
    stLrnDeviceCmd.rectDevice = cv::Rect2f( 38, 22, 78, 41 );
    PR_LRN_DEVICE_RPY stLrnDeviceRpy;
    
    PR_SetDebugMode(PR_DEBUG_MODE::DISABLED);
    enStatus = PR_LrnDevice ( &stLrnDeviceCmd, &stLrnDeviceRpy );
    if ( enStatus != VisionStatus::OK )
    {
        std::cout << "Failed to learn device" << std::endl;
        return;
    }
    
    std::cout << "Success to learn device" << std::endl;

    PR_INSP_DEVICE_CMD stInspDeviceCmd;
    stInspDeviceCmd.matInput = cv::imread(".\\data\\RotatedDevice.png");
    stInspDeviceCmd.nElectrodeThreshold = stLrnDeviceRpy.nElectrodeThreshold;
    stInspDeviceCmd.nDeviceCount = 1;
    stInspDeviceCmd.astDeviceInfo[0].nCriteriaNo = 0;
    stInspDeviceCmd.astDeviceInfo[0].rectSrchWindow = cv::Rect ( 33, 18, 96, 76 );
    stInspDeviceCmd.astDeviceInfo[0].stCtrPos = cv::Point(72,44);
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckMissed = true;
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckRotation = true;
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckScale = true;
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckShift = true;
    stInspDeviceCmd.astDeviceInfo[0].stSize = stLrnDeviceRpy.sizeDevice;
    stInspDeviceCmd.astCriteria[0].fMaxOffsetX = 15;
    stInspDeviceCmd.astCriteria[0].fMaxOffsetY = 15;
    stInspDeviceCmd.astCriteria[0].fMaxRotate = 5;
    stInspDeviceCmd.astCriteria[0].fMaxScale = 1.3f;
    stInspDeviceCmd.astCriteria[0].fMinScale = 0.7f;

    PR_INSP_DEVICE_RPY stInspDeviceRpy;    
    PR_InspDevice( &stInspDeviceCmd, &stInspDeviceRpy );
    std::cout << "Device inspection status " << stInspDeviceRpy.astDeviceResult[0].nStatus << std::endl;

    stInspDeviceCmd.matInput = cv::imread(".\\data\\ShiftedDevice.png");
    stInspDeviceCmd.astDeviceInfo[0].stCtrPos = cv::Point(85, 48);
    stInspDeviceCmd.astDeviceInfo[0].stSize = cv::Size2f(99, 52);
    stInspDeviceCmd.astDeviceInfo[0].rectSrchWindow = cv::Rect ( 42, 16, 130, 100 );
    
    PR_InspDevice( &stInspDeviceCmd, &stInspDeviceRpy );
    std::cout << "Device inspection status " << stInspDeviceRpy.astDeviceResult[0].nStatus << std::endl;
}

void TestInspDeviceAutoThreshold()
{
    std::cout << std::endl << "---------------------------------------------------------";
    std::cout << std::endl << "INSPECT DEVICE AUTO THRESHOLD REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------------------";
    std::cout << std::endl;

    PR_FreeAllRecord();

    VisionStatus enStatus;
    PR_LRN_DEVICE_CMD stLrnDeviceCmd;
    stLrnDeviceCmd.matInput = cv::imread(".\\data\\TmplResistor.png");
    stLrnDeviceCmd.bAutoThreshold = true;
    stLrnDeviceCmd.nElectrodeThreshold = 0;
    stLrnDeviceCmd.rectDevice = cv::Rect2f( 38, 22, 78, 41 );
    PR_LRN_DEVICE_RPY stLrnDeviceRpy;
    
    PR_SetDebugMode(PR_DEBUG_MODE::DISABLED);
    enStatus = PR_LrnDevice ( &stLrnDeviceCmd, &stLrnDeviceRpy );
    if ( enStatus != VisionStatus::OK )
    {
        std::cout << "Failed to learn device" << std::endl;
        return;
    }
    
    std::cout << "Success to learn device" << std::endl;

    PR_INSP_DEVICE_CMD stInspDeviceCmd;
    stInspDeviceCmd.matInput = cv::imread(".\\data\\RotatedDevice.png");
    stInspDeviceCmd.nElectrodeThreshold = stLrnDeviceRpy.nElectrodeThreshold;
    stInspDeviceCmd.nDeviceCount = 1;
    stInspDeviceCmd.astDeviceInfo[0].nCriteriaNo = 0;
    stInspDeviceCmd.astDeviceInfo[0].rectSrchWindow = cv::Rect ( 33, 18, 96, 76 );
    stInspDeviceCmd.astDeviceInfo[0].stCtrPos = cv::Point(72,44);
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckMissed = true;
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckRotation = true;
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckScale = true;
    stInspDeviceCmd.astDeviceInfo[0].stInspItem.bCheckShift = true;
    stInspDeviceCmd.astDeviceInfo[0].stSize = stLrnDeviceRpy.sizeDevice;
    stInspDeviceCmd.astCriteria[0].fMaxOffsetX = 15;
    stInspDeviceCmd.astCriteria[0].fMaxOffsetY = 15;
    stInspDeviceCmd.astCriteria[0].fMaxRotate = 5;
    stInspDeviceCmd.astCriteria[0].fMaxScale = 1.3f;
    stInspDeviceCmd.astCriteria[0].fMinScale = 0.7f;

    PR_INSP_DEVICE_RPY stInspDeviceRpy;    
    PR_InspDevice( &stInspDeviceCmd, &stInspDeviceRpy );
    std::cout << "Device inspection status " << stInspDeviceRpy.astDeviceResult[0].nStatus << std::endl;

    stInspDeviceCmd.matInput = cv::imread(".\\data\\ShiftedDevice.png");
    stInspDeviceCmd.astDeviceInfo[0].stCtrPos = cv::Point(85, 48);
    stInspDeviceCmd.astDeviceInfo[0].stSize = cv::Size2f(99, 52);
    stInspDeviceCmd.astDeviceInfo[0].rectSrchWindow = cv::Rect ( 42, 16, 130, 100 );
    
    PR_InspDevice( &stInspDeviceCmd, &stInspDeviceRpy );
    std::cout << "Device inspection status " << stInspDeviceRpy.astDeviceResult[0].nStatus << std::endl;
}

}
}