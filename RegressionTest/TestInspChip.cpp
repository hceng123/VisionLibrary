#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

static void PrintResult(const PR_INSP_CHIP_RPY &stRpy) {
    char chArrMsg[100];
    std::cout << "Inspect chip status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        _snprintf ( chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f)", stRpy.rotatedRectResult.center.x, stRpy.rotatedRectResult.center.y );
        std::cout << "Center: " << chArrMsg << std::endl;
        _snprintf ( chArrMsg, sizeof ( chArrMsg ), "(%.2f, %.2f)", stRpy.rotatedRectResult.size.width, stRpy.rotatedRectResult.size.height );
        std::cout << "Size: " << chArrMsg << std::endl;
        std::cout << std::fixed << std::setprecision ( 2 ) << "Angle  = " << stRpy.rotatedRectResult.angle << std::endl;
    }
};

void TestInspChipHead() {
    PR_INSP_CHIP_CMD stCmd;
    PR_INSP_CHIP_RPY stRpy;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSP CHIP HEAD REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/InspChip.png");
    stCmd.enInspMode = PR_INSP_CHIP_MODE::HEAD;
    stCmd.nRecordId = -1;
    stCmd.rectSrchWindow = cv::Rect( cv::Point(1452, 1590), cv::Point(1732, 2062));
    PR_InspChip(&stCmd, &stRpy);
    PrintResult ( stRpy );

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSP CHIP HEAD REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/InspChip.png");
    stCmd.enInspMode = PR_INSP_CHIP_MODE::HEAD;
    stCmd.nRecordId = -1;
    stCmd.rectSrchWindow = cv::Rect( cv::Point(1456, 1538), cv::Point(1748, 2118));
    PR_InspChip(&stCmd, &stRpy);
    PrintResult ( stRpy );
}

void TestInspChipSquare() {
    PR_INSP_CHIP_CMD stCmd;
    PR_INSP_CHIP_RPY stRpy;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSP CHIP SQUARE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/InspChip.png");
    stCmd.enInspMode = PR_INSP_CHIP_MODE::SQUARE;
    stCmd.nRecordId = -1;
    stCmd.rectSrchWindow = cv::Rect( cv::Point(2296, 1942), cv::Point(2924, 2654));
    PR_InspChip(&stCmd, &stRpy);
    PrintResult ( stRpy );
}

void TestInspChipBody() {
    PR_INSP_CHIP_CMD stCmd;
    PR_INSP_CHIP_RPY stRpy;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSP CHIP BODY REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/PCB_With_Resistor.png");
    stCmd.enInspMode = PR_INSP_CHIP_MODE::BODY;
    stCmd.nRecordId = -1;
    stCmd.rectSrchWindow = cv::Rect( cv::Point(1038, 1189), cv::Point(1102, 1288));
    PR_InspChip(&stCmd, &stRpy);
    PrintResult ( stRpy );
}

void TestInspChipCAE() {
    PR_INSP_CHIP_CMD stCmd;
    PR_INSP_CHIP_RPY stRpy;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSP CHIP CAE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/PCB_With_CAE.png");
    stCmd.enInspMode = PR_INSP_CHIP_MODE::CAE;
    stCmd.nRecordId = -1;
    stCmd.rectSrchWindow = cv::Rect( cv::Point(1420, 982), cv::Point(2368, 2046));
    PR_InspChip(&stCmd, &stRpy);
    PrintResult ( stRpy );
}

void TestInspChipCircular() {
    PR_INSP_CHIP_CMD stCmd;
    PR_INSP_CHIP_RPY stRpy;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSP CHIP CIRCULAR REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    stCmd.matInputImg = cv::imread("./data/PCB_With_Circular.png");
    stCmd.enInspMode = PR_INSP_CHIP_MODE::CIRCULAR;
    stCmd.nRecordId = -1;
    stCmd.rectSrchWindow = cv::Rect( cv::Point(1280, 1578), cv::Point(2360, 2730));
    PR_InspChip(&stCmd, &stRpy);
    PrintResult ( stRpy );
}

}
}