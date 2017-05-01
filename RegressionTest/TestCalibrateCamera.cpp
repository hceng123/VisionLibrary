#include <iostream>
#include <iomanip>
#include "UtilityFunc.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"

namespace AOI
{
namespace Vision
{

void TestCalibrateCamera()
{
    auto PrintRpy = [](const PR_CALIBRATE_CAMERA_RPY &stRpy)    {
        char chArrMsg[100];
        std::cout << "Calibrate camera status " << ToInt32(stRpy.enStatus) << std::endl;
        if (VisionStatus::OK == stRpy.enStatus)   {
            //std::cout << "Camera Intrinsic camera:" << std::endl;
            //printfMat<double> ( stRpy.matIntrinsicMatrix );

            //std::cout << "Camera Extrinsic camera:" << std::endl;
            //printfMat<double> ( stRpy.matExtrinsicMatrix );

            _snprintf(chArrMsg, sizeof(chArrMsg), "Camera resolution (%.2f, %.2f)", stRpy.dResolutionX, stRpy.dResolutionY );
            std::cout << chArrMsg << std::endl;      

            std::cout << "Camera distoration coefficients:" << std::endl;
            printfMat<double> ( stRpy.matDistCoeffs );
        }
    };

    PR_CALIBRATE_CAMERA_CMD stCmd;
    PR_CALIBRATE_CAMERA_RPY stRpy;

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIBRATE CAMERA REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInput = cv::imread("./data/ChessBoard.png", cv::IMREAD_GRAYSCALE );
    stCmd.fPatternDist = 2;
    stCmd.szBoardPattern = cv::Size(25, 23);

    PR_CalibrateCamera ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIBRATE CAMERA REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInput = cv::imread("./data/ChessBoard_2.png", cv::IMREAD_GRAYSCALE );
    stCmd.fPatternDist = 2;
    stCmd.szBoardPattern = cv::Size(8, 5);

    PR_CalibrateCamera ( &stCmd, &stRpy );
    PrintRpy(stRpy);

    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIBRATE CAMERA REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;
    
    stCmd.matInput = cv::imread("./data/ChessBoard_3.png", cv::IMREAD_GRAYSCALE );
    stCmd.fPatternDist = 2;
    stCmd.szBoardPattern = cv::Size(8, 5);

    PR_CalibrateCamera ( &stCmd, &stRpy );
    PrintRpy(stRpy);
}

}
}