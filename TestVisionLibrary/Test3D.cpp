#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestCalib3dBase() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = "./data/0715184554_10ms_80_Plane1/";
    PR_CALIB_3D_BASE_CMD stCmd;
    PR_CALIB_3D_BASE_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bGuassianFilter = true;
    stCmd.bReverseSeq = true;
    PR_Calib3DBase ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DBase status " << ToInt32( stRpy.enStatus ) << std::endl;
}