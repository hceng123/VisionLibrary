#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>

using namespace AOI::Vision;

void TestTmplMatch()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "TEMPLATE MATCH REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    cv::Mat mat = cv::imread("./data/F6-313-1-Gray.bmp", cv::IMREAD_GRAYSCALE);
    if ( mat.empty() )  {
        std::cout << "Read input image fail" << std::endl;
        return;
    }

    PR_SRCH_FIDUCIAL_MARK_CMD stCmd;
    stCmd.enType = PR_FIDUCIAL_MARK_TYPE::SQUARE;
    stCmd.fSize = 54;
    stCmd.fMargin = 8;
    stCmd.matInput = mat;
    stCmd.rectSrchRange = cv::Rect(1459,155, 500, 500 );

    PR_SRCH_FIDUCIAL_MARK_RPY stRpy;
    VisionStatus enStatus = PR_SrchFiducialMark(&stCmd, &stRpy);
    std::cout << "Search fiducial status " << stRpy.nStatus << std::endl; 
    std::cout << "Search fiducial result " << stRpy.ptPos.x << ", " << stRpy.ptPos.y << std::endl;
}