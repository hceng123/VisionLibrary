#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

void TestCalcCameraMTF() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALC CAMERA MTF REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_CALC_CAMERA_MTF_CMD stCmd;
    PR_CALC_CAMERA_MTF_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/MTFTarget.png", cv::IMREAD_GRAYSCALE );
    stCmd.rectVBigPatternROI = cv::Rect ( 508, 744, 206, 210 );
    stCmd.rectHBigPatternROI = cv::Rect ( 508, 954, 206, 210 );
    stCmd.rectVSmallPatternROI = cv::Rect ( 989, 983, 27, 27 );
    stCmd.rectHSmallPatternROI = cv::Rect ( 989, 1010, 27, 27 );
    PR_CalcCameraMTF ( &stCmd, &stRpy );
    std::cout << "PR_CalcCameraMTF status " << ToInt32( stRpy.enStatus ) << std::endl;
    std::cout << std::fixed << std::setprecision ( 2 );
    std::cout << "Big pattern vertical absolute mtf " << stRpy.fBigPatternAbsMtfV << std::endl;
    std::cout << "Big pattern horizontal absolute mtf " << stRpy.fBigPatternAbsMtfH << std::endl;
    std::cout << "Small pattern vertical absolute mtf " << stRpy.fSmallPatternAbsMtfV << std::endl;
    std::cout << "Small pattern horizontal absolute mtf " << stRpy.fSmallPatternAbsMtfH << std::endl;
    std::cout << "Small pattern vertical relative mtf " << stRpy.fSmallPatternRelMtfV << std::endl;
    std::cout << "Small pattern horizontal relative mtf " << stRpy.fSmallPatternRelMtfH << std::endl;
}

}
}