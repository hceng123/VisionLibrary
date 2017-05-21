#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>

namespace AOI
{
namespace Vision
{

static void _TestAutoThreshold(int testCaseNo, const std::string &strImagePath)
{
    PR_AUTO_THRESHOLD_CMD stCmd;
    PR_AUTO_THRESHOLD_RPY stRpy;
    auto PrintRpy = [](const PR_AUTO_THRESHOLD_RPY &stRpy)  {
        std::cout << "Auto threshold status: " << ToInt32(stRpy.enStatus) << std::endl;
        std::cout << "Auto threshold result: ";
        for ( auto threshold : stRpy.vecThreshold )
            std::cout << threshold << " ";
        std::cout << std::endl;

    };

    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "AUTO THRESHOLD REGRESSION TEST #" << testCaseNo << " STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    for (int nThresholdNum = 1; nThresholdNum <= PR_AUTO_THRESHOLD_MAX_NUM; ++ nThresholdNum ) {
        std::cout << std::endl << "Test with threshold number " << nThresholdNum << std::endl;
        stCmd.matInputImg = cv::imread(strImagePath, cv::IMREAD_GRAYSCALE);
        stCmd.rectROI = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows);
        stCmd.nThresholdNum = nThresholdNum;

        PR_AutoThreshold(&stCmd, &stRpy);
        PrintRpy(stRpy);
    }
}

void TestAutoThreshold()
{
    _TestAutoThreshold(1, "./data/F-16 Jet.png");
    _TestAutoThreshold(2, "./data/House.png");
    _TestAutoThreshold(3, "./data/Lena.png");
    _TestAutoThreshold(4, "./data/Peppers.png");
}

}
}