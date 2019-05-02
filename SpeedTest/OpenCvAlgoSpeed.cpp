#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include "../VisionLibrary/StopWatch.h"

namespace AOI
{
namespace Vision
{

static const int REPEAT_TIME = 10;

static void TransposeSpeedTest_1()
{
    std::cout << std::endl << "--------------------------------------------------------";
    std::cout << std::endl << "TRANSPOSE WITHOUT PREPARED MEMORY SPEED TEST #1 STARTING";
    std::cout << std::endl << "--------------------------------------------------------";
    std::cout << std::endl;

    uint64_t nTotalTime = 0;
    
    for( int i = 0; i < REPEAT_TIME; ++ i ) {
        cv::Mat matImage = cv::Mat::eye(2048, 2040, CV_32FC1);

        CStopWatch stopWatch;
        cv::transpose(matImage, matImage);
        nTotalTime += stopWatch.Now();
        auto nAvgTime = nTotalTime / REPEAT_TIME;
    }
    auto nAvgTime = nTotalTime / REPEAT_TIME;
    std::cout << "Repeat " << REPEAT_TIME << " times. Total time: " << nTotalTime << ", average time: " << nAvgTime << std::endl;
}

static void TransposeSpeedTest_2()
{
    std::cout << std::endl << "-----------------------------------------------------";
    std::cout << std::endl << "TRANSPOSE WITH PREPARED MEMORY SPEED TEST #1 STARTING";
    std::cout << std::endl << "-----------------------------------------------------";
    std::cout << std::endl;

    cv::Mat matImage = cv::Mat::eye(2048, 2040, CV_32FC1);
    cv::Mat matResult = cv::Mat::zeros(2040, 2048, CV_32FC1);
    CStopWatch stopWatch;
    for( int i = 0; i < REPEAT_TIME; ++ i ) {
        cv::transpose(matImage, matResult);
    }

    uint64_t nTotalTime = stopWatch.Now();
    auto nAvgTime = nTotalTime / REPEAT_TIME;
    std::cout << "Repeat " << REPEAT_TIME << " times. Total time: " << nTotalTime << ", average time: " << nAvgTime << std::endl;
}

void TestTransposeSpeed() {
    TransposeSpeedTest_1();
    TransposeSpeedTest_2();
}

}
}