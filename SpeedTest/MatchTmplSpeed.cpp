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

void MatchTmplSpeedTest_1()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE SPEED TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/CapsLock_1.png");
    stLrnCmd.rectROI = cv::Rect (0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows );
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE;
    PR_LrnTmpl ( &stLrnCmd, &stLrnRpy );
    if ( stLrnRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/CapsLock_2.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE;
    stCmd.nRecordId = stLrnRpy.nRecordId;
    stCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    
    CStopWatch stopWatch;
    for( int i = 0; i < REPEAT_TIME; ++ i ) {
        PR_MatchTmpl(&stCmd, &stRpy);
        if ( VisionStatus::OK != stRpy.enStatus )
            return;
    }
    auto nTotalTime = stopWatch.Now();
    auto nAvgTime = nTotalTime / REPEAT_TIME;
    std::cout << "Template size: " << stLrnCmd.matInputImg.size() << std::endl;
    std::cout << "Search size: " << stCmd.rectSrchWindow.size() << std::endl;
    std::cout << "Repeat " << REPEAT_TIME << " times. Total time: " << nTotalTime << ", average time: " << nAvgTime << std::endl;
}

void MatchTmplSpeedTest() {
    MatchTmplSpeedTest_1();
}

}
}