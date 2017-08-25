#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestTemplate() {
    cv::Mat matInputImg = cv::imread ( "./data/F8-58-2.bmp", cv::IMREAD_GRAYSCALE );
    cv::Rect rect ( 234, 160, 160, 160 );
    cv::Rect rect1( 250, 800, 160, 160 );

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = matInputImg;
    stCmd.rectSrchWindow = rect;
    //stCmd.matTmpl = cv::Mat ( matInputImg, rect1 );
    stCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    PR_MatchTmpl ( &stCmd, &stRpy );
    float fScore = stRpy.fMatchScore;
}