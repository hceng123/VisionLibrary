#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestLrnObj() {
    PR_LRN_OBJ_CMD stCmd;
    PR_LRN_OBJ_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/OCV/two.png", cv::IMREAD_GRAYSCALE );    
    if ( stCmd.matInputImg.empty() ) {
        std::cout << "Failed to read image!" << std::endl;
        return;
    }
    stCmd.matInputImg -= 100;
    stCmd.matInputImg *= 4;
    cv::imshow("Input image", stCmd.matInputImg );
    cv::waitKey(0);

    stCmd.enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SURF;
    stCmd.rectLrn = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    PR_LrnObj ( &stCmd, &stRpy );
    std::cout << "Lrn Obj status " << ToInt32( stRpy.enStatus ) << std::endl;
}

void TestLrnObj_1() {
    PR_LRN_OBJ_CMD stCmd;
    PR_LRN_OBJ_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/OCV/Arabic.png", cv::IMREAD_GRAYSCALE );
    //stCmd.matInputImg -= 100;
    //stCmd.matInputImg *= 3;
    if ( stCmd.matInputImg.empty() ) {
        std::cout << "Failed to read image!" << std::endl;
        return;
    }
    cv::imshow("Input image", stCmd.matInputImg );
    cv::waitKey(0);

    stCmd.enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SURF;
    stCmd.rectLrn = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    PR_LrnObj ( &stCmd, &stRpy );
    std::cout << "Lrn Obj status " << ToInt32( stRpy.enStatus ) << std::endl;
}

void TestSrchObj() {
    PR_SRCH_OBJ_CMD stCmd;
    PR_SRCH_OBJ_RPY stRpy;

    stCmd.enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SIFT;
    stCmd.nRecordID = 1;
    stCmd.matInputImg = cv::imread("./data/OCV/SrchImg.png", cv::IMREAD_GRAYSCALE );
    if ( stCmd.matInputImg.empty() ) {
        std::cout << "Failed to read image!" << std::endl;
        return;
    }
    stCmd.matInputImg -= 100;
    stCmd.matInputImg *= 3;    
    cv::imshow("Input image", stCmd.matInputImg );
    cv::waitKey(0);

    //stCmd.rectLrn = cv::Rect(0, 0, 68, 94 );
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.ptExpectedPos = cv::Point(104, 58);
    PR_Init();
    PR_SrchObj ( &stCmd, &stRpy );
    std::cout << "Srch Obj status " << ToInt32( stRpy.enStatus ) << std::endl;
}