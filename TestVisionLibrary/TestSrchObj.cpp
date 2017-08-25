#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestLrnObj() {
    PR_LRN_OBJ_CMD stCmd;
    PR_LRN_OBJ_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/OCV/ProcessedTwo.png", cv::IMREAD_GRAYSCALE );    
    if ( stCmd.matInputImg.empty() ) {
        std::cout << "Failed to read image!" << std::endl;
        return;
    }
    //stCmd.matInputImg -= 100;
    //stCmd.matInputImg *= 4;
    cv::imshow("Input image", stCmd.matInputImg );
    cv::waitKey(0);

    stCmd.enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SIFT;
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

    stCmd.enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SIFT;
    stCmd.rectLrn = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    PR_LrnObj ( &stCmd, &stRpy );
    std::cout << "Lrn Obj status " << ToInt32( stRpy.enStatus ) << std::endl;
}

void TestSrchObj() {
    
}

void TestSrchDie() {
    PR_LRN_OBJ_CMD stLrnCmd;
    PR_LRN_OBJ_RPY stLrnRpy;
    stLrnCmd.enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SURF;
    stLrnCmd.matInputImg = cv::imread("./data/die/Template.png", cv::IMREAD_GRAYSCALE );
    stLrnCmd.rectLrn = cv::Rect(0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows );
    PR_LrnObj ( &stLrnCmd, &stLrnRpy );
    if ( stLrnRpy.enStatus != VisionStatus::OK )
        return;
    cv::imwrite("./data/die/LrnObjResult.png", stLrnRpy.matResultImg );

    PR_SRCH_OBJ_CMD stSrchCmd;
    PR_SRCH_OBJ_RPY stSrchRpy;

    stSrchCmd.enAlgorithm = PR_SRCH_OBJ_ALGORITHM::SURF;
    stSrchCmd.nRecordId = stLrnRpy.nRecordId;
    stSrchCmd.matInputImg = cv::imread("./data/die/Srch.png", cv::IMREAD_GRAYSCALE );
    if ( stSrchCmd.matInputImg.empty() ) {
        std::cout << "Failed to read image!" << std::endl;
        return;
    }
    //stCmd.matInputImg -= 100;
    //stCmd.matInputImg *= 3;
    //cv::imshow("Input image", stCmd.matInputImg );
    //cv::waitKey(0);

    //stCmd.rectLrn = cv::Rect(0, 0, 68, 94 );
    stSrchCmd.rectSrchWindow = cv::Rect(0, 0, stSrchCmd.matInputImg.cols, stSrchCmd.matInputImg.rows );
    stSrchCmd.ptExpectedPos = cv::Point(stSrchCmd.matInputImg.cols / 2, stSrchCmd.matInputImg.rows / 2 );
    //PR_Init();
    PR_SrchObj ( &stSrchCmd, &stSrchRpy );
    std::cout << "Srch Obj status " << ToInt32( stSrchRpy.enStatus ) << std::endl;
    std::cout << "Srch Obj position " << stSrchRpy.ptObjPos << std::endl;
    std::cout << "Srch Obj angle " << stSrchRpy.fRotation << std::endl;

    cv::imwrite("./data/die/SrchObjResult.png", stSrchRpy.matResultImg );

    PR_DumpTimeLog("./Vision/Time.log");
}