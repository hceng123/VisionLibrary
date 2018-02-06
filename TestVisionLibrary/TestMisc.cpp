#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestCombineImage() {
    PR_COMBINE_IMG_CMD stCmd;
    PR_COMBINE_IMG_RPY stRpy;
    stCmd.nCountOfImgPerFrame = 4;
    stCmd.nCountOfFrameX = 16;
    stCmd.nCountOfFrameY = 6;
    stCmd.nOverlapX = 104;
    stCmd.nOverlapY = 173;
    stCmd.nCountOfImgPerRow = 61;

    const std::string strFolder = "D:/Data/KB_FOR_Procedure/FOVImage/";

    for (int nRow = 0; nRow < stCmd.nCountOfFrameY; ++ nRow)
    //Column start from right to left.
    for (int nCol = 0; nCol < stCmd.nCountOfFrameX; ++ nCol)
    for (int imgNo = 1; imgNo <= stCmd.nCountOfImgPerFrame; ++ imgNo)
    {
        char arrCharFileName[100];
        int nImageIndex = nCol * stCmd.nCountOfImgPerFrame + nRow * stCmd.nCountOfImgPerRow + imgNo;
        _snprintf(arrCharFileName, sizeof(arrCharFileName), "F%d-%d-1.bmp", nRow + 1, nImageIndex);
        std::string strImagePath = strFolder + std::string ( arrCharFileName );
        cv::Mat mat = cv::imread(strImagePath, cv::IMREAD_GRAYSCALE);
        if ( mat.empty() )
            continue;
        cv::Mat matColor;
        cv::cvtColor(mat, matColor, CV_BayerGR2BGR);
        stCmd.vecInputImages.push_back( matColor );
    }

    PR_CombineImg ( &stCmd, &stRpy );
    int imgNo = 0;
    for ( const auto &mat : stRpy.vecResultImages ) {
        char arrChFileName[100];
        _snprintf(arrChFileName, sizeof(arrChFileName), "CombineResult_%d.bmp", imgNo);
        std::string strResultFile = strFolder + arrChFileName;
        cv::imwrite(strResultFile, mat);
        ++ imgNo;
    }
}

void TestCombineImage_1() {
    PR_COMBINE_IMG_CMD stCmd;
    PR_COMBINE_IMG_RPY stRpy;
    stCmd.nCountOfImgPerFrame = 4;
    stCmd.nCountOfFrameX = 18;
    stCmd.nCountOfFrameY = 13;
    stCmd.nOverlapX = 102;
    stCmd.nOverlapY = 140;
    stCmd.nCountOfImgPerRow = 72;

    const std::string strFolder = "D:/Data/DEMO_Board/FOVImage/";

    for (int nRow = 0; nRow < stCmd.nCountOfFrameY; ++ nRow)
    //Column start from right to left.
    for (int nCol = 0; nCol < stCmd.nCountOfFrameX; ++ nCol)
    for (int imgNo = 1; imgNo <= stCmd.nCountOfImgPerFrame; ++ imgNo)
    {
        char arrCharFileName[100];
        int nImageIndex = nCol * stCmd.nCountOfImgPerFrame + nRow * stCmd.nCountOfImgPerRow + imgNo;
        _snprintf(arrCharFileName, sizeof(arrCharFileName), "F%d-%d-1.bmp", nRow + 1, nImageIndex);
        std::string strImagePath = strFolder + std::string ( arrCharFileName );
        cv::Mat mat = cv::imread(strImagePath, cv::IMREAD_GRAYSCALE);
        if ( mat.empty() )
            continue;
        cv::Mat matColor;
        cv::cvtColor(mat, matColor, CV_BayerGR2BGR);
        stCmd.vecInputImages.push_back( matColor );
    }

    PR_CombineImg ( &stCmd, &stRpy );
    if ( stRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to combine image" << std::endl;
        return;
    }

    double dScale = 0.5;
    int imgNo = 0;
    for ( const auto &mat : stRpy.vecResultImages ) {
        char arrChFileName[100];
        _snprintf(arrChFileName, sizeof(arrChFileName), "CombineResult_%d.bmp", imgNo);
        std::string strResultFile = strFolder + arrChFileName;
        cv::Mat matResize;
        cv::resize ( mat, matResize, cv::Size(), dScale, dScale );
        cv::imwrite(strResultFile, matResize);
        ++ imgNo;
    }
}

void TestCombineImage_2() {
    PR_COMBINE_IMG_CMD stCmd;
    PR_COMBINE_IMG_RPY stRpy;
    stCmd.nCountOfImgPerFrame = 1;
    stCmd.nCountOfFrameX = 2;
    stCmd.nCountOfFrameY = 1;
    stCmd.nOverlapX = 102;
    stCmd.nOverlapY = 0;
    stCmd.nCountOfImgPerRow = 2;

    const std::string strFolder = "D:/Data/DEMO_Board/FOVImage/";

    for (int nRow = 0; nRow < stCmd.nCountOfFrameY; ++ nRow)
    //Column start from right to left.
    for (int nCol = 0; nCol < stCmd.nCountOfFrameX; ++ nCol)
    for (int imgNo = 1; imgNo <= stCmd.nCountOfImgPerFrame; ++ imgNo)
    {
        char arrCharFileName[100];
        int nImageIndex = nCol * stCmd.nCountOfImgPerFrame + nRow * stCmd.nCountOfImgPerRow + imgNo;
        _snprintf(arrCharFileName, sizeof(arrCharFileName), "F%d-%d-1.bmp", nRow + 1, nImageIndex);
        std::string strImagePath = strFolder + std::string ( arrCharFileName );
        cv::Mat mat = cv::imread(strImagePath, cv::IMREAD_GRAYSCALE);
        if ( mat.empty() )
            continue;
        cv::Mat matColor;
        cv::cvtColor(mat, matColor, CV_BayerGR2BGR);
        stCmd.vecInputImages.push_back( matColor );
    }

    PR_CombineImg ( &stCmd, &stRpy );
    if ( stRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to combine image" << std::endl;
        return;
    }

    double dScale = 0.5;
    int imgNo = 0;
    for ( const auto &mat : stRpy.vecResultImages ) {
        char arrChFileName[100];
        _snprintf(arrChFileName, sizeof(arrChFileName), "TestCombineResult_%d.bmp", imgNo);
        std::string strResultFile = strFolder + arrChFileName;
        cv::Mat matResize;
        cv::resize ( mat, matResize, cv::Size(), dScale, dScale );
        cv::imwrite(strResultFile, matResize);
        ++ imgNo;
    }
}

void TestCombineImage_HaoYu() {
    PR_COMBINE_IMG_CMD stCmd;
    PR_COMBINE_IMG_RPY stRpy;
    stCmd.nCountOfImgPerFrame = 5;
    stCmd.nCountOfFrameX = 4;
    stCmd.nCountOfFrameY = 1;
    stCmd.nOverlapX = 277;
    stCmd.nOverlapY = 0;
    stCmd.nCountOfImgPerRow = 20;
    stCmd.enScanDir = PR_SCAN_IMAGE_DIR::LEFT_TO_RIGHT;

    const std::string strFolder = "D:/Data/20180206115742/";

    for (int nRow = 0; nRow < stCmd.nCountOfFrameY; ++ nRow)
    //Column start from right to left.
    for (int nCol = 0; nCol < stCmd.nCountOfFrameX; ++ nCol)
    for (int imgNo = 1; imgNo <= stCmd.nCountOfImgPerFrame; ++ imgNo)
    {
        char arrCharFileName[100];
        _snprintf(arrCharFileName, sizeof(arrCharFileName), "I%d_%d.bmp", nCol + 1, imgNo );
        std::string strImagePath = strFolder + std::string ( arrCharFileName );
        cv::Mat mat = cv::imread(strImagePath, cv::IMREAD_COLOR);
        if ( mat.empty() )
            continue;
        stCmd.vecInputImages.push_back( mat );
    }

    PR_CombineImg ( &stCmd, &stRpy );
    if ( stRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to combine image" << std::endl;
        return;
    }

    double dScale = 0.5;
    int imgNo = 0;
    for ( const auto &mat : stRpy.vecResultImages ) {
        char arrChFileName[100];
        _snprintf(arrChFileName, sizeof(arrChFileName), "CombineResult_%d.bmp", imgNo);
        std::string strResultFile = strFolder + arrChFileName;
        cv::Mat matResize;
        cv::resize ( mat, matResize, cv::Size(), dScale, dScale );
        cv::imwrite(strResultFile, matResize);
        ++ imgNo;
    }
}