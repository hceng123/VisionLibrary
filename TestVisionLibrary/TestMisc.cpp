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