#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace AOI::Vision;

void TestCalibCamera()
{
    PR_CALIBRATE_CAMERA_CMD stCmd;
    PR_CALIBRATE_CAMERA_RPY stRpy;
    stCmd.matInput = cv::imread("./data/ChessBoard.png", cv::IMREAD_GRAYSCALE );
    stCmd.fPatternDist = 2.;
    stCmd.szBoardPattern = cv::Size(25, 23);

    PR_CalibrateCamera ( &stCmd, &stRpy );
    if ( VisionStatus::OK != stRpy.enStatus) {
        std::cout << "Failed to calibrate camera!" << std::endl;
        return;
    }

    PR_RESTORE_IMG_CMD stRetoreCmd;
    PR_RESTORE_IMG_RPY stRetoreRpy;
    stRetoreCmd.matInput = cv::imread("./data/ChessBoard.png", cv::IMREAD_GRAYSCALE );
    stRetoreCmd.vecMatRestoreImage = stRpy.vecMatRestoreImage;
    PR_RestoreImage ( &stRetoreCmd, &stRetoreRpy );
    if ( VisionStatus::OK != stRetoreRpy.enStatus) {
        std::cout << "Failed to restore image!" << std::endl;
        return;
    }

    //cv::FileStorage fs("./data/InputMat.yml", cv::FileStorage::WRITE);
    //cv::write(fs , "MapX", stRetoreCmd.vecMatRestoreImage[0]);
    //fs.release();

    //cv::imwrite ( "./data/mapX.png", )

    cv::Mat matInputFloat;
    stRetoreCmd.matInput.convertTo ( matInputFloat, CV_32FC1 );

    //cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - stRetoreRpy.matResult );
    cv::imwrite("./data/chessboardRestored.png", stRetoreRpy.matResult);

    cv::Mat matReadRestored = cv::imread("./data/chessboardRestored.png", cv::IMREAD_GRAYSCALE );
    cv::Mat matReadAgainFloat;
    matReadRestored.convertTo ( matReadAgainFloat, CV_32FC1 );

    cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - matReadAgainFloat );
    cv::imwrite("./data/chessboard_diff.png", matDiff);
    PR_DumpTimeLog("./data/TimeLog.log");
}

void TestCompareInputAndResult() {
    cv::Mat matInput = cv::imread ( "./Vision/Logcase/RestoreImg_2017_04_27_16_48_50_093/image.png");
    if ( matInput.empty() )
        return;

    cv::Mat matResult = cv::imread ( "./Vision/Logcase/RestoreImg_2017_04_27_16_48_50_093/result.png" );
    if ( matResult.empty() )
        return;

    cv::Mat matInputFloat, matResultFloat;
    matInput.convertTo  ( matInputFloat, CV_32FC1 );
    matResult.convertTo ( matResultFloat, CV_32FC1 );

    cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - matResultFloat );
    cv::imwrite("./Vision/Logcase/RestoreImg_2017_04_27_16_48_50_093/chessboard_diff.png", matDiff);
}

void TestRunRestoreImgLogCase() {
    PR_RunLogCase ("./Vision/Logcase/RestoreImg_2017_04_27_16_48_50_093");
}