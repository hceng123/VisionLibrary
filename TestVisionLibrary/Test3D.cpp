#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

cv::Mat drawHeightGrid(const cv::Mat &matHeight, int nGridRow, int nGridCol) {
    double dMinValue = 0, dMaxValue = 0;
    cv::Mat matMask = matHeight == matHeight;
    cv::minMaxIdx ( matHeight, &dMinValue, &dMaxValue, 0, 0, matMask );
    
    cv::Mat matNewPhase = matHeight - dMinValue;

    float dRatio = 255.f / ToFloat( dMaxValue - dMinValue );
    matNewPhase = matNewPhase * dRatio;

    cv::Mat matResultImg;
    matNewPhase.convertTo ( matResultImg, CV_8UC1);
    cv::cvtColor ( matResultImg, matResultImg, CV_GRAY2BGR );

    int ROWS = matNewPhase.rows;
    int COLS = matNewPhase.cols;
    int nIntervalX = matNewPhase.cols / nGridCol;
    int nIntervalY = matNewPhase.rows / nGridRow;    

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 3;
    for ( int j = 0; j < nGridRow; ++ j ) {
        for ( int i = 0; i < nGridCol; ++ i ) {
            cv::Rect rectROI ( i * nIntervalX, j * nIntervalY, nIntervalX, nIntervalY );
            cv::Mat matROI ( matHeight, rectROI );
            cv::Mat matMask = matROI == matROI;
            float fAverage = ToFloat ( cv::mean ( matROI, matMask )[0] );

            char strAverage[100];
            _snprintf(strAverage, sizeof(strAverage), "%.4f", fAverage);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(strAverage, fontFace, fontScale, thickness, &baseline);
            //The height use '+' because text origin start from left-bottom.
            cv::Point ptTextOrg( rectROI.x + (rectROI.width - textSize.width)/2, rectROI.y + ( rectROI.height + textSize.height ) / 2 );
            cv::putText ( matResultImg, strAverage, ptTextOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness );
        }
    }

    int nGridLineSize = 3;
    cv::Scalar scalarCyan(255, 255, 0);
    for ( int i = 1; i < nGridCol; ++ i )
        cv::line ( matResultImg, cv::Point(i * nIntervalX, 0), cv::Point(i * nIntervalX, ROWS), scalarCyan, nGridLineSize );
    for ( int i = 1; i < nGridRow; ++ i )
        cv::line ( matResultImg, cv::Point(0, i * nIntervalY), cv::Point(COLS, i * nIntervalY), scalarCyan, nGridLineSize );

    return matResultImg;
}

void TestCalib3dBase() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = "./data/0831223425_45_Plane_Sch/";
    PR_CALIB_3D_BASE_CMD stCmd;
    PR_CALIB_3D_BASE_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    PR_Calib3DBase ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DBase status " << ToInt32( stRpy.enStatus ) << std::endl;

    std::string strResultMatPath = "./data/CalibPP.yml";
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return;

    write ( fs, "K", stRpy.matThickToThinStripeK );
    write ( fs, "PPz", stRpy.matBaseSurfaceParam );
    fs.release();
}

void TestCalib3DHeight() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = "./data/0715190516_10ms_80_Step/";
    PR_CALIB_3D_HEIGHT_CMD stCmd;
    PR_CALIB_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    stCmd.fMinIntensityDiff = 3;
    stCmd.fMinAvgIntensity = 3;
    stCmd.nBlockStepCount = 3;
    stCmd.fBlockStepHeight = 1.f;

    std::string strResultMatPath = "./data/CalibPP.yml";
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K"];
    cv::read ( fileNode, stCmd.matThickToThinStripeK, cv::Mat() );

    fileNode = fs["PPz"];
    cv::read ( fileNode, stCmd.matBaseSurfaceParam, cv::Mat() );
    fs.release();

    PR_Calib3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if ( ! fs1.isOpened() )
        return;
    if ( VisionStatus::OK == stRpy.enStatus )
        cv::write ( fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK );

    fs1.release();

    //double dMinValue = 0, dMaxValue = 0;
    //cv::Mat matMask = stRpy.matHeight == stRpy.matHeight;
    //cv::minMaxIdx (stRpy.matHeight, &dMinValue, &dMaxValue, 0, 0, matMask );
    ////cv::compare ( matNewPhase, )
    //
    //auto matNewPhase = stRpy.matHeight - dMinValue;
    //auto vecMatNewPhase = matToVector<float> ( matNewPhase );

    //float dRatio = 255.f/( dMaxValue - dMinValue );
    //matNewPhase = matNewPhase * dRatio;
    //vecMatNewPhase = matToVector<float> ( matNewPhase );
    //cv::imwrite("./data/HeightToGray.png", matNewPhase );
}

void TestCalc3DHeight() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = "./data/0901210205/";
    PR_CALC_3D_HEIGHT_CMD stCmd;
    PR_CALC_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    stCmd.fMinIntensityDiff = 3;
    stCmd.fMinAvgIntensity = 3;

    std::string strResultMatPath = "./data/CalibPP.yml";
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K"];
    cv::read ( fileNode, stCmd.matThickToThinStripeK, cv::Mat() );

    fileNode = fs["PPz"];
    cv::read ( fileNode, stCmd.matBaseSurfaceParam, cv::Mat() );

    fileNode = fs["PhaseToHeightK"];
    cv::read ( fileNode, stCmd.matPhaseToHeightK, cv::Mat() );

    fs.release();

    PR_Calc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        cv::Mat matHeightResultImg = drawHeightGrid ( stRpy.matHeight, 9, 9 );
        cv::imwrite("./data/HeightGridImg.png", matHeightResultImg );
        cv::Mat matPhaseResultImg = drawHeightGrid ( stRpy.matPhase, 9, 9 );
        cv::imwrite("./data/PhaseGridImg.png", matPhaseResultImg );
    }
}

void TestCalcMTF() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = "./data/0715184554_10ms_80_Plane1/";
    PR_CALC_MTF_CMD stCmd;
    PR_CALC_MTF_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.fMagnitudeOfDLP = 161;

    PR_CalcMTF ( &stCmd, &stRpy );
    std::cout << "PR_CalcMTF status " << ToInt32( stRpy.enStatus ) << std::endl;
}