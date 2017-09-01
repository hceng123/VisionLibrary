#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestCalib3dBase() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = "./data/0715184554_10ms_80_Plane1/";
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

void TestCalc3DHeight() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = "./data/0715190516_10ms_80_Step/";
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
    fs.release();

    PR_Calc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;

    double dMinValue = 0, dMaxValue = 0;
    cv::Mat matMask = stRpy.matHeight == stRpy.matHeight;
    cv::minMaxIdx (stRpy.matHeight, &dMinValue, &dMaxValue, 0, 0, matMask );
    //cv::compare ( matNewPhase, )
    
    auto matNewPhase = stRpy.matHeight - dMinValue;
    auto vecMatNewPhase = matToVector<float> ( matNewPhase );

    float dRatio = 255.f/( dMaxValue - dMinValue );
    matNewPhase = matNewPhase * dRatio;
    vecMatNewPhase = matToVector<float> ( matNewPhase );
    cv::imwrite("./data/HeightToGray.png", matNewPhase );
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