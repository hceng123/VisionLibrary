#include <iostream>
#include <iomanip>
#include "UtilityFunc.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"

namespace AOI
{
namespace Vision
{

void TestCalib3dBase() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIB 3D BASE TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

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
    std::cout << "ThickToThinStripeK: " << std::endl;
    printfMat<float>(stRpy.matThickToThinStripeK);

    std::cout << "BaseSurfaceParam: " << std::endl;
    printfMat<float>(stRpy.matBaseSurfaceParam);
    std::cout << "BaseStartAvgPhase: " << stRpy.fBaseStartAvgPhase << std::endl;

    std::string strResultMatPath = "./data/CalibPP.yml";
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return;
    cv::write ( fs, "K", stRpy.matThickToThinStripeK );
    cv::write ( fs, "PPz", stRpy.matBaseSurfaceParam );
    cv::write ( fs, "BaseStartAvgPhase", stRpy.fBaseStartAvgPhase );
    fs.release();
}

void TestCalib3DHeight() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIB 3D HEIGHT TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

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
    stCmd.fMinAvgIntensity = 1.5;
    stCmd.nBlockStepCount = 3;
    stCmd.fBlockStepHeight = 1.f;
    stCmd.bReverseHeight = true;
    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = "./data/CalibPP.yml";
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K"];
    cv::read ( fileNode, stCmd.matThickToThinStripeK, cv::Mat() );
    fileNode = fs["PPz"];
    cv::read ( fileNode, matBaseSurfaceParam, cv::Mat() );
    fileNode = fs["PhaseToHeightK"];
    cv::read ( fileNode, stCmd.fBaseStartAvgPhase, 0.f );
    fileNode = fs["BaseStartAvgPhase"];
    cv::read ( fileNode, stCmd.fBaseStartAvgPhase, 0.f );
    fs.release();

    PR_CALC_3D_BASE_CMD stCalc3DBaseCmd;
    PR_CALC_3D_BASE_RPY stCalc3DBaseRpy;
    stCalc3DBaseCmd.matBaseSurfaceParam = matBaseSurfaceParam;
    PR_Calc3DBase ( &stCalc3DBaseCmd, &stCalc3DBaseRpy );
    if ( VisionStatus::OK != stCalc3DBaseRpy.enStatus ) {
        std::cout << "PR_Calc3DBase fail. Status = " << ToInt32 ( stCalc3DBaseRpy.enStatus ) << std::endl;
        return;
    }

    stCmd.matBaseSurface = stCalc3DBaseRpy.matBaseSurface;
    PR_Calib3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;
    std::cout << "PhaseToHeightK: " << std::endl;
    printfMat<float>(stRpy.matPhaseToHeightK);

    std::cout << "Fetched StepPhase: " << std::endl;
    printfVectorOfVector<float> ( stRpy.vecVecStepPhase );

    std::cout << "Step-Phase slope: " << std::endl;
    for ( const auto &value : stRpy.vecStepPhaseSlope)
        printf("%.2f ", value);
    std::cout << std::endl;

    std::cout << "StepPhaseDiff: " << std::endl;
    printfVectorOfVector<float> ( stRpy.vecVecStepPhaseDiff );

    cv::imwrite ( "./data/PR_Calib3DHeight_DivideStepResultImg.png", stRpy.matDivideStepResultImg );
    cv::imwrite ( "./data/PR_Calib3DHeight_Calib3DHeightResultImg.png", stRpy.matResultImg );
}

void TestComb3DCalib() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "COMBINE 3D HEIGHT CALIBRATION #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_COMB_3D_CALIB_CMD stCmd;
    PR_COMB_3D_CALIB_RPY stRpy;
    stCmd.vecVecStepPhaseNeg = readDataFromFile("./data/StepPhaseData/StepPhaseNeg.csv");
    stCmd.vecVecStepPhasePos = readDataFromFile("./data/StepPhaseData/StepPhasePos.csv");
    PR_Comb3DCalib ( &stCmd, &stRpy );
    std::cout << "PR_Comb3DCalib status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        std::cout << "Result PhaseToHeightK: " << std::endl;
        printfMat<float>(stRpy.matPhaseToHeightK);
        std::cout << "Step Phase Difference: " << std::endl;
        printfVectorOfVector<float>(stRpy.vecVecStepPhaseDiff );
    }
}

void Test3D() {
    TestCalib3dBase();
    TestCalib3DHeight();
    TestComb3DCalib();
}

}
}