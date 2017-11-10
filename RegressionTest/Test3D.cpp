#include <iostream>
#include <iomanip>
#include "UtilityFunc.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"

namespace AOI
{
namespace Vision
{

static cv::Mat _drawHeightGrid(const cv::Mat &matHeight, int nGridRow, int nGridCol, const cv::Size &szMeasureWinSize = cv::Size(40, 40)) {
    double dMinValue = 0, dMaxValue = 0;
    cv::Mat matMask = ( matHeight == matHeight );
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
    int thickness = 2;
    for ( int j = 0; j < nGridRow; ++ j ) {
        for ( int i = 0; i < nGridCol; ++ i ) {
            cv::Rect rectROI ( i * nIntervalX + nIntervalX / 2 - szMeasureWinSize.width  / 2,
                               j * nIntervalY + nIntervalY / 2 - szMeasureWinSize.height / 2,
                               szMeasureWinSize.width, szMeasureWinSize.height );
            cv::rectangle ( matResultImg, rectROI, cv::Scalar ( 0, 255, 0 ), 3 );

            cv::Mat matROI ( matHeight, rectROI );
            cv::Mat matMask = (matROI == matROI);
            float fAverage = ToFloat ( cv::mean ( matROI, matMask )[0] );

            char strAverage[100];
            _snprintf ( strAverage, sizeof(strAverage), "%.3f", fAverage );
            std::cout << strAverage << " ";
            int baseline = 0;
            cv::Size textSize = cv::getTextSize ( strAverage, fontFace, fontScale, thickness, &baseline );
            //The height use '+' because text origin start from left-bottom.
            cv::Point ptTextOrg ( rectROI.x + (rectROI.width - textSize.width) / 2, rectROI.y + (rectROI.height + textSize.height) / 2 );
            cv::putText ( matResultImg, strAverage, ptTextOrg, fontFace, fontScale, cv::Scalar ( 255, 0, 0 ), thickness );
        }
        std::cout << std::endl;
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
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIB 3D BASE TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    const int IMAGE_COUNT = 12;
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
    std::cout << "ThickToThinK: " << std::endl;
    printfMat<float>(stRpy.matThickToThinK);
    std::cout << "ThickToThinnestK: " << std::endl;
    printfMat<float>(stRpy.matThickToThinnestK);

    std::string strResultMatPath = "./data/CalibPP.yml";
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return;
    cv::write ( fs, "K1", stRpy.matThickToThinK );
    cv::write ( fs, "K2", stRpy.matThickToThinnestK );
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

void TestIntegrate3DCalib() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INTEGRATE 3D HEIGHT CALIBRATION #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_INTEGRATE_3D_CALIB_CMD stCmd;
    PR_INTEGRATE_3D_CALIB_RPY stRpy;
    String strWorkingFolder("./data/HaoYu_20171017_Integrate3DCalibData/");
    for ( int i = 1; i <= 3; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%d.yml", i );
        std::string strDataFile = strWorkingFolder + chArrFileName;
        
        std::string strCalibDataFile( strDataFile );
        cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::READ );
        if ( ! fsCalibData.isOpened() ) {
            std::cout << "Failed to open file: " << strCalibDataFile << std::endl;
            return;
        }

        PR_INTEGRATE_3D_CALIB_CMD::SINGLE_CALIB_DATA stCalibData;
        cv::FileNode fileNode = fsCalibData["Phase"];
        cv::read ( fileNode, stCalibData.matPhase, cv::Mat() );

        fileNode = fsCalibData["DivideStepIndex"];
        cv::read ( fileNode, stCalibData.matDivideStepIndex, cv::Mat() );
        fsCalibData.release();

        stCmd.vecCalibData.push_back ( stCalibData );
    }

    std::string strCalibDataFile( strWorkingFolder + "H5.yml" );
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::READ );
    if (!fsCalibData.isOpened ()) {
        std::cout << "Failed to open file: " << strCalibDataFile << std::endl;
        return;
    }
    cv::FileNode fileNode = fsCalibData["Phase"];
    cv::read ( fileNode, stCmd.matTopSurfacePhase, cv::Mat() );
    fsCalibData.release();

    stCmd.fTopSurfaceHeight = 5;

    PR_Integrate3DCalib ( &stCmd, &stRpy );
    std::cout << "PR_Integrate3DCalib status " << ToInt32 ( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    int i = 1;
    for ( const auto &matResultImg : stRpy.vecMatResultImg ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "ResultImg_%02d.png", i );
        std::string strDataFile = strWorkingFolder + chArrFileName;
        cv::imwrite ( strDataFile, matResultImg );
        ++ i;
    }
    std::cout << "IntegratedK" << std::endl;
    printfMat<float>( stRpy.matIntegratedK, 5 );

    std::string strCalibResultFile( strWorkingFolder + "IntegrateCalibResult.yml" );
    cv::FileStorage fsCalibResultData ( strCalibResultFile, cv::FileStorage::WRITE );
    if (!fsCalibResultData.isOpened ()) {
        std::cout << "Failed to open file: " << strCalibResultFile << std::endl;
        return;
    }
    cv::write ( fsCalibResultData, "IntegratedK", stRpy.matIntegratedK );
    cv::write ( fsCalibResultData, "Order3CurveSurface", stRpy.matOrder3CurveSurface );
    fsCalibResultData.release();
}

void TestFastCalc3DHeight() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "FAST CALC 3D HEIGHT CALIBRATION #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    const int IMAGE_COUNT = 8;
    String strWorkingFolder("./data/FastCalc3DHeightCase/");
    std::string strFolder = strWorkingFolder + "0920235128_rightbottom/";
    PR_FAST_CALC_3D_HEIGHT_CMD stCmd;
    PR_FAST_CALC_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = false;
    stCmd.bReverseSeq = true;
    stCmd.fMinAmplitude = 2;

    cv::Mat matBaseSurfaceParam;
    {
        std::string strResultMatPath = strWorkingFolder + "CalibPP.yml";
        cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
        cv::FileNode fileNode = fs["K"];
        cv::read ( fileNode, stCmd.matThickToThinStripeK, cv::Mat () );

        fileNode = fs["PPz"];
        cv::read ( fileNode, matBaseSurfaceParam, cv::Mat () );

        fileNode = fs["BaseStartAvgPhase"];
        cv::read ( fileNode, stCmd.fBaseStartAvgPhase, 0.f );

        fileNode = fs["BaseWrappedAlpha"];
        cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat () );

        fileNode = fs["BaseWrappedBeta"];
        cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat () );

        fs.release ();
    }

    {
        std::string strIntegratedCalibResultPath = strWorkingFolder + "IntegrateCalibResult.yml";
        cv::FileStorage fs ( strIntegratedCalibResultPath, cv::FileStorage::READ );
        cv::FileNode fileNode = fs["IntegratedK"];
        cv::read ( fileNode, stCmd.matIntegratedK, cv::Mat () );

        fileNode = fs["Order3CurveSurface"];
        cv::read ( fileNode, stCmd.matOrder3CurveSurface, cv::Mat () );
        fs.release ();
    }

    PR_FastCalc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_FastCalc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    std::cout << "Grid phase result: " << std::endl;
    cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 10, 10 );
    cv::imwrite ( strWorkingFolder + "PR_FastCalc3DHeight_PhaseGridImg.png", matPhaseResultImg );

    std::cout << "Grid height result: " << std::endl;
    cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 10, 10 );
    cv::imwrite ( strWorkingFolder + "PR_FastCalc3DHeight_HeightGridImg.png", matHeightResultImg );
    
}

void Test3D() {
    TestCalib3dBase();
    TestCalib3DHeight();
    TestComb3DCalib();
    TestIntegrate3DCalib();
    TestFastCalc3DHeight();
}

}
}