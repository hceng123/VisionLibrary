#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include "TestSub.h"

void TestCalc3DHeightDiff(const cv::Mat &matHeight);

using namespace AOI::Vision;

VectorOfFloat split ( const std::string &s, char delim ) {
    std::vector<float> elems;
    std::stringstream ss ( s );
    std::string strItem;
    while( std::getline ( ss, strItem, delim ) ) {
        elems.push_back ( ToFloat ( std::atof ( strItem.c_str() ) ) );
    }
    return elems;
}

VectorOfVectorOfFloat parseData(const std::string &strContent) {
    std::stringstream ss ( strContent );
    std::string strLine;
    VectorOfVectorOfFloat vevVecResult;
    while( std::getline ( ss, strLine ) ) {
        vevVecResult.push_back ( split ( strLine, ',' ) );
    }
    return vevVecResult;
}

VectorOfVectorOfFloat readDataFromFile(const std::string &strFilePath) {
    /* Open input file */
    FILE *fp;
    fopen_s ( &fp, strFilePath.c_str(), "r" );
    if(fp == NULL) {
        return VectorOfVectorOfFloat();
    }
    /* Get file size */
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    /* Allocate buffer and read */
    char *buff = new char[size + 1];
    size_t bytes = fread(buff, 1, size, fp);
    buff[bytes] = '\0';
    std::string strContent(buff);
    delete []buff;
    fclose ( fp );
    return parseData ( strContent );
}

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

//static std::string gstrCalibResultFile("./data/capture/CalibPP.yml");
static std::string gstrWorkingFolder("./data/0916_NK_Step5/");
static std::string gstrCalibResultFile = gstrWorkingFolder + "CalibPP.yml";
void TestCalib3dBase() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = gstrWorkingFolder + "0916113354NK_Plane/";
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
    stCmd.fRemoveHarmonicWaveK = 0.f;
    PR_Calib3DBase ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DBase status " << ToInt32( stRpy.enStatus ) << std::endl;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return;

    write ( fs, "K", stRpy.matThickToThinStripeK );
    write ( fs, "PPz", stRpy.matBaseSurfaceParam );
    write ( fs, "BaseStartAvgPhase", stRpy.fBaseStartAvgPhase );
    fs.release();
}

void TestCalib3DHeight() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = gstrWorkingFolder + "0916113626NK_Step/";
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
    stCmd.bReverseHeight = false;
    stCmd.fMinIntensityDiff = 3;
    stCmd.fMinAvgIntensity = 3;
    stCmd.nBlockStepCount = 5;
    stCmd.fBlockStepHeight = 1.f;
    stCmd.nResultImgGridRow = 10;
    stCmd.nResultImgGridCol = 10;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K"];
    cv::read ( fileNode, stCmd.matThickToThinStripeK, cv::Mat() );

    fileNode = fs["PPz"];
    cv::read ( fileNode, stCmd.matBaseSurfaceParam, cv::Mat() );

    fileNode = fs["BaseStartAvgPhase"];
    cv::read ( fileNode, stCmd.fBaseStartAvgPhase, 0.f );
    fs.release();

    PR_Calib3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::imwrite ( gstrWorkingFolder + "DivideStepResultImg.png", stRpy.matDivideStepResultImg );
    cv::imwrite ( gstrWorkingFolder + "Calib3DHeightResultImg.png", stRpy.matResultImg );

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if ( ! fs1.isOpened() )
        return;
    if ( VisionStatus::OK == stRpy.enStatus )
        cv::write ( fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK );

    fs1.release();
}

void TestCalc3DHeight() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = gstrWorkingFolder + "0909220722_Step/";
    //std::string strFolder = "./data/0913212217_Unwrap_Not_Finish/";
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

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K"];
    cv::read ( fileNode, stCmd.matThickToThinStripeK, cv::Mat() );

    fileNode = fs["PPz"];
    cv::read ( fileNode, stCmd.matBaseSurfaceParam, cv::Mat() );

    fileNode = fs["BaseStartAvgPhase"];
    cv::read ( fileNode, stCmd.fBaseStartAvgPhase, 0.f );

    fileNode = fs["PhaseToHeightK"];
    cv::read ( fileNode, stCmd.matPhaseToHeightK, cv::Mat() );

    fs.release();

    PR_Calc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        cv::Mat matHeightResultImg = drawHeightGrid ( stRpy.matHeight, 9, 9 );
        cv::imwrite( gstrWorkingFolder + "PR_Calc3DHeight_HeightGridImg.png", matHeightResultImg );
        cv::Mat matPhaseResultImg = drawHeightGrid ( stRpy.matPhase, 9, 9 );
        cv::imwrite( gstrWorkingFolder + "PR_Calc3DHeight_PhaseGridImg.png", matPhaseResultImg );

        TestCalc3DHeightDiff ( stRpy.matHeight );
    }
}

void TestComb3DCalib() {
    PR_COMB_3D_CALIB_CMD stCmd;
    PR_COMB_3D_CALIB_RPY stRpy;
    stCmd.vecVecStepPhaseNeg = readDataFromFile("./data/StepPhaseData/StepPhaseNeg.csv");
    stCmd.vecVecStepPhasePos = readDataFromFile("./data/StepPhaseData/StepPhasePos.csv");
    PR_Comb3DCalib ( &stCmd, &stRpy );
    std::cout << "PR_Comb3DCalib status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        std::cout << "Result PhaseToHeightK " << std::endl;
        printfMat<float>(stRpy.matPhaseToHeightK);
        std::cout << "Step Phase Difference: " << std::endl;
        printfVectorOfVector<float>(stRpy.vecVecStepPhaseDiff );
    }
}

void TestCalc3DHeightDiff(const cv::Mat &matHeight) {
    PR_CALC_3D_HEIGHT_DIFF_CMD stCmd;
    PR_CALC_3D_HEIGHT_DIFF_RPY stRpy;
    stCmd.matHeight = matHeight;
    //stCmd.vecRectBases.push_back ( cv::Rect (761, 1783, 161, 113 ) );
    //stCmd.vecRectBases.push_back ( cv::Rect (539, 1370, 71, 32 ) );
    //stCmd.rectROI = cv::Rect(675, 1637, 103, 78 );
    stCmd.vecRectBases.push_back ( cv::Rect (550, 787, 95, 184 ) );
    stCmd.rectROI = cv::Rect ( 709, 852, 71, 69 );
    PR_Calc3DHeightDiff ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeightDiff status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        std::cout << "Height Diff Result " << stRpy.fHeightDiff << std::endl;
    }
}

void TestCalcMTF() {
    const int IMAGE_COUNT = 12;
    std::string strFolder = "./data/0831213010_25_Plane/";
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

void TestCalcPD() {
    const int IMAGE_COUNT = 12;
    std::string strFolder = "./data/0715184554_10ms_80_Plane1/";
    PR_CALC_PD_CMD stCmd;
    PR_CALC_PD_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.fMagnitudeOfDLP = 161;

    PR_CalcPD ( &stCmd, &stRpy );
    std::cout << "PR_CalcPD status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        cv::imwrite("./data/CaptureRegionImg.png", stRpy.matCaptureRegionImg );
    }
}