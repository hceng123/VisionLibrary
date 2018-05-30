#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
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
    VectorOfVectorOfFloat vecVecResult;
    while( std::getline ( ss, strLine ) ) {
        vecVecResult.push_back ( split ( strLine, ',' ) );
    }
    return vecVecResult;
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

void copyVectorOfVectorToMat(const VectorOfVectorOfFloat &vecVecInput, cv::Mat &matOutput) {
    matOutput = cv::Mat ( ToInt32 ( vecVecInput.size() ), ToInt32 ( vecVecInput[0].size() ), CV_32FC1 );
    for ( int row = 0; row < matOutput.rows; ++ row )
    for ( int col = 0; col < matOutput.cols; ++ col ) {
        matOutput.at<float>(row, col) = vecVecInput[row][col];
    }
}

cv::Mat readMatFromCsvFile(const std::string &strFilePath) {
    auto vecVecData = readDataFromFile ( strFilePath );
    cv::Mat matOutput;
    copyVectorOfVectorToMat ( vecVecData, matOutput );
    return matOutput;
}

void saveMatToCsv(cv::Mat &matrix, std::string filename){
    std::ofstream outputFile(filename);
    outputFile << cv::format ( matrix, cv::Formatter::FMT_CSV ) << std::endl;
    outputFile.close();
}

void saveMatToMatlab(cv::Mat &matrix, std::string filename){
    std::ofstream outputFile(filename);
    outputFile << cv::format ( matrix, cv::Formatter::FMT_MATLAB ) << std::endl;
    outputFile.close();
}

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
    for ( int j = 0; j < nGridRow; ++ j )
    for ( int i = 0; i < nGridCol; ++ i ) {
        cv::Rect rectROI ( i * nIntervalX + nIntervalX / 2 - szMeasureWinSize.width  / 2,
                           j * nIntervalY + nIntervalY / 2 - szMeasureWinSize.height / 2,
                           szMeasureWinSize.width, szMeasureWinSize.height );
        cv::rectangle ( matResultImg, rectROI, cv::Scalar ( 0, 255, 0 ), 3 );

        cv::Mat matROI ( matHeight, rectROI );
        cv::Mat matMask = (matROI == matROI);
        float fAverage = ToFloat ( cv::mean ( matROI, matMask )[0] );

        char strAverage[100];
        _snprintf ( strAverage, sizeof(strAverage), "%.4f", fAverage );
        int baseline = 0;
        cv::Size textSize = cv::getTextSize ( strAverage, fontFace, fontScale, thickness, &baseline );
        //The height use '+' because text origin start from left-bottom.
        cv::Point ptTextOrg ( rectROI.x + (rectROI.width - textSize.width) / 2, rectROI.y + (rectROI.height + textSize.height) / 2 );
        cv::putText ( matResultImg, strAverage, ptTextOrg, fontFace, fontScale, cv::Scalar ( 255, 0, 0 ), thickness );
    }

    int nGridLineSize = 3;
    cv::Scalar scalarCyan(255, 255, 0);
    for ( int i = 1; i < nGridCol; ++ i )
        cv::line ( matResultImg, cv::Point(i * nIntervalX, 0), cv::Point(i * nIntervalX, ROWS), scalarCyan, nGridLineSize );
    for ( int i = 1; i < nGridRow; ++ i )
        cv::line ( matResultImg, cv::Point(0, i * nIntervalY), cv::Point(COLS, i * nIntervalY), scalarCyan, nGridLineSize );

    return matResultImg;
}

static std::string gstrWorkingFolder("./data/HaoYu_20171208_2/newLens1/");
//static std::string gstrWorkingFolder("./data/HaoYu_20171208/");
static std::string gstrCalibResultFile = gstrWorkingFolder + "CalibPP.yml";
static std::string gstrIntegrateCalibResultFile( gstrWorkingFolder + "IntegrateCalibResult.yml" );
bool gbUseGamma = true;
const int IMAGE_COUNT = 12;
void TestCalib3dBase() {
    std::string strFolder = gstrWorkingFolder + "1208213334_base1/";
    PR_CALIB_3D_BASE_CMD stCmd;
    PR_CALIB_3D_BASE_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        if ( mat.empty() ) {
            std::cout << "Failed to read image " << strImageFile << std::endl;
            return;
        }
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.fRemoveHarmonicWaveK = 0.f;
    PR_Calib3DBase ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DBase status " << ToInt32( stRpy.enStatus ) << std::endl;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return;

    cv::write ( fs, "K1", stRpy.matThickToThinK );
    cv::write ( fs, "K2", stRpy.matThickToThinnestK );
    cv::write ( fs, "BaseWrappedAlpha", stRpy.matBaseWrappedAlpha );
    cv::write ( fs, "BaseWrappedBeta",  stRpy.matBaseWrappedBeta );
    cv::write ( fs, "BaseWrappedGamma", stRpy.matBaseWrappedGamma );
    fs.release();
}

void TestCalib3DHeight_01() {
    std::string strFolder = gstrWorkingFolder + "0920235040_lefttop/";
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
    stCmd.bUseThinnestPattern = gbUseGamma;

    stCmd.fMinAmplitude = 8.f;
    stCmd.nBlockStepCount = 5;
  
    stCmd.fBlockStepHeight = 1.f;
    stCmd.nResultImgGridRow = 10;
    stCmd.nResultImgGridCol = 10;
    //stCmd.nRemoveBetaJumpSpanX = 0;
    //stCmd.nRemoveBetaJumpSpanY = 0;
    //stCmd.nRemoveGammaJumpSpanX = 0;
    //stCmd.nRemoveGammaJumpSpanY = 0;
    
    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fs.release();

    PR_Calib3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus ) {
        cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 10, 10 );
        cv::imwrite ( gstrWorkingFolder + "PR_Calib3DHeight_PhaseGridImg_01.png", matPhaseResultImg );
        return;
    }

    cv::imwrite ( gstrWorkingFolder + "DivideStepResultImg_01.png", stRpy.matDivideStepResultImg );
    cv::imwrite ( gstrWorkingFolder + "Calib3DHeightResultImg_01.png", stRpy.matResultImg );

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if ( ! fs1.isOpened() )
        return;
    cv::write ( fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK );
    fs1.release();

    //Write the phase and divide step result for PR_Integrate3DCalib.
    std::string strCalibDataFile(gstrWorkingFolder + "01.yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    cv::write ( fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex );
    fsCalibData.release();
}

void TestCalib3DHeight_02() {
    std::string strFolder = gstrWorkingFolder + "0920235128_rightbottom/";
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
    stCmd.bUseThinnestPattern = gbUseGamma;

    stCmd.fMinAmplitude = 8;
    stCmd.nBlockStepCount = 5;
    stCmd.fBlockStepHeight = 1.f;
    stCmd.nResultImgGridRow = 10;
    stCmd.nResultImgGridCol = 10;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fs.release();

    PR_Calib3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus ) {
        cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 10, 10 );
        cv::imwrite ( gstrWorkingFolder + "PR_Calib3DHeight_PhaseGridImg_02.png", matPhaseResultImg );
        return;
    }

    cv::imwrite ( gstrWorkingFolder + "DivideStepResultImg_02.png", stRpy.matDivideStepResultImg );
    cv::imwrite ( gstrWorkingFolder + "Calib3DHeightResultImg_02.png", stRpy.matResultImg );

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if ( ! fs1.isOpened() )
        return;
    cv::write ( fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK );
    fs1.release();

    //Write the phase and divide step result for PR_Integrate3DCalib.
    std::string strCalibDataFile(gstrWorkingFolder + "02.yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    cv::write ( fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex );
    fsCalibData.release();
}

void TestCalib3DHeight_03() {
    std::string strFolder = gstrWorkingFolder + "0920235405_negitive/";
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
    stCmd.bReverseHeight = true;
    stCmd.bUseThinnestPattern = gbUseGamma;

    stCmd.fMinAmplitude = 8;
    stCmd.nBlockStepCount = 3;
    stCmd.fBlockStepHeight = 1.f;
    stCmd.nResultImgGridRow = 10;
    stCmd.nResultImgGridCol = 10;
    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fs.release();

    PR_Calib3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::imwrite ( gstrWorkingFolder + "DivideStepResultImg_03.png", stRpy.matDivideStepResultImg );
    cv::imwrite ( gstrWorkingFolder + "Calib3DHeightResultImg_03.png", stRpy.matResultImg );

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if ( ! fs1.isOpened() )
        return;
    cv::write ( fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK );
    fs1.release();

    //Write the phase and divide step result for PR_Integrate3DCalib.
    std::string strCalibDataFile(gstrWorkingFolder + "03.yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    cv::write ( fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex );
    fsCalibData.release();
}

void TestCalc3DHeight() {
    std::string strFolder = gstrWorkingFolder + "1017000903_H5/";
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
    stCmd.bUseThinnestPattern = gbUseGamma;

    stCmd.fMinAmplitude = 1.5;

    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fs.release();

    PR_Calc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::Mat matMask = stRpy.matPhase == stRpy.matPhase;
    auto dMeanPhase = cv::mean ( stRpy.matPhase, matMask )[0];
    std::cout << "Mean Phase: " << dMeanPhase << std::endl;

    cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 9, 9 );
    cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_HeightGridImg.png", matHeightResultImg );
    cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 9, 9 );
    cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_PhaseGridImg.png", matPhaseResultImg );

    //cv::patchNaNs ( stRpy.matPhase, 0 );
    //saveMatToCsv ( stRpy.matPhase, gstrWorkingFolder + "Phase_Correction.csv");
    std::string strCalibDataFile(gstrWorkingFolder + "H5.yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    fsCalibData.release();
}

void TestIntegrate3DCalib() {
    PR_INTEGRATE_3D_CALIB_CMD stCmd;
    PR_INTEGRATE_3D_CALIB_RPY stRpy;
    for ( int i = 1; i <= 3; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.yml", i );
        std::string strDataFile = gstrWorkingFolder + chArrFileName;
        
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

    std::string strCalibDataFile( gstrWorkingFolder + "H5.yml" );
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::READ );
    if (!fsCalibData.isOpened ()) {
        std::cout << "Failed to open file: " << strCalibDataFile << std::endl;
        return;
    }
    cv::FileNode fileNode = fsCalibData["Phase"];
    cv::read ( fileNode, stCmd.matTopSurfacePhase, cv::Mat() );
    fsCalibData.release();

    cv::medianBlur ( stCmd.matTopSurfacePhase, stCmd.matTopSurfacePhase, 5 );
    stCmd.fTopSurfaceHeight = 5;

    PR_Integrate3DCalib ( &stCmd, &stRpy );
    std::cout << "PR_Integrate3DCalib status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    int i = 1;
    for ( const auto &matResultImg : stRpy.vecMatResultImg ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "ResultImg_%02d.png", i );
        std::string strDataFile = gstrWorkingFolder + chArrFileName;
        cv::imwrite ( strDataFile, matResultImg );
        ++ i;
    }

    cv::FileStorage fsCalibResultData ( gstrIntegrateCalibResultFile, cv::FileStorage::WRITE );
    if (!fsCalibResultData.isOpened ()) {
        std::cout << "Failed to open file: " << gstrIntegrateCalibResultFile << std::endl;
        return;
    }
    cv::write ( fsCalibResultData, "IntegratedK", stRpy.matIntegratedK );
    cv::write ( fsCalibResultData, "Order3CurveSurface", stRpy.matOrder3CurveSurface );
    fsCalibResultData.release();
}

void TestCalc3DHeight_With_NormalCalibParam() {
    std::string strFolder = gstrWorkingFolder + "1017001229_negative/";
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
    stCmd.fMinAmplitude = 1.5;

    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fileNode = fs["PhaseToHeightK"];
    cv::read ( fileNode, stCmd.matPhaseToHeightK, cv::Mat() );
    fs.release();

    PR_Calc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 10, 10 );
    cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_HeightGridImg_NormalalibParam.png", matHeightResultImg );
    cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 9, 9 );
    cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_PhaseGridImg_NormalCalibParam.png", matPhaseResultImg );
}

void TestCalc3DHeight_With_IntegrateCalibParam() {
    std::string strFolder = gstrWorkingFolder + "1208214502_1/";
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
    stCmd.bEnableGaussianFilter = false;
    stCmd.bReverseSeq = true;
    stCmd.fMinAmplitude = 5;
    stCmd.bUseThinnestPattern = gbUseGamma;
    stCmd.nRemoveGammaJumpSpanX = 23;
    stCmd.nRemoveGammaJumpSpanY = 20;

    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fs.release();

    fs.open(gstrIntegrateCalibResultFile, cv::FileStorage::READ);
    if (!fs.isOpened ()) {
        std::cout << "Failed to open file: " << gstrIntegrateCalibResultFile << std::endl;
        return;
    }
    fileNode = fs["IntegratedK"];
    cv::read ( fileNode, stCmd.matIntegratedK, cv::Mat() );
    fileNode = fs["Order3CurveSurface"];
    cv::read ( fileNode, stCmd.matOrder3CurveSurface, cv::Mat() );
    fs.release();

    PR_Calc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::Mat matMask = (stRpy.matHeight == stRpy.matHeight); //Find out value is not NAN.
    cv::Mat matNanMask = 255 - matMask;
    double dMinValue, dMaxValue;
    cv::Point ptMin, ptMax;
    cv::minMaxLoc ( stRpy.matHeight, &dMinValue, &dMaxValue, &ptMin, &ptMax, matMask );
    std::cout << "Minimum position " << ptMin << " min value: " << dMinValue << std::endl;
    std::cout << "Maximum position " << ptMax << " max value: " << dMaxValue << std::endl;
    double dMean = cv::mean ( stRpy.matHeight, matMask)[0];
    std::cout << "Mean height " << dMean << std::endl;
    
    cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 10, 10 );
    cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_HeightGridImg_IntegrateCalibParam.png", matHeightResultImg );
    cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 9, 9 );
    cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_PhaseGridImg_IntegrateCalibParam.png", matPhaseResultImg );

    cv::Mat matNonNan = stRpy.matHeight == stRpy.matHeight;
    cv::Mat matNan = 255 - matNonNan;
    cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_NanMask.png", matNan );

    saveMatToCsv ( stRpy.matHeight, gstrWorkingFolder + "Height.csv");
    saveMatToCsv ( stRpy.matPhase,  gstrWorkingFolder + "Phase.csv");

    std::string strHeightFile( gstrWorkingFolder + "Height.yml" );
    fs.open(strHeightFile, cv::FileStorage::WRITE);
    if (!fs.isOpened ()) {
        std::cout << "Failed to open file: " << strHeightFile << std::endl;
        return;
    }
    cv::write ( fs, "Height", stRpy.matHeight );
    fs.release();
}

void TestIntegrate3DCalibHaoYu() {
    PR_INTEGRATE_3D_CALIB_CMD stCmd;
    PR_INTEGRATE_3D_CALIB_RPY stRpy;
    for ( int i = 1; i <= 3; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%d.yml", i );
        std::string strDataFile = gstrWorkingFolder + chArrFileName;
        
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

    std::string strCalibDataFile( gstrWorkingFolder + "H5.yml" );
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
    std::cout << "PR_Integrate3DCalib status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    int i = 1;
    for ( const auto &matResultImg : stRpy.vecMatResultImg ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "ResultImg_%02d.png", i );
        std::string strDataFile = gstrWorkingFolder + chArrFileName;
        cv::imwrite ( strDataFile, matResultImg );
        ++ i;
    }

    std::string strCalibResultFile( gstrWorkingFolder + "IntegrateCalibResult.yml" );
    cv::FileStorage fsCalibResultData ( strCalibResultFile, cv::FileStorage::WRITE );
    if (!fsCalibResultData.isOpened ()) {
        std::cout << "Failed to open file: " << strCalibResultFile << std::endl;
        return;
    }
    cv::write ( fsCalibResultData, "IntegratedK", stRpy.matIntegratedK );
    cv::write ( fsCalibResultData, "Order3CurveSurface", stRpy.matOrder3CurveSurface );
    fsCalibResultData.release();
}

static cv::Mat calcOrder3Surface(const cv::Mat &matX, const cv::Mat &matY, const cv::Mat &matK) {
    cv::Mat matXPow2 = matX.    mul ( matX );
    cv::Mat matXPow3 = matXPow2.mul ( matX );
    cv::Mat matYPow2 = matY.    mul ( matY );
    cv::Mat matYPow3 = matYPow2.mul ( matY );
    cv::Mat matArray[] = { 
        matXPow3 * matK.at<float>(0),
        matYPow3 * matK.at<float>(1),
        matXPow2.mul ( matY ) * matK.at<float>(2),
        matYPow2.mul ( matX ) * matK.at<float>(3),
        matXPow2 * matK.at<float>(4),
        matYPow2 * matK.at<float>(5),
        matX.mul ( matY ) * matK.at<float>(6),
        matX * matK.at<float>(7),
        matY * matK.at<float>(8),
        cv::Mat::ones ( matX.size(), matX.type() ) * matK.at<float>(9),
    };
    const int RANK_3_SURFACE_COEFF_COL = 10;
    static_assert ( sizeof( matArray ) / sizeof(cv::Mat) == RANK_3_SURFACE_COEFF_COL, "Size of array not correct" );
    cv::Mat matResult = matArray[0];
    for ( int i = 1; i < RANK_3_SURFACE_COEFF_COL; ++ i )
        matResult = matResult + matArray[i];
    return matResult;
}

void TestMerge3DHeight() {
    PR_MERGE_3D_HEIGHT_CMD stCmd;
    PR_MERGE_3D_HEIGHT_RPY stRpy;

    std::string strWorkingFolder("./data/HaoYu_20171208_2/");

    std::string strArrayHeightFile[2] = {strWorkingFolder + "newLens1/Height.yml",
                                         strWorkingFolder + "newLens2/Height.yml"};

    for ( int i = 0; i < 2; ++ i ) {
        cv::Mat matHeight;
        cv::FileStorage fs ( strArrayHeightFile[i], cv::FileStorage::READ );
        cv::FileNode fileNode = fs["Height"];
        cv::read ( fileNode, matHeight, cv::Mat () );
        fs.release ();
        stCmd.vecMatHeight.push_back ( matHeight );
    }

    stCmd.fRemoveLowerNoiseRatio = 0.005f;
    PR_Merge3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Merge3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 10, 10 );
    cv::imwrite ( strWorkingFolder + "PR_TestMerge3DHeight_HeightGridImg.png", matHeightResultImg );

    saveMatToCsv ( stRpy.matHeight, strWorkingFolder + "MergeHeight.csv");

    double dMinValue = 0, dMaxValue = 0;
    cv::Point ptMin, ptMax;
    cv::Mat matMask = (stRpy.matHeight == stRpy.matHeight); //Find out value is not NAN.
    cv::Mat matNanMask = 255 - matMask;
    cv::minMaxLoc ( stRpy.matHeight, &dMinValue, &dMaxValue, &ptMin, &ptMax, matMask );
    std::cout << "Minimum position " << ptMin << " min value: " << dMinValue << std::endl;
    std::cout << "Maximum position " << ptMax << " max value: " << dMaxValue << std::endl;
    double dMean = cv::mean ( stRpy.matHeight, matMask)[0];
    std::cout << "Mean height " << dMean << std::endl;
    VectorOfPoint vecNanPoints;
    cv::findNonZero ( matNanMask, vecNanPoints );
    std::cout << "Count of nan: " << vecNanPoints.size() << std::endl;
    for ( const auto &point : vecNanPoints )
        std::cout << point << std::endl;
    cv::Mat matNorm;
    cv::normalize ( stRpy.matHeight, matNorm, 0, 255, cv::NormTypes::NORM_MINMAX, CV_8UC1 );
    cv::imwrite ( strWorkingFolder + "PR_TestMerge3DHeight_NormHeight.png", matNorm );
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
    //std::string strFolder = "./data/0927225721_invert/";
    std::string strFolder = "./data/0715/0715184554_10ms_80_Plane1/";
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

void TestMotor3DCalib() {
    PR_MOTOR_CALIB_3D_CMD stCmd;
    PR_MOTOR_CALIB_3D_RPY stRpy;

    std::string strDataFolder = "./data/Motor3DCalib/", strFile;

    for (int i = 1; i <= 5; ++ i) {
        PairHeightPhase pairHeightPhase;
        pairHeightPhase.first = -i;
        strFile = strDataFolder + "HP" + std::to_string(i) + ".csv";
        pairHeightPhase.second = readMatFromCsvFile(strFile);
        stCmd.vecPairHeightPhase.push_back(pairHeightPhase);
    }

    for (int i = 1; i <= 5; ++ i) {
        PairHeightPhase pairHeightPhase;
        pairHeightPhase.first = i;
        strFile = strDataFolder + "HN" + std::to_string(i) + ".csv";
        pairHeightPhase.second = readMatFromCsvFile(strFile);
        stCmd.vecPairHeightPhase.push_back(pairHeightPhase);
    }

    PR_MotorCalib3D(&stCmd, &stRpy);
    std::cout << "PR_MotorCalib3D status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    int i = 1;
    for (const auto &matResultImg : stRpy.vecMatResultImg) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof (chArrFileName), "ResultImg_%02d.png", i);
        std::string strDataFile = strDataFolder + chArrFileName;
        cv::imwrite(strDataFile, matResultImg);
        ++ i;
    }
}

void TestSolve() {
    std::string strResultFile("IntermediateResult.yml");
    cv::FileStorage fsData(strResultFile, cv::FileStorage::READ);
    if (!fsData.isOpened()) {
        std::cout << "Failed to open file: " << strResultFile << std::endl;
        return;
    }

    cv::Mat matSum1, matSum2, matK;
    cv::FileNode fileNode = fsData["matSum1"];
    cv::read(fileNode, matSum1, cv::Mat());

    fileNode = fsData["matSum2"];
    cv::read(fileNode, matSum2, cv::Mat());

    cv::solve ( matSum1, matSum2, matK, cv::DecompTypes::DECOMP_SVD );
    cv::Mat matEign(1, 5, CV_32FC1 );
    cv::eigen ( matSum1, matEign );
    printfMat<float>(matK);
    std::cout << "Eign value: " << std::endl;
    printfMat<float>(matEign);
}

void TestCalib3dBase_1() {
    std::string strFolder = "D:/Data/DLPImage/";
    PR_CALIB_3D_BASE_CMD stCmd;
    PR_CALIB_3D_BASE_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        if ( mat.empty() ) {
            std::cout << "Failed to read image " << strImageFile << std::endl;
            return;
        }
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.fRemoveHarmonicWaveK = 0.f;
    PR_Calib3DBase ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DBase status " << ToInt32( stRpy.enStatus ) << std::endl;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return;

    cv::write ( fs, "K1", stRpy.matThickToThinK );
    cv::write ( fs, "K2", stRpy.matThickToThinnestK );
    cv::write ( fs, "BaseWrappedAlpha", stRpy.matBaseWrappedAlpha );
    cv::write ( fs, "BaseWrappedBeta",  stRpy.matBaseWrappedBeta );
    cv::write ( fs, "BaseWrappedGamma", stRpy.matBaseWrappedGamma );
    fs.release();
}

void PrintResult(const PR_CALC_FRAME_VALUE_CMD &stCmd, const PR_CALC_FRAME_VALUE_RPY &stRpy) {
    std::cout << "PR_CalcFrameValue status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK == stRpy.enStatus)
        std::cout << "PR_CalcFrameValue target " << stCmd.ptTargetFrameCenter << " result: " << stRpy.fResult << std::endl;
}

void TestCalcFrameValue() {
    PR_CALC_FRAME_VALUE_CMD stCmd;
    PR_CALC_FRAME_VALUE_RPY stRpy;

    int ROWS = 5, COLS = 5;
    float fIntervalX = 100, fIntervalY = 80;
    for (int row = 0; row < ROWS; ++ row) {
        VectorOfPoint2f vecPoints;
        VectorOfFloat vecValues;
        vecPoints.reserve(COLS);
        vecValues.reserve(COLS);
        for (int col = 0; col < COLS; ++ col) {
            vecPoints.emplace_back((col + 1) * fIntervalX, 1000.f - (row + 1) * fIntervalY);
            vecValues.push_back(row * 1.01f + col * 1.3f + 10.f);
        }
        stCmd.vecVecRefFrameCenters.push_back(vecPoints);
        stCmd.vecVecRefFrameValues.push_back(vecValues);
    }   
    
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl << "CALC FRAME VALUE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;

    stCmd.ptTargetFrameCenter = cv::Point(60, 500);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintResult(stCmd, stRpy);

    stCmd.ptTargetFrameCenter = cv::Point(150, 180);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintResult(stCmd, stRpy);

    stCmd.ptTargetFrameCenter = cv::Point(500, 180);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintResult(stCmd, stRpy);
}