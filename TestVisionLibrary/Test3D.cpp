#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include "TestSub.h"
#include "../RegressionTest/UtilityFunc.h"
#include "../VisionLibrary/CalcUtils.hpp"

void TestCalc3DHeightDiff(const cv::Mat &matHeight);

using namespace AOI::Vision;

void saveMatToCsv(cv::Mat &matrix, std::string filename){
    std::ofstream outputFile(filename);
    outputFile << cv::format(matrix, cv::Formatter::FMT_CSV);
    outputFile.close();
}

void saveMatToMatlab(cv::Mat &matrix, std::string filename){
    std::ofstream outputFile(filename);
    outputFile << cv::format(matrix, cv::Formatter::FMT_MATLAB) << std::endl;
    outputFile.close();
}

static cv::Mat _drawHeightGrid(const cv::Mat &matHeight, int nGridRow, int nGridCol, const cv::Size &szMeasureWinSize = cv::Size(40, 40)) {
    double dMinValue = 0, dMaxValue = 0;
    cv::Mat matMask = (matHeight == matHeight);
    cv::minMaxIdx(matHeight, &dMinValue, &dMaxValue, 0, 0, matMask);

    cv::Mat matNewPhase = matHeight - dMinValue;

    float dRatio = 255.f / ToFloat(dMaxValue - dMinValue);
    matNewPhase = matNewPhase * dRatio;

    cv::Mat matResultImg;
    matNewPhase.convertTo(matResultImg, CV_8UC1);
    cv::cvtColor(matResultImg, matResultImg, CV_GRAY2BGR);

    int ROWS = matNewPhase.rows;
    int COLS = matNewPhase.cols;
    int nIntervalX = matNewPhase.cols / nGridCol;
    int nIntervalY = matNewPhase.rows / nGridRow;

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 2;
    for (int j = 0; j < nGridRow; ++j)
        for (int i = 0; i < nGridCol; ++i) {
            cv::Rect rectROI(i * nIntervalX + nIntervalX / 2 - szMeasureWinSize.width / 2,
                j * nIntervalY + nIntervalY / 2 - szMeasureWinSize.height / 2,
                szMeasureWinSize.width, szMeasureWinSize.height);
            cv::rectangle(matResultImg, rectROI, cv::Scalar(0, 255, 0), 3);

            cv::Mat matROI(matHeight, rectROI);
            cv::Mat matMask = (matROI == matROI);
            float fAverage = ToFloat(cv::mean(matROI, matMask)[0]);

            char strAverage[100];
            _snprintf(strAverage, sizeof(strAverage), "%.4f", fAverage);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(strAverage, fontFace, fontScale, thickness, &baseline);
            //The height use '+' because text origin start from left-bottom.
            cv::Point ptTextOrg(rectROI.x + (rectROI.width - textSize.width) / 2, rectROI.y + (rectROI.height + textSize.height) / 2);
            cv::putText(matResultImg, strAverage, ptTextOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness);
        }

    int nGridLineSize = 3;
    cv::Scalar scalarCyan(255, 255, 0);
    for (int i = 1; i < nGridCol; ++i)
        cv::line(matResultImg, cv::Point(i * nIntervalX, 0), cv::Point(i * nIntervalX, ROWS), scalarCyan, nGridLineSize);
    for (int i = 1; i < nGridRow; ++i)
        cv::line(matResultImg, cv::Point(0, i * nIntervalY), cv::Point(COLS, i * nIntervalY), scalarCyan, nGridLineSize);

    return matResultImg;
}

static std::string gstrWorkingFolder("./data/HaoYu_20171114/test1/NewLens2/");
//static std::string gstrWorkingFolder("./data/HaoYu_20171208/");
static std::string gstrCalibResultFile = gstrWorkingFolder + "CalibPP.yml";
static std::string gstrIntegrateCalibResultFile(gstrWorkingFolder + "IntegrateCalibResult.yml");
bool gbUseGamma = true;
const int IMAGE_COUNT = 12;
void TestCalib3dBase() {
    std::string strFolder = gstrWorkingFolder + "1114193238_base/";
    PR_CALIB_3D_BASE_CMD stCmd;
    PR_CALIB_3D_BASE_RPY stRpy;
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.bmp", i);
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread(strImageFile, cv::IMREAD_GRAYSCALE);
        if (mat.empty()) {
            std::cout << "Failed to read image " << strImageFile << std::endl;
            return;
        }
        stCmd.vecInputImgs.push_back(mat);
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.fRemoveHarmonicWaveK = 0.f;
    PR_Calib3DBase(&stCmd, &stRpy);
    std::cout << "PR_Calib3DBase status " << ToInt32(stRpy.enStatus) << std::endl;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        return;

    cv::write(fs, "ProjectDir", ToInt32(stRpy.enProjectDir));
    cv::write(fs, "ScanDir", ToInt32(stRpy.enScanDir));
    cv::write(fs, "K1", stRpy.matThickToThinK);
    cv::write(fs, "K2", stRpy.matThickToThinnestK);
    cv::write(fs, "BaseWrappedAlpha", stRpy.matBaseWrappedAlpha);
    cv::write(fs, "BaseWrappedBeta", stRpy.matBaseWrappedBeta);
    cv::write(fs, "BaseWrappedGamma", stRpy.matBaseWrappedGamma);
    fs.release();
}

void TestCalib3DHeight_01() {
    std::string strFolder = gstrWorkingFolder + "0920235040_lefttop/";
    PR_CALIB_3D_HEIGHT_CMD stCmd;
    PR_CALIB_3D_HEIGHT_RPY stRpy;
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.bmp", i);
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread(strImageFile, cv::IMREAD_GRAYSCALE);
        stCmd.vecInputImgs.push_back(mat);
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
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::READ);
    cv::FileNode fileNode = fs["K1"];
    cv::read(fileNode, stCmd.matThickToThinK, cv::Mat());
    fileNode = fs["K2"];
    cv::read(fileNode, stCmd.matThickToThinnestK, cv::Mat());
    fileNode = fs["BaseWrappedAlpha"];
    cv::read(fileNode, stCmd.matBaseWrappedAlpha, cv::Mat());
    fileNode = fs["BaseWrappedBeta"];
    cv::read(fileNode, stCmd.matBaseWrappedBeta, cv::Mat());
    fileNode = fs["BaseWrappedGamma"];
    cv::read(fileNode, stCmd.matBaseWrappedGamma, cv::Mat());
    fs.release();

    PR_Calib3DHeight(&stCmd, &stRpy);
    std::cout << "PR_Calib3DHeight status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus) {
        cv::Mat matPhaseResultImg = _drawHeightGrid(stRpy.matPhase, 10, 10);
        cv::imwrite(gstrWorkingFolder + "PR_Calib3DHeight_PhaseGridImg_01.png", matPhaseResultImg);
        return;
    }

    cv::imwrite(gstrWorkingFolder + "DivideStepResultImg_01.png", stRpy.matDivideStepResultImg);
    cv::imwrite(gstrWorkingFolder + "Calib3DHeightResultImg_01.png", stRpy.matResultImg);

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if (!fs1.isOpened())
        return;
    cv::write(fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK);
    fs1.release();

    //Write the phase and divide step result for PR_Integrate3DCalib.
    std::string strCalibDataFile(gstrWorkingFolder + "01.yml");
    cv::FileStorage fsCalibData(strCalibDataFile, cv::FileStorage::WRITE);
    if (!fsCalibData.isOpened())
        return;
    cv::write(fsCalibData, "Phase", stRpy.matPhase);
    cv::write(fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex);
    fsCalibData.release();
}

void TestCalib3DHeight_02() {
    std::string strFolder = gstrWorkingFolder + "0920235128_rightbottom/";
    PR_CALIB_3D_HEIGHT_CMD stCmd;
    PR_CALIB_3D_HEIGHT_RPY stRpy;
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.bmp", i);
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread(strImageFile, cv::IMREAD_GRAYSCALE);
        stCmd.vecInputImgs.push_back(mat);
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
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::READ);
    cv::FileNode fileNode = fs["K1"];
    cv::read(fileNode, stCmd.matThickToThinK, cv::Mat());
    fileNode = fs["K2"];
    cv::read(fileNode, stCmd.matThickToThinnestK, cv::Mat());
    fileNode = fs["BaseWrappedAlpha"];
    cv::read(fileNode, stCmd.matBaseWrappedAlpha, cv::Mat());
    fileNode = fs["BaseWrappedBeta"];
    cv::read(fileNode, stCmd.matBaseWrappedBeta, cv::Mat());
    fileNode = fs["BaseWrappedGamma"];
    cv::read(fileNode, stCmd.matBaseWrappedGamma, cv::Mat());
    fs.release();

    PR_Calib3DHeight(&stCmd, &stRpy);
    std::cout << "PR_Calib3DHeight status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus) {
        cv::Mat matPhaseResultImg = _drawHeightGrid(stRpy.matPhase, 10, 10);
        cv::imwrite(gstrWorkingFolder + "PR_Calib3DHeight_PhaseGridImg_02.png", matPhaseResultImg);
        return;
    }

    cv::imwrite(gstrWorkingFolder + "DivideStepResultImg_02.png", stRpy.matDivideStepResultImg);
    cv::imwrite(gstrWorkingFolder + "Calib3DHeightResultImg_02.png", stRpy.matResultImg);

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if (!fs1.isOpened())
        return;
    cv::write(fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK);
    fs1.release();

    //Write the phase and divide step result for PR_Integrate3DCalib.
    std::string strCalibDataFile(gstrWorkingFolder + "02.yml");
    cv::FileStorage fsCalibData(strCalibDataFile, cv::FileStorage::WRITE);
    if (!fsCalibData.isOpened())
        return;
    cv::write(fsCalibData, "Phase", stRpy.matPhase);
    cv::write(fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex);
    fsCalibData.release();
}

void TestCalib3DHeight_03() {
    std::string strFolder = gstrWorkingFolder + "0920235405_negitive/";
    PR_CALIB_3D_HEIGHT_CMD stCmd;
    PR_CALIB_3D_HEIGHT_RPY stRpy;
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.bmp", i);
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread(strImageFile, cv::IMREAD_GRAYSCALE);
        stCmd.vecInputImgs.push_back(mat);
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
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::READ);
    cv::FileNode fileNode = fs["K1"];
    cv::read(fileNode, stCmd.matThickToThinK, cv::Mat());
    fileNode = fs["K2"];
    cv::read(fileNode, stCmd.matThickToThinnestK, cv::Mat());
    fileNode = fs["BaseWrappedAlpha"];
    cv::read(fileNode, stCmd.matBaseWrappedAlpha, cv::Mat());
    fileNode = fs["BaseWrappedBeta"];
    cv::read(fileNode, stCmd.matBaseWrappedBeta, cv::Mat());
    fileNode = fs["BaseWrappedGamma"];
    cv::read(fileNode, stCmd.matBaseWrappedGamma, cv::Mat());
    fs.release();

    PR_Calib3DHeight(&stCmd, &stRpy);
    std::cout << "PR_Calib3DHeight status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    cv::imwrite(gstrWorkingFolder + "DivideStepResultImg_03.png", stRpy.matDivideStepResultImg);
    cv::imwrite(gstrWorkingFolder + "Calib3DHeightResultImg_03.png", stRpy.matResultImg);

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if (!fs1.isOpened())
        return;
    cv::write(fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK);
    fs1.release();

    //Write the phase and divide step result for PR_Integrate3DCalib.
    std::string strCalibDataFile(gstrWorkingFolder + "03.yml");
    cv::FileStorage fsCalibData(strCalibDataFile, cv::FileStorage::WRITE);
    if (!fsCalibData.isOpened())
        return;
    cv::write(fsCalibData, "Phase", stRpy.matPhase);
    cv::write(fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex);
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
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::READ);
    cv::FileNode fileNode = fs["K1"];
    cv::read(fileNode, stCmd.matThickToThinK, cv::Mat());
    fileNode = fs["K2"];
    cv::read(fileNode, stCmd.matThickToThinnestK, cv::Mat());

    Int32 projectDir, scanDir;
    fileNode = fs["ProjectDir"];
    cv::read(fileNode, projectDir, 0);
    stCmd.enProjectDir = static_cast<PR_DIRECTION>(projectDir);
    fileNode = fs["ScanDir"];
    cv::read(fileNode, scanDir, 0);
    stCmd.enScanDir = static_cast<PR_DIRECTION>(scanDir);

    fileNode = fs["BaseWrappedAlpha"];
    cv::read(fileNode, stCmd.matBaseWrappedAlpha, cv::Mat());
    fileNode = fs["BaseWrappedBeta"];
    cv::read(fileNode, stCmd.matBaseWrappedBeta, cv::Mat());
    fileNode = fs["BaseWrappedGamma"];
    cv::read(fileNode, stCmd.matBaseWrappedGamma, cv::Mat());
    fs.release();

    PR_Calc3DHeight(&stCmd, &stRpy);
    std::cout << "PR_Calc3DHeight status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    cv::Mat matMask = stRpy.matPhase == stRpy.matPhase;
    auto dMeanPhase = cv::mean(stRpy.matPhase, matMask)[0];
    std::cout << "Mean Phase: " << dMeanPhase << std::endl;

    cv::Mat matHeightResultImg = _drawHeightGrid(stRpy.matHeight, 9, 9);
    cv::imwrite(gstrWorkingFolder + "PR_Calc3DHeight_HeightGridImg.png", matHeightResultImg);
    cv::Mat matPhaseResultImg = _drawHeightGrid(stRpy.matPhase, 9, 9);
    cv::imwrite(gstrWorkingFolder + "PR_Calc3DHeight_PhaseGridImg.png", matPhaseResultImg);

    //cv::patchNaNs ( stRpy.matPhase, 0 );
    //saveMatToCsv ( stRpy.matPhase, gstrWorkingFolder + "Phase_Correction.csv");
    std::string strCalibDataFile(gstrWorkingFolder + "H5.yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write (fsCalibData, "Phase", stRpy.matPhase );
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
    std::string strFolder = gstrWorkingFolder + "1114194025_02/";
    //std::string strFolder = "./data/0913212217_Unwrap_Not_Finish/";
    PR_CALC_3D_HEIGHT_CMD stCmd;
    PR_CALC_3D_HEIGHT_RPY stRpy;
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.bmp", i);
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread(strImageFile, cv::IMREAD_GRAYSCALE);
        stCmd.vecInputImgs.push_back(mat);
    }
    stCmd.bEnableGaussianFilter = false;
    stCmd.bReverseSeq = true;
    stCmd.fMinAmplitude = 5;
    stCmd.bUseThinnestPattern = gbUseGamma;
    stCmd.nRemoveGammaJumpSpanX = 23;
    stCmd.nRemoveGammaJumpSpanY = 20;

    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::READ);
    cv::FileNode fileNode = fs["K1"];
    cv::read(fileNode, stCmd.matThickToThinK, cv::Mat());
    fileNode = fs["K2"];
    cv::read(fileNode, stCmd.matThickToThinnestK, cv::Mat());

    Int32 projectDir, scanDir;
    fileNode = fs["ProjectDir"];
    cv::read(fileNode, projectDir, 0);
    stCmd.enProjectDir = static_cast<PR_DIRECTION>(projectDir);
    fileNode = fs["ScanDir"];
    cv::read(fileNode, scanDir, 0);
    stCmd.enScanDir = static_cast<PR_DIRECTION>(scanDir);

    fileNode = fs["BaseWrappedAlpha"];
    cv::read(fileNode, stCmd.matBaseWrappedAlpha, cv::Mat());
    fileNode = fs["BaseWrappedBeta"];
    cv::read(fileNode, stCmd.matBaseWrappedBeta, cv::Mat());
    fileNode = fs["BaseWrappedGamma"];
    cv::read(fileNode, stCmd.matBaseWrappedGamma, cv::Mat());
    fs.release();

    fs.open(gstrIntegrateCalibResultFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "Failed to open file: " << gstrIntegrateCalibResultFile << std::endl;
        return;
    }
    fileNode = fs["IntegratedK"];
    cv::read(fileNode, stCmd.matIntegratedK, cv::Mat());
    fileNode = fs["Order3CurveSurface"];
    cv::read(fileNode, stCmd.matOrder3CurveSurface, cv::Mat());
    fs.release();

    PR_Calc3DHeight(&stCmd, &stRpy);
    std::cout << "PR_Calc3DHeight status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    //cv::Mat matMask = (stRpy.matHeight == stRpy.matHeight); //Find out value is not NAN.
    cv::Mat matNanMask = stRpy.matNanMask;
    double dMinValue, dMaxValue;
    cv::Point ptMin, ptMax;
    cv::minMaxLoc(stRpy.matHeight, &dMinValue, &dMaxValue, &ptMin, &ptMax, matNanMask);
    std::cout << "Minimum position " << ptMin << " min value: " << dMinValue << std::endl;
    std::cout << "Maximum position " << ptMax << " max value: " << dMaxValue << std::endl;
    double dMean = cv::mean(stRpy.matHeight, matNanMask)[0];
    std::cout << "Mean height " << dMean << std::endl;

    cv::Mat matHeightResultImg = _drawHeightGrid(stRpy.matHeight, 10, 10);
    cv::imwrite(gstrWorkingFolder + "PR_Calc3DHeight_HeightGridImg_IntegrateCalibParam.png", matHeightResultImg);
    cv::Mat matPhaseResultImg = _drawHeightGrid(stRpy.matPhase, 9, 9);
    cv::imwrite(gstrWorkingFolder + "PR_Calc3DHeight_PhaseGridImg_IntegrateCalibParam.png", matPhaseResultImg);

    cv::Mat matNonNan = stRpy.matHeight == stRpy.matHeight;
    cv::Mat matNan = 255 - matNonNan;
    cv::imwrite(gstrWorkingFolder + "PR_Calc3DHeight_NanMask.png", matNan);

    saveMatToCsv(stRpy.matHeight, gstrWorkingFolder + "Height.csv");
    saveMatToCsv(stRpy.matPhase, gstrWorkingFolder + "Phase.csv");

    std::string strHeightFile(gstrWorkingFolder + "Height.yml");
    fs.open(strHeightFile, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cout << "Failed to open file: " << strHeightFile << std::endl;
        return;
    }
    cv::write(fs, "Height", stRpy.matHeight);
    cv::write(fs, "NanMask", stRpy.matNanMask);
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

static void PrintMinMaxLoc(const cv::Mat &matHeight, const std::string &strTitle = "") {
    double dMinValue = 0, dMaxValue = 0;
    cv::Point ptMin, ptMax;
    cv::Mat matMask = (matHeight == matHeight); //Find out value is not NAN.
    cv::Mat matNanMask = 255 - matMask;
    cv::minMaxLoc(matHeight, &dMinValue, &dMaxValue, &ptMin, &ptMax, matMask);
    std::cout << strTitle << std::endl;
    std::cout << "Minimum position " << ptMin << " min value: " << dMinValue << std::endl;
    std::cout << "Maximum position " << ptMax << " max value: " << dMaxValue << std::endl;
}

void TestMerge3DHeight() {
    PR_MERGE_3D_HEIGHT_CMD stCmd;
    PR_MERGE_3D_HEIGHT_RPY stRpy;

    std::string strWorkingFolder("./data/HaoYu_20171114/test1/");

    std::string strArrayHeightFile[2] = {strWorkingFolder + "newLens1/Height.yml",
                                         strWorkingFolder + "newLens2/Height.yml"};

    for (int i = 0; i < 2; ++i) {
        cv::Mat matHeight, matNanMask;
        cv::FileStorage fs(strArrayHeightFile[i], cv::FileStorage::READ);
        cv::FileNode fileNode = fs["Height"];
        cv::read(fileNode, matHeight, cv::Mat());
        fileNode = fs["NanMask"];
        cv::read(fileNode, matNanMask, cv::Mat());
        fs.release();
        stCmd.vecMatHeight.push_back(matHeight);
        stCmd.vecMatNanMask.push_back(matNanMask);
    }

    stCmd.enMethod = PR_MERGE_3D_HT_METHOD::SELECT_MAX;
    stCmd.fHeightDiffThreshold = 0.1f;
    stCmd.fRemoveLowerNoiseRatio = 0.005f;
    PR_Merge3DHeight(&stCmd, &stRpy);
    std::cout << "PR_Merge3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::Mat matHeightResultImg = _drawHeightGrid(stRpy.matHeight, 10, 10);
    cv::imwrite(strWorkingFolder + "PR_TestMerge3DHeight_HeightGridImg.png", matHeightResultImg);

    saveMatToCsv(stRpy.matHeight, strWorkingFolder + "MergeHeight.csv");

    double dMinValue = 0, dMaxValue = 0;
    cv::Point ptMin, ptMax;
    cv::Mat matMask = (stRpy.matHeight == stRpy.matHeight); //Find out value is not NAN.
    cv::Mat matNanMask = 255 - matMask;
    cv::minMaxLoc(stRpy.matHeight, &dMinValue, &dMaxValue, &ptMin, &ptMax, matMask);
    std::cout << "Minimum position " << ptMin << " min value: " << dMinValue << std::endl;
    std::cout << "Maximum position " << ptMax << " max value: " << dMaxValue << std::endl;
    double dMean = cv::mean(stRpy.matHeight, matMask)[0];
    std::cout << "Mean height " << dMean << std::endl;
    VectorOfPoint vecNanPoints;
    cv::findNonZero(matNanMask, vecNanPoints);
    std::cout << "Count of nan: " << vecNanPoints.size() << std::endl;
    //for (const auto &point : vecNanPoints)
    //    std::cout << point << std::endl;
    cv::Mat matNorm;
    cv::normalize(stRpy.matHeight, matNorm, 0, 255, cv::NormTypes::NORM_MINMAX, CV_8UC1);
    cv::imwrite(strWorkingFolder + "PR_TestMerge3DHeight_NormHeight.png", matNorm);

    TestCalc3DHeightDiff(stRpy.matHeight);
}

void TestCalc3DHeightDiff(const cv::Mat &matHeight) {
    PR_CALC_3D_HEIGHT_DIFF_CMD stCmd;
    PR_CALC_3D_HEIGHT_DIFF_RPY stRpy;
    stCmd.matHeight = matHeight;
    stCmd.matMask = cv::Mat::zeros(matHeight.size(), CV_8UC1);
    cv::Mat matMaskROI(stCmd.matMask, cv::Rect(727, 714, 100, 100));
    matMaskROI.setTo(PR_MAX_GRAY_LEVEL);
    //stCmd.vecRectBases.push_back ( cv::Rect (761, 1783, 161, 113 ) );
    //stCmd.vecRectBases.push_back ( cv::Rect (539, 1370, 71, 32 ) );
    //stCmd.rectROI = cv::Rect(675, 1637, 103, 78 );
    stCmd.vecRectBases.push_back(cv::Rect(727, 714, 169, 128));
    stCmd.rectROI = cv::Rect(859, 887, 98, 81);
    PR_Calc3DHeightDiff(&stCmd, &stRpy);
    std::cout << "PR_Calc3DHeightDiff status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK == stRpy.enStatus) {
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

    printfMat<float>(stRpy.matIntegratedK, 3);
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

    cv::solve(matSum1, matSum2, matK, cv::DecompTypes::DECOMP_SVD);
    cv::Mat matEign(1, 5, CV_32FC1);
    cv::eigen(matSum1, matEign);
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
    PR_Calib3DBase(&stCmd, &stRpy);
    std::cout << "PR_Calib3DBase status " << ToInt32( stRpy.enStatus ) << std::endl;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        return;

    cv::write(fs, "K1", stRpy.matThickToThinK);
    cv::write(fs, "K2", stRpy.matThickToThinnestK);
    cv::write(fs, "BaseWrappedAlpha", stRpy.matBaseWrappedAlpha);
    cv::write(fs, "BaseWrappedBeta", stRpy.matBaseWrappedBeta);
    cv::write(fs, "BaseWrappedGamma", stRpy.matBaseWrappedGamma);
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

void TestCalcFrameValue_1() {
    PR_CALC_FRAME_VALUE_CMD stCmd;
    PR_CALC_FRAME_VALUE_RPY stRpy;

    int ROWS = 1, COLS = 5;
    float fIntervalX = 100;
    for (int row = 0; row < ROWS; ++ row) {
        VectorOfPoint2f vecPoints;
        VectorOfFloat vecValues;
        vecPoints.reserve(COLS);
        vecValues.reserve(COLS);
        for (int col = 0; col < COLS; ++ col) {
            vecPoints.emplace_back((col + 1) * fIntervalX, 0.f);
            vecValues.push_back(col * 0.01f - 0.02f);
        }
        stCmd.vecVecRefFrameCenters.push_back(vecPoints);
        stCmd.vecVecRefFrameValues.push_back(vecValues);
    }   
    
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl << "CALC FRAME VALUE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;

    stCmd.ptTargetFrameCenter = cv::Point(60, 0);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintResult(stCmd, stRpy);

    stCmd.ptTargetFrameCenter = cv::Point(150, 0);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintResult(stCmd, stRpy);

    stCmd.ptTargetFrameCenter = cv::Point(500, 0);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintResult(stCmd, stRpy);
}

void TestCalcFrameValue_2() {
    PR_CALC_FRAME_VALUE_CMD stCmd;
    PR_CALC_FRAME_VALUE_RPY stRpy;

    auto vecPoints = VectorOfPoint2f{cv::Point2f(-130.124084, 151.156494), cv::Point2f(-104.631355, 151.156494), cv::Point2f(-79.1386414, 151.156494), cv::Point2f(-53.6459198, 151.156494)};
    stCmd.vecVecRefFrameCenters.push_back(vecPoints);

    vecPoints = VectorOfPoint2f{cv::Point2f(-130.124084, 118.996170), cv::Point2f(-104.631355, 118.996170), cv::Point2f(-79.1386414, 118.996170), cv::Point2f(-53.6459198, 118.996170)};
    stCmd.vecVecRefFrameCenters.push_back(vecPoints);

    vecPoints = VectorOfPoint2f{cv::Point2f(-130.124084, 86.8358307), cv::Point2f(-104.631355, 86.8358307), cv::Point2f(-79.1386414, 86.8358307), cv::Point2f(-53.6459198, 86.8358307)};
    stCmd.vecVecRefFrameCenters.push_back(vecPoints);

    vecPoints = VectorOfPoint2f{cv::Point2f(-130.124084, 54.6754990), cv::Point2f(-104.631355, 54.6754990), cv::Point2f(-79.1386414, 54.6754990), cv::Point2f(-53.6459198, 54.6754990)};
    stCmd.vecVecRefFrameCenters.push_back(vecPoints);

    VectorOfFloat vecValues{0.0468321592f, 0.0429658033f, 0.0480033569f, 0.0454535373f};
    stCmd.vecVecRefFrameValues.push_back(vecValues);

    vecValues = VectorOfFloat{0.0454526432f, 0.0450277366f, 0.0459417216f, 0.0449718386f};
    stCmd.vecVecRefFrameValues.push_back(vecValues);

    vecValues = VectorOfFloat{0.0467235260f, 0.0470155515f, 0.0483370610f, 0.0471665189f};
    stCmd.vecVecRefFrameValues.push_back(vecValues);

    vecValues = VectorOfFloat{0.0498712249f, 0.0471298359f, 0.0499374978f, 0.0486724712f};
    stCmd.vecVecRefFrameValues.push_back(vecValues);

    //stCmd.ptTargetFrameCenter = cv::Point2f(-53.f, 102.f);
    stCmd.ptTargetFrameCenter = cv::Point2f(-135.f, 102.f);

    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintResult(stCmd, stRpy);
}

VectorOfMat ReadFrameImage(const std::string &path)
{
    VectorOfMat vecImages;
    for (int i = 1; i <= IMAGE_COUNT * 4; ++ i) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = path + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        if ( mat.empty() ) {
            std::cout << "Failed to read image " << strImageFile << std::endl;
            return vecImages;
        }
        vecImages.push_back(mat);
    }
    return vecImages;
}

void TestCalc4DLPHeightOffset()
{
    PR_CALC_3D_HEIGHT_CMD stCalcHeightCmds[4];
    PR_CALC_3D_HEIGHT_RPY stCalcHeightRpys[4];
    for (int nStation = 1; nStation <= 4; ++ nStation)
    {
        bool b3DDetectCaliUseThinPattern = false;
        bool b3DDetectGaussionFilter = false;
        //bool b3DDetectReverseSeq = System->getParam("3d_detect_reverse_seq").toBool();
        //double d3DDetectMinIntDiff = System->getParam("3d_detect_min_intensity_diff").toDouble();
        //double d3DDetectPhaseShift = System->getParam("3d_detect_phase_shift").toDouble();
        stCalcHeightCmds[nStation - 1].bEnableGaussianFilter = b3DDetectGaussionFilter;
        //m_stCalcHeightCmds[nStation - 1].bReverseSeq = b3DDetectReverseSeq;
        //stCalcHeightCmds[nStation - 1].fMinAmplitude = d3DDetectMinIntDiff;
        stCalcHeightCmds[nStation - 1].bUseThinnestPattern = b3DDetectCaliUseThinPattern;
        //stCalcHeightCmds[nStation - 1].fPhaseShift = d3DDetectPhaseShift;
        //stCalcHeightCmds[nStation - 1].nRemoveBetaJumpSpanX = 0;
        //stCalcHeightCmds[nStation - 1].nRemoveBetaJumpSpanY = 0;
        //stCalcHeightCmds[nStation - 1].nRemoveGammaJumpSpanX = 0;
        //stCalcHeightCmds[nStation - 1].nRemoveGammaJumpSpanY = 0;

        stCalcHeightCmds[nStation - 1].fMinAmplitude = 0;

        cv::Mat matBaseSurfaceParam;

        // read config file
        char filePath[100];
        _snprintf(filePath, sizeof(filePath), "./data/3D/Config/calibration3D_dlp%d.yml", nStation);
        cv::FileStorage fs(filePath, cv::FileStorage::READ);
        cv::FileNode fileNode = fs["K1"];
        cv::read(fileNode, stCalcHeightCmds[nStation - 1].matThickToThinK, cv::Mat());
        fileNode = fs["K2"];
        cv::read(fileNode, stCalcHeightCmds[nStation - 1].matThickToThinnestK, cv::Mat());
        fileNode = fs["BaseWrappedAlpha"];
        cv::read(fileNode, stCalcHeightCmds[nStation - 1].matBaseWrappedAlpha, cv::Mat());
        fileNode = fs["BaseWrappedBeta"];
        cv::read(fileNode, stCalcHeightCmds[nStation - 1].matBaseWrappedBeta, cv::Mat());
        fileNode = fs["BaseWrappedGamma"];
        cv::read(fileNode, stCalcHeightCmds[nStation - 1].matBaseWrappedGamma, cv::Mat());
        fileNode = fs["ReverseSeq"];
        cv::read(fileNode, stCalcHeightCmds[nStation - 1].bReverseSeq, 0);
        fs.release();

        _snprintf(filePath, sizeof(filePath), "./data/3D/Config/IntegrateCalibResult%d.yml", nStation);
        cv::FileStorage fsIntegrated(filePath, cv::FileStorage::READ);
        cv::FileNode fileNodeIntegrated = fsIntegrated["IntegratedK"];
        cv::read(fileNodeIntegrated, stCalcHeightCmds[nStation - 1].matIntegratedK, cv::Mat());
        fileNodeIntegrated = fsIntegrated["Order3CurveSurface"];
        cv::read(fileNodeIntegrated, stCalcHeightCmds[nStation - 1].matOrder3CurveSurface, cv::Mat());
        fsIntegrated.release();
    }

    //std::string arrStrFrameFolder[] =
    //{
    //    "D:/BaiduNetdiskDownload/0602163838/",
    //    "D:/BaiduNetdiskDownload/0602163936/",
    //    "D:/BaiduNetdiskDownload/0602163951/",
    //    "D:/BaiduNetdiskDownload/0602164005/",
    //};

    //std::string arrStrFrameFolder[] =
    //{
    //    "D:/BaiduNetdiskDownload/180606/Steel_1/0606150314/",
    //    "D:/BaiduNetdiskDownload/180606/Steel_1/0606150408/",
    //    "D:/BaiduNetdiskDownload/180606/Steel_1/0606150421/",
    //    "D:/BaiduNetdiskDownload/180606/Steel_1/0606150431/",
    //};

    /*std::string arrStrFrameFolder[] =
    {
        "D:/BaiduNetdiskDownload/180606/Steel_2/0606150705/",
        "D:/BaiduNetdiskDownload/180606/Steel_2/0606150716/",
        "D:/BaiduNetdiskDownload/180606/Steel_2/0606150728/",
        "D:/BaiduNetdiskDownload/180606/Steel_2/0606150740/",
    };*/

    std::string arrStrFrameFolder[] =
    {
        "D:/BaiduNetdiskDownload/180611/0611104300/",
        "D:/BaiduNetdiskDownload/180611/0611104356/",
        "D:/BaiduNetdiskDownload/180611/0611104411/",
        "D:/BaiduNetdiskDownload/180611/0611104426/",
    };

    int nFrameCount = sizeof(arrStrFrameFolder) / sizeof(arrStrFrameFolder[0]);

    std::vector<PR_CALC_DLP_OFFSET_RPY> vecDlpOffsetRpy;
    for (int nFrame = 0; nFrame < nFrameCount; ++ nFrame) {
        auto vecImages = ReadFrameImage(arrStrFrameFolder[nFrame]);
        if (vecImages.empty())
            return;

        for (int nDlp = 0; nDlp < 4; ++ nDlp) {
            stCalcHeightCmds[nDlp].vecInputImgs = VectorOfMat(vecImages.begin() + nDlp * IMAGE_COUNT, vecImages.begin() + (nDlp + 1) * IMAGE_COUNT);
            PR_Calc3DHeight(&stCalcHeightCmds[nDlp], &stCalcHeightRpys[nDlp]);

            //saveMatToCsv(stCalcHeightRpys[nDlp].matHeight, arrStrFrameFolder[nFrame] + "Dlp_" + std::to_string(nDlp + 1) + "_Height.csv");

            PR_HEIGHT_TO_GRAY_CMD stCmd;
            PR_HEIGHT_TO_GRAY_RPY stRpy;
            stCmd.matHeight = stCalcHeightRpys[nDlp].matHeight;
            PR_HeightToGray(&stCmd, &stRpy);

            //cv::imwrite(arrStrFrameFolder[nFrame] + "Dlp_" + std::to_string(nDlp + 1) + "_HeightGray.png", stRpy.matGray);            
        }

        PR_CALC_DLP_OFFSET_CMD stCalcDlpOffsetCmd;
        stCalcDlpOffsetCmd.matHeight1 = stCalcHeightRpys[0].matHeight;
        stCalcDlpOffsetCmd.matHeight2 = stCalcHeightRpys[1].matHeight;
        stCalcDlpOffsetCmd.matHeight3 = stCalcHeightRpys[2].matHeight;
        stCalcDlpOffsetCmd.matHeight4 = stCalcHeightRpys[3].matHeight;
        PR_CALC_DLP_OFFSET_RPY stCalcDlpOffsetRpy;
        PR_CalcDlpOffset(&stCalcDlpOffsetCmd, &stCalcDlpOffsetRpy);
        vecDlpOffsetRpy.push_back(stCalcDlpOffsetRpy);
    }

    std::cout << "Run here" << std::endl;
}

void TestCalc4DLPHeight()
{
    std::string strParentFolder = "./data/machine_image/20180903/";
    std::string strImageFolder = strParentFolder + "0903164442/";
    std::string strResultFolder = strParentFolder + "Frame_1_Result/";

    auto vecImages = ReadFrameImage(strImageFolder);
    if (vecImages.empty())
        return;

    PR_CALC_3D_HEIGHT_CMD stCalcHeightCmds[4];
    PR_CALC_3D_HEIGHT_RPY stCalcHeightRpys[4];
    bool b3DDetectCaliUseThinPattern = true;
    bool b3DDetectGaussionFilter = true;

    //float dlpOffset[4] = {5.3551e-04, -0.0014f, 0.0694, 0};
    float dlpOffset[4] = {0.0193f, -0.0032f, -0.0899, 0};

    for (int nDlp = 0; nDlp < 4; ++ nDlp) {
        stCalcHeightCmds[nDlp].bEnableGaussianFilter = b3DDetectGaussionFilter;
        stCalcHeightCmds[nDlp].bUseThinnestPattern = b3DDetectCaliUseThinPattern;
        stCalcHeightCmds[nDlp].fMinAmplitude = 5.f;
        //stCalcHeightCmds[nDlp].nRemoveBetaJumpMaxSpan = 0;
        //stCalcHeightCmds[nDlp].nRemoveBetaJumpMinSpan = 0;

        cv::Mat matBaseSurfaceParam;

        // read config file
        char filePath[100];
        _snprintf(filePath, sizeof(filePath), "./data/3D/Config/calibration3D_dlp%d.yml", nDlp + 1);
        cv::FileStorage fs(filePath, cv::FileStorage::READ);
        cv::FileNode fileNode = fs["K1"];
        cv::read(fileNode, stCalcHeightCmds[nDlp].matThickToThinK, cv::Mat());
        fileNode = fs["K2"];
        cv::read(fileNode, stCalcHeightCmds[nDlp].matThickToThinnestK, cv::Mat());

        Int32 projectDir, scanDir;
        fileNode = fs["ProjectDir"];
        cv::read(fileNode, projectDir, 0);
        stCalcHeightCmds[nDlp].enProjectDir = static_cast<PR_DIRECTION>(projectDir);
        fileNode = fs["ScanDir"];
        cv::read(fileNode, scanDir, 0);
        stCalcHeightCmds[nDlp].enScanDir = static_cast<PR_DIRECTION>(scanDir);

        fileNode = fs["BaseWrappedAlpha"];
        cv::read(fileNode, stCalcHeightCmds[nDlp].matBaseWrappedAlpha, cv::Mat());
        fileNode = fs["BaseWrappedBeta"];
        cv::read(fileNode, stCalcHeightCmds[nDlp].matBaseWrappedBeta, cv::Mat());
        fileNode = fs["BaseWrappedGamma"];
        cv::read(fileNode, stCalcHeightCmds[nDlp].matBaseWrappedGamma, cv::Mat());
        fileNode = fs["ReverseSeq"];
        cv::read(fileNode, stCalcHeightCmds[nDlp].bReverseSeq, 0);
        fs.release();

        _snprintf(filePath, sizeof(filePath), "./data/3D/Config/IntegrateCalibResult%d.yml", nDlp + 1);
        cv::FileStorage fsIntegrated(filePath, cv::FileStorage::READ);
        cv::FileNode fileNodeIntegrated = fsIntegrated["IntegratedK"];
        cv::read(fileNodeIntegrated, stCalcHeightCmds[nDlp].matIntegratedK, cv::Mat());
        fileNodeIntegrated = fsIntegrated["Order3CurveSurface"];
        cv::read(fileNodeIntegrated, stCalcHeightCmds[nDlp].matOrder3CurveSurface, cv::Mat());
        fsIntegrated.release();

        stCalcHeightCmds[nDlp].vecInputImgs = VectorOfMat(vecImages.begin() + nDlp * IMAGE_COUNT, vecImages.begin() + (nDlp + 1) * IMAGE_COUNT);
        PR_Calc3DHeight(&stCalcHeightCmds[nDlp], &stCalcHeightRpys[nDlp]);
        stCalcHeightRpys[nDlp].matHeight = stCalcHeightRpys[nDlp].matHeight + dlpOffset[nDlp];

        PrintMinMaxLoc(stCalcHeightRpys[nDlp].matHeight, "DLP_" + std::to_string(nDlp + 1) + "_result");

        PR_HEIGHT_TO_GRAY_CMD stCmd;
        PR_HEIGHT_TO_GRAY_RPY stRpy;
        stCmd.matHeight = stCalcHeightRpys[nDlp].matHeight;
        PR_HeightToGray(&stCmd, &stRpy);

        cv::imwrite(strResultFolder + "Dlp_" + std::to_string(nDlp + 1) + "_HeightGray.png", stRpy.matGray);
        cv::imwrite(strResultFolder + "Dlp_" + std::to_string(nDlp + 1) + "_NanMask.png", stCalcHeightRpys[nDlp].matNanMask);
        saveMatToCsv(stCalcHeightRpys[nDlp].matHeight, strResultFolder + "Dlp_" + std::to_string(nDlp + 1) + "_Height.csv");
    }

    VectorOfMat vecMatHeightMerges;
    VectorOfMat vecMatHeightNanMasks;
    for (int j = 0; j < 2; ++ j)
    {
        PR_MERGE_3D_HEIGHT_CMD stCmd;
        PR_MERGE_3D_HEIGHT_RPY stRpy;

        stCmd.vecMatHeight.push_back(stCalcHeightRpys[j + 0].matHeight);
        stCmd.vecMatHeight.push_back(stCalcHeightRpys[j + 2].matHeight);

        stCmd.vecMatNanMask.push_back(stCalcHeightRpys[j + 0].matNanMask);
        stCmd.vecMatNanMask.push_back(stCalcHeightRpys[j + 2].matNanMask);

        stCmd.enMethod = PR_MERGE_3D_HT_METHOD::SELECT_NEAREST_INTERSECT;
        stCmd.fHeightDiffThreshold = 0.1f;
        stCmd.enProjDir = stCalcHeightCmds[j + 0].enProjectDir;

        VisionStatus retStatus = PR_Merge3DHeight(&stCmd, &stRpy);
        if (retStatus == VisionStatus::OK) {
            vecMatHeightMerges.push_back(stRpy.matHeight);
            vecMatHeightNanMasks.push_back(stRpy.matNanMask);

            PR_HEIGHT_TO_GRAY_CMD stCovertCmd;
            PR_HEIGHT_TO_GRAY_RPY stConvertRpy;
            stCovertCmd.matHeight = stRpy.matHeight;
            PR_HeightToGray(&stCovertCmd, &stConvertRpy);

            cv::imwrite(strResultFolder + "Merge_" + std::to_string(j + 1) + "_HeightGray.png", stConvertRpy.matGray);
            cv::imwrite(strResultFolder + "Merge_" + std::to_string(j + 1) + "_NanMask.png", stRpy.matNanMask);
            saveMatToCsv(stCovertCmd.matHeight, strResultFolder + "Merge_" + std::to_string(j + 1) + "Height.csv");
        }
        else
        {
            std::cout << "PR_Merge3DHeight failed, status " << ToInt32(retStatus) << " at line " << __LINE__ << std::endl;
            return;
        }
    }

    PR_MERGE_3D_HEIGHT_CMD stMergeHCmd;
    PR_MERGE_3D_HEIGHT_RPY stMergeHRpy;

    stMergeHCmd.vecMatHeight = vecMatHeightMerges;
    stMergeHCmd.vecMatNanMask = vecMatHeightNanMasks;

    stMergeHCmd.enMethod = PR_MERGE_3D_HT_METHOD::SELECT_MAX;
    stMergeHCmd.fHeightDiffThreshold = 0.1f;
    stMergeHCmd.enProjDir = stCalcHeightCmds[0].enProjectDir;

    VisionStatus retStatus = PR_Merge3DHeight(&stMergeHCmd, &stMergeHRpy);
    if (retStatus == VisionStatus::OK)
    {
        PR_HEIGHT_TO_GRAY_CMD stCmd;
        PR_HEIGHT_TO_GRAY_RPY stRpy;
        stCmd.matHeight = stMergeHRpy.matHeight;
        PR_HeightToGray(&stCmd, &stRpy);

        cv::imwrite(strResultFolder + "Final_HeightGray.png", stRpy.matGray);
        cv::imwrite(strResultFolder + "Final_NanMask.png", stMergeHRpy.matNanMask);
        saveMatToCsv(stMergeHRpy.matHeight, strResultFolder + "Final_Height.csv");
    }
    else
    {
        std::cout << "PR_Merge3DHeight failed, status " << ToInt32(retStatus) << " at line " << __LINE__ << std::endl;
        return;
    }

    std::cout << "Success to calculate 4 DLP height" << std::endl;
}

void TestMergeHeightMax() {
    std::string strParentFolder = "./data/machine_image/20180903/";
    std::string strImageFolder = strParentFolder + "0903164501/";
    std::string strResultFolder = strParentFolder + "Frame_4_Result/";

    PR_MERGE_3D_HEIGHT_CMD stMergeHCmd;
    PR_MERGE_3D_HEIGHT_RPY stMergeHRpy;   
    stMergeHCmd.enMethod = PR_MERGE_3D_HT_METHOD::SELECT_MAX;
    stMergeHCmd.fHeightDiffThreshold = 0.2f;

    for (int i = 0; i < 2; ++ i) {
        cv::Mat matHeight = readMatFromCsvFile(strResultFolder + "Merge_" + std::to_string(i + 1) + "Height.csv");
        cv::Mat matMask = cv::imread(strResultFolder + "Merge_" + std::to_string(i + 1) + "_NanMask.png", cv::IMREAD_GRAYSCALE);

        stMergeHCmd.vecMatHeight.push_back(matHeight);
        stMergeHCmd.vecMatNanMask.push_back(matMask);
    }

    VisionStatus retStatus = PR_Merge3DHeight(&stMergeHCmd, &stMergeHRpy);

    if (retStatus == VisionStatus::OK)
    {
        PR_HEIGHT_TO_GRAY_CMD stCmd;
        PR_HEIGHT_TO_GRAY_RPY stRpy;
        stCmd.matHeight = stMergeHRpy.matHeight;
        PR_HeightToGray(&stCmd, &stRpy);

        cv::imwrite(strResultFolder + "Final_HeightGray.png", stRpy.matGray);
        cv::imwrite(strResultFolder + "Final_NanMask.png", stMergeHRpy.matNanMask);
        saveMatToCsv(stMergeHRpy.matHeight, strResultFolder + "Final_Height.csv");
    }
    else
    {
        std::cout << "PR_Merge3DHeight failed, status " << ToInt32(retStatus) << " at line " << __LINE__ << std::endl;
        return;
    }
    std::cout << "Merge 3 height finished";
}

void TestInsp3DSolder() {
    const std::string strParentFolder = "./data/TestInsp3DSolder/";
    PR_INSP_3D_SOLDER_CMD stCmd;
    PR_INSP_3D_SOLDER_RPY stRpy;

    stCmd.matHeight = readMatFromCsvFile(strParentFolder + "H3.csv");
    stCmd.matColorImg = cv::imread(strParentFolder + "image3.bmp", cv::IMREAD_COLOR);

    stCmd.rectDeviceROI = cv::Rect(1175, 575, 334, 179);
    stCmd.vecRectCheckROIs.push_back(cv::Rect(1197, 605, 72, 122));
    stCmd.vecRectCheckROIs.push_back(cv::Rect(1400, 605, 76, 122));

    //stCmd.rectDeviceROI = cv::Rect(1205, 1891, 302, 140);
    //stCmd.vecRectCheckROIs.push_back(cv::Rect(1218, 1897, 67, 108));
    //stCmd.vecRectCheckROIs.push_back(cv::Rect(1424, 1899, 63, 112));

    stCmd.nBaseColorDiff = 20;
    stCmd.nBaseGrayDiff = 20;
    
    cv::Rect rectBase(400, 200, 200, 200);
    cv::Mat matBaseRef(stCmd.matColorImg, rectBase);
    stCmd.scalarBaseColor = cv::mean(matBaseRef);

    PR_Insp3DSolder(&stCmd, &stRpy);

    std::cout << "PR_Insp3DSolder status " << ToInt32(stRpy.enStatus) << " at line " << __LINE__ << std::endl;
    if (VisionStatus::OK == stRpy.enStatus) {
        for (const auto &result : stRpy.vecResults) {
            std::cout << "Component height " << result.fComponentHeight << " Solder height " << result.fSolderHeight
                << ", area " << result.fSolderArea << ", ratio " << result.fSolderRatio << std::endl;
        }
        cv::imwrite(strParentFolder + "result.png", stRpy.matResultImg);
        cv::Mat matDeviceROI(stRpy.matResultImg, stCmd.rectDeviceROI);
        cv::imwrite(strParentFolder + "device.png", matDeviceROI);
    }
}

static void rotateImage(cv::Mat &matInOut) {
    cv::transpose(matInOut, matInOut);
    cv::flip(matInOut, matInOut, 1); //transpose+flip(1)=CW
}

static void PrintInsp3DSolderResult(const PR_INSP_3D_SOLDER_RPY &stRpy) {
    std::cout << "PR_Insp3DSolder status " << ToInt32(stRpy.enStatus) << " at line " << __LINE__ << std::endl;
    if (VisionStatus::OK == stRpy.enStatus) {
        for (const auto &result : stRpy.vecResults) {
            std::cout << "Component height " << result.fComponentHeight << " Solder height " << result.fSolderHeight
                << ", area " << result.fSolderArea << ", ratio " << result.fSolderRatio << std::endl;
        }
    }
}

void TestInsp3DSolder_1() {
    const std::string strParentFolder = "./data/TestInsp3DSolder/";
    PR_INSP_3D_SOLDER_CMD stCmd;
    PR_INSP_3D_SOLDER_RPY stRpy;

    stCmd.matHeight = readMatFromCsvFile(strParentFolder + "H3_Rotated.csv");
    stCmd.matColorImg = cv::imread(strParentFolder + "image3_rotated.png", cv::IMREAD_COLOR);

    //rotateImage(stCmd.matHeight);
    //rotateImage(stCmd.matColorImg);

    //saveMatToCsv(stCmd.matHeight, strParentFolder + "H3_Rotated.csv");
    //cv::imwrite(strParentFolder + "image3_rotated.png", stCmd.matColorImg);

    stCmd.rectDeviceROI = cv::Rect(501, 1177, 191, 350);
    stCmd.vecRectCheckROIs.push_back(cv::Rect(527, 1206, 138, 122));
    stCmd.vecRectCheckROIs.push_back(cv::Rect(527, 1400, 138, 122));

    stCmd.nBaseColorDiff = 20;
    stCmd.nBaseGrayDiff = 20;
    
    cv::Rect rectBase(730, 180, 200, 200);
    cv::Mat matBaseRef(stCmd.matColorImg, rectBase);
    stCmd.scalarBaseColor = cv::mean(matBaseRef);

    PR_Insp3DSolder(&stCmd, &stRpy);

    std::cout << "PR_Insp3DSolder status " << ToInt32(stRpy.enStatus) << " at line " << __LINE__ << std::endl;
    if (VisionStatus::OK == stRpy.enStatus) {
        for (const auto &result : stRpy.vecResults) {
            std::cout << "Component height " << result.fComponentHeight << " Solder height " << result.fSolderHeight
                << ", area " << result.fSolderArea << ", ratio " << result.fSolderRatio << std::endl;
        }
        cv::imwrite(strParentFolder + "result.png", stRpy.matResultImg);
        cv::Mat matDeviceROI(stRpy.matResultImg, stCmd.rectDeviceROI);
        cv::imwrite(strParentFolder + "device.png", matDeviceROI);
    }
}

void TestInsp3DSolder_2() {
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl << "INSP 3D SOLDER REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;

    const std::string strParentFolder = "./data/TestInsp3DSolder_2/";
    PR_INSP_3D_SOLDER_CMD stCmd;
    PR_INSP_3D_SOLDER_RPY stRpy;

    cv::FileStorage fs(strParentFolder + "Height.yml", cv::FileStorage::READ);
    cv::FileNode fileNode = fs["Height"];
    cv::read(fileNode, stCmd.matHeight, cv::Mat());
    fs.release();

    saveMatToCsv(stCmd.matHeight, strParentFolder + "height.csv");

    stCmd.matColorImg = cv::imread(strParentFolder + "image.png", cv::IMREAD_COLOR);

    stCmd.rectDeviceROI = cv::Rect(0, 0, 143, 73);
    stCmd.vecRectCheckROIs.push_back(cv::Rect(9, 15, 45, 47));
    stCmd.vecRectCheckROIs.push_back(cv::Rect(94, 14, 35, 47));

    stCmd.nBaseColorDiff = 20;
    stCmd.nBaseGrayDiff = 20;
    
    stCmd.scalarBaseColor = cv::Scalar(37, 22, 5);

    PR_Insp3DSolder(&stCmd, &stRpy);
    PrintInsp3DSolderResult(stRpy);
    cv::imwrite(strParentFolder + "result.png", stRpy.matResultImg);
}

void TestConvertToCsv() {
    const std::string strParentFolder = "./Vision/LogCase/Insp3DSolder_2018_10_01_23_49_04_553 - Copy/";
    cv::FileStorage fs(strParentFolder + "Height.yml", cv::FileStorage::READ);
    cv::FileNode fileNode = fs["Height"];
    cv::Mat matHeight;
    cv::read(fileNode, matHeight, cv::Mat());
    fs.release();

    saveMatToCsv(matHeight, strParentFolder + "height.csv");
}

int assignFrames(
        float                            left,
        float                            top,
        float                            right,
        float                            bottom,
        float                            fovWidth,
        float                            fovHeight,
        VectorOfVectorOfPoint2f         &vecVecFrameCtr,
        float                           &fOverlapX,
        float                           &fOverlapY)
{
    vecVecFrameCtr.clear();
    fOverlapX = 0.f, fOverlapY = 0.f;

    if (right <= left || top < bottom)
        return -1;

    int frameCountX = static_cast<int>((right - left) / fovWidth) + 1;
    int frameCountY = static_cast<int>((top - bottom) / fovHeight) + 1;
    
    if (frameCountX > 1)
        fOverlapX = (frameCountX * fovWidth - (right - left)) / (frameCountX - 1);
    else
        fOverlapX = 0.f;
    if (frameCountY > 1)
        fOverlapY = (frameCountY * fovHeight - (top - bottom)) / (frameCountY - 1);
    else
        fOverlapY = 0.f;

    for (int row = 0; row < frameCountY; ++ row) {
        VectorOfPoint2f vecFrameCtr;
        for (int col = 0; col < frameCountX; ++ col) {
            float frameCtrX = 0.f, frameCtrY = 0.f;
            if (frameCountX > 1)
                frameCtrX = left + (col * (fovWidth - fOverlapX) + fovWidth / 2.f);
            else
                frameCtrX = (right + left) / 2.f;

            if (frameCountY > 1)
                frameCtrY = top - (row * (fovHeight - fOverlapY) + fovHeight / 2.f);
            else
                frameCtrY = (top + bottom) / 2.f;
            vecFrameCtr.emplace_back(frameCtrX, frameCtrY);
        }
        vecVecFrameCtr.push_back(vecFrameCtr);
    }
    return 0;
}

void TestQueryDlpOffset()
{
    PR_CALC_FRAME_VALUE_CMD stCmd;
    PR_CALC_FRAME_VALUE_RPY stRpy;

    float fLeft = -134.61f, fTop = 0.f, fRight = -9.71f, fBottom = 0.f;
    float fFovWidth  = 2040 * 15.89f / 1000.f;
    float fFovHeight = 2048 * 15.89f / 1000.f;

    VectorOfVectorOfPoint2f         vecVecFrameCtr;
    float                           fOverlapX;
    float                           fOverlapY;

    assignFrames(fLeft, fTop, fRight, fBottom, fFovWidth, fFovHeight, vecVecFrameCtr, fOverlapX, fOverlapY);

    stCmd.vecVecRefFrameCenters = vecVecFrameCtr;
    stCmd.vecVecRefFrameValues = VectorOfVectorOfFloat(1, {0.15f, 0.04f, -0.06f, -0.09f});

    float fLeftFrame = vecVecFrameCtr[0].front().x;
    float fRightFrame = vecVecFrameCtr[0].back().x;
    VectorOfFloat vecQueryResult;
    for ( float fQuery = fLeftFrame; fQuery < fRightFrame; ++ fQuery) {
        stCmd.ptTargetFrameCenter.x = fQuery;
        stCmd.ptTargetFrameCenter.y = 0;

        PR_CalcFrameValue(&stCmd, &stRpy);
        if (stRpy.enStatus != VisionStatus::OK)
            return;

        vecQueryResult.push_back(stRpy.fResult);
    }
    
    std::cout << "Run to here" << std::endl;
}

static const std::string DLP_OFFSET_CALIB_WORKING_FOLDER("C:/Users/shengguang/Documents/AOI/3D/");
const int CALIB_FOV_ROWS = 5;
const int CALIB_FOV_COLS = 4;

static cv::Mat _calculateSurfaceConvert3D(const cv::Mat& z1, const VectorOfFloat& param, int ss, const cv::Mat& xxt1) {
    auto zmin = param[4];
    auto zmax = param[5];
    cv::Mat z1t; cv::transpose(z1, z1t);
    auto zInRow = z1t.reshape(1, z1.rows * z1.cols);
    cv::Mat w = (zInRow - zmin) / (zmax - zmin);
    int len = zInRow.rows;

    assert(len == z1.rows * z1.cols);

    cv::Mat Ps1 = cv::repeat(CalcUtils::intervals<float>(0.f, 1.f, ToFloat(ss - 1)).reshape(1, 1),  len, 1);
    cv::Mat Ps2 = cv::repeat(CalcUtils::intervals<float>(ToFloat(ss - 1), -1.f, 0.f).reshape(1, 1), len, 1);
    cv::Mat Ps3 = cv::repeat(w, 1, ss);
    cv::Mat Ps4 = cv::repeat(1 - w, 1, ss);

    auto matPolyParass = CalcUtils::paraFromPolyNomial(ss);

    // Matlab: P3 = (Ps3.^Ps1).*(Ps4.^Ps2).*repmat(PolyParass, len, 1);
    cv::Mat matTmpPmPow, P3;
    cv::multiply(CalcUtils::power<float>(Ps3, Ps1), CalcUtils::power<float>(Ps4, Ps2), matTmpPmPow);
    cv::multiply(matTmpPmPow, cv::repeat(matPolyParass, len, 1), P3);

    // Matlab: zp1 = sum(xxt1.*P3, 2);
    cv::multiply(xxt1, P3, P3);
    cv::Mat matResult;
    cv::reduce(P3, matResult, 1, cv::ReduceTypes::REDUCE_SUM);
    matResult = matResult.reshape(1, z1.rows);

    return matResult;
}

static bool CalcDlpHeight(VectorOfMat *pVecOfMat) {
    cv::Mat matX1, matY1, matX1T, matY1T;
    CalcUtils::meshgrid<float>(1.f, 1.f, 2040.f, 1.f, 1.f, 2048.f, matX1, matY1);
    std::vector<float> vecParamMinMaxXY{ 1.f, 2040.f, 1.f, 2048.f, -6.f, 6.f};
    int mm = 5, nn = 5, ss = 4;

    cv::transpose(matX1, matX1T);
    cv::transpose(matY1, matY1T);
    cv::Mat matXInRow = matX1T.reshape(1, matX1.rows * matX1.cols);
    cv::Mat matYInRow = matY1T.reshape(1, matX1.rows * matX1.cols);

    auto xxt = CalcUtils::generateBezier(matXInRow, matYInRow, vecParamMinMaxXY, mm, nn);
    const int _2D_PARAMS = xxt.cols;

    for (int nDlp = 0; nDlp < NUM_OF_DLP; ++ nDlp) {
        char strBezierParamFileName[100];
        _snprintf(strBezierParamFileName, sizeof(strBezierParamFileName), "BezierParameter/K%d.csv", nDlp + 1);
        auto vecOfVecFloat = readDataFromFile(DLP_OFFSET_CALIB_WORKING_FOLDER + strBezierParamFileName);        

        if (vecOfVecFloat.empty()) {
            std::cout << "Failed to read phase data from: " << DLP_OFFSET_CALIB_WORKING_FOLDER << strBezierParamFileName << std::endl;
            return false;
        }
        std::cout << "Success to read data from: " << DLP_OFFSET_CALIB_WORKING_FOLDER << strBezierParamFileName << std::endl;

        cv::Mat matK = vectorToMat(vecOfVecFloat);
        cv::Mat mat3DBezierSurface = cv::Mat::zeros(ss, xxt.rows, CV_32FC1);
        //Matlab: for ii = 1:ss
        for (int ii = 1; ii <= ss; ++ii) {
            // Matlab: xxt1(:, ii) = xxt*K((ii - 1)*len1 + 1:ii*len1);
            cv::Mat matXxt1ROI(mat3DBezierSurface, cv::Range(ii - 1, ii), cv::Range::all());
            cv::Mat matResult = xxt * cv::Mat(matK, cv::Range((ii - 1)* _2D_PARAMS, ii* _2D_PARAMS), cv::Range::all());
            matResult = matResult.reshape(1, 1);
            matResult.copyTo(matXxt1ROI);
        }

        cv::transpose(mat3DBezierSurface, mat3DBezierSurface);

        for (int row = 0; row < CALIB_FOV_ROWS; ++row)
            for (int col = 0; col < CALIB_FOV_COLS; ++col) {
                char strPhaseDataName[100];
                _snprintf(strPhaseDataName, sizeof(strPhaseDataName), "PhaseFOV/FOV1/DLP%d/H%d%d.csv", nDlp + 1, row, col);
                auto vecOfVecFloat = readDataFromFile(DLP_OFFSET_CALIB_WORKING_FOLDER + strPhaseDataName);
                if (vecOfVecFloat.empty()) {
                    std::cout << "Failed to read phase data from: " << DLP_OFFSET_CALIB_WORKING_FOLDER << strPhaseDataName << std::endl;
                    return false;
                }
                std::cout << "Success to read data from: " << DLP_OFFSET_CALIB_WORKING_FOLDER << strPhaseDataName << std::endl;

                cv::Mat matPhase = vectorToMat(vecOfVecFloat);
                cv::GaussianBlur(matPhase, matPhase, cv::Size(5, 5), 5, 5, cv::BorderTypes::BORDER_REPLICATE);
                
                auto matHeight = _calculateSurfaceConvert3D(matPhase, vecParamMinMaxXY, 4, mat3DBezierSurface);
                pVecOfMat[nDlp].push_back(matHeight);

                char strHeightDataName[100];
                _snprintf(strHeightDataName, sizeof(strHeightDataName), "PhaseFOV/FOV1/DLP%d/HResult%d%d.csv", nDlp + 1, row, col);
                CalcUtils::saveMatToCsv(matHeight, DLP_OFFSET_CALIB_WORKING_FOLDER + strHeightDataName);
                std::cout << "Success to write data to: " << DLP_OFFSET_CALIB_WORKING_FOLDER << strHeightDataName << std::endl;
            }
    }
    return true;
}

static bool ReadDlpHeight(VectorOfMat *pVecOfMat) {
    for (int nDlp = 0; nDlp < NUM_OF_DLP; ++ nDlp) {
        for (int row = 0; row < CALIB_FOV_ROWS; ++row)
            for (int col = 0; col < CALIB_FOV_COLS; ++col) {
                char strHeightDataName[100];
                _snprintf(strHeightDataName, sizeof(strHeightDataName), "PhaseFOV/FOV1/DLP%d/HResult%d%d.csv", nDlp + 1, row, col);
                auto vecOfVecFloat = readDataFromFile(DLP_OFFSET_CALIB_WORKING_FOLDER + strHeightDataName);
                if (vecOfVecFloat.empty()) {
                    std::cout << "Failed to read phase data from: " << DLP_OFFSET_CALIB_WORKING_FOLDER << strHeightDataName << std::endl;
                    return false;
                }
                std::cout << "Success to read data from: " << DLP_OFFSET_CALIB_WORKING_FOLDER << strHeightDataName << std::endl;

                cv::Mat matHeight = vectorToMat(vecOfVecFloat);
                pVecOfMat[nDlp].push_back(matHeight);
            }
    }
    return true;
}

void TestCalibDlpOffset_1() {
    PR_CALIB_DLP_OFFSET_CMD stCmd;
    PR_CALIB_DLP_OFFSET_RPY stRpy;

    if (!ReadDlpHeight(stCmd.arrVecDlpH))
        return;

    stCmd.fResolution = 0.0159f;
    stCmd.fFrameDistX = 27.7946f;
    stCmd.fFrameDistY = 24.739f;
    stCmd.nCalibPosRows = CALIB_FOV_ROWS;
    stCmd.nCalibPosCols = CALIB_FOV_COLS;

    PR_CalibDlpOffset(&stCmd, &stRpy);
    std::cout << "Dlp offset: " << std::endl;
    for (int nDlp = 0; nDlp < NUM_OF_DLP; ++ nDlp) {
        std::cout << stRpy.arrOffset[nDlp] << std::endl;

        char strSurfaceDataName[100];
        _snprintf(strSurfaceDataName, sizeof(strSurfaceDataName), "PhaseFOV/FOV1/Dlp_Offset_Surface_%d.csv", nDlp + 1);
        CalcUtils::saveMatToCsv(stRpy.arrMatRotationSurface[nDlp], DLP_OFFSET_CALIB_WORKING_FOLDER + strSurfaceDataName);
        std::cout << "Success to write surface data to: " << DLP_OFFSET_CALIB_WORKING_FOLDER << strSurfaceDataName << std::endl;
    }
}
