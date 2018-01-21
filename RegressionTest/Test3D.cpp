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

void TestCalib3dBaseSub(
    int             nCaseNo,
    const String   &strParentPath,
    const String   &strImageFolder) {
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "CALIB 3D BASE TEST #" << nCaseNo << " STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    const int IMAGE_COUNT = 12;
    std::string strFolder = strParentPath + strImageFolder + "/";
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

    std::string strResultMatPath = strParentPath + "CalibPP.yml";
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

void TestCalib3DHeightSub(
    int             nCaseNo,
    int             nResultNo,
    bool            bUseGamma,
    const String   &strParentPath,
    const String   &strImageFolder,
    bool            bReverseHeight,
    int             nBlockStepCount,
    const cv::Mat  &matThickToThinK,
    const cv::Mat  &matThickToThinnestK,
    const cv::Mat  &matBaseWrappedAlpha,
    const cv::Mat  &matBaseWrappedBeta,
    const cv::Mat  &matBaseWrappedGamma
    ) {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALIB 3D HEIGHT TEST #" << nCaseNo << " STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    const int IMAGE_COUNT = 12;
    PR_CALIB_3D_HEIGHT_CMD stCmd;
    PR_CALIB_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strParentPath + strImageFolder + "/" + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    stCmd.bUseThinnestPattern = bUseGamma;
    stCmd.bReverseHeight = bReverseHeight;
    if ( stCmd.bUseThinnestPattern )
        stCmd.fMinAmplitude = 8.f;
    else
        stCmd.fMinAmplitude = 2.f;
    stCmd.nBlockStepCount = nBlockStepCount;
    stCmd.fBlockStepHeight = 1.f;    

    stCmd.matThickToThinK = matThickToThinK;
    stCmd.matThickToThinnestK = matThickToThinnestK;
    stCmd.matBaseWrappedAlpha = matBaseWrappedAlpha;
    stCmd.matBaseWrappedBeta = matBaseWrappedBeta;
    stCmd.matBaseWrappedGamma = matBaseWrappedGamma;

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

    String strCaseNo = std::to_string(nCaseNo);
    cv::imwrite ( strParentPath + "PR_Calib3DHeight_DivideStepResultImg_"    + strCaseNo + ".png", stRpy.matDivideStepResultImg );
    cv::imwrite ( strParentPath + "PR_Calib3DHeight_Calib3DHeightResultImg_" + strCaseNo + ".png", stRpy.matResultImg );

    String strResultNo = std::to_string(nResultNo);
    std::string strCalibDataFile(strParentPath + strResultNo + ".yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    cv::write ( fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex );
    fsCalibData.release();
}

void TestCalcTopSurface3DHeightSub(
    int             nCaseNo,
    bool            bUseGamma,
    const String   &strParentPath,
    const String   &strImageFolder,
    const cv::Mat  &matThickToThinK,
    const cv::Mat  &matThickToThinnestK,
    const cv::Mat  &matBaseWrappedAlpha,
    const cv::Mat  &matBaseWrappedBeta,
    const cv::Mat  &matBaseWrappedGamma,
    float           fPhaseShift = 0.f
    ) {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALC 3D HEIGHT TEST #" << nCaseNo << " STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    const int IMAGE_COUNT = 12;
    PR_CALC_3D_HEIGHT_CMD stCmd;
    PR_CALC_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strParentPath + strImageFolder + "/" + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    stCmd.bUseThinnestPattern = bUseGamma;
    stCmd.fMinAmplitude = 1.5;
    stCmd.fPhaseShift = fPhaseShift;

    stCmd.matThickToThinK = matThickToThinK;
    stCmd.matThickToThinnestK = matThickToThinnestK;
    stCmd.matBaseWrappedAlpha = matBaseWrappedAlpha;
    stCmd.matBaseWrappedBeta = matBaseWrappedBeta;
    stCmd.matBaseWrappedGamma = matBaseWrappedGamma;

    PR_Calc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::Mat matMask = stRpy.matPhase == stRpy.matPhase;
    float fMeanPhase = ToFloat ( cv::mean ( stRpy.matPhase, matMask )[0] );
    std::cout << "Mean Phase: " << fMeanPhase << std::endl;

    std::cout << "Grid snoop phase: " << std::endl;
    cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matHeight, 10, 10 );
    cv::imwrite ( strParentPath + "PR_Calc3DHeight_H5_PhaseGridImg.png", matPhaseResultImg );

    std::string strCalibDataFile( strParentPath + "H5.yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    fsCalibData.release();
}

void TestIntegrate3DCalibSub(
    int              nCaseNo,
    const String    &strParentPath) {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INTEGRATE 3D HEIGHT CALIBRATION #" << nCaseNo << " STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_INTEGRATE_3D_CALIB_CMD stCmd;
    PR_INTEGRATE_3D_CALIB_RPY stRpy;
    for ( int i = 1; i <= 3; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%d.yml", i );
        std::string strDataFile = strParentPath + chArrFileName;
        
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

    std::string strCalibDataFile( strParentPath + "H5.yml" );
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
        _snprintf( chArrFileName, sizeof (chArrFileName), "PR_Integrate3DCalib_ResultImg_%02d.png", i );
        std::string strDataFile = strParentPath + chArrFileName;
        cv::imwrite ( strDataFile, matResultImg );
        ++ i;
    }
    std::cout << "IntegratedK" << std::endl;
    printfMat<float>( stRpy.matIntegratedK, 5 );

    std::string strCalibResultFile( strParentPath + "IntegrateCalibResult.yml" );
    cv::FileStorage fsCalibResultData ( strCalibResultFile, cv::FileStorage::WRITE );
    if (!fsCalibResultData.isOpened ()) {
        std::cout << "Failed to open file: " << strCalibResultFile << std::endl;
        return;
    }
    cv::write ( fsCalibResultData, "IntegratedK", stRpy.matIntegratedK );
    cv::write ( fsCalibResultData, "Order3CurveSurface", stRpy.matOrder3CurveSurface );
    fsCalibResultData.release();
}

void TestCalib3dBase_1() {
    String strParentPath = "./data/HaoYu_20171017/";
    String strImageFolder = "1017000816_base";
    TestCalib3dBaseSub ( 1, strParentPath, strImageFolder );
}

void TestCalib3DHeight_1() {
    String strParentPath = "./data/HaoYu_20171017/";
    String strCalibResultFile = strParentPath + "CalibPP.yml";
    cv::FileStorage fs ( strCalibResultFile, cv::FileStorage::READ );
    cv::Mat matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma;
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, matBaseWrappedGamma, cv::Mat() );
    fs.release();

    TestCalib3DHeightSub ( 1, 1, false, strParentPath, "1017001014_LT",       false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub ( 2, 2, false, strParentPath, "1017001107_RB",       false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub ( 3, 3, false, strParentPath, "1017001229_negative", true,  3, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalcTopSurface3DHeightSub (1, false, strParentPath, "1017000903_H5",            matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestIntegrate3DCalibSub ( 1, strParentPath );
}

void TestCalib3DHeight_2() {
    String strParentPath = "./data/HaoYu_20171017/";
    String strCalibResultFile = strParentPath + "CalibPP.yml";
    cv::FileStorage fs ( strCalibResultFile, cv::FileStorage::READ );
    cv::Mat matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma;
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, matBaseWrappedGamma, cv::Mat() );
    fs.release();

    TestCalib3DHeightSub ( 4, 1, true, strParentPath, "1017001014_LT",       false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub ( 5, 2, true, strParentPath, "1017001107_RB",       false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub ( 6, 3, true, strParentPath, "1017001229_negative", true,  3, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalcTopSurface3DHeightSub ( 2, true, strParentPath, "1017000903_H5",           matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestIntegrate3DCalibSub ( 2, strParentPath );
}

/******************************************/
/*The second group test case*/
/******************************************/
void TestCalib3dBase_2() {
    String strParentPath = "./data/HaoYu_20170920/";
    String strImageFolder = "0920234214_base";
    TestCalib3dBaseSub ( 2, strParentPath, strImageFolder );
}

void TestCalib3DHeight_3() {
    String strParentPath = "./data/HaoYu_20170920/";
    String strCalibResultFile = strParentPath + "CalibPP.yml";
    cv::FileStorage fs ( strCalibResultFile, cv::FileStorage::READ );
    cv::Mat matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma;
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, matBaseWrappedGamma, cv::Mat() );
    fs.release();

    TestCalib3DHeightSub ( 7, 1, false, strParentPath, "0920235040_lefttop",     false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub ( 8, 2, false, strParentPath, "0920235128_rightbottom", false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub ( 9, 3, false, strParentPath, "0920235405_negitive",    true,  3, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalcTopSurface3DHeightSub (3, false, strParentPath, "0921202157_H5mm",             matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma, 0.1f );
    TestIntegrate3DCalibSub ( 3, strParentPath );
}

void TestCalib3DHeight_4() {
    String strParentPath = "./data/HaoYu_20170920/";
    String strCalibResultFile = strParentPath + "CalibPP.yml";
    cv::FileStorage fs ( strCalibResultFile, cv::FileStorage::READ );
    cv::Mat matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma;
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, matBaseWrappedGamma, cv::Mat() );
    fs.release();

    bool bUseGamma = true;
    TestCalib3DHeightSub ( 10, 1, bUseGamma, strParentPath,  "0920235040_lefttop",     false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub ( 11, 2, bUseGamma, strParentPath,  "0920235128_rightbottom", false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub ( 12, 3, bUseGamma, strParentPath,  "0920235405_negitive",    true,  3, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalcTopSurface3DHeightSub ( 4, false, strParentPath, "0921202157_H5mm",                  matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma, 0.1f );
    TestIntegrate3DCalibSub ( 4, strParentPath );
}

void Test3D() {
    TestCalib3dBase_1();
    TestCalib3DHeight_1();
    TestCalib3DHeight_2();

    TestCalib3dBase_2();
    TestCalib3DHeight_3();
    TestCalib3DHeight_4();
}

}
}