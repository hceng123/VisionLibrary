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
    for (int j = 0; j < nGridRow; ++j) {
        for (int i = 0; i < nGridCol; ++i) {
            cv::Rect rectROI(i * nIntervalX + nIntervalX / 2 - szMeasureWinSize.width / 2,
                j * nIntervalY + nIntervalY / 2 - szMeasureWinSize.height / 2,
                szMeasureWinSize.width, szMeasureWinSize.height);
            cv::rectangle(matResultImg, rectROI, cv::Scalar(0, 255, 0), 3);

            cv::Mat matROI(matHeight, rectROI);
            cv::Mat matMask = (matROI == matROI);
            float fAverage = ToFloat(cv::mean(matROI, matMask)[0]);

            char strAverage[100];
            _snprintf(strAverage, sizeof(strAverage), "%.3f", fAverage);
            std::cout << strAverage << " ";
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(strAverage, fontFace, fontScale, thickness, &baseline);
            //The height use '+' because text origin start from left-bottom.
            cv::Point ptTextOrg(rectROI.x + (rectROI.width - textSize.width) / 2, rectROI.y + (rectROI.height + textSize.height) / 2);
            cv::putText(matResultImg, strAverage, ptTextOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness);
        }
        std::cout << std::endl;
    }

    int nGridLineSize = 3;
    cv::Scalar scalarCyan(255, 255, 0);
    for (int i = 1; i < nGridCol; ++i)
        cv::line(matResultImg, cv::Point(i * nIntervalX, 0), cv::Point(i * nIntervalX, ROWS), scalarCyan, nGridLineSize);
    for (int i = 1; i < nGridRow; ++i)
        cv::line(matResultImg, cv::Point(0, i * nIntervalY), cv::Point(COLS, i * nIntervalY), scalarCyan, nGridLineSize);

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
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.bmp", i);
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread(strImageFile, cv::IMREAD_GRAYSCALE);
        stCmd.vecInputImgs.push_back(mat);
    }
    stCmd.bEnableGaussianFilter = true;
    PR_Calib3DBase(&stCmd, &stRpy);
    std::cout << "PR_Calib3DBase status " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << "ReverseSeq: " << stRpy.bReverseSeq << std::endl;
    std::cout << "Project Dir: " << ToInt32(stRpy.enProjectDir) << std::endl;
    std::cout << "Scan Dir: " << ToInt32(stRpy.enScanDir) << std::endl;
    std::cout << "ThickToThinK: " << std::endl;
    printfMat<float>(stRpy.matThickToThinK);
    std::cout << "ThickToThinnestK: " << std::endl;
    printfMat<float>(stRpy.matThickToThinnestK);

    std::string strResultMatPath = strParentPath + "CalibPP.yml";
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        return;
    cv::write(fs, "ReverseSeq", stRpy.bReverseSeq);
    cv::write(fs, "K1", stRpy.matThickToThinK);
    cv::write(fs, "K2", stRpy.matThickToThinnestK);
    cv::write(fs, "BaseWrappedAlpha", stRpy.matBaseWrappedAlpha);
    cv::write(fs, "BaseWrappedBeta", stRpy.matBaseWrappedBeta);
    cv::write(fs, "BaseWrappedGamma", stRpy.matBaseWrappedGamma);
    fs.release();
}

void TestCalib3DHeightSub(
    int             nCaseNo,
    int             nResultNo,
    bool            bUseGamma,
    const String   &strParentPath,
    const String   &strImageFolder,
    bool            bReverseSeq,
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
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.bmp", i);
        std::string strImageFile = strParentPath + strImageFolder + "/" + chArrFileName;
        cv::Mat mat = cv::imread(strImageFile, cv::IMREAD_GRAYSCALE);
        stCmd.vecInputImgs.push_back(mat);
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = bReverseSeq;
    stCmd.bUseThinnestPattern = bUseGamma;
    stCmd.bReverseHeight = bReverseHeight;
    if (stCmd.bUseThinnestPattern)
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

    PR_Calib3DHeight(&stCmd, &stRpy);
    std::cout << "PR_Calib3DHeight status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;
    std::cout << "PhaseToHeightK: " << std::endl;
    printfMat<float>(stRpy.matPhaseToHeightK);

    std::cout << "Fetched StepPhase: " << std::endl;
    printfVectorOfVector<float>(stRpy.vecVecStepPhase);

    std::cout << "Step-Phase slope: " << std::endl;
    for (const auto &value : stRpy.vecStepPhaseSlope)
        printf("%.2f ", value);
    std::cout << std::endl;

    std::cout << "StepPhaseDiff: " << std::endl;
    printfVectorOfVector<float>(stRpy.vecVecStepPhaseDiff);

    String strCaseNo = std::to_string(nCaseNo);
    cv::imwrite(strParentPath + "PR_Calib3DHeight_DivideStepResultImg_" + strCaseNo + ".png", stRpy.matDivideStepResultImg);
    cv::imwrite(strParentPath + "PR_Calib3DHeight_Calib3DHeightResultImg_" + strCaseNo + ".png", stRpy.matResultImg);

    String strResultNo = std::to_string(nResultNo);
    std::string strCalibDataFile(strParentPath + strResultNo + ".yml");
    cv::FileStorage fsCalibData(strCalibDataFile, cv::FileStorage::WRITE);
    if (!fsCalibData.isOpened())
        return;
    cv::write(fsCalibData, "Phase", stRpy.matPhase);
    cv::write(fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex);
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
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.bmp", i);
        std::string strImageFile = strParentPath + strImageFolder + "/" + chArrFileName;
        cv::Mat mat = cv::imread(strImageFile, cv::IMREAD_GRAYSCALE);
        stCmd.vecInputImgs.push_back(mat);
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
    stCmd.nRemoveBetaJumpMaxSpan = 0;
    stCmd.nRemoveBetaJumpMinSpan = 0;
    stCmd.nRemoveGammaJumpSpanX = 0;
    stCmd.nRemoveGammaJumpSpanY = 0;

    PR_Calc3DHeight(&stCmd, &stRpy);
    std::cout << "PR_Calc3DHeight status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    cv::Mat matMask = stRpy.matPhase == stRpy.matPhase;
    float fMeanPhase = ToFloat(cv::mean(stRpy.matPhase, matMask)[0]);
    std::cout << "Mean Phase: " << fMeanPhase << std::endl;

    std::cout << "Grid snoop phase: " << std::endl;
    cv::Mat matPhaseResultImg = _drawHeightGrid(stRpy.matHeight, 10, 10);
    cv::imwrite(strParentPath + "PR_Calc3DHeight_H5_PhaseGridImg.png", matPhaseResultImg);

    std::string strCalibDataFile(strParentPath + "H5.yml");
    cv::FileStorage fsCalibData(strCalibDataFile, cv::FileStorage::WRITE);
    if (!fsCalibData.isOpened())
        return;
    cv::write(fsCalibData, "Phase", stRpy.matPhase);
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
    for (int i = 1; i <= 3; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%d.yml", i);
        std::string strDataFile = strParentPath + chArrFileName;

        std::string strCalibDataFile(strDataFile);
        cv::FileStorage fsCalibData(strCalibDataFile, cv::FileStorage::READ);
        if (!fsCalibData.isOpened()) {
            std::cout << "Failed to open file: " << strCalibDataFile << std::endl;
            return;
        }

        PR_INTEGRATE_3D_CALIB_CMD::SINGLE_CALIB_DATA stCalibData;
        cv::FileNode fileNode = fsCalibData["Phase"];
        cv::read(fileNode, stCalibData.matPhase, cv::Mat());

        fileNode = fsCalibData["DivideStepIndex"];
        cv::read(fileNode, stCalibData.matDivideStepIndex, cv::Mat());
        fsCalibData.release();

        stCmd.vecCalibData.push_back(stCalibData);
    }

    std::string strCalibDataFile(strParentPath + "H5.yml");
    cv::FileStorage fsCalibData(strCalibDataFile, cv::FileStorage::READ);
    if (!fsCalibData.isOpened()) {
        std::cout << "Failed to open file: " << strCalibDataFile << std::endl;
        return;
    }
    cv::FileNode fileNode = fsCalibData["Phase"];
    cv::read(fileNode, stCmd.matTopSurfacePhase, cv::Mat());
    fsCalibData.release();

    stCmd.fTopSurfaceHeight = 5;

    PR_Integrate3DCalib(&stCmd, &stRpy);
    std::cout << "PR_Integrate3DCalib status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    int i = 1;
    for (const auto &matResultImg : stRpy.vecMatResultImg) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "PR_Integrate3DCalib_ResultImg_%02d.png", i);
        std::string strDataFile = strParentPath + chArrFileName;
        cv::imwrite(strDataFile, matResultImg);
        ++i;
    }
    std::cout << "IntegratedK" << std::endl;
    printfMat<float>(stRpy.matIntegratedK, 4);

    std::string strCalibResultFile(strParentPath + "IntegrateCalibResult.yml");
    cv::FileStorage fsCalibResultData(strCalibResultFile, cv::FileStorage::WRITE);
    if (!fsCalibResultData.isOpened()) {
        std::cout << "Failed to open file: " << strCalibResultFile << std::endl;
        return;
    }
    cv::write(fsCalibResultData, "IntegratedK", stRpy.matIntegratedK);
    cv::write(fsCalibResultData, "Order3CurveSurface", stRpy.matOrder3CurveSurface);
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
    cv::FileStorage fs(strCalibResultFile, cv::FileStorage::READ);
    cv::Mat matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma;
    bool bReverseSeq = false;
    cv::FileNode fileNode;
    fileNode = fs["ReverseSeq"];
    cv::read(fileNode, bReverseSeq, false);
    fileNode = fs["K1"];
    cv::read(fileNode, matThickToThinK, cv::Mat());
    fileNode = fs["K2"];
    cv::read(fileNode, matThickToThinnestK, cv::Mat());
    fileNode = fs["BaseWrappedAlpha"];
    cv::read(fileNode, matBaseWrappedAlpha, cv::Mat());
    fileNode = fs["BaseWrappedBeta"];
    cv::read(fileNode, matBaseWrappedBeta, cv::Mat());
    fileNode = fs["BaseWrappedGamma"];
    cv::read(fileNode, matBaseWrappedGamma, cv::Mat());
    fs.release();

    TestCalib3DHeightSub(1, 1, false, strParentPath, "1017001014_LT", bReverseSeq,       false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma);
    TestCalib3DHeightSub(2, 2, false, strParentPath, "1017001107_RB", bReverseSeq,       false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma);
    TestCalib3DHeightSub(3, 3, false, strParentPath, "1017001229_negative", bReverseSeq, true,  3, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma);
    TestCalcTopSurface3DHeightSub(1, false, strParentPath, "1017000903_H5", matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma);
    TestIntegrate3DCalibSub(1, strParentPath);
}

void TestCalib3DHeight_2() {
    String strParentPath = "./data/HaoYu_20171017/";
    String strCalibResultFile = strParentPath + "CalibPP.yml";
    cv::FileStorage fs(strCalibResultFile, cv::FileStorage::READ);
    cv::Mat matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma;
    bool bReverseSeq = false;
    cv::FileNode fileNode;
    fileNode = fs["ReverseSeq"];
    cv::read(fileNode, bReverseSeq, false);
    fileNode = fs["K1"];
    cv::read(fileNode, matThickToThinK, cv::Mat());
    fileNode = fs["K2"];
    cv::read(fileNode, matThickToThinnestK, cv::Mat());
    fileNode = fs["BaseWrappedAlpha"];
    cv::read(fileNode, matBaseWrappedAlpha, cv::Mat());
    fileNode = fs["BaseWrappedBeta"];
    cv::read(fileNode, matBaseWrappedBeta, cv::Mat());
    fileNode = fs["BaseWrappedGamma"];
    cv::read(fileNode, matBaseWrappedGamma, cv::Mat());
    fs.release();

    TestCalib3DHeightSub(4, 1, true, strParentPath, "1017001014_LT",       bReverseSeq, false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma);
    TestCalib3DHeightSub(5, 2, true, strParentPath, "1017001107_RB",       bReverseSeq, false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma);
    TestCalib3DHeightSub(6, 3, true, strParentPath, "1017001229_negative", bReverseSeq, true,  3, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma);
    TestCalcTopSurface3DHeightSub(2, true, strParentPath, "1017000903_H5",                        matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma);
    TestIntegrate3DCalibSub(2, strParentPath);
}

/******************************************/
/*The second group test case*/
/******************************************/
void TestCalib3dBase_2() {
    String strParentPath = "./data/HaoYu_20170920/";
    String strImageFolder = "0920234214_base";
    TestCalib3dBaseSub(2, strParentPath, strImageFolder);
}

void TestCalib3DHeight_3() {
    String strParentPath = "./data/HaoYu_20170920/";
    String strCalibResultFile = strParentPath + "CalibPP.yml";
    cv::FileStorage fs ( strCalibResultFile, cv::FileStorage::READ );
    cv::Mat matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma;
    bool bReverseSeq = false;
    cv::FileNode fileNode;
    fileNode = fs["ReverseSeq"];
    cv::read(fileNode, bReverseSeq, false);
    fileNode = fs["K1"];
    cv::read(fileNode, matThickToThinK, cv::Mat());
    fileNode = fs["K2"];
    cv::read(fileNode, matThickToThinnestK, cv::Mat());
    fileNode = fs["BaseWrappedAlpha"];
    cv::read(fileNode, matBaseWrappedAlpha, cv::Mat());
    fileNode = fs["BaseWrappedBeta"];
    cv::read(fileNode, matBaseWrappedBeta, cv::Mat());
    fileNode = fs["BaseWrappedGamma"];
    cv::read(fileNode, matBaseWrappedGamma, cv::Mat());
    fs.release();

    TestCalib3DHeightSub(7, 1, false, strParentPath, "0920235040_lefttop",     bReverseSeq, false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub(8, 2, false, strParentPath, "0920235128_rightbottom", bReverseSeq, false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub(9, 3, false, strParentPath, "0920235405_negitive",    bReverseSeq, true,  3, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalcTopSurface3DHeightSub(3, false, strParentPath, "0921202157_H5mm",                          matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma, 0.1f );
    TestIntegrate3DCalibSub( 3, strParentPath );
}

void TestCalib3DHeight_4() {
    String strParentPath = "./data/HaoYu_20170920/";
    String strCalibResultFile = strParentPath + "CalibPP.yml";
    cv::FileStorage fs ( strCalibResultFile, cv::FileStorage::READ );
    cv::Mat matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma;
    bool bReverseSeq = false;
    cv::FileNode fileNode;
    fileNode = fs["ReverseSeq"];
    cv::read(fileNode, bReverseSeq, false);
    fileNode = fs["K1"];
    cv::read(fileNode, matThickToThinK, cv::Mat());
    fileNode = fs["K2"];
    cv::read(fileNode, matThickToThinnestK, cv::Mat());
    fileNode = fs["BaseWrappedAlpha"];
    cv::read(fileNode, matBaseWrappedAlpha, cv::Mat());
    fileNode = fs["BaseWrappedBeta"];
    cv::read(fileNode, matBaseWrappedBeta, cv::Mat());
    fileNode = fs["BaseWrappedGamma"];
    cv::read(fileNode, matBaseWrappedGamma, cv::Mat());
    fs.release();

    bool bUseGamma = true;
    TestCalib3DHeightSub(10, 1, bUseGamma, strParentPath,  "0920235040_lefttop",     bReverseSeq, false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub(11, 2, bUseGamma, strParentPath,  "0920235128_rightbottom", bReverseSeq, false, 5, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalib3DHeightSub(12, 3, bUseGamma, strParentPath,  "0920235405_negitive",    bReverseSeq, true,  3, matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma );
    TestCalcTopSurface3DHeightSub(4, false, strParentPath, "0921202157_H5mm",                               matThickToThinK, matThickToThinnestK, matBaseWrappedAlpha, matBaseWrappedBeta, matBaseWrappedGamma, 0.1f );
    TestIntegrate3DCalibSub(4, strParentPath );
}

void TestCalib3dBase_3() {
    String strParentPath = "./data/";
    String strImageFolder = "DLPImage_ReverseSeqCheck";
    TestCalib3dBaseSub(3, strParentPath, strImageFolder);
}

void TestMotor3DCalib_1() {
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl << "MOTOR 3D CALIB REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;

    PR_MOTOR_CALIB_3D_CMD stCmd;
    PR_MOTOR_CALIB_3D_RPY stRpy;

    std::string strDataFolder = "./data/Motor3DCalib/", strFile;

    for (int i = 1; i <= 5; ++i) {
        PairHeightPhase pairHeightPhase;
        pairHeightPhase.first = ToFloat(-i);
        strFile = strDataFolder + "HP" + std::to_string(i) + ".csv";
        pairHeightPhase.second = readMatFromCsvFile(strFile);
        stCmd.vecPairHeightPhase.push_back(pairHeightPhase);
    }

    for (int i = 1; i <= 5; ++i) {
        PairHeightPhase pairHeightPhase;
        pairHeightPhase.first = ToFloat(i);
        strFile = strDataFolder + "HN" + std::to_string(i) + ".csv";
        pairHeightPhase.second = readMatFromCsvFile(strFile);
        stCmd.vecPairHeightPhase.push_back(pairHeightPhase);
    }

    PR_MotorCalib3D(&stCmd, &stRpy);
    std::cout << "PR_MotorCalib3D status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    printfMat<float>(stRpy.matIntegratedK, 3);
}

void TestMotor3DCalibNew_1() {
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "MOTOR 3D CALIB NEW REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    PR_MOTOR_CALIB_3D_CMD stCmd;
    PR_MOTOR_CALIB_3D_NEW_RPY stRpy;

    std::string strDataFolder = "./data/Motor3DCalib/", strFile;

    for (int i = 1; i <= 5; ++i) {
        PairHeightPhase pairHeightPhase;
        pairHeightPhase.first = ToFloat(-i);
        strFile = strDataFolder + "HP" + std::to_string(i) + ".csv";
        pairHeightPhase.second = readMatFromCsvFile(strFile);
        stCmd.vecPairHeightPhase.push_back(pairHeightPhase);
    }

    for (int i = 1; i <= 5; ++i) {
        PairHeightPhase pairHeightPhase;
        pairHeightPhase.first = ToFloat(i);
        strFile = strDataFolder + "HN" + std::to_string(i) + ".csv";
        pairHeightPhase.second = readMatFromCsvFile(strFile);
        stCmd.vecPairHeightPhase.push_back(pairHeightPhase);
    }

    PR_MotorCalib3DNew(&stCmd, &stRpy);
    std::cout << "PR_MotorCalib3D status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    printfMat<float>(stRpy.mat3DBezierK, 3);
    for (const auto& gridHeight : stRpy.vecHeightGridHeights) {
        std::cout << "Height " << gridHeight.first << " grid result " << std::endl;
        for (int row = 0; row < stCmd.nResultImgGridRow; ++ row) {
            for (int col = 0; col < stCmd.nResultImgGridCol; ++ col) {
                int index = col + row * stCmd.nResultImgGridCol;
                std::cout << gridHeight.second[index] << " ";
            }
            std::cout << std::endl;
        }
    }
}

static cv::Mat readMatFromYml(const std::string& filename, const std::string& key) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return cv::Mat();

    cv::FileNode fileNode = fs[key];
    cv::Mat matResult;
    cv::read(fileNode, matResult, cv::Mat());
    fs.release();

    return matResult;
}

void TestMerge4DlpHeight() {
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl << "MERGE 4 DLP HEIGHT REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "----------------------------------------------";
    std::cout << std::endl;

    std::string strResultFolder = "./Data/TestMerge4DlpHeight/";

    PR_CALC_3D_HEIGHT_NEW_CMD stCalcHeightCmds[4];
    PR_CALC_3D_HEIGHT_RPY stCalcHeightRpys[4];

    stCalcHeightCmds[0].enProjectDir = static_cast<PR_DIRECTION>(2);
    stCalcHeightCmds[1].enProjectDir = static_cast<PR_DIRECTION>(1);
    stCalcHeightCmds[2].enProjectDir = static_cast<PR_DIRECTION>(3);
    stCalcHeightCmds[3].enProjectDir = static_cast<PR_DIRECTION>(0);
    for (int nDlp = 0; nDlp < 4; ++nDlp) {
        stCalcHeightRpys[nDlp].matHeight = readMatFromYml(strResultFolder + "Dlp_" + std::to_string(nDlp + 1) + "_Height.yml", "Height");
        stCalcHeightRpys[nDlp].matNanMask = cv::imread(strResultFolder + "Dlp_" + std::to_string(nDlp + 1) + "_NanMask.png", cv::IMREAD_GRAYSCALE);
    }

    VectorOfMat vecMatHeightMerges;
    VectorOfMat vecMatHeightNanMasks;

    PR_MERGE_3D_HEIGHT_CMD stMerge3DCmd;
    PR_MERGE_3D_HEIGHT_RPY stMerge3DRpy;

    for (int i = 0; i < 4; ++i) {
        stCalcHeightRpys[i].matHeight.setTo(NAN, stCalcHeightRpys[i].matNanMask);
        stMerge3DCmd.vecMatHeight.push_back(stCalcHeightRpys[i].matHeight);
        stMerge3DCmd.vecMatNanMask.push_back(stCalcHeightRpys[i].matNanMask);
        stMerge3DCmd.vecProjDir.push_back(stCalcHeightCmds[i].enProjectDir);
    }

    stMerge3DCmd.enMethod = PR_MERGE_3D_HT_METHOD::SELECT_NEAREST_INTERSECT;
    stMerge3DCmd.fHeightDiffThreshold = 0.1f;

    PR_Merge3DHeight(&stMerge3DCmd, &stMerge3DRpy);
    if (stMerge3DRpy.enStatus != VisionStatus::OK) {
        std::cout << "PR_Merge3DHeight failed, status " << ToInt32(stMerge3DRpy.enStatus) << " at line " << __LINE__ << std::endl;
        return;
    }

    std::cout << "Grid snoop phase: " << std::endl;
    cv::Mat matPhaseResultImg = _drawHeightGrid(stMerge3DRpy.matHeight, 10, 10);
    cv::imwrite(strResultFolder + "PR_Merge3DHeight_Result.png", matPhaseResultImg);
}

void PrintCalcFrameResult(const PR_CALC_FRAME_VALUE_CMD &stCmd, const PR_CALC_FRAME_VALUE_RPY &stRpy) {
    std::cout << "PR_CalcFrameValue status " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    if (VisionStatus::OK == stRpy.enStatus)
        std::cout << "PR_CalcFrameValue target " << stCmd.ptTargetFrameCenter << " result: " << stRpy.fResult << std::endl;
}

void TestCalcFrameValue_1() {
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
    PrintCalcFrameResult(stCmd, stRpy);

    stCmd.ptTargetFrameCenter = cv::Point(150, 180);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintCalcFrameResult(stCmd, stRpy);

    stCmd.ptTargetFrameCenter = cv::Point(500, 180);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintCalcFrameResult(stCmd, stRpy);
}

void TestCalcFrameValue_2() {
    PR_CALC_FRAME_VALUE_CMD stCmd;
    PR_CALC_FRAME_VALUE_RPY stRpy;

    int ROWS = 1, COLS = 5;
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
    std::cout << std::endl << "CALC FRAME VALUE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;

    stCmd.ptTargetFrameCenter = cv::Point(150, 180);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintCalcFrameResult(stCmd, stRpy);

    stCmd.ptTargetFrameCenter = cv::Point(500, 180);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintCalcFrameResult(stCmd, stRpy);
}

void TestCalcFrameValue_3() {
    PR_CALC_FRAME_VALUE_CMD stCmd;
    PR_CALC_FRAME_VALUE_RPY stRpy;

    int ROWS = 5, COLS = 1;
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
    std::cout << std::endl << "CALC FRAME VALUE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;

    stCmd.ptTargetFrameCenter = cv::Point(150, 180);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintCalcFrameResult(stCmd, stRpy);

    stCmd.ptTargetFrameCenter = cv::Point(500, 180);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintCalcFrameResult(stCmd, stRpy);
}

void TestCalcFrameValue_4() {
    PR_CALC_FRAME_VALUE_CMD stCmd;
    PR_CALC_FRAME_VALUE_RPY stRpy;

    auto vecPoints = VectorOfPoint2f{cv::Point2f(-130.124084f, 151.156494f), cv::Point2f(-104.631355f, 151.156494f), cv::Point2f(-79.1386414f, 151.156494f), cv::Point2f(-53.6459198f, 151.156494f)};
    stCmd.vecVecRefFrameCenters.push_back(vecPoints);

    vecPoints = VectorOfPoint2f{cv::Point2f(-130.124084f, 118.996170f), cv::Point2f(-104.631355f, 118.996170f), cv::Point2f(-79.1386414f, 118.996170f), cv::Point2f(-53.6459198f, 118.996170f)};
    stCmd.vecVecRefFrameCenters.push_back(vecPoints);

    vecPoints = VectorOfPoint2f{cv::Point2f(-130.124084f, 86.8358307f), cv::Point2f(-104.631355f, 86.8358307f), cv::Point2f(-79.1386414f, 86.8358307f), cv::Point2f(-53.6459198f, 86.8358307f)};
    stCmd.vecVecRefFrameCenters.push_back(vecPoints);

    vecPoints = VectorOfPoint2f{cv::Point2f(-130.124084f, 54.6754990f), cv::Point2f(-104.631355f, 54.6754990f), cv::Point2f(-79.1386414f, 54.6754990f), cv::Point2f(-53.6459198f, 54.6754990f)};
    stCmd.vecVecRefFrameCenters.push_back(vecPoints);

    VectorOfFloat vecValues{0.0468321592f, 0.0429658033f, 0.0480033569f, 0.0454535373f};
    stCmd.vecVecRefFrameValues.push_back(vecValues);

    vecValues = VectorOfFloat{0.0454526432f, 0.0450277366f, 0.0459417216f, 0.0449718386f};
    stCmd.vecVecRefFrameValues.push_back(vecValues);

    vecValues = VectorOfFloat{0.0467235260f, 0.0470155515f, 0.0483370610f, 0.0471665189f};
    stCmd.vecVecRefFrameValues.push_back(vecValues);

    vecValues = VectorOfFloat{0.0498712249f, 0.0471298359f, 0.0499374978f, 0.0486724712f};
    stCmd.vecVecRefFrameValues.push_back(vecValues);

    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl << "CALC FRAME VALUE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;
    stCmd.ptTargetFrameCenter = cv::Point2f(-53.f, 102.f);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintCalcFrameResult(stCmd, stRpy);

    stCmd.ptTargetFrameCenter = cv::Point2f(-135.f, 102.f);
    PR_CalcFrameValue(&stCmd, &stRpy);
    PrintCalcFrameResult(stCmd, stRpy);
}

static void CheckOriginalDataChanged(const cv::Mat &matOriginal, const cv::Mat &matAfter) {
    cv::Mat matDiff = matOriginal - matAfter;
    std::cout << "The difference between original data and after function run data: " << cv::sum(matDiff)[0] << std::endl;
}

void TestCalc3DHeightDiff_1() {
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl << "CALC 3D HEIGHT DIFF REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "-----------------------------------------------";
    std::cout << std::endl;

    String strWorkingFolder = "./data/TestCacl3DHeightDiff/";
    PR_CALC_3D_HEIGHT_DIFF_CMD stCmd;
    PR_CALC_3D_HEIGHT_DIFF_RPY stRpy;

    cv::FileStorage fs(strWorkingFolder + "Height.yml", cv::FileStorage::READ);
    cv::FileNode fileNode = fs["Height"];
    cv::read(fileNode, stCmd.matHeight, cv::Mat());
    fs.release();
    auto matOriginal = stCmd.matHeight.clone();

    stCmd.matMask = cv::imread(strWorkingFolder + "mask.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectROI = cv::Rect(132, 173, 98, 81);
    stCmd.vecRectBases.emplace_back(0, 0, 169, 128);

    PR_Calc3DHeightDiff(&stCmd, &stRpy);
    std::cout << "PR_Calc3DHeightDiff status " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    if (VisionStatus::OK == stRpy.enStatus) {
        std::cout << "Height Diff Result " << stRpy.fHeightDiff << std::endl;
    }
    CheckOriginalDataChanged(matOriginal, stCmd.matHeight);
}

static void PrintInsp3DSolderResult(const PR_INSP_3D_SOLDER_RPY &stRpy) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "PR_Insp3DSolder status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK == stRpy.enStatus) {
        for (const auto &result : stRpy.vecResults) {
            std::cout << "Component height " << result.fComponentHeight << " Solder height " << result.fSolderHeight
                << ", area " << result.fSolderArea << ", ratio " << result.fSolderRatio << std::endl;
        }
    }
}

void TestInsp3DSolder_1() {
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl << "INSP 3D SOLDER REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;

    const std::string strParentFolder = "./data/TestInsp3DSolder_1/";
    PR_INSP_3D_SOLDER_CMD stCmd;
    PR_INSP_3D_SOLDER_RPY stRpy;

    stCmd.matHeight = readMatFromCsvFile(strParentFolder + "H3.csv");
    auto matOriginal = stCmd.matHeight.clone();

    stCmd.matColorImg = cv::imread(strParentFolder + "image3.bmp", cv::IMREAD_COLOR);

    stCmd.rectDeviceROI = cv::Rect(1175, 575, 334, 179);
    stCmd.vecRectCheckROIs.push_back(cv::Rect(1197, 605, 72, 122));
    stCmd.vecRectCheckROIs.push_back(cv::Rect(1400, 605, 76, 122));

    stCmd.nBaseColorDiff = 20;
    stCmd.nBaseGrayDiff = 20;
    
    cv::Rect rectBase(400, 200, 200, 200);
    cv::Mat matBaseRef(stCmd.matColorImg, rectBase);
    stCmd.scalarBaseColor = cv::mean(matBaseRef);
    stCmd.fHeightVariationCoverage = 0.75f;

    PR_Insp3DSolder(&stCmd, &stRpy);
    PrintInsp3DSolderResult(stRpy);

    if (!stRpy.matResultImg.empty())
        cv::imwrite(strParentFolder + "result.png", stRpy.matResultImg);
    CheckOriginalDataChanged(matOriginal, stCmd.matHeight);
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

    auto matOriginal = stCmd.matHeight.clone();

    stCmd.matColorImg = cv::imread(strParentFolder + "image.png", cv::IMREAD_COLOR);

    stCmd.rectDeviceROI = cv::Rect(0, 0, 143, 73);
    stCmd.vecRectCheckROIs.push_back(cv::Rect(9, 15, 45, 47));
    stCmd.vecRectCheckROIs.push_back(cv::Rect(94, 14, 35, 47));

    stCmd.nBaseColorDiff = 20;
    stCmd.nBaseGrayDiff = 20;    
    stCmd.scalarBaseColor = cv::Scalar(37, 22, 5);

    stCmd.fHeightVariationCoverage = 0.75f;

    PR_Insp3DSolder(&stCmd, &stRpy);
    PrintInsp3DSolderResult(stRpy);
    if (!stRpy.matResultImg.empty())
        cv::imwrite(strParentFolder + "result.png", stRpy.matResultImg);
    CheckOriginalDataChanged(matOriginal, stCmd.matHeight);
}

void TestInsp3DSolder_3() {
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl << "INSP 3D SOLDER REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;

    const std::string strParentFolder = "./data/TestInsp3DSolder_3/";
    PR_INSP_3D_SOLDER_CMD stCmd;
    PR_INSP_3D_SOLDER_RPY stRpy;

    cv::FileStorage fs(strParentFolder + "Height.yml", cv::FileStorage::READ);
    cv::FileNode fileNode = fs["Height"];
    cv::read(fileNode, stCmd.matHeight, cv::Mat());
    fs.release();

    auto matOriginal = stCmd.matHeight.clone();

    stCmd.matColorImg = cv::imread(strParentFolder + "image.png", cv::IMREAD_COLOR);

    stCmd.rectDeviceROI = cv::Rect(0, 0, 191, 350);
    stCmd.vecRectCheckROIs.push_back(cv::Rect(26, 29, 138, 122));
    stCmd.vecRectCheckROIs.push_back(cv::Rect(26, 223, 138, 122));

    stCmd.nBaseColorDiff = 20;
    stCmd.nBaseGrayDiff = 20;    
    stCmd.scalarBaseColor = cv::Scalar(42, 25, 7);

    stCmd.fHeightVariationCoverage = 0.75f;

    PR_Insp3DSolder(&stCmd, &stRpy);
    PrintInsp3DSolderResult(stRpy);
    if (!stRpy.matResultImg.empty())
        cv::imwrite(strParentFolder + "result.png", stRpy.matResultImg);
    CheckOriginalDataChanged(matOriginal, stCmd.matHeight);
}

void TestInsp3DSolder_4() {
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl << "INSP 3D SOLDER REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "--------------------------------------------";
    std::cout << std::endl;

    const std::string strParentFolder = "./data/TestInsp3DSolder_3/";
    PR_INSP_3D_SOLDER_CMD stCmd;
    PR_INSP_3D_SOLDER_RPY stRpy;

    cv::FileStorage fs(strParentFolder + "Height.yml", cv::FileStorage::READ);
    cv::FileNode fileNode = fs["Height"];
    cv::read(fileNode, stCmd.matHeight, cv::Mat());
    fs.release();

    auto matOriginal = stCmd.matHeight.clone();

    stCmd.matColorImg = cv::imread(strParentFolder + "image.png", cv::IMREAD_COLOR);

    stCmd.rectDeviceROI = cv::Rect(0, 0, 191, 350);
    stCmd.vecRectCheckROIs.push_back(cv::Rect(26, 223, 138, 122));
    stCmd.vecRectCheckROIs.push_back(cv::Rect(26, 29, 138, 122));

    stCmd.nBaseColorDiff = 20;
    stCmd.nBaseGrayDiff = 20;
    stCmd.scalarBaseColor = cv::Scalar(42, 25, 7);

    stCmd.fHeightVariationCoverage = 0.75f;

    PR_Insp3DSolder(&stCmd, &stRpy);
    PrintInsp3DSolderResult(stRpy);
    if (!stRpy.matResultImg.empty())
        cv::imwrite(strParentFolder + "result_swap_roi.png", stRpy.matResultImg);
    CheckOriginalDataChanged(matOriginal, stCmd.matHeight);
}

void Test3D() {
    TestCalib3dBase_1();
    TestCalib3DHeight_1();
    TestCalib3DHeight_2();

    TestCalib3dBase_2();
    TestCalib3DHeight_3();
    TestCalib3DHeight_4();

    TestCalib3dBase_3();

    TestMotor3DCalib_1();

    TestMotor3DCalibNew_1();

    TestMerge4DlpHeight();

    TestCalcFrameValue_1();
    TestCalcFrameValue_2();
    TestCalcFrameValue_3();
    TestCalcFrameValue_4();

    TestCalc3DHeightDiff_1();

    TestInsp3DSolder_1();
    TestInsp3DSolder_2();
    TestInsp3DSolder_3();
    TestInsp3DSolder_4();
}

}
}
