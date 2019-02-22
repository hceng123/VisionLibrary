#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>

#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include "../RegressionTest/UtilityFunc.h"
#include "../VisionLibrary/CalcUtils.hpp"

using namespace AOI::Vision;
using namespace boost::filesystem;

cv::Mat matRestoreMap12_1, matRestoreMap12_2, matRestoreMap34_1, matRestoreMap34_2;
const static std::string WORKING_FOLDER = "./data/TestCombineImage_2/";
const bool LOAD_COLOR_IMAGE = false;
const float CHANGE_DISTORTION_MATRIX_Y = 250.f;
const int IMAGE_WIDTH = 2040, IMEGE_HEIGHT = 2048;

bool loadCalibCameraResult_12() {
    cv::FileStorage fs(WORKING_FOLDER + "CameraCalibrate12.yml", cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    cv::FileNode fileNode;
    fileNode = fs["RestoreImageMap1"];
    cv::read(fileNode, matRestoreMap12_1, cv::Mat());

    fileNode = fs["RestoreImageMap2"];
    cv::read(fileNode, matRestoreMap12_2, cv::Mat());

    fs.release();
    return true;
}

bool loadCalibCameraResult_34() {
    cv::FileStorage fs(WORKING_FOLDER + "CameraCalibrate34.yml", cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    cv::FileNode fileNode;
    fileNode = fs["RestoreImageMap1"];
    cv::read(fileNode, matRestoreMap34_1, cv::Mat());

    fileNode = fs["RestoreImageMap2"];
    cv::read(fileNode, matRestoreMap34_2, cv::Mat());

    fs.release();
    return true;
}

cv::Mat restoreImage_12(const cv::Mat& matInput) {
    PR_RESTORE_IMG_CMD stRetoreCmd;
    PR_RESTORE_IMG_RPY stRetoreRpy;
    stRetoreCmd.matInputImg = matInput;
    stRetoreCmd.vecMatRestoreMap.push_back(matRestoreMap12_1);
    stRetoreCmd.vecMatRestoreMap.push_back(matRestoreMap12_2);
    PR_RestoreImage(&stRetoreCmd, &stRetoreRpy);
    if (VisionStatus::OK != stRetoreRpy.enStatus) {
        return cv::Mat();
    }
    return stRetoreRpy.matResultImg;
}

cv::Mat restoreImage_34(const cv::Mat& matInput) {
    PR_RESTORE_IMG_CMD stRetoreCmd;
    PR_RESTORE_IMG_RPY stRetoreRpy;
    stRetoreCmd.matInputImg = matInput;
    stRetoreCmd.vecMatRestoreMap.push_back(matRestoreMap34_1);
    stRetoreCmd.vecMatRestoreMap.push_back(matRestoreMap34_2);
    PR_RestoreImage(&stRetoreCmd, &stRetoreRpy);
    if (VisionStatus::OK != stRetoreRpy.enStatus) {
        return cv::Mat();
    }
    return stRetoreRpy.matResultImg;
}

cv::Mat restoreImageWithMap(const cv::Mat& matInput, const cv::Mat& matMap) {
    PR_RESTORE_IMG_CMD stRetoreCmd;
    PR_RESTORE_IMG_RPY stRetoreRpy;
    stRetoreCmd.matInputImg = matInput;
    stRetoreCmd.vecMatRestoreMap.push_back(matMap);
    stRetoreCmd.vecMatRestoreMap.push_back(cv::Mat());
    PR_RestoreImage(&stRetoreCmd, &stRetoreRpy);
    if (VisionStatus::OK != stRetoreRpy.enStatus) {
        return cv::Mat();
    }
    return stRetoreRpy.matResultImg;
}

static bool readImageFromFolder(VectorOfMat& vecFrameImages, const VectorOfVectorOfFloat& vecVecFrameCtrs) {
    VectorOfMat vectorOfImage;
    path p(WORKING_FOLDER);

    if (!is_directory(p))
        return false;

    std::vector<directory_entry> v; // To save the file names in a vector.
    copy(directory_iterator(p), directory_iterator(), back_inserter(v));

    int index = 0;
    for (std::vector<directory_entry>::const_iterator it = v.begin(); it != v.end(); ++it)
    {
        if (!is_directory(it->path()))
            continue;

        auto strSubFolder = (*it).path().string();

        if (LOAD_COLOR_IMAGE) {
            std::cout << "Reading image from " << strSubFolder << std::endl;
            cv::Mat matR = cv::imread(strSubFolder + "/51.bmp", cv::IMREAD_GRAYSCALE);
            cv::Mat matG = cv::imread(strSubFolder + "/52.bmp", cv::IMREAD_GRAYSCALE);
            cv::Mat matB = cv::imread(strSubFolder + "/53.bmp", cv::IMREAD_GRAYSCALE);

            if (matR.empty() || matG.empty() || matB.empty()) {
                std::cout << "Failed to read image from " << strSubFolder << std::endl;
                return false;
            }

            VectorOfMat vecChannels{ matB, matG, matR };
            cv::Mat matPseudocolorImage;
            cv::merge(vecChannels, matPseudocolorImage);

            vecFrameImages.push_back(matPseudocolorImage);
        }else {
            cv::Mat matGray = cv::imread(strSubFolder + "/54.bmp", cv::IMREAD_GRAYSCALE);

            if (matGray.empty()) {
                std::cout << "Failed to read image from " << strSubFolder << std::endl;
                return false;
            }

            cv::imwrite(strSubFolder + "/54_Restore.bmp", matGray);
            vecFrameImages.push_back(matGray);
        }

        ++ index;
    }
    return true;
}

static bool readImageFromFolderWithRestore(VectorOfMat& vecFrameImages, const VectorOfVectorOfFloat& vecVecFrameCtrs) {
    VectorOfMat vectorOfImage;
    path p(WORKING_FOLDER);

    if (!is_directory(p))
        return false;

    std::vector<directory_entry> v; // To save the file names in a vector.
    copy(directory_iterator(p), directory_iterator(), back_inserter(v));

    int index = 0;
    for (std::vector<directory_entry>::const_iterator it = v.begin(); it != v.end(); ++it)
    {
        if (!is_directory(it->path()))
            continue;

        auto strSubFolder = (*it).path().string();

        if (LOAD_COLOR_IMAGE) {
            std::cout << "Reading image from " << strSubFolder << std::endl;
            cv::Mat matR = cv::imread(strSubFolder + "/51.bmp", cv::IMREAD_GRAYSCALE);
            cv::Mat matG = cv::imread(strSubFolder + "/52.bmp", cv::IMREAD_GRAYSCALE);
            cv::Mat matB = cv::imread(strSubFolder + "/53.bmp", cv::IMREAD_GRAYSCALE);

            if (matR.empty() || matG.empty() || matB.empty()) {
                std::cout << "Failed to read image from " << strSubFolder << std::endl;
                return false;
            }

            if (vecVecFrameCtrs[index][1] > CHANGE_DISTORTION_MATRIX_Y) {
                matG = restoreImage_12(matG);
                matR = restoreImage_12(matR);
                matB = restoreImage_12(matB);
            }
            else {
                matG = restoreImage_34(matG);
                matR = restoreImage_34(matR);
                matB = restoreImage_34(matB);
            }

            if (matR.empty() || matG.empty() || matB.empty()) {
                std::cout << "Failed to restore image " << strSubFolder << std::endl;
                return false;
            }

            VectorOfMat vecChannels{ matB, matG, matR };
            cv::Mat matPseudocolorImage;
            cv::merge(vecChannels, matPseudocolorImage);

            vecFrameImages.push_back(matPseudocolorImage);
        }else {
            cv::Mat matGray = cv::imread(strSubFolder + "/54.bmp", cv::IMREAD_GRAYSCALE);

            if (matGray.empty()) {
                std::cout << "Failed to read image from " << strSubFolder << std::endl;
                return false;
            }

            if (vecVecFrameCtrs[index][1] > CHANGE_DISTORTION_MATRIX_Y) {
                matGray = restoreImage_12(matGray);
            }
            else {
                matGray = restoreImage_34(matGray);
            }

            if (matGray.empty()) {
                std::cout << "Failed to restore image " << strSubFolder << std::endl;
                return false;
            }

            cv::imwrite(strSubFolder + "/54_Restore.bmp", matGray);

            vecFrameImages.push_back(matGray);
        }

        ++ index;
    }
    return true;
}

void TestCombineMachineImage() {
    //float fResolutionX = 15.8655f;
    //float fResolutionY = 15.90f;

    //float fResolutionX = 15.883068f;
    //float fResolutionY = 15.884272f;

    float fResolutionX = 15.8729f;
    float fResolutionY = 15.8736f;

    const int ROWS = 6;
    const int COLS = 8;
    const int TOTAL_FRAME = ROWS * COLS;

    auto vecOfVecFloat = readDataFromFile(WORKING_FOLDER + "ScanImagePositions.csv");
    if (vecOfVecFloat.size() != TOTAL_FRAME) {
        std::cout << "Number of scan image position " << vecOfVecFloat.size() << " not match with total frame " << TOTAL_FRAME << std::endl;
        return;
    }

    if (!loadCalibCameraResult_12()) {
        std::cout << "Failed to load camera calibration result 12" << std::endl;
        return;
    }

    if (!loadCalibCameraResult_34()) {
        std::cout << "Failed to load camera calibration result 34" << std::endl;
        return;
    }

    PR_COMBINE_IMG_NEW_CMD stCmd;
    PR_COMBINE_IMG_NEW_RPY stRpy;
    if (!readImageFromFolderWithRestore(stCmd.vecInputImages, vecOfVecFloat))
        return;

    cv::Point2f ptTopLeftFrameTableCtr(vecOfVecFloat[0][0], vecOfVecFloat[0][1]);
    cv::Point ptTopLeftFrameCtr(cv::Point(stCmd.vecInputImages[0].cols / 2, stCmd.vecInputImages[0].rows / 2));
    int index = 0;
    for (int row = 0; row < ROWS; ++row) {
        VectorOfPoint vecFrameCtr;
        for (int col = 0; col < COLS; ++col) {
            cv::Point2f frameTableCtr(vecOfVecFloat[index][0], vecOfVecFloat[index][1]);
            cv::Point frameCtr;
            frameCtr.x = ptTopLeftFrameCtr.x + (frameTableCtr.x - ptTopLeftFrameTableCtr.x) * 1000.f / fResolutionX;
            frameCtr.y = ptTopLeftFrameCtr.y - (frameTableCtr.y - ptTopLeftFrameTableCtr.y) * 1000.f / fResolutionX; // Table direction is reversed with image direction
            std::cout << "Row " << row << " Col " << col << " frame center " << frameCtr << std::endl;
            vecFrameCtr.push_back(frameCtr);
            ++ index;
        }
        stCmd.vecVecFrameCtr.push_back(vecFrameCtr);
    }

    stCmd.bDrawFrame = true;
    stCmd.nCutBorderPixel = 5;
    PR_CombineImgNew(&stCmd, &stRpy);
    std::cout << "PR_CombineImgNew status " << ToInt32(stRpy.enStatus) << std::endl;
    if (!stRpy.matResultImage.empty())
        cv::imwrite(WORKING_FOLDER + "CombineImageNewResult.png", stRpy.matResultImage);
}

void FindFramePositionFromBigImage() {
    //cv::Point ptBigFramePos(9061, 9335);
    cv::Point ptBigFramePos(9063, 10507);
    float fResolutionX = 15.8729f;
    float fResolutionY = 15.8736f;

    const int ROWS = 8;
    const int COLS = 6;
    const int TOTAL_FRAME = ROWS * COLS;

    auto vecOfVecFloat = readDataFromFile(WORKING_FOLDER + "ScanImagePositions.csv");
    if (vecOfVecFloat.size() != TOTAL_FRAME) {
        std::cout << "Number of scan image position " << vecOfVecFloat.size() << " not match with total frame " << TOTAL_FRAME << std::endl;
        return;
    }

    cv::Point2f ptTopLeftFrameTableCtr(vecOfVecFloat[0][0], vecOfVecFloat[0][1]);
    cv::Point ptTopLeftFrameCtr(IMAGE_WIDTH / 2, IMEGE_HEIGHT / 2);

    int index = 0;
    for (int row = 0; row < ROWS; ++row) {
        VectorOfPoint vecFrameCtr;
        for (int col = 0; col < COLS; ++col) {
            cv::Point2f frameTableCtr(vecOfVecFloat[index][0], vecOfVecFloat[index][1]);
            cv::Point frameCtr;
            frameCtr.x = ptTopLeftFrameCtr.x + (frameTableCtr.x - ptTopLeftFrameTableCtr.x) * 1000.f / fResolutionX;
            frameCtr.y = ptTopLeftFrameCtr.y - (frameTableCtr.y - ptTopLeftFrameTableCtr.y) * 1000.f / fResolutionX; // Table direction is reversed with image direction
            
            if (abs(ptBigFramePos.x - frameCtr.x) < IMAGE_WIDTH / 2 && abs(ptBigFramePos.y - frameCtr.y) < IMEGE_HEIGHT / 2) {
                cv::Point ptInFrame(ptBigFramePos.x - frameCtr.x + IMAGE_WIDTH / 2, ptBigFramePos.y - frameCtr.y + IMEGE_HEIGHT / 2);
                std::cout << "Frame row " << row << " col " << col << " index " << index << " contains this point, position in frame " << ptInFrame << std::endl;
                
            }
            ++index;
        }
    }
}

cv::Mat readMapFromFile(int idx) {
    std::string strFileName = WORKING_FOLDER + "Frame_" + std::to_string(idx) + ".txt";
    auto vecOfVecFloat = readDataFromFile(strFileName);
    if (vecOfVecFloat.empty())
        std::cout << "Failed to read map data from " << strFileName << std::endl;
    else
        std::cout << "Success to read map data from " << strFileName << std::endl;
    cv::Mat matResult(2040, 2048, CV_16SC2);
    int index = 0;
    for (const auto& vecPoint : vecOfVecFloat) {
        matResult.at<cv::Vec2s>(index) = cv::Vec2s(std::floor(vecPoint[0]) - 1, std::floor(vecPoint[1]) - 1);
        ++ index;
    }
    // The row and col sequence of Matlab and OpenCV is reversed, need to reverse back.
    cv::transpose(matResult, matResult);
    return matResult;
}

void DynamicCalculateDistortion() {
    int mm = 7, nn = 9;
    const int ROWS = 6;
    const int COLS = 8;
    const int TOTAL_FRAME = ROWS * COLS;
    const int CTF = 6;
    const int TOTAL_DISTORTION_MAP = 11;
    float fResolutionX = 15.88;
    float fResolutionY = 15.88;

    cv::Mat matXOffsetParam = readMatFromCsvFile(WORKING_FOLDER + "XOffsetParams.csv");
    cv::Mat matYOffsetParam = readMatFromCsvFile(WORKING_FOLDER + "YOffsetParams.csv");

    //Load map from data
    matXOffsetParam = matXOffsetParam.reshape(1, mm);
    cv::transpose(matXOffsetParam, matXOffsetParam);

    matYOffsetParam = matYOffsetParam.reshape(1, mm);
    cv::transpose(matYOffsetParam, matYOffsetParam);

#ifdef _DEBUG
    auto vecVecXOffsetParam = matToVector<float>(matXOffsetParam);
    auto vecVecYOffsetParam = matToVector<float>(matYOffsetParam);
#endif

    auto vecOfVecFloat = readDataFromFile(WORKING_FOLDER + "ScanImagePositions.csv");
    if (vecOfVecFloat.size() != TOTAL_FRAME) {
        std::cout << "Number of scan image position " << vecOfVecFloat.size() << " not match with total frame " << TOTAL_FRAME << std::endl;
        return;
    }

    VectorOfFloat vecX, vecY;
    for (const auto& vecFloat : vecOfVecFloat) {
        vecX.push_back(vecFloat[0]);
        vecY.push_back(vecFloat[1]);
    }
    cv::Mat matX(vecX), matY(vecY);

    cv::Mat PPz3 = (mm - 1) * CalcUtils::diff(matYOffsetParam, 1, 2);
    cv::transpose(PPz3, PPz3);
    PPz3 = PPz3.reshape(1, nn*(mm - 1));
    VectorOfFloat vecParamMinMaxXY{-396.152f, 20.547f, 32.017f, 351.187f};
    auto matBezierSurface = CalcUtils::generateBezier(matX, matY, vecParamMinMaxXY, mm - 1, nn);

    cv::Mat dydu = matBezierSurface * PPz3;

#ifdef _DEBUG
    auto vecVecBezierSurface = matToVector<float>(matBezierSurface);
    auto vecVecDydu = matToVector<float>(dydu);
#endif

    cv::Mat dydx = dydu / (0.5f / 2048.f) / (vecParamMinMaxXY[1] - vecParamMinMaxXY[0]);

    cv::Mat dtheta = dydx.reshape(1, ROWS);

#ifdef _DEBUG
    auto vecVecBeforeDTheta = matToVector<float>(dtheta);
#endif

    //dtheta = dtheta * (2048.f / 2.f);
    CalcUtils::round<float>(dtheta);

#ifdef _DEBUG
    auto vecVecDTheta = matToVector<float>(dtheta);
#endif

    cv::Mat arrMap[TOTAL_DISTORTION_MAP];

    PR_COMBINE_IMG_NEW_CMD stCmd;
    PR_COMBINE_IMG_NEW_RPY stRpy;
    if (!readImageFromFolder(stCmd.vecInputImages, vecOfVecFloat))
        return;

    std::cout << "Success to read image from folder" << std::endl;

    cv::Point2f ptTopLeftFrameTableCtr(vecOfVecFloat[0][0], vecOfVecFloat[0][1]);
    cv::Point ptTopLeftFrameCtr(IMAGE_WIDTH / 2, IMEGE_HEIGHT / 2);

    int index = 0;
    for (int row = 0; row < ROWS; ++row) {
        VectorOfPoint vecFrameCtr;
        for (int col = 0; col < COLS; ++col) {
            int idx = -dtheta.at<float>(row, col) + CTF;
            if (arrMap[idx].empty()) {
                arrMap[idx] = readMapFromFile(idx);
                std::cout << "Sucess to read map for idx " << idx << std::endl;
            }

            if (arrMap[idx].empty()) {
                std::cout << "Distortion map " << idx << " is empty" << std::endl;
                return;
            }

            stCmd.vecInputImages[index] = restoreImageWithMap(stCmd.vecInputImages[index], arrMap[idx]);

            //restoreImageWithMap()
            cv::Point2f frameTableCtr(vecOfVecFloat[index][0], vecOfVecFloat[index][1]);
            cv::Point frameCtr;
            frameCtr.x = ptTopLeftFrameCtr.x + (frameTableCtr.x - ptTopLeftFrameTableCtr.x) * 1000.f / fResolutionX;
            frameCtr.y = ptTopLeftFrameCtr.y - (frameTableCtr.y - ptTopLeftFrameTableCtr.y) * 1000.f / fResolutionX; // Table direction is reversed with image direction
            std::cout << "Row " << row << " Col " << col << " frame center " << frameCtr << std::endl;
            vecFrameCtr.push_back(frameCtr);
            ++index;
        }
        stCmd.vecVecFrameCtr.push_back(vecFrameCtr);
    }

    stCmd.bDrawFrame = true;
    stCmd.nCutBorderPixel = 5;
    PR_CombineImgNew(&stCmd, &stRpy);
    std::cout << "PR_CombineImgNew status " << ToInt32(stRpy.enStatus) << std::endl;
    if (!stRpy.matResultImage.empty())
        cv::imwrite(WORKING_FOLDER + "CombineImageNewResult.png", stRpy.matResultImage);
}

void TestCalcRestoreIdx() {
    const int ROWS = 6;
    const int COLS = 8;
    const int TOTAL_FRAME = ROWS * COLS;

    PR_CALC_RESTORE_IDX_CMD stCmd;
    PR_CALC_RESTORE_IDX_RPY stRpy;

    stCmd.matXOffsetParam = readMatFromCsvFile(WORKING_FOLDER + "XOffsetParams.csv");
    stCmd.matYOffsetParam = readMatFromCsvFile(WORKING_FOLDER + "YOffsetParams.csv");
    stCmd.fMinX = -396.152f;
    stCmd.fMaxX = 20.547f;
    stCmd.fMinY = 32.017f;
    stCmd.fMaxY = 351.187f;
    stCmd.fRestoreMatrixAngleInterval = 0.5f / 2048.f;
    stCmd.nCenterIdx = 6;
    stCmd.nBezierRankX = 7;
    stCmd.nBezierRankY = 9;

    auto vecOfVecFloat = readDataFromFile(WORKING_FOLDER + "ScanImagePositions.csv");
    if (vecOfVecFloat.size() != TOTAL_FRAME) {
        std::cout << "Number of scan image position " << vecOfVecFloat.size() << " not match with total frame " << TOTAL_FRAME << std::endl;
        return;
    }

    int index = 0;
    for (int row = 0; row < ROWS; ++row) {
        VectorOfPoint2f vecFrameCtr;
        for (int col = 0; col < COLS; ++col) {
            cv::Point2f ptCenter(vecOfVecFloat[index][0], vecOfVecFloat[index][1]);
            vecFrameCtr.push_back(ptCenter);
            ++index;
        }
        stCmd.vecVecFrameCtr.push_back(vecFrameCtr);
    }

    PR_CalcRestoreIdx(&stCmd, &stRpy);

    std::cout << "PR_CalcRestoreIdx status " << ToInt32(stRpy.enStatus) << std::endl;
    if (!stRpy.vecVecRestoreIndex.empty())
        printfVectorOfVector<int>(stRpy.vecVecRestoreIndex);
}
