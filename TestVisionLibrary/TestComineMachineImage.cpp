#include <iostream>
#include <iomanip>

#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include "../RegressionTest/UtilityFunc.h"
#include <boost/filesystem.hpp>

using namespace AOI::Vision;
using namespace boost::filesystem;

cv::Mat matRestoreMap12_1, matRestoreMap12_2, matRestoreMap34_1, matRestoreMap34_2;
const static std::string WORKING_FOLDER = "./data/TestCombineImage_2/";
const bool LOAD_COLOR_IMAGE = false;
const float CHANGE_DISTORTION_MATRIX_Y = 250.f;

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

bool readImageFromFolder(VectorOfMat& vecFrameImages, const VectorOfVectorOfFloat& vecVecFrameCtrs) {
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

            cv:imwrite(strSubFolder + "/54_Restore.bmp", matGray);

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

    const int ROWS = 8;
    const int COLS = 6;
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
    if (!readImageFromFolder(stCmd.vecInputImages, vecOfVecFloat))
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
    PR_CombineImgNew(&stCmd, &stRpy);
    std::cout << "PR_CombineImgNew status " << ToInt32(stRpy.enStatus) << std::endl;
    if (!stRpy.matResultImage.empty())
        cv::imwrite(WORKING_FOLDER + "CombineImageNewResult.png", stRpy.matResultImage);
}

void FindFramePositionFromBigImage() {
    cv::Point ptBigFramePos(9061, 9335);
    float fResolutionX = 15.8729f;
    float fResolutionY = 15.8736f;

    const int ROWS = 8;
    const int COLS = 6;
    const int TOTAL_FRAME = ROWS * COLS;
    const int IMAGE_WIDTH = 2040, IMEGE_HEIGHT = 2048;

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
                std::cout << "Frame row " << row << " col " << col << " at index " << index << " contains this point" << std::endl;
            }
            ++index;
        }
    }
}
