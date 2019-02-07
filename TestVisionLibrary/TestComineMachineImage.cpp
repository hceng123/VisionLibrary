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
const static std::string WORKING_FOLDER = "./data/TestCombineImage/";

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
        std::cout << "Reading image from " << strSubFolder << std::endl;
        cv::Mat matR =  cv::imread(strSubFolder + "/51.bmp", cv::IMREAD_GRAYSCALE);
        cv::Mat matG =  cv::imread(strSubFolder + "/52.bmp", cv::IMREAD_GRAYSCALE);
        cv::Mat matB =  cv::imread(strSubFolder + "/53.bmp", cv::IMREAD_GRAYSCALE);

        if (matR.empty() || matG.empty() || matB.empty()) {
            std::cout << "Failed to read image from " << strSubFolder << std::endl;
            return false;
        }

        if (vecVecFrameCtrs[index][1] > 190.f) {
            matG = restoreImage_12(matG);
            matR = restoreImage_12(matR);
            matB = restoreImage_12(matB);
        }else {
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
        ++ index;
    }
    return true;
}

void TestCombineMachineImage() {
    //float fResolutionX = 15.8655f;
    //float fResolutionY = 15.9024f;
    float fResolutionX = 15.883068f;
    float fResolutionY = 15.884272f;

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
    cv::Point2f ptTopLeftFrameTableCtr;
    if (!readImageFromFolder(stCmd.vecInputImages, vecOfVecFloat))
        return;

    cv::Point ptTopLeftFrameCtr(cv::Point(stCmd.vecInputImages[0].cols / 2, stCmd.vecInputImages[0].rows / 2));
    int index = 0;
    for (int row = 0; row < ROWS; ++row) {
        VectorOfPoint vecFrameCtr;
        for (int col = 0; col < COLS; ++col) {
            if (row == 0 && col == 0) {
                ptTopLeftFrameTableCtr.x = vecOfVecFloat[index][2];
                ptTopLeftFrameTableCtr.y = vecOfVecFloat[index][3];
            }

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
