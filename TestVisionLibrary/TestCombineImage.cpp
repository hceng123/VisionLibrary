#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

using namespace AOI::Vision;

cv::Mat generateChessBoardImage(int totalRow, int totalCol, int blockSize) {
    cv::Mat matResult = cv::Mat::zeros(totalRow * blockSize, totalCol * blockSize, CV_8UC1);
    for (int row = 0; row < totalRow; ++ row) {
        for (int col = 0; col < totalCol; ++ col) {
            int index = row * totalCol + col;
            if (index % 2 == 0) {
                cv::Mat matROI(matResult, cv::Rect(col * blockSize, row * blockSize, blockSize, blockSize));
                matROI.setTo(255);
            }
        }
    }
    return matResult;
}

void TestCombineImageNew_1() {
    cv::Mat matChessBoardImage = generateChessBoardImage(11, 11, 20);
    PR_COMBINE_IMG_NEW_CMD stCmd;
    PR_COMBINE_IMG_NEW_RPY stRpy;
    stCmd.vecVecFrameCtr = VectorOfVectorOfPoint{VectorOfPoint{cv::Point(110, 110), cv::Point(310, 110)},
        VectorOfPoint{cv::Point(110, 310), cv::Point(310, 310)}};
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.vecInputImages.push_back(matChessBoardImage);
    //stCmd.bDrawFrame = true;

    PR_CombineImgNew(&stCmd, &stRpy);
    std::cout << "PR_CombineImgNew status " << ToInt32(stRpy.enStatus) << std::endl;
    if (!stRpy.matResultImage.empty())
        cv::imwrite("./data/CombineImageNewResult_1.png", stRpy.matResultImage);

    if (VisionStatus::OK != stRpy.enStatus)
        return;

    cv::Mat matResultChessBoardImage = generateChessBoardImage(21, 21, 20);
    cv::Mat matCompare = stRpy.matResultImage - matResultChessBoardImage;
    int nonZeroCount = cv::countNonZero(matCompare);
    std::cout << "Compare with difference points cout " << nonZeroCount << std::endl;
}

void TestCombineImageNew_2() {
    cv::Mat matChessBoardImage = generateChessBoardImage(11, 11, 20);
    PR_COMBINE_IMG_NEW_CMD stCmd;
    PR_COMBINE_IMG_NEW_RPY stRpy;
    stCmd.vecVecFrameCtr = VectorOfVectorOfPoint{ VectorOfPoint{ cv::Point(110, 110), cv::Point(312, 109)},
        VectorOfPoint{ cv::Point(111, 312), cv::Point(310, 310) } };
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.bDrawFrame = true;

    PR_CombineImgNew(&stCmd, &stRpy);
    std::cout << "PR_CombineImgNew status " << ToInt32(stRpy.enStatus) << std::endl;
    if (!stRpy.matResultImage.empty())
        cv::imwrite("./data/CombineImageNewResult_2.png", stRpy.matResultImage);

    if (VisionStatus::OK != stRpy.enStatus)
        return;

    cv::Mat matResultChessBoardImage = generateChessBoardImage(21, 21, 20);
    cv::Mat matCompare = stRpy.matResultImage - matResultChessBoardImage;
    int nonZeroCount = cv::countNonZero(matCompare);
    std::cout << "Compare with difference points cout " << nonZeroCount << std::endl;
}
