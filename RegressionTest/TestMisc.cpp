#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

void TestCalcCameraMTF() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "CALC CAMERA MTF REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_CALC_CAMERA_MTF_CMD stCmd;
    PR_CALC_CAMERA_MTF_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/MTFTarget.png", cv::IMREAD_GRAYSCALE );
    stCmd.rectVBigPatternROI = cv::Rect ( 508, 744, 206, 210 );
    stCmd.rectHBigPatternROI = cv::Rect ( 508, 954, 206, 210 );
    stCmd.rectVSmallPatternROI = cv::Rect ( 989, 983, 27, 27 );
    stCmd.rectHSmallPatternROI = cv::Rect ( 989, 1010, 27, 27 );
    PR_CalcCameraMTF ( &stCmd, &stRpy );
    std::cout << "PR_CalcCameraMTF status " << ToInt32( stRpy.enStatus ) << std::endl;
    std::cout << std::fixed << std::setprecision ( 2 );
    std::cout << "Big pattern vertical absolute mtf " << stRpy.fBigPatternAbsMtfV << std::endl;
    std::cout << "Big pattern horizontal absolute mtf " << stRpy.fBigPatternAbsMtfH << std::endl;
    std::cout << "Small pattern vertical absolute mtf " << stRpy.fSmallPatternAbsMtfV << std::endl;
    std::cout << "Small pattern horizontal absolute mtf " << stRpy.fSmallPatternAbsMtfH << std::endl;
    std::cout << "Small pattern vertical relative mtf " << stRpy.fSmallPatternRelMtfV << std::endl;
    std::cout << "Small pattern horizontal relative mtf " << stRpy.fSmallPatternRelMtfH << std::endl;
}

static cv::Mat generateChessBoardImage(int totalRow, int totalCol, int blockSize) {
    cv::Mat matResult = cv::Mat::zeros(totalRow * blockSize, totalCol * blockSize, CV_8UC1);
    for (int row = 0; row < totalRow; ++row) {
        for (int col = 0; col < totalCol; ++col) {
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
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "COMBINE IMAGE NEW REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    cv::Mat matChessBoardImage = generateChessBoardImage(11, 11, 20);
    PR_COMBINE_IMG_NEW_CMD stCmd;
    PR_COMBINE_IMG_NEW_RPY stRpy;
    stCmd.vecVecFrameCtr = VectorOfVectorOfPoint{ VectorOfPoint{ cv::Point(110, 110), cv::Point(310, 110) },
        VectorOfPoint{ cv::Point(110, 310), cv::Point(310, 310) } };
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.vecInputImages.push_back(matChessBoardImage);
    stCmd.vecInputImages.push_back(matChessBoardImage);

    PR_CombineImgNew(&stCmd, &stRpy);
    std::cout << "PR_CombineImgNew status " << ToInt32(stRpy.enStatus) << std::endl;
    if (!stRpy.matResultImage.empty())
        cv::imwrite("./data/CombineImageNewResult.png", stRpy.matResultImage);

    if (VisionStatus::OK != stRpy.enStatus)
        return;

    cv::Mat matResultChessBoardImage = generateChessBoardImage(21, 21, 20);
    cv::Mat matCompare = stRpy.matResultImage - matResultChessBoardImage;
    int nonZeroCount = cv::countNonZero(matCompare);
    std::cout << "Compare with difference points cout " << nonZeroCount << std::endl;
}

void TestCombineImageNew_2() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "COMBINE IMAGE NEW REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

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

void TestCombineImageNew() {
    TestCombineImageNew_1();
    TestCombineImageNew_2();
}

}
}