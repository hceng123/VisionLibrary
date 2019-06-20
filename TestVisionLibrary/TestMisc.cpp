#include "stdafx.h"
#include <iostream>
#include <boost/filesystem.hpp>

#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include "TestSub.h"

using namespace AOI::Vision;
namespace fs = boost::filesystem;

void TestCombineImage() {
    PR_COMBINE_IMG_CMD stCmd;
    PR_COMBINE_IMG_RPY stRpy;
    stCmd.nCountOfImgPerFrame = 4;
    stCmd.nCountOfFrameX = 16;
    stCmd.nCountOfFrameY = 6;
    stCmd.nOverlapX = 104;
    stCmd.nOverlapY = 173;
    stCmd.nCountOfImgPerRow = 61;

    const std::string strFolder = "D:/Data/KB_FOR_Procedure/FOVImage/";

    for (int nRow = 0; nRow < stCmd.nCountOfFrameY; ++ nRow)
    //Column start from right to left.
    for (int nCol = 0; nCol < stCmd.nCountOfFrameX; ++ nCol)
    for (int imgNo = 1; imgNo <= stCmd.nCountOfImgPerFrame; ++ imgNo)
    {
        char arrCharFileName[100];
        int nImageIndex = nCol * stCmd.nCountOfImgPerFrame + nRow * stCmd.nCountOfImgPerRow + imgNo;
        _snprintf(arrCharFileName, sizeof(arrCharFileName), "F%d-%d-1.bmp", nRow + 1, nImageIndex);
        std::string strImagePath = strFolder + std::string ( arrCharFileName );
        cv::Mat mat = cv::imread(strImagePath, cv::IMREAD_GRAYSCALE);
        if ( mat.empty() )
            continue;
        cv::Mat matColor;
        cv::cvtColor(mat, matColor, CV_BayerGR2BGR);
        stCmd.vecInputImages.push_back( matColor );
    }

    PR_CombineImg ( &stCmd, &stRpy );
    int imgNo = 0;
    for ( const auto &mat : stRpy.vecResultImages ) {
        char arrChFileName[100];
        _snprintf(arrChFileName, sizeof(arrChFileName), "CombineResult_%d.bmp", imgNo);
        std::string strResultFile = strFolder + arrChFileName;
        cv::imwrite(strResultFile, mat);
        ++ imgNo;
    }
}

void TestCombineImage_1() {
    PR_COMBINE_IMG_CMD stCmd;
    PR_COMBINE_IMG_RPY stRpy;
    stCmd.nCountOfImgPerFrame = 4;
    stCmd.nCountOfFrameX = 18;
    stCmd.nCountOfFrameY = 13;
    stCmd.nOverlapX = 102;
    stCmd.nOverlapY = 140;
    stCmd.nCountOfImgPerRow = 72;

    const std::string strFolder = "D:/Data/DEMO_Board/FOVImage/";

    for (int nRow = 0; nRow < stCmd.nCountOfFrameY; ++ nRow)
    //Column start from right to left.
    for (int nCol = 0; nCol < stCmd.nCountOfFrameX; ++ nCol)
    for (int imgNo = 1; imgNo <= stCmd.nCountOfImgPerFrame; ++ imgNo)
    {
        char arrCharFileName[100];
        int nImageIndex = nCol * stCmd.nCountOfImgPerFrame + nRow * stCmd.nCountOfImgPerRow + imgNo;
        _snprintf(arrCharFileName, sizeof(arrCharFileName), "F%d-%d-1.bmp", nRow + 1, nImageIndex);
        std::string strImagePath = strFolder + std::string ( arrCharFileName );
        cv::Mat mat = cv::imread(strImagePath, cv::IMREAD_GRAYSCALE);
        if ( mat.empty() )
            continue;
        cv::Mat matColor;
        cv::cvtColor(mat, matColor, CV_BayerGR2BGR);
        stCmd.vecInputImages.push_back( matColor );
    }

    PR_CombineImg ( &stCmd, &stRpy );
    if ( stRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to combine image" << std::endl;
        return;
    }

    double dScale = 0.5;
    int imgNo = 0;
    for ( const auto &mat : stRpy.vecResultImages ) {
        char arrChFileName[100];
        _snprintf(arrChFileName, sizeof(arrChFileName), "CombineResult_%d.bmp", imgNo);
        std::string strResultFile = strFolder + arrChFileName;
        cv::Mat matResize;
        cv::resize ( mat, matResize, cv::Size(), dScale, dScale );
        cv::imwrite(strResultFile, matResize);
        ++ imgNo;
    }
}

void TestCombineImage_2() {
    PR_COMBINE_IMG_CMD stCmd;
    PR_COMBINE_IMG_RPY stRpy;
    stCmd.nCountOfImgPerFrame = 1;
    stCmd.nCountOfFrameX = 2;
    stCmd.nCountOfFrameY = 1;
    stCmd.nOverlapX = 102;
    stCmd.nOverlapY = 0;
    stCmd.nCountOfImgPerRow = 2;

    const std::string strFolder = "D:/Data/DEMO_Board/FOVImage/";

    for (int nRow = 0; nRow < stCmd.nCountOfFrameY; ++ nRow)
    //Column start from right to left.
    for (int nCol = 0; nCol < stCmd.nCountOfFrameX; ++ nCol)
    for (int imgNo = 1; imgNo <= stCmd.nCountOfImgPerFrame; ++ imgNo)
    {
        char arrCharFileName[100];
        int nImageIndex = nCol * stCmd.nCountOfImgPerFrame + nRow * stCmd.nCountOfImgPerRow + imgNo;
        _snprintf(arrCharFileName, sizeof(arrCharFileName), "F%d-%d-1.bmp", nRow + 1, nImageIndex);
        std::string strImagePath = strFolder + std::string ( arrCharFileName );
        cv::Mat mat = cv::imread(strImagePath, cv::IMREAD_GRAYSCALE);
        if ( mat.empty() )
            continue;
        cv::Mat matColor;
        cv::cvtColor(mat, matColor, CV_BayerGR2BGR);
        stCmd.vecInputImages.push_back( matColor );
    }

    PR_CombineImg ( &stCmd, &stRpy );
    if ( stRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to combine image" << std::endl;
        return;
    }

    double dScale = 0.5;
    int imgNo = 0;
    for ( const auto &mat : stRpy.vecResultImages ) {
        char arrChFileName[100];
        _snprintf(arrChFileName, sizeof(arrChFileName), "TestCombineResult_%d.bmp", imgNo);
        std::string strResultFile = strFolder + arrChFileName;
        cv::Mat matResize;
        cv::resize ( mat, matResize, cv::Size(), dScale, dScale );
        cv::imwrite(strResultFile, matResize);
        ++ imgNo;
    }
}

void TestCombineImage_HaoYu() {
    PR_COMBINE_IMG_CMD stCmd;
    PR_COMBINE_IMG_RPY stRpy;
    stCmd.nCountOfImgPerFrame = 5;
    stCmd.nCountOfFrameX = 4;
    stCmd.nCountOfFrameY = 1;
    stCmd.nOverlapX = 277;
    stCmd.nOverlapY = 0;
    stCmd.nCountOfImgPerRow = 20;
    stCmd.enScanDir = PR_SCAN_IMAGE_DIR::LEFT_TO_RIGHT;

    const std::string strFolder = "D:/Data/20180206115742/";

    for (int nRow = 0; nRow < stCmd.nCountOfFrameY; ++ nRow)
    //Column start from right to left.
    for (int nCol = 0; nCol < stCmd.nCountOfFrameX; ++ nCol)
    for (int imgNo = 1; imgNo <= stCmd.nCountOfImgPerFrame; ++ imgNo)
    {
        char arrCharFileName[100];
        _snprintf(arrCharFileName, sizeof(arrCharFileName), "I%d_%d.bmp", nCol + 1, imgNo );
        std::string strImagePath = strFolder + std::string ( arrCharFileName );
        cv::Mat mat = cv::imread(strImagePath, cv::IMREAD_COLOR);
        if ( mat.empty() )
            continue;
        stCmd.vecInputImages.push_back( mat );
    }

    PR_CombineImg ( &stCmd, &stRpy );
    if ( stRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to combine image" << std::endl;
        return;
    }

    double dScale = 0.5;
    int imgNo = 0;
    for ( const auto &mat : stRpy.vecResultImages ) {
        char arrChFileName[100];
        _snprintf(arrChFileName, sizeof(arrChFileName), "CombineResult_%d.bmp", imgNo);
        std::string strResultFile = strFolder + arrChFileName;
        cv::Mat matResize;
        cv::resize ( mat, matResize, cv::Size(), dScale, dScale );
        cv::imwrite(strResultFile, matResize);
        ++ imgNo;
    }
}

void TestPrThreshold() {
    PR_THRESHOLD_CMD stCmd;
    PR_THRESHOLD_RPY stRpy;

    stCmd.matInputImg = cv::imread("./data/ShiftedDevice.png");
    stCmd.bDoubleThreshold = true;
    stCmd.nThreshold1 = 20;
    stCmd.nThreshold2 = 180;
    stCmd.bInverseResult = true;

    PR_Threshold(&stCmd, &stRpy);
    if (stRpy.enStatus != VisionStatus::OK) {
        std::cout << "Failed do threshold!" << std::endl;
        return;
    }
    cv::imwrite("./data/DoubleThresholdResultInverse.png", stRpy.matResultImg);
}

void drawDashRect(cv::Mat &matImage, int linelength, int dashlength, const cv::Rect rect, const cv::Scalar &color, int thickness) {
    int tl_x = rect.x;
    int tl_y = rect.y;

    int totallength = dashlength + linelength;
    int nCountX = ToInt32(std::ceil((float)rect.width  / totallength));
    int nCountY = ToInt32(std::ceil((float)rect.height / totallength));

    cv::Point start, end;//start and end point of each dash
    //draw the horizontal lines
    start.y = tl_y;
    start.x = tl_x;

    end.x = tl_x;
    end.y = tl_y;

    for (int i = 0; i < nCountX; i++)
    {
        end.x = tl_x + (i + 1) * totallength - dashlength; //draw top dash line
        if (end.x > rect.br().x)
            end.x = rect.br().x;
        end.y = tl_y;
        start.x = tl_x + i * totallength;
        start.y = tl_y;
        cv::line(matImage, start, end, color, thickness);
    }

    for (int i = 0; i < nCountX; i++)
    {
        start.x = tl_x + i * totallength;
        start.y = tl_y + rect.height;
        end.x = tl_x + (i + 1) * totallength - dashlength;//draw bottom dash line
        if (end.x > rect.br().x)
            end.x = rect.br().x;
        end.y = tl_y + rect.height;
        cv::line(matImage, start, end, color, thickness);
    }

    for (int i = 0; i < nCountY; i++)
    {
        start.x = tl_x;
        start.y = tl_y + i*totallength;
        end.y = tl_y + (i + 1)*totallength - dashlength;//draw left dash line
        if (end.y > rect.br().y)
            end.y = rect.br().y;
        end.x = tl_x;
        cv::line(matImage, start, end, color, thickness);
    }

    for (int i = 0; i < nCountY; i++)
    {
        start.x = tl_x + rect.width;
        start.y = tl_y + i*totallength;
        end.y = tl_y + (i + 1)*totallength - dashlength;//draw right dash line
        if (end.y > rect.br().y)
            end.y = rect.br().y;
        end.x = tl_x + rect.width;
        cv::line(matImage, start, end, color, thickness);
    }
}

void TestDrawDashRect()
{
    cv::Mat matResult = cv::Mat::ones(500, 500, CV_8UC3) * 255;
    drawDashRect(matResult, 10, 8, cv::Rect(100, 100, 295, 295), cv::Scalar(128,128,128), 2);
    cv::imwrite("./data/TestDashResult.png", matResult);
}

void TestGenerateSelectedImage() {
    VectorOfMat vectorOfImage;
    fs::path p("./data/ButtonImages/");

    if (!is_directory(p))
        return;

    std::vector<fs::directory_entry> v; // To save the file names in a vector.
    std::copy(fs::directory_iterator(p), fs::directory_iterator(), back_inserter(v));

    for (std::vector<fs::directory_entry>::const_iterator it = v.begin(); it != v.end(); ++it)
    {
        if (!is_regular_file(it->path()))
            continue;

        auto strFilePath = (*it).path().string();
        cv::Mat matImage = cv::imread(strFilePath, cv::IMREAD_COLOR);
        if (matImage.empty()) {
            std::cout << "Failed to read image " << strFilePath << std::endl;
            return;
        }

        cv::resize(matImage, matImage, cv::Size(128, 128));
        cv::imwrite(strFilePath, matImage);

        const int lineWidth = 5;
        cv::Point point1(matImage.cols / 2, matImage.rows * 3 / 4);
        cv::Point point2(point1.x + 20, point1.y + 20);
        cv::Point point3(point2.x + 40, point2.y - 40);
        cv::line(matImage, point1, point2, cv::Scalar(0, 255, 0), lineWidth);
        cv::line(matImage, point2, point3, cv::Scalar(0, 255, 0), lineWidth);

        auto strFileName = (*it).path().filename().string();
        std::string fileNameWithoutExe = strFileName.substr(0, strFileName.length() - 4);
        std::string strResultName = "./data/ButtonImagesResult/" + fileNameWithoutExe + "Selected.png";
        cv::imwrite(strResultName, matImage);
    }
}

template<typename T>
static std::vector<size_t> sort_index_value(std::vector<T> &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    std::vector<T> vClone(v);
    for (size_t i = 0; i < idx.size(); ++i)
        v[i] = vClone[idx[i]];
    return idx;
}

void swapValue(int& value1, int &value2) {
    value1 = value1 + value2;
    value2 = value1 - value2;
    value1 = value1 - value2;
}

void kernel_sort_index_value(int* values, int size, int *sortedIndex) {
    for (int i = 0; i < size; ++ i)
        sortedIndex[i] = i;

    for(int i = 0; i < size - 1; ++ i) {
        for (int j = i + 1; j < size; ++j) {
            if (values[i] > values[j]) {
                swapValue(values[i], values[j]);
                swapValue(sortedIndex[i], sortedIndex[j]);
            }
        }
    }
}

void TestSortIndexValue() {
    const int SIZE = 10;
    int values[SIZE] = {5,3,8,12,7,67,15,40,9,2};
    int sortedIndex[SIZE];
    kernel_sort_index_value(values, SIZE, sortedIndex);

    std::cout << "kernel_sort_index_value test result: " << std::endl;
    for (int i = 0; i < SIZE; ++i) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < SIZE; ++i) {
        std::cout << sortedIndex[i] << " ";
    }
    std::cout << std::endl;
}

void TestMeasureDist() {
    PR_MEASURE_DIST_CMD stCmd;
    PR_MEASURE_DIST_RPY stRpy;

    {
        stCmd.ptStart = cv::Point2f(100, 100);
        stCmd.ptEnd = cv::Point2f(200, 200);
        stCmd.fFiducialSlope = 0.1f;
        stCmd.enMeasureMode = PR_MEASURE_DIST_CMD::MODE::X_DIRECTION;

        PR_MeasureDist(&stCmd, &stRpy);
        std::cout << "PR_MeasureDist status " << ToInt32(stRpy.enStatus) << std::endl;
        std::cout << "Cross point " << stRpy.ptCross << std::endl;
        std::cout << "Distance " << stRpy.fDistance << std::endl;
        std::cout << "Measure with start " << stRpy.bMeasureWithStart << std::endl;

        cv::Mat matResult = cv::Mat::ones(500, 500, CV_8UC3) * PR_MAX_GRAY_LEVEL;
        cv::line(matResult, stCmd.ptStart, stRpy.ptCross, cv::Scalar(0, 255, 0), stRpy.bMeasureWithStart ? 2 : 1);
        cv::line(matResult, stCmd.ptEnd, stRpy.ptCross, cv::Scalar(0, 255, 0), stRpy.bMeasureWithStart ? 1 : 2);
        cv::imwrite("./data/MeasureDistResult_1.png", matResult);
    }

    {
        stCmd.ptStart = cv::Point2f(100, 100);
        stCmd.ptEnd = cv::Point2f(200, 200);
        stCmd.fFiducialSlope = -0.1f;
        stCmd.enMeasureMode = PR_MEASURE_DIST_CMD::MODE::X_DIRECTION;

        PR_MeasureDist(&stCmd, &stRpy);
        std::cout << "PR_MeasureDist status " << ToInt32(stRpy.enStatus) << std::endl;
        std::cout << "Cross point " << stRpy.ptCross << std::endl;
        std::cout << "Distance " << stRpy.fDistance << std::endl;
        std::cout << "Measure with start " << stRpy.bMeasureWithStart << std::endl;

        cv::Mat matResult = cv::Mat::ones(500, 500, CV_8UC3) * PR_MAX_GRAY_LEVEL;
        cv::line(matResult, stCmd.ptStart, stRpy.ptCross, cv::Scalar(0, 255, 0), stRpy.bMeasureWithStart ? 2 : 1);
        cv::line(matResult, stCmd.ptEnd, stRpy.ptCross, cv::Scalar(0, 255, 0), stRpy.bMeasureWithStart ? 1 : 2);
        cv::imwrite("./data/MeasureDistResult_2.png", matResult);
    }

    {
        stCmd.ptStart = cv::Point2f(1330, 3811.5);
        stCmd.ptEnd = cv::Point2f(11280, 13261.5);
        stCmd.fFiducialSlope = 0.000166291298f;
        stCmd.enMeasureMode = PR_MEASURE_DIST_CMD::MODE::X_DIRECTION;

        PR_MeasureDist(&stCmd, &stRpy);
        std::cout << "PR_MeasureDist status " << ToInt32(stRpy.enStatus) << std::endl;
        std::cout << "Cross point " << stRpy.ptCross << std::endl;
        std::cout << "Distance " << stRpy.fDistance << std::endl;
        std::cout << "Measure with start " << stRpy.bMeasureWithStart << std::endl;
    }

    {
        stCmd.ptStart = cv::Point2f(1272, 3869);
        stCmd.ptEnd = cv::Point2f(11224.50, 13190);
        stCmd.fFiducialSlope = 0.000166291298f;
        stCmd.enMeasureMode = PR_MEASURE_DIST_CMD::MODE::Y_DIRECTION;

        PR_MeasureDist(&stCmd, &stRpy);
        std::cout << "PR_MeasureDist status " << ToInt32(stRpy.enStatus) << std::endl;
        std::cout << "Cross point " << stRpy.ptCross << std::endl;
        std::cout << "Distance " << stRpy.fDistance << std::endl;
        std::cout << "Measure with start " << stRpy.bMeasureWithStart << std::endl;
    }
}