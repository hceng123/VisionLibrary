#ifndef _VISION_REGRESSION_TEST_UTILITY_FUNCTION_H_
#define _VISION_REGRESSION_TEST_UTILITY_FUNCTION_H_

#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

template<class T>
void printfMat ( const cv::Mat &mat, int nPrecision = 1)
{
    std::cout << std::fixed << std::setprecision(nPrecision);
    for (short row = 0; row < mat.rows; ++row)
    {
        for (short col = 0; col < mat.cols; ++col)
        {
            std::cout << mat.at<T>(row, col) << " ";
        }
        std::cout << std::endl;
    }
}

template<class T>
void printfVectorOfVector(const std::vector<std::vector<T>> &vevVecInput)
{
    for (const auto &vecInput : vevVecInput )
    {
        for (const auto value : vecInput )
        {
            printf("%.2f ", value);
        }
        printf("\n");
    }
}

namespace AOI
{
namespace Vision
{

VectorOfFloat split ( const std::string &s, char delim );
VectorOfVectorOfFloat parseData(const std::string &strContent);
VectorOfVectorOfFloat readDataFromFile(const std::string &strFilePath);
void copyVectorOfVectorToMat(const VectorOfVectorOfFloat &vecVecInput, cv::Mat &matOutput);
cv::Mat readMatFromCsvFile(const std::string &strFilePath);

template<class T>
void printPRLine(const PR_Line_<T> &line) {
    std::cout << "Point1 " << line.pt1 << ", Point2 " << line.pt2 << std::endl;
}

}
}
#endif /*_VISION_REGRESSION_TEST_UTILITY_FUNCTION_H_*/